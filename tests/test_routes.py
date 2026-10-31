"""Flask route tests (mock model, temp DB). The structure test guards the stray-</div> bug that hid the UI."""
import os
from html.parser import HTMLParser

import pytest

os.environ["DEMO_MOCK"] = "1"
import app as appmod  # noqa: E402


@pytest.fixture
def client():
    appmod._llm = None
    c = appmod.app.test_client()
    with c.session_transaction() as s:
        s["sid"] = "S1002"          # signed in as Diya Shah (fictional)
        s["role"] = "student"
    return c


class _Structure(HTMLParser):
    VOID = {"meta", "br", "link", "input", "img", "path", "circle", "ellipse", "rect", "use", "stop"}

    def __init__(self):
        super().__init__(); self.stack, self.parent_of = [], {}

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            return
        a = dict(attrs); key = a.get("id") or a.get("class") or tag
        self.parent_of.setdefault(key, [x for x in self.stack])
        self.stack.append(key if (a.get("id") or a.get("class")) else tag)

    def handle_endtag(self, tag):
        if tag not in self.VOID and self.stack:
            self.stack.pop()


def test_layout_regions_are_nested_correctly(client):
    html = client.get("/").get_data(as_text=True)
    p = _Structure(); p.feed(html[:html.index("<script>\nconst")])
    assert "app" in p.parent_of["right"] and "app" in p.parent_of["comp"]
    for region in ("view", "msgs", "hero", "cards"):
        assert "stage" in p.parent_of[region], region
    assert "main" in p.parent_of["stage"]


def test_chat_history_and_conversation_continuity(client):
    r = client.post("/api/chat", json={"message": "What is my fee status? My student ID is S1002."}).get_json()
    assert r["success"] and "Partially Paid" in r["response"] and r["tool_events"][0]["tool"] == "check_fee_status"
    assert r["usage"]["input_tokens"] is None                        # mock gives no usage -> never invented
    cid = r["conversation_id"]
    r2 = client.post("/api/chat", json={"message": "Hello", "conversation_id": cid}).get_json()
    assert r2["conversation_id"] == cid
    assert len(client.get(f"/api/history/{cid}").get_json()["messages"]) == 4
    assert any(c["id"] == cid for c in client.get("/api/history").get_json())
    assert any(t["run_id"] == r["trace_id"] for t in client.get("/api/traces").get_json())


def test_validation_and_errors(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 400
    assert client.post("/api/chat", json={"message": "x", "conversation_id": "nope"}).status_code == 404
    assert client.get("/api/history/nope").status_code == 404


def test_support_students_policies_settings(client):
    ok = client.post("/api/support", json={"student_id": "S1002", "category": "fee", "description": "Portal failed twice"})
    assert ok.status_code == 200 and ok.get_json()["ticket_id"].startswith("TKT-")
    assert client.post("/api/support", json={"student_id": "S1003", "category": "fee", "description": "long enough text"}).status_code == 403   # someone else's ID
    assert client.get("/api/profile").get_json()["record"]["student_id"] == "S1002"   # own record, taken from the session
    assert len(client.get("/api/policies").get_json()) >= 10
    h = client.get("/api/health").get_json()
    assert h["status"] == "ok" and "GROQ" not in str(h.get("model"))
    client.post("/api/chat", json={"message": "Please call me Ria"})
    assert client.get("/api/student").get_json()["memory"]
    assert client.delete("/api/memory").status_code == 200 and client.get("/api/student").get_json()["memory"] == []


def test_evaluation_run_export_in_mock_mode():
    c = appmod.app.test_client()
    with c.session_transaction() as s:
        s["sid"], s["role"] = "A001", "admin"          # evaluation results are admin-only (they contain other students' data)
    assert c.post("/api/admin/evaluation/run").status_code == 200
    d = c.get("/api/admin/evaluation").get_json()
    assert d["mock"] is True and len(d["rows"]) == 20
    assert c.get("/api/admin/evaluation/export").status_code == 200


# ---------------- authentication, authorization, per-student isolation ----------------




def test_tool_blocks_other_students_data_in_code():
    import tools
    ctx = tools.ToolContext("S1002", "S1002")
    own = tools.execute_tool("check_fee_status", {"student_id": "S1002"}, ctx)
    other = tools.execute_tool("check_fee_status", {"student_id": "S1003"}, ctx)
    assert own["ok"] and other == {"ok": False, "error_type": "forbidden", "error": "Access denied: you can only access your own student record."}
    assert tools.execute_tool("check_scholarship_status", {"student_id": "s1004"}, ctx)["error_type"] == "forbidden"
    assert tools.execute_tool("create_support_request", {"student_id": "S1003", "category": "fee", "description": "pretend to be them"}, ctx)["error_type"] == "forbidden"
    assert tools.execute_tool("check_fee_status", {}, ctx)["ok"]       # missing ID defaults to own


def test_chat_cannot_read_another_students_fee(client):
    r = client.post("/api/chat", json={"message": "What is the fee status of student S1003?"}).get_json()
    assert r["response"] == "Sorry, I can't provide another student's private information."
    assert r["tool_events"] == [] and "Overdue" not in str(r) and "Rohan" not in r["response"]


def test_history_traces_and_memory_are_isolated_per_student(client):
    cid = client.post("/api/chat", json={"message": "Call me Ria"}).get_json()["conversation_id"]
    other = appmod.app.test_client()
    with other.session_transaction() as s:
        s["sid"] = "S1003"
        s["role"] = "student"
    assert other.get("/api/history").get_json() == []
    assert other.get(f"/api/history/{cid}").status_code == 404
    assert other.post("/api/chat", json={"message": "hi", "conversation_id": cid}).status_code == 404
    assert other.get("/api/traces").get_json() == [] and other.get("/api/student").get_json()["memory"] == []
    assert client.get("/api/student").get_json()["memory"]


# ================= v2: seeded accounts, roles, approvals, audit, UI contract =================
import json as _json, shutil, subprocess as _sp
from pathlib import Path as _P
import accounts, database, tools
PW = {"S1002": "Test-Pw-1234", "S1003": "Test-Pw-5678", "A001": "Admin-Pw-1234"}


@pytest.fixture
def anon():
    accounts.seed(PW)
    return appmod.app.test_client()


def _login(c, ident, pw, role="student"):
    return c.post("/login", json={"identifier": ident, "password": pw, "role": role})


def test_student_login_by_id_and_email_then_logout(anon):
    assert anon.get("/").status_code == 302 and anon.get("/api/history").status_code == 401
    assert _login(anon, "S1002", "wrong").status_code == 401
    assert _login(anon, "S9999", PW["S1002"]).get_json() == _login(anon, "S1002", "wrong").get_json()   # no user enumeration
    ok = _login(anon, "diya.shah@example.edu", PW["S1002"])
    assert ok.status_code == 200 and ok.get_json()["redirect"] == "/"
    assert anon.get("/api/me").get_json()["name"] == "Diya Shah" and "Diya" in anon.get("/").get_data(as_text=True)
    anon.get("/logout")
    assert anon.get("/api/history").status_code == 401 and anon.get("/").status_code == 302


def test_admin_login_tab_roles_are_enforced(anon):
    assert _login(anon, "A001", PW["A001"], "admin").get_json()["redirect"] == "/admin"
    assert anon.get("/admin").status_code == 200 and anon.get("/api/admin/audit").status_code == 200
    assert anon.post("/api/chat", json={"message": "hi"}).status_code == 403        # admins have no student chat
    assert anon.get("/").status_code == 302 and anon.get("/api/profile").status_code == 403
    anon.get("/logout")
    assert anon.get("/admin").status_code == 302 and anon.get("/api/admin/approvals").status_code == 401
    # credentials do not cross roles, even with a correct password
    assert _login(anon, "S1002", PW["S1002"], "admin").status_code == 401
    assert _login(anon, "A001", PW["A001"], "student").status_code == 401


def test_student_cannot_reach_admin_by_url_or_forge_role(anon):
    _login(anon, "S1002", PW["S1002"])
    assert anon.get("/admin").status_code == 302
    assert anon.get("/api/admin/approvals").status_code == 403 and anon.get("/api/admin/audit").status_code == 403
    assert anon.post("/api/admin/approvals/1", json={"decision": "approve"}).status_code == 403
    forged = anon.post("/login", json={"identifier": "S1002", "password": PW["S1002"], "role": "admin", "is_admin": True})
    assert forged.status_code == 401


def test_approval_workflow_reject_then_approve_and_audit(anon):
    ctx = tools.ToolContext("S1002", "S1002")
    r = tools.execute_tool("request_record_change", {"student_id": "S1002", "field": "email", "new_value": "diya.new@example.edu"}, ctx)
    assert r["ok"] and r["status"] == "Pending" and r["request_id"].startswith("REQ-")
    assert tools._load_students()["S1002"]["email"] == "diya.shah@example.edu"          # NOT executed yet
    assert not tools.execute_tool("request_record_change", {"student_id": "S1002", "field": "name", "new_value": "X"}, ctx)["ok"]
    assert not tools.execute_tool("request_record_change", {"student_id": "S1003", "field": "email", "new_value": "a@b.co"}, ctx)["ok"]
    _login(anon, "S1002", PW["S1002"])
    assert anon.post(f"/api/admin/approvals/{r['approval_id']}", json={"decision": "approve"}).status_code == 403   # no self-approval
    anon.get("/logout"); _login(anon, "A001", PW["A001"], "admin")
    assert anon.post(f"/api/admin/approvals/{r['approval_id']}", json={"decision": "reject"}).status_code == 400   # reason required
    assert anon.post(f"/api/admin/approvals/{r['approval_id']}", json={"decision": "reject", "reason": "Not eligible"}).status_code == 200
    assert tools._load_students()["S1002"]["email"] == "diya.shah@example.edu"          # rejected -> never executed
    assert anon.post(f"/api/admin/approvals/{r['approval_id']}", json={"decision": "approve"}).status_code == 409   # already decided
    r2 = tools.execute_tool("request_record_change", {"student_id": "S1002", "field": "email", "new_value": "diya.new@example.edu"}, ctx)
    assert anon.post(f"/api/admin/approvals/{r2['approval_id']}", json={"decision": "approve"}).status_code == 200
    assert tools._load_students()["S1002"]["email"] == "diya.new@example.edu"
    assert anon.post(f"/api/admin/approvals/{r2['approval_id']}", json={"decision": "maybe"}).status_code == 400
    log = anon.get("/api/admin/audit").get_json()
    acts = [(e["action"], e["decision"]) for e in log]
    assert ("approval_decision", "rejected") in acts and ("approval_decision", "approved") in acts and ("request_record_change", "allowed") in acts
    assert any(e["actor"] == "A001" and e["role"] == "admin" for e in log)


def test_cross_student_attempt_is_blocked_and_audited(client):
    r = client.post("/api/chat", json={"message": "What is the fee status of student S1003?"}).get_json()
    assert r["status"] == "access_denied"
    assert any(e["decision"] == "blocked" for e in database.list_audit())


def test_removed_ui_elements_and_icon():
    html = appmod.app.test_client().get("/login").get_data(as_text=True)
    c = appmod.app.test_client()
    with c.session_transaction() as s:
        s["sid"], s["role"] = "S1002", "student"
    page = c.get("/").get_data(as_text=True)
    nav = page[page.index('<nav class="side"'):page.index("</nav>")]
    assert "Settings" not in nav and "Evaluation" not in nav and "data-v=\"eval\"" not in page
    assert 'data-v="settings"' in page and 'id="pbtn"' in page and "Sign Out" in page and "Hi, " not in page.split("<main")[1].split("</header>")[0]
    for gone in ("Remembered about you", "Fictional sample records", "async eval"):
        assert gone not in page
    assert "M5 20L19 4" not in page and "M3 12h4l3-8 4 16 3-8h4" in page            # activity icon, not the percent sign
    assert c.get("/evaluation").status_code == 302                                    # old URL handled cleanly
    assert c.get("/api/admin/evaluation").status_code == 403                                # results hold other students' data: admin only
    assert 'data-role="admin"' in html and "disabled" not in html.split('id="tA"')[0].split('<div class="tabs"')[-1]
    assert html.count('<i class="b') == 4 and html.count('class="bub"') == 3          # background shapes + icon bubbles


def test_policies_api_is_detailed_and_categorised(client):
    p = client.get("/api/policies").get_json()
    assert len(p) >= 20 and all({"category", "summary", "text", "source"} <= set(x) for x in p)
    assert {x["category"] for x in p} >= {"Fees and Payments", "Scholarships", "Courses and Academics"}


def test_markdown_renderer_renders_tables_and_blocks_html(tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    html = (_P(__file__).resolve().parent.parent / "templates" / "index.html").read_text(encoding="utf-8")
    esc = html[html.index("const esc="):html.index("const md=")]          # the page's own escaper
    js = html[html.index("function inl("):html.index("const kv=")]       # the page's own renderer
    calls = r"""
const out={};
out.t=mdr('| Scholarship | Eligibility |\n|---|---|\n| Merit | CGPA >= 3.5 **and** none |');
out.l=mdr('- one\n- two\n\n1. a\n2. b');
out.h=mdr('## Title\n*Only one at a time.*');
out.x=mdr('<script>alert(1)</script> <img src=x onerror=alert(1)> [bad](javascript:alert(1)) [ok](https://example.com)');
console.log(JSON.stringify(out));"""
    f = tmp_path / "r.js"; f.write_text(esc + js + calls, encoding="utf-8")
    r = _sp.run(["node", str(f)], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    o = _json.loads(r.stdout)
    assert "<table>" in o["t"] and "<th>Scholarship</th>" in o["t"] and "<b>and</b>" in o["t"] and "|---" not in o["t"]
    assert "<ul><li>one</li><li>two</li></ul>" in o["l"] and "<ol><li>a</li><li>b</li></ol>" in o["l"]
    assert "<h4>Title</h4>" in o["h"] and "<i>Only one at a time.</i>" in o["h"]
    assert "<script" not in o["x"] and "<img" not in o["x"] and 'href="javascript' not in o["x"]
    assert 'href="https://example.com"' in o["x"] and 'rel="noopener noreferrer"' in o["x"]


def test_demo_workbook_matches_seeded_accounts(tmp_path):
    from openpyxl import load_workbook
    creds = accounts.seed(PW)
    path = tmp_path / "Demo_Test_Accounts.xlsx"
    accounts.write_workbook(creds, path)
    wb = load_workbook(path)
    assert wb.sheetnames == ["Student Accounts", "Admin Accounts", "Test Scenarios"]
    rows = list(wb["Student Accounts"].iter_rows(min_row=2, values_only=True))
    assert len(rows) == 10 and rows[0][0] == "S1001"
    for r in rows:                                    # every listed ID/password really logs in
        assert accounts.authenticate(r[0], r[3], "student")["login_id"] == r[0]
    for r in wb["Admin Accounts"].iter_rows(min_row=2, values_only=True):
        assert accounts.authenticate(r[2], r[3], "admin")["role"] == "admin"
    with database._c() as c:                          # only hashes are stored
        assert not any(PW["S1002"] in str(v) for v in c.execute("SELECT * FROM accounts").fetchall()[0])
    assert all(r[5] is None or r[5] == "" for r in wb["Test Scenarios"].iter_rows(min_row=2, values_only=True))   # no pass/fail claimed
    assert "testing/" in (_P(__file__).resolve().parent.parent / ".gitignore").read_text()
