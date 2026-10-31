"""Assignment-2 workflow tests: ticket draft/confirm, student<->admin request sync, admin pages, UI contract.
All use the mock model / temp DB (no API key). 'Live' behaviour of the real LLM is NOT covered here."""
import json, os, shutil, sqlite3, subprocess, tempfile
from pathlib import Path

import pytest

os.environ["DEMO_MOCK"] = "1"
import accounts, app as appmod, config, database, tools  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PW = {"S1002": "Stu-Pw-1111", "S1003": "Stu-Pw-2222", "A001": "Adm-Pw-3333"}


def _session(c, sid, role="student"):
    with c.session_transaction() as s:
        s["sid"], s["role"] = sid, role
    return c


@pytest.fixture
def student():
    appmod._llm = None
    return _session(appmod.app.test_client(), "S1002")


@pytest.fixture
def admin():
    return _session(appmod.app.test_client(), "A001", "admin")


def _count(table, where="1=1", args=()):
    with database._c() as c:
        return c.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", args).fetchone()[0]


# ---------------- support-ticket conversation flow ----------------
def test_ticket_request_asks_for_reason_first_and_creates_nothing(student):
    r = student.post("/api/chat", json={"message": "Raise a support ticket"}).get_json()
    assert "What issue are you facing?" in r["response"] and r["tool_events"] == [] and r["draft"] is None
    assert _count("support_tickets") == 0 and _count("ticket_drafts") == 0


def test_draft_then_confirm_creates_exactly_one_ticket(student):
    cid = student.post("/api/chat", json={"message": "Raise a support ticket"}).get_json()["conversation_id"]
    r = student.post("/api/chat", json={"message": "My fee payment failed twice on the portal", "conversation_id": cid}).get_json()
    assert r["draft"]["status"] == "awaiting_confirmation" and "Would you like me to submit" in r["response"]
    assert [e["tool"] for e in r["tool_events"]] == ["prepare_support_ticket"]
    assert _count("support_tickets") == 0                                   # still NOTHING before confirmation
    did = r["draft"]["draft_id"]
    assert student.get(f"/api/history/{cid}").get_json()["pending_drafts"][0]["id"] == did   # survives a page reload
    ok = student.post("/api/tickets/confirm", json={"draft_id": did})
    t = ok.get_json()["ticket"]
    assert ok.status_code == 200 and t["ticket_id"].startswith("TKT-") and t["status"] == "Open" and t["expected_response"]
    again = student.post("/api/tickets/confirm", json={"draft_id": did})     # double click / retry
    assert again.get_json()["ticket"]["ticket_id"] == t["ticket_id"] and again.get_json()["duplicate"] is True
    assert _count("support_tickets") == 1
    assert any(e["action"] == "create_support_request" for e in database.list_audit())
    assert [x["ticket_id"] for x in student.get("/api/my/tickets").get_json()] == [t["ticket_id"]]


def test_cancel_creates_no_ticket_and_blocks_later_confirm(student):
    r = student.post("/api/chat", json={"message": "Please raise a support ticket: my scholarship amount is wrong"}).get_json()
    did = r["draft"]["draft_id"]
    assert student.post("/api/tickets/cancel", json={"draft_id": did}).status_code == 200
    assert student.post("/api/tickets/confirm", json={"draft_id": did}).status_code == 409
    assert _count("support_tickets") == 0


def test_cannot_confirm_someone_elses_draft_and_model_cannot_create_tickets(student):
    did = student.post("/api/chat", json={"message": "Raise a support ticket: my fee payment failed twice"}).get_json()["draft"]["draft_id"]
    other = _session(appmod.app.test_client(), "S1003")
    assert other.post("/api/tickets/confirm", json={"draft_id": did}).status_code == 404
    assert other.post("/api/tickets/cancel", json={"draft_id": did}).status_code == 404
    assert "create_support_request" not in [s["function"]["name"] for s in tools.TOOL_SCHEMAS]
    assert student.post("/api/tickets/confirm", json={"draft_id": "x"}).status_code == 404
    assert _count("support_tickets") == 0


def test_failed_creation_gives_error_not_success_and_stays_retryable(student, monkeypatch):
    did = student.post("/api/chat", json={"message": "Raise a support ticket: my fee payment failed twice"}).get_json()["draft"]["draft_id"]
    monkeypatch.setattr(tools, "create_support_request", lambda **k: {"ok": False, "error": "disk full"})
    monkeypatch.setitem(tools.TOOL_REGISTRY, "create_support_request", tools.create_support_request)
    r = student.post("/api/tickets/confirm", json={"draft_id": did})
    assert r.status_code == 400 and "ticket" not in r.get_json() and _count("support_tickets") == 0
    assert database.get_draft(did)["status"] == "pending"


# ---------------- request <-> approval synchronisation ----------------
def _submit(student, field="elective", value="Machine Learning Ops"):
    return student.post("/api/requests", json={"field": field, "value": value})


def test_request_round_trip_approve_updates_everywhere(student, admin):
    r = _submit(student); j = r.get_json()
    assert r.status_code == 201 and j["status"] == "Pending" and j["request_id"].startswith("REQ-")
    assert _submit(student).get_json()["approval_id"] == j["approval_id"]    # duplicate pending request collapses
    mine = student.get("/api/my/requests").get_json()
    assert mine[0]["request_id"] == j["request_id"] and mine[0]["status"] == "pending"
    assert student.get("/api/profile").get_json()["record"]["elective"] in (None, "")   # not applied yet
    rows = admin.get("/api/admin/approvals").get_json()
    assert rows[0]["request_id"] == j["request_id"] and rows[0]["student_name"] == "Diya Shah"
    assert admin.get("/api/admin/overview").get_json()["pending"] == 1
    d = admin.post(f"/api/admin/approvals/{j['approval_id']}", json={"decision": "approve", "reason": "Seat available"})
    assert d.status_code == 200
    assert student.get("/api/my/requests").get_json()[0]["status"] == "approved"                   # same source of truth
    assert student.get("/api/my/requests").get_json()[0]["decision_reason"] == "Seat available"
    assert student.get("/api/profile").get_json()["record"]["elective"] == "Machine Learning Ops"  # applied after approval
    assert admin.get("/api/admin/overview").get_json()["pending"] == 0
    assert any(e["action"] == "approval_decision" and e["decision"] == "approved" and e["actor"] == "A001" for e in admin.get("/api/admin/audit").get_json())
    assert admin.post(f"/api/admin/approvals/{j['approval_id']}", json={"decision": "reject", "reason": "x"}).status_code == 409   # no conflicting 2nd decision


def test_rejected_request_leaves_record_unchanged(student, admin):
    before = student.get("/api/profile").get_json()["record"]["email"]
    j = _submit(student, "email", "diya.rejected@example.edu").get_json()
    assert admin.post(f"/api/admin/approvals/{j['approval_id']}", json={"decision": "reject"}).status_code == 400   # reason required
    assert admin.post(f"/api/admin/approvals/{j['approval_id']}", json={"decision": "reject", "reason": "Domain not allowed"}).status_code == 200
    r = student.get("/api/my/requests").get_json()[0]
    assert r["status"] == "rejected" and r["decision_reason"] == "Domain not allowed"
    assert student.get("/api/profile").get_json()["record"]["email"] == before


def test_request_validation_and_authorization(student):
    assert _submit(student, "email", "not-an-email").status_code == 400
    assert _submit(student, "fee_total", "0").status_code == 400
    assert _submit(student, "fee_concession", "short").status_code == 400
    assert student.post("/api/admin/approvals/1", json={"decision": "approve"}).status_code == 403
    other = _session(appmod.app.test_client(), "S1003")
    assert other.get("/api/my/requests").get_json() == []                     # students see only their own


def test_decision_is_atomic_rollback_on_failure(student, admin, monkeypatch):
    j = _submit(student).get_json()
    real = database._c
    class Boom:                                   # make the audit INSERT (last statement of the transaction) fail
        def __init__(s): s.c = real()
        def __enter__(s): s.c.__enter__(); return s
        def __exit__(s, *a): return s.c.__exit__(*a)
        def execute(s, q, *a):
            if "INTO audit_log" in q: raise sqlite3.OperationalError("disk I/O error")
            return s.c.execute(q, *a)
    monkeypatch.setattr(database, "_c", lambda: Boom())
    r = admin.post(f"/api/admin/approvals/{j['approval_id']}", json={"decision": "approve"})
    monkeypatch.setattr(database, "_c", real)
    assert r.status_code == 500 and "nothing was changed" in r.get_json()["error"]
    assert database.get_approval(j["approval_id"])["status"] == "pending"      # status rolled back
    assert student.get("/api/profile").get_json()["record"]["elective"] in (None, "")   # record not changed


# ---------------- admin pages are real and protected ----------------
def test_every_admin_page_api_returns_database_data(student, admin):
    _submit(student); student.post("/api/chat", json={"message": "Hello"})
    did = student.post("/api/chat", json={"message": "Raise a support ticket: my fee payment failed twice"}).get_json()["draft"]["draft_id"]
    student.post("/api/tickets/confirm", json={"draft_id": did})
    ov = admin.get("/api/admin/overview").get_json()
    assert ov["pending"] == 1 and ov["open_tickets"] == 1 and ov["students"] == 10 and ov["agent"]["runs"] >= 2
    assert ov["agent"]["total_tokens"] is None                                  # mock reports no usage -> "Unavailable"
    assert len(admin.get("/api/admin/students?q=diya").get_json()) == 1 and len(admin.get("/api/admin/students").get_json()) == 10
    assert admin.get("/api/admin/students/s1003").get_json()["name"] == "Rohan Iyer" and admin.get("/api/admin/students/S9").status_code == 404
    t = admin.get("/api/admin/tickets").get_json()[0]
    assert admin.post(f"/api/admin/tickets/{t['ticket_id']}", json={"status": "In Progress"}).status_code == 200
    assert student.get("/api/my/tickets").get_json()[0]["status"] == "In Progress"
    assert admin.post(f"/api/admin/tickets/{t['ticket_id']}", json={"status": "Bogus"}).status_code == 400
    assert len(admin.get("/api/admin/policies").get_json()) >= 20 and len(admin.get("/api/admin/traces").get_json()) >= 2
    assert any(e["action"] == "ticket_status_update" for e in admin.get("/api/admin/audit").get_json())
    assert admin.get("/admin").status_code == 200


def test_students_cannot_use_any_admin_endpoint(student):
    for u in ("/api/admin/overview", "/api/admin/approvals", "/api/admin/students", "/api/admin/students/S1003", "/api/admin/tickets", "/api/admin/traces", "/api/admin/audit", "/api/admin/policies"):
        assert student.get(u).status_code == 403, u
    assert student.post("/api/admin/tickets/TKT-1001", json={"status": "Closed"}).status_code == 403
    assert student.get("/admin", follow_redirects=False).status_code == 302


# ---------------- UI contract (templates) ----------------
def test_ui_contract_login_settings_composer_menu(student, admin):
    login = appmod.app.test_client().get("/login").get_data(as_text=True)
    assert "Demo accounts" not in login and "Demo_Test_Accounts" not in login and "google" not in login.lower()
    assert 'data-role="student"' in login and 'data-role="admin"' in login
    page = student.get("/").get_data(as_text=True)
    menu = page[page.index('id="mpop"'):page.index("</div></div>", page.index('id="mpop"'))]
    for item in ("Profile", "Settings", "Sign Out"):
        assert item in menu
    assert "themeMenu" not in page and "Light Mode" not in menu and 'class="sw"' not in menu     # toggle removed from dropdown
    assert ".comp textarea:focus-visible{outline:none" in page and ".comp:focus-within" in page      # no inner rectangle, soft ring
    nav = page[page.index('<nav class="side"'):page.index("</nav>")]
    assert "Settings" not in nav and "Evaluation" not in nav
    adm = admin.get("/admin").get_data(as_text=True)
    for item in ("Overview", "Approval Requests", "Student Records", "Support Tickets", "Policies", "Agent Traces", "Audit Log", "Sign Out", "Profile", "Settings"):
        assert item in adm
    assert "Student Support Agent · Administrator" in adm and "(A001)" not in adm and "Demo accounts" not in adm
    amenu = adm[adm.index('id="mpop"'):adm.index("</div></div>", adm.index('id="mpop"'))]
    assert "Light Mode" not in amenu and "tg2" not in adm and "Sign Out" in amenu


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_markdown_numbering_continues_across_bulleted_steps(tmp_path):
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    pre = html[html.index("const esc="):html.index("const md=")] + html[html.index("function inl("):html.index("const kv=")]
    md = ("**How to change an elective course**\\n\\n1. **Timeframe**\\n- Submit within the first two weeks.\\n\\n2. **What you need to do**\\n"
          "- Fill out the form.\\n- Get it signed by:\\n  - Your faculty advisor\\n  - The programme coordinator\\n\\n3. **Submission**\\n- Submit the signed form.\\n"
          "\\n## Notes\\n1. one\\n2. two\\n\\n| A | B |\\n|---|---|\\n| 1 | 2 |")
    f = tmp_path / "t.js"; f.write_text(pre + f"console.log(JSON.stringify(mdr('{md}')));", encoding="utf-8")
    out = json.loads(subprocess.run(["node", str(f)], capture_output=True, text=True, check=True).stdout)
    assert out.count("<ol") == 2                                    # ONE list for the 3 steps (+ the separate Notes list)
    steps = out[out.index("<ol"):out.index("</ol>")]
    assert steps.count("<li><b>") == 3 and "<li><b>Timeframe</b><ul>" in steps and "<li><b>Submission</b>" in steps
    assert "<ul><li>Your faculty advisor</li><li>The programme coordinator</li></ul>" in steps      # nested list
    assert "<h4>Notes</h4>" in out and "<table>" in out and "|---" not in out
    assert out.count('start="') == 0                                # numbering is natural 1,2,3 (not restarted)