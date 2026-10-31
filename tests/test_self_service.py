"""Revoke tickets/requests, admin Update button flow, delete chat, and several asks in ONE chat message.
Mock model / temp DB. Whether the real LLM emits one tool call per ask is NOT proven here (prompt + tools are)."""
import os
from pathlib import Path

import pytest

os.environ["DEMO_MOCK"] = "1"
import accounts, app as appmod, database, memory, tools  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _as(sid, role="student"):
    c = appmod.app.test_client()
    with c.session_transaction() as s:
        s["sid"], s["role"] = sid, role
    return c


@pytest.fixture
def stu():
    appmod._llm = None
    accounts.seed({})
    return _as("S1001")


@pytest.fixture
def adm():
    return _as("A001", "admin")


def _ticket(c, text="My fee payment failed twice on the portal"):
    did = c.post("/api/chat", json={"message": "Raise a support ticket: " + text}).get_json()["draft"]["draft_id"]
    return c.post("/api/tickets/confirm", json={"draft_id": did}).get_json()["ticket"]["ticket_id"]


# ---------------- revoke ----------------
def test_student_can_revoke_open_ticket_and_admin_sees_it_final(stu, adm):
    tid = _ticket(stu)
    assert stu.post(f"/api/my/tickets/{tid}/revoke").status_code == 200
    assert stu.get("/api/my/tickets").get_json()[0]["status"] == "Revoked"
    assert adm.get("/api/admin/tickets").get_json()[0]["status"] == "Revoked"
    assert adm.post(f"/api/admin/tickets/{tid}", json={"status": "Resolved"}).status_code == 400   # revoked is final
    assert stu.post(f"/api/my/tickets/{tid}/revoke").status_code == 409                             # no double revoke
    assert adm.get("/api/admin/overview").get_json()["open_tickets"] == 0
    assert any(e["action"] == "ticket_revoked" and e["actor"] == "S1001" for e in database.list_audit())


def test_only_open_tickets_and_only_own_tickets_can_be_revoked(stu, adm):
    tid = _ticket(stu)
    assert adm.post(f"/api/admin/tickets/{tid}", json={"status": "In Progress"}).status_code == 200
    assert stu.post(f"/api/my/tickets/{tid}/revoke").status_code == 409                             # staff already working on it
    assert _as("S1002").post(f"/api/my/tickets/{tid}/revoke").status_code == 404                    # not theirs
    assert adm.post(f"/api/my/tickets/{tid}/revoke").status_code == 403                             # admins are not students
    assert stu.post("/api/my/tickets/TKT-9999/revoke").status_code == 404


def test_student_can_revoke_pending_request_nothing_applied(stu, adm):
    r = stu.post("/api/requests", json={"field": "email", "value": "new.mail@example.edu"}).get_json()
    assert stu.post(f"/api/my/requests/{r['approval_id']}/revoke").status_code == 200
    mine = stu.get("/api/my/requests").get_json()[0]
    assert mine["status"] == "revoked" and mine["decision_reason"] == "Withdrawn by student"
    assert adm.post(f"/api/admin/approvals/{r['approval_id']}", json={"decision": "approve"}).status_code == 409   # cannot approve a revoked request
    assert stu.get("/api/profile").get_json()["record"]["email"] == "aarav.mehta@example.edu"
    assert adm.get("/api/admin/overview").get_json()["pending"] == 0
    assert stu.post(f"/api/my/requests/{r['approval_id']}/revoke").status_code == 409
    assert any(e["action"] == "request_revoked" for e in database.list_audit())


def test_decided_or_foreign_requests_cannot_be_revoked(stu, adm):
    r = stu.post("/api/requests", json={"field": "elective", "value": "Machine Learning Ops"}).get_json()
    assert _as("S1002").post(f"/api/my/requests/{r['approval_id']}/revoke").status_code == 404
    adm.post(f"/api/admin/approvals/{r['approval_id']}", json={"decision": "approve"})
    assert stu.post(f"/api/my/requests/{r['approval_id']}/revoke").status_code == 409
    assert stu.get("/api/profile").get_json()["record"]["elective"] == "Machine Learning Ops"        # approval stands


# ---------------- admin: choose status, press Update, student sees it ----------------
def test_admin_status_update_reaches_the_students_support_page(stu, adm):
    tid = _ticket(stu)
    assert stu.get("/api/my/tickets").get_json()[0]["status"] == "Open"
    for new in ("In Progress", "Resolved"):
        assert adm.post(f"/api/admin/tickets/{tid}", json={"status": new}).status_code == 200
        assert stu.get("/api/my/tickets").get_json()[0]["status"] == new
    assert adm.post(f"/api/admin/tickets/{tid}", json={"status": "Revoked"}).status_code == 400       # admins cannot fake a revoke
    assert [e["result"] for e in database.list_audit() if e["action"] == "ticket_status_update"][0] == "In Progress -> Resolved"


def test_admin_template_has_update_button_and_no_autosave():
    a = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
    assert 'data-upd="' in a and ">Update</button>" in a and "data-tk]')" not in a.replace("[data-tk=", "")
    assert "document.addEventListener('change',async e=>{const s=e.target.closest('[data-tk]')" not in a


# ---------------- delete a chat ----------------
def test_student_can_delete_one_chat_only(stu):
    c1 = stu.post("/api/chat", json={"message": "What is my fee status?"}).get_json()["conversation_id"]
    c2 = stu.post("/api/chat", json={"message": "Hello"}).get_json()["conversation_id"]
    assert stu.delete(f"/api/history/{c1}").status_code == 200
    assert [c["id"] for c in stu.get("/api/history").get_json()] == [c2]                                # other chat untouched
    assert stu.get(f"/api/history/{c1}").status_code == 404 and database.get_messages(c1) == []
    assert stu.delete(f"/api/history/{c1}").status_code == 404
    assert _as("S1002").delete(f"/api/history/{c2}").status_code == 404                               # not theirs
    assert appmod.app.test_client().delete(f"/api/history/{c2}").status_code == 401                   # signed out
    assert any(e["action"] == "chat_deleted" for e in database.list_audit())


def test_deleting_a_chat_cancels_its_pending_draft_but_keeps_created_tickets(stu):
    cid = stu.post("/api/chat", json={"message": "Raise a support ticket"}).get_json()["conversation_id"]
    did = stu.post("/api/chat", json={"message": "my fee payment failed twice", "conversation_id": cid}).get_json()["draft"]["draft_id"]
    tid = _ticket(stu, "scholarship amount is wrong again")
    stu.delete(f"/api/history/{cid}")
    assert stu.post("/api/tickets/confirm", json={"draft_id": did}).status_code == 409
    assert [t["ticket_id"] for t in stu.get("/api/my/tickets").get_json()] == [tid]


def test_history_template_uses_an_icon_only_delete_button():
    p = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    row = p[p.index('class="ib" data-del'):]
    row = row[:row.index("</button>") + 9]
    assert 'aria-label="Delete chat' in row and 'title="Delete chat"' in row and '<use href="#i-trash"/>' in row
    assert ">Delete<" not in row and "data-rvt" in p and "data-rvr" in p and "confirm('Delete this chat?" in p


# ---------------- several asks in ONE message ----------------
MSG = ("what is my name and my course?? also give me my fee status and scholarship status? "
       "Please correct my name to Shyam Radadia and I want to change my programme to PGDM AI and Data Science.")


def test_one_message_with_name_and_course_change_creates_both_requests(stu, adm):
    r = stu.post("/api/chat", json={"message": MSG}).get_json()
    events = [(e["tool"], e["arguments"]["field"], e["status"]) for e in r["tool_events"]]
    assert events == [("request_record_change", "name", "success"), ("request_record_change", "course", "success")]
    reqs = stu.get("/api/my/requests").get_json()
    assert {(x["field"], x["new_value"], x["status"]) for x in reqs} == {("name", "Shyam Radadia", "pending"), ("course", "AI & DS", "pending")}   # course normalised
    assert r["response"].count("REQ-") == 2 and "cannot" not in r["response"].lower()
    for x in reqs:                                                                                       # admin approves both
        assert adm.post(f"/api/admin/approvals/{x['id']}", json={"decision": "approve"}).status_code == 200
    rec = stu.get("/api/profile").get_json()["record"]
    assert rec["name"] == "Shyam Radadia" and rec["course"] == "AI & DS" and rec["programme"] == "PGDM"
    assert accounts.get("S1001")["name"] == "Shyam Radadia"                                              # sign-in display name follows
    assert any(n["name"] == "Shyam Radadia" for n in adm.get("/api/admin/students?q=shyam").get_json())


def test_name_and_course_request_validation(stu):
    assert stu.post("/api/requests", json={"field": "name", "value": "X"}).status_code == 400
    assert stu.post("/api/requests", json={"field": "name", "value": "<script>alert(1)</script>"}).status_code == 400
    assert stu.post("/api/requests", json={"field": "name", "value": "Shyam Radadia"}).status_code == 201
    assert stu.post("/api/requests", json={"field": "course", "value": "MBA Finance"}).status_code == 400
    for ok in ("PGDM AI and Data Science", "pgdm, bda", "AI & DS", "Big Data Analytics"):
        assert memory.normalise_course(ok) in ("AI & DS", "BDA"), ok


def test_several_ticket_drafts_in_one_turn_are_all_returned(stu):
    from conftest import ScriptedLLM, text_resp, tool_resp
    from llm import LLMResponse, ToolRequest
    import agent, json
    two = LLMResponse("", [ToolRequest("a", "prepare_support_ticket", json.dumps({"student_id": "S1001", "category": "fee", "description": "Payment failed twice on portal"})),
                           ToolRequest("b", "prepare_support_ticket", json.dumps({"student_id": "S1001", "category": "other", "description": "Library card is not working"}))], model="x")
    r = agent.run_agent("two problems", [], "S1001", ScriptedLLM([two, text_resp("Drafted both.")]), save_trace=False, student_id="S1001", conversation_id="c1")
    assert [e["status"] for e in r.trace.tool_calls] == ["success", "success"]
    assert len(database.list_drafts("c1", "S1001")) == 2 and database.list_tickets("S1001") == []       # drafts only: no ticket yet
    p = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert "(d.drafts||[]).map(draftUI).join('')" in p
