"""Tools, in the course pattern:  Python function  +  JSON schema  +  registry  +  executor.

The LLM only *requests* a tool; this module's Python code executes it (Single_Tool_Calling).
Every tool returns a dict with "ok": True/False (Unit II notebooks) and NEVER raises.

Assignment-2 hook: every call goes through `execute_tool(name, args, ctx)`. That single
chokepoint is where authentication, authorization, approval and audit logging will be added
later, using `ctx.user_id` and the `TOOL_META` flags (e.g. writes=True needs approval)."""
import csv
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import config
import database
import memory
import retrieval

STUDENT_ID_RE = re.compile(r"S\d{4}")
CATEGORIES = ["fee", "scholarship", "course_change", "academic", "other"]
URGENT_CATEGORIES = {"fee"}   # payment problems get the 1-working-day SLA (see handbook)


@dataclass
class ToolContext:
    """Who is calling. Today just the memory profile id; Assignment 2 adds the signed-in student."""
    user_id: str = "demo_student"
    student_id: str | None = None   # signed-in student; when set, private tools are restricted to this ID
    conversation_id: str | None = None


# ------------------------------------------------------------------ data access
def _load_students() -> dict[str, dict]:
    with open(config.STUDENTS_CSV, newline="", encoding="utf-8") as f:
        data = {row["student_id"]: row for row in csv.DictReader(f)}
    for sid, fields in database.get_overrides().items():   # admin-approved changes
        if sid in data: data[sid].update(fields)
    return data


def _lookup(student_id: str) -> tuple[dict | None, dict | None]:
    """Validate the id and fetch the record. Returns (record, error_dict)."""
    sid = (student_id or "").strip().upper()
    if not STUDENT_ID_RE.fullmatch(sid):
        return None, {"ok": False, "error_type": "invalid_input",
                      "error": f"'{student_id}' is not a valid student ID (expected like S1001)."}
    record = _load_students().get(sid)
    if record is None:
        return None, {"ok": False, "error_type": "not_found",
                      "error": f"No student record found for ID {sid}."}
    return record, None


# ------------------------------------------------------------------ tools
def check_fee_status(student_id: str) -> dict:
    rec, err = _lookup(student_id)
    if err:
        return err
    if rec["fee_service"] != "online":   # simulated outage, used to test tool-failure handling
        return {"ok": False, "error_type": "tool_failure",
                "error": "The fee ledger service is temporarily unavailable. Please try again later."}
    total, paid = int(rec["fee_total"]), int(rec["fee_paid"])
    return {"ok": True, "student_id": rec["student_id"], "name": rec["name"],
            "fee_total_inr": total, "fee_paid_inr": paid, "balance_inr": total - paid,
            "next_due_date": rec["fee_due_date"], "fee_status": rec["fee_status"]}


def check_scholarship_status(student_id: str) -> dict:
    rec, err = _lookup(student_id)
    if err:
        return err
    return {"ok": True, "student_id": rec["student_id"], "name": rec["name"],
            "scholarship": rec["scholarship_name"], "status": rec["scholarship_status"],
            "last_updated": rec["scholarship_updated"] or None, "note": rec["scholarship_note"]}


def get_student_record(student_id: str) -> dict:
    rec, err = _lookup(student_id)
    if err:
        return err
    return {"ok": True, "student_id": rec["student_id"], "name": rec["name"],
            "programme": rec["programme"], "course": rec["course"], "semester": int(rec["semester"]),
            "email": rec["email"]}


def _tickets_conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS support_tickets (
        ticket_id TEXT PRIMARY KEY, student_id TEXT, category TEXT, description TEXT,
        priority TEXT, status TEXT, created_at TEXT)""")
    return conn


def create_support_request(student_id: str, category: str, description: str) -> dict:
    rec, err = _lookup(student_id)
    if err:
        return err
    if category not in CATEGORIES:
        return {"ok": False, "error_type": "invalid_input",
                "error": f"category must be one of {CATEGORIES}."}
    description = (description or "").strip()
    if len(description) < 10:
        return {"ok": False, "error_type": "invalid_input",
                "error": "Please describe the problem in at least 10 characters."}
    if memory.is_sensitive(description):
        return {"ok": False, "error_type": "invalid_input",
                "error": "Do not include passwords, OTPs or card numbers in a support request."}
    sid = rec["student_id"]
    with _tickets_conn() as conn:
        dup = conn.execute("SELECT * FROM support_tickets WHERE student_id=? AND category=? "
                           "AND description=? AND status='Open'", (sid, category, description)).fetchone()
        if dup:   # idempotent: same open request is not created twice
            return {"ok": True, "ticket_id": dup["ticket_id"], "priority": dup["priority"],
                    "status": dup["status"], "note": "An identical open request already exists."}
        n = conn.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]
        ticket_id = f"TKT-{1001 + n}"
        priority = "Urgent" if category in URGENT_CATEGORIES else "Normal"
        conn.execute("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)",
                     (ticket_id, sid, category, description, priority, "Open",
                      datetime.now(timezone.utc).isoformat()))
    sla = "1 working day" if priority == "Urgent" else "3 working days"
    return {"ok": True, "ticket_id": ticket_id, "student_id": sid, "category": category,
            "priority": priority, "status": "Open", "expected_response": sla}


def prepare_support_ticket(student_id: str, category: str, description: str, _ctx: ToolContext | None = None) -> dict:
    """Creates a DRAFT only. A ticket exists only after the student presses Confirm (see /api/tickets/confirm)."""
    rec, err = _lookup(student_id)
    if err: return err
    description = (description or "").strip()
    if category not in CATEGORIES: return {"ok": False, "error_type": "invalid_input", "error": f"category must be one of {CATEGORIES}."}
    if len(description) < 10: return {"ok": False, "error_type": "invalid_input", "error": "Ask the student to describe the problem in more detail (10+ characters)."}
    if memory.is_sensitive(description): return {"ok": False, "error_type": "invalid_input", "error": "Do not include passwords, OTPs or card numbers in a support request."}
    did = database.create_draft(rec["student_id"], getattr(_ctx, "conversation_id", None), category, description)
    return {"ok": True, "status": "awaiting_confirmation", "draft_id": did, "category": category, "description": description,
            "message": "Draft prepared. NO ticket exists yet; the student must press Confirm to submit it."}


FIELD_RULES = {"name": (r"[A-Za-z][A-Za-z .'-]{1,58}[A-Za-z.]", "your full name (letters, spaces, . ' - only)"),
               "course": (r"AI & DS|BDA", "the course: AI & DS or BDA"),
               "email": (r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "a valid email address"),
               "elective": (r".{3,80}", "the elective course name (3-80 characters)"),
               "fee_concession": (r".{10,200}", "the reason for the concession (10-200 characters)")}


def request_record_change(student_id: str, field: str, new_value: str) -> dict:
    """SENSITIVE action: never applied directly. Creates a pending request an administrator must decide."""
    rec, err = _lookup(student_id)
    if err: return err
    new_value = (new_value or "").strip()
    if field not in FIELD_RULES: return {"ok": False, "error_type": "invalid_input", "error": f"field must be one of {sorted(FIELD_RULES)}."}
    if field == "course":
        canon = memory.normalise_course(new_value)          # accepts "PGDM AI and Data Science", "bda", ...
        if not canon: return {"ok": False, "error_type": "invalid_input", "error": "The course must be AI & DS or BDA (the programme is always PGDM)."}
        new_value = canon
    pattern, what = FIELD_RULES[field]
    if not re.fullmatch(pattern, new_value) or memory.is_sensitive(new_value):
        return {"ok": False, "error_type": "invalid_input", "error": f"Please provide {what}."}
    aid = database.create_approval(rec["student_id"], field, new_value)
    row = database.get_approval(aid)
    return {"ok": True, "status": "Pending", "approval_id": aid, "request_id": row["request_id"], "request_type": row["type_label"],
            "message": "Request submitted. The change is NOT applied until an administrator approves it."}


def search_policy(query: str) -> dict:
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error_type": "invalid_input", "error": "query is empty."}
    chunks = retrieval.search_policy(query)
    return {"ok": True, "chunks": chunks,
            "note": None if chunks else
            "No relevant section found in the student handbook for this query."}


def save_memory(key: str, value: str, _ctx: ToolContext | None = None) -> dict:
    return memory.save_memory((_ctx or ToolContext()).user_id, key, value)


# ------------------------------------------------------------------ schemas (what the LLM sees)
def _fn(name, description, props, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required,
                       "additionalProperties": False}}}


_SID = {"type": "string", "description": "Student ID such as S1001"}

TOOL_SCHEMAS = [
    _fn("check_fee_status",
        "Get a student's fee status: total, paid, balance, next due date. "
        "Use for questions about the student's own fees or dues. Needs a student ID.",
        {"student_id": _SID}, ["student_id"]),
    _fn("check_scholarship_status",
        "Get the status of a student's scholarship application. Needs a student ID.",
        {"student_id": _SID}, ["student_id"]),
    _fn("get_student_record",
        "Get basic student details: name, programme (PGDM), course (AI & DS or BDA), semester, email. Needs a student ID.",
        {"student_id": _SID}, ["student_id"]),
    _fn("prepare_support_ticket",
        "Prepare a DRAFT support ticket after the student has described their problem. It does NOT submit anything: "
        "the student must press Confirm. Never call it before you know the problem.",
        {"student_id": _SID, "category": {"type": "string", "enum": CATEGORIES}, "description": {"type": "string"}},
        ["student_id", "category", "description"]),
    _fn("request_record_change",
        "Request a change to the student's own record: field name (correct their name), course (AI & DS or BDA), email, elective or fee_concession. Needs administrator "
        "approval and is NOT applied immediately. Use only when the student asks for such a change.",
        {"student_id": _SID, "field": {"type": "string", "enum": ["email", "elective", "fee_concession", "course", "name"]}, "new_value": {"type": "string"}},
        ["student_id", "field", "new_value"]),
    _fn("search_policy",
        "Search the student handbook (fees, late fees, payment failure, instalments, scholarships, "
        "course change, support/escalation, attendance, re-evaluation). Use for ANY question about "
        "rules, deadlines or procedures. Returns only the relevant sections.",
        {"query": {"type": "string", "description": "Short keyword query about the policy topic"}},
        ["query"]),
    _fn("save_memory",
        "Save a useful, NON-sensitive fact the student explicitly shared, for future sessions. "
        "Allowed keys: preferred_name, programme, student_id, preference. "
        "NEVER save passwords, OTPs, tokens, card numbers or temporary information.",
        {"key": {"type": "string", "enum": sorted(memory.ALLOWED_KEYS)},
         "value": {"type": "string"}}, ["key", "value"]),
]

TOOL_REGISTRY = {
    "check_fee_status": check_fee_status,
    "check_scholarship_status": check_scholarship_status,
    "get_student_record": get_student_record,
    "create_support_request": create_support_request,
    "prepare_support_ticket": prepare_support_ticket,
    "request_record_change": request_record_change,
    "search_policy": search_policy,
    "save_memory": save_memory,
}

# Metadata for Assignment 2 (authorization / approval policy will read this).
TOOL_META = {
    "check_fee_status": {"reads_private_data": True, "writes": False},
    "check_scholarship_status": {"reads_private_data": True, "writes": False},
    "get_student_record": {"reads_private_data": True, "writes": False},
    "create_support_request": {"reads_private_data": True, "writes": True},
    "prepare_support_ticket": {"reads_private_data": True, "writes": False},
    "request_record_change": {"reads_private_data": True, "writes": True, "needs_approval": True},
    "search_policy": {"reads_private_data": False, "writes": False},
    "save_memory": {"reads_private_data": False, "writes": True},
}


STUDENT_TOOLS = {"check_fee_status", "check_scholarship_status", "get_student_record", "create_support_request", "prepare_support_ticket", "request_record_change"}


def _audit(ctx, action, target, decision, result):
    try: database.audit(ctx.student_id or ctx.user_id, "student", action, target, decision, result)
    except Exception: pass   # auditing must never crash the agent loop


def execute_tool(name: str, args: dict, ctx: ToolContext | None = None) -> dict:
    """Generic executor (Multi_Tool_ReAct). Unknown tool, bad arguments or a crash become an error dict."""
    ctx = ctx or ToolContext()
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error_type": "unknown_tool", "error": f"Unknown tool: {name}"}
    if not isinstance(args, dict):
        return {"ok": False, "error_type": "invalid_input", "error": "Tool arguments must be an object."}
    if ctx.student_id and name in STUDENT_TOOLS:
        sid = str(args.get("student_id") or ctx.student_id).strip().upper()
        if sid != ctx.student_id:   # authorization enforced in code, not by the prompt
            _audit(ctx, name, sid, "blocked", "forbidden: another student's record")
            return {"ok": False, "error_type": "forbidden",
                    "error": "Access denied: you can only access your own student record."}
        args = {**args, "student_id": sid}
    try:
        result = fn(**args, _ctx=ctx) if name in ("save_memory", "prepare_support_ticket") else fn(**args)
        if name in ("create_support_request", "request_record_change", "prepare_support_ticket"):
            _audit(ctx, name, args.get("student_id", ""), "allowed", "ok " + str(result.get("ticket_id") or result.get("request_id") or result.get("draft_id") or "") if result.get("ok") else "failed: " + str(result.get("error")))
        return result
    except TypeError as e:
        return {"ok": False, "error_type": "invalid_input", "error": f"Bad arguments: {e}"}
    except Exception as e:  # noqa: BLE001 - tools must never crash the agent loop
        return {"ok": False, "error_type": "tool_failure", "error": f"Tool crashed: {e}"}
