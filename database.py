"""Chat history (conversations + messages) in SQLite. Separate from persistent memory."""
import sqlite3, uuid
from datetime import datetime, timezone
import config

def _c():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH); c.row_factory = sqlite3.Row
    c.executescript("""CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY,title TEXT,created_at TEXT,updated_at TEXT);
    CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,conversation_id TEXT,role TEXT,content TEXT,trace_id TEXT,created_at TEXT);""")
    c.executescript("""CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,actor TEXT,role TEXT,action TEXT,target TEXT,decision TEXT,result TEXT);
    CREATE TABLE IF NOT EXISTS approvals(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id TEXT,field TEXT,new_value TEXT,status TEXT,requested_at TEXT,decided_by TEXT,decided_at TEXT);
    CREATE TABLE IF NOT EXISTS overrides(student_id TEXT,field TEXT,value TEXT,PRIMARY KEY(student_id,field));
    CREATE TABLE IF NOT EXISTS accounts(login_id TEXT PRIMARY KEY,email TEXT,name TEXT,role TEXT,password_hash TEXT,status TEXT);""")
    c.executescript("""CREATE TABLE IF NOT EXISTS support_tickets(ticket_id TEXT PRIMARY KEY,student_id TEXT,category TEXT,description TEXT,priority TEXT,status TEXT,created_at TEXT);
    CREATE TABLE IF NOT EXISTS ticket_drafts(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id TEXT,conversation_id TEXT,category TEXT,description TEXT,status TEXT,ticket_id TEXT,created_at TEXT);""")
    for col in ("request_type", "details", "decision_reason"):
        try: c.execute(f"ALTER TABLE approvals ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError: pass
    try: c.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError: pass
    return c

def now(): return datetime.now(timezone.utc).isoformat()

def create_conversation(user_id, title="New chat"):
    cid = uuid.uuid4().hex[:12]
    with _c() as c: c.execute("INSERT INTO conversations(id,title,created_at,updated_at,user_id) VALUES(?,?,?,?,?)", (cid, title, now(), now(), user_id))
    return cid

def add_message(cid, role, content, trace_id=None):
    with _c() as c:
        c.execute("INSERT INTO messages(conversation_id,role,content,trace_id,created_at) VALUES(?,?,?,?,?)", (cid, role, content, trace_id, now()))
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now(), cid))
        if role == "user" and c.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=? AND role='user'", (cid,)).fetchone()[0] == 1:
            c.execute("UPDATE conversations SET title=? WHERE id=?", (content[:60], cid))

def exists(cid, user_id):
    with _c() as c: return c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (cid, user_id)).fetchone() is not None

def list_conversations(user_id, limit=50):
    with _c() as c: return [dict(r) for r in c.execute("SELECT id,title,created_at,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit))]

def get_messages(cid):
    with _c() as c: return [dict(r) for r in c.execute("SELECT role,content,trace_id,created_at FROM messages WHERE conversation_id=? ORDER BY id", (cid,))]


# ---------------- audit log, approvals (Assignment 2) ----------------
def audit(actor, role, action, target, decision, result):
    with _c() as c:
        c.execute("INSERT INTO audit_log(ts,actor,role,action,target,decision,result) VALUES(?,?,?,?,?,?,?)",
                  (now(), actor, role, action, str(target)[:200], decision, str(result)[:300]))

def list_audit(limit=100):
    with _c() as c: return [dict(r) for r in c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]

REQUEST_TYPES = {"name": "Correct Name", "course": "Change Course", "email": "Change Contact Email", "elective": "Change Elective", "fee_concession": "Fee Concession"}
TICKET_STATUSES = ("Open", "In Progress", "Resolved", "Closed")

def _req_view(r):
    r = dict(r); r["request_id"] = f"REQ-{(r['requested_at'] or '0000')[:4]}-{r['id']:03d}"; r["type_label"] = REQUEST_TYPES.get(r["field"], r["field"]); return r

def create_approval(sid, field, value):
    with _c() as c:
        dup = c.execute("SELECT id FROM approvals WHERE student_id=? AND field=? AND new_value=? AND status='pending'", (sid, field, value)).fetchone()
        if dup: return dup["id"]          # no duplicate pending requests
        return c.execute("INSERT INTO approvals(student_id,field,new_value,status,requested_at,request_type) VALUES(?,?,?,'pending',?,?)", (sid, field, value, now(), field)).lastrowid

def list_approvals(student_id=None, limit=200):
    with _c() as c:
        q, a = "SELECT * FROM approvals", ()
        if student_id: q, a = q + " WHERE student_id=?", (student_id,)
        return [_req_view(r) for r in c.execute(q + " ORDER BY id DESC LIMIT ?", a + (limit,))]

def get_approval(aid):
    with _c() as c: r = c.execute("SELECT * FROM approvals WHERE id=?", (aid,)).fetchone()
    return _req_view(r) if r else None

def decide_approval(aid, admin_id, approve, reason=""):
    """ONE transaction: status change + (if approved) record change + audit entry. Any failure rolls everything back."""
    with _c() as c:
        r = c.execute("SELECT * FROM approvals WHERE id=?", (aid,)).fetchone()
        if not r: return None
        cur = c.execute("UPDATE approvals SET status=?,decided_by=?,decided_at=?,decision_reason=? WHERE id=? AND status='pending'",
                        ("approved" if approve else "rejected", admin_id, now(), (reason or "")[:300], aid))
        if cur.rowcount != 1: return None            # already decided -> no conflicting second decision
        if approve:
            c.execute("INSERT OR REPLACE INTO overrides VALUES(?,?,?)", (r["student_id"], r["field"], r["new_value"]))
            if r["field"] == "name":      # keep the sign-in display name in step with the approved correction
                c.execute("UPDATE accounts SET name=? WHERE login_id=?", (r["new_value"], r["student_id"]))
        c.execute("INSERT INTO audit_log(ts,actor,role,action,target,decision,result) VALUES(?,?,?,?,?,?,?)",
                  (now(), admin_id, "admin", "approval_decision", f"REQ-{r['requested_at'][:4]}-{aid:03d} {r['student_id']}.{r['field']}",
                   "approved" if approve else "rejected", "change applied" if approve else "not executed"))
    return get_approval(aid)

# ---- support tickets (same table the tools write to) ----
def list_tickets(student_id=None, limit=200):
    with _c() as c:
        q, a = "SELECT * FROM support_tickets", ()
        if student_id: q, a = q + " WHERE student_id=?", (student_id,)
        return [dict(r) for r in c.execute(q + " ORDER BY created_at DESC LIMIT ?", a + (limit,))]

def update_ticket_status(tid, status, admin_id):
    if status not in TICKET_STATUSES: return None
    with _c() as c:
        old = c.execute("SELECT status FROM support_tickets WHERE ticket_id=?", (tid,)).fetchone()
        if not old or old["status"] == "Revoked": return None     # revoked by the student: final
        c.execute("UPDATE support_tickets SET status=? WHERE ticket_id=?", (status, tid))
        c.execute("INSERT INTO audit_log(ts,actor,role,action,target,decision,result) VALUES(?,?,?,?,?,?,?)",
                  (now(), admin_id, "admin", "ticket_status_update", tid, "allowed", f"{old['status']} -> {status}"))
    return True

# ---- support-ticket drafts: the agent may only DRAFT; a ticket exists after the student confirms ----
def create_draft(sid, cid, category, description):
    with _c() as c:
        dup = c.execute("SELECT id FROM ticket_drafts WHERE student_id=? AND IFNULL(conversation_id,'')=IFNULL(?,'') AND description=? AND status='pending'", (sid, cid, description)).fetchone()
        if dup: return dup["id"]
        return c.execute("INSERT INTO ticket_drafts(student_id,conversation_id,category,description,status,created_at) VALUES(?,?,?,?,'pending',?)", (sid, cid, category, description, now())).lastrowid

def get_draft(did):
    with _c() as c: r = c.execute("SELECT * FROM ticket_drafts WHERE id=?", (did,)).fetchone()
    return dict(r) if r else None

def claim_draft(did, sid):
    """Atomic pending -> confirming; only ONE concurrent/double-click request can win."""
    with _c() as c: return c.execute("UPDATE ticket_drafts SET status='confirming' WHERE id=? AND student_id=? AND status='pending'", (did, sid)).rowcount == 1

def finish_draft(did, status, ticket_id=None):
    with _c() as c: c.execute("UPDATE ticket_drafts SET status=?,ticket_id=? WHERE id=?", (status, ticket_id, did))

def list_drafts(cid, sid, status="pending"):
    with _c() as c: return [dict(r) for r in c.execute("SELECT * FROM ticket_drafts WHERE conversation_id=? AND student_id=? AND status=?", (cid, sid, status))]

def get_overrides():
    with _c() as c: rows = c.execute("SELECT * FROM overrides").fetchall()
    out = {}
    for r in rows: out.setdefault(r["student_id"], {})[r["field"]] = r["value"]
    return out


# ---------------- student self-service: revoke a ticket / request, delete a chat ----------------
def _audit_in(c, actor, action, target, result):
    c.execute("INSERT INTO audit_log(ts,actor,role,action,target,decision,result) VALUES(?,?,?,?,?,?,?)", (now(), actor, "student", action, target, "allowed", result))

def revoke_ticket(tid, sid):
    """Student withdraws their OWN ticket, only while it is still Open. Returns 'ok' | 'not_found' | 'conflict'."""
    with _c() as c:
        t = c.execute("SELECT student_id FROM support_tickets WHERE ticket_id=?", (tid,)).fetchone()
        if not t or t["student_id"] != sid: return "not_found"
        if c.execute("UPDATE support_tickets SET status='Revoked' WHERE ticket_id=? AND student_id=? AND status='Open'", (tid, sid)).rowcount != 1: return "conflict"
        _audit_in(c, sid, "ticket_revoked", tid, "Open -> Revoked by student")
    return "ok"

def revoke_approval(aid, sid):
    """Student withdraws their OWN request, only while it is still pending (nothing was applied)."""
    with _c() as c:
        r = c.execute("SELECT student_id,requested_at FROM approvals WHERE id=?", (aid,)).fetchone()
        if not r or r["student_id"] != sid: return "not_found"
        if c.execute("UPDATE approvals SET status='revoked',decided_by=?,decided_at=?,decision_reason='Withdrawn by student' WHERE id=? AND student_id=? AND status='pending'", (sid, now(), aid, sid)).rowcount != 1: return "conflict"
        _audit_in(c, sid, "request_revoked", f"REQ-{r['requested_at'][:4]}-{aid:03d}", "pending -> revoked by student")
    return "ok"

def delete_conversation(cid, sid):
    """Deletes one of the student's OWN chats (messages included). Pending ticket drafts from it are cancelled; tickets/requests already created are kept."""
    with _c() as c:
        if not c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (cid, sid)).fetchone(): return False
        c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        c.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, sid))
        c.execute("UPDATE ticket_drafts SET status='cancelled' WHERE conversation_id=? AND student_id=? AND status='pending'", (cid, sid))
        _audit_in(c, sid, "chat_deleted", cid, "conversation and messages removed")
    return True
