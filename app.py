"""Flask app: student chat + admin dashboard.  Run: python app.py  (DEMO_MOCK=1 python app.py for a mock-model preview)"""
import csv, json, os, secrets, subprocess, sys
import accounts, agent, config, database, memory, retrieval, tools
from flask import Flask, jsonify, redirect, render_template, request, send_file, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or secrets.token_hex(32)
VERSION = "2.0"
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        if os.environ.get("DEMO_MOCK") == "1":
            from evaluation.mock_llm import MockLLM; _llm = MockLLM()
        else:
            from llm import GroqLLM; _llm = GroqLLM()
    return _llm

def uid(): return session["sid"]
def role(): return session.get("role")
def err(msg, code): return jsonify(error=msg), code

# ---------------------------------------------------------------- access control (server-side)
@app.before_request
def gate():
    p = request.path
    if p in ("/login", "/api/health") or p.startswith("/static"): return None
    api = p.startswith("/api/")
    if "sid" not in session:
        return err("Please sign in.", 401) if api else redirect("/login")
    admin_area = p.startswith("/admin") or p.startswith("/api/admin")
    if role() == "admin" and not (admin_area or p in ("/logout", "/api/me")):
        return err("Administrator accounts use the admin dashboard.", 403) if api else redirect("/admin")
    if role() != "admin" and admin_area:
        return err("Forbidden.", 403) if api else redirect("/")
    return None

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    accounts.ensure_seeded()
    d = request.get_json(silent=True) or {}
    r = "admin" if d.get("role") == "admin" else "student"       # role comes from the tab, but is checked against the DB
    acc = accounts.authenticate(d.get("identifier"), d.get("password"), r)
    if not acc:
        return err("Incorrect ID/email or password.", 401)        # same message for every failure
    session.clear(); session["sid"] = acc["login_id"]; session["role"] = acc["role"]; session.permanent = bool(d.get("remember"))
    database.audit(acc["login_id"], acc["role"], "login", "-", "allowed", "signed in")
    return jsonify(ok=True, redirect="/admin" if r == "admin" else "/")

@app.get("/logout")
def logout():
    if "sid" in session: database.audit(uid(), role(), "logout", "-", "allowed", "signed out")
    session.clear(); return redirect("/login")

@app.get("/")
def index():
    a = accounts.get(uid()) or {"name": uid(), "email": ""}
    return render_template("index.html", name=a["name"].split()[0], full=a["name"], login_id=uid(), email=a["email"],
                           mock=os.environ.get("DEMO_MOCK") == "1",
                           problems=[p for p in config.validate_config() if not (os.environ.get("DEMO_MOCK") == "1" and "GROQ" in p)])

@app.get("/evaluation")
def old_evaluation(): return redirect("/")       # dashboard removed from the UI; evaluation scripts remain

@app.get("/api/me")
def me():
    a = accounts.get(uid()) or {}
    return jsonify(name=a.get("name"), role=a.get("role"), login_id=uid(), email=a.get("email"))

# ---------------------------------------------------------------- chat + history
@app.post("/api/chat")
def chat():
    d = request.get_json(silent=True) or {}
    msg = (d.get("message") or "").strip()
    if not msg or len(msg) > 2000: return err("Message must be 1-2000 characters.", 400)
    cid = d.get("conversation_id")
    if cid and not database.exists(cid, uid()): return err("Unknown conversation.", 404)
    try:
        cid = cid or database.create_conversation(uid())
        hist = [{"role": m["role"], "content": m["content"]} for m in database.get_messages(cid)]
        database.add_message(cid, "user", msg)
        r = agent.run_agent(msg, hist, uid(), get_llm(), student_id=uid(), conversation_id=cid)
        t = r.trace.to_dict()
        database.add_message(cid, "assistant", r.answer, t["run_id"])
    except Exception:
        app.logger.exception("chat failed")
        return err("Something went wrong. Please try again.", 500)
    u = t["token_usage"]
    drafts = [e["result"] for e in t["events"] if e["type"] == "tool_call" and e["tool"] == "prepare_support_ticket" and e["status"] == "success"]
    return jsonify(draft=drafts[-1] if drafts else None, drafts=drafts, success=not t["stopping_reason"].startswith("llm_error"), response=r.answer, conversation_id=cid,
                   trace_id=t["run_id"], model=t["model"], latency_ms=t["latency_ms"],
                   usage={"input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"],
                          "total_tokens": (u["input_tokens"] + u["output_tokens"]) if u["available"] else None},
                   tool_events=[e for e in t["events"] if e["type"] == "tool_call"], sources=t["retrieved_chunks"],
                   memory_references=t["memory_used"], status=t["stopping_reason"], error=("; ".join(t["errors"]) or None), trace=t)

@app.get("/api/history")
def history(): return jsonify(database.list_conversations(uid()))

@app.delete("/api/history/<cid>")
def history_delete(cid):
    return (jsonify(ok=True) if database.delete_conversation(cid, uid()) else err("Not found", 404))

@app.get("/api/history/<cid>")
def history_one(cid):
    return jsonify(id=cid, messages=database.get_messages(cid), pending_drafts=database.list_drafts(cid, uid())) if database.exists(cid, uid()) else err("Not found", 404)

def _traces():
    if not config.TRACE_FILE.exists(): return []
    return [json.loads(l) for l in config.TRACE_FILE.read_text(encoding="utf-8").splitlines() if l.strip()][::-1]

@app.get("/api/traces")
def traces(): return jsonify([t for t in _traces() if t.get("user_id") == uid()])

# ---------------------------------------------------------------- student data (identity from the SESSION only)
PROFILE_FIELDS = ("student_id", "name", "email", "programme", "course", "semester", "elective", "fee_total", "fee_paid", "fee_due_date",
                  "fee_status", "fee_concession", "scholarship_name", "scholarship_status", "scholarship_note")

@app.get("/api/profile")
def profile():
    rec = tools._load_students().get(uid())
    return jsonify(record={k: rec.get(k) for k in PROFILE_FIELDS} if rec else None, approvals=database.list_approvals(uid(), 10))

@app.get("/api/my/requests")
def my_requests(): return jsonify(database.list_approvals(uid()))      # own requests only

@app.get("/api/my/tickets")
def my_tickets(): return jsonify(database.list_tickets(uid()))

_REVOKE = {"ok": (200, None), "not_found": (404, "Not found."), "conflict": (409, "This can no longer be revoked.")}

@app.post("/api/my/tickets/<tid>/revoke")
def revoke_ticket(tid):
    res = database.revoke_ticket(tid, uid()); code, msg = _REVOKE[res]
    return (jsonify(ok=True) if res == "ok" else err("Only tickets that are still Open can be revoked." if res == "conflict" else msg, code))

@app.post("/api/my/requests/<int:aid>/revoke")
def revoke_request(aid):
    res = database.revoke_approval(aid, uid()); code, msg = _REVOKE[res]
    return (jsonify(ok=True) if res == "ok" else err("Only pending requests can be revoked." if res == "conflict" else msg, code))

@app.post("/api/requests")
def new_request():
    d = request.get_json(silent=True) or {}
    r = tools.execute_tool("request_record_change", {"student_id": uid(), "field": d.get("field", ""), "new_value": d.get("value", "")}, tools.ToolContext(uid(), uid()))
    return jsonify(r), (201 if r["ok"] else 400)

# ---- support tickets: confirm / cancel a draft ----
def _ticket_payload(tid):
    t = next((x for x in database.list_tickets(uid()) if x["ticket_id"] == tid), None)
    return {**t, "expected_response": "1 working day" if t["priority"] == "Urgent" else "3 working days"} if t else None

@app.post("/api/tickets/confirm")
def ticket_confirm():
    did = (request.get_json(silent=True) or {}).get("draft_id")
    d = database.get_draft(did) if isinstance(did, int) else None
    if not d or d["student_id"] != uid(): return err("Draft not found.", 404)
    if d["status"] == "confirmed": return jsonify(ok=True, duplicate=True, ticket=_ticket_payload(d["ticket_id"]))   # double-click / retry
    if d["status"] == "cancelled": return err("This draft was cancelled.", 409)
    if not database.claim_draft(did, uid()): return err("This request is already being submitted.", 409)
    try:
        r = tools.execute_tool("create_support_request", {"student_id": uid(), "category": d["category"], "description": d["description"]}, tools.ToolContext(uid(), uid()))
    except Exception:
        r = {"ok": False, "error": "Could not create the ticket."}
    if not r["ok"]:
        database.finish_draft(did, "pending"); return err(r.get("error", "Could not create the ticket."), 400)
    database.finish_draft(did, "confirmed", r["ticket_id"])
    return jsonify(ok=True, ticket=_ticket_payload(r["ticket_id"]))

@app.post("/api/tickets/cancel")
def ticket_cancel():
    did = (request.get_json(silent=True) or {}).get("draft_id")
    d = database.get_draft(did) if isinstance(did, int) else None
    if not d or d["student_id"] != uid(): return err("Draft not found.", 404)
    if d["status"] != "pending": return err("This draft can no longer be cancelled.", 409)
    database.finish_draft(did, "cancelled"); database.audit(uid(), "student", "ticket_draft_cancelled", f"draft {did}", "allowed", "no ticket created")
    return jsonify(ok=True)

@app.get("/api/student")
def student(): return jsonify(memory=memory.list_memories(uid()))

@app.delete("/api/memory")
def clear_memory():
    memory.clear_memories(uid()); database.audit(uid(), "student", "clear_memory", "-", "allowed", "ok"); return jsonify(ok=True)

@app.post("/api/support")
def support():
    d = request.get_json(silent=True) or {}
    r = tools.execute_tool("create_support_request", {"student_id": d.get("student_id") or uid(), "category": d.get("category", "other"),
                                                      "description": d.get("description", "")}, tools.ToolContext(uid(), uid()))
    return jsonify(r), (200 if r["ok"] else 403 if r.get("error_type") == "forbidden" else 400)

# ---------------------------------------------------------------- policies
CATS = [("Fees and Payments", ("fee", "payment", "refund", "hardship", "receipt", "instalment")),
        ("Scholarships", ("scholarship",)), ("Courses and Academics", ("course", "grading", "attendance", "examination", "backlog", "leave", "integrity")),
        ("Student Support and Records", ("support", "correction", "identity", "grievance", "privacy"))]

def _cat(title):
    t = title.lower()
    return next((c for c, ks in CATS if any(k in t for k in ks)), "Other")

@app.get("/api/policies")
def policies():
    out = []
    for c in retrieval.load_chunks():
        first = c["text"].split(". ")[0].strip().rstrip(".") + "."
        out.append({**c, "category": _cat(c["title"]), "summary": first, "source": "student_policy.md"})
    return jsonify(out)

@app.get("/policies/download")
def dl(): return send_file(config.POLICY_MD, as_attachment=True)

# ---------------------------------------------------------------- admin
@app.get("/admin")
def admin_page():
    a = accounts.get(uid()) or {}
    return render_template("admin.html", name=a.get("name", uid()), login_id=uid(), email=a.get("email", ""))

def _names():
    return {k: v["name"] for k, v in tools._load_students().items()}

@app.get("/api/admin/approvals")
def admin_approvals():
    n = _names()
    return jsonify([{**r, "student_name": n.get(r["student_id"], "?")} for r in database.list_approvals()])

@app.get("/api/admin/approvals/<int:aid>")
def admin_approval_one(aid):
    r = database.get_approval(aid)
    if not r: return err("Not found", 404)
    s = tools._load_students().get(r["student_id"], {})
    return jsonify({**r, "student_name": s.get("name"), "current_value": s.get(r["field"]) or "—"})

@app.post("/api/admin/approvals/<int:aid>")
def admin_decide(aid):
    d = request.get_json(silent=True) or {}
    if d.get("decision") not in ("approve", "reject"): return err("decision must be 'approve' or 'reject'.", 400)
    if d["decision"] == "reject" and not (d.get("reason") or "").strip(): return err("A reason is required when rejecting.", 400)
    try:
        r = database.decide_approval(aid, uid(), d["decision"] == "approve", d.get("reason", ""))
    except Exception:
        app.logger.exception("decision failed"); return err("The decision could not be saved; nothing was changed.", 500)
    return jsonify(ok=True, approval=r) if r else err("Request not found or already decided.", 409)

@app.get("/api/admin/audit")
def admin_audit(): return jsonify(database.list_audit(500))

@app.get("/api/admin/students")
def admin_students():
    q = (request.args.get("q") or "").strip().lower()
    keys = ("student_id", "name", "email", "programme", "course", "semester", "fee_status", "scholarship_status")
    return jsonify([{k: r[k] for k in keys} for r in tools._load_students().values() if not q or q in (r["student_id"] + r["name"] + r["email"]).lower()])

@app.get("/api/admin/students/<sid>")
def admin_student(sid):
    r = tools._load_students().get(sid.upper())
    return jsonify({k: v for k, v in r.items() if k != "fee_service"}) if r else err("Not found", 404)

@app.get("/api/admin/tickets")
def admin_tickets():
    n = _names(); return jsonify([{**t, "student_name": n.get(t["student_id"], "?")} for t in database.list_tickets()])

@app.post("/api/admin/tickets/<tid>")
def admin_ticket_update(tid):
    ok = database.update_ticket_status(tid, (request.get_json(silent=True) or {}).get("status"), uid())
    return jsonify(ok=True) if ok else err("Unknown ticket or invalid status.", 400)

@app.get("/api/admin/policies")
def admin_policies(): return policies()

@app.get("/api/admin/traces")
def admin_traces(): return jsonify(_traces()[:100])

@app.get("/api/admin/overview")
def admin_overview():
    ap, tk, tr = database.list_approvals(), database.list_tickets(), _traces()
    n = _names(); lat = [t["latency_ms"] for t in tr]; tu = [t["token_usage"] for t in tr]
    toks = sum((u["input_tokens"] or 0) + (u["output_tokens"] or 0) for u in tu) if tu and all(u["available"] for u in tu) else None
    return jsonify(pending=sum(r["status"] == "pending" for r in ap), open_tickets=sum(t["status"] in ("Open", "In Progress") for t in tk), students=len(n),
                   agent={"runs": len(tr), "avg_latency_ms": round(sum(lat) / len(lat)) if lat else None, "total_tokens": toks},
                   pending_requests=[{**r, "student_name": n.get(r["student_id"])} for r in ap if r["status"] == "pending"][:3],
                   recent_tickets=tk[:3], recent_audit=database.list_audit(4))

# ---------------------------------------------------------------- evaluation (API kept; UI page removed)
def _eval_file():
    d = config.BASE_DIR / "evaluation"
    for name, mock in (("results.csv", False), ("results_mock.csv", True)):
        if (d / name).exists(): return d / name, mock
    return None, False

@app.get("/api/admin/evaluation")
def evaluation():
    p, mock = _eval_file()
    return jsonify(mock=mock, rows=list(csv.DictReader(open(p, encoding="utf-8"))) if p else [])

@app.get("/api/admin/evaluation/export")
def eval_export():
    p, _ = _eval_file()
    return send_file(p, as_attachment=True) if p else err("No results yet.", 404)

@app.post("/api/admin/evaluation/run")
def eval_run():
    mock = os.environ.get("DEMO_MOCK") == "1"
    if not mock and not config.GROQ_API_KEY: return err("GROQ_API_KEY is not set, so live evaluation is unavailable.", 503)
    p = subprocess.run([sys.executable, "-m", "evaluation.evaluate", "--mock" if mock else "--live"], cwd=config.BASE_DIR, capture_output=True, text=True, timeout=900)
    return (jsonify(ok=True, mode="mock" if mock else "live"), 200) if p.returncode == 0 else err("Evaluation run failed.", 500)

@app.get("/api/health")
def health():
    return jsonify(status="ok", version=VERSION, model=config.MODEL, api_key_configured=bool(config.GROQ_API_KEY), setup_problems=config.validate_config())

if __name__ == "__main__":
    accounts.ensure_seeded()
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
