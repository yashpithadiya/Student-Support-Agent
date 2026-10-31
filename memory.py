"""Persistent memory (SQLite) with three deliberate controls:

SAVE      only a small whitelist of keys can be stored, and `is_sensitive()` rejects
          anything that looks like a password / OTP / token / card number.
          This is enforced here in code - NOT just by a prompt instruction.
          Raw chat messages are never stored here; only explicit facts the agent chose to save.
RETRIEVE  `retrieve_relevant()` returns only items whose words overlap with the CURRENT
          question (max MEMORY_TOP_K), never the whole memory table.
LIFETIME  rows persist in the SQLite file, so they survive restarts / new chat sessions.
"""
import re
import sqlite3
from datetime import datetime, timezone

import config

# key -> single-valued (new value replaces old) or multi-valued
ALLOWED_KEYS = {
    "preferred_name": "single",   # how the student wants to be addressed
    "programme": "single",        # always "PGDM"
    "course": "single",           # "AI & DS" or "BDA"
    "student_id": "single",       # demo identifier (Assignment 2 replaces this with real login)
    "preference": "multi",        # explicit preferences, e.g. "prefers short answers"
}
MAX_VALUE_LEN = 120

_SENSITIVE_WORDS = re.compile(
    r"\b(otp|one[- ]?time|password|passcode|passwd|pin|cvv|cvc|token|api[ _-]?key|"
    r"secret|card number|credit card|debit card|aadhaar|aadhar|pan number|"
    r"bank account|account number|login|security answer)\b", re.I)
_LONG_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d{4,}(?![A-Za-z0-9])")  # standalone 4+ digit run


def is_sensitive(text: str) -> bool:
    """True if the text looks like a credential, OTP, token or financial identifier."""
    return bool(_SENSITIVE_WORDS.search(text or "") or _LONG_NUMBER.search(text or ""))


def _conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    _migrate_programme(conn)
    return conn


def _migrate_programme(conn):
    """Safe, idempotent: old combined values ("PGDM Data Science & AI") become programme=PGDM + course=AI & DS / BDA."""
    for r in conn.execute("SELECT id,user_id,value FROM memories WHERE key='programme' AND value!='PGDM'").fetchall():
        course = "AI & DS" if re.search(r"data science|\bai\b", r["value"], re.I) else "BDA" if re.search(r"big data|bda", r["value"], re.I) else None
        conn.execute("UPDATE memories SET value='PGDM' WHERE id=?", (r["id"],))
        if course and not conn.execute("SELECT 1 FROM memories WHERE user_id=? AND key='course'", (r["user_id"],)).fetchone():
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("INSERT INTO memories (user_id,key,value,created_at,updated_at) VALUES (?,?,?,?,?)", (r["user_id"], "course", course, now, now))
    conn.commit()


COURSE_ALIASES = {"ai & ds": "AI & DS", "ai and ds": "AI & DS", "aids": "AI & DS", "ai&ds": "AI & DS", "bda": "BDA",
                  "artificial intelligence & data science": "AI & DS", "data science & ai": "AI & DS", "big data analytics": "BDA", "ai and data science": "AI & DS", "ai & data science": "AI & DS", "data science and ai": "AI & DS",
                  "bigdata analytics": "BDA", "ai/ds": "AI & DS", "ai ds": "AI & DS"}


def normalise_course(value: str):
    v = re.sub(r"^\s*pgdm[\s,:\-]*", "", (value or "").strip(), flags=re.I)      # "PGDM AI and Data Science" -> course part only
    return COURSE_ALIASES.get(re.sub(r"\s+", " ", v.strip().rstrip(".")).lower())


def save_memory(user_id: str, key: str, value: str) -> dict:
    """Save one fact. Returns {"ok": bool, ...}; never raises for bad input."""
    key = (key or "").strip().lower()
    value = (value or "").strip()
    if key not in ALLOWED_KEYS:
        return {"ok": False, "error": f"Cannot store '{key}'. Allowed keys: {sorted(ALLOWED_KEYS)}."}
    if not value or len(value) > MAX_VALUE_LEN:
        return {"ok": False, "error": f"Value must be 1-{MAX_VALUE_LEN} characters."}
    if is_sensitive(value) or is_sensitive(key):
        return {"ok": False, "refused": True,
                "error": "Refused: this looks like sensitive information "
                         "(password/OTP/token/number) and is never stored as long-term memory."}
    if key == "programme":
        if value.strip().upper() != "PGDM": return {"ok": False, "error": "The programme is always PGDM."}
        value = "PGDM"
    if key == "course":
        value = normalise_course(value)
        if not value: return {"ok": False, "error": "The course must be AI & DS or BDA."}
    if key == "student_id" and not re.fullmatch(r"S\d{4}", value.upper()):
        return {"ok": False, "error": "student_id must look like S1001."}
    if key == "student_id":
        value = value.upper()
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        if ALLOWED_KEYS[key] == "single":
            conn.execute("DELETE FROM memories WHERE user_id=? AND key=?", (user_id, key))
        elif conn.execute("SELECT 1 FROM memories WHERE user_id=? AND key=? AND value=?",
                          (user_id, key, value)).fetchone():
            return {"ok": True, "saved": {"key": key, "value": value}, "note": "already stored"}
        conn.execute("INSERT INTO memories (user_id,key,value,created_at,updated_at) "
                     "VALUES (?,?,?,?,?)", (user_id, key, value, now, now))
    return {"ok": True, "saved": {"key": key, "value": value}}


def list_memories(user_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT id,key,value FROM memories WHERE user_id=? ORDER BY id",
                            (user_id,)).fetchall()
    return [dict(r) for r in rows]


def clear_memories(user_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM memories WHERE user_id=?", (user_id,))


# ---------------- relevance filtering ----------------
_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or", "what",
         "how", "do", "does", "did", "can", "i", "it", "that", "this", "with", "about", "be"}
_PERSONAL = {"my", "me", "mine", "i", "i'm", "am", "myself"}
# memory key -> (words that make it relevant, must ALSO include a personal word?)
_TRIGGERS = {
    "preferred_name": ({"name", "call", "called", "address", "greet", "who"}, False),
    "programme": ({"programme", "program", "batch", "enrolled", "studying", "semester", "pgdm", "elective"}, False),
    "course": ({"course", "specialization", "specialisation", "stream", "branch", "bda", "ai", "ds"}, False),
    "student_id": ({"fee", "fees", "scholarship", "status", "balance", "due", "record",
                    "payment", "paid", "ticket", "request", "id", "details", "pay"}, True),
    "preference": ({"prefer", "preference", "format", "short", "detailed", "language",
                    "style", "reminder", "contact"}, False),
}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def retrieve_relevant(user_id: str, question: str, k: int | None = None) -> list[dict]:
    """Return at most k memory items relevant to THIS question (not the whole table)."""
    k = k or config.MEMORY_TOP_K
    q = _tokens(question)
    scored = []
    for m in list_memories(user_id):
        triggers, need_personal = _TRIGGERS[m["key"]]
        score = len(q & triggers)
        if m["key"] == "preference":                       # also match on the stored text itself
            score += len((q - _STOP) & _tokens(m["value"]))
        if need_personal and not (q & _PERSONAL):
            score = 0
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda s: -s[0])
    return [m for _, m in scored[:k]]
