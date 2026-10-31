"""The agent: a framework-free ReAct-style loop (Multi_Tool_ReAct / Unit II notebooks).

    select context -> LLM decides -> (tool request -> Python executes -> observation -> LLM decides)* -> answer

Nothing here fixes the order of tool calls: the model chooses whether to answer directly,
search the handbook, call a student-data tool, save memory, or ask the student for missing info.
The loop stops on: final answer, MAX_TURNS, or an unrecoverable LLM error."""
import json
import time

import difflib
import re

import config
import database
import memory
import tools
from llm import LLMError
from tracing import Trace

SYSTEM_PROMPT = """You are the Student Support Agent for a fictional college (demonstration data).
You help students with fees, scholarships, course changes and support requests.

Rules:
- Student-specific facts (fees, scholarship, record) MUST come from a tool result. Never guess them.
- Those tools need a student ID like S1001. If it is not in the conversation or the
  'Remembered facts' below, ask the student for it. Do not call the tool without it.
- For ANY question about rules, deadlines, fees policy or procedures, call search_policy first and
  answer ONLY from the returned sections. If nothing relevant is returned, say the handbook does not
  contain that information and offer to create a support request. Never invent policies or deadlines.
- Support tickets: when a student asks to raise a ticket, FIRST ask what issue they are facing (\"Sure! I can help you raise a support ticket. What issue are you facing?\") and do NOT call any tool yet. Once you know the problem (and category), call prepare_support_ticket, summarise the details and ask \"Would you like me to submit this support request?\". You cannot submit tickets: the student presses Confirm or Cancel. Never claim a ticket exists.
- If a tool returns an error, tell the student plainly what went wrong and what they can do next.
- Use save_memory only when the student explicitly shares a stable, non-sensitive fact (preferred name,
  programme, student ID, or a preference). NEVER save passwords, OTPs, tokens, card or bank numbers or
  one-off temporary details. If the student shares such a secret, tell them not to share it in chat.
- General or small-talk questions can be answered directly without tools. Politely decline requests
  unrelated to student support.
- For ordered steps use ONE numbered list with sequential numbers (1., 2., 3.) and indented bullets under each step.
- Changes to the student's OWN record (name, course AI & DS/BDA, email, elective, fee concession) are submitted with request_record_change: it creates a Pending request for administrator approval. Never say a name or course cannot be changed: submit the request and say it needs approval. Other problems use prepare_support_ticket.
- If ONE message contains SEVERAL asks (questions, changes, tickets), handle EVERY one: call the needed tools for each ask in the same turn, answer all questions, and report each request ID or draft. Never drop, merge or postpone an ask.
- Write answers in Markdown: use a table for comparisons, bullet or numbered lists for steps, and name the handbook section you used. Never output raw JSON or tool payloads.
- The programme is PGDM; the only courses are AI & DS and BDA. Never mention any other programme or course.
- If the student asks about ANY other student, reply exactly: Sorry, I can't provide another student's private information. Do not name them, do not offer tickets or other ways to get it.
- Do not use HTML tags. For several items inside a table cell, separate them with semicolons.
- Be concise and friendly. Address the student by their remembered preferred name if one is given."""

LLM_RETRIES = 1
RETRY_DELAY_S = 1.0
RETRYABLE = {"rate_limit", "connection", "server", "bad_request"}

FRIENDLY_LLM_ERRORS = {
    "rate_limit": "The AI service is rate-limited right now. Please wait a minute and try again.",
    "connection": "I could not reach the AI service. Please check your connection and try again.",
    "auth": "The AI service rejected the API key. Please check GROQ_API_KEY in your .env file.",
    "server": "The AI service is having problems. Please try again shortly.",
}


class AgentResult:
    def __init__(self, answer: str, trace: Trace):
        self.answer, self.trace = answer, trace



# ---------------------------------------------------------------- strict student-data access control
REFUSAL = "Sorry, I can't provide another student's private information."
_PRONOUN = re.compile(r"\b(he|him|his|she|her|hers|they|them|their|that student|same (?:student|person))\b", re.I)


def references_other_student(text: str, own_sid: str) -> bool:
    """Deterministic backend check (no LLM): does the message point at a student other than the signed-in one?
    Catches IDs ("S1001", "s 1001", "student 1001"), other students' emails, and names (incl. small typos)."""
    students = tools._load_students()
    own = students.get(own_sid, {})
    t = (text or "").lower()
    for rx in (r"\bs\s*-?\s*(\d{4})\b", r"\b(?:student|id|roll)\s*(?:id|number|no\.?)?\s*#?\s*:?\s*(\d{4})\b"):
        if any(f"S{n}" != own_sid for n in re.findall(rx, t)):
            return True
    others = {k: v for k, v in students.items() if k != own_sid}
    if any(e in {v["email"].lower() for v in others.values()} for e in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", t)):
        return True
    own_tokens = set(re.findall(r"[a-z]+", (own.get("name", "") + " " + own.get("email", "").split("@")[0]).lower()))
    names = {tok for v in others.values() for tok in re.findall(r"[a-z]+", (v["name"] + " " + v["email"].split("@")[0]).lower()) if len(tok) >= 3} - own_tokens
    long_names = [n for n in names if len(n) >= 5]
    for w in re.findall(r"[a-z]+", t):
        if w in names or (len(w) >= 5 and difflib.get_close_matches(w, long_names, n=1, cutoff=0.85)):
            return True
    return False


# --------------------------------------------------------------------- context selection
def build_messages(question: str, history: list[dict], relevant_memory: list[dict], student_id: str | None = None) -> list[dict]:
    """Build the model input from ONLY what this question needs:
       system prompt + relevant memory (<= MEMORY_TOP_K) + last HISTORY_TURNS exchanges + question.
       The handbook is NOT included here - the model pulls relevant sections via search_policy.
       Tool results from earlier turns are not replayed; only the final user/assistant text is."""
    system = SYSTEM_PROMPT
    if student_id:
        system += (f"\n\nThe signed-in student is {student_id}. Use this ID for student tools. "
                   "Refuse requests about any other student; the system also blocks them.")
    if relevant_memory:
        facts = "\n".join(f"- {m['key']}: {m['value']}" for m in relevant_memory)
        system += f"\n\nRemembered facts from earlier sessions (relevant to this question):\n{facts}"
    recent = [m for m in history if m["role"] in ("user", "assistant")]
    recent = recent[-2 * config.HISTORY_TURNS:]
    return ([{"role": "system", "content": system}]
            + [{"role": m["role"], "content": m["content"]} for m in recent]
            + [{"role": "user", "content": question}])


# --------------------------------------------------------------------- the loop
def _call_llm(llm, messages, trace: Trace, turn: int):
    """One model call with a single retry for transient errors; records the call in the trace."""
    for attempt in range(LLM_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            resp = llm.chat(messages, tools.TOOL_SCHEMAS)
        except LLMError as e:
            trace.add_error(f"LLM error ({e.kind}) on turn {turn}, attempt {attempt + 1}: {str(e)[:200]}")
            if e.kind in RETRYABLE and attempt < LLM_RETRIES:
                time.sleep(RETRY_DELAY_S)
                continue
            raise
        trace.add_model_call(turn, (time.perf_counter() - t0) * 1000, resp.usage,
                             [t.name for t in resp.tool_requests])
        return resp


def run_agent(question: str, history: list[dict] | None = None, user_id: str = "demo_student",
              llm=None, save_trace: bool = True, student_id: str | None = None,
              conversation_id: str | None = None) -> AgentResult:
    history = history or []
    ctx = tools.ToolContext(user_id=user_id, student_id=student_id, conversation_id=conversation_id)
    if student_id and (references_other_student(question, student_id) or
                       (history and history[-1]["role"] == "assistant" and history[-1]["content"].strip() == REFUSAL and _PRONOUN.search(question))):
        trace = Trace("[blocked: request about another student]", getattr(llm, "model", config.MODEL), user_id)
        try:
            database.audit(student_id, "student", "access_other_student", "-", "blocked", "refused before any model or data access")
        except Exception:
            pass
        trace.finish(REFUSAL, "access_denied")
        if save_trace:
            trace.save()
        return AgentResult(REFUSAL, trace)
    if llm is None:
        from llm import GroqLLM
        llm = GroqLLM()
    trace = Trace(question, getattr(llm, "model", config.MODEL), user_id)

    relevant = memory.retrieve_relevant(user_id, question)       # context selection (memory)
    trace.memory_used = [{"id": m["id"], "key": m["key"]} for m in relevant]
    messages = build_messages(question, history, relevant, student_id)
    trace.history_messages = len([m for m in messages[1:-1]])

    answer, reason, denied = "", "max_turns", False
    try:
        for turn in range(1, config.MAX_TURNS + 1):
            resp = _call_llm(llm, messages, trace, turn)           # REASON / DECIDE
            messages.append(resp.as_message())
            if not resp.tool_requests:                            # no tool => final answer
                answer = resp.content.strip() or "Sorry, I could not produce an answer. Please rephrase."
                reason = "final_answer"
                break
            for req in resp.tool_requests:                        # ACT + OBSERVE
                try:
                    args = json.loads(req.arguments or "{}")
                except json.JSONDecodeError:
                    args = None
                t0 = time.perf_counter()
                result = (tools.execute_tool(req.name, args, ctx) if args is not None else
                          {"ok": False, "error_type": "invalid_input",
                           "error": "Tool arguments were not valid JSON."})
                logged_args = args if args is not None else {}
                if result.get("error_type") == "forbidden":      # another student's data was requested
                    denied = True                                # never log the other student's ID or details
                    result = {"ok": False, "error_type": "forbidden", "error": "Access denied."}
                    logged_args = {"student_id": "[redacted]"}
                trace.add_tool_call(turn, req.name, logged_args, result, (time.perf_counter() - t0) * 1000)
                messages.append({"role": "tool", "tool_call_id": req.id, "name": req.name,
                                 "content": json.dumps(result)})
            if denied:                                            # fixed neutral refusal, whatever the model would say
                answer, reason = REFUSAL, "access_denied"
                break
        else:
            answer = ("I could not finish this request within the allowed number of steps. "
                      "Please try rephrasing, or ask me to create a support request.")
    except LLMError as e:
        answer = FRIENDLY_LLM_ERRORS.get(e.kind, "Something went wrong while contacting the AI service. "
                                                 "Please try again.")
        reason = f"llm_error:{e.kind}"

    trace.finish(answer, reason)
    if save_trace:
        trace.save()
    return AgentResult(answer, trace)
