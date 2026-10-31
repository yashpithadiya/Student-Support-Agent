"""A deterministic RULE-BASED stand-in for the language model.

*** It is NOT a real model and its results say NOTHING about real model quality. ***
Purpose: let the evaluation harness, graders, tracing and CSV/summary generation be run and
verified offline (no API key). The mock's own 'decisions' are keyword rules, so a high mock
score only shows that the harness and the agent plumbing (tools, memory, retrieval, tracing)
work end to end. It reports usage=None, so token counts show as unavailable."""
import json
import re

from llm import LLMResponse, ToolRequest

SID = re.compile(r"\bS\d{4}\b", re.I)
POLICY_WORDS = ["appeal", "portal", "elective", "course change", "curfew", "hostel", "visa",
                "placement", "attendance", "re-evaluation", "late fee", "escalat", "renewal",
                "instalment", "next working day", "deadline"]


class MockLLM:
    model = "mock-rule-based (NOT a real model)"

    def __init__(self):
        self._n = 0

    def _req(self, name, args):
        self._n += 1
        return ToolRequest(f"mock_{self._n}", name, json.dumps(args))

    def chat(self, messages, tools):
        if messages[-1]["role"] == "tool":
            return LLMResponse(self._compose(messages), model=self.model)
        q = messages[-1]["content"]
        ql = q.lower()
        system = messages[0]["content"]

        if re.search(r"\b(otp|password|passcode)\b", ql):
            return LLMResponse("Please don't share OTPs or passwords in chat. I will not store them "
                               "and you should keep them private.", model=self.model)
        sid = self._find_sid(messages, system)
        changes = []                                   # several asks in ONE message -> one tool call for each
        if m := re.search(r"(?:change|correct|update)\s+my\s+name\s+to\s+([A-Za-z.' -]{2,60}?)(?:\s+and\b|[.!?,]|$)", q, re.I):
            changes.append(("name", m.group(1).strip()))
        if m := re.search(r"(?:change|switch)\s+my\s+(?:course|programme)\s+to\s+(.+?)(?:[.!?]|$)", q, re.I):
            changes.append(("course", m.group(1).strip()))
        if changes and sid:
            return LLMResponse("", [self._req("request_record_change", {"student_id": sid, "field": f, "new_value": v}) for f, v in changes], model=self.model)
        reqs = []
        if m := re.search(r"call me (\w+)", ql):
            reqs.append(self._req("save_memory", {"key": "preferred_name", "value": m.group(1).capitalize()}))
        if re.search(r"\bPGDM\b", q):
            reqs.append(self._req("save_memory", {"key": "programme", "value": "PGDM"}))
        if m := re.search(r"\b(AI & DS|BDA)\b", q):
            reqs.append(self._req("save_memory", {"key": "course", "value": m.group(1)}))
        if reqs:
            return LLMResponse("", reqs, model=self.model)

        sid = self._find_sid(messages, system)
        prev = [x for x in messages[:-1] if x["role"] == "assistant"]
        if sid and prev and "What issue are you facing" in prev[-1]["content"] and not prev[-1].get("tool_calls"):
            cat = "fee" if re.search(r"fee|payment", ql) else "other"      # the student has now described the problem
            return LLMResponse("", [self._req("prepare_support_ticket", {"student_id": sid, "category": cat, "description": q[:200]})], model=self.model)
        need_id = LLMResponse("I need your student ID (for example S1001) to look that up. "
                              "Could you share it?", model=self.model)

        if re.search(r"support (request|ticket)|raise a ticket", ql):
            if not sid:
                return need_id
            detail = re.sub(r"\b(please|raise|create|make|a|an|the|support|ticket|request|for|me|i|want|to|need|s\d{4}|submit)\b", " ", ql)
            if len(re.findall(r"[a-z]{3,}", detail)) < 3:           # no problem described yet -> ask first, no tool
                return LLMResponse("Sure! I can help you raise a support ticket. What issue are you facing?", model=self.model)
            cat = "fee" if re.search(r"fee|payment", ql) else "other"
            return LLMResponse("", [self._req("prepare_support_ticket",
                               {"student_id": sid, "category": cat, "description": q[:200]})], model=self.model)
        if re.search(r"which programme is student|enrolled in", ql) and sid:
            return LLMResponse("", [self._req("get_student_record", {"student_id": sid})], model=self.model)
        if any(w in ql for w in POLICY_WORDS):
            return LLMResponse("", [self._req("search_policy", {"query": q})], model=self.model)
        if "scholarship" in ql:
            return LLMResponse("", [self._req("check_scholarship_status", {"student_id": sid})],
                               model=self.model) if sid else need_id
        if re.search(r"fee|owe|balance|dues", ql):
            return LLMResponse("", [self._req("check_fee_status", {"student_id": sid})],
                               model=self.model) if sid else need_id
        if re.search(r"which (programme|course) did i", ql):
            m = re.search(r"(?:programme|course): (.+)", system)
            return LLMResponse(f"You told me you are in {m.group(1)}." if m else
                               "I don't have that on record.", model=self.model)
        if re.search(r"hello|what can you help", ql):
            return LLMResponse("Hello! I can help with fees, scholarships, course changes and support requests.",
                               model=self.model)
        return LLMResponse("Sorry, I can only help with student-support questions such as fees, "
                           "scholarships, course changes and support requests.", model=self.model)

    @staticmethod
    def _find_sid(messages, system):
        for m in reversed([m for m in messages if m["role"] == "user"]):
            if f := SID.search(m["content"]):
                return f.group(0).upper()
        if f := re.search(r"signed-in student is (S\d{4})|student_id: (S\d{4})", system):
            return f.group(1) or f.group(2)
        return None

    @staticmethod
    def _compose(messages):
        i = max(k for k, m in enumerate(messages) if m["role"] == "user")
        results = [json.loads(m["content"]) for m in messages[i:] if m["role"] == "tool"]
        out = []
        for r in results:
            if not r.get("ok"):
                if r.get("refused"):
                    out.append("I can't store that because it looks sensitive.")
                elif r.get("error_type") == "not_found":
                    out.append(f"{r['error']} Please check the ID.")
                else:
                    out.append(f"Sorry, that did not work: {r['error']}")
            elif "fee_status" in r:
                out.append(f"{r['name']}, your fee status is {r['fee_status']}: INR {r['fee_paid_inr']} paid of "
                           f"{r['fee_total_inr']}, balance {r['balance_inr']}, next due {r['next_due_date']}.")
            elif "scholarship" in r:
                out.append(f"Your {r['scholarship']} application is: {r['status']}. {r['note']}")
            elif "programme" in r:
                out.append(f"{r['name']} is enrolled in {r['programme']}, course {r['course']} (semester {r['semester']}).")
            elif "draft_id" in r:
                out.append(f"I have prepared a support ticket draft (category: {r['category']}): {r['description']} "
                           "Would you like me to submit this support request? Please Confirm or Cancel.")
            elif "request_id" in r:
                out.append(f"Your request {r['request_id']} ({r['request_type']}) is {r['status']} and needs administrator approval.")
            elif "ticket_id" in r:
                out.append(f"I created support request {r['ticket_id']} (priority {r['priority']}).")
            elif "chunks" in r:
                c = r["chunks"]
                out.append(f"According to the handbook ({c[0]['title']}): {c[0]['text']}" if c else
                           "The handbook does not contain information on that, so I can't answer it. "
                           "I can create a support request if you like.")
            elif "saved" in r:
                out.append("Thanks, I'll remember that.")
        return " ".join(out) or "Done."
