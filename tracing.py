"""Manual tracing, in the style of the course notebook `Unit_II_Manual_Tracing_and_Langfuse`.

One Trace per agent run. It records every model call, tool call, retrieval and error,
then is appended as one JSON line to TRACE_FILE. Token counts are NEVER invented:
if the provider reports no usage, the value stays None ("unavailable")."""
import json
import time
import uuid
from datetime import datetime, timezone

import config


class Trace:
    def __init__(self, question: str, model: str, user_id: str = ""):
        self.run_id = uuid.uuid4().hex[:12]
        self.question = question
        self.model = model
        self.user_id = user_id
        self.start_time = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()
        self.events: list[dict] = []          # ordered: model_call / tool_call / error
        self.memory_used: list[dict] = []     # [{"id", "key"}]  (no values logged)
        self.retrieved_chunks: list[dict] = []  # [{"chunk_id", "title", "score"}]
        self.errors: list[str] = []
        self.history_messages = 0
        self.final_response = ""
        self.stopping_reason = ""
        self.end_time = ""
        self.latency_ms = 0.0
        self.evaluation: dict | None = None   # filled by evaluation/evaluate.py

    # ---- recording helpers ----
    def add_model_call(self, turn: int, latency_ms: float, usage: dict | None,
                       requested_tools: list[str]):
        self.events.append({"type": "model_call", "turn": turn,
                            "latency_ms": round(latency_ms, 1),
                            "usage": usage,  # None = unavailable
                            "requested_tools": requested_tools})

    def add_tool_call(self, turn: int, name: str, arguments: dict, result: dict,
                      latency_ms: float):
        status = "success" if result.get("ok") else "error"
        self.events.append({"type": "tool_call", "turn": turn, "tool": name,
                            "arguments": arguments, "status": status,
                            "error": result.get("error"), "result": result,
                            "latency_ms": round(latency_ms, 1)})
        if name == "search_policy":
            for c in result.get("chunks", []):
                self.retrieved_chunks.append({"chunk_id": c["chunk_id"],
                                              "title": c["title"], "score": c["score"]})

    def add_error(self, message: str):
        self.errors.append(message)
        self.events.append({"type": "error", "message": message})

    # ---- derived views ----
    @property
    def tool_calls(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "tool_call"]

    @property
    def tools_called(self) -> list[str]:
        return [e["tool"] for e in self.tool_calls]

    def token_usage(self) -> dict:
        """Sum usage across model calls. 'available' is False if any call lacked usage."""
        calls = [e for e in self.events if e["type"] == "model_call"]
        if not calls or any(c["usage"] is None for c in calls):
            return {"available": False, "input_tokens": None, "output_tokens": None}
        return {"available": True,
                "input_tokens": sum(c["usage"]["input_tokens"] for c in calls),
                "output_tokens": sum(c["usage"]["output_tokens"] for c in calls)}

    def finish(self, final_response: str, stopping_reason: str):
        self.final_response = final_response
        self.stopping_reason = stopping_reason
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.latency_ms = round((time.perf_counter() - self._t0) * 1000, 1)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "user_id": self.user_id, "question": self.question,
                "model": self.model, "start_time": self.start_time, "end_time": self.end_time,
                "latency_ms": self.latency_ms, "token_usage": self.token_usage(),
                "history_messages": self.history_messages, "memory_used": self.memory_used, "retrieved_chunks": self.retrieved_chunks,
                "tools_called": self.tools_called, "events": self.events,
                "errors": self.errors, "final_response": self.final_response,
                "stopping_reason": self.stopping_reason, "evaluation": self.evaluation}

    def save(self, path=None):
        path = path or config.TRACE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
