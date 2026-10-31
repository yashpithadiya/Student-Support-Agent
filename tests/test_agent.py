import json

import pytest

import agent
import config
import memory
from conftest import ScriptedLLM, text_resp, tool_resp
from llm import LLMError


def test_direct_answer_uses_no_tool():
    llm = ScriptedLLM([text_resp("Hello! I can help with fees.")])
    r = agent.run_agent("Hello", llm=llm, save_trace=False)
    assert r.answer.startswith("Hello") and r.trace.tools_called == []
    assert r.trace.stopping_reason == "final_answer"


def test_tool_result_is_fed_back_before_the_final_answer():
    llm = ScriptedLLM([tool_resp("check_fee_status", {"student_id": "S1002"}),
                       text_resp("Your balance is 225000.")])
    r = agent.run_agent("fee status of S1002?", llm=llm, save_trace=False)
    second_call = llm.calls[1]
    tool_msg = [m for m in second_call if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"])["balance_inr"] == 225000      # real tool output reached the model
    assert r.trace.tools_called == ["check_fee_status"] and r.trace.tool_calls[0]["status"] == "success"


def test_multi_step_loop_chains_tools():
    llm = ScriptedLLM([tool_resp("get_student_record", {"student_id": "S1001"}, id="a"),
                       tool_resp("check_fee_status", {"student_id": "S1001"}, id="b"),
                       text_resp("done")])
    r = agent.run_agent("details and fees for S1001", llm=llm, save_trace=False)
    assert r.trace.tools_called == ["get_student_record", "check_fee_status"] and len(llm.calls) == 3


def test_tool_error_reaches_model_and_is_traced():
    llm = ScriptedLLM([tool_resp("check_fee_status", {"student_id": "S9999"}), text_resp("No such student.")])
    r = agent.run_agent("fees for S9999", llm=llm, save_trace=False)
    ev = r.trace.tool_calls[0]
    assert ev["status"] == "error" and ev["error"].startswith("No student record")
    assert "not_found" in [m for m in llm.calls[1] if m["role"] == "tool"][0]["content"]


def test_invalid_tool_arguments_json_is_handled():
    bad = tool_resp("check_fee_status", {})
    bad.tool_requests[0].arguments = "{not json"
    llm = ScriptedLLM([bad, text_resp("Sorry, could not do that.")])
    r = agent.run_agent("fees", llm=llm, save_trace=False)
    assert r.trace.tool_calls[0]["status"] == "error" and r.trace.stopping_reason == "final_answer"


def test_unknown_tool_requested_by_model_is_handled():
    llm = ScriptedLLM([tool_resp("drop_database", {}), text_resp("I can't do that.")])
    r = agent.run_agent("do it", llm=llm, save_trace=False)
    assert r.trace.tool_calls[0]["result"]["error_type"] == "unknown_tool"


def test_max_turns_stops_runaway_loop(monkeypatch):
    monkeypatch.setattr(config, "MAX_TURNS", 3)
    llm = ScriptedLLM([tool_resp("search_policy", {"query": "late fee"}, id=str(i)) for i in range(10)])
    r = agent.run_agent("loop forever", llm=llm, save_trace=False)
    assert r.trace.stopping_reason == "max_turns" and len(llm.calls) == 3 and "steps" in r.answer


# ---------------- API errors ----------------
def test_rate_limit_is_retried_once_then_reported_gracefully():
    llm = ScriptedLLM([LLMError("rate_limit", "429"), LLMError("rate_limit", "429")])
    r = agent.run_agent("hi", llm=llm, save_trace=False)
    assert "rate-limited" in r.answer and r.trace.stopping_reason == "llm_error:rate_limit"
    assert len(r.trace.errors) == 2 and len(llm.calls) == 2


def test_transient_error_recovers_on_retry():
    llm = ScriptedLLM([LLMError("connection", "timeout"), text_resp("Hello")])
    r = agent.run_agent("hi", llm=llm, save_trace=False)
    assert r.answer == "Hello" and len(r.trace.errors) == 1


def test_auth_error_not_retried_and_message_mentions_key():
    llm = ScriptedLLM([LLMError("auth", "bad key")])
    r = agent.run_agent("hi", llm=llm, save_trace=False)
    assert "GROQ_API_KEY" in r.answer and len(llm.calls) == 1


# ---------------- context selection ----------------
def test_context_contains_only_needed_information():
    memory.save_memory("u1", "preferred_name", "Ria")
    memory.save_memory("u1", "programme", "PGDM")
    history = []
    for i in range(6):
        history += [{"role": "user", "content": f"old question {i}"},
                    {"role": "assistant", "content": f"old answer {i}"}]
    llm = ScriptedLLM([text_resp("ok")])
    agent.run_agent("Which programme am I in?", history, "u1", llm, save_trace=False)
    msgs = llm.calls[0]
    system = msgs[0]["content"]
    assert "programme: PGDM" in system         # relevant memory included
    assert "preferred_name" not in system                         # irrelevant memory excluded
    assert "DEMONSTRATION DATA" not in system and "Late Fee Rules" not in system   # no handbook dump
    contents = [m["content"] for m in msgs[1:]]
    assert len(contents) == 2 * config.HISTORY_TURNS + 1          # limited history + current question
    assert "old question 0" not in contents and contents[-1] == "Which programme am I in?"


def test_no_memory_block_when_nothing_relevant():
    memory.save_memory("u1", "preferred_name", "Ria")
    llm = ScriptedLLM([text_resp("ok")])
    r = agent.run_agent("What is the capital of France?", [], "u1", llm, save_trace=False)
    assert "from earlier sessions" not in llm.calls[0][0]["content"] and r.trace.memory_used == []


# ---------------- memory through the agent ----------------
def test_agent_saves_memory_and_next_session_uses_it():
    llm1 = ScriptedLLM([tool_resp("save_memory", {"key": "student_id", "value": "S1005"}), text_resp("Noted")])
    agent.run_agent("I'm S1005", [], "u1", llm1, save_trace=False)
    llm2 = ScriptedLLM([text_resp("ok")])                          # brand-new session: empty history
    r = agent.run_agent("What is my scholarship status?", [], "u1", llm2, save_trace=False)
    assert "student_id: S1005" in llm2.calls[0][0]["content"]
    assert r.trace.memory_used[0]["key"] == "student_id"


def test_sensitive_statement_is_not_stored_even_if_model_tries():
    llm = ScriptedLLM([tool_resp("save_memory", {"key": "preference", "value": "OTP is 482910"}),
                       text_resp("Please don't share OTPs.")])
    r = agent.run_agent("my OTP is 482910, remember it", [], "u1", llm, save_trace=False)
    assert memory.list_memories("u1") == []
    assert r.trace.tool_calls[0]["status"] == "error"


# ---------------- trace ----------------
def test_trace_contains_required_fields_and_is_saved(tmp_path):
    memory.save_memory("u1", "student_id", "S1002")
    llm = ScriptedLLM([tool_resp("search_policy", {"query": "portal failed pay next working day"},
                                 usage={"input_tokens": 100, "output_tokens": 10}),
                       text_resp("Report within 24 hours.", usage={"input_tokens": 300, "output_tokens": 20})])
    r = agent.run_agent("My payment portal failed, can I pay tomorrow?", [], "u1", llm)
    t = r.trace.to_dict()
    for k in ["run_id", "question", "model", "start_time", "end_time", "latency_ms", "token_usage",
              "memory_used", "retrieved_chunks", "tools_called", "events", "errors", "final_response",
              "stopping_reason", "evaluation"]:
        assert k in t
    assert t["retrieved_chunks"][0]["chunk_id"] == "payment-portal-failure"
    assert t["token_usage"] == {"available": True, "input_tokens": 400, "output_tokens": 30}
    assert t["latency_ms"] >= 0 and t["model"] == "scripted-fake"
    line = json.loads(config.TRACE_FILE.read_text().splitlines()[-1])    # persisted as JSONL
    assert line["run_id"] == t["run_id"]


def test_missing_token_usage_is_marked_unavailable_not_invented():
    r = agent.run_agent("hi", llm=ScriptedLLM([text_resp("hello")]), save_trace=False)
    assert r.trace.token_usage() == {"available": False, "input_tokens": None, "output_tokens": None}


def test_trace_does_not_log_memory_values():
    memory.save_memory("u1", "programme", "PGDM")
    r = agent.run_agent("Which programme am I in?", [], "u1", ScriptedLLM([text_resp("x")]), save_trace=False)
    assert "PGDM" not in json.dumps(r.trace.memory_used)


def test_policy_question_trace_records_no_chunks_when_unsupported():
    llm = ScriptedLLM([tool_resp("search_policy", {"query": "hostel curfew"}), text_resp("Not in handbook.")])
    r = agent.run_agent("hostel curfew?", llm=llm, save_trace=False)
    assert r.trace.retrieved_chunks == [] and r.trace.tools_called == ["search_policy"]
