"""Tests llm.GroqLLM against the REAL Groq SDK classes, but with a fake HTTP transport (no network,
no API key). Verifies request/response parsing, usage extraction and error mapping.
This is NOT a live test: it cannot show how the real model behaves."""
import json

import groq
import httpx
import pytest

from llm import GroqLLM, LLMError


def make_llm(handler):
    client = groq.Groq(api_key="test-key", base_url="http://fake.invalid", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return GroqLLM(client=client, model="openai/gpt-oss-20b")


def completion(message, usage=True):
    body = {"id": "x", "object": "chat.completion", "created": 1, "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}]}
    if usage:
        body["usage"] = {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
    return httpx.Response(200, json=body)


def test_text_response_and_usage_parsed():
    llm = make_llm(lambda req: completion({"role": "assistant", "content": "Hi there"}))
    r = llm.chat([{"role": "user", "content": "hi"}], [])
    assert r.content == "Hi there" and r.tool_requests == []
    assert r.usage == {"input_tokens": 123, "output_tokens": 45}


def test_tool_call_parsed_and_request_shape():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return completion({"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "check_fee_status", "arguments": "{\"student_id\": \"S1002\"}"}}]})
    r = make_llm(handler).chat([{"role": "user", "content": "fees"}], [{"type": "function", "function": {
        "name": "check_fee_status", "parameters": {"type": "object", "properties": {}}}}])
    assert r.tool_requests[0].name == "check_fee_status"
    assert json.loads(r.tool_requests[0].arguments) == {"student_id": "S1002"}
    assert r.as_message()["tool_calls"][0]["function"]["name"] == "check_fee_status"
    assert seen["model"] == "openai/gpt-oss-20b" and seen["temperature"] == 0 and seen["tool_choice"] == "auto"


def test_missing_usage_stays_none_not_invented():
    llm = make_llm(lambda req: completion({"role": "assistant", "content": "ok"}, usage=False))
    assert llm.chat([{"role": "user", "content": "hi"}], []).usage is None


@pytest.mark.parametrize("status,kind", [(429, "rate_limit"), (401, "auth"), (400, "bad_request"),
                                         (500, "server")])
def test_http_errors_are_mapped(status, kind):
    llm = make_llm(lambda req: httpx.Response(status, json={"error": {"message": "boom"}}))
    with pytest.raises(LLMError) as e:
        llm.chat([{"role": "user", "content": "hi"}], [])
    assert e.value.kind == kind


def test_connection_error_is_mapped():
    def handler(req):
        raise httpx.ConnectError("no route")
    with pytest.raises(LLMError) as e:
        make_llm(handler).chat([{"role": "user", "content": "hi"}], [])
    assert e.value.kind == "connection"


def test_agent_end_to_end_over_fake_http_with_real_sdk(monkeypatch):
    """Whole stack: agent -> GroqLLM -> real SDK -> fake HTTP. Model asks for a tool, then answers."""
    import agent
    replies = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "check_fee_status", "arguments": "{\"student_id\": \"S1002\"}"}}]},
        {"role": "assistant", "content": "Your balance is INR 225000."}]
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return completion(replies.pop(0))
    r = agent.run_agent("fee status for S1002", [], "u1", make_llm(handler), save_trace=False)
    assert r.answer == "Your balance is INR 225000."
    assert r.trace.token_usage() == {"available": True, "input_tokens": 246, "output_tokens": 90}
    tool_msg = [m for m in bodies[1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"])["balance_inr"] == 225000
