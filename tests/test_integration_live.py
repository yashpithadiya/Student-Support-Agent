"""LIVE integration tests - call the real Groq API. Skipped automatically without GROQ_API_KEY.
Run with:  GROQ_API_KEY=... pytest -m live -v"""
import os

import pytest

import agent

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")]


def test_live_fee_question_calls_tool():
    r = agent.run_agent("What is my fee status? My student ID is S1002.", [], "live_test", save_trace=False)
    assert "check_fee_status" in r.trace.tools_called
    assert r.trace.token_usage()["available"] in (True, False)       # reported honestly either way


def test_live_policy_question_retrieves_chunk():
    r = agent.run_agent("The payment portal failed. Can I pay on the next working day?", [], "live_test",
                        save_trace=False)
    assert "payment-portal-failure" in [c["chunk_id"] for c in r.trace.retrieved_chunks]
