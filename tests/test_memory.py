import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import config
import memory
import tools

ROOT = Path(__file__).resolve().parent.parent


def test_save_and_list():
    assert memory.save_memory("u1", "preferred_name", "Ria")["ok"]
    assert memory.list_memories("u1")[0]["value"] == "Ria"


def test_memory_persists_across_sessions_in_a_new_process():
    """A brand-new Python process (= a new app session) can read what an earlier one saved."""
    memory.save_memory("u1", "programme", "PGDM")
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(ROOT)!r})
        import config, pathlib; config.DB_PATH = pathlib.Path({str(config.DB_PATH)!r})
        import memory
        print(memory.retrieve_relevant("u1", "Which programme am I in?")[0]["value"])
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "PGDM"


@pytest.mark.parametrize("secret", [
    "my OTP is 482910", "password Winter@2026", "482910", "token abc", "api key sk-123",
    "my card number is 4111111111111111", "PIN 1234", "aadhaar 123456789012"])
def test_sensitive_values_are_refused(secret):
    r = memory.save_memory("u1", "preference", secret)
    assert not r["ok"] and r.get("refused")
    assert memory.list_memories("u1") == []


def test_sensitive_refused_through_the_tool_layer_too():
    r = tools.execute_tool("save_memory", {"key": "preference", "value": "OTP 998877"},
                           tools.ToolContext("u1"))
    assert not r["ok"] and memory.list_memories("u1") == []


def test_only_whitelisted_keys():
    r = memory.save_memory("u1", "bank_balance", "lots")
    assert not r["ok"] and memory.list_memories("u1") == []


def test_single_valued_key_is_replaced_and_multi_valued_deduplicated():
    memory.save_memory("u1", "preferred_name", "Ria")
    memory.save_memory("u1", "preferred_name", "Riya")
    assert [m["value"] for m in memory.list_memories("u1")] == ["Riya"]
    memory.save_memory("u1", "preference", "prefers short answers")
    memory.save_memory("u1", "preference", "prefers short answers")
    assert len([m for m in memory.list_memories("u1") if m["key"] == "preference"]) == 1


def test_student_id_format_validated():
    assert not memory.save_memory("u1", "student_id", "12345678")["ok"]
    assert memory.save_memory("u1", "student_id", "s1002")["saved"]["value"] == "S1002"


def test_only_relevant_memory_is_retrieved():
    memory.save_memory("u1", "preferred_name", "Ria")
    memory.save_memory("u1", "programme", "PGDM")
    memory.save_memory("u1", "student_id", "S1002")
    keys = lambda q: {m["key"] for m in memory.retrieve_relevant("u1", q)}   # noqa: E731
    assert keys("Which programme am I in?") == {"programme"}
    assert keys("What should you call me?") == {"preferred_name"}
    assert keys("What is my fee status?") == {"student_id"}
    assert keys("How do I change my elective course?") == {"programme"}   # elective -> programme context
    assert keys("What is the capital of France?") == set()
    assert "student_id" not in keys("What is the late fee rule?")      # not personal -> not needed


def test_memory_is_per_user():
    memory.save_memory("alice", "preferred_name", "Alice")
    assert memory.list_memories("bob") == []
