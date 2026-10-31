import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import config  # noqa: E402
from llm import LLMError, LLMResponse, ToolRequest  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own SQLite DB and trace file; no real data is touched."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "TRACE_FILE", tmp_path / "traces.jsonl")
    monkeypatch.setattr(agent, "RETRY_DELAY_S", 0)


class ScriptedLLM:
    """Fake model: returns a pre-written list of responses (or raises LLMError) and records inputs."""
    model = "scripted-fake"

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, messages, tools):
        self.calls.append([dict(m) if isinstance(m, dict) else m for m in messages])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_resp(name, args, usage=None, id="c1"):
    import json
    return LLMResponse("", [ToolRequest(id, name, json.dumps(args))], usage=usage, model="scripted-fake")


def text_resp(text, usage=None):
    return LLMResponse(text, [], usage=usage, model="scripted-fake")


@pytest.fixture
def scripted():
    return ScriptedLLM
