import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import config
from evaluation import evaluate as ev
from llm import LLMResponse

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_COLUMNS = ["case_id", "category", "task_success", "tool_selection_correct", "tool_execution_ok",
                    "answer_correct", "retrieval_ok", "memory_ok", "tools_called", "latency_ms",
                    "input_tokens", "output_tokens", "retrieved_chunks", "memory_used", "observations"]


def test_golden_dataset_is_valid_and_covers_required_categories():
    cases = json.loads(config.GOLDEN_DATASET.read_text())["cases"]
    assert 15 <= len(cases) <= 20 and len({c["id"] for c in cases}) == len(cases)
    cats = {c["category"] for c in cases}
    assert {"direct", "fee_tool", "scholarship_tool", "support_ticket", "policy_rag", "memory_read",
            "missing_record", "unsupported_policy", "tool_failure", "sensitive"} <= cats


def test_mock_evaluation_generates_report_files(tmp_path):
    out = subprocess.run([sys.executable, "-m", "evaluation.evaluate", "--mock", "--out-dir", str(tmp_path)],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    df = pd.read_csv(tmp_path / "results_mock.csv")
    assert set(REQUIRED_COLUMNS) <= set(df.columns) and len(df) == 20
    assert (df["input_tokens"] == "unavailable").all()            # mock reports no usage -> never invented
    assert df["latency_ms"].astype(float).gt(0).all()
    summary = (tmp_path / "evaluation_summary_mock.md").read_text()
    assert "MOCK RUN" in summary and "Task success" in summary
    traces = [json.loads(l) for l in (tmp_path / "eval_traces_mock.jsonl").read_text().splitlines()]
    assert len(traces) == 20 and all(t["evaluation"] for t in traces)


def test_harness_fails_a_model_that_fabricates_and_skips_tools():
    class Liar:
        model = "liar"

        def chat(self, messages, tools):
            return LLMResponse("Your fee is fully paid. The hostel curfew is 10 pm.", model="liar")
    cases = {c["id"]: c for c in json.loads(config.GOLDEN_DATASET.read_text())["cases"]}
    for cid in ["G02_fee_partial", "G08_policy_portal_failure", "G13_unsupported_policy", "G14_tool_failure"]:
        row, _ = ev.run_case(cases[cid], "mock", Liar)
        assert row["task_success"] is False, cid


def test_graders_unit():
    assert ev.grade_numbers([450000], "Total INR 4,50,000") is True
    assert ev.grade_numbers([10], "within 10 working days") is True
    assert ev.grade_numbers([225000], "nothing here") is False
    assert ev.grade_forbidden([r"\b10\s?pm\b"], "curfew is 10 pm") is False
    assert ev.grade_no_tool(True, ["search_policy"]) is False
    assert ev.grade_required_tools([], ["x"]) is None
