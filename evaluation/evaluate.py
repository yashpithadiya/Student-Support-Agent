"""Repeatable evaluation harness (pattern from Unit_II_Agent_Evaluation_Fundamentals_Golden_Dataset).

EVALUATION CRITERIA (defined before any run; separate dimensions, as in the course notes):
  1. tool_selection_correct : required tools were called, no prohibited tool, no tool when
                              expect_no_tool, and key arguments match (e.g. student_id).   [process]
  2. tool_execution_ok      : the tools that ran returned ok=True - OR returned exactly the
                              error_type the case expects (not_found / tool_failure).      [process]
  3. answer_correct         : final answer contains an expected keyword/number (any-of / all-of)
                              and none of the forbidden patterns (fabricated values).      [outcome]
  4. retrieval_ok           : expected policy chunk ids were retrieved; or NO chunk when the
                              handbook has no answer.                                      [evidence]
  5. memory_ok              : memory written / not written / used as expected.               [state]
  6. task_success           : ALL applicable checks above pass (n/a checks are ignored).
Latency, tokens, retrieved chunks and memory used come from the trace of the graded (last) turn.
Tokens are reported ONLY if the provider returned them; otherwise 'unavailable'.

Usage:  python -m evaluation.evaluate --live        (real Groq model; needs GROQ_API_KEY)
        python -m evaluation.evaluate --mock        (offline rule-based mock: verifies the harness ONLY)
        add --repeats 3 to repeat the suite and observe non-determinism."""
import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
import config  # noqa: E402
import memory  # noqa: E402
from llm import LLMError  # noqa: E402

EVAL_USER = "eval_user"


class FaultyLLM:
    """Raises a provider error on every call (used for the rate-limit test case)."""
    model = "faulty-llm (injected error)"

    def __init__(self, kind):
        self.kind = kind

    def chat(self, messages, tools):
        raise LLMError(self.kind, "injected error for evaluation")


# ------------------------------------------------------------------ graders (deterministic)
def grade_required_tools(required, called):
    return set(required).issubset(called) if required else None


def grade_prohibited_tools(prohibited, called):
    return not (set(prohibited) & set(called)) if prohibited else None


def grade_no_tool(expect_no_tool, called):
    return (len(called) == 0) if expect_no_tool else None


def grade_tool_arguments(expected, tool_calls):
    if not expected:
        return None
    for exp in expected:
        ok = any(c["tool"] == exp["tool"] and all(
            str(c["arguments"].get(k, "")).lower() == str(v).lower() for k, v in exp["arguments"].items())
            for c in tool_calls)
        if not ok:
            return False
    return True


def grade_numbers(expected, text):
    if not expected:
        return None
    clean = re.sub(r"(?<=\d),(?=\d)", "", text or "")
    found = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", clean)]
    return all(any(math.isclose(float(e), f, abs_tol=1e-6) for f in found) for e in expected)


def grade_keywords_any(keywords, text):
    return any(k.lower() in (text or "").lower() for k in keywords) if keywords else None


def grade_forbidden(patterns, text):
    if not patterns:
        return None
    return not any(re.search(p, text or "", re.I) for p in patterns)


def grade_tool_execution(case_expect, tool_calls):
    """Separate 'did the tool run correctly' from 'is the answer correct'."""
    relevant = [c for c in tool_calls if c["tool"] in (case_expect.get("required_tools") or [])]
    if not relevant:
        return None
    want_err = case_expect.get("tool_error_type")
    for c in relevant:
        err_type = (c["result"] or {}).get("error_type")
        if want_err:
            if err_type != want_err:
                return False
        elif c["status"] != "success":
            return False
    return True


# ------------------------------------------------------------------ one case
def run_case(case, mode, llm_factory):
    exp = case["expect"]
    tmp = Path(tempfile.mkdtemp()) / "eval.db"
    config.DB_PATH = tmp                                    # isolated DB per case
    for m in case.get("setup_memory", []):                  # simulates an EARLIER session
        memory.save_memory(EVAL_USER, m["key"], m["value"])

    history, result = [], None
    for i, turn in enumerate(case["turns"]):
        if case.get("fresh_session_between_turns") and i > 0:
            history = []
        if case.get("inject_llm_error"):
            llm, agent.RETRY_DELAY_S = FaultyLLM(case["inject_llm_error"]), 0
        else:
            llm = llm_factory()
        result = agent.run_agent(turn, history, EVAL_USER, llm, save_trace=False)
        history += [{"role": "user", "content": turn}, {"role": "assistant", "content": result.answer}]
    agent.RETRY_DELAY_S = 1.0

    tr = result.trace
    called, calls, answer = tr.tools_called, tr.tool_calls, result.answer
    retrieved_ids = [c["chunk_id"] for c in tr.retrieved_chunks]
    mems = memory.list_memories(EVAL_USER)
    obs = []

    # --- tool selection (process) ---
    parts = [grade_required_tools(exp.get("required_tools"), called),
             grade_prohibited_tools(exp.get("prohibited_tools"), called),
             grade_no_tool(exp.get("expect_no_tool"), called),
             grade_tool_arguments(exp.get("tool_arguments"), calls)]
    applicable = [p for p in parts if p is not None]
    tool_sel = all(applicable) if applicable else None
    for name, p in zip(["required_tools", "prohibited_tools", "no_tool", "tool_arguments"], parts):
        if p is False:
            obs.append(f"tool check failed: {name}; called={called}")

    # --- tool execution (process) ---
    tool_exec = grade_tool_execution(exp, calls)
    if tool_exec is False:
        obs.append("tool execution status not as expected: " +
                   str([(c['tool'], c['status'], (c['result'] or {}).get('error_type')) for c in calls]))

    # --- answer correctness (outcome) ---
    ans_parts = [grade_keywords_any(exp.get("keywords_any"), answer),
                 grade_numbers(exp.get("expected_numbers"), answer),
                 grade_forbidden(exp.get("forbidden_patterns"), answer)]
    if exp.get("stopping_reason"):
        ans_parts.append(tr.stopping_reason == exp["stopping_reason"])
    ans_app = [p for p in ans_parts if p is not None]
    answer_ok = all(ans_app) if ans_app else None
    if answer_ok is False:
        labels = ["keywords_any", "expected_numbers", "forbidden_patterns"] + \
                 (["stopping_reason"] if exp.get("stopping_reason") else [])
        failed = [l for l, p in zip(labels, ans_parts) if p is False]
        obs.append(f"answer check failed: {failed}; answer={answer[:120]!r}")

    # --- retrieval (evidence) ---
    ret = []
    if exp.get("expected_chunks"):
        ret.append(set(exp["expected_chunks"]).issubset(retrieved_ids))
    if exp.get("expect_no_chunks"):
        ret.append(len(retrieved_ids) == 0)
    retrieval_ok = all(ret) if ret else None
    if retrieval_ok is False:
        obs.append(f"retrieval mismatch: got {retrieved_ids}")

    # --- memory / state ---
    mem = []
    for key, want in (exp.get("memory_after") or {}).items():
        val = next((m["value"] for m in mems if m["key"] == key), None)
        mem.append(val is not None and (want[9:] in val if want.startswith("contains:") else val == want))
    if exp.get("memory_empty_after"):
        mem.append(len(mems) == 0)
    for bad in exp.get("memory_forbidden_values", []):
        mem.append(not any(bad in m["value"] for m in mems))
    for key in exp.get("expected_memory_used", []):
        mem.append(key in [m["key"] for m in tr.memory_used])
    if exp.get("ticket_created_for"):
        with sqlite3.connect(tmp) as c:
            try:
                n = c.execute("SELECT COUNT(*) FROM support_tickets WHERE student_id=?",
                              (exp["ticket_created_for"],)).fetchone()[0]
            except sqlite3.OperationalError:
                n = 0
        mem.append(n >= 1)
    for key, table, want in (("draft_created_for", "ticket_drafts", True), ("no_ticket_for", "support_tickets", False)):
        if exp.get(key):
            with sqlite3.connect(tmp) as c:
                try:
                    n = c.execute(f"SELECT COUNT(*) FROM {table} WHERE student_id=?", (exp[key],)).fetchone()[0]
                except sqlite3.OperationalError:
                    n = 0
            mem.append((n >= 1) if want else (n == 0))
    memory_ok = all(mem) if mem else None
    if memory_ok is False:
        obs.append(f"memory/state check failed; stored keys={[m['key'] for m in mems]}")

    checks = [tool_sel, tool_exec, answer_ok, retrieval_ok, memory_ok]
    success = all(c for c in checks if c is not None)
    outcome = {"task_success": success, "tool_selection_correct": tool_sel, "tool_execution_ok": tool_exec,
               "answer_correct": answer_ok, "retrieval_ok": retrieval_ok, "memory_ok": memory_ok,
               "observations": obs}
    tr.evaluation = {"case_id": case["id"], "mode": mode, **outcome}
    usage = tr.token_usage()
    return {
        "case_id": case["id"], "category": case["category"], "input": case["turns"][-1],
        "task_success": success, "tool_selection_correct": tool_sel, "tool_execution_ok": tool_exec,
        "answer_correct": answer_ok, "retrieval_ok": retrieval_ok, "memory_ok": memory_ok,
        "tools_called": ";".join(called), "latency_ms": tr.latency_ms,
        "input_tokens": usage["input_tokens"] if usage["available"] else "unavailable",
        "output_tokens": usage["output_tokens"] if usage["available"] else "unavailable",
        "retrieved_chunks": ";".join(retrieved_ids),
        "memory_used": ";".join(m["key"] for m in tr.memory_used),
        "stopping_reason": tr.stopping_reason, "model": tr.model, "mode": mode,
        "final_answer": answer.replace("\n", " ")[:300],
        "observations": " | ".join(obs),
    }, tr


# ------------------------------------------------------------------ suite + reports
def pct(series):
    s = series.dropna()
    return f"{100 * s.astype(float).mean():.1f}% ({int(s.astype(bool).sum())}/{len(s)})" if len(s) else "n/a"


def write_summary(df, path, mode):
    lat = df["latency_ms"].astype(float)
    tok_ok = df["input_tokens"].apply(lambda v: v != "unavailable").all()
    lines = []
    if mode == "mock":
        lines += ["> **MOCK RUN — NOT MODEL PERFORMANCE.** These numbers come from a deterministic rule-based "
                  "stand-in, used only to verify the evaluation harness without an API key. Run "
                  "`python -m evaluation.evaluate --live` with a real `GROQ_API_KEY` to measure the actual model.", ""]
    lines += [f"# Evaluation summary ({mode})", "",
              f"- Model: `{df['model'].iloc[0]}`  |  cases: {df['case_id'].nunique()}  |  "
              f"runs: {len(df)}", "",
              "| Metric | Result |", "|---|---|",
              f"| Task success (all applicable checks) | {pct(df['task_success'])} |",
              f"| Tool selection correct | {pct(df['tool_selection_correct'])} |",
              f"| Tool execution as expected | {pct(df['tool_execution_ok'])} |",
              f"| Answer correct | {pct(df['answer_correct'])} |",
              f"| Retrieval evidence correct | {pct(df['retrieval_ok'])} |",
              f"| Memory/state correct | {pct(df['memory_ok'])} |",
              f"| Latency mean / median / max | {lat.mean():.0f} / {lat.median():.0f} / {lat.max():.0f} ms |"]
    if tok_ok:
        i, o = df["input_tokens"].astype(float), df["output_tokens"].astype(float)
        lines.append(f"| Tokens per run (input / output), mean | {i.mean():.0f} / {o.mean():.0f} |")
    else:
        lines.append("| Token usage | unavailable (provider returned no usage for these runs) |")
    lines += ["", "## By category", "", "| Category | Success |", "|---|---|"]
    for cat, g in df.groupby("category", sort=False):
        lines.append(f"| {cat} | {pct(g['task_success'])} |")
    fails = df[~df["task_success"].astype(bool)]
    lines += ["", "## Failed cases", ""]
    lines += ["None."] if fails.empty else [f"- **{r.case_id}** — {r.observations}" for r in fails.itertuples()]
    lines += ["", "## Known limitations", "",
              "- Deterministic graders check keywords/numbers/tools; they cannot judge groundedness or tone "
              "(Unit II 'limitation of deterministic graders').",
              "- 20 cases is small; one run is not proof of reliability (use `--repeats`).",
              "- Latency includes network + model time and varies per run; it is graded-turn latency only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true")
    g.add_argument("--mock", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out-dir", default=str(ROOT / "evaluation"))
    args = ap.parse_args()
    mode = "live" if args.live else "mock"

    if args.live:
        problems = config.validate_config()
        if problems:
            sys.exit("Cannot run live evaluation:\n- " + "\n- ".join(problems))
        from llm import GroqLLM
        factory = GroqLLM
    else:
        from evaluation.mock_llm import MockLLM
        factory = MockLLM

    cases = json.loads(config.GOLDEN_DATASET.read_text(encoding="utf-8"))["cases"]
    rows, trace_path = [], Path(args.out_dir) / f"eval_traces_{mode}.jsonl"
    trace_path.unlink(missing_ok=True)
    for rep in range(1, args.repeats + 1):
        for case in cases:
            print(f"[{rep}/{args.repeats}] {case['id']} ...", end=" ", flush=True)
            try:
                row, tr = run_case(case, mode, factory)
                tr.save(trace_path)
            except Exception as e:  # a crashed case is a failed case, not a crashed suite
                row = {"case_id": case["id"], "category": case["category"], "input": case["turns"][-1],
                       "task_success": False, "tool_selection_correct": None, "tool_execution_ok": None,
                       "answer_correct": None, "retrieval_ok": None, "memory_ok": None, "tools_called": "",
                       "latency_ms": 0, "input_tokens": "unavailable", "output_tokens": "unavailable",
                       "retrieved_chunks": "", "memory_used": "", "stopping_reason": "crash",
                       "model": "n/a", "mode": mode, "final_answer": "", "observations": f"crash: {e}"}
            row["repeat"] = rep
            rows.append(row)
            print("PASS" if row["task_success"] else "FAIL")
    df = pd.DataFrame(rows)
    suffix = "" if mode == "live" else "_mock"
    out = Path(args.out_dir)
    df.to_csv(out / f"results{suffix}.csv", index=False)
    write_summary(df, out / f"evaluation_summary{suffix}.md", mode)
    print(f"\nWrote {out / f'results{suffix}.csv'} and evaluation_summary{suffix}.md")
    print(f"Task success: {pct(df['task_success'])}")


if __name__ == "__main__":
    main()
