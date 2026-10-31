> **MOCK RUN — NOT MODEL PERFORMANCE.** These numbers come from a deterministic rule-based stand-in, used only to verify the evaluation harness without an API key. Run `python -m evaluation.evaluate --live` with a real `GROQ_API_KEY` to measure the actual model.

# Evaluation summary (mock)

- Model: `mock-rule-based (NOT a real model)`  |  cases: 20  |  runs: 20

| Metric | Result |
|---|---|
| Task success (all applicable checks) | 100.0% (20/20) |
| Tool selection correct | 100.0% (18/18) |
| Tool execution as expected | 100.0% (15/15) |
| Answer correct | 100.0% (19/19) |
| Retrieval evidence correct | 100.0% (4/4) |
| Memory/state correct | 100.0% (5/5) |
| Latency mean / median / max | 6 / 3 / 14 ms |
| Token usage | unavailable (provider returned no usage for these runs) |

## By category

| Category | Success |
|---|---|
| direct | 100.0% (1/1) |
| fee_tool | 100.0% (2/2) |
| scholarship_tool | 100.0% (2/2) |
| record_tool | 100.0% (1/1) |
| support_ticket | 100.0% (1/1) |
| policy_rag | 100.0% (3/3) |
| missing_info | 100.0% (1/1) |
| missing_record | 100.0% (1/1) |
| unsupported_policy | 100.0% (1/1) |
| tool_failure | 100.0% (1/1) |
| memory_write | 100.0% (1/1) |
| memory_read | 100.0% (2/2) |
| sensitive | 100.0% (1/1) |
| followup | 100.0% (1/1) |
| llm_error | 100.0% (1/1) |

## Failed cases

None.

## Known limitations

- Deterministic graders check keywords/numbers/tools; they cannot judge groundedness or tone (Unit II 'limitation of deterministic graders').
- 20 cases is small; one run is not proof of reliability (use `--repeats`).
- Latency includes network + model time and varies per run; it is graded-turn latency only.
