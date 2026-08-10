# Experiment 1 smoke test

Status: **PASS**

Selection is deterministic and subtype-overlap-first: CA_0929adea730d48f1, CA_2b7372b55b21300b, CA_4027b2c0efdfc079, KR_0ff7accd62d3ee4d, KR_258d6cc3522c11bc, KR_6274659eedc5c67d

| Check | Result |
|---|---|
| 12 successful calls (3 KR + 3 CA × 2) | PASS |
| Five requested sections | PASS |
| No truncation | PASS |
| Placeholder retention | PASS |
| No refusal/follow-up request | PASS |
| Raw response and metadata saved | PASS |
| No duplicate unique request | PASS |
| Blind request excludes case metadata | PASS |
| Second --resume run made zero calls | PASS |

Successful records: 12/12
Generation parameters: model_requested=gpt-5.6-luna, model_returned=gpt-5.6-luna, temperature=1.0, top_p=1.0, max_output_tokens=8000, reasoning_effort=low, seed_base=20260730.
Optional evaluator smoke: 12 schema-valid evaluations; 4 schema retries.
