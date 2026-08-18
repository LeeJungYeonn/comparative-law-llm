# Exp 1 smoke test — blocked API attempt 2

Status: **BLOCKED_API_USAGE_LIMIT**

- Primary requests: 0/16 successful
- Letsur result: `HTTP 429 usage_limit_exceeded — COST limit exceeded` for all requests
- Additional low/medium/high check: not run because the primary-pass precondition was not met
- Model requested: `gpt-5.6-luna`; returned model identifier: unavailable
- Prompt version: `exp1-court-opinion-v1`
- Corpus files remained unchanged
- Hypothesis-marker evaluation and PCA were not performed
- Pipeline ready to freeze for full Exp 1: **FALSE**

The legacy fixed-heading analyst prompt was replaced only to satisfy the required judicial-opinion format and KO/EN semantic equivalence. No change was based on jurisdictional-marker strength.
