# Experiment 1 conclusion reanalysis v2

## Why the legacy conclusion results were provisional

The legacy analysis matched evaluator-generated party strings exactly and selected `Counter.most_common()` when three replicate labels tied. It reproduced 432 matched comparisons, 57.9% agreement, κ=0.424, 63/70 cases with any change, 2 direct-flip cases, and 107 matched comparisons with a modal tie. The tie result depended on JSONL row order.

The source-derived registry contains 524 canonical case-party rows. The KO and EN source party sets matched in all 70 cases. Attached-role, grouped-party, duplicate-party, and ambiguous-placeholder cases remain audit flags rather than being silently merged.

## Canonical-party primary result

- Canonical case-party registry: 524
- Primary eligible parties: 333 / 524
- Agreement: 238/333 = 71.5%
- Disagreement: 95/333 = 28.5%
- Unweighted Cohen's κ: 0.569
- Eligible cases: 70
- Cases with any eligible conclusion change: 51/70 = 72.9%
- Direct likely↔unlikely candidates: 0 parties in 0 cases
- Likely/unlikely↔conditional/uncertain: 51 parties in 36 cases

By origin, KR-origin cases contributed 196/297 eligible parties, with 149/196 agreement (76.0%), κ=0.586, and 25/35 cases with a change. CA-origin cases contributed 137/227 eligible parties, with 89/137 agreement (65.0%), κ=0.523, and 26/35 cases with a change.

The two legacy direct-flip cases did not survive the v2 rules. For `CA_59b8f41e992ca4b0` / `PERSON_D`, the v2 result was KO `not_assessed` versus EN `unlikely`, so it was ineligible. For `KR_db98921e71fc497c` / `PERSON_D`, the v2 canonical recoding was `unlikely` in both languages.

## Replicate stability and exclusions

- Replicate-disagreement language-party units: 47/1048 = 4.5%
- `not_assessed` consensus units: 254/1048 = 24.2%
- Excluded canonical parties: 191/524
- Within-language instability: 0.263
- Cross-language discordance: 0.331
- Cross-minus-within: 0.071, 95% CI [0.044, 0.097], permutation p=5e-05

The primary exclusion reasons are nonexclusive: KO replicate disagreement 29, EN replicate disagreement 18, KO `not_assessed` consensus 131, and EN `not_assessed` consensus 123. A sensitivity analysis treating `not_assessed` as a category included 483 parties and produced 335/483 agreement (69.4%) and κ=0.586; it is not mixed with the primary estimate.

## Execution and validation

- Generation API calls: 0
- Full conclusion-evaluator API response attempts: 469
- Successful response recodes: 420/420
- Schema-invalid attempts retried: 49
- Transport retries: 0
- Final failures: 0
- Final resume verification: 0 API calls and 420 cache hits
- Separate two-response smoke test: 2/2 success, 0 schema retries, and 0 calls on resume
- Automated tests: 173 passed
- Protected input/result hashes: unchanged

The broad legal-system-marker statistics were not recomputed or altered. Their protected `summary.json` SHA-256 remains `203a2fc5f3bf075822b1c39b8144ed513d4bc9c8a50aab5971057d59278a482d`.

## Status

All v2 conclusion estimates remain automatically coded and provisional until review of `human_conclusion_validation_v2.csv`. It contains 231 rows covering every replicate disagreement, required source/mapping audit flag and language-discordant conclusion, plus stratified agreement examples. There are no v2 direct-flip candidates; reviewer fields are intentionally blank.

Remaining limitations include evaluator error, ambiguity in converting a response's multiple liability theories into one overall label, translation being confounded with input language, correlated parties within cases, and only three model replicates per language.
