# Revised corpus v2 — implementation and run report

Run date: 2026-08-10 (Asia/Seoul)

## Repository audit

Reusable legacy components were the atomic/checkpoint output pattern, stable
hashing, COLD opinion-type handling, factual-sufficiency categories, and
deterministic leakage checks. The old Korean collector explicitly targeted
appellate records and excluded Supreme Court records. The old U.S. collector
targeted California `court_type=SA`, not multiple state courts of last resort.
The old rule-based fact builder is useful for cleanup and QC but is not a final
source-grounded extractor. All old code and pilot artifacts remain under
`legacy/`.

## Actual sources and run results

- KR: `lbox/lbox_open`, `precedent_corpus`, cached revision
  `10429acf7e13d7ef2ea4187ffbd685490289a82c`, CC BY-NC 4.0.
- U.S.: `harvard-lil/cold-cases`, `default`, revision
  `5d8d0d8457ef63b6463af9737da21d3badd924ad`, CC0 1.0.
- Window: 2000-01-01 through 2025-12-31.

The full U.S. metadata audit covered 544,841 `court_type=S` records and froze:

| State | Region | Total S | Metadata liability candidates | General | Medical | Product | Employer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pennsylvania | Northeast | 50,959 | 501 | 17 | 38 | 139 | 153 |
| Michigan | Midwest | 42,637 | 154 | 4 | 5 | 62 | 7 |
| Louisiana | South | 66,339 | 707 | 9 | 67 | 233 | 155 |
| Nevada | West | 16,797 | 413 | 100 | 11 | 26 | 2 |
| West Virginia | South | 11,667 | 2,511 | 2,330 | 41 | 92 | 0 |

`usable_full_text_count` is zero/deferred in this metadata-only audit by design;
opinion text was not downloaded for all 544,841 records. Actual full-text
usability was checked during candidate collection. The frozen choice was not
revised afterward.

### Deterministic funnels

KR: 1,100 source rows scanned → 300 broad candidates → 0 high-confidence
court/date eligible → 259 adequate-length opinions → 270 heuristic
fact-sufficient (overlapping gate) → 0 strict source eligible.

U.S.: 544,841 records in the full state audit; a pinned three-shard collection
frame returned 388 SQL-prefiltered rows → 377 Python-confirmed broad candidates
→ 377 date/court eligible → 359 usable controlling opinions → 258 heuristic
fact-sufficient → 211 strict source eligible.

Strict U.S. pool by state: Louisiana 48, Michigan 37, Nevada 34,
Pennsylvania 36, West Virginia 56. Strict pool by domain: general 105,
medical 81, product 12, employer 7, other 6. Thus the original 40/20/20/20
target is already unavailable on the U.S. strict pool before fact extraction.

### Lower-court supplementation

KR had no high-confidence highest-court record on which to attempt linkage.
U.S. attempted 119 fact-insufficient cases; exact identifiers produced 0
reliable links, 0 successful supplements, and 119 still fact-insufficient
records. No fuzzy-title link or inferred fact was accepted.

## Blocking evidence

The cached LBox `precedent_corpus` exposes only `id` and `precedent`. In the 300
saved broad candidates, every record lacked a high-confidence decision date and
none had enough mutually reinforcing metadata/header evidence for
high-confidence Supreme Court status (40 were medium, 260 low/unknown). The
pipeline therefore correctly admitted zero Korean records rather than relaxing
the specification.

`OPENAI_API_KEY` was absent. A three-case extraction smoke exercised the
failure/checkpoint path and recorded three explicit failures; it made no API
request. Consequently source extraction, translation, LLM-assisted QC, and the
final 100+100 sample were not fabricated. `collection_summary.json` records
status `shortfall` and false final invariants.

## Files added

- `pipeline_v2/`: schemas, deterministic rules, IO, Hugging Face access,
  collection evaluators, and resumable Responses runtime.
- `collect_kr_supreme_cases_v2.py`, `audit_us_state_cases_v2.py`,
  `collect_us_state_highcourt_cases_v2.py`, `link_case_families_v2.py`.
- `extract_neutral_facts_v2.py`, `translate_neutral_facts_v2.py`,
  `qc_neutral_facts_v2.py`, `finalize_case_sample_v2.py`.
- `prompts_v2/`, `requirements-v2.txt`, and `tests/test_v2_pipeline.py`.
- README v2 documentation and this report.

The temporary diagnostic helper used while validating remote Parquet access is
not part of the production pipeline.
