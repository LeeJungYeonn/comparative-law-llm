# Stage 2 implementation report

Version: `stage2-neutral-facts-35x35-v1`

## 1–8. Repository, inputs, and candidate extraction

The Git repository root is the inner `comparative-law-llm/` directory. Existing
Stage 2-adjacent code was `build_fact_patterns.py`, `preprocess_cases.py`,
`qc_rules.py`, `pipeline/text_utils.py`, and `pipeline/io_utils.py`. It was a
legacy deterministic candidate builder, not a source-grounded multi-call Stage
2 implementation. The new code reuses its fact-section extraction and text
normalization as a prefilter.

Canonical immutable inputs:

| Origin | Path | Count | Source field | SHA-256 |
|---|---|---:|---|---|
| KR | `outputs/raw/kr_v4/kr_cases_selected_35.jsonl` | 35 | `raw_text` | `ca53460a99df2a59ffa1b4047cdfa406dd2afac54b053dccb99724dd850b8a49` |
| CA | `outputs/raw/ca_v4/ca_cases_selected_35.jsonl` | 35 | `main_opinion_text` | `35f9028cb5be3f331bc3df54511388986910ea86c7100ebb070d7ca2a2595aeb` |

KR subtype distribution: traffic 10, medical 7, premises 6, employer 4,
product 4, and privacy, intentional, general personal injury, property damage
1 each. CA distribution: premises 9, traffic 7, general personal injury 6,
medical 4, product 4, employer 2, and privacy, intentional, property damage 1
each. It includes all five former CA reserves and all 14 CA records with
`auto_strict_eligible=false`.

`pipeline/source_segmentation.py` assigns stable `SRC####` identifiers and
deterministic offsets. If the full source fits `--max-input-tokens`, every
segment is sent in one ordered extraction call. Longer sources are split into
ordered overlapping chunks and every source segment is processed. Keyword and
fact-section signals are metadata/priority hints only; they never exclude the
rest of the source. Segment and character coverage, missing ranges, and call
counts are recorded per case.

## 9–15. Prompt source (complete authoritative text)

Each linked UTF-8 file is the complete prompt, not an excerpt. Generation also
copies byte-identical snapshots into the run's `prompts/` directory.

1. Evidence extraction: `prompts/extract_evidence_ko_v1.txt`, `prompts/extract_evidence_en_v1.txt`
2. Source neutralization: `prompts/neutralize_ko_v2.txt`, `prompts/neutralize_en_v2.txt`
3. Translation: `prompts/translate_ko_to_en_v2.txt`, `prompts/translate_en_to_ko_v3.txt`
4. Source grounding: `prompts/verify_grounding_ko_v1.txt`, `prompts/verify_grounding_en_v1.txt`
5. Translation verification: `prompts/verify_translation_ko_en_v1.txt`, `prompts/verify_translation_en_ko_v1.txt`

All prompts treat source material as untrusted, forbid following embedded
instructions, and require one complete JSON object. Translation prompts receive
only master facts, fact IDs, epistemic metadata, and placeholders—not raw text,
evidence, subtype, title, court, citation, date, or outcome.

## 16–20. Schemas, deterministic gates, and API operations

The complete authoritative schemas and taxonomies are in
`pipeline/stage2_schema.py`. Dataclass adaptation, count/uniqueness/hash checks,
and manifest writing are in `pipeline/stage2_input.py`.

`pipeline/leakage_checks.py` treats source sentence IDs as authoritative and
reconstructs exact excerpts deterministically. It validates fact/evidence links;
legal, disposition, jurisdiction, currency, case-number, and placeholder
leakage; source completeness; fact order; bilingual numbers/fractions; units;
semantic anchors; lexical negation warnings; and target-language residue. It
normalizes mph→km/h, feet→m, comma-formatted miles→km, pounds→kg, and °F→°C.
Actual normalized value changes are hard failures; surface-form and unresolved
normalization differences are warnings.

`pipeline/llm_client.py` uses the OpenAI-compatible `/chat/completions`
contract without adding a runtime package dependency. It reads only
`LETSUR_API_KEY`, tries JSON Schema, falls back to JSON object and then JSON-only
mode, and drops unsupported seed/temperature only after an explicit gateway
error. It records requested versus effective parameters. Exponential backoff,
`Retry-After`, atomic raw responses, request hashes, caches, and token usage are
implemented. `pipeline/response_parser.py` handles fences, surrounding prose,
and trailing commas but rejects truncation or missing critical fields without
inventing content. `pipeline/checkpoint.py` recovers only a malformed final
partial JSONL line.

## 21–23. Tests and calibration status

- Full suite: **109 passed**; Stage 2 regression subset: **39 passed**.
- Actual-input validation: **pass**, KR 35 + CA 35, total 70, unique IDs and
  source hashes valid, exact expected subtype distributions.
- Actual-input dry-run: **pass**, all 70 segmented; no API calls.
- Mock end-to-end: **pass**, one KR and one CA case through extraction,
  source neutralization, both translations, both independent verifiers, and
  merge; `neutral_pairs_all.jsonl` retained all 70 manifest IDs and two mock
  records passed every gate.
- Real Stage A calibration: **completed for KR 3 + CA 3 only** under
  `outputs/neutral/stage2-neutral-35x35-v2/stage-a-calibration`. All six have
  100% segment and character coverage, event, harm, and causal sequence. The
  longest source used two ordered extraction calls. All six translations have
  no deterministic hard failure; all grounding and translation verifiers have
  no validated hard failure or consistency violation. Final automatic results
  are 1 pass and 5 warnings, all pending human QC. The merge is
  `completed_subset`: 6 complete pairs and 64 missing, with no count mismatch.
  Re-running all three API phases with default `--resume` made **0 API calls**.
  All current calibration provenance is real; no mock record is mixed in.

## 24–26. Full-run commands, execution status, limitations

The complete commands are in the Stage 2 README section. Required order:

1. generation with `--resume`;
2. translation with `--resume`;
3. both verifiers with `--resume`;
4. merge after reviewing every stage result.

The 70-case real API run has **not** been performed. Stage A is complete, but
five cases remain automatic warnings and human QC is pending. Stage B must not
start without explicit user approval. Stages B, C, and D are separate cumulative
batches and are never chained automatically.

Known limitations are lexical anonymization/name detection, conservative
cross-language negation and epistemic heuristics, and verifier correlation.
Human QC is mandatory, and pre-QC results must not be called gold data.

## Files added or changed

Entry points: `generate_neutral_fact_patterns.py`,
`translate_neutral_fact_patterns.py`, `verify_neutral_fact_patterns.py`, and
`merge_neutral_pairs.py`.

Pipeline modules: `pipeline/stage2_input.py`, `source_segmentation.py`,
`factual_evidence.py`, `canonical_neutralization.py`,
`neutral_translation.py`, `neutral_verification.py`, `leakage_checks.py`,
`response_parser.py`, `llm_client.py`, `stage2_schema.py`, `checkpoint.py`, and
`stage2_runtime.py`.

Tests: `tests/test_stage2.py`. Documentation: `README.md` and this report.

Calibration artifacts: `calibration_report_stage_a.json`,
`calibration_report_stage_a.csv`, `quality_report.csv`, versioned raw responses,
append-only `run_history`, and `quarantine.jsonl` in the Stage A output directory.
