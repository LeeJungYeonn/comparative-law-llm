# Stage 2 paired QC implementation report

## Generation-version correction

The paired-QC source is now `stage2-neutral-facts-35x35-v4`, generated with the
`english-v1` profile through `run_stage2_v4.py`.

- Generation instructions are English in all 12 generation prompts.
- Korean records use only `neutralize_ko_v7_en`; the earlier
  `neutralize_ko_v5`/`neutralize_ko_v6` mixture is not reused.
- Korean source text and Korean master output remain Korean. Only the prompt
  instructions and schema terminology are standardized in English.
- The dataset version is `stage2-neutral-facts-35x35-v4`.
- The schema version is `stage2-v3.1`.
- The neutralization policy is `canonical-neutralization-en-v1`.
- `run_manifest.json`, `input_manifest.json`, prompt provenance, and generated
  records use the same generation profile and dataset version.
- The earlier v3 artifacts remain unchanged for auditability.

`run_stage2_v3.py` retains the legacy v3 profile and also accepts
`--generation-profile english-v1`. The v4 wrapper selects the English profile
and v4 output directory by default.

## Paired-QC design

The entry point is `qc_neutral_fact_pairs.py`; reusable logic is in
`pipeline/paired_qc.py`. QC output is
`outputs/neutral/stage2-paired-qc-v1`.

Source QC and translation QC have separate inputs, prompts, schemas, automatic
outputs, human-review sheets, imports, and checkpoints. Deterministic hard
findings override model passes. Human edits create separate validated artifacts
and invalidate child translations by parent hash.

The seven QC prompts have English instructions:

- `prompts/qc_source_neutral_ko_v1_en.txt`
- `prompts/qc_source_neutral_en_v1_en.txt`
- `prompts/qc_translation_ko_to_en_v1_en.txt`
- `prompts/qc_translation_en_to_ko_v1_en.txt`
- `prompts/qc_back_translate_en_to_ko_v1_en.txt`
- `prompts/qc_back_translate_ko_to_en_v1_en.txt`
- `prompts/qc_disagreement_adjudication_v1_en.txt`

## Stage A result and stop point

The six-case Stage A generation subset was produced in
`outputs/neutral/stage2-neutral-35x35-v4`. Generation-version consistency
passed. Some generated translations or verifiers are absent because their
upstream automatic quality gates failed; they were not silently promoted.

Automatic source QC completed for three Korean and three California cases:

| Validated status | Count |
| --- | ---: |
| pass | 1 |
| warning | 2 |
| fail | 3 |

The individual results are:

- pass: `KR_043490cec7ae93fa`
- warning: `CA_90588b6bc671dd08`, `CA_59b8f41e992ca4b0`
- fail: `KR_09a496b96b9d302d`, `KR_0f10f050f02ad48d`,
  `CA_78a282aae14272a7`

The pending six-row human source-review sheet is:

`outputs/neutral/stage2-paired-qc-v1/human_source_review.csv`

All six source masters were subsequently imported as
`accepted_with_edits`. Translations were regenerated from the exact
human-validated masters into `regenerated_translations.jsonl`, without
overwriting the v4 generation artifacts. All stale-translation markers were
resolved.

Automatic translation QC then completed for all six cases. Its validated result
is fail 6, so execution stopped at the human bilingual-review gate:

`outputs/neutral/stage2-paired-qc-v1/human_translation_review.csv`

No bilingual human decisions have been imported and Stage B was not started.

The input-validation phase has `generation_consistency_status: pass` and
`quality_status: fail`. The latter reflects missing/failed downstream
translation artifacts in the generated subset; it does not invalidate the
available source masters for source QC.

## Verification

`python -m pytest -q` passes: **140 passed**.

Tests cover profile isolation, manifest consistency, canonical relation-metadata
normalization, provenance, mixed-prompt detection, deterministic overrides,
recognition risk, human imports, invalid edits, parent hashes, stale
translations, accepted-only exports, diagnostic back-translation, and
non-overwrite behavior.

## Resume sequence after human source review

```powershell
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --import-human-source-review outputs\neutral\stage2-paired-qc-v1\human_source_review.csv
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --regenerate-stale-translations --resume
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --translation-qc-only
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --import-human-translation-review outputs\neutral\stage2-paired-qc-v1\human_translation_review.csv
```
