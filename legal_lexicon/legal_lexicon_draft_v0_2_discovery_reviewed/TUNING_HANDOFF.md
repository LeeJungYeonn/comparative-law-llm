# Next stage: tuning 10 only

Status of this package: DISCOVERY-REVIEWED DRAFT. It is NOT frozen v1.0.

## Before tuning
1. Researcher reviews `discovery_candidates_human_review_filled_draft.csv`.
2. Confirm/revise rows with `human_keep=true`.
3. Do not inspect validation or confirmatory text.

## Tuning
Use ONLY case IDs in `tuning_case_manifest.csv` (KR 5 + US 5).

Run `lexicon_matcher_draft_v0_2.py` on each tuning `main_opinion_text` and save exact hit spans.

Review:
- false positives;
- category errors;
- concept-family errors;
- overly broad literals/regexes;
- false negatives by sampling no-hit sentences and reading tuning opinions.

Any draft change must be recorded in `tuning_revision_log.csv`, and:
- `evidence_split` must be `tuning`, `discovery`, or `official_resource`;
- never `validation`;
- never `confirmatory`.

After tuning changes are complete, create a PRE-VALIDATION candidate lexicon and hash it.

## Validation
Run the pre-validation candidate ONCE on IDs in `validation_case_manifest_LOCKED.csv`.
Validation is evaluation only. Never modify the lexicon after seeing validation results.

Then freeze the unchanged pre-validation candidate as `lexicon_v1.0` and apply it to confirmatory 160.
