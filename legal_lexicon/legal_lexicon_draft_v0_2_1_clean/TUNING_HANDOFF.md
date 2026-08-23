# Tuning handoff — draft v0.2.1 clean

Use only `tuning_case_manifest.csv` (KR 5 + US 5).
Do not inspect validation or confirmatory text.

Review TP/FP/category/concept errors and sample no-hit text for FN.
Record every allowed change in `tuning_revision_log.csv`.

After tuning:
1. freeze/hash a pre-validation candidate;
2. run validation 10 once;
3. do not modify based on validation;
4. freeze unchanged candidate as lexicon_v1.0.
