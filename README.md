# comparative-law-llm

Reproducible pipeline for comparing how input language may shift legal knowledge
sources and reasoning-unit distributions in LLM-generated liability analysis.

Stage 1 builds the reusable case corpus and neutral fact-pattern candidates. It
does not translate, call an LLM, or evaluate model outputs.

## Stage 1: Data Collection And Fact Patterns

### 1. Collect Korean damages / civil-liability cases

The Korean collector reuses the pilot logic in `collect_kr_cases.py`. It loads
`lbox/lbox_open::precedent_corpus`, filters damages/civil-liability keywords,
and keeps trial/appellate-oriented cases where possible.

```powershell
python collect_kr_cases.py --limit 50 --output outputs/kr_cases.csv
```

Use `--overwrite` only when you intentionally want to replace an existing file.
For a tiny smoke test:

```powershell
python collect_kr_cases.py --limit 5 --output outputs/kr_cases_sample.csv --overwrite
```

### 2. Collect U.S. tort / damages cases

The U.S. collector extracts the pilot notebook logic from `load_dataset.ipynb`
into `collect_us_cases.py`. It streams `harvard-lil/cold-cases`, reads
`opinions[*].opinion_text`, filters tort/damages/civil-liability keywords, and
adds state-specific filtering.

```powershell
python collect_us_cases.py --state California --limit 50 --output outputs/us_cases.csv
```

Supported states:

- `California`
- `New York`

State filtering records `state_filter_status` as `exact`, `inferred`,
`ambiguous`, or `unavailable`. Ambiguous rows are excluded by default; pass
`--include-ambiguous` only if you want them saved with QC flags.

Smoke test:

```powershell
python collect_us_cases.py --state California --limit 5 --output outputs/us_cases_sample.csv --overwrite
```

### 3. Preprocess collected CSVs

`preprocess_cases.py` keeps its original interface and still expects:

- `outputs/kr_cases.csv`
- `outputs/us_cases.csv`

Run:

```powershell
python preprocess_cases.py
```

This writes:

- `outputs/preprocessed_cases.csv`
- `outputs/case_metadata.csv`
- `outputs/preprocessing_summary.json`

### 4. Build unified case table and neutral fact candidates

`build_fact_patterns.py` reads `outputs/preprocessed_cases.csv` when available.
It can also read a compatible collected CSV or case table. The script writes a
unified case table and deterministic, rule-based neutral fact-pattern candidates.

```powershell
python build_fact_patterns.py --input outputs/preprocessed_cases.csv --output outputs/fact_patterns.jsonl
```

Smoke test:

```powershell
python build_fact_patterns.py --input outputs/preprocessed_cases.csv --output outputs/fact_patterns_sample.jsonl --limit 5 --overwrite
```

## Output Relationships

- `outputs/kr_cases.csv`: Korean collected source cases, compatible with
  `preprocess_cases.py`.
- `outputs/us_cases.csv`: U.S. collected source cases, compatible with
  `preprocess_cases.py`.
- `outputs/preprocessed_cases.csv`: cleaned/normalized case text and metadata
  used as the preferred fact-pattern input.
- `outputs/case_table.csv`: unified downstream table with stable IDs,
  normalized jurisdiction labels, raw text, collection notes, and quality flags.
- `outputs/fact_patterns.jsonl`: neutral fact-pattern candidates. In Stage 1,
  the neutral text remains in the source language and `neutral_fact_en` is
  always `null`; translation is reserved for Stage 2.
- `outputs/fact_pattern_failures.jsonl`: rows where extraction failed or failed
  QC.
- `outputs/fact_pattern_qc.csv`: per-case QC status, flags, removed legal
  signals, and legal-signal leakage checks.

Inspect QC quickly:

```powershell
Import-Csv outputs/fact_pattern_qc.csv | Group-Object status
Import-Csv outputs/fact_pattern_qc.csv | Select-Object case_id,status,quality_flags -First 10
```

## Important CLI Options

Collectors support:

- `--limit`
- `--seed`
- `--output`
- `--overwrite`
- `--min-text-length`
- `--max-text-length`

The U.S. collector also supports:

- `--state California`
- `--state New York`
- `--include-ambiguous`
- `--scan-limit`

## Notes

- API keys are not used in Stage 1.
- Stage 1 uses deterministic heuristics only.
- Failed or uncertain fact extraction is saved with quality flags rather than
  silently dropped.
