# comparative-law-llm

Reproducible pipeline for comparing how input language may shift legal knowledge
sources and reasoning-unit distributions in LLM-generated liability analysis.

Stage 1 builds the reusable Korean and California case corpora. It does not
translate cases, call an LLM, or evaluate model outputs.

## Stage 1: Raw Case Collection

The canonical collectors are:

- `collect_kr_raw_cases.py`: Korean direct-tort appellate cases (v4)
- `collect_ca_raw_cases.py`: California state Court of Appeal tort cases

Both collectors apply deterministic screening and QC without calling an LLM.

Run a full collection from PowerShell:

```powershell
& .venv\Scripts\python.exe collect_kr_raw_cases.py --export-all-candidates --build-shortlist --select-final-sample --target-count 50 --court-level appellate --strict-direct-tort-only --do-not-use-year-for-sampling --sampling-config configs/tort_n50_sampling.yaml --seed 42
& .venv\Scripts\python.exe collect_ca_raw_cases.py --target-count 50 --scan-limit 750000 --seed 42 --overwrite
```

Smoke-test the collectors without writing outputs:

```powershell
& .venv\Scripts\python.exe collect_kr_raw_cases.py --scan-limit 100 --preview-only
& .venv\Scripts\python.exe collect_ca_raw_cases.py --target-count 3 --scan-limit 1000 --preview-only
```

Use `--overwrite` only when existing output files should be replaced.

### Default outputs

Korean v4 outputs are written under `outputs/raw/kr_v4`, including:

- complete broad, appellate, direct-tort, strict-eligible, and excluded pools
- `kr_direct_tort_shortlist_100.jsonl`
- `kr_direct_tort_shortlist_100_qc.csv`
- `kr_cases_selected_50_pre_qc.jsonl`
- `kr_cases_selected_50_final.jsonl`
- `kr_cases_summary.json`

The Korean manifest is written to
`outputs/manifests/kr_v4_case_manifest.csv`. Existing v3 outputs are retained.

California outputs are written under `outputs/raw/ca_v3`, including:

- `ca_cases_selected_<target-count>.jsonl`
- `ca_cases_qc.csv`
- `ca_cases_summary.json`
- candidate-pool JSONL files

California manifest and sampling-alignment outputs are written to:

- `outputs/manifests/ca_v3_case_manifest.csv`
- `outputs/manifests/kr_ca_sampling_alignment.csv`

Output locations can be changed with `--output-dir`, `--manifest-output`, and,
for California, `--alignment-output`.

## Downstream Fact-Pattern Utilities

`preprocess_cases.py` supports the legacy collected CSV interface and writes:

- `outputs/preprocessed_cases.csv`
- `outputs/case_metadata.csv`
- `outputs/preprocessing_summary.json`

```powershell
& .venv\Scripts\python.exe preprocess_cases.py
```

`build_fact_patterns.py` builds a unified case table and deterministic neutral
fact-pattern candidates from a compatible preprocessed case table.

```powershell
& .venv\Scripts\python.exe build_fact_patterns.py --input outputs/preprocessed_cases.csv --output outputs/fact_patterns.jsonl
```

Smoke test:

```powershell
& .venv\Scripts\python.exe build_fact_patterns.py --input outputs/preprocessed_cases.csv --output outputs/fact_patterns_sample.jsonl --limit 5 --overwrite
```

Stage 1 uses deterministic heuristics only. Failed or uncertain extraction is
recorded with QC flags rather than silently dropped.

## Stage 2: Neutral facts (KR 35 + California 35)

Version `stage2-neutral-facts-35x35-v1` creates source-grounded neutral facts
for the four experimental conditions KR-case-KO, KR-case-EN, CA-case-EN, and
CA-case-KO. The immutable Stage 1 snapshots are:

- `outputs/raw/kr_v4/kr_cases_selected_35.jsonl` — SHA-256 `ca53460a99df2a59ffa1b4047cdfa406dd2afac54b053dccb99724dd850b8a49`
- `outputs/raw/ca_v4/ca_cases_selected_35.jsonl` — SHA-256 `35f9028cb5be3f331bc3df54511388986910ea86c7100ebb070d7ca2a2595aeb`

All 35 California records are inputs, including five former reserve records
and records whose automatic Stage 1 eligibility flag is false. Selection and
subtype balancing are not rerun. KR uses `raw_text`; CA uses
`main_opinion_text`.

The generation order is deliberately sequential:

```text
KR raw → Korean source master → English translation
CA raw → English source master → Korean translation
```

Raw opinions never enter translation requests. Extraction, source
neutralization, source-grounding verification, translation, and translation
verification are separate model calls. Each phase has deterministic validation,
atomic checkpoints, request-hash caching, raw-response preservation, and
resume support. A verifier records findings but never edits generated text.

### Pipeline commands

Set `LETSUR_API_KEY` in the environment; never commit it. Validate and inspect
the complete deterministic request plan without API calls:

```powershell
& .venv\Scripts\python.exe generate_neutral_fact_patterns.py --kr-input outputs\raw\kr_v4\kr_cases_selected_35.jsonl --ca-input outputs\raw\ca_v4\ca_cases_selected_35.jsonl --output-dir outputs\neutral\stage2-neutral-35x35-v1 --dry-run --resume
```

Run source extraction/neutralization only after dry-run and mock tests pass:

```powershell
& .venv\Scripts\python.exe generate_neutral_fact_patterns.py --kr-input outputs\raw\kr_v4\kr_cases_selected_35.jsonl --ca-input outputs\raw\ca_v4\ca_cases_selected_35.jsonl --output-dir outputs\neutral\stage2-neutral-35x35-v1 --model gpt-5.6-luna --base-url https://gw.letsur.ai/v1 --concurrency 2 --max-retries 5 --resume
```

```powershell
& .venv\Scripts\python.exe translate_neutral_fact_patterns.py --kr-source-neutral outputs\neutral\stage2-neutral-35x35-v1\source_neutral_kr.jsonl --ca-source-neutral outputs\neutral\stage2-neutral-35x35-v1\source_neutral_ca.jsonl --output-dir outputs\neutral\stage2-neutral-35x35-v1 --model gpt-5.6-luna --base-url https://gw.letsur.ai/v1 --concurrency 2 --max-retries 5 --resume
```

```powershell
& .venv\Scripts\python.exe verify_neutral_fact_patterns.py --source-neutral-input outputs\neutral\stage2-neutral-35x35-v1 --translation-input outputs\neutral\stage2-neutral-35x35-v1 --output-dir outputs\neutral\stage2-neutral-35x35-v1 --verifier-model gpt-5.6-luna --base-url https://gw.letsur.ai/v1 --resume
& .venv\Scripts\python.exe merge_neutral_pairs.py --input-dir outputs\neutral\stage2-neutral-35x35-v1
```

Use `--case-id` for the required KR/CA real-API smoke cases before starting all
70. `--mock-response-dir` accepts `<dir>/<stage>/<case_id>.json` fixtures. Run
all deterministic and mock tests with:

```powershell
& .venv\Scripts\python.exe -m pytest -q
```

### Progressive rollout and resume safety

Never run all 70 cases as one unattended batch. Use separate cumulative case-ID
files for Stage A (3+3), Stage B (10+10), Stage C (20+20), and Stage D (35+35).
Review each batch report and stop conditions before explicitly starting the next
stage. The implemented entry points support `--case-id`, `--case-id-file`,
`--batch-name`, `--max-cases-per-origin`, `--stop-on-hard-failure`,
`--retry-failed`, `--retry-warnings`, `--recheck-deterministic`, and configurable
hard/API failure rates. Failed cases are appended to `quarantine.jsonl`.

Default `--resume` never calls the API for a case with a stored response,
regardless of pass/warning/fail status. `--recheck-deterministic` uses cached
records with zero API calls. Only explicit retry/regenerate flags can make a new
request. Raw responses are versioned, mock/real cache provenance must match, and
run history is append-only.

The completed six-case calibration is in
`outputs/neutral/stage2-neutral-35x35-v2/stage-a-calibration`. Its merge is
`completed_subset` (6 complete, 64 missing); the 70-case run has not started.

### Outputs and human QC

Outputs live under `outputs/neutral/stage2-neutral-35x35-v1`. The manifest and
validation report pin the inputs. Source segments, evidence, masters,
translations, both verifier results, API errors/usage, raw responses, prompt
snapshots, and request cache are separate artifacts. `neutral_pairs_all.jsonl`
always follows all 70 manifest case IDs, including missing or failed stages;
`neutral_pairs_pass.jsonl` contains only records passing every automatic gate.

Review `human_qc_template.csv`, compare fact units with cited evidence, and
fill `human_qc_status` and notes before selecting experimental inputs. Automatic
passes are not a gold dataset.

Known limitations: deterministic name/leakage checks cannot prove semantic
anonymization; negation and epistemic checks are conservative lexical checks;
model verifiers can still agree on the same error; metric conversion and
long-opinion candidate selection require human review. Automatic regeneration
is intentionally disabled so original outputs remain auditable.

See `STAGE2_IMPLEMENTATION_REPORT.md` for the implementation inventory,
subtype distributions, prompt and schema sources, test evidence, and smoke/full
run status.
