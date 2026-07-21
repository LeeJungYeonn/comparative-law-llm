# comparative-law-llm

Reproducible pipeline for comparing how input language may shift legal knowledge
sources and reasoning-unit distributions in LLM-generated liability analysis.

Stage 1 builds the reusable Korean and California case corpora. It does not
translate cases, call an LLM, or evaluate model outputs.

## Stage 1 v3: Raw Case Collection

The canonical collectors are:

- `collect_kr_raw_cases.py`: Korean tort appellate cases
- `collect_ca_raw_cases.py`: California state Court of Appeal tort cases

Both collectors stream their source datasets, apply deterministic screening and
QC, sample with a fixed seed, and create their output directories automatically.
When neither `--export-all-candidates` nor `--select-final-sample` is supplied,
both outputs are enabled.

Run a full collection from PowerShell:

```powershell
& .venv\Scripts\python.exe collect_kr_raw_cases.py --target-count 50 --scan-limit 200000 --seed 42 --overwrite
& .venv\Scripts\python.exe collect_ca_raw_cases.py --target-count 50 --scan-limit 750000 --seed 42 --overwrite
```

Smoke-test the collectors without writing outputs:

```powershell
& .venv\Scripts\python.exe collect_kr_raw_cases.py --target-count 3 --scan-limit 100 --preview-only
& .venv\Scripts\python.exe collect_ca_raw_cases.py --target-count 3 --scan-limit 1000 --preview-only
```

Use `--overwrite` only when existing output files should be replaced.

### Default outputs

Korean outputs are written under `outputs/raw/kr_v3`, including:

- `kr_cases_selected_<target-count>.jsonl`
- `kr_cases_qc.csv`
- `kr_cases_summary.json`
- candidate-pool JSONL files

The Korean manifest is written to
`outputs/manifests/kr_v3_case_manifest.csv`.

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
