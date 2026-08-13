# comparative-law-llm

Reproducible pipeline for comparing how input language may shift legal knowledge
sources and reasoning-unit distributions in LLM-generated liability analysis.

The original Korean/California pilot is preserved unchanged under `legacy/`.
Commands and outputs in that directory describe the earlier experiment only.

## Revised KR–U.S. State-Law Corpus v2

The v2 pipeline builds a separate 200-case research corpus without running the
later judgment-generation, PCA, marker, or statistical experiments. Its source
window is `2000-01-01` through `2025-12-31` for both countries.

### Sources and inclusion rules

- Korea: `legalize-kr/precedent-kr` at a pinned Git revision. This repository
  derives from the Korean National Law Information Center API. The collector
  reads YAML-frontmatter Markdown only from `민사/대법원` and requires
  `사건종류=민사`, `법원명=대법원`, `법원등급=대법원`, an in-window
  `선고일자`, and a usable `판례내용` section. LBox is no longer used to guess
  court or date eligibility.
- United States: `harvard-lil/cold-cases`, config `default` (CC0 1.0). The
  source revision is pinned in `us_state_selection.json`. Main-corpus records
  require structured `court_type == "S"`; `SA`, `ST`, and every federal type
  are excluded. The controlling opinion preference is `010combined`,
  `015unamimous`, `020lead`, then `080onthemerits`. Concurrences and dissents
  remain separate, and plurality records require review.

The full U.S. availability audit projects only metadata columns from every
COLD Parquet shard. It computes state-high-court and domain availability for
all represented states without downloading 32 GB of opinion text. Consequently
`usable_full_text_count` is explicitly marked as deferred in the metadata audit;
actual opinion usability is checked record-by-record during candidate
collection. Five states are then frozen deterministically using availability,
domain coverage, publication metadata, and regional diversity. Candidate
collection scans all 32 pinned Parquet shards; a positive `--parquet-shards`
value is partial smoke mode. It never changes the frozen state set based on
downstream results.

Every candidate receives one `primary_domain`: general negligence/personal
injury, medical/professional liability, product liability, or other civil
liability. Employer and supervisory doctrines are nonexclusive
`liability_theories`/`secondary_tags`. After full collection, a bounded-flow
calculation freezes one identical KR/US domain allocation, preferring 40--55
general, 20--30 medical, 10--25 product, and the remainder other. The U.S.
sample totals 100 with all five states represented at 10--30 records each.

### Family linkage and neutral facts

`link_case_families_v2.py` accepts only exact case numbers, source IDs, or
structured history/cross-reference identifiers. Fuzzy title similarity is not
sufficient. The highest-court opinion always remains the reference judgment;
linked lower-court text is supplied only as a separately labeled factual
supplement only when at least one mandatory dimension is missing. Mandatory
dimensions are parties/relationships, conduct or omission, harm, and a minimal
causal/event sequence. Context, detailed chronology, and defense facts are
optional enrichment; 5/7 remains preferred but is not an absolute gate.

Neutral facts are built in three distinct stages:

1. `extract_neutral_facts_v2.py` extracts atomic fact units with source spans,
   epistemic status, stable placeholders, and normalization metadata in the
   source language.
2. `translate_neutral_facts_v2.py` translates only those units and preserves
   fact IDs, placeholders, numbers, units, negation, and epistemic status.
3. `qc_neutral_facts_v2.py` runs deterministic grounding, leakage, alignment,
   and sufficiency checks, followed optionally by a separate evidence-bearing
   LLM audit. LLM QC proposes corrections but never edits facts silently.

The API stages use the project's established Letsur OpenAI-compatible gateway:
`https://gw.letsur.ai/v1/chat/completions`, bearer authentication from
`LETSUR_API_KEY`, and JSON Schema with JSON-object/JSON-only fallbacks. `.env`
is loaded by `python-dotenv`; the key value is never printed or persisted. The
model is configurable by `--model`, `FACT_EXTRACTION_MODEL`, or `LETSUR_MODEL`
and defaults to the previously used `gpt-5.6-luna`. Each request key includes
the case ID, stage, prompt version, input hash, model, and gateway identifier. Raw responses, request
IDs, returned model IDs, timestamps, usage, hashes, errors, and bounded retries
are retained under `outputs_v2/raw_api_responses/`. `--resume` skips successful
requests. `--mock-response-dir` and `--dry-run` provide no-API smoke paths.

### Reproduction commands

```powershell
& .venv\Scripts\python.exe -m pip install -r requirements-v2.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/legalize-kr/precedent-kr.git .cache_v2/precedent-kr
git -C .cache_v2/precedent-kr config core.longpaths true
git -C .cache_v2/precedent-kr sparse-checkout set --no-cone "/민사/대법원/"
& .venv\Scripts\python.exe smoke_test_letsur_v2.py
& .venv\Scripts\python.exe collect_kr_supreme_cases_v2.py --source-dir .cache_v2/precedent-kr --start-date 2000-01-01 --end-date 2025-12-31 --candidate-target 1200 --seed 20260810 --output-dir outputs_v2
& .venv\Scripts\python.exe audit_us_state_cases_v2.py --start-date 2000-01-01 --end-date 2025-12-31 --source-mode parquet --preserve-states-from outputs_v2/us_state_selection.json --output-dir outputs_v2 --overwrite
& .venv\Scripts\python.exe collect_us_state_highcourt_cases_v2.py --states-from outputs_v2/us_state_selection.json --candidate-target 750 --source-mode parquet --parquet-shards 0 --seed 20260810 --output-dir outputs_v2
& .venv\Scripts\python.exe assess_candidate_feasibility_v2.py --overwrite
& .venv\Scripts\python.exe link_case_families_v2.py --input outputs_v2/kr_supreme_candidates.jsonl --input outputs_v2/us_state_highcourt_candidates.jsonl --output-dir outputs_v2
& .venv\Scripts\python.exe extract_neutral_facts_v2.py --input outputs_v2/candidates_with_family_links.jsonl --limit-per-country 3 --output-dir outputs_v2/smoke_3x3 --resume
& .venv\Scripts\python.exe prepare_bulk_manifest_v2.py --overwrite
& .venv\Scripts\python.exe extract_neutral_facts_v2.py --input outputs_v2/bulk_extraction_manifest_200.jsonl --output-dir outputs_v2/bulk_200 --resume
& .venv\Scripts\python.exe translate_neutral_facts_v2.py --input outputs_v2/bulk_200/neutral_facts_source.jsonl --output-dir outputs_v2/bulk_200 --resume
& .venv\Scripts\python.exe qc_neutral_facts_v2.py --input outputs_v2/bulk_200/neutral_facts_bilingual.jsonl --llm-model gpt-5.6-luna --llm-warnings-only --output-dir outputs_v2/bulk_200 --resume
& .venv\Scripts\python.exe finalize_case_sample_v2.py --facts-input outputs_v2/bulk_200/neutral_facts_bilingual.jsonl --qc-input outputs_v2/bulk_200/neutral_fact_qc.csv --kr-target 100 --us-target 100 --seed 20260810 --output-dir outputs_v2 --overwrite
& .venv\Scripts\python.exe -m pytest -q tests/test_v2_pipeline.py
```

Existing outputs are never replaced unless `--overwrite` is explicit. Use
`--limit --dry-run` for collection plans and small deterministic diagnostics.

### Output relationships

Candidate JSONL files retain raw and main opinions; their QC CSV companions
contain auditable exclusions without duplicating long text. Family-link outputs
feed source extraction. Source fact units feed source-language neutral facts,
which feed aligned bilingual facts and QC. Only QC-passing records reach the
matched finalizer. Final case files preserve `full_opinion_text` and
`main_opinion_text`; final fact-pattern files are separate. The manifest holds
provenance, split, salience, QC, and SHA-256 fields. `collection_summary.json`
contains the actual funnel and every final invariant, including explicit false
values when a source or API shortfall prevents finalization.

Experiment runbooks are in `EXP1_GUIDE.md` and `EXP2_GUIDE.md`. Experiment 2
reuses the Experiment 1 pipeline and adds only an explicit, language-matched
jurisdiction instruction; its outputs are isolated under `outputs/exp2`.

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

# Paired Stage 2 QC

The consistent English-instruction generation profile is
`stage2-neutral-facts-35x35-v4`. Run it through `run_stage2_v4.py`; Korean
source masters remain Korean, while every generation prompt uses English
instructions. The v4 manifests pin profile `english-v1`, schema
`stage2-v3.1`, and policy `canonical-neutralization-en-v1`.

```powershell
python run_stage2_v4.py --case-id-file configs\stage2_calibration_a_6.txt --batch-name stage-a-english --dry-run
python run_stage2_v4.py --case-id-file configs\stage2_calibration_a_6.txt --batch-name stage-a-english --resume
```

The paired QC entrypoint is `qc_neutral_fact_pairs.py`. It validates one
consistent Stage 2 generation version, runs source-neutral QC, stops for human
source review, then runs translation QC only after the validated master is
imported.

```powershell
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --dry-run
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --source-qc-only
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --import-human-source-review outputs\neutral\stage2-paired-qc-v1\human_source_review.csv
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --regenerate-stale-translations --resume
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --translation-qc-only
python qc_neutral_fact_pairs.py --generation-output-dir outputs\neutral\stage2-neutral-35x35-v4 --batch-name stage-a --import-human-translation-review outputs\neutral\stage2-paired-qc-v1\human_translation_review.csv
```

For the current six-case Stage A, all six source masters were accepted with
human edits. Translations regenerated from those exact validated masters are
stored separately in
`outputs/neutral/stage2-paired-qc-v1/regenerated_translations.jsonl`; the v4
generation artifacts are not overwritten. Automatic translation QC currently
reports six failures and has stopped at
`outputs/neutral/stage2-paired-qc-v1/human_translation_review.csv`.

See `PAIRED_QC_IMPLEMENTATION_REPORT.md` for verified paths, gate results,
workflow, and canonical prompt locations.
