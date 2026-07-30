# Experiment 1: input-language effect

This experiment sends each usable neutral fact pattern independently in Korean
and English, preserves raw prose, evaluates one response at a time under a
strict JSON schema, and analyzes only paired KO–EN observations.

The immutable input is
`outputs/neutral/stage2-paired-qc-v1/accepted_pairs.jsonl`. The scripts only read
it. `--model` has no default: pass an exact model snapshot explicitly. API
credentials are read from `LETSUR_API_KEY`; `.env` is loaded by default and its
values are never written or printed. Use `--env-file` to select another file.

## 1. Preflight and dry run

```powershell
& .venv\Scripts\python.exe audit_exp1.py
& .venv\Scripts\python.exe run_exp1_generation.py --model EXACT_SNAPSHOT --dry-run --repetitions 3
```

The dry run creates the deterministic shuffled request plan, config, manifest,
prompt hashes, input hash, git/Python/package provenance, and makes zero calls.

## 2. Required smoke test

```powershell
& .venv\Scripts\python.exe run_exp1_generation.py --output-dir outputs\exp1_smoke --model EXACT_SNAPSHOT --temperature 1 --max-output-tokens 8000 --reasoning-effort low --concurrency 4 --smoke-test --repetitions 1 --resume
& .venv\Scripts\python.exe run_exp1_generation.py --output-dir outputs\exp1_smoke --model EXACT_SNAPSHOT --temperature 1 --max-output-tokens 8000 --reasoning-effort low --concurrency 4 --smoke-test --repetitions 1 --resume
& .venv\Scripts\python.exe check_exp1_smoke.py --raw outputs\exp1_smoke\raw_responses.jsonl --manifest outputs\exp1_smoke\run_manifest.json --model EXACT_SNAPSHOT
```

The first command plans 3 KR and 3 CA cases with overlapping subtypes where
possible (12 calls). The second must report 12 resume skips and zero calls.
Review `reports/exp1_smoke_test.md` before full generation.

## 3. Full generation

Use the separate canonical full-run output directory below after the smoke test
passes. This keeps smoke and full-run request-order provenance distinct.

```powershell
& .venv\Scripts\python.exe run_exp1_generation.py --input outputs\neutral\stage2-paired-qc-v1\accepted_pairs.jsonl --output-dir outputs\exp1 --model EXACT_SNAPSHOT --temperature 1 --max-output-tokens 8000 --reasoning-effort low --seed 20260730 --repetitions 3 --concurrency 4 --resume
```

`--limit N` limits cases. `--case-ids ID1,ID2` or a path containing one case ID
per line selects cases. Other supported controls include `--top-p`,
`--max-output-tokens`, `--reasoning-effort`, `--preflight-only`, and
`--base-url`.

## 4. Blind automatic evaluation and analysis

```powershell
& .venv\Scripts\python.exe evaluate_exp1.py --model EXACT_EVALUATOR_SNAPSHOT --temperature 1 --max-output-tokens 12000 --reasoning-effort low --concurrency 4 --resume
& .venv\Scripts\python.exe analyze_exp1.py
```

Evaluation receives one raw response only: it does not receive origin,
translation status, paired response, or hypothesis. Schema failures are retried
at most twice in addition to transport backoff. Analysis produces case-unique
pair metrics, bootstrap intervals, exact McNemar tests, paired permutation
tests with BH-FDR correction, origin strata/interactions, replicate stability,
five graphs, and a stratified human-validation sheet.

Run all tests with:

```powershell
& .venv\Scripts\python.exe -m pytest tests\test_exp1.py -q
```
