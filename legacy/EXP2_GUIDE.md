# Experiment 2: explicit jurisdiction condition

Exp 2 uses the Exp 1 cases, paired KO/EN facts, model settings, sampling settings,
response format, repetitions, and evaluator unchanged. The shared generator adds
one language-matched jurisdiction sentence plus a blank line immediately before
the existing Exp 1 user prompt. Outputs default to `outputs/exp2`; the runner
rejects any Exp 2 output path inside `outputs/exp1`.

Local validation and a no-API smoke plan:

```powershell
& .venv\Scripts\python.exe run_exp2_generation.py --output-dir outputs\exp2_smoke_local --smoke-test --repetitions 1 --dry-run
& .venv\Scripts\python.exe validate_exp2.py --exp2-dir outputs\exp2_smoke_local
```

Canonical full generation (do not run until authorized):

```powershell
& .venv\Scripts\python.exe run_exp2_generation.py --resume
```

The canonical run automatically reads the exact model and generation settings
from `outputs/exp1/config.json` and rejects mismatches. Evaluation and comparison:

```powershell
& .venv\Scripts\python.exe evaluate_exp1.py --input outputs\exp2\raw_responses.jsonl --output-dir outputs\exp2 --model EXACT_EXP1_EVALUATOR_SNAPSHOT --resume
& .venv\Scripts\python.exe analyze_exp2.py
& .venv\Scripts\python.exe validate_exp2.py
```

`analyze_exp2.py` reports conclusion stability, instruction/jurisdiction
alignment, wrong-jurisdiction terms, remedy-category shift, and Jensen-Shannon
distance between reasoning-unit distributions.
