# comparative-law-llm

## Korean 1st/2nd Instance Collection

Regenerate the Korean sample with trial/appellate-oriented filtering:

```powershell
.\.venv\Scripts\python.exe collect_kr_cases.py --sample-size 50 --output outputs\kr_cases.csv
.\.venv\Scripts\python.exe preprocess_cases.py
```

The collector uses `court`/`case_number` style metadata when available. For
`lbox/lbox_open::precedent_corpus`, which only exposes `id` and `precedent`, it
falls back to target civil case-number codes (`가합`, `가단`, `가소`, `나`) and
text markers that exclude Supreme Court-style decisions.
