# Legal Lexicon Draft v0.2.1 Clean

Status: **PRE-TUNING CLEANED DRAFT**

No tuning, validation, or confirmatory opinion text was used.

## Changes
- S0145 `3배 이내`: inactive for primary matching.
- S0146 `세 배`: inactive for primary matching.
- S0190 `3배 손해배상`: retained as requested, still inactive.
- Simultaneously-active exact duplicates reduced to one active representative.
- Translation/wrong-jurisdiction duplicate rows remain as inactive provenance.
- Added `lost income`, `loss of income`, `loss of earnings` under `LOST_EARNINGS`.
- `loss of earning capacity` and `lost earning capacity` use separate `EARNING_CAPACITY_LOSS`.
- Bare `loss of earning` is deliberately absent.
- Orientation remains disabled for these generic shared remedy surfaces.

## Counts
- concepts: 151
- surface/regex rows: 340
- primary-active rows: 225
- inactive rows: 115
- active exact duplicate groups: 0
- duplicate rows deactivated in cleanup: 7

Next: run this exact draft on tuning 10 only and record all changes in `tuning_revision_log.csv`.
