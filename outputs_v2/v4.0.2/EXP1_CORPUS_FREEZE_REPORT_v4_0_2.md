# Exp 1 corpus freeze v4.0.2

Status: **FROZEN**

Only two user-directed neutral-fact corrections were applied:

- `US_9dc742b4409b555931`: `The Fierles` / `피얼스 부부` was replaced by the existing `[PERSON_A]` and `[PERSON_B]` placeholders.
- `KR_6e91996913ed443be6`: `a Japanese company` / `일본 업체` was replaced by `[COMPANY_F]`.

Affected KO/EN/source hashes were recomputed and corresponding fact units were synchronized. Case metadata, domains, splits, all non-target facts, prompts, and source opinions were unchanged. Collection and corpus QC were not rerun; the frozen v4.0.1 canonical QC is referenced by hash and the two edits received deterministic scope/hash/synchronization checks.

## Verification

- total_200: **TRUE**
- case_roster_and_order_unchanged: **TRUE**
- case_metadata_unchanged_excluding_version: **TRUE**
- all_non_target_facts_unchanged_excluding_version: **TRUE**
- all_non_target_units_unchanged_excluding_version: **TRUE**
- only_two_target_facts_changed: **TRUE**
- domains_unchanged: **TRUE**
- splits_unchanged: **TRUE**
- all_neutral_fact_hashes_current: **TRUE**
- all_fact_units_synchronized: **TRUE**
- all_residual_target_strings_removed: **TRUE**
- kr_development_20: **TRUE**
- kr_confirmatory_80: **TRUE**
- us_development_20: **TRUE**
- us_confirmatory_80: **TRUE**
