# v4.0.1 metadata repair and freeze report

Status: **FROZEN**

- Case roster changes: 0
- `substantive_civil_liability_central` synchronized to `true`: 200/200
- Prior values preserved in `substantive_civil_liability_central_pre_v4`: 200/200
- Prior-value distribution: {'false': 23, 'true': 177}
- User manual neutral-fact edits preserved without text changes: 2/2
- Neutral-fact hashes refreshed: 2 (KR_80cd6399334ab774b2, KR_cd0b887a7a431369b9)
- Canonical final QC: 200/200 pass
- KR domains: {'general_negligence_personal_injury': 42, 'medical_professional_liability': 16, 'other_civil_liability': 33, 'product_liability': 9}
- US domains: {'general_negligence_personal_injury': 42, 'medical_professional_liability': 16, 'other_civil_liability': 33, 'product_liability': 9}
- US states: {'Louisiana': 16, 'Michigan': 19, 'Nevada': 18, 'Pennsylvania': 20, 'West Virginia': 27}

## Final invariants

- total_200: **TRUE**
- kr_100: **TRUE**
- us_100: **TRUE**
- case_roster_unchanged: **TRUE**
- case_id_and_case_family_id_unchanged: **TRUE**
- all_non_target_case_metadata_unchanged: **TRUE**
- all_substantive_civil_liability_central_true: **TRUE**
- all_pre_v4_values_preserved: **TRUE**
- all_kr_eligible_supreme_court_merits: **TRUE**
- all_us_eligible_state_highest_court_merits: **TRUE**
- all_dates_2000_2025: **TRUE**
- no_substantive_duplicate_case_families: **TRUE**
- all_us_controlling_opinions_validated: **TRUE**
- kr_us_primary_domain_counts_equal: **TRUE**
- all_five_us_states_represented: **TRUE**
- each_us_state_between_10_and_30: **TRUE**
- all_neutral_fact_text_preserved: **TRUE**
- all_fact_unit_text_preserved: **TRUE**
- all_neutral_facts_qc_pass: **TRUE**
- all_ko_en_pairs_equivalent: **TRUE**
- all_neutral_fact_source_synchronized: **TRUE**
- all_neutral_fact_hashes_current: **TRUE**
- no_unresolved_qc_flags: **TRUE**
- kr_development_20: **TRUE**
- kr_confirmatory_80: **TRUE**
- us_development_20: **TRUE**
- us_confirmatory_80: **TRUE**

## Core artifact SHA-256

- `outputs_v2/v4.0.1/final_cases_200_v4_0_1.jsonl`: `c9b6cffa2b1c75b67c2d7e031a604a26b49c57c317a61a649731f27127d1ec23`
- `outputs_v2/v4.0.1/final_fact_patterns_200_v4_0_1.jsonl`: `5e50e1c067abd9c7167f0bd36896523c6a5264ece5b2ee06d8a46a22d3814a8a`
- `outputs_v2/v4.0.1/final_fact_units_200_v4_0_1.jsonl`: `65bc8417b0c8b174cce0a1a1aee01edca2be0cfceff9296898e63f038521c902`
- `outputs_v2/v4.0.1/canonical_final_qc_200_v4_0_1.jsonl`: `e22bf7366553291b3fbb39b17395b5c1eb9313bb1cf4eca6b6d01f86e9a6b270`
- `outputs_v2/v4.0.1/us_raw_sources_100_v4_0_1.jsonl`: `ca9545cd77bc9312e9ffa4398c79efffa0e06950392b47825bf605cbba12ee46`
- `outputs_v2/v4.0.1/us_controlling_opinions_100_v4_0_1.jsonl`: `672280b7c65217a0e642fed187c603c64bf37cdcc85057dd8ba9649243d01e30`
- `outputs_v2/v4.0.1/final_roster_manifest_200_v4_0_1.csv`: `b2d7569eba508ca6e04bd3da2c16f570254127095e6abab3074a19bc3d6b3d9e`
- `outputs_v2/v4.0.1/final_qc_summary_v4_0_1.json`: `58be1d6e1416a71cb63fbeb34c198cd533b8d4d973796dac055d4518f86a1965`
- `outputs_v2/v4.0.1/source_duplicate_validation_v4_0_1.json`: `c8f042630812a7d38dde3f9d52812091b2ae25d217c413f98aeceb003fd1a743`
- `outputs_v2/v4.0.1/metadata_sync_audit_v4_0_1.json`: `38de5ad56687d10d3b05a35d473a9257a168cab5c1afd663201fd7c4bc18ee8f`
