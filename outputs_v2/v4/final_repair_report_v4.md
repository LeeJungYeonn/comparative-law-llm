# v4 corpus repair and freeze report

Status: **FROZEN**

- Replaced cases: 30
- Retained neutral facts corrected: 36
- Replacement neutral facts freshly extracted: 30
- Duplicate case families removed: 2
- Noncontrolling U.S. source opinions replaced: 6
- Decision-date/source-header corrections: 2
- U.S. controlling opinions split before appended separate opinions: 39
- Canonical final QC: 200/200 pass
- Substantive duplicate families remaining: 0
- KR domains: {'general_negligence_personal_injury': 42, 'medical_professional_liability': 16, 'other_civil_liability': 33, 'product_liability': 9}
- US domains: {'general_negligence_personal_injury': 42, 'medical_professional_liability': 16, 'other_civil_liability': 33, 'product_liability': 9}
- US states: {'Louisiana': 16, 'Michigan': 19, 'Nevada': 18, 'Pennsylvania': 20, 'West Virginia': 27}
- Other-civil-liability subtypes: {'KR': {'employment_workplace': 9, 'financial_insurance_business': 12, 'other_economic_or_personal': 5, 'professional_services': 2, 'property_environmental_construction': 3, 'public_entity_institutional': 2}, 'US': {'defamation_privacy_reputation': 1, 'employment_workplace': 14, 'financial_insurance_business': 8, 'other_economic_or_personal': 2, 'professional_services': 5, 'property_environmental_construction': 1, 'public_entity_institutional': 2}}

## Freeze invariants

- total_200: **TRUE**
- kr_100: **TRUE**
- us_100: **TRUE**
- all_kr_eligible_supreme_court_merits: **TRUE**
- all_us_eligible_state_highest_court_merits: **TRUE**
- all_dates_2000_2025: **TRUE**
- no_substantive_duplicate_case_families: **TRUE**
- all_us_controlling_opinions_validated: **TRUE**
- kr_us_primary_domain_counts_equal: **TRUE**
- all_five_us_states_represented: **TRUE**
- each_us_state_between_10_and_30: **TRUE**
- all_neutral_facts_qc_pass: **TRUE**
- all_ko_en_pairs_equivalent: **TRUE**
- all_neutral_fact_source_synchronized: **TRUE**
- no_unresolved_qc_flags: **TRUE**
- kr_development_20: **TRUE**
- kr_confirmatory_80: **TRUE**
- us_development_20: **TRUE**
- us_confirmatory_80: **TRUE**

## Replacements

- `US_a07db3e0a97840c5b1` → `US_59362b11f24be41b92`: extraordinary-writ-only opinion
- `US_c387cc2f242da3c0dc` → `US_a15bd1ea6aa4ea1ed9`: administrative prison-policy appeal
- `US_b2c7f8c81dc1b0c361` → `US_5374560fdfdf536ed1`: public civil-enforcement retroactivity case
- `US_0852f7949bcaa2bb53` → `US_4a0f056bcec6698d7d`: jury-selection procedure was the high-court issue
- `US_1cfe591dc59be857da` → `US_3aa1784247c016e937`: leave-denial order; substance appeared only in separate opinions
- `US_8636c5a0c5ede55860` → `US_979762a92670a0552a`: leave-denial order; substance appeared only in dissent; the initially considered wet-floor candidate was also rejected because its controlling opinion addressed only lost-evidence procedure
- `US_9b0dcb923fdb190dd3` → `US_6fc55880fb74718547`: leave-denial order; substance appeared only in concurrence
- `US_8026310f8d6a3819bd` → `US_f57bc43fc6478fd8b2`: controlling decision was issued in 1999
- `US_5905b849c15a6fb3d6` → `US_384364b54638f78623`: duplicate Grove litigation/case family
- `US_9cac14d9a55b155cd4` → `US_b675276d7ca44f64e2`: duplicate Craig litigation/case family
- `US_fa07baffd06e422309` → `US_472f038dea6238a51e`: stored reference text is a concurring-and-dissenting opinion, not the majority
- `US_67cc0f02a85dfe7f69` → `US_a4ba9463b694840f34`: stored reference text is a concurrence/opinion in support of affirmance, not a controlling majority
- `US_e0f4e43491c639af90` → `US_0ffb071db74bea0121`: stored reference text is a dissent, not the controlling majority
- `US_231c5e7fb867d9b8c2` → `US_2999172a4de14d7523`: source-scope recheck: contractual indemnity and defense-cost allocation, not direct civil-liability merits
- `US_04cf5031ab373b2ef6` → `US_107f5beb1553bc775c`: source-scope recheck: multi-claim mortgage pleading appeal, not a clean substantive civil-liability merits case
- `US_d15af88bf67507bd93` → `US_b267b8668935378e25`: source-scope recheck: implied-indemnity defense costs after the underlying action
- `US_7f556627654c5d3f9a` → `US_97456da392d82990a3`: source-scope recheck: post-judgment compensation-fund payment procedure
- `US_74c861440b330e42dc` → `US_0bdc6cb112770284df`: source-scope recheck: the controlling opinion concerns discovery conduct and trial sanctions in a construction-contract dispute, not civil-liability merits
- `KR_4a417b3e4eb96c19c1` → `KR_c0c025922aca493f10`: HARD judicial-evaluation leakage; minimum product-to-general domain swap
- `KR_f3fe352a4c02d16f80` → `KR_d47698374deaa59285`: HARD fact insufficiency: controlling source omitted the accident conduct and causal sequence
- `KR_5beca7bd705b05bb94` → `KR_25ab15d21ac1967afd`: HARD fact insufficiency: controlling source omitted the patient outcome and surrounding treatment sequence
- `KR_adf561060b125d6b87` → `KR_e61f8571b55fd48e4c`: HARD legal-rule leakage and controlling source omitted the accident conduct needed for a sufficient neutral fact
- `KR_882ce243b36c889fda` → `KR_1346b7ea49f678b93c`: source-scope recheck: controlling opinion centered on a special-hiring clause rather than the civil-liability merits
- `KR_1e296dbb0bcebaab8f` → `KR_6a3e8f577c00b9c20c`: procedural-role wording; minimum other-to-medical domain swap
- `KR_c9b0b61cc030200a6a` → `KR_a1301398c0525889ea`: semantic redundancy; minimum additional other-to-general swap after rejecting a fee-only U.S. candidate
- `KR_c7ec53e12d00dabce6` → `KR_fd8f69f92c35adf0ac`: minimum other-to-general swap required after direct source review reclassified the Nevada replacement
- `KR_293763f5a50c79c279` → `KR_6078abc75c440541ab`: minimum other-to-general domain swap required by the U.S. source-scope replacements
- `KR_644078b169f572b0fe` → `KR_8f9a8d4ff8f0f379fb`: minimum other-to-general domain swap required by the U.S. source-scope replacements
- `KR_d35502ba94fbfc6c92` → `KR_de99291f16b6fe1b2a`: minimum other-to-general domain swap required by the U.S. source-scope replacements
- `KR_c2567a4fb9e71fc1f7` → `KR_46eecae742ccab2781`: minimum general-to-medical swap required after rejecting the procedure-only wet-floor U.S. replacement

## Core artifact SHA-256

- `outputs_v2/v4/final_cases_200_v4.jsonl`: `bdde98317f030bb92643fcb1856e0789e6c92408e409415ab37647b51dfcf2d5`
- `outputs_v2/v4/final_fact_patterns_200_v4.jsonl`: `d05796dfc24369c82a1af66d1398b71f9154ee0394f7716f9618dd36202c37a2`
- `outputs_v2/v4/final_fact_units_200_v4.jsonl`: `9e299c15c0ab6b2ee65edce9c6a296ad48f5b18c7ed4bdf76e313b4e78b64250`
- `outputs_v2/v4/canonical_final_qc_200_v4.jsonl`: `3bbac0caac90790075cd344bfc1b159eb914c27534c4ccd1b5e04ec105c6b515`
- `outputs_v2/v4/us_raw_sources_100_v4.jsonl`: `ca9545cd77bc9312e9ffa4398c79efffa0e06950392b47825bf605cbba12ee46`
- `outputs_v2/v4/us_controlling_opinions_100_v4.jsonl`: `672280b7c65217a0e642fed187c603c64bf37cdcc85057dd8ba9649243d01e30`
- `outputs_v2/v4/final_roster_manifest_200_v4.csv`: `b2d7569eba508ca6e04bd3da2c16f570254127095e6abab3074a19bc3d6b3d9e`
- `outputs_v2/v4/independent_qc_issue_resolution_v4.csv`: `d0554e25c842f5a7e2a03f0214f1e0737b3141d7df9284127d4922fb751a44dd`
- `outputs_v2/v4/final_qc_summary_v4.json`: `c4ab5ee05e4e0e8b0617fabdad97e2c22f8927fa8b94c9ba6adc8fbef36b1fa7`
- `outputs_v2/v4/source_duplicate_validation_v4.json`: `d77c3b83eeed4cc9c7a3fd7b983ab57a6c7016ce4b7462b49940a12a7c4865bd`
