# Neutral Fact Pattern Quality Review

## Scope

Reviewed the uploaded 200-case corpus by comparing neutral fact patterns against the corresponding original/highest-court opinion text in `final_cases_200.jsonl`, with additional bilingual and neutrality checks.

The review focused on:

1. factual distortion or unsupported transformation;
2. Korean/English translation mismatch;
3. duplicated factual units;
4. residual court/legal conclusions or procedural disposition;
5. jurisdiction-specific leakage in units, currency, institutions, and identifiers;
6. placeholder consistency across Korean and English;
7. clear source-case eligibility failures that cannot be repaired by editing the neutral fact text.

The original uploaded files were not overwritten.

## Main findings

### 1. Text-level corrections

High-confidence corrections were made in **73 / 200 cases**.

Major error types included:

- 19 Korean-origin records whose Korean master neutral fact was actually English or mixed English/Korean;
- exact duplicated sentences/factual units;
- court findings such as `원심은 ... 판단하였다`, jury verdicts, and explicit legal-causation/foreseeability conclusions left in neutral facts;
- litigation-role/procedural sentences that were not necessary factual background;
- legal-theory wording such as negligence/duty labels where the underlying factual allegation could be stated neutrally;
- jurisdiction-sensitive currency, U.S. customary units, Korean government/institution labels, resident-registration terminology, and `[BOROUGH_*]` placeholders in the core-neutralized text;
- isolated English grammar corruption and duplicated entity placeholders.

After correction, automated consistency checks found:

- 200 records present;
- no Korean-origin Korean field with the prior English-master failure;
- no KO/EN placeholder-set mismatch;
- no exact duplicated sentence;
- no remaining literal won/dollar amount in the cases targeted for normalization;
- no remaining targeted `miles/feet`, Korean government-title, resident-registration, or `[BOROUGH_*]` leakage patterns.

### 2. Cases that require source-level replacement

**18 U.S. cases** should not be treated as valid main-corpus civil-liability merits cases even after their neutral-fact text is cleaned.

These include attorney-disciplinary proceedings, workers' compensation proceedings, insurance-coverage/indemnity disputes, purely procedural or extraordinary-writ decisions, and one record with unreliable controlling-opinion selection.

A neutral-fact rewrite cannot fix this source-selection problem. They should be replaced from the eligible candidate pool and the KR/US domain-state allocation should then be recomputed.

Replacement-required cases:

- `US_f6a3b4ef121a4e3b04` — insurance coverage/pollution-exclusion dispute; underlying injury is not the merits issue
- `US_2b3ba71069d1e5eab9` — workers' compensation statutory proceeding rather than ordinary civil liability/damages
- `US_f360178db4c0409aa2` — medical-malpractice notice/statute-of-limitations procedural decision rather than malpractice merits
- `US_a072ff18b300a13dfd` — attorney disciplinary proceeding, not a civil-liability/damages merits case
- `US_39be177337afd8d1be` — attorney disciplinary proceeding, not a civil-liability/damages merits case
- `US_162eb334cc7d5a8b91` — attorney disciplinary proceeding, not a civil-liability/damages merits case
- `US_a83924c6ae07314b33` — attorney disciplinary proceeding, not a civil-liability/damages merits case
- `US_bf6cccdd9b8a33c4a5` — prescription/direct-action procedural issue after an automobile tort settlement, not underlying liability merits
- `US_b1a79acce1af757b58` — attorney disciplinary proceeding, not a civil-liability/damages merits case
- `US_e1da2d37a9dee8ec7b` — attorney disciplinary proceeding, not a civil-liability/damages merits case
- `US_18fd4288d7f32131ce` — attorney disciplinary proceeding, not a civil-liability/damages merits case
- `US_1bd83c629182d24617` — extraordinary-writ/settlement-allocation proceeding; substantive liability is not the issue decided
- `US_af3a72d6a4fbf2544a` — claim/issue-preclusion decision; underlying negligence was already adjudicated elsewhere
- `US_c2f1eaa3640f0a9bbf` — insurance-contract indemnity interpretation dispute rather than civil-liability merits
- `US_0bdc6cb112770284df` — action against insurer concerning statutory/coverage and bad-faith issues rather than underlying tort merits
- `US_ace4efc8607c66874a` — declaratory insurance-coverage dispute rather than civil-liability merits
- `US_4e8bd43839d97e3e1f` — writ of prohibition focused on personal jurisdiction in an underlying product case, not liability merits
- `US_cf1fe4f8bf2f7fa43b` — stored main opinion begins with a dissent in an extraordinary-writ case; controlling-opinion selection is unreliable

## Important domain-label warning

This review also found evidence that the existing `primary_domain` classifier should be re-audited before the generation experiment. For example, some Korean records labeled `medical_professional_liability` concern non-medical topics such as association governance, lease/key-money disputes, patents/technology royalties, or other non-medical civil disputes.

I did **not** silently rewrite all domain labels in this correction file, because doing so requires a separate full source-level domain reclassification and would change the matched 55/28/12/5 sampling allocation.

## Output files

### `final_fact_patterns_200_qc_corrected.jsonl`

Schema-preserving text-corrected version of all 200 records.

Use this file for inspecting the corrected neutral fact text, but do **not** yet treat it as the final experimental 200-case corpus because the source-level replacement cases remain present.

### `neutral_fact_qc_audit_200.jsonl`

One row per case with:

- text review status;
- correction issue types;
- retain / replacement-required status;
- source-level note for replacement cases.

### `final_fact_patterns_182_retainable_after_qc.jsonl`

Convenience subset excluding the 18 definite source-level replacement cases.

This is **not** a final balanced experimental corpus: removing cases changes U.S. state/domain counts. Use it only as a review/intermediate artifact until replacement cases are selected.

## Recommended next step

1. Replace the 18 source-ineligible U.S. cases from the pre-LLM eligible candidate pool.
2. Re-run the same source-level eligibility screen on replacement candidates.
3. Re-run domain classification for all 200 cases, not just replacements.
4. Generate/translate neutral facts only for replacements.
5. Run this QC again and freeze the final 100 KR + 100 U.S. manifest only after all case-level and text-level checks pass.
