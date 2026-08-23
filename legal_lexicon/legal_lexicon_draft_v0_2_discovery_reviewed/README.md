# Legal Lexicon Draft v0.2 — discovery-reviewed

**Status: DISCOVERY-REVIEWED DRAFT, BEFORE TUNING**

This package is rebuilt from:
1. external official-resource seed `v0.1.1`; and
2. the nested development **discovery 20 only**.

Historical all-40-development v0.9 artifacts are not inputs.

## Separation
- discovery: 20 = KR 10 + US 10 — used for candidate generation/review
- tuning: 10 = KR 5 + US 5 — IDs exported, text NOT used in this build
- validation: 10 = KR 5 + US 5 — locked, text NOT used in this build
- confirmatory: 160 — untouched

## Discovery review
Input candidates: 8261
Accepted as primary/sensitivity evidence: 82
Rejected: 8179

The filled `human_*` fields are **model-assisted legal review**. They are not a claim that a human
researcher has reviewed every row. Use `researcher_signoff_discovery_review.csv` for final researcher approval.

## Key discovery-driven changes
- Added Korean-system concepts for `주의의무`, `과실`, and generic `인과관계`, instead of treating
  Korean surface forms as translations of U.S. concepts.
- Added Korean appellate procedure terms such as `상고인`, `피상고인`, `원심판결`, appellate
  error grounds, and final-appeal dismissal language.
- Promoted discovery-confirmed U.S. procedure seed candidates such as `standard of review`,
  `summary judgment`, `abuse of discretion`, `motion for summary judgment`, and `de novo review`.
- Added U.S. doctrine/remedy phrases supported by discovery contexts, including `standard of care`,
  `duty to warn`, `malpractice`, `negligent supervision`, `intentional infliction of emotional distress`,
  `reckless conduct`, and `loss of consortium`.
- Added U.S. procedure phrases including `trial court`, `motion in limine`, `jury verdict`,
  `judgment as a matter of law`, `directed verdict`, `burden of proof`, and
  `preponderance of the evidence`.
- Demoted bare `emotional distress` from primary remedy matching because discovery contexts also use
  the phrase inside the IIED cause of action.
- Added draft named-party argument-verb regexes. These MUST be tuned for precision on tuning 10.

## Next
Researcher sign-off -> tuning 10 matcher audit -> tuning-only revisions -> pre-validation hash ->
one-shot validation 10 -> NO modification -> freeze v1.0 -> confirmatory 160.
