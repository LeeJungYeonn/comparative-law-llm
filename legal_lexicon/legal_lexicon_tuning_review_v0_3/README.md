# Legal Lexicon Tuning Review v0.3

**Status: TUNING-REVIEWED PRE-VALIDATION CANDIDATE — VALIDATION NOT ACCESSED**

The review used only the nested tuning 10 opinions (KR 5 + US 5).
No validation or confirmatory opinion text was used to decide any revision.

## Hit review
- input matcher hits: 1322
- TP: 1300
- FP: 22
- reviewed hit precision: 0.9834
- concept-mapping errors: 26

This is hit precision only. **No recall/F1 claim is made**, because the FN audit is curated rather than exhaustive gold-span annotation.

## Main precision fixes
- `answer`: restricted to pleading context.
- English party-argument bare verbs: require explicit named/role subject; court/amicus/pronoun-only matches are suppressed.
- `plaintiff's claim`: removed from party-argument noun regex while `argument/contention/position/defense` remain.
- English `transfer`: deactivated after non-procedural personnel-transfer FP.
- English `reversal`: remapped to U.S. appellate-disposition concept.
- English `mental anguish`: remapped to U.S. non-economic-damages concept.

## Main recall fixes
Canonical tuning FNs added/activated include:
- KR: product-defect context, `증명책임`, `공동소송`, `의료과실`, `가동연한`,
  `사실오인`, `자유심증주의`, `치료비`.
- US: `factual cause`, `reasonably careful`, `comparatively negligent`,
  `wrongful death`, `survival statute`, `res ipsa loquitur`, `professional negligence`,
  `expert affidavit`, `certified question`, `jury instruction`, `post-trial motion`,
  appellate error terms, `manifest error`, `JNOV`, `motion to dismiss`,
  `funeral expenses`, `pecuniary loss`, `loss of enjoyment of life`,
  `impairment of earning capacity`, `due process`, constitutional excessiveness,
  punitive reprehensibility/ratio guidepost, and multiple-tortfeasor liability.

## Methodological guard
The current files are a **pre-validation candidate**. Researcher sign-off is still pending.
Once approved, hash/freeze this exact candidate, run validation 10 once, and never revise from validation results.
