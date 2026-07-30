# Experiment 1 results

## Run

- 70 cases, 420 responses (KO/EN × 3 replicates), 70 case-level pairs
- Generation: 420 successful responses; 420 recorded API calls; 0 transport retries
- Evaluation: 420 valid evaluations from 496 API response attempts; 76 schema-invalid intermediate attempts

## Conclusion stability

- Party-level agreement: 0.579
- Party-level change: 0.421
- Case-level any change: 0.900
- Direct likely/unlikely flip: 0.029
- Unweighted Cohen's kappa (nominal categories): 0.424
- Case-level change by origin: KR 0.886; CA 0.914

## Legal-system signals

- KR-oriented marker prevalence: KO 0.995, EN 0.129; paired difference 0.867, bootstrap 95% CI [0.810, 0.919], McNemar p = 5.42e-20
- US/common-law marker prevalence: KO 0.081, EN 0.971; paired difference -0.890, bootstrap 95% CI [-0.938, -0.833], McNemar p = 5.42e-20
- Strong A marker prevalence: KO 0.852, EN 0.162; paired difference 0.690
- Explicit-jurisdiction prevalence: KO 0.895, EN 0.186
- Statute-reference prevalence: KO 0.824, EN 0.014
- Hallucinated authority detected: KO 0.000, EN 0.000

## Reasoning composition

Significant paired proportion differences (positive means more weight in KO):

- `conclusion`: KO − EN = 0.074, BH q = 0.0008499
- `fault_or_intent`: KO − EN = 0.028, BH q = 0.0008499
- `damages_scope`: KO − EN = 0.028, BH q = 0.00306
- `plaintiff_fault_or_defense`: KO − EN = -0.014, BH q = 0.00425
- `multiple_tortfeasors`: KO − EN = 0.013, BH q = 0.002975
- `procedural_reasoning`: KO − EN = -0.013, BH q = 0.0017

- Mean output length: KO 8265 chars; EN 18186 chars
- Mean cross-language JS divergence: 0.093
- Mean within-language replicate JS divergence: 0.075
- Cross-minus-within JS difference: 0.018, bootstrap 95% CI [0.014, 0.022], permutation p = 9.999e-05
- Matched-replicate cross-language conclusion discordance: 0.424
- Within-language conclusion instability: 0.379

## Origin interaction

- KR-marker language effect (KO − EN): KR-origin 0.819; CA-origin 0.914; interaction contrast -0.095, bootstrap 95% CI [-0.200, 0.010]
- US-marker language effect (KO − EN): KR-origin -0.914; CA-origin -0.867; interaction contrast -0.048, bootstrap 95% CI [-0.152, 0.057]
- Master − translated contrast: KR marker -0.048; US marker -0.024; output length -172 chars

## Interpretation and limitations

The language conditions show large legal-system-marker shifts, but conclusion coding is less stable and requires human review. Cross-language reasoning divergence was lower than within-language replicate divergence on average, so reasoning-composition differences should not be overstated. Input language and translation status are not independent: translation status is the language-by-origin interaction in this 2×2 design. Automated evaluator coding, exact placeholder party matching, unequal response lengths, and multiple model samples are additional limitations.
