# Pilot Feature Cliff Delta Table

Direction: positive Cliff_delta means KR values tend to be larger than US values. CI: percentile bootstrap, 10,000 resamples, seed=20260714.

| feature | n_KR | n_US | KR_mean | US_mean | Cliff_delta | 95% CI | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| precedent_citation_count | 50 | 50 | 0.2000 | 30.5000 | -0.9696 | [-0.9992, -0.9180] | large |
| precedent_per_1k | 50 | 50 | 0.1518 | 5.9929 | -0.9500 | [-0.9992, -0.8828] | large |
| doctrine_per_1k | 50 | 50 | 11.8045 | 0.7745 | 0.9496 | [0.8944, 0.9872] | large |
| citation_count | 50 | 50 | 1.4800 | 33.0600 | -0.9060 | [-0.9772, -0.8108] | large |
| doc_length_sentences | 50 | 50 | 34.2600 | 174.1000 | -0.8804 | [-0.9708, -0.7692] | large |
| procedure_term_count | 50 | 50 | 2.9600 | 17.0400 | -0.8528 | [-0.9480, -0.7372] | large |
| doc_length_tokens | 50 | 50 | 1218.5400 | 4184.9600 | -0.8028 | [-0.9184, -0.6616] | large |
| citation_density | 50 | 50 | 1.5392 | 6.5367 | -0.7360 | [-0.8680, -0.5788] | large |
| remedy_per_1k | 50 | 50 | 5.7525 | 1.8411 | 0.6696 | [0.4872, 0.8284] | large |
| doctrine_term_count | 50 | 50 | 10.8600 | 3.5400 | 0.6332 | [0.4564, 0.7896] | large |
| party_arg_density | 50 | 50 | 4.8253 | 0.3869 | 0.6004 | [0.4296, 0.7624] | large |
| jurisdiction_mention_count | 50 | 50 | 1.7800 | 6.3800 | -0.4988 | [-0.6776, -0.3036] | large |
| procedure_per_1k | 50 | 50 | 3.2954 | 4.9229 | -0.3928 | [-0.5984, -0.1712] | medium |
| conclusion_position | 50 | 50 | 0.4791 | 0.1522 | 0.3396 | [0.1164, 0.5592] | medium |
| avg_sentence_length | 50 | 50 | 38.5828 | 24.5944 | 0.3116 | [0.0840, 0.5248] | small |
| remedy_term_count | 50 | 50 | 5.1800 | 6.8600 | 0.2328 | [0.0032, 0.4532] | small |
| jurisdiction_per_1k | 50 | 50 | 1.9246 | 1.4825 | -0.2280 | [-0.4512, 0.0024] | small |
| statute_per_1k | 50 | 50 | 1.3874 | 0.5438 | 0.1968 | [0.0024, 0.3864] | small |
| statute_ref_count | 50 | 50 | 1.2800 | 2.5600 | 0.0560 | [-0.1456, 0.2588] | negligible |
