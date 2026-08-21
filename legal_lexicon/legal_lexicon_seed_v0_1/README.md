# Legal Lexicon Seed v0.1

**Development-stage seed; not frozen for confirmatory analysis.**

Files:
- `sources.csv`: source registry and URLs
- `concepts.csv`: concept-level schema
- `surface_forms.csv`: bilingual literal/regex forms
- `seed_lexicon_combined.csv`: joined view for manual review
- `review_template.csv`: human validation template
- `lexicon_matcher.py`: deterministic matcher

Methodological rule:
External legal resources supply candidate concepts. Only the held-out development corpus may be used
to add/remove surface forms, tune regexes, and reject false positives. Once v1.0 is frozen, no
confirmatory document may be used to change the lexicon.

Provenance:
- source_exact
- official_translation
- source_variant
- corpus_candidate
- researcher_translation_candidate
- corpus_pattern_seed

Bilingual caution:
`concept_family` denotes functional comparability, not doctrinal identity. For example,
`KR_ADEQUATE_CAUSATION` and `US_PROXIMATE_CAUSE` may share `LEGAL_CAUSATION` while remaining
separate concept IDs.

Overlap:
1. longest-match within a category,
2. remedy accepted first,
3. doctrine hits overlapping accepted remedy spans are excluded,
4. procedure and party_arg are independent dimensions,
5. exact start/end offsets are retained.

Seed size:
- concepts: 125
- surface forms/patterns: 296
- sources: 23

Before v1.0 freeze:
- validate all translation candidates on held-out development outputs,
- audit category-level precision,
- inspect no-hit documents for false negatives,
- remove or demote high-ambiguity bare terms,
- freeze exact files and SHA-256 hashes,
- only then run confirmatory reference/output analysis.

## Activation rule after matcher smoke test
Researcher translation candidates, corpus candidates, and unverified official-translation candidates are stored with `primary_include=0` and `orientation_include=0` until validated on the held-out development set. This prevents translated U.S. terms such as `duty of care` -> `주의의무` from being misread as U.S.-specific evidence in Korean judgments.
