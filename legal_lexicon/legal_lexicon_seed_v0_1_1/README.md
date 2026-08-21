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


## v0.1.1 sanity fixes

These changes were made **before development-corpus tuning** and therefore do not use
confirmatory data:

1. English literal matching now uses whole-token boundaries.
   - Prevents `affirm` from matching `affirmative`.
   - Prevents `answer` from matching `answered`.
2. Same-span duplicate surfaces are collapsed deterministically.
   - If the same functional expression exists in KR and U.S. seed rows,
     the hit is labeled `SHARED::<concept_family>` and excluded from orientation scoring.
3. `orientation_include` is now conservative and separate from doctrine-density eligibility.
   - Generic terms such as `negligence`, `duty of care`, `reasonable care`,
     `foreseeability`, generic product-liability terms, and English KLRI translations
     are not treated as jurisdiction markers solely because they appear in one source.
4. Party-argument regexes are sentence-local and include:
   - singular/plural party roles,
   - Korean and English role terms,
   - anonymized placeholders such as `[PERSON_A]` and `[COMPANY_B]`.
5. Several over-strong source-provenance labels were downgraded to
   `source_variant` or `corpus_candidate`.
6. All edited records use version `0.1.1`.

This remains a **development-stage seed**. The next step is to run the matcher only on
the held-out development set, inspect exact hit spans and no-hit documents, revise the
lexicon using development evidence, and then freeze v1.0 before confirmatory analysis.
