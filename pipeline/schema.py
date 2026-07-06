from __future__ import annotations

CASE_TABLE_COLUMNS = [
    "case_id",
    "case_origin",
    "jurisdiction",
    "source_dataset",
    "source_id",
    "title",
    "date",
    "court",
    "trial_level",
    "raw_text",
    "text_length_chars",
    "collection_notes",
    "quality_flags",
]

FACT_PATTERN_FIELDS = [
    "case_id",
    "case_origin",
    "jurisdiction",
    "source_title",
    "raw_text_excerpt",
    "neutral_fact_ko",
    "neutral_fact_en",
    "neutralization_method",
    "removed_legal_signals",
    "quality_flags",
    "qc",
]

QUALITY_FLAGS = {
    "too_short",
    "too_long",
    "no_fact_section_detected",
    "legal_conclusion_may_remain",
    "jurisdiction_signal_may_remain",
    "legal_term_leakage",
    "translation_needed",
    "manual_review_recommended",
    "ambiguous_us_state",
    "extraction_failed",
}
