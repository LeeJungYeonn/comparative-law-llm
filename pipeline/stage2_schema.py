from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


DATASET_VERSION = "stage2-neutral-facts-35x35-v1"
SCHEMA_VERSION = "stage2-v2"
EPISTEMIC_STATUSES = {
    "undisputed", "court_found", "jury_found", "alleged_by_plaintiff",
    "alleged_by_defendant", "assumed_true_for_pleading", "disputed",
    "testimony", "unclear",
}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
FACT_TYPES = {
    "party_relation", "action", "event", "harm", "causation_relevant",
    "defense_relevant", "disputed_fact", "warning", "knowledge",
    "physical_environment", "economic_harm", "other",
}


@dataclass(frozen=True)
class Stage2CaseInput:
    case_id: str
    case_origin: Literal["KR", "CA"]
    source_language: Literal["ko", "en"]
    source_text: str
    source_text_field: str
    source_text_sha256: str
    case_subtype: str | None
    source_dataset: str | None
    source_record_id: str | None
    input_file_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EVIDENCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["case_id", "evidence_units"],
    "properties": {
        "case_id": {"type": "string"},
        "evidence_units": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "source_sentence_ids", "short_quote", "proposed_fact_type", "epistemic_status", "epistemic_status_confidence"],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "source_sentence_ids": {"type": "array", "items": {"type": "string"}},
                    "short_quote": {"type": ["string", "null"]},
                    "proposed_fact_type": {"type": "array", "items": {"type": "string", "enum": sorted(FACT_TYPES)}},
                    "epistemic_status": {"type": "string", "enum": sorted(EPISTEMIC_STATUSES)},
                    "epistemic_status_confidence": {"type": "string", "enum": sorted(CONFIDENCE_LEVELS)},
                },
            },
        },
    },
}

NEUTRAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["case_id", "master_neutral_text", "fact_units", "removed_legal_signals", "removed_jurisdiction_signals", "anonymization_warnings", "grounding_warnings", "insufficient_factual_detail"],
    "properties": {
        "case_id": {"type": "string"},
        "master_neutral_text": {"type": "string"},
        "fact_units": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["fact_id", "master_text", "epistemic_status", "epistemic_status_confidence", "fact_types", "source_evidence_ids"], "properties": {
            "fact_id": {"type": "string"}, "master_text": {"type": "string"},
            "epistemic_status": {"type": "string", "enum": sorted(EPISTEMIC_STATUSES)},
            "epistemic_status_confidence": {"type": "string", "enum": sorted(CONFIDENCE_LEVELS)},
            "fact_types": {"type": "array", "items": {"type": "string", "enum": sorted(FACT_TYPES)}},
            "source_evidence_ids": {"type": "array", "items": {"type": "string"}},
        }}},
        "removed_legal_signals": {"type": "array", "items": {"type": "string"}},
        "removed_jurisdiction_signals": {"type": "array", "items": {"type": "string"}},
        "anonymization_warnings": {"type": "array", "items": {"type": "string"}},
        "grounding_warnings": {"type": "array", "items": {"type": "string"}},
        "insufficient_factual_detail": {"type": "boolean"},
    },
}

TRANSLATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["case_id", "translated_neutral_text", "translated_fact_units"],
    "properties": {
        "case_id": {"type": "string"},
        "translated_neutral_text": {"type": "string"},
        "translated_fact_units": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["fact_id", "translated_text"], "properties": {"fact_id": {"type": "string"}, "translated_text": {"type": "string"}}}},
    },
}

GROUNDING_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["case_id", "grounded", "unsupported_fact_ids", "overstated_fact_ids", "missing_material_facts", "epistemic_status_errors", "legal_conclusion_leakage", "jurisdiction_leakage", "verifier_status", "verifier_notes"],
    "properties": {
        "case_id": {"type": "string"}, "grounded": {"type": "boolean"},
        "unsupported_fact_ids": {"type": "array", "items": {"type": "string"}},
        "overstated_fact_ids": {"type": "array", "items": {"type": "string"}},
        "missing_material_facts": {"type": "array", "items": {"type": "string"}},
        "epistemic_status_errors": {"type": "array", "items": {"type": "string"}},
        "legal_conclusion_leakage": {"type": "array", "items": {"type": "string"}},
        "jurisdiction_leakage": {"type": "array", "items": {"type": "string"}},
        "verifier_status": {"type": "string", "enum": ["pass", "warning", "fail"]},
        "verifier_notes": {"type": "array", "items": {"type": "string"}},
    },
}

TRANSLATION_VERIFIER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["case_id", "meaning_preserved", "missing_fact_ids", "added_information", "omitted_information", "changed_negation", "changed_temporal_relation", "changed_epistemic_status", "legal_term_reintroduction", "placeholder_errors", "translation_status", "translation_notes"],
    "properties": {
        "case_id": {"type": "string"}, "meaning_preserved": {"type": "boolean"},
        "missing_fact_ids": {"type": "array", "items": {"type": "string"}},
        "added_information": {"type": "array", "items": {"type": "string"}},
        "omitted_information": {"type": "array", "items": {"type": "string"}},
        "changed_negation": {"type": "array", "items": {"type": "string"}},
        "changed_temporal_relation": {"type": "array", "items": {"type": "string"}},
        "changed_epistemic_status": {"type": "array", "items": {"type": "string"}},
        "legal_term_reintroduction": {"type": "array", "items": {"type": "string"}},
        "placeholder_errors": {"type": "array", "items": {"type": "string"}},
        "translation_status": {"type": "string", "enum": ["pass", "warning", "fail"]},
        "translation_notes": {"type": "array", "items": {"type": "string"}},
    },
}
