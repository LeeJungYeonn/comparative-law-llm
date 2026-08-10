from __future__ import annotations

from pipeline.stage2_schema import Stage2CaseInput


DATASET_VERSION = "stage2-neutral-facts-35x35-v3"
SCHEMA_VERSION = "stage2-v3.1"

EPISTEMIC_STATUSES = {
    "undisputed_descriptive",
    "documented_record",
    "court_found_descriptive",
    "party_allegation",
    "opposing_party_allegation",
    "witness_testimony",
    "expert_opinion",
    "disputed_descriptive",
    "assumed_true_for_pleading",
    "court_found_legal_conclusion",
    "court_found_causation_conclusion",
    "court_found_fault_allocation",
    "court_found_damages_calculation",
    "court_found_evidentiary_evaluation",
    "procedural_result",
    "unclear",
}

EXCLUDED_FINAL_STATUSES = {
    "court_found_legal_conclusion",
    "court_found_causation_conclusion",
    "court_found_fault_allocation",
    "court_found_damages_calculation",
    "court_found_evidentiary_evaluation",
    "procedural_result",
}

ATTRIBUTED_STATUSES = {
    "party_allegation",
    "opposing_party_allegation",
    "witness_testimony",
    "expert_opinion",
    "disputed_descriptive",
    "assumed_true_for_pleading",
}

CONFIDENCE_LEVELS = {"high", "medium", "low"}
FACT_TYPES = {
    "party_relation",
    "action",
    "omission",
    "event",
    "harm",
    "causation_relevant_sequence",
    "defense_relevant",
    "disputed_fact",
    "warning",
    "knowledge",
    "physical_environment",
    "economic_harm",
    "testimony",
    "expert_opinion",
    "other",
}

RELATION_TYPES = {
    "spouse_of",
    "parent_of",
    "child_of",
    "employee_of",
    "employer_of",
    "owned_by",
    "possessed_by",
    "drove",
    "operated",
    "controlled",
    "maintained",
    "treated",
    "examined",
    "did_not_examine",
    "spoke_by_phone_with",
    "prescribed_to",
    "did_not_prescribe_to",
    "performed_surgery_on",
    "manufactured",
    "distributed",
    "wholesaled",
    "retailed",
    "sold",
    "issued_warranty_for",
    "designed",
    "warned",
    "failed_to_warn",
    "knew_location_of",
    "did_not_determine_location_of",
    "alleged",
    "testified",
    "expert_opined",
    "injured",
    "died_in",
    "located_at",
    "moved_toward",
    "moved_away_from",
    "preceded",
    "followed",
    "caused_physical_sequence",
}

ENTITY_TYPES = {
    "person",
    "company",
    "medical_institution",
    "public_agency",
    "educational_institution",
    "property",
    "product",
    "location",
    "other",
}


def _object(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


EVIDENCE_SCHEMA = _object(
    {
        "case_id": {"type": "string"},
        "evidence_units": {
            "type": "array",
            "items": _object(
                {
                    "provisional_evidence_id": {"type": "string"},
                    "source_sentence_ids": {"type": "array", "items": {"type": "string"}},
                    "optional_short_quote": {"type": ["string", "null"]},
                    "fact_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": sorted(FACT_TYPES)},
                    },
                    "epistemic_status": {
                        "type": "string",
                        "enum": sorted(EPISTEMIC_STATUSES),
                    },
                    "epistemic_status_confidence": {
                        "type": "string",
                        "enum": sorted(CONFIDENCE_LEVELS),
                    },
                    "actor_entity_mentions": {"type": "array", "items": {"type": "string"}},
                    "object_entity_mentions": {"type": "array", "items": {"type": "string"}},
                    "materiality": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                [
                    "provisional_evidence_id",
                    "source_sentence_ids",
                    "optional_short_quote",
                    "fact_types",
                    "epistemic_status",
                    "epistemic_status_confidence",
                    "actor_entity_mentions",
                    "object_entity_mentions",
                    "materiality",
                ],
            ),
        },
    },
    ["case_id", "evidence_units"],
)

ENTITY_RELATION_SCHEMA = _object(
    {
        "case_id": {"type": "string"},
        "entities": {
            "type": "array",
            "items": _object(
                {
                    "entity_id": {"type": "string"},
                    "placeholder": {"type": "string"},
                    "entity_type": {"type": "string", "enum": sorted(ENTITY_TYPES)},
                    "source_mentions": {"type": "array", "items": {"type": "string"}},
                    "source_sentence_ids": {"type": "array", "items": {"type": "string"}},
                    "roles": {"type": "array", "items": {"type": "string"}},
                },
                [
                    "entity_id",
                    "placeholder",
                    "entity_type",
                    "source_mentions",
                    "source_sentence_ids",
                    "roles",
                ],
            ),
        },
        "relations": {
            "type": "array",
            "items": _object(
                {
                    "relation_id": {"type": "string"},
                    "subject_entity_id": {"type": "string"},
                    "relation_type": {"type": "string", "enum": sorted(RELATION_TYPES)},
                    "object_entity_id": {"type": "string"},
                    "source_sentence_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": sorted(CONFIDENCE_LEVELS)},
                    "material_relation": {"type": "boolean"},
                },
                [
                    "relation_id",
                    "subject_entity_id",
                    "relation_type",
                    "object_entity_id",
                    "source_sentence_ids",
                    "confidence",
                    "material_relation",
                ],
            ),
        },
        "completeness_warnings": {"type": "array", "items": {"type": "string"}},
    },
    ["case_id", "entities", "relations", "completeness_warnings"],
)

REALIZED_RELATION_SCHEMA = _object(
    {
        "subject_placeholder": {"type": "string"},
        "relation_type": {"type": "string", "enum": sorted(RELATION_TYPES)},
        "object_placeholder": {"type": "string"},
    },
    ["subject_placeholder", "relation_type", "object_placeholder"],
)

FACT_UNIT_SCHEMA = _object(
    {
        "fact_id": {"type": "string"},
        "master_text": {"type": "string"},
        "epistemic_status": {"type": "string", "enum": sorted(EPISTEMIC_STATUSES)},
        "epistemic_status_confidence": {
            "type": "string",
            "enum": sorted(CONFIDENCE_LEVELS),
        },
        "fact_types": {"type": "array", "items": {"type": "string", "enum": sorted(FACT_TYPES)}},
        "source_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "relation_ids": {"type": "array", "items": {"type": "string"}},
        "realized_relations": {"type": "array", "items": REALIZED_RELATION_SCHEMA},
        "material_relation": {"type": "boolean"},
    },
    [
        "fact_id",
        "master_text",
        "epistemic_status",
        "epistemic_status_confidence",
        "fact_types",
        "source_evidence_ids",
        "relation_ids",
        "realized_relations",
        "material_relation",
    ],
)

NEUTRAL_SCHEMA = _object(
    {
        "case_id": {"type": "string"},
        "master_neutral_text": {"type": "string"},
        "fact_units": {"type": "array", "items": FACT_UNIT_SCHEMA},
        "removed_legal_signals": {"type": "array", "items": {"type": "string"}},
        "removed_jurisdiction_signals": {"type": "array", "items": {"type": "string"}},
        "anonymization_warnings": {"type": "array", "items": {"type": "string"}},
        "grounding_warnings": {"type": "array", "items": {"type": "string"}},
        "insufficient_factual_detail": {"type": "boolean"},
    },
    [
        "case_id",
        "master_neutral_text",
        "fact_units",
        "removed_legal_signals",
        "removed_jurisdiction_signals",
        "anonymization_warnings",
        "grounding_warnings",
        "insufficient_factual_detail",
    ],
)

GROUNDING_SCHEMA = _object(
    {
        "case_id": {"type": "string"},
        "grounded": {"type": "boolean"},
        "unsupported_fact_ids": {"type": "array", "items": {"type": "string"}},
        "overstated_fact_ids": {"type": "array", "items": {"type": "string"}},
        "missing_material_facts": {"type": "array", "items": {"type": "string"}},
        "epistemic_status_errors": {"type": "array", "items": {"type": "string"}},
        "entity_role_errors": {"type": "array", "items": {"type": "string"}},
        "role_relation_errors": {"type": "array", "items": {"type": "string"}},
        "subject_object_errors": {"type": "array", "items": {"type": "string"}},
        "ownership_employment_errors": {"type": "array", "items": {"type": "string"}},
        "medical_provider_role_errors": {"type": "array", "items": {"type": "string"}},
        "product_chain_role_errors": {"type": "array", "items": {"type": "string"}},
        "material_relation_errors": {"type": "array", "items": {"type": "string"}},
        "legal_conclusion_leakage": {"type": "array", "items": {"type": "string"}},
        "fault_allocation_leakage": {"type": "array", "items": {"type": "string"}},
        "causation_conclusion_leakage": {"type": "array", "items": {"type": "string"}},
        "evidentiary_evaluation_leakage": {"type": "array", "items": {"type": "string"}},
        "damages_calculation_leakage": {"type": "array", "items": {"type": "string"}},
        "jurisdiction_leakage": {"type": "array", "items": {"type": "string"}},
        "verifier_status": {"type": "string", "enum": ["pass", "warning", "fail"]},
        "verifier_notes": {"type": "array", "items": {"type": "string"}},
    },
    [
        "case_id",
        "grounded",
        "unsupported_fact_ids",
        "overstated_fact_ids",
        "missing_material_facts",
        "epistemic_status_errors",
        "entity_role_errors",
        "role_relation_errors",
        "subject_object_errors",
        "ownership_employment_errors",
        "medical_provider_role_errors",
        "product_chain_role_errors",
        "material_relation_errors",
        "legal_conclusion_leakage",
        "fault_allocation_leakage",
        "causation_conclusion_leakage",
        "evidentiary_evaluation_leakage",
        "damages_calculation_leakage",
        "jurisdiction_leakage",
        "verifier_status",
        "verifier_notes",
    ],
)

TRANSLATION_SCHEMA = _object(
    {
        "case_id": {"type": "string"},
        "translated_neutral_text": {"type": "string"},
        "translated_fact_units": {
            "type": "array",
            "items": _object(
                {
                    "fact_id": {"type": "string"},
                    "translated_text": {"type": "string"},
                    "relation_ids": {"type": "array", "items": {"type": "string"}},
                    "realized_relations": {"type": "array", "items": REALIZED_RELATION_SCHEMA},
                },
                ["fact_id", "translated_text", "relation_ids", "realized_relations"],
            ),
        },
    },
    ["case_id", "translated_neutral_text", "translated_fact_units"],
)

TRANSLATION_VERIFIER_SCHEMA = _object(
    {
        "case_id": {"type": "string"},
        "meaning_preserved": {"type": "boolean"},
        "missing_fact_ids": {"type": "array", "items": {"type": "string"}},
        "added_information": {"type": "array", "items": {"type": "string"}},
        "omitted_information": {"type": "array", "items": {"type": "string"}},
        "changed_negation": {"type": "array", "items": {"type": "string"}},
        "changed_temporal_relation": {"type": "array", "items": {"type": "string"}},
        "changed_epistemic_status": {"type": "array", "items": {"type": "string"}},
        "changed_subject_object": {"type": "array", "items": {"type": "string"}},
        "changed_entity_role": {"type": "array", "items": {"type": "string"}},
        "changed_possession": {"type": "array", "items": {"type": "string"}},
        "changed_employment_or_ownership": {"type": "array", "items": {"type": "string"}},
        "changed_medical_provider_role": {"type": "array", "items": {"type": "string"}},
        "changed_product_chain_role": {"type": "array", "items": {"type": "string"}},
        "changed_directionality": {"type": "array", "items": {"type": "string"}},
        "changed_material_relation": {"type": "array", "items": {"type": "string"}},
        "legal_term_reintroduction": {"type": "array", "items": {"type": "string"}},
        "jurisdiction_reintroduction": {"type": "array", "items": {"type": "string"}},
        "jurisdiction_term_reintroduction": {"type": "array", "items": {"type": "string"}},
        "placeholder_errors": {"type": "array", "items": {"type": "string"}},
        "translation_status": {"type": "string", "enum": ["pass", "warning", "fail"]},
        "translation_notes": {"type": "array", "items": {"type": "string"}},
    },
    [
        "case_id",
        "meaning_preserved",
        "missing_fact_ids",
        "added_information",
        "omitted_information",
        "changed_negation",
        "changed_temporal_relation",
        "changed_epistemic_status",
        "changed_subject_object",
        "changed_entity_role",
        "changed_possession",
        "changed_employment_or_ownership",
        "changed_medical_provider_role",
        "changed_product_chain_role",
        "changed_directionality",
        "changed_material_relation",
        "legal_term_reintroduction",
        "jurisdiction_reintroduction",
        "jurisdiction_term_reintroduction",
        "placeholder_errors",
        "translation_status",
        "translation_notes",
    ],
)
