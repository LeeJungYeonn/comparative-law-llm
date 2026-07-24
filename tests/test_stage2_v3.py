from __future__ import annotations

from collections import Counter

import pytest

from pipeline.stage2_v3_pipeline import (
    canonical_quantity_tokens,
    normalize_entity_relation_graph,
    source_checks,
    translation_checks,
    validate_grounding_payload,
)
from pipeline.stage2_v3_schema import (
    DATASET_VERSION,
    EPISTEMIC_STATUSES,
    EXCLUDED_FINAL_STATUSES,
    RELATION_TYPES,
)


def _graph() -> dict:
    return {
        "graph_status": "pass",
        "entities": [
            {"entity_id": "ENT001", "placeholder": "[PERSON_A]"},
            {"entity_id": "ENT002", "placeholder": "[PERSON_B]"},
        ],
        "relations": [{
            "relation_id": "R001",
            "subject_entity_id": "ENT001",
            "relation_type": "treated",
            "object_entity_id": "ENT002",
            "material_relation": True,
        }],
    }


def _evidence() -> dict:
    return {
        "source_coverage": {"coverage_status": "complete"},
        "evidence_units": [
            {"evidence_id": "E001", "epistemic_status": "documented_record"},
            {"evidence_id": "E002", "epistemic_status": "documented_record"},
        ],
    }


def _master_payload() -> dict:
    fact_units = [
        {
            "fact_id": "F001",
            "master_text": "[PERSON_A] treated [PERSON_B].",
            "epistemic_status": "documented_record",
            "fact_types": ["action"],
            "source_evidence_ids": ["E001"],
            "relation_ids": ["R001"],
            "realized_relations": [{
                "subject_placeholder": "[PERSON_A]",
                "relation_type": "treated",
                "object_placeholder": "[PERSON_B]",
            }],
        },
        {
            "fact_id": "F002",
            "master_text": "[PERSON_B] suffered a bodily injury.",
            "epistemic_status": "documented_record",
            "fact_types": ["harm"],
            "source_evidence_ids": ["E002"],
            "relation_ids": [],
            "realized_relations": [],
        },
    ]
    return {
        "master_neutral_text": " ".join(unit["master_text"] for unit in fact_units),
        "fact_units": fact_units,
    }


def test_v3_version_and_taxonomy_are_distinct() -> None:
    assert DATASET_VERSION == "stage2-neutral-facts-35x35-v3"
    assert "court_found" not in EPISTEMIC_STATUSES
    assert "court_found_descriptive" in EPISTEMIC_STATUSES
    assert "court_found_causation_conclusion" in EXCLUDED_FINAL_STATUSES
    assert {"treated", "spoke_by_phone_with", "manufactured", "distributed"} <= RELATION_TYPES


@pytest.mark.parametrize(
    ("ko", "en"),
    [
        ("3분의 1", "one-third"),
        ("8주", "8 weeks"),
        ("23세", "23 years old"),
        ("25%", "25 percent"),
        ("7.6 m였고", "7.6 m long"),
        ("시속 60킬로미터", "60 kilometers per hour"),
    ],
)
def test_v3_quantity_surface_equivalence(ko: str, en: str) -> None:
    assert canonical_quantity_tokens(ko) == canonical_quantity_tokens(en)


def test_v3_source_checks_require_material_relation_realization() -> None:
    checked = source_checks(_master_payload(), _evidence(), _graph(), "en")
    assert checked["status"] == "pass"
    broken = _master_payload()
    broken["fact_units"][0]["relation_ids"] = []
    broken["fact_units"][0]["realized_relations"] = []
    broken["master_neutral_text"] = " ".join(
        unit["master_text"] for unit in broken["fact_units"]
    )
    checked = source_checks(broken, _evidence(), _graph(), "en")
    assert checked["status"] == "fail"
    assert checked["missing_material_relation_ids"] == ["R001"]


def test_v3_graph_drops_nonreflexive_self_relations_and_renumbers() -> None:
    payload = {
        "case_id": "X",
        "entities": [
            {
                "entity_id": "a",
                "entity_type": "person",
                "source_sentence_ids": ["SRC0001"],
            },
            {
                "entity_id": "b",
                "entity_type": "person",
                "source_sentence_ids": ["SRC0001"],
            },
        ],
        "relations": [
            {
                "relation_id": "old1",
                "subject_entity_id": "a",
                "relation_type": "injured",
                "object_entity_id": "a",
                "source_sentence_ids": ["SRC0001"],
            },
            {
                "relation_id": "old2",
                "subject_entity_id": "a",
                "relation_type": "injured",
                "object_entity_id": "b",
                "source_sentence_ids": ["SRC0001"],
            },
        ],
        "completeness_warnings": [],
    }
    graph = normalize_entity_relation_graph(
        payload,
        case_id="X",
        case_origin="CA",
        evidence={"evidence_units": [{"source_sentence_ids": ["SRC0001"]}]},
    )
    assert [item["relation_id"] for item in graph["relations"]] == ["R001"]
    assert graph["relations"][0]["subject_entity_id"] != graph["relations"][0]["object_entity_id"]
    assert graph["graph_status"] == "warning"
    assert graph["completeness_warnings"] == [
        "dropped_nonreflexive_self_relation:R001"
    ]


@pytest.mark.parametrize(
    "master_text",
    [
        "[PERSON_A] was negligent and treated [PERSON_B].",
        "[PERSON_A] was liable and treated [PERSON_B].",
        "[PERSON_A] treated [PERSON_B] in California.",
    ],
)
def test_v3_source_checks_reject_legal_or_jurisdiction_leakage(
    master_text: str,
) -> None:
    payload = _master_payload()
    payload["fact_units"][0]["master_text"] = master_text
    payload["master_neutral_text"] = " ".join(
        unit["master_text"] for unit in payload["fact_units"]
    )
    assert source_checks(payload, _evidence(), _graph(), "en")["status"] == "fail"


def test_v3_korean_role_word_is_not_legal_conclusion() -> None:
    payload = _master_payload()
    payload["fact_units"][0]["master_text"] = (
        "[PERSON_A]는 치료 책임자로서 [PERSON_B]를 치료했다."
    )
    payload["master_neutral_text"] = " ".join(
        unit["master_text"] for unit in payload["fact_units"]
    )
    assert source_checks(payload, _evidence(), _graph(), "ko")["status"] == "pass"


def test_v3_grounding_recheck_excludes_required_legal_omissions() -> None:
    payload = {
        "grounded": True,
        "unsupported_fact_ids": [],
        "overstated_fact_ids": [],
        "missing_material_facts": ["법원의 과실 판단이 누락되었다."],
        "epistemic_status_errors": [],
        "entity_role_errors": [],
        "subject_object_errors": [],
        "material_relation_errors": [],
        "legal_conclusion_leakage": [],
        "jurisdiction_leakage": [],
        "verifier_status": "warning",
    }
    checked = validate_grounding_payload(
        payload,
        {"neutralization_status": "pass"},
    )
    assert checked["missing_material_facts"] == []
    assert checked["validated_verifier_status"] == "pass"
    assert checked["verifier_consistency_violation"] is False


def _translation(master: dict, texts: list[str]) -> dict:
    units = []
    for source, text in zip(master["fact_units"], texts):
        units.append({
            "fact_id": source["fact_id"],
            "translated_text": text,
            "relation_ids": source["relation_ids"],
            "realized_relations": [
                dict(relation) for relation in source["realized_relations"]
            ],
        })
    return {
        "translated_fact_units": units,
        "translated_neutral_text": " ".join(texts),
    }


def test_v3_translation_relation_direction_is_hard_gate() -> None:
    master = _master_payload()
    translated = _translation(
        master,
        [
            "[PERSON_A]가 [PERSON_B]를 치료했다.",
            "[PERSON_B]는 신체적 부상을 입었다.",
        ],
    )
    assert translation_checks(master, translated, "ko")["status"] != "fail"
    translated["translated_fact_units"][0]["realized_relations"][0][
        "subject_placeholder"
    ] = "[PERSON_B]"
    translated["translated_fact_units"][0]["realized_relations"][0][
        "object_placeholder"
    ] = "[PERSON_A]"
    assert translation_checks(master, translated, "ko")["status"] == "fail"


def test_v3_placeholder_occurrence_difference_is_not_hard_failure() -> None:
    master = _master_payload()
    translated = _translation(
        master,
        [
            "[PERSON_A]가 [PERSON_B]를 치료했고 [PERSON_A]는 떠났다.",
            "[PERSON_B]는 신체적 부상을 입었다.",
        ],
    )
    checked = translation_checks(master, translated, "ko")
    assert "placeholder_occurrence_count_differs" in checked["warnings"]
    assert "placeholder_identity_mismatch" not in checked["errors"]


def test_v3_quantity_counter_preserves_duplicate_values() -> None:
    assert canonical_quantity_tokens("8 weeks and 8 weeks") == Counter({"week:8": 2})
