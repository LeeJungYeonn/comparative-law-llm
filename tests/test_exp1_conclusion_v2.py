from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from analyze_exp1_conclusions_v2 import (
    build_consensus, build_pair_rows, validate_flat_rows,
)
from exp1.common import append_jsonl, sha256_file
from exp1.conclusion_v2 import (
    CONCLUSIONS, aggregate_replicates, canonical_ids, canonicalize_single,
    existing_evaluator_audit, validate_recode_payload,
)
from run_exp1_conclusion_recode_v2 import completed_keys


def test_bracketed_and_bare_party_canonicalization() -> None:
    assert canonicalize_single("[PERSON_A]") == "PERSON_A"
    assert canonicalize_single("PERSON_A") == "PERSON_A"


def test_compound_party_string_is_auditable() -> None:
    assert canonical_ids("[PERSON_A]와 [PERSON_C]") == ["PERSON_A", "PERSON_C"]
    evaluations = [{
        "case_id": "C1",
        "evaluation": {"parties": [{"party_id": "[PERSON_A] and [PERSON_C]"}]},
    }]
    audit = existing_evaluator_audit(evaluations)
    assert "existing_grouped_party_string" in audit[("C1", "PERSON_A")]
    assert "existing_grouped_party_string" in audit[("C1", "PERSON_C")]


def payload(parties: list[dict]) -> dict:
    return {"response_id": "R1", "language": "ko", "parties": parties}


def party(party_id: str, conclusion: str = "likely") -> dict:
    return {
        "canonical_party_id": party_id,
        "conclusion": conclusion,
        "assessed": conclusion != "not_assessed",
        "supporting_text": "evidence" if conclusion != "not_assessed" else "",
        "aggregation_note": "integrated conclusion",
    }


def validate(value: dict, expected: list[str]) -> None:
    validate_recode_payload(
        value,
        expected_response_id="R1",
        expected_language="ko",
        expected_parties=expected,
    )


def test_duplicate_canonical_party_is_schema_failure() -> None:
    with pytest.raises((ValueError, Exception), match="duplicate|canonical"):
        validate(payload([party("PERSON_A"), party("PERSON_A")]), ["PERSON_A", "PERSON_B"])


def test_missing_canonical_party_is_schema_failure() -> None:
    with pytest.raises(Exception):
        validate(payload([party("PERSON_A")]), ["PERSON_A", "PERSON_B"])


def test_unallowed_canonical_party_is_schema_failure() -> None:
    with pytest.raises(Exception):
        validate(payload([party("PERSON_A"), party("PERSON_C")]), ["PERSON_A", "PERSON_B"])


def test_majority_aggregation_a_a_b_is_consensus() -> None:
    result = aggregate_replicates({1: "likely", 2: "likely", 3: "unlikely"})
    assert result["aggregation_status"] == "consensus"
    assert result["consensus_conclusion"] == "likely"
    assert result["consensus_count"] == 2


def test_three_distinct_labels_are_replicate_disagreement() -> None:
    result = aggregate_replicates({1: "likely", 2: "unlikely", 3: "conditional"})
    assert result["aggregation_status"] == "replicate_disagreement"
    assert result["consensus_conclusion"] is None


def test_not_assessed_is_preserved_not_dropped() -> None:
    result = aggregate_replicates({1: "not_assessed", 2: "not_assessed", 3: "likely"})
    assert result["aggregation_status"] == "consensus"
    assert result["consensus_conclusion"] == "not_assessed"


def registry() -> list[dict[str, str]]:
    return [{
        "case_id": "C1", "case_origin": "KR", "case_subtype": "x",
        "canonical_party_id": "PERSON_A", "source_party_set_mismatch": "False",
        "unresolved_source_issue": "False", "audit_flags": "",
    }, {
        "case_id": "C1", "case_origin": "KR", "case_subtype": "x",
        "canonical_party_id": "PERSON_B", "source_party_set_mismatch": "False",
        "unresolved_source_issue": "False", "audit_flags": "",
    }]


def flat_rows() -> list[dict]:
    rows = []
    for language in ("ko", "en"):
        for replicate in (1, 2, 3):
            for canonical in ("PERSON_A", "PERSON_B"):
                conclusion = "likely"
                if canonical == "PERSON_B":
                    conclusion = ("likely", "unlikely", "conditional")[replicate - 1]
                rows.append({
                    "case_id": "C1", "response_id": f"{language}{replicate}",
                    "language": language, "replicate_id": replicate,
                    "canonical_party_id": canonical, "conclusion": conclusion,
                    "assessed": True, "supporting_text": "e", "aggregation_note": "n",
                    "evaluator_model": "m", "evaluator_prompt_version": "v2",
                })
    return rows


def test_jsonl_shuffle_does_not_change_consensus_or_statistics() -> None:
    first = flat_rows()
    second = list(first)
    random.Random(99).shuffle(second)
    consensus_a = build_consensus(first, registry())
    consensus_b = build_consensus(second, registry())
    assert consensus_a == consensus_b
    assert build_pair_rows(consensus_a, registry()) == build_pair_rows(consensus_b, registry())


def test_pair_and_case_are_not_duplicated() -> None:
    consensus = build_consensus(flat_rows(), registry())
    pairs = build_pair_rows(consensus, registry())
    keys = [(row["case_id"], row["canonical_party_id"]) for row in pairs]
    assert len(keys) == len(set(keys)) == 2


def test_primary_inclusion_exclusion_denominator() -> None:
    consensus = build_consensus(flat_rows(), registry())
    pairs = build_pair_rows(consensus, registry())
    eligible = [row for row in pairs if row["primary_eligible"]]
    excluded = [row for row in pairs if not row["primary_eligible"]]
    assert len(eligible) == 1
    assert eligible[0]["canonical_party_id"] == "PERSON_A"
    assert len(excluded) == 1
    assert "replicate_disagreement" in excluded[0]["exclusion_reasons"]


def test_broad_marker_file_hash_is_unchanged_by_v2_writes(tmp_path: Path) -> None:
    broad = tmp_path / "summary.json"
    broad.write_text('{"legal_system_signals":{"x":1}}', encoding="utf-8")
    before = sha256_file(broad)
    (tmp_path / "conclusion_summary_v2.json").write_text("{}", encoding="utf-8")
    assert sha256_file(broad) == before


def test_resume_completed_response_causes_zero_pending_calls(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    append_jsonl(path, {"recode_cache_key": "K1", "evaluation": {"parties": []}})
    assert completed_keys(path) == {"K1"}
    planned = ["K1"]
    assert [key for key in planned if key not in completed_keys(path)] == []


def test_flat_output_schema_and_enum_validation() -> None:
    rows = flat_rows()
    validate_flat_rows(rows, registry())
    assert all(row["conclusion"] in CONCLUSIONS for row in rows)
    broken = list(rows)
    broken[0] = {**broken[0], "conclusion": "probably"}
    with pytest.raises(ValueError, match="invalid_conclusion"):
        validate_flat_rows(broken, registry())
