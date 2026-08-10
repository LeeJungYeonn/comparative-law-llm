from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from analyze_exp1 import LABELS, main as analyze_main, reasoning_vector
from evaluate_exp1 import parse_json_content
from exp1.common import (
    FORBIDDEN_REQUEST_FIELDS, SCHEMA_PATH, append_jsonl, assert_request_is_blind,
    load_env_file, load_prompts, prompt_hashes, read_jsonl, render_generation, request_payload,
    sha256_text, unique_key,
)
from run_exp1_generation import build_plan, completed_keys


def case(case_id: str = "KR_1") -> dict:
    return {
        "case_id": case_id,
        "case_origin": "KR",
        "case_subtype": "traffic_accident",
        "neutral_fact_ko": "[원고]가 다쳤다.",
        "neutral_fact_en": "[Plaintiff] was injured.",
        "source_language": "ko",
        "master_language": "ko",
    }


def valid_evaluation() -> dict:
    return {
        "parties": [{
            "party_id": "[Defendant]", "conclusion": "conditional",
            "evidence_span": "may be liable", "confidence": 0.8,
        }],
        "jurisdiction_signals": {
            "explicit_jurisdiction": "NONE", "explicit_statute_reference": False,
            "explicit_precedent_reference": False, "civil_law_authority_marker": False,
            "common_law_authority_marker": False, "jurisdiction_specific_doctrine": False,
            "unsupported_or_hallucinated_authority": False, "evidence_spans": [],
        },
        "concepts": [{
            "concept_id": "general_causation", "present": True,
            "evidence_span": "caused", "confidence": 0.9, "marker_strength": "C",
        }],
        "reasoning_units": [
            {"unit_id": 1, "text_span": "A duty may exist.", "labels": ["governing_rule", "duty_or_protected_interest"]},
            {"unit_id": 2, "text_span": "Liability is uncertain.", "labels": ["conclusion"]},
        ],
        "damages": [{
            "damage_id": "medical_expenses", "present": True,
            "evidence_span": "medical expenses", "confidence": 0.9,
        }],
        "evaluator_confidence": 0.85,
        "evaluator_notes": "",
    }


def test_ko_en_requests_match_each_case_and_replicate() -> None:
    plan = build_plan([case("KR_1"), case("KR_2")], repetitions=3, model="snapshot", seed=7)
    keys = {(x["row"]["case_id"], x["replicate_id"]) for x in plan}
    for key in keys:
        conditions = [x["condition"] for x in plan if (x["row"]["case_id"], x["replicate_id"]) == key]
        seeds = {x["seed"] for x in plan if (x["row"]["case_id"], x["replicate_id"]) == key}
        assert sorted(conditions) == ["en", "ko"]
        assert len(seeds) == 1


def test_request_payload_contains_no_case_metadata() -> None:
    prompts = load_prompts()
    system, user = render_generation(case(), "ko", prompts)
    body = request_payload(
        model="snapshot", system_prompt=system, user_prompt=user, temperature=.2,
        top_p=1, max_output_tokens=1000, seed=5, reasoning_effort=None,
    )
    assert_request_is_blind(body)
    serialized = json.dumps(body["messages"], ensure_ascii=False)
    assert not any(f'"{field}"' in serialized for field in FORBIDDEN_REQUEST_FIELDS)


def test_generation_parameters_identical_across_conditions() -> None:
    prompts = load_prompts()
    bodies = []
    for condition in ("ko", "en"):
        system, user = render_generation(case(), condition, prompts)
        bodies.append(request_payload(
            model="snapshot", system_prompt=system, user_prompt=user, temperature=.2,
            top_p=.95, max_output_tokens=2000, seed=10, reasoning_effort="medium",
        ))
    for key in ("model", "temperature", "top_p", "max_tokens", "seed", "reasoning_effort"):
        assert bodies[0][key] == bodies[1][key]


def test_prompt_hashes_are_exact_sha256() -> None:
    prompts = load_prompts()
    hashes = prompt_hashes()
    assert all(len(value) == 64 for value in hashes.values())
    assert hashes["generation_ko_v1_system.txt"] == sha256_text(prompts["generation_ko_v1_system.txt"])


def test_resume_skips_completed_unique_key(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    key = unique_key("KR_1", "ko", 1, "snapshot", "generation_v1")
    append_jsonl(path, {"unique_key": key, "raw_response": "ok", "error": None})
    assert completed_keys(path) == {key}


def test_jsonl_append_is_not_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"a": "한글"})
    append_jsonl(path, {"a": "English"})
    assert read_jsonl(path) == [{"a": "한글"}, {"a": "English"}]


def test_evaluator_schema_and_conclusion_enum() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(valid_evaluation())
    invalid = valid_evaluation()
    invalid["parties"][0]["conclusion"] = "probably"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(invalid)


def test_strict_json_parser() -> None:
    value = valid_evaluation()
    assert parse_json_content("```json\n" + json.dumps(value) + "\n```") == value


def test_reasoning_unit_proportions_use_unit_denominator() -> None:
    evaluation = valid_evaluation()
    vector = reasoning_vector(evaluation)
    assert vector["governing_rule"] == .5
    assert vector["duty_or_protected_interest"] == .5
    assert vector["conclusion"] == .5
    assert all(0 <= vector[label] <= 1 for label in LABELS)
    # Multi-label coding means category proportions need not sum to one.
    assert sum(vector.values()) == 1.5


def test_request_order_is_deterministic_and_unique() -> None:
    first = build_plan([case("KR_1"), case("KR_2")], 2, "snapshot", 42)
    second = build_plan([case("KR_1"), case("KR_2")], 2, "snapshot", 42)
    signature = lambda plan: [(x["unique_key"], x["request_order"]) for x in plan]
    assert signature(first) == signature(second)
    assert len({x["unique_key"] for x in first}) == len(first)


def test_env_loader_does_not_override_existing_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EXP1_TEST_SECRET='from-file'\n", encoding="utf-8")
    monkeypatch.setenv("EXP1_TEST_SECRET", "existing")
    assert load_env_file(env_file)
    import os
    assert os.environ["EXP1_TEST_SECRET"] == "existing"


def test_pair_analysis_emits_one_row_per_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluations = tmp_path / "evaluations.jsonl"
    for case_id, origin in (("KR_1", "KR"), ("CA_1", "CA")):
        for condition in ("ko", "en"):
            append_jsonl(evaluations, {
                "evaluation_key": f"{case_id}-{condition}",
                "response_unique_key": f"{case_id}-{condition}",
                "case_id": case_id,
                "case_origin": origin,
                "case_subtype": "traffic_accident",
                "condition": condition,
                "replicate_id": 1,
                "raw_response": "analysis",
                "evaluation": valid_evaluation(),
            })
    output = tmp_path / "out"
    reports = tmp_path / "reports"
    monkeypatch.setattr("sys.argv", [
        "analyze_exp1.py", "--evaluations", str(evaluations),
        "--output-dir", str(output), "--reports-dir", str(reports),
        "--human-random-pairs", "2",
    ])
    analyze_main()
    with (output / "pair_metrics.csv").open(encoding="utf-8-sig") as handle:
        import csv
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert len({row["case_id"] for row in rows}) == 2
    assert len(list((output / "graphs").glob("*.png"))) == 5
