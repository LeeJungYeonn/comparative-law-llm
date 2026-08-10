from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.llm_client import LLMClient
from pipeline.paired_qc import (
    SOURCE_REVIEW_FIELDS,
    TRANSLATION_REVIEW_FIELDS,
    adjudicate_source_qc,
    expected_provenance,
    finalize_pairs,
    generation_consistency,
    import_source_reviews,
    import_translation_reviews,
    master_sha256,
    recognition_metrics,
    source_review_rows,
    translation_readiness,
    translation_review_rows,
    translation_sha256,
    write_csv,
)
from qc_neutral_fact_pairs import (
    _overlay_regenerated_translations,
    _run_back_translation,
    _source_qc_ready,
    _unreviewed_records,
)


def _master() -> dict:
    units = [
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
            "master_text": "[PERSON_B] suffered an injury.",
            "epistemic_status": "documented_record",
            "fact_types": ["harm"],
            "source_evidence_ids": ["E002"],
            "relation_ids": [],
            "realized_relations": [],
        },
    ]
    return {
        "dataset_version": "stage2-neutral-facts-35x35-test",
        "case_id": "KR_TEST", "case_origin": "KR",
        "source_language": "ko", "master_language": "ko",
        "master_neutral_text": " ".join(unit["master_text"] for unit in units),
        "fact_units": units,
        "source_coverage": {"coverage_status": "complete"},
        "deterministic_checks": {
            "event_present": True, "harm_present": True,
            "event_harm_sequence_present": True,
        },
    }


def _translation(master: dict | None = None) -> dict:
    master = master or _master()
    units = [
        {
            "fact_id": "F001",
            "translated_text": "[PERSON_A]가 [PERSON_B]를 치료했다.",
            "relation_ids": ["R001"],
            "realized_relations": [{
                "subject_placeholder": "[PERSON_A]",
                "relation_type": "treated",
                "object_placeholder": "[PERSON_B]",
            }],
        },
        {
            "fact_id": "F002",
            "translated_text": "[PERSON_B]가 부상을 입었다.",
            "relation_ids": [], "realized_relations": [],
        },
    ]
    return {
        "case_id": "KR_TEST", "case_origin": "KR",
        "translation_direction": "ko_to_en",
        "master_neutral_text": master["master_neutral_text"],
        "translated_neutral_text": " ".join(
            unit["translated_text"] for unit in units
        ),
        "translated_fact_units": units,
    }


def _case() -> dict:
    master = _master()
    return {
        "case_id": "KR_TEST", "case_origin": "KR",
        "case_subtype": "medical_malpractice",
        **expected_provenance("KR"),
        "raw_source": "source", "raw_record": {"original_outcome": "hidden"},
        "master": master, "translation": _translation(master),
        "graph": {"entities": [], "relations": []},
    }


def _source_qc(status: str = "pass", hard: bool = False) -> dict:
    finding = {
        "fact_id": "F001", "master_span": "treated",
        "source_sentence_ids": ["SRC001"], "source_excerpt": "source",
        "error_type": "wrong_actor", "severity": "hard",
        "explanation": "actor differs",
    }
    return {
        "case_id": "KR_TEST", "case_origin": "KR",
        "validated_source_qc_status": status,
        "unresolved_hard_failure": hard,
        "source_qc_prompt_version": "qc_source_neutral_ko_v1_en",
        "source_qc": {
            "unsupported_facts": [], "overstated_facts": [],
            "missing_material_facts": [], "entity_role_errors": (
                [finding] if hard else []
            ),
            "epistemic_status_errors": [], "legal_conclusion_leakage": [],
            "jurisdiction_leakage": [], "model_confidence": "high",
            "factual_sufficiency": "sufficient", "source_copy_risk": "low",
        },
        "deterministic_precheck": {
            "source_coverage": {"coverage_status": "complete"},
        },
    }


def _translation_qc(status: str = "pass", hard: bool = False) -> dict:
    return {
        "case_id": "KR_TEST", "translation_direction": "ko_to_en",
        "validated_translation_qc_status": status,
        "unresolved_hard_failure": hard,
        "translation_qc_prompt_version": "qc_translation_ko_to_en_v1_en",
        "translation_qc": {
            "semantic_equivalence": status,
            **{
                field: [] for field in (
                    "added_information", "omitted_information",
                    "subject_object_shifts", "entity_role_shifts",
                    "epistemic_status_shifts", "polarity_shifts",
                    "temporal_relation_shifts", "causal_direction_shifts",
                    "spatial_direction_shifts", "number_unit_shifts",
                    "legal_terms_reintroduced",
                    "jurisdiction_signals_reintroduced",
                    "fact_structure_errors",
                )
            },
        },
        "deterministic_precheck": {
            "checks": {
                "fact_id_match": True, "placeholder_identity_match": True,
            },
        },
    }


def test_expected_provenance_has_source_master_then_translation() -> None:
    assert expected_provenance("KR") == {
        "source_language": "ko", "master_language": "ko",
        "translation_direction": "ko_to_en",
        "ko_generation_type": "source_neutralized",
        "en_generation_type": "translated",
    }
    assert expected_provenance("CA")["translation_direction"] == "en_to_ko"


def test_recognition_metrics_detect_names_dates_amounts_and_copy() -> None:
    source = "Acme Hospital treated the patient on 2020-01-02 for $500."
    master = "Acme Hospital treated the patient on 2020-01-02 for $500."
    graph = {"entities": [{"source_mentions": ["Acme Hospital"]}]}
    checked = recognition_metrics(source, master, graph)
    assert checked["source_copy_risk"] == "high"
    assert checked["actual_name_retention"] == ["Acme Hospital"]
    assert checked["unique_date_retention"]
    assert checked["unique_amount_retention"]


def test_recognition_metrics_do_not_treat_generic_roles_as_actual_names() -> None:
    source = "The truck carried lumber and the mother saw the accident."
    master = "The truck carried lumber and the mother saw the accident."
    graph = {
        "entities": [{
            "source_mentions": ["truck", "lumber", "mother", "accident"],
        }],
    }
    checked = recognition_metrics(source, master, graph)
    assert checked["actual_name_retention"] == []


def test_deterministic_hard_rule_overrides_gpt_pass() -> None:
    case = _case()
    finding = {
        "fact_id": "F001", "master_span": "A", "source_sentence_ids": ["S1"],
        "source_excerpt": "B", "error_type": "wrong_actor",
        "severity": "hard", "explanation": "wrong actor",
    }
    model = {
        "source_qc": {
            "unsupported_facts": [], "overstated_facts": [],
            "missing_material_facts": [], "entity_role_errors": [finding],
            "epistemic_status_errors": [], "legal_conclusion_leakage": [],
            "jurisdiction_leakage": [], "source_copy_risk": "low",
            "factual_sufficiency": "sufficient",
            "model_source_qc_status": "pass", "model_confidence": "high",
        }
    }
    deterministic = {
        "deterministic_source_qc_status": "pass", "errors": [],
        "warnings": [], "source_coverage": {"coverage_status": "complete"},
    }
    result = adjudicate_source_qc(case, deterministic, model)
    assert result["validated_source_qc_status"] == "fail"
    assert result["model_source_qc_status"] == "pass"
    assert "wrong_actor" in result["validation_reasons"]


def test_generation_consistency_detects_mixed_prompt_language_and_version(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "neutralize_ko_v5.txt").write_text(
        "한국어 지시를 사용한다.", encoding="utf-8",
    )
    (prompts / "neutralize_ko_v6.txt").write_text(
        "Use evidence to generate output.", encoding="utf-8",
    )
    (prompts / "neutralize_en_v6.txt").write_text(
        "Use evidence to generate output.", encoding="utf-8",
    )
    indexes = {
        key: {} for key in (
            "segments", "evidence", "graph", "master", "translation",
            "source_checks", "source_verifier", "translation_checks",
            "translation_verifier",
        )
    }
    for case_id, origin, version in (
        ("KR_A", "KR", "neutralize_ko_v5"),
        ("KR_B", "KR", "neutralize_ko_v6"),
        ("CA_A", "CA", "neutralize_en_v6"),
    ):
        indexes["master"][case_id] = {
            "case_origin": origin, "dataset_version": "v3",
            "model_provenance": {
                "model": "gpt-test", "prompt_version": version, "mock": False,
            },
        }
    result = generation_consistency(tmp_path, indexes)
    assert result["status"] == "fail"
    assert "generation_instruction_language_not_uniform_english" in result["errors"]
    assert "neutralization_prompt_version_mixed_within_origin" in result["errors"]


def test_human_source_import_edit_requires_notes_and_fact_units(
    tmp_path: Path,
) -> None:
    case = _case()
    cases = {"KR_TEST": case}
    auto = _source_qc()
    row = source_review_rows("stage-a", cases, [auto])[0]
    row.update({
        "human_source_action": "edit_master",
        "human_source_status": "accepted_with_edits",
        "edited_fact_ids": "F001", "reviewer_id": "reviewer_1",
        "human_validated_master_text": "edited", "reviewer_notes": "",
        "human_validated_fact_units_json": json.dumps(
            case["master"]["fact_units"]
        ),
    })
    path = tmp_path / "source.csv"
    write_csv(path, SOURCE_REVIEW_FIELDS, [row])
    with pytest.raises(ValueError, match="notes required"):
        import_source_reviews(path, cases, tmp_path)


def test_source_edit_marks_translation_stale(tmp_path: Path) -> None:
    case = _case()
    cases = {"KR_TEST": case}
    row = source_review_rows("stage-a", cases, [_source_qc()])[0]
    units = [dict(unit) for unit in case["master"]["fact_units"]]
    units[0] = {**units[0], "master_text": "[PERSON_A] examined [PERSON_B]."}
    row.update({
        "human_source_action": "edit_master",
        "human_source_status": "accepted_with_edits",
        "edited_fact_ids": "F001", "reviewer_id": "reviewer_1",
        "reviewer_notes": "Corrected provider role.",
        "edit_reasons": "corrected provider role",
        "human_validated_master_text": " ".join(
            unit["master_text"] for unit in units
        ),
        "human_validated_fact_units_json": json.dumps(units),
    })
    path = tmp_path / "source.csv"
    write_csv(path, SOURCE_REVIEW_FIELDS, [row])
    import_source_reviews(path, cases, tmp_path)
    stale = json.loads(
        (tmp_path / "translations_requiring_regeneration.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert stale["translation_requires_regeneration"] is True


def test_source_import_allows_fully_reviewed_fact_structure_replacement(
    tmp_path: Path,
) -> None:
    case = _case()
    cases = {"KR_TEST": case}
    row = source_review_rows("stage-a", cases, [_source_qc()])[0]
    units = [dict(case["master"]["fact_units"][0])]
    units[0]["fact_id"] = "F001"
    units[0]["master_text"] = "[PERSON_A] examined [PERSON_B]."
    row.update({
        "human_source_action": "edit_master",
        "human_source_status": "accepted_with_edits",
        "edited_fact_ids": "F001",
        "reviewer_id": "reviewer_1",
        "reviewer_notes": "Rebuilt the complete material-fact structure.",
        "human_validated_master_text": units[0]["master_text"],
        "human_validated_fact_units_json": json.dumps(units),
    })
    path = tmp_path / "source.csv"
    write_csv(path, SOURCE_REVIEW_FIELDS, [row])
    import_source_reviews(path, cases, tmp_path)
    imported = json.loads(
        (tmp_path / "human_validated_masters.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert imported["fact_structure_changed"] is True
    assert imported["original_fact_ids"] == ["F001", "F002"]
    assert imported["human_validated_fact_ids"] == ["F001"]


def test_source_import_allows_lossless_split_with_marked_text_edits(
    tmp_path: Path,
) -> None:
    case = _case()
    cases = {"KR_TEST": case}
    row = source_review_rows("stage-a", cases, [_source_qc()])[0]
    units = [
        {
            **case["master"]["fact_units"][0],
            "master_text": "[PERSON_A] carefully examined [PERSON_B].",
        },
        {
            **case["master"]["fact_units"][1],
            "fact_id": "F002",
            "master_text": "[PERSON_B] was injured.",
        },
        {
            **case["master"]["fact_units"][1],
            "fact_id": "F003",
            "master_text": "Treatment followed.",
        },
    ]
    case["master"]["fact_units"][1]["master_text"] = (
        "[PERSON_B] was injured. Treatment followed."
    )
    case["master"]["master_neutral_text"] = " ".join(
        unit["master_text"] for unit in case["master"]["fact_units"]
    )
    row = source_review_rows("stage-a", cases, [_source_qc()])[0]
    row.update({
        "human_source_action": "edit_master",
        "human_source_status": "accepted_with_edits",
        "edited_fact_ids": "F001",
        "reviewer_id": "reviewer_1",
        "reviewer_notes": "Edited F001 and split F002 without text changes.",
        "human_validated_master_text": " ".join(
            unit["master_text"] for unit in units
        ),
        "human_validated_fact_units_json": json.dumps(units),
    })
    path = tmp_path / "source.csv"
    write_csv(path, SOURCE_REVIEW_FIELDS, [row])
    import_source_reviews(path, cases, tmp_path)
    imported = json.loads(
        (tmp_path / "human_validated_masters.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert imported["fact_structure_changed"] is True
    assert imported["edited_fact_ids"] == ["F001"]


def test_source_import_rejects_unmarked_structural_text_edit(
    tmp_path: Path,
) -> None:
    case = _case()
    cases = {"KR_TEST": case}
    units = [
        dict(case["master"]["fact_units"][0]),
        {
            **case["master"]["fact_units"][1],
            "fact_id": "F002",
            "master_text": "Substantively changed text.",
        },
        {
            **case["master"]["fact_units"][1],
            "fact_id": "F003",
            "master_text": "Additional fact.",
        },
    ]
    row = source_review_rows("stage-a", cases, [_source_qc()])[0]
    row.update({
        "human_source_action": "edit_master",
        "human_source_status": "accepted_with_edits",
        "edited_fact_ids": "F001",
        "reviewer_id": "reviewer_1",
        "reviewer_notes": "Incomplete edit markers.",
        "human_validated_master_text": " ".join(
            unit["master_text"] for unit in units
        ),
        "human_validated_fact_units_json": json.dumps(units),
    })
    path = tmp_path / "source.csv"
    write_csv(path, SOURCE_REVIEW_FIELDS, [row])
    with pytest.raises(ValueError, match="substantively changed"):
        import_source_reviews(path, cases, tmp_path)


def test_translation_readiness_never_uses_stale_translation() -> None:
    case = _case()
    cases = {"KR_TEST": case}
    edited = {
        "case_id": "KR_TEST",
        "human_source_status": "accepted_with_edits",
        "human_validated_master_text": "changed master",
        "human_validated_fact_units": case["master"]["fact_units"],
        "human_validated_master_sha256": "changed",
    }
    ready, stale = translation_readiness(
        cases, {"KR_TEST": _source_qc("warning")},
        {"KR_TEST": edited},
    )
    assert ready == []
    assert stale[0]["translation_requires_regeneration"] is True


def test_translation_import_and_final_export_only_accept_reviewed_pair(
    tmp_path: Path,
) -> None:
    case = _case()
    cases = {"KR_TEST": case}
    master = case["master"]
    source_review = {
        "case_id": "KR_TEST", "human_source_status": "accepted",
        "human_validated_master_text": master["master_neutral_text"],
        "human_validated_fact_units": master["fact_units"],
        "human_validated_master_sha256": master_sha256(master),
        "reviewer_id": "source_reviewer", "review_timestamp": "now",
    }
    ready_item = {
        "case": case, "master": master, "translation": case["translation"],
    }
    translation_qc = _translation_qc()
    row = translation_review_rows(
        "stage-a", {"KR_TEST": ready_item}, [translation_qc],
    )[0]
    row.update({
        "human_translation_action": "accept_translation",
        "human_translation_status": "accepted",
        "reviewer_id": "translation_reviewer",
    })
    path = tmp_path / "translation.csv"
    write_csv(path, TRANSLATION_REVIEW_FIELDS, [row])
    import_translation_reviews(
        path, {"KR_TEST": ready_item}, tmp_path,
    )
    human_translation = json.loads(
        (tmp_path / "human_validated_translations.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    report = finalize_pairs(
        tmp_path, cases, {"KR_TEST": _source_qc()},
        {"KR_TEST": translation_qc},
        {"KR_TEST": source_review},
        {"KR_TEST": human_translation},
        tmp_path / "experiments", tmp_path / "manifests", "stage-a",
    )
    accepted = json.loads(
        (tmp_path / "accepted_pairs.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert accepted["case_is_finally_usable"] is True
    assert accepted["source_master_sha256"] == accepted[
        "translation_parent_master_sha256"
    ]
    assert report["accepted_count"] == 1
    experiment = json.loads(
        (tmp_path / "experiments" / "no_jurisdiction_pairs.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert "case_origin" not in experiment
    assert "QC" not in json.dumps(experiment)


def test_translation_import_normalizes_legacy_master_text_units(
    tmp_path: Path,
) -> None:
    case = _case()
    item = {
        "case": case, "master": case["master"],
        "translation": case["translation"],
    }
    row = translation_review_rows(
        "stage-a", {"KR_TEST": item}, [_translation_qc("fail")],
    )[0]
    units = [
        {
            **unit,
            "master_text": unit["translated_text"],
        }
        for unit in case["translation"]["translated_fact_units"]
    ]
    for unit in units:
        unit.pop("translated_text")
    row.update({
        "human_translation_action": "edit_translation",
        "human_translation_status": "accepted_with_edits",
        "edited_fact_ids": "F001",
        "reviewer_notes": "Human bilingual correction.",
        "reviewer_id": "reviewer",
        "human_validated_translation_text": " ".join(
            unit["master_text"] for unit in units
        ),
        "human_validated_translation_fact_units_json": json.dumps(units),
    })
    path = tmp_path / "translation.csv"
    write_csv(path, TRANSLATION_REVIEW_FIELDS, [row])
    import_translation_reviews(path, {"KR_TEST": item}, tmp_path)
    imported = json.loads(
        (tmp_path / "human_validated_translations.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert all(
        unit.get("translated_text")
        for unit in imported["human_validated_translation_fact_units"]
    )
    assert all(
        "master_text" not in unit
        for unit in imported["human_validated_translation_fact_units"]
    )


def test_prompts_are_english_instructions_and_forbid_repairs() -> None:
    root = Path(__file__).resolve().parents[1] / "prompts"
    for path in root.glob("qc_*_v1_en.txt"):
        text = path.read_text(encoding="utf-8")
        assert "Do not" in text or "must never" in text
        assert "Return one JSON object" in text
    translation = (
        root / "qc_translation_ko_to_en_v1_en.txt"
    ).read_text(encoding="utf-8")
    assert "raw judgment is intentionally absent" in translation
    assert "Do not rewrite" in translation


def test_mocked_back_translation_is_diagnostic_and_does_not_overwrite(
    tmp_path: Path,
) -> None:
    case = _case()
    item = {
        "case": case, "master": case["master"],
        "translation": case["translation"],
    }
    original_master = json.dumps(case["master"], sort_keys=True)
    original_translation = json.dumps(case["translation"], sort_keys=True)
    mock_dir = tmp_path / "mocks"
    mock_path = mock_dir / "qc_back_translation"
    mock_path.mkdir(parents=True)
    (mock_path / "KR_TEST.json").write_text(json.dumps({
        "case_id": "KR_TEST",
        "back_translated_fact_units": [
            {"fact_id": "F001", "back_translated_text": "diagnostic one"},
            {"fact_id": "F002", "back_translated_text": "diagnostic two"},
        ],
        "diagnostic_notes": ["No material shift detected."],
    }), encoding="utf-8")
    client = LLMClient(
        output_dir=tmp_path, model="mock", base_url="https://invalid",
        mock_response_dir=mock_dir,
    )
    _run_back_translation(
        object(), {"KR_TEST": item}, {
            "KR_TEST": {
                **_translation_qc("warning"),
                "validation_reasons": ["deterministic_and_gpt_disagreement"],
            },
        }, tmp_path, client,
    )
    assert json.dumps(case["master"], sort_keys=True) == original_master
    assert json.dumps(case["translation"], sort_keys=True) == original_translation
    output = json.loads(
        (tmp_path / "optional_back_translations.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert output["diagnostic_only"] is True
    assert output["cannot_replace_final_text"] is True


def test_regenerated_translation_overlay_requires_current_human_master(
    tmp_path: Path,
) -> None:
    case = _case()
    cases = {"KR_TEST": case}
    review = {
        "case_id": "KR_TEST",
        "human_validated_master_sha256": "current-master",
    }
    translated = {
        **_translation(case["master"]),
        "case_id": "KR_TEST",
        "parent_master_sha256": "current-master",
    }
    (tmp_path / "regenerated_translations.jsonl").write_text(
        json.dumps(translated) + "\n", encoding="utf-8",
    )
    assert _overlay_regenerated_translations(
        cases, {"KR_TEST": review}, tmp_path,
    ) == 1
    assert cases["KR_TEST"]["translation"] == translated

    cases["KR_TEST"]["translation"] = {"case_id": "KR_TEST", "old": True}
    review["human_validated_master_sha256"] = "newer-master"
    assert _overlay_regenerated_translations(
        cases, {"KR_TEST": review}, tmp_path,
    ) == 0
    assert cases["KR_TEST"]["translation"]["old"] is True


def test_cumulative_batch_exports_only_unreviewed_cases() -> None:
    cases = {"A": {}, "B": {}, "C": {}}
    records = [
        {"case_id": "A"}, {"case_id": "B"}, {"case_id": "OUTSIDE"},
    ]
    assert _unreviewed_records(records, cases, {"A": {"accepted": True}}) == [
        {"case_id": "B"},
    ]


def test_source_qc_does_not_require_prior_generation_verifier() -> None:
    case = {
        "segments": {"ok": True},
        "evidence": {"ok": True},
        "graph": {"ok": True},
        "master": {"ok": True},
        "source_checks": {"ok": True},
        "source_verifier": None,
        "raw_source": "source text",
    }
    assert _source_qc_ready(case) is True
