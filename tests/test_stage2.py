from __future__ import annotations

import json
from pathlib import Path

import pytest

from generate_neutral_fact_patterns import main as generate_main
from merge_neutral_pairs import main as merge_main
from pipeline.factual_evidence import extract_evidence
from pipeline.canonical_neutralization import _redact_monetary_values
from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl, read_jsonl_recover
from pipeline.leakage_checks import canonical_numbers, normalize_units, source_neutral_checks, translation_checks, validate_evidence
from pipeline.llm_client import LLMClient, LLMRequestError
from pipeline.neutral_translation import translate
from pipeline.neutral_verification import validate_grounding_verifier, validate_translation_verifier
from pipeline.response_parser import ResponseParseError, parse_json_response
from pipeline.source_segmentation import segment_source, select_candidate_chunks
from pipeline.stage2_input import validate_inputs
from pipeline.stage2_schema import Stage2CaseInput
from translate_neutral_fact_patterns import main as translate_main
from verify_neutral_fact_patterns import main as verify_main


ROOT = Path(__file__).resolve().parents[1]
KR_INPUT = ROOT / "outputs/raw/kr_v4/kr_cases_selected_35.jsonl"
CA_INPUT = ROOT / "outputs/raw/ca_v4/ca_cases_selected_35.jsonl"


def test_final_inputs_are_35x35_expected_and_immutable(tmp_path: Path) -> None:
    before = (KR_INPUT.read_bytes(), CA_INPUT.read_bytes())
    cases, report, manifest = validate_inputs(KR_INPUT, CA_INPUT, tmp_path)
    assert report["status"] == "pass" and len(cases) == 70
    assert report["kr_count"] == report["ca_count"] == 35
    assert report["case_id_unique_across_all_inputs"]
    assert report["origins"]["KR"]["required_source_field"] == "raw_text"
    assert report["origins"]["CA"]["required_source_field"] == "main_opinion_text"
    assert not report["warnings"]
    assert manifest["kr_input_file_sha256"] == "ca53460a99df2a59ffa1b4047cdfa406dd2afac54b053dccb99724dd850b8a49"
    assert manifest["ca_input_file_sha256"] == "35f9028cb5be3f331bc3df54511388986910ea86c7100ebb070d7ca2a2595aeb"
    assert before == (KR_INPUT.read_bytes(), CA_INPUT.read_bytes())


def test_ca_reserves_and_auto_false_remain_in_snapshot() -> None:
    rows = [json.loads(line) for line in CA_INPUT.read_text(encoding="utf-8").splitlines() if line]
    assert sum(row.get("selection_tier") == "reserve" for row in rows) == 5
    assert sum(row.get("auto_strict_eligible") is False for row in rows) == 14
    assert len(rows) == 35


def test_segmentation_is_deterministic_and_offsets_exact() -> None:
    text = "First event happened. Second event followed.\n\nFinal harm occurred."
    first, second = segment_source(text), segment_source(text)
    assert first == second
    assert [row.source_sentence_id for row in first] == ["SRC0001", "SRC0002", "SRC0003"]
    assert all(text[row.start_char:row.end_char] == row.text for row in first)


def test_candidate_selection_samples_late_opinion() -> None:
    text = "\n\n".join(f"Paragraph {i} has procedural material." for i in range(40)) + "\n\nThe injury occurred near the end."
    case = Stage2CaseInput("CA_X", "CA", "en", text, "main_opinion_text", "x", "general_personal_injury", None, None, "y")
    segments = segment_source(text)
    chunks, metadata = select_candidate_chunks(case, segments, max_input_tokens=2000, overlap_sentences=1, max_chunks=5)
    selected_ids = {sid for chunk in chunks for sid in chunk["source_sentence_ids"]}
    assert segments[-1].source_sentence_id in selected_ids
    assert selected_ids == {segment.source_sentence_id for segment in segments}
    assert metadata["segment_coverage_ratio"] == metadata["character_coverage_ratio"] == 1.0
    assert metadata["coverage_complete"] is True


def test_ordered_chunk_coverage_includes_crushing_and_death_sentences() -> None:
    text = "\n\n".join(["Background paragraph."] * 33 + ["The movement pushed the empty trailer into the lumber pile, crushing the person.", "The person was caught between the lumber and the trailer, resulting in death."] + ["Later procedure."] * 10)
    case = Stage2CaseInput("CA_TRUCK", "CA", "en", text, "main_opinion_text", "x", "traffic_accident", None, None, "y")
    segments = segment_source(text); chunks, metadata = select_candidate_chunks(case, segments, max_input_tokens=120, overlap_sentences=1)
    processed = {sid for chunk in chunks for sid in chunk["source_sentence_ids"]}
    crushing = [segment.source_sentence_id for segment in segments if "crushing" in segment.text or "resulting in death" in segment.text]
    assert set(crushing) <= processed
    assert metadata["coverage_complete"] and metadata["extraction_call_count"] > 1


def test_evidence_rejects_unknown_id_but_deterministic_excerpt_is_authoritative() -> None:
    segment_record = {"segments": [{"source_sentence_id": "SRC0001", "text": "Actual text."}]}
    payload = {"case_id": "X", "evidence_units": [{"evidence_id": "E001", "source_sentence_ids": ["SRC9999"], "exact_excerpts": ["invented"], "epistemic_status": "undisputed"}]}
    assert any("unknown_source_sentence_id" in error for error in validate_evidence(payload, "X", segment_record))
    payload["evidence_units"][0]["source_sentence_ids"] = ["SRC0001"]
    assert any("non_deterministic_exact_excerpt" in error for error in validate_evidence(payload, "X", segment_record))


def test_quote_punctuation_does_not_drop_evidence_when_source_id_is_valid(tmp_path: Path) -> None:
    case = Stage2CaseInput("CA_X", "CA", "en", "Actual text.", "main_opinion_text", "x", None, None, None, "y")
    segment_record = {"candidate_chunks": [{"chunk_id": "CHUNK001", "source_sentence_ids": ["SRC0001"], "text": "<SRC0001>Actual text.</SRC0001>"}], "segments": [{"source_sentence_id": "SRC0001", "text": "Actual text."}], "candidate_metadata": {"coverage_complete": True}}
    mock = {"case_id": "CA_X", "evidence_units": [
        {"evidence_id": "E001", "source_sentence_ids": ["SRC0001"], "short_quote": "Actual text", "proposed_fact_type": ["event"], "epistemic_status": "undisputed", "epistemic_status_confidence": "high"},
    ]}
    atomic_write_json(tmp_path / "mocks/factual_evidence/CA_X.json", mock)
    client = LLMClient(output_dir=tmp_path / "out", model="mock", base_url="https://invalid", mock_response_dir=tmp_path / "mocks")
    result = extract_evidence(case, segment_record, client, ROOT)
    assert result["extraction_status"] == "pass"
    assert [unit["evidence_id"] for unit in result["evidence_units"]] == ["E001"]
    assert result["evidence_units"][0]["exact_excerpts"] == ["Actual text."]


@pytest.mark.parametrize("raw", ['```json\n{"a":1}\n```', 'prefix {"a":1,} suffix'])
def test_response_parser_fallback(raw: str) -> None:
    assert parse_json_response(raw, ("a",)) == {"a": 1}


def test_response_parser_rejects_truncation_and_missing_fields() -> None:
    with pytest.raises(ResponseParseError): parse_json_response('{"a":', ("a",))
    with pytest.raises(ResponseParseError): parse_json_response('{"a":1}', ("b",))


def test_checkpoint_recovers_only_partial_final_line(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"; path.write_text('{"case_id":"A"}\n{"case_id":', encoding="utf-8")
    rows, recovered = read_jsonl_recover(path)
    assert recovered and rows == [{"case_id": "A"}]


def test_unit_normalization_uses_metric_and_reasonable_rounding() -> None:
    text, conversions = normalize_units("The vehicle moved at 30 mph for 2 miles and stopped 10 feet away.")
    assert "48.3 km/h" in text and "3.2 km" in text and "3 m" in text
    assert len(conversions) == 3


def test_source_gate_detects_legal_and_jurisdiction_leakage() -> None:
    payload = {"master_neutral_text": "The negligent person was liable in California.", "fact_units": [{"fact_id": "F001", "epistemic_status": "undisputed", "source_evidence_ids": ["E001"]}]}
    evidence = {"evidence_units": [{"evidence_id": "E001"}]}
    checks = source_neutral_checks(payload, evidence, "source words", "en")
    assert checks["status"] == "fail"
    assert "legal_term_leakage" in checks["errors"] and "jurisdiction_leakage" in checks["errors"]


def test_translation_checks_fact_order_placeholders_numbers_and_legal_terms() -> None:
    master = {"master_neutral_text": "[PERSON_A] waited 2 hours.", "fact_units": [{"fact_id": "F001"}]}
    translated = {"translated_neutral_text": "[PERSON_B] was liable after 3 hours.", "translated_fact_units": [{"fact_id": "F002", "translated_text": "x"}]}
    checks = translation_checks(master, translated, "en")
    assert {"fact_id_or_order_mismatch", "placeholder_identity_mismatch", "legal_term_reintroduced"} <= set(checks["errors"])


@pytest.mark.parametrize(("ko", "en"), [
    ("다툼이 없었다.", "It was undisputed."),
    ("비용의 3분의 1을 지출했다.", "One-third of the cost was spent."),
    ("편도 1차선 도로였다.", "It was a one-lane road."),
    ("약 8주 동안 치료가 필요했다.", "Treatment was needed for approximately 8 weeks."),
    ("[PERSON_A]는 23세였다.", "[PERSON_A] was 23 years old."),
    ("길이는 7.6 m였고 손상되었다.", "It was 7.6 m long and was damaged."),
    ("경사는 25%였다.", "The grade was 25 percent."),
])
def test_bilingual_surface_equivalents_are_not_hard_failures(ko: str, en: str) -> None:
    master = {"master_neutral_text": ko, "fact_units": [{"fact_id": "F001", "master_text": ko, "epistemic_status": "undisputed"}]}
    translated = {"translated_neutral_text": en, "translated_fact_units": [{"fact_id": "F001", "translated_text": en}]}
    assert translation_checks(master, translated, "en")["status"] != "fail"


def test_placeholder_occurrence_count_difference_is_warning_only() -> None:
    ko = "[PERSON_A]는 [PERSON_B]를 보았고 [PERSON_A]는 기다렸다."
    en = "[PERSON_A] saw [PERSON_B] and waited."
    checks = translation_checks({"master_neutral_text": ko, "fact_units": [{"fact_id": "F001", "master_text": ko}]}, {"translated_neutral_text": en, "translated_fact_units": [{"fact_id": "F001", "translated_text": en}]}, "en")
    assert checks["status"] == "warning" and "placeholder_occurrence_count_differs" in checks["warnings"]


@pytest.mark.parametrize(("master_text", "translated_text", "expected"), [
    ("거리는 2 m였다.", "The distance was 20 m.", "numerical_value_changed"),
    ("[PERSON_A]는 사망했다.", "[PERSON_A] suffered a minor injury.", "independent_fact_added_or_omitted"),
    ("[PERSON_A]는 위험을 알고 있었다.", "[PERSON_A] did not know of the danger.", "clear_polarity_reversal"),
    ("[PERSON_A]는 기다렸다.", "[PERSON_B] waited.", "placeholder_identity_mismatch"),
    ("[PERSON_A]는 서 있었다.", "[PERSON_A] stood and struck [PERSON_B].", "independent_fact_added_or_omitted"),
    ("[PERSON_A]는 기다렸다.", "[PERSON_A] was negligent while waiting.", "legal_term_reintroduced"),
    ("[PERSON_A]는 기다렸다.", "[PERSON_A] waited in California.", "jurisdiction_term_reintroduced"),
])
def test_corrupted_translation_hard_failures(master_text: str, translated_text: str, expected: str) -> None:
    master = {"master_neutral_text": master_text, "fact_units": [{"fact_id": "F001", "master_text": master_text, "epistemic_status": "undisputed"}]}
    translated = {"translated_neutral_text": translated_text, "translated_fact_units": [{"fact_id": "F001", "translated_text": translated_text}]}
    checks = translation_checks(master, translated, "en")
    assert checks["status"] == "fail" and any(expected in error for error in checks["errors"])


def test_deleted_fact_unit_is_hard_failure() -> None:
    master = {"master_neutral_text": "A. B.", "fact_units": [{"fact_id": "F001", "master_text": "A."}, {"fact_id": "F002", "master_text": "B."}]}
    translated = {"translated_neutral_text": "A.", "translated_fact_units": [{"fact_id": "F001", "translated_text": "A."}]}
    checks = translation_checks(master, translated, "en")
    assert checks["status"] == "fail" and "fact_id_or_order_mismatch" in checks["errors"]


def test_translation_request_excludes_raw_source_and_metadata(tmp_path: Path) -> None:
    mock = tmp_path / "mock" / "translation"; mock.mkdir(parents=True)
    atomic_write_json(mock / "KR_X.json", {"case_id": "KR_X", "translated_neutral_text": "[PERSON_A] waited.", "translated_fact_units": [{"fact_id": "F001", "translated_text": "[PERSON_A] waited."}]})
    client = LLMClient(output_dir=tmp_path / "out", model="mock", base_url="https://invalid", mock_response_dir=tmp_path / "mock")
    master = {"case_id": "KR_X", "case_origin": "KR", "master_language": "ko", "master_neutral_text": "[PERSON_A]는 기다렸다.", "fact_units": [{"fact_id": "F001", "master_text": "[PERSON_A]는 기다렸다.", "epistemic_status": "undisputed"}], "raw_text": "SECRET", "case_subtype": "secret", "court_name": "secret"}
    result = translate(master, client, ROOT)
    assert result["translation_direction"] == "ko_to_en"
    cache = next((tmp_path / "out/request_cache/translation").glob("*.json"))
    assert "SECRET" not in cache.read_text(encoding="utf-8")


def test_structured_output_and_unsupported_seed_fallback(tmp_path: Path) -> None:
    class FakeClient(LLMClient):
        calls: list[dict] = []
        def _http(self, body):
            self.calls.append(dict(body))
            if "seed" in body:
                error = LLMRequestError("HTTP 400: unsupported seed parameter"); error.status = 400; error.headers = {}
                raise error
            if body.get("response_format", {}).get("type") == "json_schema":
                error = LLMRequestError("HTTP 400: json_schema unsupported"); error.status = 400; error.headers = {}
                raise error
            return {"choices": [{"message": {"content": '{"case_id":"X"}'}}], "usage": {}}, {}
    client = FakeClient(output_dir=tmp_path, model="model", base_url="https://example.invalid/v1", seed=7)
    result = client.call(case_id="X", stage="test", system_prompt="prompt", user_payload={}, schema={"type": "object"}, required_fields=("case_id",), prompt_version="v1")
    assert result.payload == {"case_id": "X"}
    assert result.provenance["structured_output_mode"] == "json_object"
    assert result.provenance["requested_generation_parameters"]["seed"] == 7
    assert "seed" not in result.provenance["effective_generation_parameters"]


def test_rate_limit_retry_respects_retry_path(tmp_path: Path) -> None:
    class RetryClient(LLMClient):
        calls = 0
        def _http(self, body):
            self.calls += 1
            if self.calls == 1:
                error = LLMRequestError("HTTP 429"); error.status = 429; error.headers = {"Retry-After": "0"}
                raise error
            return {"choices": [{"message": {"content": '{"case_id":"X"}'}}], "usage": {}}, {}
    client = RetryClient(output_dir=tmp_path, model="model", base_url="https://example.invalid/v1", max_retries=1)
    assert client.call(case_id="X", stage="retry", system_prompt="prompt", user_payload={}, schema={"type": "object"}, required_fields=("case_id",), prompt_version="v1").payload["case_id"] == "X"
    assert client.calls == 2


def test_verifier_status_is_corrected_from_payload_arrays() -> None:
    grounding = {"verifier_status": "pass", "grounded": True, "unsupported_fact_ids": ["F001"], "overstated_fact_ids": [], "missing_material_facts": [], "epistemic_status_errors": [], "legal_conclusion_leakage": [], "jurisdiction_leakage": []}
    result = validate_grounding_verifier(grounding)
    assert result["validated_verifier_status"] == "fail" and result["verifier_consistency_violation"]
    translation = {"translation_status": "pass", "meaning_preserved": True, "missing_fact_ids": [], "added_information": ["new act"], "omitted_information": [], "changed_negation": [], "changed_temporal_relation": [], "changed_epistemic_status": [], "legal_term_reintroduction": [], "placeholder_errors": []}
    result = validate_translation_verifier(translation)
    assert result["validated_verifier_status"] == "fail" and result["verifier_consistency_violation"]


def test_target_language_residue_is_warning() -> None:
    master_text, translated_text = "The grade was 25 percent.", "경사는 25 percent였다."
    checks = translation_checks({"master_neutral_text": master_text, "fact_units": [{"fact_id": "F001", "master_text": master_text}]}, {"translated_neutral_text": translated_text, "translated_fact_units": [{"fact_id": "F001", "translated_text": translated_text}]}, "ko")
    assert checks["status"] == "warning" and "percent" in checks["language_residue"]


def test_comma_imperial_units_and_korean_number_words_normalize() -> None:
    normalized, conversions = normalize_units("40,000 miles and 23,500 miles")
    assert normalized == "64374 km and 37819 km"
    assert len(conversions) == 2
    assert canonical_numbers("two children, age eleven and age eight")[0] == canonical_numbers("두 자녀, 열한 살과 여덟 살")[0]


def test_monetary_redaction_preserves_category_and_removes_unsupported_time() -> None:
    text, conversions = _redact_monetary_values("사고 전 치료비는 6,282,940원이었다.")
    assert text == "치료비는 [AMOUNT]이었다."
    assert {item["type"] for item in conversions} == {"monetary_redaction", "unsupported_temporal_modifier_removal"}


def test_merge_preserves_missing_cases(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "input_manifest.json", {"dataset_version": "stage2-neutral-facts-35x35-v1", "kr_case_ids": ["KR_A"], "ca_case_ids": ["CA_A"]})
    for name in ("source_neutral_kr.jsonl", "source_neutral_ca.jsonl", "translated_pairs_kr.jsonl", "translated_pairs_ca.jsonl", "source_grounding_verification.jsonl", "translation_verification.jsonl"):
        atomic_write_jsonl(tmp_path / name, [])
    assert merge_main(["--input-dir", str(tmp_path)]) == 0
    rows = [json.loads(line) for line in (tmp_path / "neutral_pairs_all.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["case_id"] for row in rows] == ["KR_A", "CA_A"]
    assert all(row["source_neutral_status"] == "missing" for row in rows)


def test_mock_end_to_end_cli_preserves_all_70_manifest_cases(tmp_path: Path) -> None:
    cases, _, _ = validate_inputs(KR_INPUT, CA_INPUT)
    chosen = [next(case for case in cases if case.case_origin == origin) for origin in ("KR", "CA")]
    mock_dir, output = tmp_path / "mocks", tmp_path / "output"
    for case in chosen:
        segments = segment_source(case.source_text)
        chunks, _ = select_candidate_chunks(case, segments)
        source_id = chunks[0]["source_sentence_ids"][0]
        source_text = next(segment.text for segment in segments if segment.source_sentence_id == source_id)
        excerpt = source_text[: min(20, len(source_text))]
        fact_text = "[PERSON_A]는 현장에 있었다." if case.case_origin == "KR" else "[PERSON_A] was present at the location."
        translated_text = "[PERSON_A] was present at the location." if case.case_origin == "KR" else "[PERSON_A]는 현장에 있었다."
        fixtures = {
                "factual_evidence": {"case_id": case.case_id, "evidence_units": [{"evidence_id": "E001", "source_sentence_ids": [source_id], "short_quote": excerpt, "proposed_fact_type": ["event", "harm"], "epistemic_status": "undisputed", "epistemic_status_confidence": "high"}]},
                "source_neutral": {"case_id": case.case_id, "master_neutral_text": fact_text, "fact_units": [{"fact_id": "F001", "master_text": fact_text, "epistemic_status": "undisputed", "epistemic_status_confidence": "high", "fact_types": ["event", "harm"], "source_evidence_ids": ["E001"]}], "removed_legal_signals": [], "removed_jurisdiction_signals": [], "anonymization_warnings": [], "grounding_warnings": [], "insufficient_factual_detail": False},
            "translation": {"case_id": case.case_id, "translated_neutral_text": translated_text, "translated_fact_units": [{"fact_id": "F001", "translated_text": translated_text}]},
            "grounding_verifier": {"case_id": case.case_id, "grounded": True, "unsupported_fact_ids": [], "overstated_fact_ids": [], "missing_material_facts": [], "epistemic_status_errors": [], "legal_conclusion_leakage": [], "jurisdiction_leakage": [], "verifier_status": "pass", "verifier_notes": []},
            "translation_verifier": {"case_id": case.case_id, "meaning_preserved": True, "missing_fact_ids": [], "added_information": [], "omitted_information": [], "changed_negation": [], "changed_temporal_relation": [], "changed_epistemic_status": [], "legal_term_reintroduction": [], "placeholder_errors": [], "translation_status": "pass", "translation_notes": []},
        }
        for stage, payload in fixtures.items(): atomic_write_json(mock_dir / stage / f"{case.case_id}.json", payload)
    case_args = [part for case in chosen for part in ("--case-id", case.case_id)]
    assert generate_main(["--kr-input", str(KR_INPUT), "--ca-input", str(CA_INPUT), "--output-dir", str(output), "--mock-response-dir", str(mock_dir), *case_args]) == 0
    usage_before = (output / "api_usage.csv").read_bytes()
    assert generate_main(["--kr-input", str(KR_INPUT), "--ca-input", str(CA_INPUT), "--output-dir", str(output), "--mock-response-dir", str(mock_dir), "--resume", *case_args]) == 0
    assert (output / "api_usage.csv").read_bytes() == usage_before
    assert translate_main(["--kr-source-neutral", str(output / "source_neutral_kr.jsonl"), "--ca-source-neutral", str(output / "source_neutral_ca.jsonl"), "--output-dir", str(output), "--mock-response-dir", str(mock_dir), "--resume"]) == 0
    assert verify_main(["--source-neutral-input", str(output), "--translation-input", str(output), "--output-dir", str(output), "--mock-response-dir", str(mock_dir), "--resume"]) == 0
    assert merge_main(["--input-dir", str(output)]) == 0
    all_rows = [json.loads(line) for line in (output / "neutral_pairs_all.jsonl").read_text(encoding="utf-8").splitlines()]
    pass_rows = [json.loads(line) for line in (output / "neutral_pairs_pass.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(all_rows) == 70 and len({row["case_id"] for row in all_rows}) == 70
    assert len(pass_rows) == 2
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["phases"]["phase_9_merge"]["execution_status"] == "completed_subset"
    assert manifest["phases"]["phase_9_merge"]["missing_master_count"] == 68
    assert manifest["run_history"][-4]["new_api_calls"] == 0  # resumed generation entry
