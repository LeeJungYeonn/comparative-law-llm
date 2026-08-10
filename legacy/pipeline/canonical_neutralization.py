from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pipeline.factual_evidence import load_prompt
from pipeline.leakage_checks import normalize_units, source_neutral_checks
from pipeline.llm_client import LLMClient
from pipeline.stage2_schema import DATASET_VERSION, NEUTRAL_SCHEMA, Stage2CaseInput


MONEY_PATTERNS = (
    re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*원"),
    re.compile(r"(?:[$€£¥₩]\s*(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)|(?:USD|KRW)\s*(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)|(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?:dollars?|won))", re.I),
)


def _redact_monetary_values(text: str) -> tuple[str, list[dict[str, Any]]]:
    conversions: list[dict[str, Any]] = []
    result = text
    for unsupported_temporal_phrase in ("사고 전 치료비", "사고 이전 치료비"):
        if unsupported_temporal_phrase in result:
            result = result.replace(unsupported_temporal_phrase, "치료비")
            conversions.append({"original": unsupported_temporal_phrase, "normalized": "치료비", "type": "unsupported_temporal_modifier_removal"})
    epistemic_replacements = {
        "얻을 수 있었던 것으로 평가되었다": "얻을 수 있을 것으로 추정되었다",
        "얻을 수 있었던 것으로 인정되었다": "얻을 수 있을 것으로 추정되었다",
    }
    for original, normalized in epistemic_replacements.items():
        if original in result:
            result = result.replace(original, normalized)
            conversions.append({"original": original, "normalized": normalized, "type": "epistemic_normalization"})
    for pattern in MONEY_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            conversions.append({"original": match.group(0), "normalized": "[AMOUNT]", "type": "monetary_redaction"})
            return "[AMOUNT]"
        result = pattern.sub(replace, result)
    return result, conversions


def _finalize_payload(payload: dict[str, Any], evidence: dict[str, Any], source_text: str, source_language: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(payload)
    payload["fact_units"] = [dict(unit) for unit in payload.get("fact_units", [])]
    model_master_text = str(payload.get("master_neutral_text") or "")
    conversions: list[dict[str, Any]] = []
    for unit in payload["fact_units"]:
        unit["master_text"], monetary_conversions = _redact_monetary_values(str(unit.get("master_text") or "")); conversions.extend(monetary_conversions)
        unit["master_text"], unit_conversions = normalize_units(unit["master_text"]); conversions.extend(unit_conversions)
    composed = " ".join(str(unit.get("master_text") or "").strip() for unit in payload["fact_units"] if str(unit.get("master_text") or "").strip())
    rebuilt = bool(composed and composed != model_master_text)
    if composed: payload["master_neutral_text"] = composed
    else: payload["master_neutral_text"], summary_conversions = normalize_units(model_master_text); conversions.extend(summary_conversions)
    conversions = list({(item["original"], item["normalized"]): item for item in conversions}.values())
    checks = source_neutral_checks(payload, evidence, source_text, source_language)
    checks["master_text_rebuilt_from_fact_units"] = rebuilt
    return payload, {"checks": checks, "conversions": conversions, "rebuilt": rebuilt}


def neutralize(case: Stage2CaseInput, evidence: dict[str, Any], client: LLMClient, root: Path) -> dict[str, Any]:
    prompt_name = "neutralize_ko_v2.txt" if case.case_origin == "KR" else "neutralize_en_v2.txt"
    minimal = [{"evidence_id": item["evidence_id"], "source_sentence_ids": item["source_sentence_ids"], "exact_excerpts_untrusted": item["exact_excerpts"], "proposed_fact_type": item["proposed_fact_type"], "epistemic_status": item["epistemic_status"], "epistemic_status_confidence": item["epistemic_status_confidence"]} for item in evidence.get("evidence_units", [])]
    result = client.call(
        case_id=case.case_id, stage="source_neutral", system_prompt=load_prompt(root, prompt_name),
        user_payload={"case_id": case.case_id, "master_language": case.source_language, "verified_evidence_untrusted": minimal},
        schema=NEUTRAL_SCHEMA, required_fields=("case_id", "master_neutral_text", "fact_units", "removed_legal_signals", "removed_jurisdiction_signals", "anonymization_warnings", "grounding_warnings", "insufficient_factual_detail"), prompt_version=prompt_name.removesuffix(".txt"),
    )
    payload, finalized = _finalize_payload(result.payload, evidence, case.source_text, case.source_language)
    checks, conversions, master_rebuilt = finalized["checks"], finalized["conversions"], finalized["rebuilt"]
    model_insufficient = bool(payload.get("insufficient_factual_detail"))
    coverage_complete = bool((evidence.get("source_coverage") or {}).get("coverage_complete", False))
    hard_failure = checks["status"] == "fail" or evidence.get("extraction_status") == "fail" or not coverage_complete
    status = "fail" if hard_failure else "warning" if checks["status"] == "warning" or model_insufficient else "pass"
    usable_for_translation = not hard_failure and checks["deterministic_factual_sufficiency"] in {"pass", "warning"}
    return {
        "dataset_version": DATASET_VERSION, "case_id": case.case_id, "case_origin": case.case_origin,
        "case_subtype": case.case_subtype, "source_language": case.source_language, "master_language": case.source_language,
        "source_text_field": case.source_text_field, "source_text_sha256": case.source_text_sha256,
        "source_dataset": case.source_dataset, "source_record_id": case.source_record_id, "input_file_sha256": case.input_file_sha256,
        "case_is_usable": usable_for_translation, "case_is_usable_for_translation": usable_for_translation, "case_is_finally_usable": None,
        "exclusion_reasons": checks["errors"] + ([] if coverage_complete else ["source_coverage_incomplete"]),
        "neutralization_status": status, "master_neutral_text": payload.get("master_neutral_text", ""), "fact_units": payload.get("fact_units", []),
        "removed_legal_signals": payload.get("removed_legal_signals", []), "removed_jurisdiction_signals": payload.get("removed_jurisdiction_signals", []),
        "anonymization_warnings": payload.get("anonymization_warnings", []), "grounding_warnings": payload.get("grounding_warnings", []) + checks["warnings"] + (["model_insufficient_factual_detail"] if model_insufficient else []) + (["master_text_rebuilt_from_fact_units"] if master_rebuilt else []),
        "memorization_risk": checks["memorization_risk"], "insufficient_factual_detail": model_insufficient, "model_insufficient_factual_detail": model_insufficient,
        "deterministic_factual_sufficiency": checks["deterministic_factual_sufficiency"], "source_coverage": evidence.get("source_coverage", {}),
        "deterministic_checks": checks, "unit_conversions": conversions, "model_provenance": result.provenance,
    }


def recheck_neutral_record(record: dict[str, Any], evidence: dict[str, Any], source_text: str) -> dict[str, Any]:
    payload = {key: record.get(key) for key in ("master_neutral_text", "fact_units", "removed_legal_signals", "removed_jurisdiction_signals", "anonymization_warnings", "grounding_warnings", "insufficient_factual_detail")}
    finalized_payload, finalized = _finalize_payload(payload, evidence, source_text, str(record.get("source_language") or "ko"))
    checks = finalized["checks"]; model_insufficient = bool(record.get("model_insufficient_factual_detail", record.get("insufficient_factual_detail")))
    coverage_complete = bool((evidence.get("source_coverage") or {}).get("coverage_complete", False))
    hard_failure = checks["status"] == "fail" or evidence.get("extraction_status") == "fail" or not coverage_complete
    updated = dict(record); updated.update({
        "master_neutral_text": finalized_payload["master_neutral_text"], "fact_units": finalized_payload["fact_units"],
        "neutralization_status": "fail" if hard_failure else "warning" if checks["status"] == "warning" or model_insufficient else "pass",
        "model_insufficient_factual_detail": model_insufficient, "deterministic_factual_sufficiency": checks["deterministic_factual_sufficiency"],
        "case_is_usable_for_translation": not hard_failure and checks["deterministic_factual_sufficiency"] in {"pass", "warning"}, "case_is_finally_usable": None,
        "source_coverage": evidence.get("source_coverage", {}), "deterministic_checks": checks, "unit_conversions": finalized["conversions"],
    }); updated["case_is_usable"] = updated["case_is_usable_for_translation"]
    return updated
