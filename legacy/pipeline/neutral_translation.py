from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.factual_evidence import load_prompt
from pipeline.leakage_checks import PLACEHOLDER_RE, translation_checks
from pipeline.llm_client import LLMClient
from pipeline.stage2_schema import TRANSLATION_SCHEMA


def translate(master: dict[str, Any], client: LLMClient, root: Path) -> dict[str, Any]:
    origin = master["case_origin"]
    prompt_name = "translate_ko_to_en_v2.txt" if origin == "KR" else "translate_en_to_ko_v3.txt"
    direction = "ko_to_en" if origin == "KR" else "en_to_ko"
    target = "en" if origin == "KR" else "ko"
    master_text = str(master.get("master_neutral_text") or "")
    # Deliberately excludes source text, evidence, subtype, case/court/date metadata.
    request = {
        "case_id": master["case_id"], "master_language": master["master_language"], "translation_direction": direction,
        "master_neutral_text": master_text,
        "fact_units": [{"fact_id": unit["fact_id"], "master_text": unit["master_text"], "epistemic_status": unit["epistemic_status"]} for unit in master.get("fact_units", [])],
        "placeholders": sorted(set(PLACEHOLDER_RE.findall(master_text))),
    }
    result = client.call(case_id=master["case_id"], stage="translation", system_prompt=load_prompt(root, prompt_name), user_payload=request, schema=TRANSLATION_SCHEMA, required_fields=("case_id", "translated_neutral_text", "translated_fact_units"), prompt_version=prompt_name.removesuffix(".txt"))
    translated_payload = dict(result.payload)
    model_translated_text = str(translated_payload.get("translated_neutral_text") or "")
    composed_translated_text = " ".join(str(unit.get("translated_text") or "").strip() for unit in translated_payload.get("translated_fact_units", []) if str(unit.get("translated_text") or "").strip())
    text_rebuilt = bool(composed_translated_text and composed_translated_text != model_translated_text)
    translated_payload["translated_neutral_text"] = composed_translated_text or model_translated_text
    checks = translation_checks(master, translated_payload, target)
    return {
        "case_id": master["case_id"], "case_origin": origin, "master_language": master["master_language"], "translation_direction": direction,
        "master_neutral_text": master_text, "translated_neutral_text": translated_payload.get("translated_neutral_text", ""),
        "translated_fact_units": translated_payload.get("translated_fact_units", []), "translation_status": checks["status"],
        "translation_warnings": checks["warnings"] + checks["errors"] + (["translated_text_rebuilt_from_fact_units"] if text_rebuilt else []), "placeholder_match": checks["placeholder_match"],
        "fact_id_match": checks["fact_id_match"], "number_unit_match": checks["number_unit_match"],
        "deterministic_checks": {**checks, "translated_text_rebuilt_from_fact_units": text_rebuilt}, "model_provenance": result.provenance,
    }


def recheck_translation_record(master: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    target = "en" if master["case_origin"] == "KR" else "ko"
    payload = {"translated_neutral_text": record.get("translated_neutral_text", ""), "translated_fact_units": record.get("translated_fact_units", [])}
    composed = " ".join(str(unit.get("translated_text") or "").strip() for unit in payload["translated_fact_units"] if str(unit.get("translated_text") or "").strip())
    rebuilt = bool(composed and composed != payload["translated_neutral_text"])
    if composed: payload["translated_neutral_text"] = composed
    checks = translation_checks(master, payload, target)
    updated = dict(record); updated.update({
        "master_neutral_text": master.get("master_neutral_text", ""), "translated_neutral_text": payload["translated_neutral_text"],
        "translation_status": checks["status"], "translation_warnings": checks["warnings"] + checks["errors"] + (["translated_text_rebuilt_from_fact_units"] if rebuilt else []),
        "placeholder_match": checks["placeholder_match"], "fact_id_match": checks["fact_id_match"], "number_unit_match": checks["number_unit_match"],
        "deterministic_checks": {**checks, "translated_text_rebuilt_from_fact_units": rebuilt},
    })
    return updated
