from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.factual_evidence import load_prompt
from pipeline.leakage_checks import PLACEHOLDER_RE
from pipeline.llm_client import LLMClient
from pipeline.stage2_schema import GROUNDING_SCHEMA, TRANSLATION_VERIFIER_SCHEMA


def validate_grounding_verifier(payload: dict[str, Any]) -> dict[str, Any]:
    model_status = str(payload.get("verifier_status") or "fail")
    fail_fields = ("unsupported_fact_ids", "overstated_fact_ids", "epistemic_status_errors", "legal_conclusion_leakage", "jurisdiction_leakage")
    hard_reasons = [field for field in fail_fields if payload.get(field)]
    if payload.get("grounded") is not True: hard_reasons.append("grounded_false")
    warnings = ["missing_material_facts"] if payload.get("missing_material_facts") else []
    validated = "fail" if hard_reasons else "warning" if warnings or model_status == "warning" else "fail" if model_status == "fail" else "pass"
    return {"model_verifier_status": model_status, "validated_verifier_status": validated, "verifier_consistency_violation": model_status == "pass" and validated != "pass", "deterministic_verifier_reasons": hard_reasons + warnings}


def validate_translation_verifier(payload: dict[str, Any]) -> dict[str, Any]:
    model_status = str(payload.get("translation_status") or "fail")
    fail_fields = ("missing_fact_ids", "added_information", "omitted_information", "changed_negation", "changed_temporal_relation", "changed_epistemic_status", "legal_term_reintroduction", "placeholder_errors")
    hard_reasons = [field for field in fail_fields if payload.get(field)]
    if payload.get("meaning_preserved") is not True: hard_reasons.append("meaning_preserved_false")
    validated = "fail" if hard_reasons or model_status == "fail" else "warning" if model_status == "warning" else "pass"
    return {"model_verifier_status": model_status, "validated_verifier_status": validated, "verifier_consistency_violation": model_status == "pass" and validated != "pass", "deterministic_verifier_reasons": hard_reasons}


def verify_grounding(master: dict[str, Any], evidence: dict[str, Any], client: LLMClient, root: Path) -> dict[str, Any]:
    prompt_name = "verify_grounding_ko_v1.txt" if master["case_origin"] == "KR" else "verify_grounding_en_v1.txt"
    minimal_evidence = [{"evidence_id": unit["evidence_id"], "exact_excerpts_untrusted": unit["exact_excerpts"], "epistemic_status": unit["epistemic_status"]} for unit in evidence.get("evidence_units", [])]
    request = {"case_id": master["case_id"], "verified_evidence_untrusted": minimal_evidence, "fact_units": master.get("fact_units", []), "master_neutral_text": master.get("master_neutral_text", "")}
    result = client.call(case_id=master["case_id"], stage="grounding_verifier", system_prompt=load_prompt(root, prompt_name), user_payload=request, schema=GROUNDING_SCHEMA, required_fields=tuple(GROUNDING_SCHEMA["required"]), prompt_version=prompt_name.removesuffix(".txt"))
    consistency = validate_grounding_verifier(result.payload)
    return {**result.payload, **consistency, "verifier_status": consistency["validated_verifier_status"], "verification_type": "source_grounding", "model_provenance": result.provenance}


def verify_translation(master: dict[str, Any], translated: dict[str, Any], client: LLMClient, root: Path) -> dict[str, Any]:
    prompt_name = "verify_translation_ko_en_v1.txt" if master["case_origin"] == "KR" else "verify_translation_en_ko_v1.txt"
    request = {"case_id": master["case_id"], "master_fact_units": [{"fact_id": unit["fact_id"], "text": unit["master_text"], "epistemic_status": unit["epistemic_status"]} for unit in master.get("fact_units", [])], "translated_fact_units": translated.get("translated_fact_units", []), "placeholders": sorted(set(PLACEHOLDER_RE.findall(str(master.get("master_neutral_text") or ""))))}
    result = client.call(case_id=master["case_id"], stage="translation_verifier", system_prompt=load_prompt(root, prompt_name), user_payload=request, schema=TRANSLATION_VERIFIER_SCHEMA, required_fields=tuple(TRANSLATION_VERIFIER_SCHEMA["required"]), prompt_version=prompt_name.removesuffix(".txt"))
    consistency = validate_translation_verifier(result.payload)
    if translated.get("translation_status") == "fail" and consistency["validated_verifier_status"] == "pass":
        consistency["validated_verifier_status"] = "fail"; consistency["verifier_consistency_violation"] = True; consistency["deterministic_verifier_reasons"].append("deterministic_translation_failure_not_acknowledged")
    return {**result.payload, **consistency, "translation_status": consistency["validated_verifier_status"], "verification_type": "translation", "model_provenance": result.provenance}


def recheck_grounding_verifier_record(record: dict[str, Any]) -> dict[str, Any]:
    consistency = validate_grounding_verifier({**record, "verifier_status": record.get("model_verifier_status", record.get("verifier_status"))})
    return {**record, **consistency, "verifier_status": consistency["validated_verifier_status"]}


def recheck_translation_verifier_record(record: dict[str, Any], translated: dict[str, Any] | None = None) -> dict[str, Any]:
    consistency = validate_translation_verifier({**record, "translation_status": record.get("model_verifier_status", record.get("translation_status"))})
    if translated and translated.get("translation_status") == "fail" and consistency["validated_verifier_status"] == "pass":
        consistency["validated_verifier_status"] = "fail"; consistency["verifier_consistency_violation"] = True; consistency["deterministic_verifier_reasons"].append("deterministic_translation_failure_not_acknowledged")
    return {**record, **consistency, "translation_status": consistency["validated_verifier_status"]}
