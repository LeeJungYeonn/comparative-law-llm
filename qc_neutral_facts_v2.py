from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import guard_outputs, read_jsonl, write_csv, write_jsonl
from pipeline_v2.llm_runtime import DEFAULT_API_KEY_ENV, DEFAULT_LETSUR_BASE_URL, call_structured
from pipeline_v2.rules import assess_fact_sufficiency, leakage_checks, translation_equivalence_checks

PROMPT_VERSION = "neutral-fact-qc-v2.2"
QC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "findings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "problem_type": {"type": "string", "enum": ["unsupported_fact", "legal_conclusion", "jurisdiction_signal", "translation_loss_or_addition", "epistemic_status_change", "fact_insufficiency"]},
            "fact_id": {"type": ["string", "null"]}, "evidence_span": {"type": "string"},
            "explanation": {"type": "string"}, "proposed_correction": {"type": ["string", "null"]},
        }, "required": ["problem_type", "fact_id", "evidence_span", "explanation", "proposed_correction"]},
        },
        "sufficient_for_independent_analysis": {"type": "boolean"},
    },
    "required": ["findings", "sufficient_for_independent_analysis"],
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run deterministic and optional secondary LLM QC on bilingual neutral facts.")
    result.add_argument("--input", type=Path, default=Path("outputs_v2/neutral_facts_bilingual.jsonl"))
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--llm-model", help="Optional separate QC model; otherwise only deterministic QC runs.")
    result.add_argument("--base-url", default=DEFAULT_LETSUR_BASE_URL)
    result.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    result.add_argument("--dotenv-path", type=Path)
    result.add_argument("--prompt", type=Path, default=Path("prompts_v2/qc_neutral_facts_v2.txt"))
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--concurrency", type=int, default=4)
    result.add_argument("--llm-warnings-only", action="store_true", help="Run secondary LLM QC only for records with deterministic translation warnings.")
    result.add_argument("--retry-review", action="store_true", help="With --resume, rerun prior LLM QC records that had findings or were insufficient.")
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def deterministic_qc(record: dict[str, Any]) -> dict[str, Any]:
    source_language = record["source_language"]
    source = record["neutral_fact_source"]
    translated = record["neutral_fact_en"] if source_language == "ko" else record["neutral_fact_ko"]
    ko_leak, en_leak = leakage_checks(record["neutral_fact_ko"]), leakage_checks(record["neutral_fact_en"])
    aligned = record.get("aligned_fact_units") or []
    source_units = record.get("source_fact_units") or []
    expected_ids = [unit["fact_id"] for unit in source_units if unit.get("include_in_neutral_fact")]
    actual_ids = [unit.get("fact_id") for unit in aligned]
    id_status = "pass" if expected_ids == actual_ids and len(actual_ids) == len(set(actual_ids)) else "fail"
    grounding = "pass" if source_units and all(unit.get("source_grounding_status") == "pass" for unit in source_units if unit.get("include_in_neutral_fact")) else "fail"
    sufficiency = assess_fact_sufficiency(source)
    fact_type_dimensions = {
        "parties": "fact_has_parties", "conduct": "fact_has_conduct", "context": "fact_has_context",
        "timeline": "fact_has_timeline", "harm": "fact_has_harm", "causation": "fact_has_causation",
        "defense_context": "fact_has_defense_context",
    }
    for unit in source_units:
        if unit.get("include_in_neutral_fact") and unit.get("fact_type") in fact_type_dimensions:
            sufficiency[fact_type_dimensions[unit["fact_type"]]] = True
    mandatory = ("fact_has_parties", "fact_has_conduct", "fact_has_harm", "fact_has_causation")
    sufficiency["fact_has_causal_sequence"] = sufficiency["fact_has_causation"]
    sufficiency["mandatory_fact_dimensions"] = {key: sufficiency[key] for key in mandatory}
    sufficiency["missing_mandatory_fact_dimensions"] = [key for key in mandatory if not sufficiency[key]]
    sufficiency["core_fact_sufficient"] = not sufficiency["missing_mandatory_fact_dimensions"]
    optional = ("fact_has_context", "fact_has_timeline", "fact_has_defense_context")
    sufficiency["fact_sufficiency_score"] = sum(bool(sufficiency[key]) for key in (*mandatory, *optional))
    sufficiency["preferred_fact_sufficiency"] = sufficiency["core_fact_sufficient"] and sufficiency["fact_sufficiency_score"] >= 5
    sufficiency["factual_background_sufficient"] = sufficiency["core_fact_sufficient"]
    legal = "pass" if ko_leak["legal_leakage_status"] == en_leak["legal_leakage_status"] == "pass" else "fail"
    jurisdiction = "pass" if ko_leak["jurisdiction_leakage_status"] == en_leak["jurisdiction_leakage_status"] == "pass" else "fail"
    translation = "pass" if record.get("translation_equivalence_status") == "pass" and id_status == "pass" and all(unit.get("translation_equivalence_status") == "pass" for unit in aligned) else "fail"
    issues = []
    if grounding != "pass": issues.append("source_grounding")
    if legal != "pass": issues.append("legal_leakage")
    if jurisdiction != "pass": issues.append("jurisdiction_leakage")
    if translation != "pass": issues.append("translation_equivalence")
    if not sufficiency["factual_background_sufficient"]: issues.append("fact_sufficiency")
    return {
        "case_id": record["case_id"], "origin_country": record.get("origin_country"), "origin_state": record.get("origin_state"),
        "primary_domain": record.get("primary_domain") or record.get("case_domain"),
        "case_domain": record.get("primary_domain") or record.get("case_domain"), "source_grounding_status": grounding,
        "legal_leakage_status": legal, "legal_leakage_evidence": {"ko": ko_leak["legal_leakage_evidence"], "en": en_leak["legal_leakage_evidence"]},
        "jurisdiction_leakage_status": jurisdiction, "jurisdiction_leakage_evidence": {"ko": ko_leak["jurisdiction_leakage_evidence"], "en": en_leak["jurisdiction_leakage_evidence"]},
        "translation_equivalence_status": translation, "translation_equivalence_issues": record.get("translation_equivalence_issues", []),
        "translation_equivalence_warnings": record.get("translation_equivalence_warnings", []),
        "fact_id_alignment_status": id_status, **sufficiency, "deterministic_qc_issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    qc_path = args.output_dir / "neutral_fact_qc.csv"
    failures_path = args.output_dir / "neutral_fact_qc_failures.jsonl"
    llm_path = args.output_dir / "neutral_fact_llm_qc.jsonl"
    guard_outputs((qc_path, llm_path), overwrite=args.overwrite, resume=args.resume)
    records = list(read_jsonl(args.input))
    if args.limit:
        records = records[:args.limit]
    deterministic = {record["case_id"]: deterministic_qc(record) for record in records}
    llm_candidates = [
        record for record in records
        if args.llm_model and (not args.llm_warnings_only or deterministic[record["case_id"]]["translation_equivalence_warnings"])
    ]
    print(json.dumps({"stage": "qc", "cases": len(records), "deterministic": True, "llm_model": args.llm_model, "api_calls": len(llm_candidates) if not args.dry_run else 0}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    prompt = args.prompt.read_text(encoding="utf-8")
    existing_llm = {row["case_id"]: row for row in read_jsonl(llm_path)} if args.resume and llm_path.exists() else {}
    if args.retry_review:
        existing_llm = {
            case_id: row for case_id, row in existing_llm.items()
            if not row.get("findings") and row.get("sufficient_for_independent_analysis")
        }
    def run_llm(record: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "case_id": record["case_id"], "source_fact_units": record.get("source_fact_units"),
            "aligned_fact_units": record.get("aligned_fact_units"), "neutral_fact_ko": record["neutral_fact_ko"], "neutral_fact_en": record["neutral_fact_en"],
        }
        parsed, provenance = call_structured(
            case_id=record["case_id"], stage="qc", prompt_version=PROMPT_VERSION, model=args.llm_model,
            system_prompt=prompt, user_payload=payload, schema_name="neutral_fact_qc", schema=QC_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        return {"case_id": record["case_id"], **parsed, "qc_provenance": provenance}
    pending_llm = [record for record in llm_candidates if record["case_id"] not in existing_llm]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(run_llm, record): record for record in pending_llm}
        for future in as_completed(futures):
            record = futures[future]
            existing_llm[record["case_id"]] = future.result()
    qc_rows = []
    # QC is recomputed for every input record, so stale failures must never carry across resume runs.
    failures = []
    for record in records:
        qc = deterministic[record["case_id"]]
        llm_result = existing_llm.get(record["case_id"])
        qc["llm_qc_status"] = "not_run" if not llm_result else ("pass" if not llm_result.get("findings") and llm_result.get("sufficient_for_independent_analysis") else "review")
        qc["llm_qc_findings"] = [] if not llm_result else llm_result.get("findings", [])
        unresolved_translation_warning = bool(qc["translation_equivalence_warnings"] and qc["llm_qc_status"] != "pass")
        qc["manual_review_recommended"] = bool(qc["deterministic_qc_issues"] or unresolved_translation_warning or qc["llm_qc_status"] == "review" or record.get("unit_normalization_status") == "review" or record.get("institution_neutralization_status") == "review")
        qc["final_eligible"] = not qc["deterministic_qc_issues"] and qc["llm_qc_status"] in {"pass", "not_run"}
        qc_rows.append(qc)
        if not qc["final_eligible"]:
            failures.append({"case_id": record["case_id"], "stage": "post_translation_qc", "deterministic_qc_issues": qc["deterministic_qc_issues"], "llm_qc_findings": qc["llm_qc_findings"]})
    write_csv(qc_path, qc_rows)
    write_jsonl(llm_path, [existing_llm[row["case_id"]] for row in records if row["case_id"] in existing_llm])
    write_jsonl(failures_path, failures)
    print(json.dumps({"qc_pass": sum(row["final_eligible"] for row in qc_rows), "qc_fail": sum(not row["final_eligible"] for row in qc_rows), "manual_review_recommended": sum(row["manual_review_recommended"] for row in qc_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
