from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import guard_outputs, read_jsonl, sha256_text, write_jsonl
from pipeline_v2.llm_runtime import call_structured, configured_model, load_mock
from pipeline_v2.rules import leakage_checks, source_span_grounding

PROMPT_VERSION = "neutral-fact-units-v2.1"
FACT_TYPES = ["parties", "conduct", "context", "timeline", "harm", "causation", "defense_context", "other"]
EPISTEMIC = ["established_record_fact", "party_allegation", "testimony", "disputed_fact"]
EXTRACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "fact_units": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "fact_id": {"type": "string", "pattern": "^F[0-9]{3}$"}, "text": {"type": "string"},
            "source_level": {"type": "string", "enum": ["highest_court", "lower_court"]}, "source_case_id": {"type": "string"},
            "source_span": {"type": "string"}, "fact_type": {"type": "string", "enum": FACT_TYPES},
            "epistemic_status": {"type": "string", "enum": EPISTEMIC}, "include_in_neutral_fact": {"type": "boolean"},
            "exclusion_reason": {"type": ["string", "null"]},
        }, "required": ["fact_id", "text", "source_level", "source_case_id", "source_span", "fact_type", "epistemic_status", "include_in_neutral_fact", "exclusion_reason"]}},
        "entity_mappings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"source_entity": {"type": "string"}, "placeholder": {"type": "string"}}, "required": ["source_entity", "placeholder"]}},
        "normalizations": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"kind": {"type": "string", "enum": ["measurement", "currency", "institution"]}, "original": {"type": "string"}, "normalized": {"type": "string"}, "status": {"type": "string", "enum": ["applied", "not_needed", "review"]}}, "required": ["kind", "original", "normalized", "status"]}},
    },
    "required": ["fact_units", "entity_mappings", "normalizations"],
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Extract source-grounded, atomic neutral fact units using the OpenAI Responses API.")
    result.add_argument("--input", type=Path, action="append", required=True)
    result.add_argument("--model")
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--prompt", type=Path, default=Path("prompts_v2/extract_neutral_facts_v2.txt"))
    result.add_argument("--concurrency", type=int, default=2)
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--max-source-chars", type=int, default=120000)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--include-ineligible", action="store_true")
    result.add_argument("--mock-response-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def validate_parsed(case: dict[str, Any], parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    highest = case.get("main_opinion_text") or ""
    lower = case.get("lower_court_fact_text") or ""
    units = parsed.get("fact_units") or []
    issues: list[str] = []
    ids = [unit.get("fact_id") for unit in units]
    if len(ids) != len(set(ids)):
        issues.append("duplicate_fact_ids")
    if ids != [f"F{index:03d}" for index in range(1, len(ids) + 1)]:
        issues.append("nonsequential_fact_ids")
    for unit in units:
        source = highest if unit.get("source_level") == "highest_court" else lower
        expected_id = case["case_id"] if unit.get("source_level") == "highest_court" else unit.get("source_case_id")
        if unit.get("source_level") == "highest_court" and unit.get("source_case_id") != case["case_id"]:
            issues.append(f"{unit.get('fact_id')}:wrong_source_case_id")
        status, start, end = source_span_grounding(source, unit.get("source_span") or "")
        unit["source_grounding_status"] = status
        unit["source_start"] = start
        unit["source_end"] = end
        if status != "pass":
            issues.append(f"{unit.get('fact_id')}:source_span_not_found")
    if not any(unit.get("include_in_neutral_fact") for unit in units):
        issues.append("no_included_fact_units")
    return units, issues


def process_case(case: dict[str, Any], args: argparse.Namespace, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    source_language = "ko" if case.get("origin_country") == "KR" else "en"
    user_payload = {
        "case_id": case["case_id"], "source_language": source_language,
        "highest_court_source": (case.get("main_opinion_text") or "")[:args.max_source_chars],
        "lower_court_supplement": (case.get("lower_court_fact_text") or "")[:args.max_source_chars],
        "lower_court_case_ids": case.get("lower_court_case_ids") or [],
    }
    if args.mock_response_dir:
        parsed = load_mock(args.mock_response_dir, "extract", case["case_id"])
        if parsed is None:
            raise RuntimeError(f"No extraction mock found for {case['case_id']}")
        provenance = {"model": "mock", "model_snapshot_or_returned_model_id": "mock", "prompt_version": PROMPT_VERSION, "request_id": None, "timestamp": datetime.now(timezone.utc).isoformat(), "usage": None, "input_hash": sha256_text(json.dumps(user_payload, ensure_ascii=False, sort_keys=True)), "output_hash": sha256_text(json.dumps(parsed, ensure_ascii=False, sort_keys=True)), "status": "success"}
    else:
        parsed, provenance = call_structured(
            case_id=case["case_id"], stage="extract", prompt_version=PROMPT_VERSION, model=model,
            system_prompt=prompt, user_payload=user_payload, schema_name="neutral_fact_units", schema=EXTRACTION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
        )
    units, issues = validate_parsed(case, parsed)
    included = [unit["text"].strip() for unit in units if unit.get("include_in_neutral_fact")]
    master = " ".join(value for value in included if value)
    leaks = leakage_checks(master)
    grounding_status = "pass" if not any("source_span" in issue for issue in issues) else "fail"
    record = {
        "case_id": case["case_id"], "case_family_id": case.get("case_family_id"), "origin_country": case.get("origin_country"),
        "origin_state": case.get("origin_state"), "case_domain": case.get("case_domain"), "source_language": source_language,
        "highest_court_case_id": case.get("highest_court_case_id", case["case_id"]),
        "lower_court_supplemented": case.get("lower_court_supplemented", False),
        "lower_court_case_ids": case.get("lower_court_case_ids", []),
        "lower_court_link_confidence": case.get("lower_court_link_confidence", "none"),
        "neutral_fact_source": master, "neutral_fact_ko": master if source_language == "ko" else "",
        "neutral_fact_en": master if source_language == "en" else "", "fact_units": units,
        "source_grounding_status": grounding_status, **leaks,
        "unit_normalization_status": "review" if any(item.get("status") == "review" for item in parsed.get("normalizations", [])) else "pass",
        "institution_neutralization_status": "review" if any(item.get("kind") == "institution" and item.get("status") == "review" for item in parsed.get("normalizations", [])) else "pass",
        "normalizations": parsed.get("normalizations", []), "extraction_validation_issues": issues,
        "extraction_provenance": provenance,
    }
    private = {"case_id": case["case_id"], "entity_mappings": parsed.get("entity_mappings", [])}
    return ("success" if grounding_status == "pass" and master else "failure"), {"record": record, "private": private}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source_path = args.output_dir / "neutral_facts_source.jsonl"
    units_path = args.output_dir / "neutral_fact_units_source.jsonl"
    private_path = args.output_dir / "neutral_fact_entity_mappings_private.jsonl"
    failures_path = args.output_dir / "neutral_fact_failures.jsonl"
    guard_outputs((source_path, units_path, private_path, failures_path), overwrite=args.overwrite, resume=args.resume)
    cases = [row for path in args.input for row in read_jsonl(path) if args.include_ineligible or row.get("strict_source_eligible") is True]
    if args.limit:
        cases = cases[:args.limit]
    prompt = args.prompt.read_text(encoding="utf-8")
    model = "mock" if args.mock_response_dir else (args.model or __import__("os").getenv("FACT_EXTRACTION_MODEL") or "UNCONFIGURED")
    plan = {"stage": "extract", "cases": len(cases), "model": model, "prompt_version": PROMPT_VERSION, "api_calls": 0 if args.dry_run or args.mock_response_dir else len(cases)}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    model = "mock" if args.mock_response_dir else configured_model(args.model)
    completed = {row["case_id"]: row for row in read_jsonl(source_path)} if args.resume and source_path.exists() else {}
    private = {row["case_id"]: row for row in read_jsonl(private_path)} if args.resume and private_path.exists() else {}
    failures = list(read_jsonl(failures_path)) if args.resume and failures_path.exists() else []
    pending = [case for case in cases if case["case_id"] not in completed]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(process_case, case, args, model, prompt): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            try:
                status, payload = future.result()
                if status == "success":
                    completed[case["case_id"]] = payload["record"]
                    private[case["case_id"]] = payload["private"]
                else:
                    failures.append({"case_id": case["case_id"], "stage": "extract_validation", "issues": payload["record"]["extraction_validation_issues"], "record": payload["record"]})
            except Exception as exc:
                failures.append({"case_id": case["case_id"], "stage": "extract_api", "error_type": type(exc).__name__, "error": str(exc)})
    ordered = [completed[case["case_id"]] for case in cases if case["case_id"] in completed]
    write_jsonl(source_path, ordered)
    write_jsonl(units_path, [{"case_id": row["case_id"], **unit} for row in ordered for unit in row["fact_units"]])
    write_jsonl(private_path, [private[case["case_id"]] for case in cases if case["case_id"] in private])
    write_jsonl(failures_path, failures)
    print(json.dumps({"extraction_pass": len(ordered), "failures": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
