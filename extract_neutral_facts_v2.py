from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import guard_outputs, read_jsonl, sha256_text, write_jsonl
from pipeline_v2.llm_runtime import DEFAULT_API_KEY_ENV, DEFAULT_LETSUR_BASE_URL, call_structured, configured_model, load_mock
from pipeline_v2.rules import leakage_checks, neutralize_jurisdiction_signals, source_span_grounding
from pipeline_v2.v3_rules import script_language_sanity

PROMPT_VERSION = "neutral-fact-units-v2.7"
FACT_TYPES = ["parties", "conduct", "context", "timeline", "harm", "causation", "defense_context", "other"]
EPISTEMIC = ["established_record_fact", "party_allegation", "testimony", "disputed_fact"]
EXTRACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "fact_units": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "fact_id": {"type": "string", "pattern": "^F[0-9]{3}$"}, "text": {"type": "string"},
            "source_level": {"type": "string", "enum": ["highest_court", "lower_court"]}, "source_case_id": {"type": "string"},
            "source_span_id": {"type": "string", "pattern": "^[HL][0-9]{4}$"}, "source_span": {"type": "string"}, "fact_type": {"type": "string", "enum": FACT_TYPES},
            "epistemic_status": {"type": "string", "enum": EPISTEMIC}, "include_in_neutral_fact": {"type": "boolean"},
            "exclusion_reason": {"type": ["string", "null"]},
        }, "required": ["fact_id", "text", "source_level", "source_case_id", "source_span_id", "source_span", "fact_type", "epistemic_status", "include_in_neutral_fact", "exclusion_reason"]}},
        "entity_mappings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"source_entity": {"type": "string"}, "placeholder": {"type": "string"}}, "required": ["source_entity", "placeholder"]}},
        "normalizations": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"kind": {"type": "string", "enum": ["measurement", "currency", "institution"]}, "original": {"type": "string"}, "normalized": {"type": "string"}, "status": {"type": "string", "enum": ["applied", "not_needed", "review"]}}, "required": ["kind", "original", "normalized", "status"]}},
    },
    "required": ["fact_units", "entity_mappings", "normalizations"],
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Extract source-grounded atomic neutral fact units through the configured Letsur gateway.")
    result.add_argument("--input", type=Path, action="append", required=True)
    result.add_argument("--model")
    result.add_argument("--base-url", default=DEFAULT_LETSUR_BASE_URL)
    result.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    result.add_argument("--dotenv-path", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--prompt", type=Path, default=Path("prompts_v2/extract_neutral_facts_v2.txt"))
    result.add_argument("--prompt-version", default=PROMPT_VERSION)
    result.add_argument("--concurrency", type=int, default=2)
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--max-source-chars", type=int, default=120000)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--limit-per-country", type=int, default=0, help="Select this many eligible KR and US cases for balanced smoke tests.")
    result.add_argument("--include-ineligible", action="store_true")
    result.add_argument("--mock-response-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--revalidate-existing", action="store_true", help="Re-run deterministic validation on saved extraction records without API calls.")
    result.add_argument("--exclude-from-llm-qc", type=Path, help="QC CSV whose legal/jurisdiction fact IDs are excluded during revalidation.")
    result.add_argument("--semantic-repair-file", type=Path, help="Semantic QC JSONL; re-extract only hard/manual failures with findings supplied as repair constraints.")
    return result


def build_passages(source: str, prefix: str, max_chars: int, passage_chars: int = 700) -> list[dict[str, Any]]:
    source = source[:max_chars]
    passages = []
    start = 0
    while start < len(source):
        hard_end = min(len(source), start + passage_chars)
        end = hard_end
        if hard_end < len(source):
            window = source[start + 200:hard_end]
            boundaries = [match.end() for match in re.finditer(r"(?:\n+|(?<=[.!?。！？])\s+)", window)]
            if boundaries:
                end = start + 200 + boundaries[-1]
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        if end <= start:
            start = hard_end
            continue
        passages.append({"span_id": f"{prefix}{len(passages) + 1:04d}", "text": source[start:end], "source_start": start, "source_end": end})
        start = max(end, hard_end if end == 0 else end)
    return passages


def validate_parsed(case: dict[str, Any], parsed: dict[str, Any], passage_map: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
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
        if unit.get("source_level") == "highest_court" and unit.get("source_case_id") != case["case_id"]:
            issues.append(f"{unit.get('fact_id')}:wrong_source_case_id")
        passage = passage_map.get(unit.get("source_span_id") or "")
        expected_prefix = "H" if unit.get("source_level") == "highest_court" else "L"
        if not passage or not str(unit.get("source_span_id", "")).startswith(expected_prefix):
            unit["source_grounding_status"] = "fail"
            unit["source_start"] = unit["source_end"] = None
            issues.append(f"{unit.get('fact_id')}:invalid_source_span_id")
            continue
        model_span = unit.get("source_span") or ""
        unit["model_source_span"] = model_span
        unit["source_span_copy_status"] = "exact" if model_span == passage["text"] else "canonicalized_from_source_span_id"
        unit["source_span"] = passage["text"]
        unit["source_start"] = passage["source_start"]
        unit["source_end"] = passage["source_end"]
        source = highest if unit.get("source_level") == "highest_court" else lower
        unit["source_grounding_status"] = source_span_grounding(source, passage["text"])[0]
        if unit["source_grounding_status"] != "pass":
            issues.append(f"{unit.get('fact_id')}:canonical_passage_not_found")
    if not any(unit.get("include_in_neutral_fact") for unit in units):
        issues.append("no_included_fact_units")
    included_types = {unit.get("fact_type") for unit in units if unit.get("include_in_neutral_fact")}
    for required_type in ("parties", "conduct", "harm", "causation"):
        if required_type not in included_types:
            issues.append(f"missing_core_fact_type:{required_type}")
    return units, issues


def normalize_neutral_units(units: list[dict[str, Any]], issues: list[str]) -> list[str]:
    issues = [issue for issue in issues if not issue.startswith("missing_core_fact_type:")]
    for unit in units:
        original_text = unit.get("text") or ""
        neutral_text, jurisdiction_evidence = neutralize_jurisdiction_signals(original_text)
        if neutral_text != original_text:
            unit.setdefault("text_before_jurisdiction_neutralization", original_text)
            unit["jurisdiction_neutralization_evidence"] = jurisdiction_evidence
            unit["text"] = neutral_text
        if unit.get("include_in_neutral_fact") and leakage_checks(unit.get("text") or "")["legal_leakage_status"] != "pass":
            unit["include_in_neutral_fact"] = False
            unit["exclusion_reason"] = "legal_or_procedural_content"
    included_types = {unit.get("fact_type") for unit in units if unit.get("include_in_neutral_fact")}
    for required_type in ("parties", "conduct", "harm", "causation"):
        if required_type not in included_types:
            issues.append(f"missing_core_fact_type:{required_type}")
    return issues


def ensure_epistemic_frame(unit: dict[str, Any], source_language: str) -> None:
    text = unit.get("text") or ""
    status = unit.get("epistemic_status")
    if status == "testimony" and not re.search(r"증언|진술|testif|according to testimony", text, re.I):
        unit["text"] = ("증언에 따르면, " if source_language == "ko" else "According to testimony, ") + text
    elif status == "disputed_fact" and not re.search(r"다투|분쟁|disput|contest", text, re.I):
        unit["text"] = ("다음 내용은 다투어졌다: " if source_language == "ko" else "The following account was disputed: ") + text
    elif status == "party_allegation" and not re.search(r"주장|alleg|assert|claim", text, re.I):
        unit["text"] = ("당사자는 다음과 같이 주장했다: " if source_language == "ko" else "A party alleged that ") + text


def process_case(case: dict[str, Any], args: argparse.Namespace, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    source_language = "ko" if case.get("origin_country") == "KR" else "en"
    highest_passages = build_passages(case.get("main_opinion_text") or "", "H", args.max_source_chars)
    lower_passages = build_passages(case.get("lower_court_fact_text") or "", "L", args.max_source_chars)
    passage_map = {item["span_id"]: item for item in [*highest_passages, *lower_passages]}
    user_payload = {
        "case_id": case["case_id"], "source_language": source_language,
        "highest_court_passages": [{"span_id": item["span_id"], "text": item["text"]} for item in highest_passages],
        "lower_court_passages": [{"span_id": item["span_id"], "text": item["text"]} for item in lower_passages],
        "lower_court_case_ids": case.get("lower_court_case_ids") or [],
    }
    semantic_findings = case.get("_semantic_repair_findings") or []
    if semantic_findings:
        user_payload["PRIOR_SEMANTIC_QC_FINDINGS_TO_CORRECT"] = semantic_findings
    if args.mock_response_dir:
        parsed = load_mock(args.mock_response_dir, "extract", case["case_id"])
        if parsed is None:
            raise RuntimeError(f"No extraction mock found for {case['case_id']}")
        provenance = {"model": "mock", "model_snapshot_or_returned_model_id": "mock", "prompt_version": args.prompt_version, "request_id": None, "timestamp": datetime.now(timezone.utc).isoformat(), "usage": None, "input_hash": sha256_text(json.dumps(user_payload, ensure_ascii=False, sort_keys=True)), "output_hash": sha256_text(json.dumps(parsed, ensure_ascii=False, sort_keys=True)), "status": "success"}
    else:
        stage = "extract-semantic-repair" if semantic_findings else "extract"
        system_prompt = prompt + ("\nA prior source-aware QC found the listed concrete defects. Rebuild a concise nonduplicative factual record that corrects all of them. Omit litigation roles, pleadings, trial/appellate posture, verdicts, awards, and dispositions. Preserve allegations, testimony, disputes, entities, and numbers exactly as supported." if semantic_findings else "")
        parsed, provenance = call_structured(
            case_id=case["case_id"], stage=stage, prompt_version=args.prompt_version, model=model,
            system_prompt=system_prompt, user_payload=user_payload, schema_name="neutral_fact_units", schema=EXTRACTION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
    units, issues = validate_parsed(case, parsed, passage_map)
    issues = normalize_neutral_units(units, issues)
    if issues and not args.mock_response_dir:
        retry_payload = {**user_payload, "VALIDATION_FAILURES_TO_CORRECT": issues}
        parsed, provenance = call_structured(
            case_id=case["case_id"], stage="extract-semantic-repair-validation" if semantic_findings else "extract-validation-retry", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt + "\nThe prior response failed deterministic validation. Correct every listed failure without inventing facts; use other supplied exact passages when necessary.",
            user_payload=retry_payload, schema_name="neutral_fact_units", schema=EXTRACTION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        units, issues = validate_parsed(case, parsed, passage_map)
        issues = normalize_neutral_units(units, issues)
    if issues and semantic_findings and not args.mock_response_dir:
        retry_payload = {**user_payload, "FOURTH_VALIDATION_FAILURES_TO_CORRECT": issues}
        parsed, provenance = call_structured(
            case_id=case["case_id"], stage="extract-semantic-repair-validation-4", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt + "\nAbsolute final constraint: if source_language is ko, every neutral text field must be Korean. If causation is missing, state a concrete event-to-harm sequence supported by an exact passage, not legal causation.",
            user_payload=retry_payload, schema_name="neutral_fact_units", schema=EXTRACTION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        units, issues = validate_parsed(case, parsed, passage_map)
        issues = normalize_neutral_units(units, issues)
    if issues and semantic_findings and not args.mock_response_dir:
        retry_payload = {**user_payload, "THIRD_VALIDATION_FAILURES_TO_CORRECT": issues}
        parsed, provenance = call_structured(
            case_id=case["case_id"], stage="extract-semantic-repair-validation-3", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt + "\nReturn Korean neutral text for Korean sources and English for U.S. sources. Explicitly label at least one distinct included unit as each of parties, conduct, harm, and causation, using exact supporting passages. Omit all litigation roles and dispositions.",
            user_payload=retry_payload, schema_name="neutral_fact_units", schema=EXTRACTION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        units, issues = validate_parsed(case, parsed, passage_map)
        issues = normalize_neutral_units(units, issues)
    if issues and not args.mock_response_dir:
        retry_payload = {**user_payload, "FINAL_VALIDATION_FAILURES_TO_CORRECT": issues}
        parsed, provenance = call_structured(
            case_id=case["case_id"], stage="extract-semantic-repair-validation-2" if semantic_findings else "extract-validation-retry-2", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt + "\nFinal repair: output only the supplied source language and include distinct source-grounded parties, conduct, harm, and concrete factual causation units. Do not reintroduce litigation posture or duplicate content.",
            user_payload=retry_payload, schema_name="neutral_fact_units", schema=EXTRACTION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        units, issues = validate_parsed(case, parsed, passage_map)
        issues = normalize_neutral_units(units, issues)
    included = [unit["text"].strip() for unit in units if unit.get("include_in_neutral_fact")]
    master = " ".join(value for value in included if value)
    language_sanity = script_language_sanity(master, source_language)
    if language_sanity["status"] != "pass":
        issues.append("source_language_sanity")
    leaks = leakage_checks(master)
    grounding_status = "pass" if units and all(unit.get("source_grounding_status") == "pass" for unit in units) else "fail"
    record = {
        "case_id": case["case_id"], "case_family_id": case.get("case_family_id"), "origin_country": case.get("origin_country"),
        "origin_state": case.get("origin_state"), "primary_domain": case.get("primary_domain") or case.get("case_domain"),
        "case_domain": case.get("primary_domain") or case.get("case_domain"),
        "liability_theories": case.get("liability_theories", []), "secondary_tags": case.get("secondary_tags", []),
        "source_language": source_language,
        "highest_court_case_id": case.get("highest_court_case_id", case["case_id"]),
        "lower_court_supplemented": case.get("lower_court_supplemented", False),
        "lower_court_case_ids": case.get("lower_court_case_ids", []),
        "lower_court_link_confidence": case.get("lower_court_link_confidence", "none"),
        "neutral_fact_source": master, "neutral_fact_ko": master if source_language == "ko" else "",
        "neutral_fact_en": master if source_language == "en" else "", "fact_units": units,
        "source_language_sanity_status": language_sanity["status"], "source_language_sanity_detail": language_sanity,
        "source_grounding_status": grounding_status, **leaks,
        "unit_normalization_status": "review" if any(item.get("status") == "review" for item in parsed.get("normalizations", [])) else "pass",
        "institution_neutralization_status": "review" if any(item.get("kind") == "institution" and item.get("status") == "review" for item in parsed.get("normalizations", [])) else "pass",
        "normalizations": parsed.get("normalizations", []), "extraction_validation_issues": issues,
        "extraction_provenance": provenance,
    }
    private = {"case_id": case["case_id"], "entity_mappings": parsed.get("entity_mappings", [])}
    validation_pass = not issues and grounding_status == "pass" and master and leaks["legal_leakage_status"] == leaks["jurisdiction_leakage_status"] == "pass"
    return ("success" if validation_pass else "failure"), {"record": record, "private": private}


def select_smoke_cases(cases: list[dict[str, Any]], per_country: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for country in ("KR", "US"):
        ranked = sorted(
            (item for item in cases if item.get("origin_country") == country),
            key=lambda item: (-int(item.get("fact_sufficiency_score") or 0), item["case_id"]),
        )
        country_selected: list[dict[str, Any]] = []
        seen_domains: set[str] = set()
        for item in ranked:
            domain = item.get("primary_domain") or item.get("case_domain") or ""
            if domain in seen_domains:
                continue
            country_selected.append(item)
            seen_domains.add(domain)
            if len(country_selected) == per_country:
                break
        if len(country_selected) < per_country:
            selected_ids = {item["case_id"] for item in country_selected}
            country_selected.extend(item for item in ranked if item["case_id"] not in selected_ids)
        selected.extend(country_selected[:per_country])
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source_path = args.output_dir / "neutral_facts_source.jsonl"
    units_path = args.output_dir / "neutral_fact_units_source.jsonl"
    private_path = args.output_dir / "neutral_fact_entity_mappings_private.jsonl"
    failures_path = args.output_dir / "neutral_fact_failures.jsonl"
    prior_source = list(read_jsonl(source_path)) if args.revalidate_existing and source_path.exists() else []
    prior_private = list(read_jsonl(private_path)) if args.revalidate_existing and private_path.exists() else []
    prior_failures = list(read_jsonl(failures_path)) if args.revalidate_existing and failures_path.exists() else []
    guard_outputs((source_path, units_path, private_path, failures_path), overwrite=args.overwrite, resume=args.resume)
    cases = [row for path in args.input for row in read_jsonl(path) if args.include_ineligible or row.get("strict_source_eligible") is True]
    if args.semantic_repair_file:
        semantic = {
            row["case_id"]: row for row in read_jsonl(args.semantic_repair_file)
            if row.get("hard_fail") or row.get("manual_review_required")
        }
        cases = [
            {**case, "_semantic_repair_findings": [*(semantic[case["case_id"]].get("issues") or []), *(semantic[case["case_id"]].get("evidence") or [])]}
            for case in cases if case["case_id"] in semantic
        ]
    if args.limit_per_country:
        cases = select_smoke_cases(cases, args.limit_per_country)
    elif args.limit:
        cases = cases[:args.limit]
    prompt = args.prompt.read_text(encoding="utf-8")
    model = "mock" if args.mock_response_dir else configured_model(args.model)
    planned_calls = len(cases)
    if args.resume and source_path.exists():
        existing_ids = {row["case_id"] for row in read_jsonl(source_path)}
        planned_calls = sum(case["case_id"] not in existing_ids for case in cases)
    plan = {"stage": "extract", "cases": len(cases), "model": model, "prompt_version": args.prompt_version, "api_calls": 0 if args.dry_run or args.mock_response_dir or args.revalidate_existing else planned_calls}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if args.revalidate_existing:
        llm_exclusions: dict[str, set[str]] = {}
        llm_epistemic_frames: dict[str, set[str]] = {}
        if args.exclude_from_llm_qc:
            with args.exclude_from_llm_qc.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    fact_ids = {
                        fact_id for finding in json.loads(row.get("llm_qc_findings") or "[]")
                        if finding.get("problem_type") in {"legal_conclusion", "jurisdiction_signal"}
                        for fact_id in re.findall(r"F\d{3}", finding.get("fact_id") or "")
                    }
                    if fact_ids:
                        llm_exclusions[row["case_id"]] = fact_ids
                    epistemic_ids = {
                        fact_id for finding in json.loads(row.get("llm_qc_findings") or "[]")
                        if finding.get("problem_type") == "epistemic_status_change"
                        for fact_id in re.findall(r"F\d{3}", finding.get("fact_id") or "")
                    }
                    if epistemic_ids:
                        llm_epistemic_frames[row["case_id"]] = epistemic_ids
        records = {row["case_id"]: row for row in prior_source}
        for failure in prior_failures:
            if failure.get("stage") == "extract_validation" and failure.get("record"):
                records[failure["case_id"]] = failure["record"]
        completed = {}
        private = {row["case_id"]: row for row in prior_private}
        failures = [failure for failure in prior_failures if failure.get("stage") == "extract_api"]
        for case in cases:
            record = records.get(case["case_id"])
            if not record:
                failures.append({"case_id": case["case_id"], "stage": "extract_revalidation", "error": "no_saved_extraction_record"})
                continue
            units = record.get("fact_units") or []
            for unit in units:
                if unit.get("fact_id") in llm_exclusions.get(record["case_id"], set()):
                    unit["include_in_neutral_fact"] = False
                    unit["exclusion_reason"] = "secondary_llm_qc_legal_or_jurisdiction"
                if unit.get("fact_id") in llm_epistemic_frames.get(record["case_id"], set()):
                    ensure_epistemic_frame(unit, record.get("source_language") or "en")
            issues = normalize_neutral_units(units, record.get("extraction_validation_issues") or [])
            record["extraction_validation_issues"] = issues
            record["neutral_fact_source"] = " ".join(
                unit["text"].strip() for unit in units if unit.get("include_in_neutral_fact") and unit.get("text")
            )
            if record.get("source_language") == "ko":
                record["neutral_fact_ko"] = record["neutral_fact_source"]
            else:
                record["neutral_fact_en"] = record["neutral_fact_source"]
            leaks = leakage_checks(record.get("neutral_fact_source") or "")
            record.update(leaks)
            grounding = "pass" if units and all(unit.get("source_grounding_status") == "pass" for unit in units) else "fail"
            record["source_grounding_status"] = grounding
            validation_pass = not issues and grounding == "pass" and record.get("neutral_fact_source") and leaks["legal_leakage_status"] == leaks["jurisdiction_leakage_status"] == "pass"
            if validation_pass:
                completed[record["case_id"]] = record
                private.setdefault(record["case_id"], {"case_id": record["case_id"], "entity_mappings": []})
            else:
                failures.append({"case_id": record["case_id"], "stage": "extract_validation", "issues": issues, "record": record})
        ordered = [completed[case["case_id"]] for case in cases if case["case_id"] in completed]
        write_jsonl(source_path, ordered)
        write_jsonl(units_path, [{"case_id": row["case_id"], **unit} for row in ordered for unit in row["fact_units"]])
        write_jsonl(private_path, [private[case["case_id"]] for case in cases if case["case_id"] in private])
        write_jsonl(failures_path, failures)
        print(json.dumps({"extraction_pass": len(ordered), "failures": len(failures)}, ensure_ascii=False))
        return 0 if not failures else 2
    model = "mock" if args.mock_response_dir else configured_model(args.model)
    completed = {row["case_id"]: row for row in read_jsonl(source_path)} if args.resume and source_path.exists() else {}
    private = {row["case_id"]: row for row in read_jsonl(private_path)} if args.resume and private_path.exists() else {}
    failures = list(read_jsonl(failures_path)) if args.resume and failures_path.exists() else []
    pending = [case for case in cases if case["case_id"] not in completed]
    if args.resume:
        pending_ids = {case["case_id"] for case in pending}
        failures = [failure for failure in failures if failure.get("case_id") not in pending_ids]
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
