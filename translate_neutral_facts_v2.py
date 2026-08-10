from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import guard_outputs, read_jsonl, sha256_text, write_jsonl
from pipeline_v2.llm_runtime import call_structured, configured_model, load_mock
from pipeline_v2.rules import leakage_checks, translation_equivalence_checks

PROMPT_VERSION = "neutral-fact-translation-v2.1"
TRANSLATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"aligned_units": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "fact_id": {"type": "string", "pattern": "^F[0-9]{3}$"}, "source_text": {"type": "string"},
        "neutral_ko": {"type": "string"}, "neutral_en": {"type": "string"},
        "translation_status": {"type": "string", "enum": ["aligned", "review"]},
    }, "required": ["fact_id", "source_text", "neutral_ko", "neutral_en", "translation_status"]}}},
    "required": ["aligned_units"],
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Translate source-language neutral fact units and preserve fact-ID alignment.")
    result.add_argument("--input", type=Path, default=Path("outputs_v2/neutral_facts_source.jsonl"))
    result.add_argument("--model")
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--prompt", type=Path, default=Path("prompts_v2/translate_neutral_facts_v2.txt"))
    result.add_argument("--concurrency", type=int, default=2)
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--mock-response-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def validate_translation(source: dict[str, Any], parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    expected = [unit for unit in source["fact_units"] if unit.get("include_in_neutral_fact")]
    aligned = parsed.get("aligned_units") or []
    issues: list[str] = []
    if [unit.get("fact_id") for unit in aligned] != [unit.get("fact_id") for unit in expected]:
        issues.append("fact_id_order_or_set_mismatch")
    expected_by_id = {unit["fact_id"]: unit for unit in expected}
    for unit in aligned:
        original = expected_by_id.get(unit.get("fact_id"))
        if not original:
            continue
        if unit.get("source_text") != original.get("text"):
            issues.append(f"{unit.get('fact_id')}:source_text_changed")
        target = unit.get("neutral_en", "") if source["source_language"] == "ko" else unit.get("neutral_ko", "")
        checks = translation_equivalence_checks(original.get("text", ""), target, source["source_language"])
        unit.update(checks)
        if checks["translation_equivalence_status"] != "pass":
            issues.extend(f"{unit.get('fact_id')}:{item}" for item in checks["translation_equivalence_issues"])
    return aligned, issues


def process_case(source: dict[str, Any], args: argparse.Namespace, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    included = [unit for unit in source["fact_units"] if unit.get("include_in_neutral_fact")]
    payload = {
        "case_id": source["case_id"], "source_language": source["source_language"],
        "target_language": "en" if source["source_language"] == "ko" else "ko",
        "fact_units": [{"fact_id": unit["fact_id"], "source_text": unit["text"], "epistemic_status": unit["epistemic_status"]} for unit in included],
    }
    if args.mock_response_dir:
        parsed = load_mock(args.mock_response_dir, "translate", source["case_id"])
        if parsed is None:
            raise RuntimeError(f"No translation mock found for {source['case_id']}")
        provenance = {"model": "mock", "model_snapshot_or_returned_model_id": "mock", "prompt_version": PROMPT_VERSION, "request_id": None, "timestamp": datetime.now(timezone.utc).isoformat(), "usage": None, "input_hash": sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True)), "output_hash": sha256_text(json.dumps(parsed, ensure_ascii=False, sort_keys=True)), "status": "success"}
    else:
        parsed, provenance = call_structured(
            case_id=source["case_id"], stage="translate", prompt_version=PROMPT_VERSION, model=model,
            system_prompt=prompt, user_payload=payload, schema_name="aligned_translation", schema=TRANSLATION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
        )
    aligned, issues = validate_translation(source, parsed)
    ko = " ".join(unit["neutral_ko"].strip() for unit in aligned)
    en = " ".join(unit["neutral_en"].strip() for unit in aligned)
    aggregate = translation_equivalence_checks(source["neutral_fact_source"], en if source["source_language"] == "ko" else ko, source["source_language"])
    issues.extend(aggregate["translation_equivalence_issues"])
    leak_ko, leak_en = leakage_checks(ko), leakage_checks(en)
    status = "pass" if not issues and leak_ko["legal_leakage_status"] == leak_en["legal_leakage_status"] == "pass" and leak_ko["jurisdiction_leakage_status"] == leak_en["jurisdiction_leakage_status"] == "pass" else "fail"
    record = {
        **{key: value for key, value in source.items() if key != "fact_units"},
        "source_fact_units": source["fact_units"],
        "neutral_fact_ko": ko, "neutral_fact_en": en, "aligned_fact_units": aligned,
        "translation_equivalence_status": "pass" if not issues else "fail", "translation_equivalence_issues": list(dict.fromkeys(issues)),
        "legal_leakage_status": "pass" if leak_ko["legal_leakage_status"] == leak_en["legal_leakage_status"] == "pass" else "fail",
        "legal_leakage_evidence": {"ko": leak_ko["legal_leakage_evidence"], "en": leak_en["legal_leakage_evidence"]},
        "jurisdiction_leakage_status": "pass" if leak_ko["jurisdiction_leakage_status"] == leak_en["jurisdiction_leakage_status"] == "pass" else "fail",
        "jurisdiction_leakage_evidence": {"ko": leak_ko["jurisdiction_leakage_evidence"], "en": leak_en["jurisdiction_leakage_evidence"]},
        "translation_provenance": provenance,
    }
    return status, record


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_path = args.output_dir / "neutral_facts_bilingual.jsonl"
    failures_path = args.output_dir / "neutral_fact_failures.jsonl"
    guard_outputs((output_path,), overwrite=args.overwrite, resume=args.resume)
    sources = list(read_jsonl(args.input))
    if args.limit:
        sources = sources[:args.limit]
    prompt = args.prompt.read_text(encoding="utf-8")
    configured = "mock" if args.mock_response_dir else (args.model or __import__("os").getenv("FACT_EXTRACTION_MODEL") or "UNCONFIGURED")
    print(json.dumps({"stage": "translate", "cases": len(sources), "model": configured, "prompt_version": PROMPT_VERSION, "api_calls": 0 if args.dry_run or args.mock_response_dir else len(sources)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    model = "mock" if args.mock_response_dir else configured_model(args.model)
    completed = {row["case_id"]: row for row in read_jsonl(output_path)} if args.resume and output_path.exists() else {}
    failures = list(read_jsonl(failures_path)) if failures_path.exists() else []
    pending = [source for source in sources if source["case_id"] not in completed]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(process_case, source, args, model, prompt): source for source in pending}
        for future in as_completed(futures):
            source = futures[future]
            try:
                status, record = future.result()
                if status == "pass":
                    completed[source["case_id"]] = record
                else:
                    failures.append({"case_id": source["case_id"], "stage": "translation_validation", "issues": record["translation_equivalence_issues"], "record": record})
            except Exception as exc:
                failures.append({"case_id": source["case_id"], "stage": "translation_api", "error_type": type(exc).__name__, "error": str(exc)})
    ordered = [completed[source["case_id"]] for source in sources if source["case_id"] in completed]
    write_jsonl(output_path, ordered)
    write_jsonl(failures_path, failures)
    print(json.dumps({"translation_pass": len(ordered), "failures_total": len(failures)}, ensure_ascii=False))
    return 0 if len(ordered) == len(sources) else 2


if __name__ == "__main__":
    raise SystemExit(main())
