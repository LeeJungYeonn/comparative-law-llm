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
from pipeline_v2.rules import PLACEHOLDER_RE, leakage_checks, neutralize_jurisdiction_signals, translation_equivalence_checks
from pipeline_v2.v3_rules import script_language_sanity

PROMPT_VERSION = "neutral-fact-translation-v2.4"
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
    result.add_argument("--base-url", default=DEFAULT_LETSUR_BASE_URL)
    result.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    result.add_argument("--dotenv-path", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--prompt", type=Path, default=Path("prompts_v2/translate_neutral_facts_v2.txt"))
    result.add_argument("--prompt-version", default=PROMPT_VERSION)
    result.add_argument("--concurrency", type=int, default=2)
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--mock-response-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--revalidate-existing", action="store_true", help="Re-run deterministic validation on saved translation-validation records without API calls.")
    result.add_argument("--retry-from-qc", type=Path, help="With --resume, retry case IDs whose QC status is review.")
    return result


def validate_translation(source: dict[str, Any], parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    expected = [unit for unit in source["fact_units"] if unit.get("include_in_neutral_fact")]
    expected_ids = {unit["fact_id"] for unit in expected}
    aligned = [unit for unit in (parsed.get("aligned_units") or []) if unit.get("fact_id") in expected_ids]
    issues: list[str] = []
    if [unit.get("fact_id") for unit in aligned] != [unit.get("fact_id") for unit in expected]:
        issues.append("fact_id_order_or_set_mismatch")
    expected_by_id = {unit["fact_id"]: unit for unit in expected}
    for unit in aligned:
        original = expected_by_id.get(unit.get("fact_id"))
        if not original:
            continue
        model_source_text = unit.get("source_text") or ""
        unit["model_source_text"] = model_source_text
        unit["source_text_copy_status"] = "exact" if model_source_text == original.get("text") else "canonicalized_from_fact_id"
        unit["source_text"] = original.get("text") or ""
        source_field = "neutral_ko" if source["source_language"] == "ko" else "neutral_en"
        target_field = "neutral_en" if source["source_language"] == "ko" else "neutral_ko"
        unit[source_field] = unit["source_text"]
        target_before = unit.get(target_field) or ""
        target_after, jurisdiction_evidence = neutralize_jurisdiction_signals(target_before)
        if target_field == "neutral_ko":
            possessive_before = target_after
            for match in reversed(list(re.finditer(r"그(?:녀)?의", target_after))):
                prior_people = list(re.finditer(r"\[PERSON_[A-Z]+\]", target_after[:match.start()]))
                if prior_people:
                    replacement = prior_people[-1].group(0) + "의"
                    target_after = target_after[:match.start()] + replacement + target_after[match.end():]
            if target_after != possessive_before:
                unit["neutral_ko_before_possessive_neutralization"] = possessive_before
        if target_after != target_before:
            unit[f"{target_field}_before_jurisdiction_neutralization"] = target_before
            unit["translation_jurisdiction_neutralization_evidence"] = jurisdiction_evidence
            unit[target_field] = target_after
        source_placeholders = set(PLACEHOLDER_RE.findall(original.get("text") or ""))
        target_placeholders = set(PLACEHOLDER_RE.findall(target_after))
        extras = target_placeholders - source_placeholders
        if target_field == "neutral_en" and extras:
            repaired = target_after
            for placeholder in sorted(extras):
                repaired = re.sub(rf"when\s+{re.escape(placeholder)}\s+arrived\s+at", "on arrival at", repaired, flags=re.I)
            if repaired != target_after:
                unit["target_before_placeholder_repair"] = target_after
                unit["placeholder_repair_reason"] = "removed_target_only_entity_from_source_arrival_phrase"
                target_after = repaired
                unit[target_field] = repaired
        target = unit.get("neutral_en", "") if source["source_language"] == "ko" else unit.get("neutral_ko", "")
        checks = translation_equivalence_checks(original.get("text", ""), target, source["source_language"])
        unit.update(checks)
        if checks["translation_equivalence_status"] != "pass":
            issues.extend(f"{unit.get('fact_id')}:{item}" for item in checks["translation_equivalence_issues"])
    return aligned, issues


def build_translation_record(source: dict[str, Any], parsed: dict[str, Any], provenance: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    aligned, issues = validate_translation(source, parsed)
    ko = " ".join(unit["neutral_ko"].strip() for unit in aligned)
    en = " ".join(unit["neutral_en"].strip() for unit in aligned)
    # Per-unit checks are authoritative. Rechecking the concatenated prose is
    # redundant and can double-count month/number concepts across boundaries.
    leak_ko, leak_en = leakage_checks(ko), leakage_checks(en)
    status = "pass" if not issues and leak_ko["legal_leakage_status"] == leak_en["legal_leakage_status"] == "pass" and leak_ko["jurisdiction_leakage_status"] == leak_en["jurisdiction_leakage_status"] == "pass" else "fail"
    record = {
        **{key: value for key, value in source.items() if key != "fact_units"},
        "source_fact_units": source["fact_units"],
        "neutral_fact_ko": ko, "neutral_fact_en": en, "aligned_fact_units": aligned,
        "translation_equivalence_status": "pass" if not issues else "fail", "translation_equivalence_issues": list(dict.fromkeys(issues)),
        "translation_equivalence_warnings": list(dict.fromkeys(
            warning for unit in aligned for warning in unit.get("translation_equivalence_warnings", [])
        )),
        "legal_leakage_status": "pass" if leak_ko["legal_leakage_status"] == leak_en["legal_leakage_status"] == "pass" else "fail",
        "legal_leakage_evidence": {"ko": leak_ko["legal_leakage_evidence"], "en": leak_en["legal_leakage_evidence"]},
        "jurisdiction_leakage_status": "pass" if leak_ko["jurisdiction_leakage_status"] == leak_en["jurisdiction_leakage_status"] == "pass" else "fail",
        "jurisdiction_leakage_evidence": {"ko": leak_ko["jurisdiction_leakage_evidence"], "en": leak_en["jurisdiction_leakage_evidence"]},
        "translation_provenance": provenance,
    }
    return status, record


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
        provenance = {"model": "mock", "model_snapshot_or_returned_model_id": "mock", "prompt_version": args.prompt_version, "request_id": None, "timestamp": datetime.now(timezone.utc).isoformat(), "usage": None, "input_hash": sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True)), "output_hash": sha256_text(json.dumps(parsed, ensure_ascii=False, sort_keys=True)), "status": "success"}
    else:
        parsed, provenance = call_structured(
            case_id=source["case_id"], stage="translate", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt, user_payload=payload, schema_name="aligned_translation", schema=TRANSLATION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
    def validated(value: dict[str, Any], prov: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        current_status, current = build_translation_record(source, value, prov)
        ko_sanity = script_language_sanity(current.get("neutral_fact_ko") or "", "ko")
        en_sanity = script_language_sanity(current.get("neutral_fact_en") or "", "en")
        current["language_sanity_status"] = "pass" if ko_sanity["status"] == en_sanity["status"] == "pass" else "fail"
        current["language_sanity_detail"] = {"ko": ko_sanity, "en": en_sanity}
        if current["language_sanity_status"] != "pass":
            current_status = "fail"
            current.setdefault("translation_equivalence_issues", []).append("language_sanity")
        return current_status, current

    status, record = validated(parsed, provenance)
    if status != "pass" and not args.mock_response_dir:
        feedback = [
            *record.get("translation_equivalence_issues", []),
            *(f"legal_leakage:{lang}:{item}" for lang, items in (record.get("legal_leakage_evidence") or {}).items() for item in items),
            *(f"jurisdiction_leakage:{lang}:{item}" for lang, items in (record.get("jurisdiction_leakage_evidence") or {}).items() for item in items),
        ]
        retry_payload = {**payload, "VALIDATION_FAILURES_TO_CORRECT": feedback}
        parsed, provenance = call_structured(
            case_id=source["case_id"], stage="translate-validation-retry", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt + "\nThe prior translation failed deterministic validation. Correct every listed issue while preserving source-language text exactly.",
            user_payload=retry_payload, schema_name="aligned_translation", schema=TRANSLATION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        status, record = validated(parsed, provenance)
    if status != "pass" and not args.mock_response_dir:
        feedback = [
            *record.get("translation_equivalence_issues", []),
            *(f"legal_leakage:{lang}:{item}" for lang, items in (record.get("legal_leakage_evidence") or {}).items() for item in items),
            *(f"jurisdiction_leakage:{lang}:{item}" for lang, items in (record.get("jurisdiction_leakage_evidence") or {}).items() for item in items),
        ]
        retry_payload = {**payload, "SECOND_VALIDATION_FAILURES_TO_CORRECT": feedback}
        parsed, provenance = call_structured(
            case_id=source["case_id"], stage="translate-validation-retry-2", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt + "\nThis is a final repair attempt. Preserve the exact source placeholder set in each target unit and remove every listed leakage from target-language wording.",
            user_payload=retry_payload, schema_name="aligned_translation", schema=TRANSLATION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        status, record = validated(parsed, provenance)
    if status != "pass" and not args.mock_response_dir:
        feedback = [*record.get("translation_equivalence_issues", [])]
        retry_payload = {**payload, "THIRD_VALIDATION_FAILURES_TO_CORRECT": feedback}
        parsed, provenance = call_structured(
            case_id=source["case_id"], stage="translate-validation-retry-3", prompt_version=args.prompt_version, model=model,
            system_prompt=prompt + "\nFinal strict repair: target placeholders must be exactly the source placeholder set for each fact. CT means computed tomography here, never a jurisdiction.",
            user_payload=retry_payload, schema_name="aligned_translation", schema=TRANSLATION_SCHEMA,
            raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
            max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        status, record = validated(parsed, provenance)
    return status, record


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_path = args.output_dir / "neutral_facts_bilingual.jsonl"
    failures_path = args.output_dir / "neutral_fact_failures.jsonl"
    prior_output = list(read_jsonl(output_path)) if args.revalidate_existing and output_path.exists() else []
    guard_outputs((output_path,), overwrite=args.overwrite, resume=args.resume)
    sources = list(read_jsonl(args.input))
    if args.limit:
        sources = sources[:args.limit]
    prompt = args.prompt.read_text(encoding="utf-8")
    configured = "mock" if args.mock_response_dir else configured_model(args.model)
    api_calls = 0 if args.dry_run or args.mock_response_dir or args.revalidate_existing else len(sources)
    print(json.dumps({"stage": "translate", "cases": len(sources), "model": configured, "prompt_version": args.prompt_version, "api_calls": api_calls}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    model = configured
    completed = {row["case_id"]: row for row in read_jsonl(output_path)} if args.resume and output_path.exists() else {}
    prior_failures = list(read_jsonl(failures_path)) if failures_path.exists() else []
    failures = [] if args.overwrite or args.revalidate_existing else prior_failures
    if args.resume and args.retry_from_qc:
        with args.retry_from_qc.open(encoding="utf-8-sig", newline="") as handle:
            retry_ids = {row["case_id"] for row in csv.DictReader(handle) if row.get("llm_qc_status") == "review"}
        for case_id in retry_ids:
            completed.pop(case_id, None)
    if args.revalidate_existing:
        saved = {row["case_id"]: row.get("record") for row in prior_failures if row.get("stage") == "translation_validation" and row.get("record")}
        saved.update({row["case_id"]: row for row in prior_output})
        for source in sources:
            prior = saved.get(source["case_id"])
            if not prior:
                failures.append({"case_id": source["case_id"], "stage": "translation_revalidation", "error": "no_saved_translation_validation_record"})
                continue
            parsed = {"aligned_units": prior.get("aligned_fact_units") or []}
            status, record = build_translation_record(source, parsed, prior.get("translation_provenance") or {})
            if status == "pass":
                completed[source["case_id"]] = record
            else:
                failures.append({"case_id": source["case_id"], "stage": "translation_validation", "issues": record["translation_equivalence_issues"], "record": record})
        ordered = [completed[source["case_id"]] for source in sources if source["case_id"] in completed]
        write_jsonl(output_path, ordered)
        write_jsonl(failures_path, failures)
        print(json.dumps({"translation_pass": len(ordered), "failures_total": len(failures)}, ensure_ascii=False))
        return 0 if len(ordered) == len(sources) else 2
    pending = [source for source in sources if source["case_id"] not in completed]
    if args.resume:
        pending_ids = {source["case_id"] for source in pending}
        failures = [failure for failure in failures if failure.get("case_id") not in pending_ids]
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
