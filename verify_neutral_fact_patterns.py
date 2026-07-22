from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl
from pipeline.llm_client import LLMClient
from pipeline.neutral_verification import (recheck_grounding_verifier_record,
    recheck_translation_verifier_record, verify_grounding, verify_translation)
from pipeline.stage2_runtime import (append_errors, append_quarantine, append_run_history, by_case,
    ensure_writable_outputs, merge_by_case, quarantine_record, record_counts, resolve_case_ids,
    run_parallel, write_usage)


ROOT = Path(__file__).resolve().parent
def _path(base: Path, filename: str) -> Path: return base / filename if base.is_dir() else base
def _evidence_path(base: Path, suffix: str) -> Path:
    if base.is_dir(): return base / f"factual_evidence_{suffix}.jsonl"
    expected = f"source_neutral_{suffix}.jsonl"
    return base.with_name(f"factual_evidence_{suffix}.jsonl") if base.name == expected else base.with_name(base.name.replace("source_neutral", "factual_evidence"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run independent source-grounding and translation verifiers.")
    p.add_argument("--source-neutral-input", type=Path, required=True); p.add_argument("--translation-input", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--verifier-model", default="gpt-5.6-luna"); p.add_argument("--base-url", default="https://gw.letsur.ai/v1"); p.add_argument("--concurrency", type=int, default=2); p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--case-id", action="append"); p.add_argument("--case-id-file", type=Path); p.add_argument("--batch-name", default="unnamed"); p.add_argument("--max-cases-per-origin", type=int)
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--mock-response-dir", type=Path)
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--retry-warnings", action="store_true"); p.add_argument("--recheck-deterministic", action="store_true"); p.add_argument("--regenerate", action="store_true"); p.add_argument("--regenerate-on-verifier-fail", action="store_true", help=argparse.SUPPRESS); p.add_argument("--include-unusable", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stop-on-hard-failure", action="store_true"); p.add_argument("--max-hard-failure-rate", type=float, default=0.10); p.add_argument("--max-api-failure-rate", type=float, default=0.05)
    return p


def _retry(record, args, field):
    if record is None: return True
    if args.regenerate: return True
    status = str(record.get(field) or "fail")
    return (args.retry_failed and status == "fail") or (args.retry_warnings and status == "warning")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); output = args.output_dir
    grounding_path, translation_path = output / "source_grounding_verification.jsonl", output / "translation_verification.jsonl"; ensure_writable_outputs((grounding_path, translation_path), resume=args.resume, overwrite=args.overwrite)
    masters = {}; evidence = {}; translations = {}
    for suffix in ("kr", "ca"):
        master_path = _path(args.source_neutral_input, f"source_neutral_{suffix}.jsonl")
        masters.update(by_case(master_path)); evidence.update(by_case(_evidence_path(args.source_neutral_input, suffix))); translations.update(by_case(_path(args.translation_input, f"translated_pairs_{suffix}.jsonl")))
    selected_ids = set(resolve_case_ids(args.case_id, args.case_id_file)); eligible = [row for cid, row in masters.items() if cid in evidence and (row.get("case_is_usable_for_translation", row.get("case_is_usable")) or args.include_unusable) and (not selected_ids or cid in selected_ids)]
    if args.max_cases_per_origin is not None:
        counts = Counter(); limited = []
        for row in eligible:
            if counts[row["case_origin"]] < args.max_cases_per_origin: limited.append(row); counts[row["case_origin"]] += 1
        eligible = limited
    grounding_items = eligible; translation_items = [row for row in eligible if row["case_id"] in translations]
    if args.dry_run:
        append_run_history(output, {"batch_name": args.batch_name, "selected_case_ids": [row["case_id"] for row in eligible], "mode": "dry_run", "new_api_calls": 0, "cache_hits": 0}); print(f"verification dry-run pass: grounding={len(grounding_items)} translation={len(translation_items)}"); return 0
    client = LLMClient(output_dir=output, model=args.verifier_model, base_url=args.base_url, max_retries=args.max_retries, mock_response_dir=args.mock_response_dir, bypass_cache=bool(args.retry_failed or args.retry_warnings or args.regenerate))
    old_ground = by_case(grounding_path) if not args.overwrite else {}; old_translation = by_case(translation_path) if not args.overwrite else {}
    ground_results = []; translation_results = []; errors = []
    if args.recheck_deterministic:
        ground_results = [recheck_grounding_verifier_record(old_ground[row["case_id"]]) for row in grounding_items if row["case_id"] in old_ground]
        translation_results = [recheck_translation_verifier_record(old_translation[row["case_id"]], translations[row["case_id"]]) for row in translation_items if row["case_id"] in old_translation]
    else:
        ground_todo = [row for row in grounding_items if _retry(old_ground.get(row["case_id"]), args, "verifier_status")]; trans_todo = [row for row in translation_items if _retry(old_translation.get(row["case_id"]), args, "translation_status")]
        def ground_worker(row): return verify_grounding(row, evidence[row["case_id"]], client, ROOT)
        def translation_worker(row): return verify_translation(row, translations[row["case_id"]], client, ROOT)
        ground_results, errors = run_parallel(ground_todo, ground_worker, args.concurrency); translation_results, more = run_parallel(trans_todo, translation_worker, args.concurrency); errors.extend(more)
    grounding_all = merge_by_case(list(old_ground.values()), ground_results); translation_all = merge_by_case(list(old_translation.values()), translation_results); atomic_write_jsonl(grounding_path, grounding_all); atomic_write_jsonl(translation_path, translation_all)
    ground_map, translation_map = {row["case_id"]: row for row in grounding_all}, {row["case_id"]: row for row in translation_all}
    append_errors(output / "api_errors.jsonl", errors); write_usage(output / "api_usage.csv", ground_results + translation_results)
    hard = []
    for master in eligible:
        case_id = master["case_id"]; reasons = []
        ground = ground_map.get(case_id); trans_verify = translation_map.get(case_id) if case_id in translations else None
        if not ground: reasons.append("missing_grounding_verification")
        elif ground.get("validated_verifier_status") == "fail": reasons.extend(ground.get("deterministic_verifier_reasons") or ["grounding_verifier_fail"])
        if case_id in translations:
            if not trans_verify: reasons.append("missing_translation_verification")
            elif trans_verify.get("validated_verifier_status") == "fail": reasons.extend(trans_verify.get("deterministic_verifier_reasons") or ["translation_verifier_fail"])
        if (ground and ground.get("verifier_consistency_violation")) or (trans_verify and trans_verify.get("verifier_consistency_violation")): reasons.append("verifier_consistency_violation")
        if reasons: hard.append((master, sorted(set(reasons))))
    quarantines = [quarantine_record(master, "verification", reasons, deterministic_recheck=True) for master, reasons in hard]; append_quarantine(output / "quarantine.jsonl", quarantines)
    selected_count = max(1, len(eligible)); hard_rate = len(hard) / selected_count; api_rate = len(errors) / selected_count
    batch = {"batch_name": args.batch_name, "selected_case_ids": [row["case_id"] for row in eligible], "new_api_calls": client.new_api_calls, "cache_hits": client.cache_hits, "completed_cases": sum(row["case_id"] in ground_map and (row["case_id"] not in translations or row["case_id"] in translation_map) for row in eligible), "pass_count": sum(ground_map.get(row["case_id"], {}).get("validated_verifier_status") == "pass" and (row["case_id"] not in translations or translation_map.get(row["case_id"], {}).get("validated_verifier_status") == "pass") for row in eligible), "warning_count": sum("warning" in {ground_map.get(row["case_id"], {}).get("validated_verifier_status"), translation_map.get(row["case_id"], {}).get("validated_verifier_status")} for row in eligible), "fail_count": len(hard), "quarantined_count": len(quarantines), "missing_count": sum(row["case_id"] not in ground_map for row in eligible), "hard_failure_rate": hard_rate, "api_failure_rate": api_rate}
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.batch_name); atomic_write_json(output / f"batch_{safe}_verification.json", batch)
    phases = {"phase_5_grounding_verifier": {"execution_status": "completed", "record_counts": record_counts(grounding_all, "validated_verifier_status", 70)}, "phase_8_translation_verifier": {"execution_status": "completed", "record_counts": record_counts(translation_all, "validated_verifier_status", 70)}}
    append_run_history(output, {**batch, "mode": "mock" if args.mock_response_dir else "api", "recheck_deterministic": args.recheck_deterministic, "retry_failed": args.retry_failed, "retry_warnings": args.retry_warnings, "regenerate": args.regenerate}, phases)
    blocked = bool(hard) or any((row.get("verifier_consistency_violation") for row in ground_results + translation_results)) or hard_rate > args.max_hard_failure_rate or api_rate > args.max_api_failure_rate
    print(f"verification complete: selected={len(eligible)} grounding_new={len(ground_results)} translation_new={len(translation_results)} api_calls={client.new_api_calls} cache_hits={client.cache_hits} hard_fail={len(hard)} blocked={blocked}")
    return 2 if blocked else 0


if __name__ == "__main__": raise SystemExit(main())
