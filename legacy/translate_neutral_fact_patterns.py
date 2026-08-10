from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl
from pipeline.llm_client import LLMClient
from pipeline.neutral_translation import recheck_translation_record, translate
from pipeline.stage2_runtime import (append_errors, append_quarantine, append_run_history, by_case,
    ensure_writable_outputs, merge_by_case, quarantine_record, record_counts, resolve_case_ids,
    run_parallel, write_usage)


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Translate source-language Stage 2 masters fact-unit by fact-unit.")
    p.add_argument("--kr-source-neutral", type=Path, required=True); p.add_argument("--ca-source-neutral", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.6-luna"); p.add_argument("--base-url", default="https://gw.letsur.ai/v1"); p.add_argument("--concurrency", type=int, default=2); p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--case-id", action="append"); p.add_argument("--case-id-file", type=Path); p.add_argument("--batch-name", default="unnamed"); p.add_argument("--max-cases-per-origin", type=int)
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--mock-response-dir", type=Path)
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--retry-warnings", action="store_true"); p.add_argument("--recheck-deterministic", action="store_true"); p.add_argument("--regenerate", action="store_true"); p.add_argument("--include-unusable", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stop-on-hard-failure", action="store_true"); p.add_argument("--max-hard-failure-rate", type=float, default=0.10); p.add_argument("--max-api-failure-rate", type=float, default=0.05)
    return p


def _retry(record, args):
    if record is None: return True
    if args.regenerate: return True
    status = str(record.get("translation_status") or "fail")
    return (args.retry_failed and status == "fail") or (args.retry_warnings and status == "warning")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); output = args.output_dir
    paths = {"KR": output / "translated_pairs_kr.jsonl", "CA": output / "translated_pairs_ca.jsonl"}; ensure_writable_outputs(paths.values(), resume=args.resume, overwrite=args.overwrite)
    masters: list[dict] = []
    for path, origin in ((args.kr_source_neutral, "KR"), (args.ca_source_neutral, "CA")):
        masters.extend(row for row in by_case(path).values() if row.get("case_origin") == origin and (row.get("case_is_usable_for_translation", row.get("case_is_usable")) or args.include_unusable) and row.get("neutralization_status") in {"pass", "warning"})
    selected_ids = set(resolve_case_ids(args.case_id, args.case_id_file)); masters = [row for row in masters if not selected_ids or row["case_id"] in selected_ids]
    if args.max_cases_per_origin is not None:
        counts = Counter(); limited = []
        for row in masters:
            if counts[row["case_origin"]] < args.max_cases_per_origin: limited.append(row); counts[row["case_origin"]] += 1
        masters = limited
    if args.dry_run:
        append_run_history(output, {"batch_name": args.batch_name, "selected_case_ids": [row["case_id"] for row in masters], "mode": "dry_run", "new_api_calls": 0, "cache_hits": 0})
        print(f"translation dry-run pass: {len(masters)} masters; raw source is absent from requests"); return 0
    client = LLMClient(output_dir=output, model=args.model, base_url=args.base_url, max_retries=args.max_retries, mock_response_dir=args.mock_response_dir, bypass_cache=bool(args.retry_failed or args.retry_warnings or args.regenerate))
    existing = {origin: by_case(path) if not args.overwrite else {} for origin, path in paths.items()}; results = []; errors = []
    if args.recheck_deterministic:
        for master in masters:
            record = existing[master["case_origin"]].get(master["case_id"])
            if record: results.append(recheck_translation_record(master, record))
    else:
        todo = [master for master in masters if _retry(existing[master["case_origin"]].get(master["case_id"]), args)]
        def worker(row): return translate(row, client, ROOT)
        results, errors = run_parallel(todo, worker, args.concurrency)
    for origin in ("KR", "CA"):
        merged = merge_by_case(list(existing[origin].values()), [row for row in results if row["case_origin"] == origin]); atomic_write_jsonl(paths[origin], merged); existing[origin] = {row["case_id"]: row for row in merged}
    append_errors(output / "api_errors.jsonl", errors); write_usage(output / "api_usage.csv", results)
    current = [row for origin in ("KR", "CA") for row in existing[origin].values()]; selected_records = [existing[row["case_origin"]].get(row["case_id"]) for row in masters]
    hard = [(master, record) for master, record in zip(masters, selected_records) if record is None or record.get("translation_status") == "fail"]
    quarantines = [quarantine_record(record or master, "translation", (record or {}).get("translation_warnings") or ["missing_translation_record"], regeneration=record is None, deterministic_recheck=record is not None) for master, record in hard]; append_quarantine(output / "quarantine.jsonl", quarantines)
    selected_count = max(1, len(masters)); hard_rate = len(hard) / selected_count; api_rate = len(errors) / selected_count
    batch = {"batch_name": args.batch_name, "selected_case_ids": [row["case_id"] for row in masters], "new_api_calls": client.new_api_calls, "cache_hits": client.cache_hits, "completed_cases": sum(row is not None for row in selected_records), "pass_count": sum(row is not None and row.get("translation_status") == "pass" for row in selected_records), "warning_count": sum(row is not None and row.get("translation_status") == "warning" for row in selected_records), "fail_count": len(hard), "quarantined_count": len(quarantines), "missing_count": sum(row is None for row in selected_records), "hard_failure_rate": hard_rate, "api_failure_rate": api_rate}
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.batch_name); atomic_write_json(output / f"batch_{safe}_translation.json", batch)
    phases = {"phase_6_translation": {"execution_status": "completed", "record_counts": record_counts(current, "translation_status", 70)}, "phase_7_deterministic_translation_checks": {"execution_status": "completed", "record_counts": record_counts(current, "translation_status", 70)}}
    append_run_history(output, {**batch, "mode": "mock" if args.mock_response_dir else "api", "recheck_deterministic": args.recheck_deterministic, "retry_failed": args.retry_failed, "retry_warnings": args.retry_warnings, "regenerate": args.regenerate}, phases)
    blocked = bool(hard) or hard_rate > args.max_hard_failure_rate or api_rate > args.max_api_failure_rate
    print(f"translation complete: selected={len(masters)} new={len(results)} api_calls={client.new_api_calls} cache_hits={client.cache_hits} hard_fail={len(hard)} blocked={blocked}")
    return 2 if blocked else 0


if __name__ == "__main__": raise SystemExit(main())
