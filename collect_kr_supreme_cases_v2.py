from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline_v2 import DATE_END, DATE_START, SEED, VERSION
from pipeline_v2.collect import collection_funnel, evaluate_kr_row, iter_lbox_rows
from pipeline_v2.io_utils import guard_outputs, write_csv, write_json, write_jsonl


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Collect Korean Supreme Court civil-liability candidates from LBox.")
    result.add_argument("--start-date", default=DATE_START)
    result.add_argument("--end-date", default=DATE_END)
    result.add_argument("--candidate-target", type=int, default=300)
    result.add_argument("--seed", type=int, default=SEED)
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--local-arrow-dir", type=Path)
    result.add_argument("--revision")
    result.add_argument("--min-opinion-chars", type=int, default=1200)
    result.add_argument("--limit", type=int, default=0, help="Maximum source rows scanned; 0 means no scan cap.")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    candidate_path = args.output_dir / "kr_supreme_candidates.jsonl"
    qc_path = args.output_dir / "kr_supreme_candidates_qc.csv"
    summary_path = args.output_dir / "kr_collection_deterministic_summary.json"
    guard_outputs((candidate_path, qc_path, summary_path), overwrite=args.overwrite, resume=args.resume)
    if args.resume and candidate_path.exists() and qc_path.exists() and summary_path.exists():
        print(f"resume: existing completed Korean collection retained at {candidate_path}")
        return 0
    rows, resolved_revision = iter_lbox_rows(args.local_arrow_dir, args.revision)
    records = []
    scanned = 0
    for row in rows:
        scanned += 1
        record = evaluate_kr_row(row, start_date=args.start_date, end_date=args.end_date, min_chars=args.min_opinion_chars)
        if record:
            records.append(record)
        if args.limit and scanned >= args.limit:
            break
        if args.candidate_target and len(records) >= args.candidate_target:
            break
    summary = {
        "collection_version": VERSION, "source_dataset": "lbox/lbox_open", "source_config": "precedent_corpus",
        "source_revision": resolved_revision or args.revision or "unresolved", "date_window": [args.start_date, args.end_date],
        "candidate_target": args.candidate_target, "seed": args.seed, "partial_run": bool(args.limit),
        "funnel": collection_funnel(records, scanned),
        "court_level_counts": dict(Counter(f"{row['court_level']}|{row['court_level_confidence']}" for row in records)),
        "domain_counts": dict(Counter(row["case_domain"] for row in records)),
        "exclusion_counts": dict(Counter(reason for row in records for reason in row["exclusion_reasons"])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        write_jsonl(candidate_path, records)
        qc_rows = [{key: value for key, value in row.items() if key not in {"full_opinion_text", "main_opinion_text", "separate_opinions"}} for row in records]
        write_csv(qc_path, qc_rows)
        write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
