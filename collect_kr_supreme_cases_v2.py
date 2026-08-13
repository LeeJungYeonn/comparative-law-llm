from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline_v2 import DATE_END, DATE_START, SEED, VERSION
from pipeline_v2.collect import collection_funnel
from pipeline_v2.io_utils import guard_outputs, sha256_text, write_csv, write_json, write_jsonl
from pipeline_v2.legalize_kr import DEFAULT_SOURCE_DIR, evaluate_legalize_file, iter_precedent_files, repository_revision
from pipeline_v2.schema import PRIMARY_DOMAINS


def select_candidate_pool(records: list[dict], target: int, seed: int) -> list[dict]:
    if target <= 0 or len(records) <= target:
        return records
    rejected_target = min(target // 5, sum(not row["strict_source_eligible"] for row in records))
    strict_target = target - rejected_target
    weights = {
        "general_negligence_personal_injury": 0.45,
        "medical_professional_liability": 0.25,
        "product_liability": 0.15,
        "other_civil_liability": 0.15,
    }

    def take(rows: list[dict], count: int, salt: str) -> list[dict]:
        ordered = sorted(rows, key=lambda row: sha256_text(f"{seed}|{salt}|{row['case_id']}"))
        chosen: list[dict] = []
        seen: set[str] = set()
        for domain in PRIMARY_DOMAINS:
            quota = round(count * weights[domain])
            for row in (item for item in ordered if item["primary_domain"] == domain):
                if len([item for item in chosen if item["primary_domain"] == domain]) >= quota:
                    break
                chosen.append(row); seen.add(row["case_id"])
        for row in ordered:
            if len(chosen) >= count:
                break
            if row["case_id"] not in seen:
                chosen.append(row); seen.add(row["case_id"])
        return chosen

    strict = take([row for row in records if row["strict_source_eligible"]], strict_target, "strict")
    rejected = take([row for row in records if not row["strict_source_eligible"]], target - len(strict), "rejected")
    return sorted([*strict, *rejected], key=lambda row: sha256_text(f"{seed}|saved|{row['case_id']}"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Collect Korean Supreme Court civil-liability candidates from structured legalize-kr Markdown.")
    result.add_argument("--start-date", default=DATE_START)
    result.add_argument("--end-date", default=DATE_END)
    result.add_argument("--candidate-target", type=int, default=300)
    result.add_argument("--seed", type=int, default=SEED)
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    result.add_argument("--source-revision")
    result.add_argument("--min-opinion-chars", type=int, default=1200)
    result.add_argument("--limit", type=int, default=0, help="Maximum source files scanned for smoke diagnostics; 0 scans the full path.")
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
    resolved_revision = args.source_revision or repository_revision(args.source_dir)
    records = []
    scanned = 0
    for path in iter_precedent_files(args.source_dir):
        # The filename date is structured and avoids loading records outside the configured window.
        date_match = __import__("re").search(r"_(\d{4}-\d{2}-\d{2})_", path.name)
        if date_match and not (args.start_date <= date_match.group(1) <= args.end_date):
            continue
        scanned += 1
        record = evaluate_legalize_file(
            path, source_dir=args.source_dir, revision=resolved_revision,
            start_date=args.start_date, end_date=args.end_date, min_chars=args.min_opinion_chars,
        )
        if record:
            records.append(record)
        if args.limit and scanned >= args.limit:
            break
    summary = {
        "collection_version": VERSION, "source_dataset": "legalize-kr/precedent-kr", "source_config": "git-markdown",
        "source_repository": "https://github.com/legalize-kr/precedent-kr",
        "source_revision": resolved_revision, "date_window": [args.start_date, args.end_date],
        "candidate_target": args.candidate_target, "seed": args.seed, "partial_run": bool(args.limit),
        "candidate_target_met": len(records) >= args.candidate_target,
        "funnel": collection_funnel(records, scanned),
        "court_level_counts": dict(Counter(f"{row['court_level']}|{row['court_level_confidence']}" for row in records)),
        "primary_domain_counts": dict(Counter(row["primary_domain"] for row in records)),
        "strict_primary_domain_counts": dict(Counter(row["primary_domain"] for row in records if row["strict_source_eligible"])),
        "liability_theory_counts": dict(Counter(tag for row in records for tag in row.get("liability_theories", []))),
        "exclusion_counts": dict(Counter(reason for row in records for reason in row["exclusion_reasons"])),
    }
    saved_records = select_candidate_pool(records, args.candidate_target, args.seed)
    summary["saved_candidate_pool"] = {
        "count": len(saved_records), "strict_source_eligible": sum(row["strict_source_eligible"] for row in saved_records),
        "rejected": sum(not row["strict_source_eligible"] for row in saved_records),
        "primary_domain_counts": dict(Counter(row["primary_domain"] for row in saved_records)),
        "selection": "deterministic 80% strict / 20% rejected with preferred primary-domain stratification and hash fill",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        write_jsonl(candidate_path, saved_records)
        qc_rows = [{key: value for key, value in row.items() if key not in {"full_opinion_text", "main_opinion_text", "separate_opinions", "판례내용"}} for row in saved_records]
        write_csv(qc_path, qc_rows)
        write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
