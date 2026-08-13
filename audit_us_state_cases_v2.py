from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline_v2 import DATE_END, DATE_START, VERSION
from pipeline_v2.hf_api import dataset_info, iter_filtered_rows
from pipeline_v2.io_utils import guard_outputs, normalized_whitespace, write_csv, write_json
from pipeline_v2.rules import REGION, classify_domain, civil_liability_candidate, select_main_opinion, state_from_jurisdiction


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Audit all COLD Cases state courts of last resort and freeze five states.")
    result.add_argument("--start-date", default=DATE_START)
    result.add_argument("--end-date", default=DATE_END)
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--preserve-states-from", type=Path, help="Keep a previously frozen five-state set if every state remains viable after the full audit.")
    result.add_argument("--reuse-complete-audit", type=Path, help="Reuse a previously generated complete revised-taxonomy audit CSV without rescanning remote shards.")
    result.add_argument("--page-size", type=int, default=20)
    result.add_argument("--min-opinion-chars", type=int, default=1200)
    result.add_argument("--source-mode", choices=("parquet", "datasets-server"), default="parquet")
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--allow-partial", action="store_true", help="Permit partial datasets-server results for smoke diagnostics only.")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def choose_states(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    ranked = sorted(rows, key=lambda row: (-row["selection_score"], row["state"]))
    metadata_only = bool(rows) and all(row.get("usable_full_text_count_method") for row in rows)
    if metadata_only:
        viable = [row for row in ranked if row["civil_liability_candidate_count"] >= 30 and row["published_count"] >= 25 and row["domain_coverage_count"] >= 3]
    else:
        viable = [row for row in ranked if row["civil_liability_candidate_count"] >= 30 and row["usable_full_text_count"] >= 25 and row["domain_coverage_count"] >= 3]
    pool = viable if len(viable) >= 5 else ranked
    selected: list[dict[str, Any]] = []
    for region in ("Northeast", "Midwest", "South", "West"):
        candidate = next((row for row in pool if row["region"] == region and row not in selected), None)
        if candidate:
            selected.append(candidate)
    for row in pool:
        if len(selected) == 5:
            break
        if row not in selected:
            selected.append(row)
    rule = ("highest deterministic metadata availability score subject to >=30 liability candidates, >=25 published records, >=3 core domains; opinion-text usability is rechecked during candidate collection; one state per Census region where feasible, then highest remaining score; alphabetical tie-break" if metadata_only else "highest deterministic availability score subject to >=30 liability candidates, >=25 usable opinions, >=3 core domains; one state per Census region where feasible, then highest remaining score; alphabetical tie-break")
    return selected[:5], rule


def parquet_audit(start_date: str, end_date: str) -> tuple[list[dict[str, Any]], str, int]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for the complete Parquet audit; install requirements-v2.txt") from exc
    with urllib.request.urlopen("https://huggingface.co/api/datasets/harvard-lil/cold-cases", timeout=120) as response:
        repo = json.load(response)
    revision = repo.get("sha") or "main-unresolved"
    names = sorted(item["rfilename"] for item in repo.get("siblings", []) if item.get("rfilename", "").endswith(".parquet"))
    if not names:
        raise RuntimeError("No Parquet files found in the COLD Cases repository metadata")
    urls = [f"https://huggingface.co/datasets/harvard-lil/cold-cases/resolve/{revision}/{name}" for name in names]
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    sql = f"""
        WITH base AS (
          SELECT court_jurisdiction,
                 lower(coalesce(case_name, '') || ' ' || coalesce(case_name_full, '') || ' ' || coalesce(nature_of_suit, '')) AS search_text,
                 lower(coalesce(precedential_status, '')) AS publication
          FROM read_parquet(?, union_by_name=true)
          WHERE court_type = 'S' AND date_filed BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ), tagged AS (
          SELECT *,
            regexp_matches(search_text, 'neglig|injur|death|malpractice|professional|product|defect|warn|vicarious|respondeat|supervis|damage|liability|tort|premises|defamation|privacy|nuisance|conversion') AS civil,
            CASE
              WHEN regexp_matches(search_text, 'malpractice|medical|physician|hospital|professional neglig') THEN 'medical_professional_liability'
              WHEN regexp_matches(search_text, 'product|defect|failure to warn') THEN 'product_liability'
              WHEN regexp_matches(search_text, 'neglig|injur|death|premises|damage|tort') THEN 'general_negligence_personal_injury'
              ELSE 'other_civil_liability' END AS domain
          FROM base
        )
        SELECT court_jurisdiction,
          count(*) AS total_state_supreme_cases,
          count(*) FILTER (WHERE civil) AS civil_liability_candidate_count,
          count(*) FILTER (WHERE civil AND domain='general_negligence_personal_injury') AS general_negligence_count,
          count(*) FILTER (WHERE civil AND domain='medical_professional_liability') AS medical_professional_count,
          count(*) FILTER (WHERE civil AND domain='product_liability') AS product_liability_count,
          count(*) FILTER (WHERE civil AND domain='other_civil_liability') AS other_civil_liability_count,
          count(*) FILTER (WHERE civil AND regexp_matches(search_text, 'vicarious|respondeat|supervis|scope of employment')) AS employer_supervision_tag_count,
          count(*) FILTER (WHERE publication LIKE '%published%' AND publication NOT LIKE '%unpublished%') AS published_count
        FROM tagged GROUP BY court_jurisdiction ORDER BY court_jurisdiction
    """
    cursor = connection.execute(sql, [urls])
    columns = [item[0] for item in cursor.description]
    raw_rows = [dict(zip(columns, values)) for values in cursor.fetchall()]
    rows = []
    for raw in raw_rows:
        state = state_from_jurisdiction(raw["court_jurisdiction"])
        if not state:
            continue
        row = {"state": state, "region": REGION.get(state, "Unknown"), "primary_court_jurisdiction": raw["court_jurisdiction"]}
        row.update({key: int(raw[key]) for key in columns if key != "court_jurisdiction"})
        row["usable_full_text_count"] = 0
        row["usable_full_text_count_method"] = "deferred_to_candidate_collection; metadata audit does not download nested opinion text"
        row["domain_coverage_count"] = sum(row[key] > 0 for key in ("general_negligence_count", "medical_professional_count", "product_liability_count", "other_civil_liability_count"))
        row["selection_score"] = row["civil_liability_candidate_count"] + row["published_count"] // 10 + row["domain_coverage_count"] * 25
        rows.append(row)
    return rows, revision, sum(row["total_state_supreme_cases"] for row in rows)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    audit_path = args.output_dir / "us_state_availability_audit.csv"
    selection_path = args.output_dir / "us_state_selection.json"
    guard_outputs((audit_path, selection_path), overwrite=args.overwrite, resume=args.resume)
    if args.resume and audit_path.exists() and selection_path.exists():
        print(f"resume: frozen state selection retained at {selection_path}")
        return 0
    if args.reuse_complete_audit:
        with args.reuse_complete_audit.open(encoding="utf-8-sig", newline="") as handle:
            audit_rows = list(csv.DictReader(handle))
        numeric_fields = {
            "total_state_supreme_cases", "civil_liability_candidate_count", "general_negligence_count",
            "medical_professional_count", "product_liability_count", "other_civil_liability_count",
            "employer_supervision_tag_count", "published_count", "usable_full_text_count",
            "domain_coverage_count", "selection_score",
        }
        for row in audit_rows:
            for key in numeric_fields:
                if key in row and row[key] != "":
                    row[key] = int(row[key])
        prior = json.loads(args.preserve_states_from.read_text(encoding="utf-8")) if args.preserve_states_from else {}
        revision = prior.get("source_revision", "complete-audit-revision-unresolved")
        scanned = sum(int(row["total_state_supreme_cases"]) for row in audit_rows)
        partial_seen = False
    elif args.source_mode == "parquet":
        if args.limit:
            raise ValueError("--limit is available only with --source-mode datasets-server")
        audit_rows, revision, scanned = parquet_audit(args.start_date, args.end_date)
        partial_seen = False
    else:
        where = f'"court_type"=\'S\' AND "date_filed">=\'{args.start_date}\' AND "date_filed"<=\'{args.end_date}\''
        cache_dir = args.output_dir / "cache" / "hf_us_audit"
        metrics: dict[str, Counter[str]] = defaultdict(Counter)
        jurisdictions: dict[str, Counter[str]] = defaultdict(Counter)
        scanned = 0
        partial_seen = False
        for row, meta in iter_filtered_rows("harvard-lil/cold-cases", where=where, page_size=args.page_size, cache_dir=cache_dir, limit=args.limit, allow_partial=args.allow_partial):
            scanned += 1
            partial_seen = partial_seen or meta["partial"]
            state = state_from_jurisdiction(row.get("court_jurisdiction"))
            if not state:
                continue
            jurisdiction = normalized_whitespace(row.get("court_jurisdiction"))
            jurisdictions[state][jurisdiction] += 1
            bucket = metrics[state]
            bucket["total_state_supreme_cases"] += 1
            opinion = select_main_opinion(row, minimum_chars=args.min_opinion_chars)
            text = "\n".join(normalized_whitespace(row.get(key)) for key in ("case_name", "nature_of_suit", "summary", "syllabus") if row.get(key)) + "\n" + opinion["main_opinion_text"]
            civil, _, exclusions = civil_liability_candidate(text)
            if civil:
                bucket["civil_liability_candidate_count"] += 1
                domain = classify_domain(text)["case_domain"]
                bucket[domain] += 1
                if opinion["main_opinion_usable"]:
                    bucket["usable_full_text_count"] += 1
                status = normalized_whitespace(row.get("precedential_status")).lower()
                if status in {"published", "precedential"} or "published" in status:
                    bucket["published_count"] += 1
        audit_rows = []
        domain_columns = {
            "general_negligence_count": "general_negligence_personal_injury",
            "medical_professional_count": "medical_professional_liability",
            "product_liability_count": "product_liability",
            "other_civil_liability_count": "other_civil_liability",
        }
        for state, bucket in metrics.items():
            row: dict[str, Any] = {"state": state, "region": REGION.get(state, "Unknown")}
            row.update({key: bucket[key] for key in ("total_state_supreme_cases", "civil_liability_candidate_count", "usable_full_text_count", "published_count")})
            row.update({column: bucket[domain] for column, domain in domain_columns.items()})
            row["domain_coverage_count"] = sum(row[column] > 0 for column in domain_columns)
            row["primary_court_jurisdiction"] = jurisdictions[state].most_common(1)[0][0]
            row["selection_score"] = row["usable_full_text_count"] * 3 + row["civil_liability_candidate_count"] + row["domain_coverage_count"] * 25
            audit_rows.append(row)
    selected, rule = choose_states(audit_rows)
    if args.preserve_states_from:
        frozen_payload = json.loads(args.preserve_states_from.read_text(encoding="utf-8"))
        frozen_names = [entry["state"] for entry in frozen_payload.get("selected_states", [])]
        by_state = {row["state"]: row for row in audit_rows}
        preserved = [by_state[state] for state in frozen_names if state in by_state]
        viable = all(
            row["civil_liability_candidate_count"] >= 30
            and row["published_count"] >= 25
            and row["domain_coverage_count"] >= 3
            for row in preserved
        )
        if len(preserved) != 5 or not viable:
            raise RuntimeError("Previously frozen states are not all viable under the complete revised audit; inspect evidence before changing them")
        selected = preserved
        rule = "preserved previously frozen five-state set after complete revised-taxonomy audit confirmed >=30 liability candidates, >=25 published records, and >=3 primary domains for every state"
    if args.source_mode != "parquet":
        revision = "unresolved"
        try:
            info = dataset_info("harvard-lil/cold-cases")
            revision = info.get("dataset_info", {}).get("dataset_revision") or info.get("dataset_revision") or "server-current"
        except RuntimeError:
            revision = "server-current-unresolved"
    payload = {
        "collection_version": VERSION, "source_dataset": "harvard-lil/cold-cases", "source_config": "default",
        "source_revision": revision, "date_window": [args.start_date, args.end_date], "court_type_required": "S",
        "selection_rule": rule, "audit_source_mode": "reused-complete-parquet-audit" if args.reuse_complete_audit else args.source_mode, "partial_audit": bool(args.limit) or partial_seen, "source_rows_scanned": scanned,
        "selected_states": [{key: row[key] for key in ("state", "region", "primary_court_jurisdiction", "selection_score", "total_state_supreme_cases", "civil_liability_candidate_count", "usable_full_text_count", "general_negligence_count", "medical_professional_count", "product_liability_count", "other_civil_liability_count")} for row in selected],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.dry_run:
        if args.limit or partial_seen:
            raise SystemExit("Refusing to freeze states from a partial audit; use --dry-run for diagnostics or obtain a complete source index")
        write_csv(audit_path, sorted(audit_rows, key=lambda row: row["state"]))
        write_json(selection_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
