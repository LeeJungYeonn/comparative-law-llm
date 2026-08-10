from __future__ import annotations

import argparse
import json
import math
import urllib.request
from collections import Counter
from pathlib import Path

from pipeline_v2 import DATE_END, DATE_START, SEED, VERSION
from pipeline_v2.collect import collection_funnel, evaluate_us_row
from pipeline_v2.hf_api import iter_filtered_rows
from pipeline_v2.io_utils import guard_outputs, write_csv, write_json, write_jsonl


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Collect civil-liability candidates from five frozen state high courts.")
    result.add_argument("--states-from", type=Path, default=Path("outputs_v2/us_state_selection.json"))
    result.add_argument("--start-date", default=DATE_START)
    result.add_argument("--end-date", default=DATE_END)
    result.add_argument("--candidate-target", type=int, default=300)
    result.add_argument("--seed", type=int, default=SEED)
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--page-size", type=int, default=10)
    result.add_argument("--source-mode", choices=("parquet", "datasets-server"), default="parquet")
    result.add_argument("--parquet-shards", type=int, default=2, help="Deterministic leading shard sampling frame; each shard is about 1 GB compressed.")
    result.add_argument("--min-opinion-chars", type=int, default=1200)
    result.add_argument("--limit", type=int, default=0, help="Maximum rows scanned per state.")
    result.add_argument("--allow-partial", action="store_true", help="Allow a partial datasets-server sampling frame; every returned record is still QC'd individually.")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def parquet_candidate_rows(selected: list[dict], revision: str, shard_count: int, per_state_target: int, seed: int, start_date: str, end_date: str):
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for Parquet collection; install requirements-v2.txt") from exc
    with urllib.request.urlopen("https://huggingface.co/api/datasets/harvard-lil/cold-cases", timeout=120) as response:
        repo = json.load(response)
    revision = revision if revision and "unresolved" not in revision else repo.get("sha") or "main"
    names = sorted(item["rfilename"] for item in repo.get("siblings", []) if item.get("rfilename", "").endswith(".parquet"))[:shard_count]
    urls = [f"https://huggingface.co/datasets/harvard-lil/cold-cases/resolve/{revision}/{name}" for name in names]
    jurisdictions = ",".join("'" + entry["primary_court_jurisdiction"].replace("'", "''") + "'" for entry in selected)
    civil = "negligence|personal injury|wrongful death|medical malpractice|professional negligence|product liability|products liability|defective product|failure to warn|premises liability|vicarious liability|respondeat superior|negligent supervision|compensatory damages|civil damages"
    sql = f"""
      WITH base AS (
        SELECT *, lower(coalesce(case_name, '') || ' ' || coalesce(nature_of_suit, '') || ' ' ||
          coalesce(array_to_string(list_transform(opinions, x -> x.opinion_text), ' '), '')) AS search_text
        FROM read_parquet(?, union_by_name=true)
        WHERE court_type='S' AND court_jurisdiction IN ({jurisdictions})
          AND date_filed BETWEEN DATE '{start_date}' AND DATE '{end_date}'
      ), civil AS (
        SELECT *, row_number() OVER (PARTITION BY court_jurisdiction ORDER BY hash(cast(id AS varchar) || '{seed}')) AS sample_rank
        FROM base WHERE regexp_matches(search_text, '{civil}', 'i')
      )
      SELECT * EXCLUDE (search_text, sample_rank) FROM civil WHERE sample_rank <= {per_state_target}
      ORDER BY court_jurisdiction, sample_rank
    """
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs"); connection.execute("LOAD httpfs")
    cursor = connection.execute(sql, [urls])
    columns = [item[0] for item in cursor.description]
    for values in cursor.fetchall():
        yield dict(zip(columns, values)), {"partial": shard_count < len(repo.get("siblings", [])), "truncated_cells": [], "parquet_shards": names, "source_revision": revision}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    candidate_path = args.output_dir / "us_state_highcourt_candidates.jsonl"
    qc_path = args.output_dir / "us_state_highcourt_candidates_qc.csv"
    summary_path = args.output_dir / "us_collection_deterministic_summary.json"
    guard_outputs((candidate_path, qc_path, summary_path), overwrite=args.overwrite, resume=args.resume)
    if args.resume and candidate_path.exists() and qc_path.exists() and summary_path.exists():
        print(f"resume: existing completed U.S. collection retained at {candidate_path}")
        return 0
    selection = json.loads(args.states_from.read_text(encoding="utf-8"))
    selected = selection.get("selected_states") or []
    if len(selected) != 5:
        raise ValueError(f"State selection must contain exactly five frozen states; got {len(selected)}")
    per_state_target = math.ceil(args.candidate_target / 5)
    parquet_per_state_target = math.ceil(per_state_target * 1.35)
    records = []
    scanned = 0
    scanned_by_state: Counter[str] = Counter()
    candidate_by_state: Counter[str] = Counter()
    partial_seen = False
    if args.source_mode == "parquet":
        source_rows = parquet_candidate_rows(selected, selection.get("source_revision", ""), args.parquet_shards, parquet_per_state_target, args.seed, args.start_date, args.end_date)
        iterators = [(None, source_rows)]
    else:
        iterators = []
        for entry in selected:
            state, jurisdiction = entry["state"], entry["primary_court_jurisdiction"]
            where = f'"court_type"=\'S\' AND "court_jurisdiction"=\'{jurisdiction}\' AND "date_filed">=\'{args.start_date}\' AND "date_filed"<=\'{args.end_date}\''
            cache_dir = args.output_dir / "cache" / "hf_us_collection" / state.replace(" ", "_")
            iterators.append((state, iter_filtered_rows("harvard-lil/cold-cases", where=where, page_size=args.page_size, cache_dir=cache_dir, limit=args.limit, allow_partial=args.allow_partial)))
    for expected_state, iterator in iterators:
        for row, meta in iterator:
            state = expected_state or next(entry["state"] for entry in selected if entry["primary_court_jurisdiction"] == str(row.get("court_jurisdiction")))
            scanned += 1
            scanned_by_state[state] += 1
            partial_seen = partial_seen or meta["partial"]
            record = evaluate_us_row(row, start_date=args.start_date, end_date=args.end_date, min_chars=args.min_opinion_chars)
            if record["civil_liability_evidence"]:
                record["source_cells_truncated"] = meta["truncated_cells"]
                record["source_filter_partial"] = meta["partial"]
                if meta["truncated_cells"]:
                    record["exclusion_reasons"] = list(dict.fromkeys([*record["exclusion_reasons"], "datasets_server_cell_truncated"]))
                    record["strict_source_eligible"] = False
                records.append(record)
                candidate_by_state[state] += 1
            if args.source_mode == "datasets-server" and candidate_by_state[state] >= per_state_target:
                break
    summary = {
        "collection_version": VERSION, "source_dataset": "harvard-lil/cold-cases", "source_config": "default",
        "source_revision": selection.get("source_revision", "unresolved"), "date_window": [args.start_date, args.end_date],
        "frozen_states": [entry["state"] for entry in selected], "candidate_target": args.candidate_target,
        "partial_run": bool(args.limit), "source_mode": args.source_mode, "parquet_shards": args.parquet_shards if args.source_mode == "parquet" else None,
        "source_filter_partial": partial_seen, "seed": args.seed, "scanned_by_state": dict(scanned_by_state),
        "candidate_by_state": dict(candidate_by_state), "funnel": collection_funnel(records, scanned),
        "domain_counts": dict(Counter(row["case_domain"] for row in records)),
        "exclusion_counts": dict(Counter(reason for row in records for reason in row["exclusion_reasons"])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        write_jsonl(candidate_path, records)
        qc_rows = [{key: value for key, value in row.items() if key not in {"full_opinion_text", "main_opinion_text", "separate_opinions", "history", "cross_reference"}} for row in records]
        write_csv(qc_path, qc_rows)
        write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
