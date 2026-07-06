from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

from pipeline.io_utils import parse_flags, read_csv, require_overwrite, write_case_table, write_jsonl
from pipeline.text_utils import (
    excerpt,
    extract_fact_section,
    find_jurisdiction_signals,
    find_legal_signals,
    neutralize_loaded_terms,
    normalize_whitespace,
    remove_named_signals,
    strip_case_citations,
    strip_statute_references,
    truncate_preserving_sentence,
    unique,
)


LOGGER = logging.getLogger(__name__)


def select_source_text(row: pd.Series) -> str:
    for column in ["reasoning_text", "clean_text", "raw_text"]:
        if column in row.index:
            text = normalize_whitespace(row.get(column, ""))
            if text:
                return text
    return ""


def build_fact_pattern(row: pd.Series, min_fact_chars: int, max_fact_chars: int) -> tuple[dict | None, dict | None]:
    origin = str(row.get("case_origin", "") or row.get("jurisdiction", "")).upper()
    if origin not in {"KR", "US"}:
        origin = "KR" if str(row.get("jurisdiction", "")).lower() == "korea" else "US"

    raw_text = select_source_text(row)
    source_title = str(row.get("title", "") or row.get("case_name", ""))
    court = str(row.get("court", "") or row.get("court_name", ""))
    case_id = str(row.get("case_id", ""))
    jurisdiction = str(row.get("jurisdiction", "") or ("Korea" if origin == "KR" else "Unknown"))
    flags = set(parse_flags(row.get("quality_flags", "")))
    removed_signals: list[str] = []

    fact_text, found_section = extract_fact_section(raw_text, origin)
    if not found_section:
        flags.add("no_fact_section_detected")

    for cleaner in [strip_case_citations, strip_statute_references]:
        fact_text, removed = cleaner(fact_text)
        removed_signals.extend(removed)

    fact_text, removed = remove_named_signals(fact_text, title=source_title, court=court)
    removed_signals.extend(removed)
    fact_text, removed = neutralize_loaded_terms(fact_text)
    removed_signals.extend(removed)
    fact_text = truncate_preserving_sentence(fact_text, max_fact_chars)

    fact_ko, fact_en = (fact_text, None) if origin == "KR" else (None, fact_text)

    remaining_legal = find_legal_signals(fact_text)
    remaining_jurisdiction = find_jurisdiction_signals(fact_text)
    if remaining_legal:
        flags.add("legal_term_leakage")
        flags.add("legal_conclusion_may_remain")
    if remaining_jurisdiction:
        flags.add("jurisdiction_signal_may_remain")
    if len(fact_text) < min_fact_chars:
        flags.add("too_short")
    if len(fact_text) >= max_fact_chars:
        flags.add("too_long")
    flags.add("translation_needed")
    if {"no_fact_section_detected", "legal_term_leakage", "jurisdiction_signal_may_remain"} & flags:
        flags.add("manual_review_recommended")

    status = "pass"
    if "too_short" in flags or not fact_text:
        status = "fail"
    elif "manual_review_recommended" in flags or "too_long" in flags:
        status = "warning"

    qc = {
        "raw_length_chars": len(raw_text),
        "fact_length_chars": len(fact_text),
        "legal_signal_count": len(remaining_legal),
        "legal_signal_terms": remaining_legal,
        "status": status,
    }

    if not fact_text:
        flags.add("extraction_failed")
        failure = {
            "case_id": case_id,
            "reason": "empty_fact_candidate",
            "source_text_excerpt": excerpt(raw_text),
            "quality_flags": sorted(flags),
        }
        return None, failure

    record = {
        "case_id": case_id,
        "case_origin": origin,
        "jurisdiction": jurisdiction,
        "source_title": source_title,
        "raw_text_excerpt": excerpt(raw_text),
        "neutral_fact_ko": fact_ko,
        "neutral_fact_en": fact_en,
        "neutralization_method": "rule_based",
        "removed_legal_signals": unique(removed_signals),
        "quality_flags": sorted(flags),
        "qc": qc,
    }

    if status == "fail":
        failure = {
            "case_id": case_id,
            "reason": "fact_candidate_failed_qc",
            "source_text_excerpt": excerpt(raw_text),
            "quality_flags": sorted(flags),
        }
        return record, failure
    return record, None


def run(args: argparse.Namespace) -> dict[str, Path | int]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    failures_path = Path(args.failures_output)
    qc_path = Path(args.qc_output)
    case_table_path = Path(args.case_table_output)

    for path in [output_path, failures_path, qc_path, case_table_path]:
        require_overwrite(path, args.overwrite)

    if not input_path.exists() and args.input == "outputs/preprocessed_cases.csv":
        fallback_path = Path("outputs/case_table.csv")
        if fallback_path.exists():
            LOGGER.warning("Input not found at %s; using %s", input_path, fallback_path)
            input_path = fallback_path

    input_df = read_csv(input_path)
    if args.limit:
        input_df = input_df.head(args.limit).copy()

    write_case_table(input_df, case_table_path, overwrite=True)
    case_table = read_csv(case_table_path)

    records: list[dict] = []
    failures: list[dict] = []
    qc_rows: list[dict] = []

    for _, row in case_table.iterrows():
        record, failure = build_fact_pattern(row, args.min_fact_chars, args.max_fact_chars)
        if record:
            records.append(record)
            qc_rows.append(
                {
                    "case_id": record["case_id"],
                    "case_origin": record["case_origin"],
                    "jurisdiction": record["jurisdiction"],
                    "status": record["qc"]["status"],
                    "raw_length_chars": record["qc"]["raw_length_chars"],
                    "fact_length_chars": record["qc"]["fact_length_chars"],
                    "legal_signal_count": record["qc"]["legal_signal_count"],
                    "legal_signal_terms": "; ".join(record["qc"]["legal_signal_terms"]),
                    "quality_flags": "; ".join(record["quality_flags"]),
                    "removed_legal_signals": "; ".join(record["removed_legal_signals"]),
                }
            )
        if failure:
            failures.append(failure)

    write_jsonl(output_path, records, overwrite=True)
    write_jsonl(failures_path, failures, overwrite=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    return {
        "case_table": case_table_path,
        "fact_patterns": output_path,
        "failures": failures_path,
        "qc": qc_path,
        "records": len(records),
        "failures_count": len(failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based neutral fact pattern candidates.")
    parser.add_argument("--input", default="outputs/preprocessed_cases.csv")
    parser.add_argument("--output", default="outputs/fact_patterns.jsonl")
    parser.add_argument("--failures-output", default="outputs/fact_pattern_failures.jsonl")
    parser.add_argument("--qc-output", default="outputs/fact_pattern_qc.csv")
    parser.add_argument("--case-table-output", default="outputs/case_table.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-fact-chars", type=int, default=200)
    parser.add_argument("--max-fact-chars", type=int, default=5_000)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    paths = run(args)
    print("Fact pattern build complete")
    print(f"records: {paths['records']}")
    print(f"failures: {paths['failures_count']}")
    for label in ["case_table", "fact_patterns", "failures", "qc"]:
        print(f"{label}: {paths[label]}")


if __name__ == "__main__":
    main()
