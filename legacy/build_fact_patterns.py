from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import pandas as pd

from pipeline.io_utils import parse_flags, read_csv, require_overwrite, write_case_table, write_jsonl
from pipeline.text_utils import (
    cleanup_kr_fact_text,
    compact_inline,
    contains_criminal_signal,
    contains_order_or_claim_section,
    criminal_signal_terms,
    excerpt,
    extract_fact_section_with_metadata,
    find_jurisdiction_signals,
    find_legal_signals,
    is_procedural_only_kr_fact,
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


def row_context_text(row: pd.Series, raw_text: str) -> str:
    parts = [raw_text]
    for column in ["title", "case_name", "court", "court_name", "collection_notes", "quality_flags"]:
        if column in row.index:
            parts.append(str(row.get(column, "")))
    return "\n".join(parts)


def make_failure(
    *,
    case_id: str,
    reason: str,
    raw_text: str,
    flags: set[str],
    qc: dict[str, object],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "reason": reason,
        "source_text_excerpt": excerpt(raw_text),
        "quality_flags": sorted(flags),
        "qc": qc,
    }


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
    context_text = row_context_text(row, raw_text)
    criminal_terms = criminal_signal_terms(context_text, origin)
    contains_criminal = contains_criminal_signal(context_text, origin)

    base_qc = {
        "raw_length_chars": len(raw_text),
        "fact_length_chars": 0,
        "legal_signal_count": 0,
        "legal_signal_terms": [],
        "has_fact_heading": False,
        "fact_extraction_method": "failed",
        "contains_order_or_claim_section": False,
        "contains_criminal_signal": contains_criminal,
        "criminal_signal_terms": criminal_terms,
        "excluded_reason": "",
        "manual_review_recommended": False,
        "status": "fail",
    }

    if contains_criminal:
        flags.add("contains_criminal_signal")
        flags.add("extraction_failed")
        base_qc["excluded_reason"] = "criminal_case_signal"
        return None, make_failure(
            case_id=case_id,
            reason="criminal_case_signal",
            raw_text=raw_text,
            flags=flags,
            qc=base_qc,
        )

    if origin == "KR" and (
        compact_inline(raw_text).startswith(("제1심 판결의 인용", "1. 제1심 판결의 인용"))
        or "이 법원이 적을 이유는" in raw_text
    ):
        flags.add("extraction_failed")
        flags.add("manual_review_recommended")
        base_qc["excluded_reason"] = "procedural_section_signal"
        base_qc["manual_review_recommended"] = True
        return None, make_failure(
            case_id=case_id,
            reason="procedural_section_signal",
            raw_text=raw_text,
            flags=flags,
            qc=base_qc,
        )

    fact_text, extraction_meta = extract_fact_section_with_metadata(raw_text, origin)
    found_section = bool(extraction_meta.get("has_fact_heading"))
    extraction_method = str(extraction_meta.get("fact_extraction_method", "failed"))
    excluded_reason = str(extraction_meta.get("excluded_reason", ""))

    if not found_section:
        flags.add("no_fact_section_detected")
    if extraction_method == "fallback":
        flags.add("manual_review_recommended")
    if extraction_method == "failed":
        flags.add("extraction_failed")

    for cleaner in [strip_case_citations, strip_statute_references]:
        fact_text, removed = cleaner(fact_text)
        removed_signals.extend(removed)

    if origin == "KR":
        fact_text, removed = cleanup_kr_fact_text(fact_text)
        removed_signals.extend(removed)

    fact_text, removed = remove_named_signals(fact_text, title=source_title, court=court)
    removed_signals.extend(removed)
    fact_text, removed = neutralize_loaded_terms(fact_text)
    removed_signals.extend(removed)
    effective_max_chars = min(max_fact_chars, 1_200) if origin == "KR" and extraction_method == "fallback" else max_fact_chars
    fact_text = truncate_preserving_sentence(fact_text, effective_max_chars)

    fact_ko, fact_en = (fact_text, None) if origin == "KR" else (None, fact_text)

    remaining_legal = find_legal_signals(fact_text)
    remaining_jurisdiction = find_jurisdiction_signals(fact_text)
    contains_order_or_claim = contains_order_or_claim_section(fact_text) if origin == "KR" else False
    if remaining_legal:
        flags.add("legal_term_leakage")
        flags.add("legal_conclusion_may_remain")
    if remaining_jurisdiction:
        flags.add("jurisdiction_signal_may_remain")
    if contains_order_or_claim:
        flags.add("contains_order_or_claim_section")
        flags.add("manual_review_recommended")
    procedural_only = origin == "KR" and is_procedural_only_kr_fact(fact_text)
    if procedural_only:
        flags.add("extraction_failed")
        flags.add("manual_review_recommended")
    if len(fact_text) < min_fact_chars:
        flags.add("too_short")
    if len(fact_text) >= max_fact_chars:
        flags.add("too_long")
    flags.add("translation_needed")
    if {"no_fact_section_detected", "legal_term_leakage", "jurisdiction_signal_may_remain"} & flags:
        flags.add("manual_review_recommended")

    status = "pass"
    if procedural_only:
        status = "fail"
    elif "too_short" in flags or not fact_text:
        status = "fail"
    elif "manual_review_recommended" in flags or "too_long" in flags:
        status = "warning"

    qc = {
        "raw_length_chars": len(raw_text),
        "fact_length_chars": len(fact_text),
        "legal_signal_count": len(remaining_legal),
        "legal_signal_terms": remaining_legal,
        "has_fact_heading": found_section,
        "fact_extraction_method": extraction_method,
        "contains_order_or_claim_section": contains_order_or_claim,
        "contains_criminal_signal": contains_criminal,
        "criminal_signal_terms": criminal_terms,
        "excluded_reason": excluded_reason if status == "fail" else "",
        "manual_review_recommended": "manual_review_recommended" in flags,
        "status": status,
    }
    if procedural_only:
        qc["excluded_reason"] = "procedural_only_fact_candidate"

    if not fact_text:
        flags.add("extraction_failed")
        qc["excluded_reason"] = excluded_reason or "empty_fact_candidate"
        failure = make_failure(
            case_id=case_id,
            reason=str(qc["excluded_reason"]),
            raw_text=raw_text,
            flags=flags,
            qc=qc,
        )
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
        qc["excluded_reason"] = str(qc.get("excluded_reason") or excluded_reason or "fact_candidate_failed_qc")
        failure = make_failure(
            case_id=case_id,
            reason=str(qc["excluded_reason"]),
            raw_text=raw_text,
            flags=flags,
            qc=qc,
        )
        return None, failure
    return record, None


def qc_row_from_record(record: dict) -> dict[str, object]:
    qc = record["qc"]
    return {
        "case_id": record["case_id"],
        "case_origin": record["case_origin"],
        "jurisdiction": record["jurisdiction"],
        "status": qc["status"],
        "raw_length_chars": qc["raw_length_chars"],
        "fact_length_chars": qc["fact_length_chars"],
        "legal_signal_count": qc["legal_signal_count"],
        "legal_signal_terms": "; ".join(qc["legal_signal_terms"]),
        "has_fact_heading": qc["has_fact_heading"],
        "fact_extraction_method": qc["fact_extraction_method"],
        "contains_order_or_claim_section": qc["contains_order_or_claim_section"],
        "contains_criminal_signal": qc["contains_criminal_signal"],
        "criminal_signal_terms": "; ".join(qc["criminal_signal_terms"]),
        "excluded_reason": qc["excluded_reason"],
        "manual_review_recommended": qc["manual_review_recommended"],
        "quality_flags": "; ".join(record["quality_flags"]),
        "removed_legal_signals": "; ".join(record["removed_legal_signals"]),
    }


def qc_row_from_failure(failure: dict[str, object], row: pd.Series) -> dict[str, object]:
    qc = failure.get("qc", {})
    return {
        "case_id": failure.get("case_id", ""),
        "case_origin": row.get("case_origin", ""),
        "jurisdiction": row.get("jurisdiction", ""),
        "status": qc.get("status", "fail"),
        "raw_length_chars": qc.get("raw_length_chars", 0),
        "fact_length_chars": qc.get("fact_length_chars", 0),
        "legal_signal_count": qc.get("legal_signal_count", 0),
        "legal_signal_terms": "; ".join(qc.get("legal_signal_terms", [])),
        "has_fact_heading": qc.get("has_fact_heading", False),
        "fact_extraction_method": qc.get("fact_extraction_method", "failed"),
        "contains_order_or_claim_section": qc.get("contains_order_or_claim_section", False),
        "contains_criminal_signal": qc.get("contains_criminal_signal", False),
        "criminal_signal_terms": "; ".join(qc.get("criminal_signal_terms", [])),
        "excluded_reason": qc.get("excluded_reason", failure.get("reason", "")),
        "manual_review_recommended": qc.get("manual_review_recommended", True),
        "quality_flags": "; ".join(failure.get("quality_flags", [])),
        "removed_legal_signals": "",
    }


def summarize_qc(qc_rows: list[dict[str, object]]) -> dict[str, object]:
    df = pd.DataFrame(qc_rows)
    if df.empty:
        return {
            "total_input_cases": 0,
            "excluded_by_origin": {},
            "criminal_excluded_cases": 0,
            "heading_based_extractions": 0,
            "fallback_extractions": 0,
            "extraction_failed_cases": 0,
            "manual_review_recommended_cases": 0,
            "starts_with_order_or_claim_remaining": False,
        }
    excluded = df[df["status"].eq("fail")]
    return {
        "total_input_cases": int(len(df)),
        "excluded_by_origin": {
            key: int(value)
            for key, value in excluded.groupby("case_origin", dropna=False).size().to_dict().items()
        },
        "criminal_excluded_cases": int(df["excluded_reason"].eq("criminal_case_signal").sum()),
        "heading_based_extractions": int(df["fact_extraction_method"].eq("heading_based").sum()),
        "fallback_extractions": int(df["fact_extraction_method"].eq("fallback").sum()),
        "extraction_failed_cases": int(df["fact_extraction_method"].eq("failed").sum()),
        "manual_review_recommended_cases": int(df["manual_review_recommended"].astype(str).str.lower().eq("true").sum()),
        "starts_with_order_or_claim_remaining": False,
    }


def sanity_check_kr_extractor() -> None:
    sample = """주문
1. 피고는 원고에게 1,000원을 지급하라.
청구취지
돈을 지급하라.
이유
1. 기초사실
가. 원고는 공사 현장에서 작업하였다.
나. 피고 직원은 장비를 이동하였다.
2. 판단
피고의 책임을 본다.
"""
    fact_text, metadata = extract_fact_section_with_metadata(sample, "KR")
    assert metadata["fact_extraction_method"] == "heading_based"
    assert "원고는 공사 현장에서 작업하였다" in fact_text
    assert "피고의 책임" not in fact_text


def run(args: argparse.Namespace) -> dict[str, Path | int]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    failures_path = Path(args.failures_output)
    qc_path = Path(args.qc_output)
    case_table_path = Path(args.case_table_output)
    summary_path = Path(args.summary_output)

    for path in [output_path, failures_path, qc_path, case_table_path, summary_path]:
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
        try:
            record, failure = build_fact_pattern(row, args.min_fact_chars, args.max_fact_chars)
        except Exception as exc:
            LOGGER.exception("Fact extraction failed for case_id=%s", row.get("case_id", ""))
            raw_text = select_source_text(row)
            flags = {"extraction_failed", "manual_review_recommended"}
            failure = make_failure(
                case_id=str(row.get("case_id", "")),
                reason=f"exception:{type(exc).__name__}",
                raw_text=raw_text,
                flags=flags,
                qc={
                    "raw_length_chars": len(raw_text),
                    "fact_length_chars": 0,
                    "legal_signal_count": 0,
                    "legal_signal_terms": [],
                    "has_fact_heading": False,
                    "fact_extraction_method": "failed",
                    "contains_order_or_claim_section": False,
                    "contains_criminal_signal": False,
                    "criminal_signal_terms": [],
                    "excluded_reason": f"exception:{type(exc).__name__}",
                    "manual_review_recommended": True,
                    "status": "fail",
                },
            )
            record = None
        if record:
            records.append(record)
            qc_rows.append(qc_row_from_record(record))
        if failure:
            failures.append(failure)
            if not record:
                qc_rows.append(qc_row_from_failure(failure, row))

    write_jsonl(output_path, records, overwrite=True)
    write_jsonl(failures_path, failures, overwrite=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    summary = summarize_qc(qc_rows)
    summary["starts_with_order_or_claim_remaining"] = any(
        compact_inline((record.get("neutral_fact_ko") or record.get("neutral_fact_en") or "")).startswith(("주문", "청구취지"))
        for record in records
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "case_table": case_table_path,
        "fact_patterns": output_path,
        "failures": failures_path,
        "qc": qc_path,
        "summary": summary_path,
        "records": len(records),
        "failures_count": len(failures),
        "summary_data": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based neutral fact pattern candidates.")
    parser.add_argument("--input", default="outputs/preprocessed_cases.csv")
    parser.add_argument("--output", default="outputs/fact_patterns.jsonl")
    parser.add_argument("--failures-output", default="outputs/fact_pattern_failures.jsonl")
    parser.add_argument("--qc-output", default="outputs/fact_pattern_qc.csv")
    parser.add_argument("--case-table-output", default="outputs/case_table.csv")
    parser.add_argument("--summary-output", default="outputs/fact_pattern_summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-fact-chars", type=int, default=200)
    parser.add_argument("--max-fact-chars", type=int, default=5_000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    if args.self_test:
        sanity_check_kr_extractor()
        print("Sanity checks passed")
        return
    paths = run(args)
    print("Fact pattern build complete")
    print(f"records: {paths['records']}")
    print(f"failures: {paths['failures_count']}")
    for label in ["case_table", "fact_patterns", "failures", "qc", "summary"]:
        print(f"{label}: {paths[label]}")
    print("summary")
    for key, value in paths["summary_data"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
