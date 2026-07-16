from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from pipeline.stage1_raw import (
    add_gate_stats,
    apply_duplicate_qc,
    compact,
    grouped_case_numbers,
    length_flags,
    make_raw_record,
    require_outputs,
    status_from_signals,
    stratified_sample_with_fallback,
    summarize_records,
    write_jsonl,
    write_manifest,
    write_qc_csv,
    write_summary,
)


LOGGER = logging.getLogger(__name__)
TEXT_COLUMNS = ("precedent", "raw_text", "text", "facts", "reason", "ruling")
CASE_NUMBER_COLUMNS = ("case_number_or_citation", "case_number", "case_no", "caseno", "사건번호")
COURT_COLUMNS = ("court_name", "court", "court_full_name", "법원명")
DATE_COLUMNS = ("decision_date", "date", "선고일자", "판결선고일")
CASE_NAME_COLUMNS = ("case_name", "casename", "사건명")

INCLUDE_PATTERNS = [r"손해배상", r"불법행위", r"위자료", r"치료비", r"일실수입", r"과실상계", r"주의의무", r"사용자책임", r"공작물책임", r"제조물책임"]
FACT_PATTERNS = [r"인정사실", r"기초사실", r"사실관계", r"다툼 없는 사실", r"사고가 발생", r"상해를 입", r"손해를 입", r"사망하", r"충돌", r"추락", r"의료"]
CRIMINAL_PATTERNS = [r"형사", r"피고인", r"징역", r"집행유예", r"공소사실", r"범죄사실", r"\d{4}\s*고(?:단|합|정)\s*\d+"]
ADMIN_PATTERNS = [r"행정", r"처분취소", r"거부처분", r"영업정지", r"\d{4}\s*(?:구합|구단|누)\s*\d+"]
FAMILY_PATTERNS = [r"이혼", r"친권", r"양육", r"재산분할", r"\d{4}\s*(?:드단|드합|르)\s*\d+"]
IP_PATTERNS = [r"특허", r"상표", r"저작권", r"디자인권"]
INSURANCE_ONLY_PATTERNS = [r"보험금", r"보험계약", r"보험자", r"면책"]
ATTORNEY_FEE_PATTERNS = [r"소송비용", r"변호사보수", r"변호사 비용"]
PROCEDURAL_PATTERNS = [r"관할", r"이송", r"각하", r"항소기간", r"재심", r"소멸시효"]
CONTRACT_PATTERNS = [r"매매계약", r"임대차", r"대여금", r"공사대금", r"약정금", r"계약해제", r"채무불이행"]
TORT_CONTEXT_PATTERNS = [r"불법행위", r"교통사고", r"의료과실", r"추락", r"충돌", r"시설", r"제품", r"안전", r"상해", r"사망"]
SUPREME_PATTERNS = [r"대법원", r"\d{4}\s*다\s*\d+"]
SUBTYPES = [
    ("traffic_accident", [r"교통사고", r"차량", r"자동차", r"충돌", r"운전"]),
    ("medical_professional", [r"의료", r"병원", r"의사", r"수술", r"진료", r"전문가"]),
    ("facility_product_safety", [r"시설", r"공작물", r"제품", r"제조물", r"추락", r"안전"]),
    ("property_economic_harm", [r"재산", r"영업", r"명예", r"신용", r"사기", r"허위"]),
    ("personal_injury", [r"상해", r"부상", r"치료비", r"후유장해", r"사망"]),
    ("intentional_tort", [r"폭행", r"모욕", r"명예훼손", r"사생활"]),
]


def regex_hits(patterns: Iterable[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def first_existing(row: dict[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        if column in row and compact(row.get(column, "")):
            return compact(row.get(column, ""))
    return ""


def first_pattern_label(items: list[tuple[str, list[str]]], text: str, default: str) -> str:
    for label, patterns in items:
        if regex_hits(patterns, text):
            return label
    return default


def extract_case_number(text: str) -> str:
    match = re.search(r"\b\d{4}\s*[가-힣]{1,5}\s*\d+\b", text)
    return compact(match.group(0)) if match else ""


def extract_decision_date(text: str) -> str:
    match = re.search(r"\b((?:18|19|20)\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?", text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def infer_court(text: str) -> str:
    match = re.search(r"([가-힣]{0,12}(?:지방법원|고등법원|대법원)(?:\s*[가-힣]+지원)?)", text[:1000])
    return compact(match.group(1)) if match else ""


def infer_court_level(court_name: str, case_number: str, text: str) -> str:
    preview = f"{court_name}\n{case_number}\n{text[:1000]}"
    if regex_hits(SUPREME_PATTERNS, preview):
        return "supreme"
    if "고등법원" in court_name or re.search(r"\d{4}\s*나\s*\d+", case_number):
        return "appellate"
    if "지방법원" in court_name or re.search(r"\d{4}\s*가(?:합|단|소)\s*\d+", case_number):
        return "trial"
    return "unknown"


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, object] | None:
    raw_text = first_existing(row, [args.text_col] if args.text_col else TEXT_COLUMNS)
    if not raw_text:
        return None
    case_number = first_existing(row, CASE_NUMBER_COLUMNS) or extract_case_number(raw_text)
    court_name = first_existing(row, COURT_COLUMNS) or infer_court(raw_text)
    decision_date = first_existing(row, DATE_COLUMNS) or extract_decision_date(raw_text)
    year_match = re.search(r"\b(18|19|20)\d{2}\b", decision_date)
    year = int(year_match.group(0)) if year_match else None
    case_name = first_existing(row, CASE_NAME_COLUMNS)
    haystack = f"{case_name}\n{case_number}\n{court_name}\n{raw_text[:12000]}"
    include_signals = regex_hits(INCLUDE_PATTERNS, haystack)
    exclude_signals = []
    quality_flags = length_flags(raw_text, args.min_text_chars, args.max_text_chars)
    court_level = infer_court_level(court_name, case_number, raw_text)
    if court_level == "supreme":
        exclude_signals.append("supreme_court_excluded")
    if court_level != "appellate":
        exclude_signals.append("non_appellate_or_unknown_court_level")
    if year is None:
        exclude_signals.append("decision_year_unknown")
    elif year < args.year_min or year > args.year_max:
        exclude_signals.append("decision_year_out_of_range")
    if regex_hits(CRIMINAL_PATTERNS, haystack):
        exclude_signals.append("criminal_case")
    if regex_hits(ADMIN_PATTERNS, haystack):
        exclude_signals.append("administrative_only")
    if regex_hits(FAMILY_PATTERNS, haystack):
        exclude_signals.append("family_only")
    if regex_hits(IP_PATTERNS, haystack):
        exclude_signals.append("ip_only")
    if regex_hits(INSURANCE_ONLY_PATTERNS, haystack) and not regex_hits(TORT_CONTEXT_PATTERNS, haystack):
        exclude_signals.append("insurance_payment_only")
    if regex_hits(ATTORNEY_FEE_PATTERNS, haystack) and not include_signals:
        exclude_signals.append("attorney_fees_or_costs_only")
    if regex_hits(PROCEDURAL_PATTERNS, haystack) and not regex_hits(FACT_PATTERNS, haystack):
        exclude_signals.append("procedural_only_or_no_factual_background")
    if regex_hits(CONTRACT_PATTERNS, haystack) and not regex_hits(TORT_CONTEXT_PATTERNS, haystack):
        exclude_signals.append("contract_only")
    if not include_signals:
        exclude_signals.append("no_civil_damages_include_signal")
    if not regex_hits(FACT_PATTERNS, haystack):
        quality_flags.append("warning_weak_fact_background_signal")
    if court_level == "unknown":
        quality_flags.append("warning_unknown_court_level")
    if not court_name:
        quality_flags.append("warning_unknown_court_name")
    if not case_number:
        quality_flags.append("warning_unknown_case_number")
    exclude_signals.extend(flag for flag in quality_flags if flag in {"too_short_or_no_full_opinion_text", "too_long"})
    subtype = "contract_only" if "contract_only" in exclude_signals else first_pattern_label(SUBTYPES, haystack, "unclear")
    return make_raw_record(
        case_origin="KR",
        jurisdiction="Korea",
        source_dataset=f"{args.dataset}::{args.config}",
        source_record_id=compact(row.get("id", "")),
        source_url_or_citation=case_number,
        case_name=case_name,
        case_number_or_citation=case_number,
        court_name=court_name or "unknown",
        court_level=court_level,
        decision_date=decision_date,
        opinion_type="main",
        procedural_posture="unknown",
        case_subtype=subtype,
        raw_text=raw_text,
        include_signals=include_signals,
        exclude_signals=exclude_signals,
        quality_flags=quality_flags,
        collection_status=status_from_signals(exclude_signals, quality_flags),
    )


def keyword_gate(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    raw_text = first_existing(row, [args.text_col] if args.text_col else TEXT_COLUMNS)
    case_number = first_existing(row, CASE_NUMBER_COLUMNS)
    court_name = first_existing(row, COURT_COLUMNS)
    case_name = first_existing(row, CASE_NAME_COLUMNS)
    haystack = f"{case_name}\n{case_number}\n{court_name}\n{raw_text[:12000]}"
    hits = regex_hits(INCLUDE_PATTERNS, haystack)
    return bool(hits), hits


def collect(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    records = []
    scanned = 0
    keyword_gate_hits = 0
    keyword_gate_skipped = 0
    for scanned, row in enumerate(dataset, start=1):
        if args.scan_limit and scanned > args.scan_limit:
            break
        gate_keep, _ = keyword_gate(row, args)
        if not gate_keep:
            keyword_gate_skipped += 1
            if args.progress_every and scanned % args.progress_every == 0:
                LOGGER.info("scanned=%s keyword_hits=%s qc_candidates=%s gate_skipped=%s", scanned, keyword_gate_hits, len(records), keyword_gate_skipped)
            continue
        keyword_gate_hits += 1
        record = evaluate_row(row, args)
        if record:
            records.append(record)
        if args.preview_only and len(records) >= args.preview_count:
            break
        if args.progress_every and scanned % args.progress_every == 0:
            LOGGER.info("scanned=%s keyword_hits=%s qc_candidates=%s gate_skipped=%s", scanned, keyword_gate_hits, len(records), keyword_gate_skipped)
    grouped_case_numbers(records)
    apply_duplicate_qc(records)
    eligible = [row for row in records if row["collection_status"] == "pass" or (args.include_warning and row["collection_status"] == "warning")]
    selected, sampling_meta = stratified_sample_with_fallback(
        eligible,
        target_count=args.target_count,
        seed=args.seed,
        primary_year_min=args.primary_year_min,
        fallback_year_min=args.fallback_year_min,
        year_max=args.year_max,
    )
    summary = summarize_records(all_records=records, selected_records=selected, args=args)
    summary.update(sampling_meta)
    add_gate_stats(
        summary,
        stream_rows_scanned=scanned,
        keyword_gate_hits=keyword_gate_hits,
        keyword_gate_skipped=keyword_gate_skipped,
        gate_patterns=INCLUDE_PATTERNS,
    )
    return records, selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 raw Korean civil liability case collector.")
    parser.add_argument("--dataset", default="lbox/lbox_open")
    parser.add_argument("--config", default="precedent_corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-col", default="")
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--scan-limit", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2020)
    parser.add_argument("--primary-year-min", type=int, default=2010)
    parser.add_argument("--fallback-year-min", type=int, default=2000)
    parser.add_argument("--min-text-chars", type=int, default=2000)
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--include-warning", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--raw-output", default="outputs/raw/kr_cases_raw.jsonl")
    parser.add_argument("--qc-output", default="outputs/raw/kr_cases_qc.csv")
    parser.add_argument("--summary-output", default="outputs/raw/kr_cases_summary.json")
    parser.add_argument("--manifest-output", default="outputs/manifests/case_manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    all_records, selected, summary = collect(args)
    if args.preview_only:
        print(summary)
        return
    paths = [Path(args.raw_output), Path(args.qc_output), Path(args.summary_output), Path(args.manifest_output)]
    require_outputs(paths, args.overwrite)
    write_jsonl(Path(args.raw_output), selected)
    write_qc_csv(Path(args.qc_output), all_records)
    write_summary(Path(args.summary_output), summary)
    write_manifest(Path(args.manifest_output), [Path(args.raw_output), Path("outputs/raw/ca_cases_raw.jsonl")], overwrite=True)
    print(f"selected={len(selected)} candidates={len(all_records)}")
    print(f"raw={args.raw_output}")
    print(f"qc={args.qc_output}")
    print(f"summary={args.summary_output}")
    print(f"manifest={args.manifest_output}")


if __name__ == "__main__":
    main()
