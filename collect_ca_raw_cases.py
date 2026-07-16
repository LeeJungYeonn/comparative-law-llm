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
    sample_records,
    status_from_signals,
    summarize_records,
    write_jsonl,
    write_manifest,
    write_qc_csv,
    write_summary,
)


LOGGER = logging.getLogger(__name__)
INCLUDE_PATTERNS = [r"\btort\b", r"\bdamages\b", r"\bnegligence\b", r"personal injur", r"wrongful death", r"property damage", r"product liability", r"premises liability", r"medical malpractice", r"misrepresentation", r"\bfraud\b", r"accident"]
FEDERAL_PATTERNS = [r"United States District Court", r"U\.S\. District Court", r"\bN\.D\. Cal\.|\bC\.D\. Cal\.|\bE\.D\. Cal\.|\bS\.D\. Cal\.", r"\b9th Cir\.|Ninth Circuit", r"\bF\. ?Supp\.|\bF\.2d\b|\bF\.3d\b|\bU\.S\.C\."]
CRIMINAL_PATTERNS = [r"\bPeople v\.", r"\bcriminal\b", r"\bhabeas\b", r"\bprison\b", r"\bwarden\b", r"\bconviction\b", r"\bsentence\b"]
ADMIN_PATTERNS = [r"Public Utilities Commission", r"administrative review", r"agency decision", r"writ of mandate", r"mandamus", r"Workers'? Compensation", r"eminent domain", r"\bcondemnation\b"]
EXCLUDE_PATTERNS = {
    "family_only": [r"\bdivorce\b", r"\bchild custody\b", r"\bspousal\b", r"\bmarital\b"],
    "ip_only": [r"\bcopyright\b", r"\bpatent\b", r"\btrademark\b", r"Lanham Act"],
    "insurance_coverage_only": [r"insurance coverage", r"duty to defend", r"\binsurer\b", r"\binsured\b"],
    "attorney_fees_only": [r"attorney fees?", r"attorneys'? fees?", r"Civil Code section 1717"],
    "contract_or_debt_only": [r"contract interpretation", r"promissory note", r"\bdebt collection\b", r"specific performance", r"breach of contract"],
}
PROCEDURAL_PATTERNS = [r"statute of limitations", r"default judgment", r"\bdemurrer\b", r"\bremand\b", r"jurisdiction", r"res judicata"]
FACT_PATTERNS = [r"FACTUAL BACKGROUND", r"\bFACTS\b", r"BACKGROUND", r"was injured", r"were injured", r"was killed", r"collision", r"accident", r"slip", r"fell", r"damaged", r"misrepresented"]
SUBTYPES = [
    ("auto_accident", [r"automobile", r"\bcar\b", r"vehicle", r"collision", r"truck", r"motorist"]),
    ("medical_professional", [r"medical malpractice", r"professional negligence", r"malpractice"]),
    ("product_safety", [r"product liability", r"defective product", r"manufacturer"]),
    ("premises_facility_safety", [r"premises liability", r"slip and fall", r"defective condition"]),
    ("property_economic_harm", [r"property damage", r"trespass", r"nuisance", r"fraud", r"misrepresentation"]),
    ("wrongful_death", [r"wrongful death", r"was killed", r"fatal"]),
    ("personal_injury", [r"personal injur", r"bodily injur", r"was injured", r"were injured"]),
]
POSTURE_PATTERNS = [("summary_judgment", [r"summary judgment"]), ("jury_verdict", [r"jury verdict", r"jury found", r"verdict"]), ("demurrer", [r"\bdemurrer\b"]), ("trial_judgment", [r"judgment after trial", r"bench trial"]), ("appeal", [r"\bappeal\b", r"appeals? from"])]


def regex_hits(patterns: Iterable[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def first_pattern_label(items: list[tuple[str, list[str]]], text: str, default: str) -> str:
    for label, patterns in items:
        if regex_hits(patterns, text):
            return label
    return default


def citations_text(row: dict[str, Any]) -> str:
    citations = row.get("citations") or []
    if not isinstance(citations, list):
        return compact(citations)
    values = []
    for citation in citations:
        if isinstance(citation, dict):
            values.append(compact(citation.get("cite") or citation.get("citation") or ""))
        else:
            values.append(compact(citation))
    return "; ".join(value for value in values if value)


def opinion_chunks(row: dict[str, Any]) -> tuple[str, str, str]:
    opinions = row.get("opinions") or []
    if not isinstance(opinions, list) or not opinions:
        return compact(row.get("opinion_text") or row.get("text") or row.get("raw_text") or ""), "unknown", ""
    main = []
    url = ""
    for opinion in opinions:
        if not isinstance(opinion, dict):
            continue
        text = opinion.get("opinion_text") or opinion.get("text") or ""
        op_type = compact(opinion.get("type") or opinion.get("opinion_type") or "").lower()
        if opinion.get("download_url") and not url:
            url = compact(opinion.get("download_url"))
        if text and (not op_type or op_type in {"majority", "lead", "main", "unanimous"}):
            main.append(text)
    if main:
        return "\n\n".join(main), "majority/main", url
    first = opinions[0] if isinstance(opinions[0], dict) else {}
    return compact(first.get("opinion_text") or first.get("text") or ""), compact(first.get("type") or first.get("opinion_type") or "unknown"), url


def is_california_state(row: dict[str, Any]) -> bool:
    haystack = "\n".join(compact(row.get(key, "")) for key in ["court_full_name", "court_short_name", "court_jurisdiction", "court_type"])
    if regex_hits(FEDERAL_PATTERNS, haystack + "\n" + citations_text(row)):
        return False
    court = compact(row.get("court_full_name") or row.get("court_short_name") or "")
    jurisdiction = compact(row.get("court_jurisdiction", "")).lower()
    court_type = compact(row.get("court_type", "")).upper()
    return court_type in {"SA", "ST", ""} and (re.search(r"California (Court of Appeal|Supreme Court)", court, flags=re.IGNORECASE) or jurisdiction in {"california", "cal.", "ca"})


def court_level(row: dict[str, Any]) -> str:
    court = compact(row.get("court_full_name") or row.get("court_short_name") or "")
    if re.search(r"Supreme Court", court, flags=re.IGNORECASE):
        return "supreme"
    if re.search(r"Court of Appeal|Court of Appeals", court, flags=re.IGNORECASE):
        return "appellate"
    return "unknown"


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, object] | None:
    raw_text, opinion_type, url = opinion_chunks(row)
    text = compact(raw_text)
    year_match = re.search(r"\b(18|19|20)\d{2}\b", compact(row.get("date_filed", "")))
    year = int(year_match.group(0)) if year_match else None
    if year is not None and args.year_min and year < args.year_min:
        return None
    if year is not None and args.year_max and year > args.year_max:
        return None
    metadata = "\n".join(compact(row.get(key, "")) for key in ["case_name", "case_name_full", "case_name_short", "court_full_name", "court_short_name", "headmatter", "summary", "syllabus", "disposition"])
    haystack = f"{metadata}\n{text[:12000]}"
    include_signals = regex_hits(INCLUDE_PATTERNS, haystack)
    exclude_signals = []
    quality_flags = length_flags(text, args.min_text_chars, args.max_text_chars)
    if not is_california_state(row):
        exclude_signals.append("not_california_state_court_or_federal")
    if regex_hits(FEDERAL_PATTERNS, haystack):
        exclude_signals.append("federal_case")
    if regex_hits(CRIMINAL_PATTERNS, haystack):
        exclude_signals.append("criminal_habeas_prison_or_sentencing")
    if regex_hits(ADMIN_PATTERNS, haystack):
        exclude_signals.append("administrative_only")
    for reason, patterns in EXCLUDE_PATTERNS.items():
        if regex_hits(patterns, haystack):
            if reason == "contract_or_debt_only" and regex_hits([r"\bfraud\b", r"misrepresentation", r"injur", r"property damage"], haystack):
                quality_flags.append("warning_contract_signal_with_possible_tort_facts")
            else:
                exclude_signals.append(reason)
    if regex_hits(PROCEDURAL_PATTERNS, haystack) and not regex_hits(FACT_PATTERNS, haystack):
        exclude_signals.append("procedural_only_or_no_factual_background")
    if not include_signals:
        exclude_signals.append("no_tort_damages_include_signal")
    if not regex_hits(FACT_PATTERNS, haystack):
        quality_flags.append("warning_weak_fact_background_signal")
    exclude_signals.extend(flag for flag in quality_flags if flag in {"too_short_or_no_full_opinion_text", "too_long"})
    title = compact(row.get("case_name") or row.get("case_name_full") or row.get("case_name_short") or "")
    citation = citations_text(row)
    return make_raw_record(
        case_origin="CA",
        jurisdiction="California",
        source_dataset=args.dataset,
        source_record_id=compact(row.get("id", "")),
        source_url_or_citation=url or citation,
        case_name=title,
        case_number_or_citation=citation,
        court_name=compact(row.get("court_full_name") or row.get("court_short_name") or ""),
        court_level=court_level(row),
        decision_date=compact(row.get("date_filed") or row.get("date") or ""),
        opinion_type=opinion_type,
        procedural_posture=first_pattern_label(POSTURE_PATTERNS, haystack, "unknown"),
        case_subtype=first_pattern_label(SUBTYPES, haystack, "unclear"),
        raw_text=raw_text,
        include_signals=include_signals,
        exclude_signals=exclude_signals,
        quality_flags=quality_flags,
        collection_status=status_from_signals(exclude_signals, quality_flags),
    )


def keyword_gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    raw_text, _, _ = opinion_chunks(row)
    metadata = "\n".join(compact(row.get(key, "")) for key in ["case_name", "case_name_full", "case_name_short", "court_full_name", "court_short_name", "headmatter", "summary", "syllabus", "disposition"])
    hits = regex_hits(INCLUDE_PATTERNS, f"{metadata}\n{compact(raw_text)[:12000]}")
    return bool(hits), hits


def collect(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    records = []
    scanned = 0
    keyword_gate_hits = 0
    keyword_gate_skipped = 0
    for scanned, row in enumerate(dataset, start=1):
        if args.scan_limit and scanned > args.scan_limit:
            break
        gate_keep, _ = keyword_gate(row)
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
    selected = sample_records([row for row in records if row["collection_status"] == "pass" or (args.include_warning and row["collection_status"] == "warning")], args.target_count, args.seed)
    summary = summarize_records(all_records=records, selected_records=selected, args=args)
    add_gate_stats(
        summary,
        stream_rows_scanned=scanned,
        keyword_gate_hits=keyword_gate_hits,
        keyword_gate_skipped=keyword_gate_skipped,
        gate_patterns=INCLUDE_PATTERNS,
    )
    return records, selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 raw California civil liability case collector.")
    parser.add_argument("--dataset", default="harvard-lil/cold-cases")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--scan-limit", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year-min", type=int, default=0)
    parser.add_argument("--year-max", type=int, default=0)
    parser.add_argument("--min-text-chars", type=int, default=3000)
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--include-warning", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--raw-output", default="outputs/raw/ca_cases_raw.jsonl")
    parser.add_argument("--qc-output", default="outputs/raw/ca_cases_qc.csv")
    parser.add_argument("--summary-output", default="outputs/raw/ca_cases_summary.json")
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
    write_manifest(Path(args.manifest_output), [Path("outputs/raw/kr_cases_raw.jsonl"), Path(args.raw_output)], overwrite=True)
    print(f"selected={len(selected)} candidates={len(all_records)}")
    print(f"raw={args.raw_output}")
    print(f"qc={args.qc_output}")
    print(f"summary={args.summary_output}")
    print(f"manifest={args.manifest_output}")


if __name__ == "__main__":
    main()
