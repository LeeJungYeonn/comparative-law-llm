from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


COLLECTION_VERSION = "stage1-v2"

DEFAULT_SUBTYPE_QUOTAS = {
    "traffic_auto_accident": 15,
    "medical_professional_liability": 10,
    "premises_facility_safety": 8,
    "product_safety": 5,
    "property_economic_harm": 7,
    "injury_death_other": 5,
}

RAW_SCHEMA_FIELDS = [
    "case_id",
    "case_origin",
    "jurisdiction",
    "source_dataset",
    "source_record_id",
    "source_url_or_citation",
    "case_name",
    "case_number_or_citation",
    "court_name",
    "court_level",
    "decision_date",
    "decision_year",
    "opinion_type",
    "procedural_posture",
    "case_subtype",
    "raw_text",
    "raw_text_sha256",
    "raw_length_chars",
    "include_signals",
    "exclude_signals",
    "quality_flags",
    "collection_status",
    "collection_version",
    "related_case_group_id",
]

QC_FIELDS = [
    "case_id",
    "source_record_id",
    "case_name",
    "case_number_or_citation",
    "court_name",
    "court_level",
    "decision_date",
    "decision_year",
    "case_subtype",
    "raw_length_chars",
    "collection_status",
    "exclusion_reason",
    "include_signals",
    "exclude_signals",
    "quality_flags",
    "related_case_group_id",
]


def normalize_whitespace(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", normalize_whitespace(value)).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalized_text_for_hash(text: str) -> str:
    value = compact(text).lower()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def short_hash(*parts: object, length: int = 16) -> str:
    joined = "\n".join(compact(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:length]


def stable_stage1_case_id(
    *,
    case_origin: str,
    source_dataset: str,
    source_record_id: str,
    case_number_or_citation: str,
    case_name: str,
    decision_date: str,
    raw_text: str,
) -> str:
    stable_source = compact(source_record_id) or compact(case_number_or_citation)
    if stable_source:
        digest = short_hash(source_dataset, stable_source, case_name, decision_date)
    else:
        digest = short_hash(source_dataset, case_name, decision_date, raw_text[:1000])
    return f"{case_origin}_{digest}"


def parse_year(value: object) -> int | None:
    match = re.search(r"\b(18|19|20)\d{2}\b", compact(value))
    return int(match.group(0)) if match else None


def unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def make_raw_record(
    *,
    case_origin: str,
    jurisdiction: str,
    source_dataset: str,
    source_record_id: str,
    source_url_or_citation: str,
    case_name: str,
    case_number_or_citation: str,
    court_name: str,
    court_level: str,
    decision_date: str,
    opinion_type: str,
    procedural_posture: str,
    case_subtype: str,
    raw_text: str,
    include_signals: Iterable[str],
    exclude_signals: Iterable[str],
    quality_flags: Iterable[str],
    collection_status: str,
    related_case_group_id: str = "",
) -> dict[str, object]:
    text = normalize_whitespace(raw_text)
    case_id = stable_stage1_case_id(
        case_origin=case_origin,
        source_dataset=source_dataset,
        source_record_id=source_record_id,
        case_number_or_citation=case_number_or_citation,
        case_name=case_name,
        decision_date=decision_date,
        raw_text=text,
    )
    record = {
        "case_id": case_id,
        "case_origin": case_origin,
        "jurisdiction": jurisdiction,
        "source_dataset": source_dataset,
        "source_record_id": compact(source_record_id),
        "source_url_or_citation": compact(source_url_or_citation),
        "case_name": compact(case_name),
        "case_number_or_citation": compact(case_number_or_citation),
        "court_name": compact(court_name),
        "court_level": compact(court_level) or "unknown",
        "decision_date": compact(decision_date),
        "decision_year": parse_year(decision_date),
        "opinion_type": compact(opinion_type) or "unknown",
        "procedural_posture": compact(procedural_posture),
        "case_subtype": compact(case_subtype) or "unclear",
        "raw_text": text,
        "raw_text_sha256": sha256_text(text),
        "raw_length_chars": len(text),
        "include_signals": unique(include_signals),
        "exclude_signals": unique(exclude_signals),
        "quality_flags": unique(quality_flags),
        "collection_status": collection_status,
        "collection_version": COLLECTION_VERSION,
        "related_case_group_id": related_case_group_id,
    }
    return {field: record.get(field, None if field == "decision_year" else "") for field in RAW_SCHEMA_FIELDS}


def length_flags(text: str, min_text_chars: int, max_text_chars: int) -> list[str]:
    flags = []
    if len(text) < min_text_chars:
        flags.append("too_short_or_no_full_opinion_text")
    if max_text_chars and len(text) > max_text_chars:
        flags.append("too_long")
    return flags


def status_from_signals(exclude_signals: list[str], quality_flags: list[str]) -> str:
    if exclude_signals:
        return "fail"
    if any(flag.startswith("warning_") for flag in quality_flags):
        return "warning"
    return "pass"


def apply_duplicate_qc(records: list[dict[str, object]], near_threshold: float = 0.96) -> None:
    exact_seen: dict[str, str] = {}
    norm_seen: dict[str, str] = {}
    citation_seen: dict[str, str] = {}
    representatives: list[tuple[str, str]] = []
    groups: dict[str, str] = {}

    def group_for(left_id: str, right_id: str) -> str:
        existing = groups.get(left_id) or groups.get(right_id)
        if existing:
            groups[left_id] = existing
            groups[right_id] = existing
            return existing
        group = f"grp_{short_hash(left_id, right_id, length=12)}"
        groups[left_id] = group
        groups[right_id] = group
        return group

    for record in records:
        case_id = str(record["case_id"])
        flags = list(record.get("quality_flags") or [])
        exact = str(record.get("raw_text_sha256", ""))
        norm = sha256_text(normalized_text_for_hash(str(record.get("raw_text", ""))))
        citation = compact(record.get("case_number_or_citation", "")).lower()
        duplicate_of = ""

        if exact and exact in exact_seen:
            flags.append("duplicate_exact_hash")
            duplicate_of = exact_seen[exact]
        elif norm and norm in norm_seen:
            flags.append("duplicate_normalized_text_hash")
            duplicate_of = norm_seen[norm]
        elif citation and citation in citation_seen:
            flags.append("duplicate_case_number_or_citation")
            duplicate_of = citation_seen[citation]
        elif record.get("collection_status") in {"pass", "warning"}:
            norm_text = normalized_text_for_hash(str(record.get("raw_text", "")))[:5000]
            for other_id, other_norm in representatives:
                if SequenceMatcher(None, norm_text, other_norm).ratio() >= near_threshold:
                    flags.append("duplicate_near_text")
                    duplicate_of = other_id
                    break
            representatives.append((case_id, norm_text))

        exact_seen.setdefault(exact, case_id)
        norm_seen.setdefault(norm, case_id)
        if citation:
            citation_seen.setdefault(citation, case_id)
        if duplicate_of:
            record["related_case_group_id"] = group_for(case_id, duplicate_of)
            record["collection_status"] = "fail"
        record["quality_flags"] = unique(flags)


def grouped_case_numbers(records: list[dict[str, object]]) -> None:
    by_number: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        number = compact(record.get("case_number_or_citation", "")).lower()
        if number:
            by_number[number].append(record)
    for number, group in by_number.items():
        if len(group) < 2:
            continue
        group_id = f"grp_{short_hash(number, length=12)}"
        for record in group:
            record["related_case_group_id"] = group_id


def eligible_records(records: list[dict[str, object]], include_warning: bool) -> list[dict[str, object]]:
    statuses = {"pass", "warning"} if include_warning else {"pass"}
    return [record for record in records if record.get("collection_status") in statuses]


def sample_records(records: list[dict[str, object]], target_count: int, seed: int) -> list[dict[str, object]]:
    pool = list(records)
    pool.sort(key=lambda row: (str(row.get("court_level", "")), str(row.get("decision_year", "")), str(row.get("case_subtype", "")), str(row.get("case_id", ""))))
    rng = random.Random(seed)
    rng.shuffle(pool)
    if target_count and len(pool) > target_count:
        pool = pool[:target_count]
    return sorted(pool, key=lambda row: str(row.get("case_id", "")))


def subtype_quota_group(subtype: object) -> str:
    value = compact(subtype).lower()
    if value in {"traffic_accident", "auto_accident"}:
        return "traffic_auto_accident"
    if value in {"medical_professional", "professional_negligence"}:
        return "medical_professional_liability"
    if value in {"premises_facility_safety", "facility_product_safety"}:
        return "premises_facility_safety"
    if value in {"product_safety", "product_liability"}:
        return "product_safety"
    if value == "property_economic_harm":
        return "property_economic_harm"
    return "injury_death_other"


def period_bucket(year: object) -> str:
    try:
        value = int(year)
    except (TypeError, ValueError):
        return "unknown"
    start = value - ((value - 2000) % 5)
    end = min(start + 4, 2020)
    return f"{start}-{end}"


def appellate_year_pool(records: list[dict[str, object]], *, year_min: int, year_max: int) -> list[dict[str, object]]:
    pool = []
    for record in records:
        if record.get("court_level") != "appellate":
            continue
        year = record.get("decision_year")
        if not isinstance(year, int):
            continue
        if year_min <= year <= year_max:
            pool.append(record)
    return pool


def stratified_quota_sample(
    records: list[dict[str, object]],
    *,
    target_count: int,
    seed: int,
    subtype_quotas: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    quotas = subtype_quotas or DEFAULT_SUBTYPE_QUOTAS
    rng = random.Random(seed)
    by_group_period: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_group_period[(subtype_quota_group(record.get("case_subtype")), period_bucket(record.get("decision_year")))].append(record)
    for values in by_group_period.values():
        values.sort(key=lambda row: str(row.get("case_id", "")))
        rng.shuffle(values)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    for group, quota in quotas.items():
        periods = sorted(period for (candidate_group, period) in by_group_period if candidate_group == group)
        while periods and sum(1 for row in selected if subtype_quota_group(row.get("case_subtype")) == group) < quota:
            progressed = False
            for period in periods:
                bucket = by_group_period[(group, period)]
                while bucket and bucket[0]["case_id"] in selected_ids:
                    bucket.pop(0)
                if not bucket:
                    continue
                row = bucket.pop(0)
                selected.append(row)
                selected_ids.add(str(row["case_id"]))
                progressed = True
                if sum(1 for item in selected if subtype_quota_group(item.get("case_subtype")) == group) >= quota:
                    break
                if len(selected) >= target_count:
                    break
            if len(selected) >= target_count or not progressed:
                break

    leftovers = [row for row in records if str(row.get("case_id", "")) not in selected_ids]
    leftovers.sort(key=lambda row: (subtype_quota_group(row.get("case_subtype")), period_bucket(row.get("decision_year")), str(row.get("case_id", ""))))
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(selected) >= target_count:
            break
        selected.append(row)
        selected_ids.add(str(row["case_id"]))

    return sorted(selected[:target_count], key=lambda row: str(row.get("case_id", "")))


def stratified_sample_with_fallback(
    records: list[dict[str, object]],
    *,
    target_count: int,
    seed: int,
    primary_year_min: int = 2010,
    fallback_year_min: int = 2000,
    year_max: int = 2020,
    subtype_quotas: dict[str, int] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    quotas = subtype_quotas or DEFAULT_SUBTYPE_QUOTAS
    primary_pool = appellate_year_pool(records, year_min=primary_year_min, year_max=year_max)
    selected = stratified_quota_sample(primary_pool, target_count=target_count, seed=seed, subtype_quotas=quotas)
    fallback_used = False
    fallback_pool: list[dict[str, object]] = []
    if len(selected) < target_count:
        fallback_used = True
        selected_ids = {str(row["case_id"]) for row in selected}
        fallback_pool = [
            row
            for row in appellate_year_pool(records, year_min=fallback_year_min, year_max=year_max)
            if str(row.get("case_id", "")) not in selected_ids
        ]
        fallback_selected = stratified_quota_sample(
            fallback_pool,
            target_count=target_count - len(selected),
            seed=seed + 1,
            subtype_quotas=quotas,
        )
        selected = sorted(selected + fallback_selected, key=lambda row: str(row.get("case_id", "")))

    selected_primary = [row for row in selected if isinstance(row.get("decision_year"), int) and int(row["decision_year"]) >= primary_year_min]
    shortage_report = {}
    for group, quota in quotas.items():
        primary_available = sum(1 for row in primary_pool if subtype_quota_group(row.get("case_subtype")) == group)
        primary_selected = sum(1 for row in selected_primary if subtype_quota_group(row.get("case_subtype")) == group)
        shortage_report[group] = {
            "quota": quota,
            "primary_available": primary_available,
            "primary_selected": primary_selected,
            "primary_shortage": max(0, quota - primary_selected),
        }
    meta = {
        "sampling_method": "subtype_x_5year_period_stratified",
        "primary_year_min": primary_year_min,
        "fallback_year_min": fallback_year_min,
        "year_max": year_max,
        "fallback_used_for_total_shortage": fallback_used,
        "primary_eligible_count": len(primary_pool),
        "fallback_eligible_count": len(fallback_pool),
        "shortage_report": shortage_report,
        "subtype_quotas": quotas,
    }
    return selected[:target_count], meta


def summarize_records(*, all_records: list[dict[str, object]], selected_records: list[dict[str, object]], args: object) -> dict[str, object]:
    def counts(field: str, rows: list[dict[str, object]]) -> dict[str, int]:
        return dict(Counter(str(row.get(field, "") or "unknown") for row in rows))

    duplicate_flags = Counter(flag for row in all_records for flag in row.get("quality_flags", []) if str(flag).startswith("duplicate_"))
    candidate_period_subtype = Counter(
        f"{period_bucket(row.get('decision_year'))}|{subtype_quota_group(row.get('case_subtype'))}"
        for row in all_records
    )
    selected_period_subtype = Counter(
        f"{period_bucket(row.get('decision_year'))}|{subtype_quota_group(row.get('case_subtype'))}"
        for row in selected_records
    )
    return {
        "collection_version": COLLECTION_VERSION,
        "total_candidates_scanned": len(all_records),
        "eligible_pool_count": len(eligible_records(all_records, bool(getattr(args, "include_warning", False)))),
        "selected_count": len(selected_records),
        "status_counts": counts("collection_status", all_records),
        "selected_court_level_counts": counts("court_level", selected_records),
        "selected_decision_year_counts": counts("decision_year", selected_records),
        "selected_case_subtype_counts": counts("case_subtype", selected_records),
        "candidate_court_level_counts": counts("court_level", all_records),
        "candidate_decision_year_counts": counts("decision_year", all_records),
        "candidate_case_subtype_counts": counts("case_subtype", all_records),
        "candidate_period_x_subtype_counts": dict(candidate_period_subtype),
        "selected_period_x_subtype_counts": dict(selected_period_subtype),
        "duplicate_flag_counts": dict(duplicate_flags),
        "seed": getattr(args, "seed", None),
        "target_count": getattr(args, "target_count", None),
        "scan_limit": getattr(args, "scan_limit", None),
        "min_text_chars": getattr(args, "min_text_chars", None),
        "max_text_chars": getattr(args, "max_text_chars", None),
    }


def add_gate_stats(summary: dict[str, object], *, stream_rows_scanned: int, keyword_gate_hits: int, keyword_gate_skipped: int, gate_patterns: Iterable[str]) -> dict[str, object]:
    summary["stream_rows_scanned"] = stream_rows_scanned
    summary["keyword_gate_hits"] = keyword_gate_hits
    summary["keyword_gate_skipped"] = keyword_gate_skipped
    summary["keyword_gate_patterns"] = list(gate_patterns)
    return summary


def require_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists: {names}. Pass --overwrite to replace it.")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def qc_row(record: dict[str, object]) -> dict[str, object]:
    row = {field: record.get(field, "") for field in QC_FIELDS}
    row["include_signals"] = ";".join(record.get("include_signals") or [])
    row["exclude_signals"] = ";".join(record.get("exclude_signals") or [])
    row["quality_flags"] = ";".join(record.get("quality_flags") or [])
    row["exclusion_reason"] = ";".join(list(record.get("exclude_signals") or []) + list(record.get("quality_flags") or [])) if record.get("collection_status") == "fail" else ""
    return row


def write_qc_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(qc_row(row))


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def read_raw_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(path: Path, raw_paths: Iterable[Path], overwrite: bool) -> None:
    require_outputs([path], overwrite)
    fieldnames = [
        "case_id",
        "case_origin",
        "jurisdiction",
        "source_dataset",
        "source_record_id",
        "case_name",
        "case_number_or_citation",
        "court_name",
        "court_level",
        "decision_year",
        "case_subtype",
        "collection_status",
        "related_case_group_id",
        "raw_text_sha256",
        "raw_length_chars",
        "raw_path",
    ]
    rows = []
    for raw_path in raw_paths:
        for record in read_raw_jsonl(raw_path):
            rows.append({field: record.get(field, "") for field in fieldnames if field != "raw_path"} | {"raw_path": str(raw_path)})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
