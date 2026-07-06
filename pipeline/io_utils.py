from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.schema import CASE_TABLE_COLUMNS
from pipeline.text_utils import compact_inline


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def write_jsonl(path: Path, rows: Iterable[dict], overwrite: bool = True) -> int:
    require_overwrite(path, overwrite)
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def short_hash(*parts: object, length: int = 10) -> str:
    joined = "\n".join(compact_inline(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:length]


def slug(value: object, fallback: str = "unknown") -> str:
    text = compact_inline(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or fallback


def source_slug(source_dataset: str) -> str:
    source = source_dataset.lower()
    if "lbox" in source:
        return "lbox"
    if "cold-cases" in source or "harvard" in source:
        return "cold_cases"
    return slug(source_dataset)


def stable_case_id(
    *,
    case_origin: str,
    jurisdiction: str,
    source_dataset: str,
    source_id: str,
    title: str,
    date: str,
    raw_text: str,
) -> str:
    origin = slug(case_origin).upper()
    jurisdiction_part = slug(jurisdiction)
    if jurisdiction_part in {"korea", "kr"}:
        jurisdiction_part = source_slug(source_dataset)
    if source_id:
        id_part = slug(source_id)
    else:
        id_part = short_hash(source_dataset, title, date, raw_text[:500])
    return f"{origin}_{jurisdiction_part}_{id_part}"


def parse_flags(value: object) -> list[str]:
    text = compact_inline(value)
    if not text:
        return []
    parts = re.split(r"[;,|]", text)
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def join_flags(flags: Iterable[str]) -> str:
    return "; ".join(dict.fromkeys(flag for flag in flags if flag))


def first_present(row: pd.Series, columns: Iterable[str]) -> str:
    for column in columns:
        if column in row.index:
            value = compact_inline(row.get(column, ""))
            if value:
                return value
    return ""


def infer_origin(row: pd.Series) -> str:
    origin = first_present(row, ["case_origin", "jurisdiction"])
    if origin.upper() in {"KR", "KOREA"}:
        return "KR"
    if origin.upper() in {"US", "USA", "CALIFORNIA", "NEW YORK"}:
        return "US"
    country = first_present(row, ["country"])
    if country.lower() == "korea":
        return "KR"
    if country.lower() == "united states":
        return "US"
    return origin.upper() if origin else "US"


def normalize_jurisdiction(row: pd.Series, origin: str) -> str:
    state = first_present(row, ["state", "jurisdiction_name", "state_name"])
    if state:
        return state
    jurisdiction = first_present(row, ["jurisdiction"])
    country = first_present(row, ["country"])
    case_id = first_present(row, ["case_id"]).lower()
    if origin == "KR":
        return "Korea"
    if jurisdiction in {"California", "New York"}:
        return jurisdiction
    if "us_california_" in case_id:
        return "California"
    if "us_new_york_" in case_id:
        return "New York"
    if country == "Korea":
        return "Korea"
    return "Unknown"


def build_case_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in df.fillna("").iterrows():
        origin = infer_origin(row)
        jurisdiction = normalize_jurisdiction(row, origin)
        source_dataset = first_present(row, ["source_dataset", "source"]) or (
            "lbox/lbox_open::precedent_corpus" if origin == "KR" else "harvard-lil/cold-cases"
        )
        raw_text = first_present(row, ["reasoning_text", "clean_text", "raw_text", "text"])
        title = first_present(row, ["title", "case_name", "name"])
        date = first_present(row, ["date", "decision_date", "date_filed"])
        court = first_present(row, ["court", "court_name", "court_full_name"])
        trial_level = first_present(row, ["trial_level", "court_level"])
        existing_case_id = first_present(row, ["case_id"])
        source_id = first_present(row, ["source_id", "id"]) or existing_case_id
        collection_notes = join_flags(
            [
                first_present(row, ["collection_notes"]),
                first_present(row, ["kr_filter_reason"]),
                first_present(row, ["us_filter_reason"]),
                first_present(row, ["state_filter_status"]),
                first_present(row, ["preprocess_notes"]),
            ]
        )
        flags = parse_flags(first_present(row, ["quality_flags"]))
        if first_present(row, ["state_filter_status"]) == "ambiguous":
            flags.append("ambiguous_us_state")
        text_length = len(raw_text)
        if text_length < 200:
            flags.append("too_short")
        if text_length > 100_000:
            flags.append("too_long")
        if existing_case_id.startswith(("KR_", "US_")):
            case_id = existing_case_id
        else:
            case_id = stable_case_id(
                case_origin=origin,
                jurisdiction=jurisdiction,
                source_dataset=source_dataset,
                source_id=source_id,
                title=title,
                date=date,
                raw_text=raw_text,
            )
        records.append(
            {
                "case_id": case_id,
                "case_origin": origin,
                "jurisdiction": jurisdiction,
                "source_dataset": source_dataset,
                "source_id": source_id,
                "title": title,
                "date": date,
                "court": court,
                "trial_level": trial_level,
                "raw_text": raw_text,
                "text_length_chars": text_length,
                "collection_notes": collection_notes,
                "quality_flags": join_flags(flags),
            }
        )
    return pd.DataFrame(records, columns=CASE_TABLE_COLUMNS)


def write_case_table(df: pd.DataFrame, output_path: Path, overwrite: bool) -> Path:
    require_overwrite(output_path, overwrite)
    ensure_parent(output_path)
    case_table = build_case_table(df)
    case_table.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return output_path
