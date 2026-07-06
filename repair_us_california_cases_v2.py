from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from collect_us_california_cases import (
    CATEGORY_OVERRIDES,
    MANUAL_EXCLUDE_CASES,
    SOURCE_DATASET,
    SOURCE_SPLIT,
    evaluate_row,
    has_concrete_liability_context,
    has_contract_only_signal,
    has_family_law_signal,
    make_record,
    qc_row,
    row_text,
)
from pipeline.io_utils import ensure_parent, require_overwrite
from pipeline.text_utils import compact_inline


LOGGER = logging.getLogger(__name__)

CATEGORY_PRIORITY = {
    "personal_injury": 0,
    "premises_liability": 0,
    "auto_accident": 1,
    "wrongful_death": 1,
    "property_damage": 2,
    "product_liability": 3,
    "professional_negligence": 3,
    "fraud_damages": 4,
    "contract_damages": 5,
    "unclear": 6,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    require_overwrite(path, overwrite)
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def title_preview(record: dict[str, Any]) -> str:
    return f"{compact_inline(record.get('source_title', ''))}\n{compact_inline(record.get('raw_text', ''))[:3000]}"


def apply_existing_record_v2_qc(record: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    record = dict(record)
    case_id = compact_inline(record.get("case_id", ""))
    category = CATEGORY_OVERRIDES.get(case_id, compact_inline(record.get("case_category_guess", "")))
    record["case_category_guess"] = category
    qc = dict(record.get("collection_qc") or {})
    text = compact_inline(record.get("raw_text", ""))
    preview = title_preview(record)
    has_concrete_context = has_concrete_liability_context(text, compact_inline(record.get("source_title", "")), category)

    removed_reason: str | None = None
    if case_id in MANUAL_EXCLUDE_CASES:
        removed_reason = MANUAL_EXCLUDE_CASES[case_id]
    elif has_family_law_signal(preview):
        removed_reason = "family_divorce_signal"
    elif has_contract_only_signal(preview, has_concrete_context):
        removed_reason = "contract_only_debt_or_rent_signal"

    if removed_reason:
        qc["status"] = "fail"
        qc["excluded_reason"] = removed_reason
    else:
        qc.setdefault("status", "pass")
        qc["excluded_reason"] = qc.get("excluded_reason") or None

    record["collection_qc"] = qc
    return record, removed_reason


def replacement_sort_key(row: pd.Series) -> tuple[int, int, int, str]:
    category = CATEGORY_OVERRIDES.get(str(row["case_id"]), str(row["case_category_guess"]))
    status_rank = 0 if row["status"] == "pass" else 1
    category_rank = CATEGORY_PRIORITY.get(category, 9)
    try:
        length_rank = -int(float(row["text_length_chars"]))
    except Exception:
        length_rank = 0
    return status_rank, category_rank, length_rank, str(row["case_id"])


def candidate_case_ids(qc_df: pd.DataFrame, excluded_ids: set[str], selected_ids: set[str]) -> list[str]:
    pool = qc_df[qc_df["status"].isin(["pass", "warning"])].copy()
    pool["case_category_guess"] = pool.apply(
        lambda row: CATEGORY_OVERRIDES.get(str(row["case_id"]), str(row["case_category_guess"])),
        axis=1,
    )
    rows = []
    for _, row in pool.iterrows():
        case_id = str(row["case_id"])
        if case_id in excluded_ids or case_id in selected_ids or case_id in MANUAL_EXCLUDE_CASES:
            continue
        rows.append(row)
    return [str(row["case_id"]) for row in sorted(rows, key=replacement_sort_key)]


def fetch_replacements(
    ordered_case_ids: list[str],
    needed_count: int,
    *,
    dataset_name: str,
    split: str,
    scan_limit: int,
) -> list[dict[str, Any]]:
    if needed_count <= 0:
        return []
    wanted = set(ordered_case_ids)
    selected: list[dict[str, Any]] = []
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for scanned, row in enumerate(dataset, start=1):
        if scan_limit and scanned > scan_limit:
            LOGGER.warning("Stopping replacement fetch at --scan-limit=%s", scan_limit)
            break
        source_id = compact_inline(row.get("id", ""))
        case_id = f"US_california_{source_id}"
        if case_id not in wanted:
            continue
        text = row_text(row)
        qc, notes, category = evaluate_row(row, text)
        if qc["status"] != "pass":
            LOGGER.info("Skipping replacement %s after v2 QC: %s", case_id, qc.get("excluded_reason"))
            wanted.remove(case_id)
            continue
        record = make_record(row, text, qc, notes, category, dataset_name)
        if record["case_id"] != case_id:
            LOGGER.warning("Stable ID mismatch while fetching %s -> %s", case_id, record["case_id"])
        record["notes"] = "; ".join(part for part in [record.get("notes", ""), "selected_replacement_v2"] if part)
        selected.append(record)
        wanted.remove(case_id)
        LOGGER.info("Selected replacement %s (%s)", case_id, category)
        if len(selected) >= needed_count:
            break
    return selected


def update_qc_v2(
    qc_df: pd.DataFrame,
    removed: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> pd.DataFrame:
    df = qc_df.copy()
    for case_id, category in CATEGORY_OVERRIDES.items():
        df.loc[df["case_id"] == case_id, "case_category_guess"] = category
    for removed_record in removed:
        case_id = removed_record["case_id"]
        reason = removed_record["removed_reason"]
        df.loc[df["case_id"] == case_id, "status"] = "fail"
        df.loc[df["case_id"] == case_id, "excluded_reason"] = reason
        df.loc[df["case_id"] == case_id, "notes"] = (
            df.loc[df["case_id"] == case_id, "notes"].astype(str).replace("nan", "")
            + "; removed_from_final_v2"
        )

    final_qc_rows = {row["case_id"]: qc_row(row) for row in final_records}
    for replacement in replacements:
        row = final_qc_rows[replacement["case_id"]]
        mask = df["case_id"] == replacement["case_id"]
        for key, value in row.items():
            if key in df.columns:
                df.loc[mask, key] = "" if value is None else str(value)
        df.loc[mask, "notes"] = (
            df.loc[mask, "notes"].astype(str).replace("nan", "")
            + "; selected_replacement_v2"
        )
    return df


def remaining_signal_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    family = 0
    contract = 0
    federal_out_crim_admin = 0
    for record in records:
        text = compact_inline(record.get("raw_text", ""))
        category = compact_inline(record.get("case_category_guess", ""))
        preview = title_preview(record)
        if has_family_law_signal(preview):
            family += 1
        has_concrete_context = has_concrete_liability_context(text, compact_inline(record.get("source_title", "")), category)
        if has_contract_only_signal(preview, has_concrete_context):
            contract += 1
        qc = record.get("collection_qc") or {}
        if any(
            bool(qc.get(key))
            for key in [
                "is_federal_case",
                "is_out_of_state_case",
                "contains_criminal_signal",
                "contains_admin_signal",
            ]
        ):
            federal_out_crim_admin += 1
    return {
        "family_divorce_signal_remaining_count": family,
        "contract_only_signal_remaining_count": contract,
        "federal_out_of_state_criminal_admin_signal_remaining_count": federal_out_crim_admin,
    }


def build_summary(
    original_count: int,
    removed: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
) -> dict[str, Any]:
    category_counts = Counter(str(row["case_category_guess"]) for row in final_records)
    summary = {
        "original_final_pass_count": original_count,
        "removed_from_original_final_count": len(removed),
        "removed_cases": [
            {
                "case_id": row["case_id"],
                "source_title": row["source_title"],
                "reason": row["removed_reason"],
            }
            for row in removed
        ],
        "replacement_cases": [
            {
                "case_id": row["case_id"],
                "source_title": row["source_title"],
                "case_category_guess": row["case_category_guess"],
            }
            for row in replacements
        ],
        "final_raw_v2_count": len(final_records),
        "category_counts": {key: int(value) for key, value in category_counts.items()},
    }
    summary.update(remaining_signal_counts(final_records))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair California raw case set without full recollection.")
    parser.add_argument("--raw-input", default="outputs/us_california_cases_raw.jsonl")
    parser.add_argument("--qc-input", default="outputs/us_california_cases_qc.csv")
    parser.add_argument("--raw-output", default="outputs/us_california_cases_raw_v2.jsonl")
    parser.add_argument("--qc-output", default="outputs/us_california_cases_qc_v2.csv")
    parser.add_argument("--summary-output", default="outputs/us_california_cases_summary_v2.json")
    parser.add_argument("--dataset", default=SOURCE_DATASET)
    parser.add_argument("--split", default=SOURCE_SPLIT)
    parser.add_argument("--scan-limit", type=int, default=500_000)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    raw_records = read_jsonl(Path(args.raw_input))
    qc_df = pd.read_csv(args.qc_input, dtype=str, encoding="utf-8-sig").fillna("")

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in raw_records:
        updated, reason = apply_existing_record_v2_qc(record)
        if reason:
            updated["removed_reason"] = reason
            removed.append(updated)
        else:
            kept.append(updated)

    candidate_ids = candidate_case_ids(
        qc_df,
        excluded_ids={row["case_id"] for row in removed},
        selected_ids={row["case_id"] for row in kept},
    )
    needed = args.target_count - len(kept)
    replacements = fetch_replacements(
        candidate_ids,
        needed,
        dataset_name=args.dataset,
        split=args.split,
        scan_limit=args.scan_limit,
    )
    final_records = kept + replacements
    if len(final_records) != args.target_count:
        raise RuntimeError(f"Expected {args.target_count} final records, got {len(final_records)}")

    raw_output = Path(args.raw_output)
    qc_output = Path(args.qc_output)
    summary_output = Path(args.summary_output)
    for path in [raw_output, qc_output, summary_output]:
        require_overwrite(path, args.overwrite)
        ensure_parent(path)

    write_jsonl(raw_output, final_records, overwrite=True)
    qc_v2 = update_qc_v2(qc_df, removed, final_records, replacements)
    qc_v2.to_csv(qc_output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    summary = build_summary(len(raw_records), removed, replacements, final_records)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
