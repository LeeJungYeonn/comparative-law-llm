"""Synchronize KR v4 derived artifacts to an authoritative human-QC selection."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path


BASE = Path("outputs/raw/kr_v4")
FINAL_PATH = BASE / "kr_cases_selected_50_final.jsonl"
AUDIT_PATH = BASE / "kr_cases_selected_50_final_selection_audit.csv"
FINAL_SUMMARY_PATH = BASE / "kr_cases_selected_50_final_summary.json"
PIPELINE_SUMMARY_PATH = BASE / "kr_cases_summary.json"
QC_PATH = BASE / "kr_direct_tort_shortlist_100_qc.csv"
MANIFEST_PATH = Path("outputs/manifests/kr_v4_case_manifest.csv")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_authoritative_files() -> tuple[list[dict[str, object]], dict[str, dict[str, object]], dict[str, object]]:
    final = read_jsonl(FINAL_PATH)
    with AUDIT_PATH.open(encoding="utf-8-sig", newline="") as handle:
        audit = list(csv.DictReader(handle))
    with FINAL_SUMMARY_PATH.open(encoding="utf-8") as handle:
        final_summary = json.load(handle)

    final_ids = [str(row["case_id"]) for row in final]
    audit_ids = [str(row["case_id"]) for row in audit]
    summary_ids = [str(case_id) for case_id in final_summary.get("selected_case_ids", [])]
    if len(final) != 50 or len(set(final_ids)) != 50:
        raise ValueError("authoritative final JSONL must contain 50 unique cases")
    if set(final_ids) != set(audit_ids) or set(final_ids) != set(summary_ids):
        raise ValueError("final JSONL, selection audit, and final summary case IDs differ")
    if int(final_summary.get("selected_count", -1)) != len(final):
        raise ValueError("final summary selected_count differs from final JSONL")
    return final, {str(row["case_id"]): row for row in final}, final_summary


def synchronize_jsonl(path: Path, canonical: dict[str, dict[str, object]]) -> None:
    rows = read_jsonl(path)
    synchronized: list[dict[str, object]] = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id in canonical:
            synchronized.append(dict(canonical[case_id]))
        else:
            row["selected"] = False
            row["sampling_rank"] = None
            row["sampling_reasons"] = []
            synchronized.append(row)
    atomic_write_jsonl(path, synchronized)


def synchronize_qc(canonical: dict[str, dict[str, object]]) -> None:
    with QC_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id in canonical:
            source = canonical[case_id]
            for field in fieldnames:
                value = source.get(field, row.get(field, ""))
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                row[field] = value
        else:
            row["human_qc_status"] = ""
            row["human_qc_notes"] = ""
    atomic_write_csv(QC_PATH, fieldnames, rows)


def synchronize_summary(final: list[dict[str, object]], final_summary: dict[str, object]) -> None:
    with PIPELINE_SUMMARY_PATH.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    distribution = dict(sorted(Counter(str(row["case_subtype"]) for row in final).items()))
    summary.update(final_summary)
    summary.update(
        {
            "selected_count": len(final),
            "preselected_count": len(final),
            "pre_qc_selected_count": len(final),
            "final_selected_subtype_distribution": distribution,
            "manual_qc_rows_applied": len(final),
            "human_qc_applied": True,
            "require_human_accept": True,
            "shortage": 0,
        }
    )
    atomic_write_json(PIPELINE_SUMMARY_PATH, summary)


def synchronize_manifest(canonical: dict[str, dict[str, object]]) -> None:
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    final_path_text = str(FINAL_PATH)
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id in canonical:
            source = canonical[case_id]
            for field in fieldnames:
                if field in source:
                    value = source[field]
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, ensure_ascii=False)
                    row[field] = value
            row["selected"] = True
            row["pre_qc_selected"] = True
            row["shortlisted"] = True
            row["raw_path"] = final_path_text
        else:
            row["selected"] = False
            row["pre_qc_selected"] = False
    atomic_write_csv(MANIFEST_PATH, fieldnames, rows)


def main() -> None:
    final, canonical, final_summary = validate_authoritative_files()
    for path in sorted(BASE.glob("*.jsonl")):
        if path == FINAL_PATH:
            continue
        if path.name == "kr_cases_selected_50_pre_qc.jsonl":
            atomic_write_jsonl(path, [dict(row) for row in final])
        else:
            synchronize_jsonl(path, canonical)
    synchronize_qc(canonical)
    synchronize_summary(final, final_summary)
    synchronize_manifest(canonical)
    print(f"synchronized_selected={len(final)}")


if __name__ == "__main__":
    main()
