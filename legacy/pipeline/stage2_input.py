from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.checkpoint import atomic_write_json
from pipeline.stage2_schema import DATASET_VERSION, Stage2CaseInput


EXPECTED_SUBTYPES = {
    "KR": {"traffic_accident": 10, "medical_professional": 7, "premises_facility_safety": 6, "employer_vicarious_liability": 4, "product_safety": 4, "privacy_reputation": 1, "intentional_tort": 1, "general_personal_injury": 1, "property_damage": 1},
    "CA": {"premises_facility_safety": 9, "traffic_accident": 7, "general_personal_injury": 6, "medical_professional": 4, "product_safety": 4, "employer_vicarious_liability": 2, "privacy_reputation": 1, "intentional_tort": 1, "property_damage": 1},
}
ORIGIN_CONFIG = {
    "KR": {"language": "ko", "field": "raw_text"},
    "CA": {"language": "en", "field": "main_opinion_text"},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Object expected at {path}:{number}")
        rows.append(value)
    return rows


def load_origin(path: Path, origin: str) -> tuple[list[Stage2CaseInput], dict[str, Any]]:
    config = ORIGIN_CONFIG[origin]
    file_hash = sha256_file(path)
    rows = _read_jsonl(path)
    cases: list[Stage2CaseInput] = []
    errors: list[str] = []
    source_hash_mismatches: list[str] = []
    for index, row in enumerate(rows, 1):
        case_id = str(row.get("case_id") or "").strip()
        source_text = str(row.get(config["field"]) or "")
        calculated_hash = sha256_bytes(source_text.encode("utf-8"))
        recorded_hash = str(row.get("raw_text_sha256") or "").lower()
        if not case_id:
            errors.append(f"row {index}: missing case_id")
        if not source_text.strip():
            errors.append(f"{case_id or index}: empty {config['field']}")
        if recorded_hash and recorded_hash != calculated_hash:
            source_hash_mismatches.append(case_id or str(index))
        cases.append(Stage2CaseInput(
            case_id=case_id, case_origin=origin, source_language=config["language"],
            source_text=source_text, source_text_field=config["field"],
            source_text_sha256=calculated_hash,
            case_subtype=str(row.get("case_subtype") or "") or None,
            source_dataset=str(row.get("source_dataset") or "") or None,
            source_record_id=str(row.get("source_record_id") or "") or None,
            input_file_sha256=file_hash,
        ))
    ids = [case.case_id for case in cases]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    subtype_distribution = dict(Counter(case.case_subtype or "" for case in cases))
    warnings: list[str] = []
    if subtype_distribution != EXPECTED_SUBTYPES[origin]:
        warnings.append("subtype_distribution_differs_from_expected")
    if len(cases) != 35:
        errors.append(f"expected 35 records, found {len(cases)}")
    if duplicates:
        errors.append(f"duplicate case_id: {duplicates}")
    if source_hash_mismatches:
        errors.append(f"source hash mismatch: {source_hash_mismatches}")
    return cases, {
        "origin": origin, "input_path": str(path.resolve()), "input_file_sha256": file_hash,
        "count": len(cases), "case_ids_unique": not duplicates,
        "required_source_field": config["field"], "source_text_present": not any(not c.source_text.strip() for c in cases),
        "source_hash_validation": "pass" if not source_hash_mismatches else "fail",
        "source_hash_mismatches": source_hash_mismatches,
        "subtype_distribution": subtype_distribution, "expected_subtype_distribution": EXPECTED_SUBTYPES[origin],
        "warnings": warnings, "errors": errors,
    }


def validate_inputs(kr_path: Path, ca_path: Path, output_dir: Path | None = None) -> tuple[list[Stage2CaseInput], dict[str, Any], dict[str, Any]]:
    before = {"KR": sha256_file(kr_path), "CA": sha256_file(ca_path)}
    kr_cases, kr_report = load_origin(kr_path, "KR")
    ca_cases, ca_report = load_origin(ca_path, "CA")
    cross_duplicates = sorted(set(c.case_id for c in kr_cases) & set(c.case_id for c in ca_cases))
    errors = kr_report["errors"] + ca_report["errors"]
    if cross_duplicates:
        errors.append(f"case_id duplicated across origins: {cross_duplicates}")
    after = {"KR": sha256_file(kr_path), "CA": sha256_file(ca_path)}
    if before != after:
        errors.append("input file changed during validation")
    report = {
        "dataset_version": DATASET_VERSION, "status": "pass" if not errors else "fail",
        "kr_count": len(kr_cases), "ca_count": len(ca_cases), "total_count": len(kr_cases) + len(ca_cases),
        "case_id_unique_across_all_inputs": not cross_duplicates,
        "input_hash_unchanged": before == after, "origins": {"KR": kr_report, "CA": ca_report},
        "warnings": kr_report["warnings"] + ca_report["warnings"], "errors": errors,
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "kr_input_path": str(kr_path.resolve()), "ca_input_path": str(ca_path.resolve()),
        "kr_input_file_sha256": before["KR"], "ca_input_file_sha256": before["CA"],
        "kr_case_ids": [c.case_id for c in kr_cases], "ca_case_ids": [c.case_id for c in ca_cases],
        "kr_count": len(kr_cases), "ca_count": len(ca_cases),
        "subtype_distributions": {"KR": kr_report["subtype_distribution"], "CA": ca_report["subtype_distribution"]},
        "source_field_mapping": {"KR": "raw_text", "CA": "main_opinion_text"},
    }
    if output_dir:
        atomic_write_json(output_dir / "input_validation_report.json", report)
        atomic_write_json(output_dir / "input_manifest.json", manifest)
    if errors:
        raise ValueError("Input validation failed: " + "; ".join(errors))
    return kr_cases + ca_cases, report, manifest

