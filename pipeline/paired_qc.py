from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl, atomic_write_text
from pipeline.stage2_runtime import by_case, json_hash, read_records
from pipeline.stage2_v3_pipeline import (
    JURISDICTION_TERMS,
    LEGAL_TERMS,
    PLACEHOLDER_RE,
    source_checks,
    stable_hash,
    translation_checks,
)
from pipeline.stage2_v3_schema import SCHEMA_VERSION as GENERATION_SCHEMA_VERSION


QC_DATASET_VERSION = "stage2-paired-qc-v1"
QC_SCHEMA_VERSION = "stage2-paired-qc-schema-v1"
SOURCE_QC_PROMPT_VERSION = {
    "ko": "qc_source_neutral_ko_v1_en",
    "en": "qc_source_neutral_en_v1_en",
}
TRANSLATION_QC_PROMPT_VERSION = {
    "ko_to_en": "qc_translation_ko_to_en_v1_en",
    "en_to_ko": "qc_translation_en_to_ko_v1_en",
}
BATCH_TARGETS = {
    "stage-a": {"KR": 3, "CA": 3},
    "stage-b": {"KR": 10, "CA": 10},
    "stage-c": {"KR": 20, "CA": 20},
    "stage-d": {"KR": 35, "CA": 35},
}

SOURCE_REVIEW_FIELDS = [
    "case_id", "batch_name", "case_origin", "case_subtype",
    "source_language", "master_language", "automatic_source_status",
    "source_error_types", "source_qc_confidence", "source_coverage",
    "core_event_present", "harm_present", "event_harm_sequence_present",
    "material_entities_complete", "entity_roles_correct",
    "source_grounding_correct", "epistemic_status_correct",
    "legal_neutrality", "jurisdiction_neutrality", "factual_sufficiency",
    "recognition_risk", "human_source_action", "edited_fact_ids",
    "human_source_status", "reviewer_notes",
    # Operational columns needed for validated, append-only imports.
    "master_text", "parent_master_sha256", "human_validated_master_text",
    "human_validated_fact_units_json", "edit_reasons", "reviewer_id",
    "review_timestamp",
]
TRANSLATION_REVIEW_FIELDS = [
    "case_id", "batch_name", "translation_direction", "master_text",
    "translation_text", "automatic_translation_status",
    "translation_error_types", "fact_ids_match",
    "placeholder_identity_match", "subject_object_preserved",
    "entity_roles_preserved", "epistemic_status_preserved",
    "negation_preserved", "temporal_order_preserved",
    "causal_direction_preserved", "spatial_direction_preserved",
    "numbers_units_preserved", "legal_terms_absent",
    "jurisdiction_signals_absent", "translation_equivalence",
    "human_translation_action", "edited_fact_ids",
    "human_translation_status", "reviewer_notes",
    # Operational columns needed for validated, append-only imports.
    "master_sha256", "parent_translation_sha256",
    "human_validated_translation_text",
    "human_validated_translation_fact_units_json", "edit_reasons",
    "reviewer_id", "review_timestamp",
]

SOURCE_ACTIONS = {
    "accept_master", "edit_master", "regenerate_master", "exclude_case", "defer",
}
TRANSLATION_ACTIONS = {
    "accept_translation", "edit_translation", "regenerate_translation",
    "exclude_case", "defer",
}
HUMAN_STATUSES = {"accepted", "accepted_with_edits", "rejected", "pending"}

SOURCE_FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "fact_id", "master_span", "source_sentence_ids", "source_excerpt",
        "error_type", "severity", "explanation",
    ],
    "properties": {
        "fact_id": {"type": "string"},
        "master_span": {"type": "string"},
        "source_sentence_ids": {"type": "array", "items": {"type": "string"}},
        "source_excerpt": {"type": "string"},
        "error_type": {"type": "string"},
        "severity": {"type": "string", "enum": ["hard", "warning"]},
        "explanation": {"type": "string"},
    },
}
SOURCE_QC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "source_language", "source_qc"],
    "properties": {
        "case_id": {"type": "string"},
        "source_language": {"type": "string", "enum": ["ko", "en"]},
        "source_qc": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "factual_support_status", "unsupported_facts",
                "overstated_facts", "missing_material_facts",
                "entity_role_errors", "epistemic_status_errors",
                "legal_conclusion_leakage", "jurisdiction_leakage",
                "source_copy_risk", "recognition_risk_reasons",
                "factual_sufficiency", "model_source_qc_status",
                "model_confidence", "recommended_human_action",
            ],
            "properties": {
                "factual_support_status": {
                    "type": "string", "enum": ["pass", "warning", "fail"],
                },
                **{
                    field: {"type": "array", "items": SOURCE_FINDING_SCHEMA}
                    for field in (
                        "unsupported_facts", "overstated_facts",
                        "missing_material_facts", "entity_role_errors",
                        "epistemic_status_errors", "legal_conclusion_leakage",
                        "jurisdiction_leakage",
                    )
                },
                "source_copy_risk": {
                    "type": "string", "enum": ["low", "medium", "high"],
                },
                "recognition_risk_reasons": {
                    "type": "array", "items": {"type": "string"},
                },
                "factual_sufficiency": {
                    "type": "string",
                    "enum": ["sufficient", "marginal", "insufficient"],
                },
                "model_source_qc_status": {
                    "type": "string", "enum": ["pass", "warning", "fail"],
                },
                "model_confidence": {
                    "type": "string", "enum": ["high", "medium", "low"],
                },
                "recommended_human_action": {
                    "type": "string",
                    "enum": [
                        "accept", "review_only", "edit_master",
                        "regenerate_master", "exclude_case",
                    ],
                },
            },
        },
    },
}

TRANSLATION_FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "fact_id", "master_span", "translation_span", "error_type",
        "severity", "explanation",
    ],
    "properties": {
        "fact_id": {"type": "string"},
        "master_span": {"type": "string"},
        "translation_span": {"type": "string"},
        "error_type": {"type": "string"},
        "severity": {"type": "string", "enum": ["hard", "warning"]},
        "explanation": {"type": "string"},
    },
}
TRANSLATION_QC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "translation_direction", "translation_qc"],
    "properties": {
        "case_id": {"type": "string"},
        "translation_direction": {
            "type": "string", "enum": ["ko_to_en", "en_to_ko"],
        },
        "translation_qc": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "semantic_equivalence", "added_information",
                "omitted_information", "subject_object_shifts",
                "entity_role_shifts", "epistemic_status_shifts",
                "polarity_shifts", "temporal_relation_shifts",
                "causal_direction_shifts", "spatial_direction_shifts",
                "number_unit_shifts", "legal_terms_reintroduced",
                "jurisdiction_signals_reintroduced", "fact_structure_errors",
                "model_translation_qc_status", "model_confidence",
                "recommended_human_action",
            ],
            "properties": {
                "semantic_equivalence": {
                    "type": "string", "enum": ["pass", "warning", "fail"],
                },
                **{
                    field: {
                        "type": "array", "items": TRANSLATION_FINDING_SCHEMA,
                    }
                    for field in (
                        "added_information", "omitted_information",
                        "subject_object_shifts", "entity_role_shifts",
                        "epistemic_status_shifts", "polarity_shifts",
                        "temporal_relation_shifts", "causal_direction_shifts",
                        "spatial_direction_shifts", "number_unit_shifts",
                        "legal_terms_reintroduced",
                        "jurisdiction_signals_reintroduced",
                        "fact_structure_errors",
                    )
                },
                "model_translation_qc_status": {
                    "type": "string", "enum": ["pass", "warning", "fail"],
                },
                "model_confidence": {
                    "type": "string", "enum": ["high", "medium", "low"],
                },
                "recommended_human_action": {
                    "type": "string",
                    "enum": [
                        "accept", "review_only", "edit_translation",
                        "regenerate_translation", "exclude_case",
                    ],
                },
            },
        },
    },
}

BACK_TRANSLATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "back_translated_fact_units", "diagnostic_notes"],
    "properties": {
        "case_id": {"type": "string"},
        "back_translated_fact_units": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["fact_id", "back_translated_text"],
                "properties": {
                    "fact_id": {"type": "string"},
                    "back_translated_text": {"type": "string"},
                },
            },
        },
        "diagnostic_notes": {"type": "array", "items": {"type": "string"}},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def master_sha256(record: dict[str, Any]) -> str:
    return stable_hash({
        "master_neutral_text": record.get("master_neutral_text") or "",
        "fact_units": record.get("fact_units") or [],
    })


def translation_sha256(record: dict[str, Any]) -> str:
    return stable_hash({
        "translated_neutral_text": record.get("translated_neutral_text") or "",
        "translated_fact_units": record.get("translated_fact_units") or [],
    })


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def merge_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    merged = by_case(path)
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError(f"case_id is required for {path}")
        merged[case_id] = row
    atomic_write_jsonl(path, merged.values())


def append_audit_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    existing = read_records(path)
    additions = list(rows)
    atomic_write_jsonl(path, [*existing, *additions])


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    atomic_write_text(path, stream.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def artifact_indexes(generation_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    files = {
        "segments": ["source_segments_kr.jsonl", "source_segments_ca.jsonl"],
        "evidence": ["factual_evidence_kr.jsonl", "factual_evidence_ca.jsonl"],
        "graph": [
            "entity_relation_graphs_kr.jsonl",
            "entity_relation_graphs_ca.jsonl",
        ],
        "master": ["source_neutral_kr.jsonl", "source_neutral_ca.jsonl"],
        "translation": ["translated_pairs_kr.jsonl", "translated_pairs_ca.jsonl"],
        "source_checks": ["deterministic_source_checks.jsonl"],
        "source_verifier": ["source_grounding_role_verification.jsonl"],
        "translation_checks": ["deterministic_translation_checks.jsonl"],
        "translation_verifier": ["translation_verification.jsonl"],
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for kind, names in files.items():
        keyed: dict[str, dict[str, Any]] = {}
        for name in names:
            path = generation_dir / name
            if path.exists():
                keyed.update(by_case(path))
        result[kind] = keyed
    return result


def _load_raw_index(path: Path, origin: str) -> dict[str, dict[str, Any]]:
    rows = read_records(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id:
            result[case_id] = row
    if not result:
        raise ValueError(f"No {origin} records in {path}")
    return result


def resolve_raw_paths(
    generation_dir: Path, raw_kr_input: Path | None, raw_ca_input: Path | None,
) -> tuple[Path, Path]:
    manifest = load_json(generation_dir / "input_manifest.json")
    kr = raw_kr_input or Path(str(manifest.get("kr_input_path") or ""))
    ca = raw_ca_input or Path(str(manifest.get("ca_input_path") or ""))
    if not kr.is_file() or not ca.is_file():
        raise FileNotFoundError(
            "Raw Stage 1 snapshots are required; pass --raw-kr-input and "
            "--raw-ca-input when manifest paths are unavailable."
        )
    return kr, ca


def _provenances(indexes: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for kind in ("evidence", "graph", "master", "translation",
                 "source_verifier", "translation_verifier"):
        for record in indexes[kind].values():
            provenance = record.get("model_provenance") or {}
            if isinstance(provenance, dict):
                values.append({"kind": kind, **provenance})
            for item in record.get("chunk_model_provenance") or []:
                if isinstance(item, dict):
                    values.append({"kind": kind, **item})
    return values


def _prompt_language(path: Path) -> str:
    if not path.is_file():
        return "missing"
    text = path.read_text(encoding="utf-8-sig")
    hangul = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "ko" if hangul >= 3 and hangul > latin // 4 else "en"


def generation_consistency(
    generation_dir: Path, indexes: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    provenances = _provenances(indexes)
    models = sorted({str(item.get("model")) for item in provenances if item.get("model")})
    mocks = sorted({bool(item.get("mock")) for item in provenances})
    versions_by_kind: dict[str, list[str]] = {}
    prompt_languages: dict[str, str] = {}
    for item in provenances:
        kind = str(item.get("kind"))
        version = str(item.get("prompt_version") or "")
        if version:
            versions_by_kind.setdefault(kind, [])
            if version not in versions_by_kind[kind]:
                versions_by_kind[kind].append(version)
            prompt_path = generation_dir / "prompts" / f"{version}.txt"
            prompt_languages[version] = _prompt_language(prompt_path)
    for values in versions_by_kind.values():
        values.sort()

    errors: list[str] = []
    if len(models) != 1:
        errors.append("generation_model_inconsistent")
    if len(mocks) > 1:
        errors.append("mock_real_provenance_mixed")
    non_english = sorted(
        version for version, language in prompt_languages.items()
        if language != "en"
    )
    if non_english:
        errors.append("generation_instruction_language_not_uniform_english")

    master_versions = versions_by_kind.get("master", [])
    per_origin_master_versions: dict[str, list[str]] = {}
    for record in indexes["master"].values():
        origin = str(record.get("case_origin") or "")
        version = str((record.get("model_provenance") or {}).get("prompt_version") or "")
        if version:
            per_origin_master_versions.setdefault(origin, [])
            if version not in per_origin_master_versions[origin]:
                per_origin_master_versions[origin].append(version)
    for values in per_origin_master_versions.values():
        values.sort()
    if any(len(values) != 1 for values in per_origin_master_versions.values()):
        errors.append("neutralization_prompt_version_mixed_within_origin")
    numeric_policies = {
        match.group(1)
        for version in master_versions
        if (match := re.search(r"_v(\d+)$", version))
    }
    if len(numeric_policies) > 1:
        errors.append("neutralization_policy_version_inconsistent")

    dataset_versions = sorted({
        str(record.get("dataset_version"))
        for kind in ("evidence", "graph", "master", "translation")
        for record in indexes[kind].values()
        if record.get("dataset_version")
    })
    if len(dataset_versions) != 1:
        errors.append("generation_dataset_version_inconsistent")
    manifest_version = ""
    run_manifest_version = ""
    if (generation_dir / "input_manifest.json").exists():
        manifest_version = str(
            load_json(generation_dir / "input_manifest.json").get("dataset_version") or ""
        )
    if (generation_dir / "run_manifest.json").exists():
        run_manifest_version = str(
            load_json(generation_dir / "run_manifest.json").get("dataset_version") or ""
        )
    if manifest_version and dataset_versions and manifest_version not in dataset_versions:
        errors.append("input_manifest_dataset_version_mismatch")
    warnings: list[str] = []
    if run_manifest_version and dataset_versions and run_manifest_version not in dataset_versions:
        warnings.append("stale_run_manifest_dataset_version")
    return {
        "status": "pass" if not errors else "fail",
        "generation_dataset_versions": dataset_versions,
        "input_manifest_dataset_version": manifest_version,
        "run_manifest_dataset_version": run_manifest_version,
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "models": models,
        "mock_provenance_values": mocks,
        "prompt_versions_by_kind": versions_by_kind,
        "master_prompt_versions_by_origin": per_origin_master_versions,
        "prompt_instruction_languages": prompt_languages,
        "non_english_instruction_prompts": non_english,
        "errors": errors,
        "warnings": warnings,
    }


def expected_provenance(origin: str) -> dict[str, str]:
    if origin == "KR":
        return {
            "source_language": "ko", "master_language": "ko",
            "translation_direction": "ko_to_en",
            "ko_generation_type": "source_neutralized",
            "en_generation_type": "translated",
        }
    return {
        "source_language": "en", "master_language": "en",
        "translation_direction": "en_to_ko",
        "en_generation_type": "source_neutralized",
        "ko_generation_type": "translated",
    }


def validate_qc_inputs(
    generation_dir: Path,
    raw_kr_input: Path,
    raw_ca_input: Path,
    selected_case_ids: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    indexes = artifact_indexes(generation_dir)
    raw_indexes = {
        "KR": _load_raw_index(raw_kr_input, "KR"),
        "CA": _load_raw_index(raw_ca_input, "CA"),
    }
    manifest = load_json(generation_dir / "input_manifest.json")
    manifest_cases = {
        str(item.get("case_id")): item for item in manifest.get("case_inputs") or []
        if item.get("case_id")
    }
    all_ids = list(dict.fromkeys([
        *[str(value) for value in manifest.get("kr_case_ids") or []],
        *[str(value) for value in manifest.get("ca_case_ids") or []],
        *manifest_cases.keys(),
    ]))
    selected = selected_case_ids or all_ids
    consistency = generation_consistency(generation_dir, indexes)
    cases: dict[str, dict[str, Any]] = {}
    case_reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case_id in selected:
        errors: list[str] = []
        if case_id in seen:
            errors.append("duplicate_case_id")
        seen.add(case_id)
        origin = "KR" if case_id.startswith("KR_") else "CA" if case_id.startswith("CA_") else ""
        if not origin:
            errors.append("invalid_case_origin")
        raw = raw_indexes.get(origin, {}).get(case_id)
        source_field = "raw_text" if origin == "KR" else "main_opinion_text"
        source_text = str((raw or {}).get(source_field) or "")
        if not source_text:
            errors.append("raw_source_missing")
        required = ("segments", "evidence", "graph", "master", "translation",
                    "source_checks", "source_verifier", "translation_checks",
                    "translation_verifier")
        artifacts = {kind: indexes[kind].get(case_id) for kind in required}
        missing = [kind for kind, value in artifacts.items() if not value]
        if missing:
            errors.extend(f"missing_required_artifact:{kind}" for kind in missing)
        master = artifacts.get("master") or {}
        translation = artifacts.get("translation") or {}
        expected = expected_provenance(origin) if origin else {}
        for field in ("source_language", "master_language"):
            if master and str(master.get(field) or "") != expected.get(field):
                errors.append(f"{field}_mismatch")
        if translation and str(translation.get("translation_direction") or "") != expected.get("translation_direction"):
            errors.append("translation_direction_mismatch")
        if master:
            fact_ids = [str(unit.get("fact_id") or "") for unit in master.get("fact_units") or []]
            if not fact_ids or "" in fact_ids or len(fact_ids) != len(set(fact_ids)):
                errors.append("master_fact_ids_missing_or_duplicate")
            source_hash = sha256_text(source_text) if source_text else ""
            if source_hash and str(master.get("source_text_sha256") or "") != source_hash:
                errors.append("source_hash_mismatch")
        if translation:
            translated_ids = [
                str(unit.get("fact_id") or "")
                for unit in translation.get("translated_fact_units") or []
            ]
            master_ids = [
                str(unit.get("fact_id") or "")
                for unit in master.get("fact_units") or []
            ]
            if translated_ids != master_ids:
                errors.append("translation_fact_id_alignment_mismatch")
            if str(translation.get("master_neutral_text") or "") != str(
                master.get("master_neutral_text") or ""
            ):
                errors.append("translation_parent_master_hash_mismatch")
        status = "pass" if not errors else "fail"
        case_report = {
            "case_id": case_id, "case_origin": origin, "status": status,
            "validation_errors": errors,
            "missing_required_artifacts": missing,
        }
        case_reports.append(case_report)
        cases[case_id] = {
            "case_id": case_id, "case_origin": origin,
            "case_subtype": (master or raw or {}).get("case_subtype") or
                            (raw or {}).get("selected_subtype") or "",
            "source_language": expected.get("source_language", ""),
            "master_language": expected.get("master_language", ""),
            "translation_direction": expected.get("translation_direction", ""),
            **expected, "raw_source": source_text, "raw_record": raw or {},
            **artifacts,
        }
    origin_counts = Counter(item["case_origin"] for item in case_reports)
    report = {
        "qc_dataset_version": QC_DATASET_VERSION,
        "generation_dataset_version": (
            consistency["generation_dataset_versions"][0]
            if len(consistency["generation_dataset_versions"]) == 1 else ""
        ),
        "validation_status": (
            "fail" if consistency["status"] == "fail" or
            any(item["status"] == "fail" for item in case_reports) else "pass"
        ),
        "selection_status": (
            "completed_subset" if len(selected) < len(all_ids) else "complete"
        ),
        "selected_case_count": len(selected),
        "selected_origin_counts": dict(origin_counts),
        "generation_consistency": consistency,
        "case_validation": case_reports,
        "created_at": utc_now(),
    }
    immutable_manifest = {
        "qc_dataset_version": QC_DATASET_VERSION,
        "qc_schema_version": QC_SCHEMA_VERSION,
        "generation_dataset_version": report["generation_dataset_version"],
        "generation_output_dir": str(generation_dir.resolve()),
        "raw_kr_input": str(raw_kr_input.resolve()),
        "raw_ca_input": str(raw_ca_input.resolve()),
        "selected_case_ids": selected,
        "selection_status": report["selection_status"],
        "generation_consistency": consistency,
        "cases": [
            {
                "case_id": case_id,
                "case_origin": cases[case_id]["case_origin"],
                **expected_provenance(cases[case_id]["case_origin"]),
                "source_sha256": sha256_text(cases[case_id]["raw_source"]),
                "source_master_sha256": (
                    master_sha256(cases[case_id]["master"])
                    if cases[case_id].get("master") else ""
                ),
                "translation_sha256": (
                    translation_sha256(cases[case_id]["translation"])
                    if cases[case_id].get("translation") else ""
                ),
            }
            for case_id in selected
        ],
        "created_at": utc_now(),
    }
    return report, cases, immutable_manifest


def tokenize(value: str) -> list[str]:
    return re.findall(r"\[[A-Z][A-Z0-9_]*\]|[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?", value.casefold())


DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[./-]\d{1,2}(?:[./-]\d{1,2})?\b|"
    r"\b(?:19|20)\d{2}\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?"
)
AMOUNT_RE = re.compile(
    r"(?:[$₩€£]\s?\d[\d,.]*|\d[\d,.]*\s*(?:원|달러|dollars?|won|USD|KRW))",
    re.I,
)
KOREAN_SURNAMES = set(
    "김이박최정강조윤장임한오서신권황안송전홍유고문양손배백허남심노"
)
GENERIC_ENTITY_MENTIONS = {
    "accident", "antibiotic", "culture and sensitivity test",
    "culture and sensitivity testing", "delivery slip", "lumber", "mother",
    "trailer", "truck", "widow", "surgery", "manufacturing defect",
    "lack of proper bonding", "오토바이", "간호사", "당직의사", "부검기관",
    "사망", "심근경색", "제왕절개수술", "태아", "폐혈전색전증",
    "혈전색전증", "방향목", "오른쪽 러프", "우안", "울창한 산",
}


def _looks_like_identifying_name(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped.casefold() in GENERIC_ENTITY_MENTIONS:
        return False
    if re.fullmatch(r"[가-힣]{3}", stripped):
        return stripped[0] in KOREAN_SURNAMES
    if re.search(
        r"[가-힣]{2,}(?:주식회사|대학교|대학병원|병원|의원|보험회사|공사|학교)$",
        stripped,
    ):
        return True
    english_words = re.findall(r"[A-Za-z][A-Za-z.'&-]*", stripped)
    title_words = [
        word for word in english_words
        if word[0].isupper() and word.casefold() not in {"the", "a", "an"}
    ]
    return (
        len(title_words) >= 2 or
        bool(re.search(
            r"\b(?:Inc|Corp|Corporation|Company|Hospital|University|Clinic|"
            r"Agency|Department)\b",
            stripped,
        ))
    )


def recognition_metrics(
    source: str, master: str, graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_tokens = tokenize(source)
    master_tokens = tokenize(master)
    match = SequenceMatcher(None, source_tokens, master_tokens, autojunk=False)
    blocks = [block for block in match.get_matching_blocks() if block.size]
    longest = max((block.size for block in blocks), default=0)
    matched = sum(block.size for block in blocks)
    source_8 = {
        tuple(source_tokens[index:index + 8])
        for index in range(max(0, len(source_tokens) - 7))
    }
    source_12 = {
        tuple(source_tokens[index:index + 12])
        for index in range(max(0, len(source_tokens) - 11))
    }
    shared_8 = sum(
        tuple(master_tokens[index:index + 8]) in source_8
        for index in range(max(0, len(master_tokens) - 7))
    )
    shared_12 = sum(
        tuple(master_tokens[index:index + 12]) in source_12
        for index in range(max(0, len(master_tokens) - 11))
    )
    dates = sorted(set(DATE_RE.findall(source)) & set(DATE_RE.findall(master)))
    amounts = sorted(set(AMOUNT_RE.findall(source)) & set(AMOUNT_RE.findall(master)))
    actual_names: list[str] = []
    for entity in (graph or {}).get("entities") or []:
        for mention in entity.get("source_mentions") or []:
            mention = str(mention).strip()
            if (
                len(mention) >= 2 and mention in master
                and not PLACEHOLDER_RE.fullmatch(mention)
                and _looks_like_identifying_name(mention)
            ):
                actual_names.append(mention)
    reasons: list[str] = []
    overlap_ratio = matched / max(1, len(master_tokens))
    if dates:
        reasons.append("unique_date_retained")
    if amounts:
        reasons.append("unique_amount_retained")
    if actual_names:
        reasons.append("actual_name_retained")
    if shared_12:
        reasons.append("distinctive_phrase_retained")
    if longest >= 20 or overlap_ratio >= 0.75 or actual_names:
        risk = "high"
    elif longest >= 12 or shared_12 or overlap_ratio >= 0.45 or dates or amounts:
        risk = "medium"
    else:
        risk = "low"
    return {
        "longest_shared_ngram": longest,
        "shared_8gram_count": shared_8,
        "shared_12gram_count": shared_12,
        "verbatim_overlap_ratio": round(overlap_ratio, 6),
        "unique_date_retention": dates,
        "unique_amount_retention": amounts,
        "actual_name_retention": sorted(set(actual_names)),
        "distinctive_phrase_retention": bool(shared_12),
        "source_copy_risk": risk,
        "recognition_risk_reasons": reasons,
    }


def deterministic_source_precheck(case: dict[str, Any]) -> dict[str, Any]:
    master = case["master"]
    existing = source_checks(
        {
            "master_neutral_text": master.get("master_neutral_text") or "",
            "fact_units": master.get("fact_units") or [],
        },
        case["evidence"],
        case["graph"],
        case["source_language"],
    )
    recognition = recognition_metrics(
        case["raw_source"], str(master.get("master_neutral_text") or ""),
        case.get("graph"),
    )
    errors = list(existing.get("errors") or [])
    warnings = list(existing.get("warnings") or [])
    if recognition["actual_name_retention"]:
        errors.append("actual_party_or_institution_name_leakage")
    if recognition["source_copy_risk"] == "high":
        warnings.append("high_recognition_risk_requires_contextual_review")
    elif recognition["source_copy_risk"] == "medium":
        warnings.append("medium_recognition_risk")
    status = "fail" if errors else "warning" if warnings else "pass"
    return {
        "case_id": case["case_id"], "case_origin": case["case_origin"],
        "deterministic_source_qc_status": status,
        "errors": sorted(set(errors)), "warnings": sorted(set(warnings)),
        "source_coverage": master.get("source_coverage") or {},
        "recognition_metrics": recognition,
        "existing_stage2_source_checks": case.get("source_checks") or {},
    }


SOURCE_ARRAY_FIELDS = (
    "unsupported_facts", "overstated_facts", "missing_material_facts",
    "entity_role_errors", "epistemic_status_errors",
    "legal_conclusion_leakage", "jurisdiction_leakage",
)
TRANSLATION_ARRAY_FIELDS = (
    "added_information", "omitted_information", "subject_object_shifts",
    "entity_role_shifts", "epistemic_status_shifts", "polarity_shifts",
    "temporal_relation_shifts", "causal_direction_shifts",
    "spatial_direction_shifts", "number_unit_shifts",
    "legal_terms_reintroduced", "jurisdiction_signals_reintroduced",
    "fact_structure_errors",
)


def aggregate_source_qc(
    case: dict[str, Any], payloads: list[dict[str, Any]],
    provenances: list[dict[str, Any]],
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("source-QC parsing failure")
    combined = {field: [] for field in SOURCE_ARRAY_FIELDS}
    risks: list[str] = []
    sufficiency: list[str] = []
    confidence: list[str] = []
    recommendations: list[str] = []
    statuses: list[str] = []
    reasons: list[str] = []
    for payload in payloads:
        if str(payload.get("case_id")) != case["case_id"]:
            raise ValueError("source-QC case_id mismatch")
        qc = payload.get("source_qc") or {}
        forbidden = {"master_text", "edited_master", "replacement_text"} & set(qc)
        if forbidden:
            raise ValueError("GPT QC attempted text repair")
        for field in SOURCE_ARRAY_FIELDS:
            combined[field].extend(qc.get(field) or [])
        risks.append(str(qc.get("source_copy_risk") or "low"))
        sufficiency.append(str(qc.get("factual_sufficiency") or "insufficient"))
        confidence.append(str(qc.get("model_confidence") or "low"))
        recommendations.append(str(qc.get("recommended_human_action") or "review_only"))
        statuses.append(str(qc.get("model_source_qc_status") or "fail"))
        reasons.extend(qc.get("recognition_risk_reasons") or [])
    rank = {"pass": 0, "warning": 1, "fail": 2}
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    suff_rank = {"sufficient": 0, "marginal": 1, "insufficient": 2}
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    action_rank = {
        "accept": 0, "review_only": 1, "edit_master": 2,
        "regenerate_master": 3, "exclude_case": 4,
    }
    return {
        "case_id": case["case_id"], "source_language": case["source_language"],
        "source_qc": {
            "factual_support_status": max(statuses, key=rank.get),
            **combined,
            "source_copy_risk": max(risks, key=risk_rank.get),
            "recognition_risk_reasons": sorted(set(reasons)),
            "factual_sufficiency": max(sufficiency, key=suff_rank.get),
            "model_source_qc_status": max(statuses, key=rank.get),
            "model_confidence": max(confidence, key=conf_rank.get),
            "recommended_human_action": max(
                recommendations, key=action_rank.get,
            ),
        },
        "model_provenance": {
            "model": provenances[0].get("model") if provenances else "",
            "prompt_version": SOURCE_QC_PROMPT_VERSION[case["source_language"]],
            "request_hashes": [item.get("request_hash") for item in provenances],
            "raw_response_paths": [
                item.get("raw_response_path") for item in provenances
            ],
            "mock": all(bool(item.get("mock")) for item in provenances),
        },
    }


SOURCE_HARD_TYPES = {
    "unsupported_material_fact", "overstated_material_fact", "wrong_actor",
    "wrong_object", "wrong_patient", "wrong_spouse_or_parent", "wrong_owner",
    "wrong_operator", "wrong_driver", "wrong_employer", "wrong_employee",
    "wrong_treating_provider", "wrong_examining_provider",
    "wrong_phone_only_provider", "wrong_prescriber", "wrong_manufacturer",
    "wrong_distributor", "wrong_wholesaler", "wrong_retailer", "wrong_seller",
    "wrong_warranty_issuer", "wrong_vehicle_manufacturer",
    "wrong_allegation_source", "missing_material_entity",
    "missing_material_relation", "merged_distinct_entities",
    "core_event_missing", "harm_missing", "event_harm_sequence_missing",
    "party_allegation_promoted_to_fact",
    "opposing_allegation_promoted_to_fact",
    "expert_opinion_promoted_to_fact", "testimony_promoted_to_fact",
    "assumed_for_pleading_fact_presented_as_established",
    "court_causation_conclusion_retained", "court_fault_allocation_retained",
    "court_damages_calculation_retained", "legal_responsibility_conclusion",
    "fault_allocation", "causation_conclusion", "damages_calculation",
    "final_remedy_or_award", "direct_jurisdiction_leakage",
    "actual_party_or_institution_name_leakage", "coverage_incomplete",
    "source_qc_parsing_failure",
}


def _finding_types(qc: dict[str, Any], fields: Iterable[str]) -> list[str]:
    return [
        str(finding.get("error_type") or "")
        for field in fields for finding in qc.get(field) or []
        if finding.get("error_type")
    ]


def adjudicate_source_qc(
    case: dict[str, Any], deterministic: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    qc = model.get("source_qc") or {}
    types = _finding_types(qc, SOURCE_ARRAY_FIELDS)
    hard_reasons = [
        finding.get("error_type") or "hard_model_finding"
        for field in SOURCE_ARRAY_FIELDS for finding in qc.get(field) or []
        if finding.get("severity") == "hard"
    ]
    hard_reasons.extend(item for item in types if item in SOURCE_HARD_TYPES)
    hard_reasons.extend(deterministic.get("errors") or [])
    coverage = (
        (deterministic.get("source_coverage") or {}).get("coverage_status")
    )
    if coverage != "complete":
        hard_reasons.append("coverage_incomplete")
    warnings = list(deterministic.get("warnings") or [])
    if qc.get("source_copy_risk") == "medium":
        warnings.append("medium_recognition_risk")
    if qc.get("source_copy_risk") == "high":
        hard_reasons.append("severe_recognition_risk")
    if qc.get("factual_sufficiency") == "marginal":
        warnings.append("factual_sufficiency_marginal")
    if qc.get("factual_sufficiency") == "insufficient":
        hard_reasons.append("factual_sufficiency_insufficient")
    if qc.get("model_confidence") == "low":
        warnings.append("model_confidence_low")
    model_status = str(qc.get("model_source_qc_status") or "fail")
    deterministic_status = str(
        deterministic.get("deterministic_source_qc_status") or "fail"
    )
    if model_status != deterministic_status:
        warnings.append("deterministic_and_gpt_disagreement")
    status = "fail" if hard_reasons else "warning" if warnings or model_status == "warning" else "pass"
    return {
        "qc_dataset_version": QC_DATASET_VERSION,
        "generation_dataset_version": case["master"].get("dataset_version") or "",
        "case_id": case["case_id"], "case_origin": case["case_origin"],
        "case_subtype": case["case_subtype"],
        "source_language": case["source_language"],
        "master_language": case["master_language"],
        "model_source_qc_status": model_status,
        "deterministic_source_qc_status": deterministic_status,
        "validated_source_qc_status": status,
        "validation_reasons": sorted(set(
            [*hard_reasons, *warnings]
        )),
        "unresolved_hard_failure": bool(hard_reasons),
        "source_qc": qc,
        "deterministic_precheck": deterministic,
        "previous_source_verifier": case.get("source_verifier") or {},
        "source_master_sha256": master_sha256(case["master"]),
        "source_qc_prompt_version": SOURCE_QC_PROMPT_VERSION[case["source_language"]],
        "model_provenance": model.get("model_provenance") or {},
    }


def deterministic_translation_precheck(
    case: dict[str, Any], master: dict[str, Any], translation: dict[str, Any],
) -> dict[str, Any]:
    target_language = "en" if case["translation_direction"] == "ko_to_en" else "ko"
    checked = translation_checks(master, translation, target_language)
    return {
        "case_id": case["case_id"], "case_origin": case["case_origin"],
        "translation_direction": case["translation_direction"],
        "deterministic_translation_qc_status": checked.get("status") or "fail",
        "errors": checked.get("errors") or [],
        "warnings": checked.get("warnings") or [],
        "checks": checked,
        "existing_stage2_translation_checks": case.get("translation_checks") or {},
    }


TRANSLATION_HARD_TYPES = {
    "fact_id_deletion_or_addition", "material_fact_addition",
    "material_fact_omission", "wrong_actor", "wrong_object",
    "material_entity_role_shift", "clear_polarity_reversal",
    "clear_temporal_reversal", "clear_causal_direction_reversal",
    "material_spatial_direction_shift", "canonical_number_or_unit_change",
    "legal_doctrine_reintroduction", "direct_jurisdiction_signal_reintroduction",
    "placeholder_identity_change", "translation_qc_parsing_failure",
}


def adjudicate_translation_qc(
    case: dict[str, Any], master: dict[str, Any],
    translation: dict[str, Any], deterministic: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    qc = model.get("translation_qc") or {}
    hard_reasons = [
        finding.get("error_type") or "hard_model_finding"
        for field in TRANSLATION_ARRAY_FIELDS for finding in qc.get(field) or []
        if finding.get("severity") == "hard"
    ]
    types = _finding_types(qc, TRANSLATION_ARRAY_FIELDS)
    hard_reasons.extend(item for item in types if item in TRANSLATION_HARD_TYPES)
    hard_reasons.extend(deterministic.get("errors") or [])
    warnings = list(deterministic.get("warnings") or [])
    model_status = str(qc.get("model_translation_qc_status") or "fail")
    deterministic_status = str(
        deterministic.get("deterministic_translation_qc_status") or "fail"
    )
    if qc.get("model_confidence") == "low":
        warnings.append("model_confidence_low")
    if model_status != deterministic_status:
        warnings.append("deterministic_and_gpt_disagreement")
    status = "fail" if hard_reasons else "warning" if warnings or model_status == "warning" else "pass"
    return {
        "qc_dataset_version": QC_DATASET_VERSION,
        "generation_dataset_version": case["master"].get("dataset_version") or "",
        "case_id": case["case_id"], "case_origin": case["case_origin"],
        "case_subtype": case["case_subtype"],
        "translation_direction": case["translation_direction"],
        "model_translation_qc_status": model_status,
        "deterministic_translation_qc_status": deterministic_status,
        "validated_translation_qc_status": status,
        "validation_reasons": sorted(set([*hard_reasons, *warnings])),
        "unresolved_hard_failure": bool(hard_reasons),
        "translation_qc": qc, "deterministic_precheck": deterministic,
        "previous_translation_verifier": case.get("translation_verifier") or {},
        "source_master_sha256": master_sha256(master),
        "translation_parent_master_sha256": master_sha256(master),
        "translation_sha256": translation_sha256(translation),
        "translation_qc_prompt_version": (
            TRANSLATION_QC_PROMPT_VERSION[case["translation_direction"]]
        ),
        "model_provenance": model.get("model_provenance") or {},
    }


def source_review_rows(
    batch_name: str, cases: dict[str, dict[str, Any]],
    validated: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in validated:
        case = cases[str(record["case_id"])]
        qc = record.get("source_qc") or {}
        pre = record.get("deterministic_precheck") or {}
        checks = case["master"].get("deterministic_checks") or {}
        types = _finding_types(qc, SOURCE_ARRAY_FIELDS)
        rows.append({
            "case_id": case["case_id"], "batch_name": batch_name,
            "case_origin": case["case_origin"],
            "case_subtype": case["case_subtype"],
            "source_language": case["source_language"],
            "master_language": case["master_language"],
            "automatic_source_status": record["validated_source_qc_status"],
            "source_error_types": ";".join(sorted(set(types))),
            "source_qc_confidence": qc.get("model_confidence") or "",
            "source_coverage": (
                (pre.get("source_coverage") or {}).get("coverage_status") or ""
            ),
            "core_event_present": checks.get("event_present"),
            "harm_present": checks.get("harm_present"),
            "event_harm_sequence_present": checks.get("event_harm_sequence_present"),
            "material_entities_complete": not any(
                value in {"missing_material_entity", "merged_distinct_entities"}
                for value in types
            ),
            "entity_roles_correct": not bool(qc.get("entity_role_errors")),
            "source_grounding_correct": not bool(
                qc.get("unsupported_facts") or qc.get("overstated_facts")
            ),
            "epistemic_status_correct": not bool(qc.get("epistemic_status_errors")),
            "legal_neutrality": not bool(qc.get("legal_conclusion_leakage")),
            "jurisdiction_neutrality": not bool(qc.get("jurisdiction_leakage")),
            "factual_sufficiency": qc.get("factual_sufficiency") or "",
            "recognition_risk": qc.get("source_copy_risk") or "",
            "human_source_action": "defer", "edited_fact_ids": "",
            "human_source_status": "pending", "reviewer_notes": "",
            "master_text": case["master"].get("master_neutral_text") or "",
            "parent_master_sha256": master_sha256(case["master"]),
            "human_validated_master_text": "",
            "human_validated_fact_units_json": "",
            "edit_reasons": "", "reviewer_id": "", "review_timestamp": "",
        })
    return rows


def _parse_ids(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,]", value or "") if item.strip()]


def _require_unique_case_rows(rows: list[dict[str, str]]) -> None:
    ids = [str(row.get("case_id") or "").strip() for row in rows]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate review row: {duplicates}")


def import_source_reviews(
    review_path: Path, cases: dict[str, dict[str, Any]], output_dir: Path,
) -> list[dict[str, Any]]:
    rows = read_csv(review_path)
    _require_unique_case_rows(rows)
    imported: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if case_id not in cases:
            raise ValueError(f"unknown case ID: {case_id}")
        case = cases[case_id]
        action = str(row.get("human_source_action") or "").strip()
        status = str(row.get("human_source_status") or "").strip()
        reviewer = str(row.get("reviewer_id") or "").strip()
        notes = str(row.get("reviewer_notes") or "").strip()
        if action not in SOURCE_ACTIONS or status not in HUMAN_STATUSES:
            raise ValueError(f"invalid source review action/status for {case_id}")
        if not reviewer:
            raise ValueError(f"reviewer ID required for {case_id}")
        if (action in {"edit_master", "regenerate_master", "exclude_case"} or
                status in {"accepted_with_edits", "rejected"}) and not notes:
            raise ValueError(f"reviewer notes required for {case_id}")
        current_hash = master_sha256(case["master"])
        if str(row.get("parent_master_sha256") or "") != current_hash:
            raise ValueError(f"parent hash changed for {case_id}")
        valid_fact_ids = {
            str(unit.get("fact_id")) for unit in case["master"].get("fact_units") or []
        }
        edited_ids = _parse_ids(str(row.get("edited_fact_ids") or ""))
        if action == "edit_master" or status == "accepted_with_edits":
            if not edited_ids:
                raise ValueError(f"edited fact ID required for {case_id}")
            text = str(row.get("human_validated_master_text") or "").strip()
            raw_units = str(row.get("human_validated_fact_units_json") or "").strip()
            if not text or not raw_units:
                raise ValueError(
                    f"edited master text and fact-unit JSON required for {case_id}"
                )
            units = json.loads(raw_units)
            if not isinstance(units, list):
                raise ValueError(f"fact-unit JSON array required for {case_id}")
            original_unit_ids = [
                str(unit.get("fact_id") or "")
                for unit in case["master"].get("fact_units") or []
            ]
            unit_ids = [str(unit.get("fact_id") or "") for unit in units]
            fact_structure_changed = unit_ids != original_unit_ids
            if fact_structure_changed:
                expected_ids = [
                    f"F{index:03d}" for index in range(1, len(unit_ids) + 1)
                ]
                if unit_ids != expected_ids or len(set(unit_ids)) != len(unit_ids):
                    raise ValueError(
                        f"replacement fact IDs must be unique and sequential "
                        f"for {case_id}"
                    )
                if set(edited_ids) != set(unit_ids):
                    raise ValueError(
                        f"all replacement fact IDs must be marked edited "
                        f"for {case_id}"
                    )
            elif not set(edited_ids) <= valid_fact_ids:
                raise ValueError(f"unknown edited fact ID for {case_id}")
            joined_text = " ".join(
                str(unit.get("master_text") or "").strip() for unit in units
            )
            if " ".join(joined_text.split()) != " ".join(text.split()):
                raise ValueError(
                    f"validated master text does not match fact units "
                    f"for {case_id}"
                )
        else:
            text = str(case["master"].get("master_neutral_text") or "")
            units = list(case["master"].get("fact_units") or [])
            original_unit_ids = [
                str(unit.get("fact_id") or "") for unit in units
            ]
            unit_ids = list(original_unit_ids)
            fact_structure_changed = False
        timestamp = str(row.get("review_timestamp") or "").strip() or utc_now()
        audit = {
            **{field: row.get(field, "") for field in SOURCE_REVIEW_FIELDS},
            "review_timestamp": timestamp,
            "imported_at": utc_now(),
            "review_source_sha256": sha256_text(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
            ),
        }
        imported.append(audit)
        if status in {"accepted", "accepted_with_edits"}:
            record = {
                "case_id": case_id,
                "parent_master_sha256": current_hash,
                "human_validated_master_text": text,
                "human_validated_fact_units": units,
                "edited_fact_ids": edited_ids,
                "edit_reasons": _parse_ids(str(row.get("edit_reasons") or "")),
                "fact_structure_changed": fact_structure_changed,
                "original_fact_ids": original_unit_ids,
                "human_validated_fact_ids": unit_ids,
                "reviewer_id": reviewer, "review_timestamp": timestamp,
                "human_source_status": status,
                "human_validated_master_sha256": stable_hash({
                    "master_neutral_text": text, "fact_units": units,
                }),
            }
            validated.append(record)
            if status == "accepted_with_edits":
                edits.append(record)
                stale.append({
                    "case_id": case_id,
                    "old_parent_master_sha256": current_hash,
                    "new_parent_master_sha256": record[
                        "human_validated_master_sha256"
                    ],
                    "translation_requires_regeneration": True,
                    "reason": "human_validated_master_edited",
                })
    append_audit_jsonl(
        output_dir / "human_source_reviews_imported.jsonl", imported,
    )
    merge_jsonl(output_dir / "human_validated_masters.jsonl", validated)
    append_audit_jsonl(output_dir / "master_edit_history.jsonl", edits)
    merge_jsonl(output_dir / "translations_requiring_regeneration.jsonl", stale)
    return imported


def validated_master_record(
    case: dict[str, Any], review: dict[str, Any],
) -> dict[str, Any]:
    return {
        **case["master"],
        "master_neutral_text": review["human_validated_master_text"],
        "fact_units": review["human_validated_fact_units"],
        "human_source_status": review["human_source_status"],
        "human_validated_master_sha256": review["human_validated_master_sha256"],
    }


def translation_readiness(
    cases: dict[str, dict[str, Any]], source_qc: dict[str, dict[str, Any]],
    human_masters: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ready: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        automatic = source_qc.get(case_id) or {}
        review = human_masters.get(case_id)
        if not review or review.get("human_source_status") not in {
            "accepted", "accepted_with_edits",
        }:
            continue
        if automatic.get("validated_source_qc_status") == "fail" and (
            review.get("human_source_status") != "accepted_with_edits"
        ):
            continue
        master = validated_master_record(case, review)
        translation = case.get("translation")
        current_hash = master_sha256(master)
        translation_matches = bool(
            translation and
            str(translation.get("master_neutral_text") or "") ==
            str(master.get("master_neutral_text") or "")
        )
        if not translation_matches:
            stale.append({
                "case_id": case_id,
                "translation_requires_regeneration": True,
                "new_parent_master_sha256": current_hash,
                "reason": "translation_parent_master_hash_mismatch",
            })
            continue
        ready.append({
            "case": case, "master": master, "translation": translation,
            "source_master_sha256": current_hash,
        })
    return ready, stale


def translation_review_rows(
    batch_name: str, ready_by_case: dict[str, dict[str, Any]],
    validated: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in validated:
        item = ready_by_case[str(record["case_id"])]
        master = item["master"]
        translation = item["translation"]
        qc = record.get("translation_qc") or {}
        pre = record.get("deterministic_precheck") or {}
        checks = pre.get("checks") or {}
        types = _finding_types(qc, TRANSLATION_ARRAY_FIELDS)
        rows.append({
            "case_id": record["case_id"], "batch_name": batch_name,
            "translation_direction": record["translation_direction"],
            "master_text": master.get("master_neutral_text") or "",
            "translation_text": translation.get("translated_neutral_text") or "",
            "automatic_translation_status": record[
                "validated_translation_qc_status"
            ],
            "translation_error_types": ";".join(sorted(set(types))),
            "fact_ids_match": checks.get("fact_id_match"),
            "placeholder_identity_match": checks.get("placeholder_identity_match"),
            "subject_object_preserved": not bool(qc.get("subject_object_shifts")),
            "entity_roles_preserved": not bool(qc.get("entity_role_shifts")),
            "epistemic_status_preserved": not bool(
                qc.get("epistemic_status_shifts")
            ),
            "negation_preserved": not bool(qc.get("polarity_shifts")),
            "temporal_order_preserved": not bool(
                qc.get("temporal_relation_shifts")
            ),
            "causal_direction_preserved": not bool(
                qc.get("causal_direction_shifts")
            ),
            "spatial_direction_preserved": not bool(
                qc.get("spatial_direction_shifts")
            ),
            "numbers_units_preserved": not bool(qc.get("number_unit_shifts")),
            "legal_terms_absent": not bool(qc.get("legal_terms_reintroduced")),
            "jurisdiction_signals_absent": not bool(
                qc.get("jurisdiction_signals_reintroduced")
            ),
            "translation_equivalence": qc.get("semantic_equivalence") or "",
            "human_translation_action": "defer", "edited_fact_ids": "",
            "human_translation_status": "pending", "reviewer_notes": "",
            "master_sha256": master_sha256(master),
            "parent_translation_sha256": translation_sha256(translation),
            "human_validated_translation_text": "",
            "human_validated_translation_fact_units_json": "",
            "edit_reasons": "", "reviewer_id": "", "review_timestamp": "",
        })
    return rows


def import_translation_reviews(
    review_path: Path, ready_by_case: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = read_csv(review_path)
    _require_unique_case_rows(rows)
    imported: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if case_id not in ready_by_case:
            raise ValueError(f"translation not ready or unknown case ID: {case_id}")
        item = ready_by_case[case_id]
        translation = item["translation"]
        master = item["master"]
        action = str(row.get("human_translation_action") or "").strip()
        status = str(row.get("human_translation_status") or "").strip()
        reviewer = str(row.get("reviewer_id") or "").strip()
        notes = str(row.get("reviewer_notes") or "").strip()
        if action not in TRANSLATION_ACTIONS or status not in HUMAN_STATUSES:
            raise ValueError(f"invalid translation action/status for {case_id}")
        if not reviewer:
            raise ValueError(f"reviewer ID required for {case_id}")
        if (action in {"edit_translation", "regenerate_translation", "exclude_case"}
                or status in {"accepted_with_edits", "rejected"}) and not notes:
            raise ValueError(f"reviewer notes required for {case_id}")
        if str(row.get("master_sha256") or "") != master_sha256(master):
            raise ValueError(f"validated master hash changed for {case_id}")
        if str(row.get("parent_translation_sha256") or "") != translation_sha256(translation):
            raise ValueError(f"translation parent hash changed for {case_id}")
        valid_fact_ids = {
            str(unit.get("fact_id"))
            for unit in translation.get("translated_fact_units") or []
        }
        edited_ids = _parse_ids(str(row.get("edited_fact_ids") or ""))
        if action == "edit_translation" or status == "accepted_with_edits":
            if not edited_ids or not set(edited_ids) <= valid_fact_ids:
                raise ValueError(f"valid edited fact ID required for {case_id}")
            text = str(
                row.get("human_validated_translation_text") or ""
            ).strip()
            raw_units = str(
                row.get("human_validated_translation_fact_units_json") or ""
            ).strip()
            if not text or not raw_units:
                raise ValueError(
                    f"edited translation and fact-unit JSON required for {case_id}"
                )
            units = json.loads(raw_units)
            if not isinstance(units, list):
                raise ValueError(f"translation fact-unit JSON required for {case_id}")
            unit_ids = [str(unit.get("fact_id") or "") for unit in units]
            if unit_ids != [
                str(unit.get("fact_id") or "")
                for unit in translation.get("translated_fact_units") or []
            ]:
                raise ValueError(f"translation fact structure changed for {case_id}")
            canonical_units: list[dict[str, Any]] = []
            for unit in units:
                canonical = dict(unit)
                if not str(canonical.get("translated_text") or "").strip():
                    legacy_text = str(canonical.get("master_text") or "").strip()
                    if legacy_text:
                        canonical["translated_text"] = legacy_text
                if not str(canonical.get("translated_text") or "").strip():
                    raise ValueError(
                        f"translated_text required for {case_id}/"
                        f"{canonical.get('fact_id')}"
                    )
                canonical.pop("master_text", None)
                canonical_units.append(canonical)
            units = canonical_units
            joined_text = " ".join(
                str(unit["translated_text"]).strip() for unit in units
            )
            if " ".join(joined_text.split()) != " ".join(text.split()):
                raise ValueError(
                    f"validated translation text does not match fact units "
                    f"for {case_id}"
                )
        else:
            text = str(translation.get("translated_neutral_text") or "")
            units = list(translation.get("translated_fact_units") or [])
        timestamp = str(row.get("review_timestamp") or "").strip() or utc_now()
        audit = {
            **{field: row.get(field, "") for field in TRANSLATION_REVIEW_FIELDS},
            "review_timestamp": timestamp, "imported_at": utc_now(),
            "review_source_sha256": sha256_text(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
            ),
        }
        imported.append(audit)
        if status in {"accepted", "accepted_with_edits"}:
            record = {
                "case_id": case_id,
                "source_master_sha256": master_sha256(master),
                "parent_translation_sha256": translation_sha256(translation),
                "human_validated_translation_text": text,
                "human_validated_translation_fact_units": units,
                "edited_fact_ids": edited_ids,
                "edit_reasons": _parse_ids(str(row.get("edit_reasons") or "")),
                "reviewer_id": reviewer, "review_timestamp": timestamp,
                "human_translation_status": status,
                "human_validated_translation_sha256": stable_hash({
                    "translated_neutral_text": text,
                    "translated_fact_units": units,
                }),
            }
            validated.append(record)
            if status == "accepted_with_edits":
                edits.append(record)
    append_audit_jsonl(
        output_dir / "human_translation_reviews_imported.jsonl", imported,
    )
    merge_jsonl(output_dir / "human_validated_translations.jsonl", validated)
    append_audit_jsonl(output_dir / "translation_edit_history.jsonl", edits)
    return imported


def _explicit_instruction(origin: str, language: str) -> str:
    if origin == "KR":
        return (
            "한국법을 적용하여 다음 사실관계를 분석하시오."
            if language == "ko" else
            "Analyze the following fact pattern under Korean law."
        )
    return (
        "캘리포니아법을 적용하여 다음 사실관계를 분석하시오."
        if language == "ko" else
        "Analyze the following fact pattern under California law."
    )


def finalize_pairs(
    output_dir: Path, cases: dict[str, dict[str, Any]],
    source_qc: dict[str, dict[str, Any]],
    translation_qc: dict[str, dict[str, Any]],
    human_masters: dict[str, dict[str, Any]],
    human_translations: dict[str, dict[str, Any]],
    experiments_dir: Path,
    manifests_dir: Path,
    batch_name: str,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        source_review = human_masters.get(case_id)
        translation_review = human_translations.get(case_id)
        sqc = source_qc.get(case_id) or {}
        tqc = translation_qc.get(case_id) or {}
        if not source_review or not translation_review:
            pending.append({
                "case_id": case_id, "reason": "pending_human_review",
            })
            continue
        source_status = str(source_review.get("human_source_status") or "")
        translation_status = str(
            translation_review.get("human_translation_status") or ""
        )
        master_hash = str(
            source_review.get("human_validated_master_sha256") or ""
        )
        translation_parent_hash = str(
            translation_review.get("source_master_sha256") or ""
        )
        unresolved = bool(
            sqc.get("unresolved_hard_failure") and
            source_status != "accepted_with_edits"
        ) or bool(
            tqc.get("unresolved_hard_failure") and
            translation_status != "accepted_with_edits"
        )
        usable = (
            source_status in {"accepted", "accepted_with_edits"} and
            translation_status in {"accepted", "accepted_with_edits"} and
            not unresolved and master_hash == translation_parent_hash
        )
        if not usable:
            rejected.append({
                "case_id": case_id,
                "human_source_status": source_status,
                "human_translation_status": translation_status,
                "unresolved_hard_failure": unresolved,
                "master_translation_parent_hash_match": (
                    master_hash == translation_parent_hash
                ),
            })
            continue
        source_text = str(source_review["human_validated_master_text"])
        translated_text = str(
            translation_review["human_validated_translation_text"]
        )
        ko_text = source_text if case["case_origin"] == "KR" else translated_text
        en_text = translated_text if case["case_origin"] == "KR" else source_text
        accepted.append({
            "qc_dataset_version": QC_DATASET_VERSION,
            "generation_dataset_version": (
                case["master"].get("dataset_version") or ""
            ),
            "case_id": case_id, "case_origin": case["case_origin"],
            "case_subtype": case["case_subtype"],
            "source_language": case["source_language"],
            "master_language": case["master_language"],
            "translation_direction": case["translation_direction"],
            "neutral_fact_ko": ko_text, "neutral_fact_en": en_text,
            "ko_generation_type": case["ko_generation_type"],
            "en_generation_type": case["en_generation_type"],
            "automatic_source_qc_status": sqc.get(
                "validated_source_qc_status"
            ),
            "human_source_qc_status": source_status,
            "automatic_translation_qc_status": tqc.get(
                "validated_translation_qc_status"
            ),
            "human_translation_qc_status": translation_status,
            "overall_human_qc_status": "accepted",
            "case_is_finally_usable": True,
            "source_master_sha256": master_hash,
            "translation_parent_master_sha256": translation_parent_hash,
            "translation_sha256": translation_review.get(
                "human_validated_translation_sha256"
            ),
            "source_qc_prompt_version": sqc.get("source_qc_prompt_version"),
            "translation_qc_prompt_version": tqc.get(
                "translation_qc_prompt_version"
            ),
            "reviewer_id": translation_review.get("reviewer_id"),
            "review_timestamp": translation_review.get("review_timestamp"),
        })
    atomic_write_jsonl(output_dir / "accepted_pairs.jsonl", accepted)
    atomic_write_jsonl(output_dir / "rejected_pairs.jsonl", rejected)
    atomic_write_jsonl(output_dir / "pending_pairs.jsonl", pending)

    no_jurisdiction: list[dict[str, Any]] = []
    explicit: list[dict[str, Any]] = []
    secure_rows: list[dict[str, Any]] = []
    for index, record in enumerate(accepted, 1):
        source_case = cases[str(record["case_id"])]
        blinded = f"PAIR_{index:04d}"
        ko_condition = f"{blinded}_KO"
        en_condition = f"{blinded}_EN"
        no_jurisdiction.append({
            "pair_id": blinded, "case_id_blinded": blinded,
            "ko_condition": {
                "condition_id": ko_condition, "input_language": "ko",
                "fact_pattern": record["neutral_fact_ko"],
            },
            "en_condition": {
                "condition_id": en_condition, "input_language": "en",
                "fact_pattern": record["neutral_fact_en"],
            },
        })
        explicit.append({
            "pair_id": blinded, "case_id_blinded": blinded,
            "ko_condition": {
                "condition_id": f"{blinded}_EXPLICIT_KO",
                "input_language": "ko",
                "jurisdiction_instruction": _explicit_instruction(
                    record["case_origin"], "ko"
                ),
                "fact_pattern": record["neutral_fact_ko"],
            },
            "en_condition": {
                "condition_id": f"{blinded}_EXPLICIT_EN",
                "input_language": "en",
                "jurisdiction_instruction": _explicit_instruction(
                    record["case_origin"], "en"
                ),
                "fact_pattern": record["neutral_fact_en"],
            },
        })
        for condition in (ko_condition, en_condition):
            secure_rows.append({
                "blinded_condition_id": condition,
                "original_case_id": record["case_id"],
                "case_origin": record["case_origin"],
                "source_language": record["source_language"],
                "translation_direction": record["translation_direction"],
                "case_subtype": record["case_subtype"],
                "hidden_original_outcome": (
                    source_case["raw_record"].get("original_outcome") or ""
                ),
                "review_status": "accepted",
            })
    atomic_write_jsonl(
        experiments_dir / "no_jurisdiction_pairs.jsonl", no_jurisdiction,
    )
    atomic_write_jsonl(
        experiments_dir / "explicit_jurisdiction_pairs.jsonl", explicit,
    )
    write_csv(
        manifests_dir / "secure_case_manifest.csv",
        [
            "blinded_condition_id", "original_case_id", "case_origin",
            "source_language", "translation_direction", "case_subtype",
            "hidden_original_outcome", "review_status",
        ],
        secure_rows,
    )
    report = {
        "qc_dataset_version": QC_DATASET_VERSION,
        "batch_name": batch_name, "accepted_count": len(accepted),
        "rejected_count": len(rejected), "pending_count": len(pending),
        "human_source_accepted_count": sum(
            bool(item) for item in human_masters.values()
        ),
        "human_translation_accepted_count": sum(
            bool(item) for item in human_translations.values()
        ),
        "unresolved_hard_failure_count": sum(
            bool((source_qc.get(case_id) or {}).get("unresolved_hard_failure"))
            for case_id in cases
        ) + sum(
            bool((translation_qc.get(case_id) or {}).get("unresolved_hard_failure"))
            for case_id in cases
        ),
        "stale_translation_count": len(read_records(
            output_dir / "translations_requiring_regeneration.jsonl"
        )),
        "stage_b_unlocked": (
            batch_name == "stage-a" and len(accepted) == 6 and
            not rejected and not pending and
            not read_records(output_dir / "translations_requiring_regeneration.jsonl")
        ),
        "next_batch_unlocked": (
            len(accepted) == sum(BATCH_TARGETS[batch_name].values()) and
            not rejected and not pending and
            not read_records(output_dir / "translations_requiring_regeneration.jsonl")
        ),
        "created_at": utc_now(),
    }
    (output_dir / "batch_reports").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "batch_reports" / f"{batch_name.replace('-', '_')}.json",
        report,
    )
    return report


def summarize(output_dir: Path, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = read_records(output_dir / "validated_source_qc.jsonl")
    translation = read_records(output_dir / "validated_translation_qc.jsonl")
    accepted = read_records(output_dir / "accepted_pairs.jsonl")
    rejected = read_records(output_dir / "rejected_pairs.jsonl")
    pending = read_records(output_dir / "pending_pairs.jsonl")
    source_reviews = list(
        by_case(output_dir / "human_source_reviews_imported.jsonl").values()
    )
    translation_reviews = list(
        by_case(output_dir / "human_translation_reviews_imported.jsonl").values()
    )

    def rate(rows: list[dict[str, Any]], field: str, arrays: tuple[str, ...]) -> float:
        if not rows:
            return 0.0
        count = sum(
            bool((row.get(field) or {}).get(name))
            for row in rows for name in arrays
        )
        return round(count / len(rows), 6)

    source_by_id = {str(row.get("case_id")): row for row in source}
    translation_by_id = {str(row.get("case_id")): row for row in translation}

    def group_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        ids = {str(case["case_id"]) for case in selected}
        source_rows = [source_by_id[case_id] for case_id in ids if case_id in source_by_id]
        translation_rows = [
            translation_by_id[case_id]
            for case_id in ids if case_id in translation_by_id
        ]
        return {
            "raw_case_count": len(ids),
            "automatic_source": dict(Counter(
                str(row.get("validated_source_qc_status") or "missing")
                for row in source_rows
            )),
            "automatic_translation": dict(Counter(
                str(row.get("validated_translation_qc_status") or "missing")
                for row in translation_rows
            )),
            "final_accepted": sum(
                str(row.get("case_id")) in ids for row in accepted
            ),
            "final_rejected": sum(
                str(row.get("case_id")) in ids for row in rejected
            ),
            "pending": sum(str(row.get("case_id")) in ids for row in pending),
        }

    by_origin = {
        origin: group_summary([
            case for case in cases.values() if case.get("case_origin") == origin
        ])
        for origin in ("KR", "CA")
    }
    subtypes = sorted({
        str(case.get("case_subtype") or "unknown") for case in cases.values()
    })
    by_subtype = {
        subtype: group_summary([
            case for case in cases.values()
            if str(case.get("case_subtype") or "unknown") == subtype
        ])
        for subtype in subtypes
    }
    lengths = [
        len(str(row.get("neutral_fact_ko") or "")) +
        len(str(row.get("neutral_fact_en") or ""))
        for row in accepted
    ]
    summary = {
        "qc_dataset_version": QC_DATASET_VERSION,
        "raw_case_count": len(cases),
        "automatic_source_counts": dict(Counter(
            str(row.get("validated_source_qc_status") or "missing")
            for row in source
        )),
        "automatic_translation_counts": dict(Counter(
            str(row.get("validated_translation_qc_status") or "missing")
            for row in translation
        )),
        "recognition_risk_distribution": dict(Counter(
            str((row.get("source_qc") or {}).get("source_copy_risk") or "missing")
            for row in source
        )),
        "human_source_counts": dict(Counter(
            str(row.get("human_source_status") or "pending")
            for row in source_reviews
        )),
        "human_translation_counts": dict(Counter(
            str(row.get("human_translation_status") or "pending")
            for row in translation_reviews
        )),
        "unsupported_fact_rate": rate(
            source, "source_qc", ("unsupported_facts",)
        ),
        "missing_material_fact_rate": rate(
            source, "source_qc", ("missing_material_facts",)
        ),
        "entity_role_error_rate": rate(
            source, "source_qc", ("entity_role_errors",)
        ),
        "epistemic_error_rate": rate(
            source, "source_qc", ("epistemic_status_errors",)
        ),
        "legal_leakage_rate": rate(
            source, "source_qc", ("legal_conclusion_leakage",)
        ),
        "jurisdiction_leakage_rate": rate(
            source, "source_qc", ("jurisdiction_leakage",)
        ),
        "translation_addition_rate": rate(
            translation, "translation_qc", ("added_information",)
        ),
        "translation_omission_rate": rate(
            translation, "translation_qc", ("omitted_information",)
        ),
        "subject_object_shift_rate": rate(
            translation, "translation_qc", ("subject_object_shifts",)
        ),
        "translation_entity_role_shift_rate": rate(
            translation, "translation_qc", ("entity_role_shifts",)
        ),
        "translation_epistemic_shift_rate": rate(
            translation, "translation_qc", ("epistemic_status_shifts",)
        ),
        "translation_polarity_shift_rate": rate(
            translation, "translation_qc", ("polarity_shifts",)
        ),
        "translation_number_unit_shift_rate": rate(
            translation, "translation_qc", ("number_unit_shifts",)
        ),
        "legal_term_reintroduction_rate": rate(
            translation, "translation_qc", ("legal_terms_reintroduced",)
        ),
        "final_accepted_pair_count": len(accepted),
        "final_rejected_pair_count": len(rejected),
        "pending_count": len(pending),
        "subtype_distribution": dict(Counter(
            str(case.get("case_subtype") or "unknown") for case in cases.values()
        )),
        "human_edited_master_count": len(read_records(
            output_dir / "master_edit_history.jsonl"
        )),
        "human_edited_translation_count": len(read_records(
            output_dir / "translation_edit_history.jsonl"
        )),
        "neutral_text_length_distribution": {
            "count": len(lengths),
            "minimum": min(lengths) if lengths else 0,
            "maximum": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
        "by_origin": by_origin,
        "by_subtype": by_subtype,
        "by_batch": {
            batch: {
                "human_source_rows": sum(
                    str(row.get("batch_name")) == batch
                    for row in source_reviews
                ),
                "human_translation_rows": sum(
                    str(row.get("batch_name")) == batch
                    for row in translation_reviews
                ),
            }
            for batch in sorted({
                str(row.get("batch_name") or "")
                for row in [*source_reviews, *translation_reviews]
                if row.get("batch_name")
            })
        },
    }
    atomic_write_json(output_dir / "dataset_summary.json", summary)
    rows = [
        {"metric": key, "value": json.dumps(value, ensure_ascii=False)}
        for key, value in summary.items()
    ]
    write_csv(output_dir / "qc_summary.csv", ["metric", "value"], rows)
    return summary
