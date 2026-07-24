from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl, atomic_write_text
from pipeline.llm_client import LLMClient
from pipeline.source_segmentation import (
    segment_source,
    segmentation_record,
    select_candidate_chunks,
)
from pipeline.stage2_input import sha256_file, validate_inputs
from pipeline.stage2_runtime import (
    append_errors,
    append_run_history,
    by_case,
    merge_by_case,
    quarantine_record,
    resolve_case_ids,
    run_parallel,
    select_by_origin_limit,
    write_usage,
)
from pipeline.stage2_v3_pipeline import (
    build_entity_relation_graph,
    canonical_quantity_tokens,
    coverage_record,
    extract_evidence,
    neutralize,
    normalize_entity_relation_graph,
    source_checks,
    stable_hash,
    translate,
    translation_checks,
    validate_grounding_payload,
    validate_translation_verifier_payload,
    verify_grounding,
    verify_translation,
)
from pipeline.stage2_v3_schema import DATASET_VERSION


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("outputs/neutral/stage2-neutral-35x35-v3")
REGRESSION_CASE_IDS = [
    "KR_043490cec7ae93fa",
    "KR_09a496b96b9d302d",
    "KR_0f10f050f02ad48d",
    "CA_90588b6bc671dd08",
    "CA_78a282aae14272a7",
    "CA_59b8f41e992ca4b0",
]
V3_PROMPTS = [
    "extract_evidence_ko_v3.txt",
    "extract_evidence_en_v3.txt",
    "extract_entity_relations_ko_v2.txt",
    "extract_entity_relations_en_v2.txt",
    "extract_entity_relations_ko_v3.txt",
    "extract_entity_relations_en_v3.txt",
    "neutralize_ko_v4.txt",
    "neutralize_en_v4.txt",
    "neutralize_ko_v6.txt",
    "neutralize_en_v6.txt",
    "verify_grounding_and_roles_ko_v3.txt",
    "verify_grounding_and_roles_en_v3.txt",
    "verify_grounding_and_roles_ko_v4.txt",
    "verify_grounding_and_roles_en_v4.txt",
    "translate_ko_to_en_v4.txt",
    "translate_en_to_ko_v4.txt",
    "verify_translation_relations_ko_en_v3.txt",
    "verify_translation_relations_en_ko_v3.txt",
    "verify_translation_relations_ko_en_v4.txt",
    "verify_translation_relations_en_ko_v4.txt",
]
HUMAN_QC_FIELDS = [
    "case_id", "case_origin", "case_subtype", "source_coverage_complete",
    "core_event_preserved", "harm_preserved", "causal_sequence_preserved",
    "source_grounding_correct", "material_facts_complete",
    "legal_conclusion_removed", "fault_percentage_removed",
    "causation_conclusion_removed", "evidentiary_evaluation_removed",
    "damages_calculation_removed", "entity_placeholders_consistent",
    "actor_action_mapping_correct", "subject_object_mapping_correct",
    "ownership_employment_mapping_correct", "medical_provider_roles_correct",
    "manufacturer_distributor_roles_correct", "warranty_issuer_correct",
    "allegation_attribution_correct", "ko_en_meaning_equivalent",
    "negation_preserved", "temporal_order_preserved",
    "directionality_preserved", "numbers_units_preserved",
    "jurisdiction_leakage_absent", "human_qc_status", "usable", "notes",
]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Stage 2 v3 source-grounded neutral fact pipeline."
    )
    value.add_argument(
        "--kr-input", type=Path,
        default=Path("outputs/raw/kr_v4/kr_cases_selected_35.jsonl"),
    )
    value.add_argument(
        "--ca-input", type=Path,
        default=Path("outputs/raw/ca_v4/ca_cases_selected_35.jsonl"),
    )
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--model", default="gpt-5.6-luna")
    value.add_argument("--base-url", default="https://gw.letsur.ai/v1")
    value.add_argument("--api-key-env", default="LETSUR_API_KEY")
    value.add_argument("--case-id", action="append")
    value.add_argument("--case-id-file", type=Path)
    value.add_argument("--batch-name", default="stage-a-calibration")
    value.add_argument("--max-cases-per-origin", type=int)
    value.add_argument("--concurrency", type=int, default=2)
    value.add_argument("--max-retries", type=int, default=5)
    value.add_argument("--max-input-tokens", type=int, default=12000)
    value.add_argument("--chunk-overlap-sentences", type=int, default=2)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--retry-failed", action="store_true")
    value.add_argument("--retry-warnings", action="store_true")
    value.add_argument("--recheck-deterministic", action="store_true")
    value.add_argument("--regenerate", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--mock-response-dir", type=Path)
    value.add_argument("--stop-on-hard-failure", action="store_true")
    value.add_argument("--max-hard-failure-rate", type=float, default=0.10)
    value.add_argument("--max-api-failure-rate", type=float, default=0.05)
    return value


def _load_local_api_key(name: str) -> None:
    if os.environ.get(name):
        return
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        if key.strip() == name:
            os.environ[name] = raw_value.strip().strip("\"'")
            return


def _snapshot_prompts(output: Path) -> None:
    for filename in V3_PROMPTS:
        source = ROOT / "prompts" / filename
        destination = output / "prompts" / filename
        content = source.read_text(encoding="utf-8")
        if destination.exists() and destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Immutable prompt snapshot differs: {destination}")
        if not destination.exists():
            atomic_write_text(destination, content)


def _write_input_artifacts(
    kr_path: Path, ca_path: Path, output: Path
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    cases, report, manifest = validate_inputs(kr_path, ca_path, None)
    report = {**report, "dataset_version": DATASET_VERSION}
    manifest = {
        **manifest,
        "dataset_version": DATASET_VERSION,
        "source_field_mapping": {
            "KR": {
                "case_origin": "KR",
                "source_language": "ko",
                "source_text_field": "raw_text",
            },
            "CA": {
                "case_origin": "CA",
                "source_language": "en",
                "source_text_field": "main_opinion_text",
            },
        },
        "case_inputs": [
            {
                "case_id": case.case_id,
                "case_origin": case.case_origin,
                "source_language": case.source_language,
                "source_text_field": case.source_text_field,
                "source_text_sha256": case.source_text_sha256,
                "case_subtype": case.case_subtype,
                "source_dataset": case.source_dataset,
                "source_record_id": case.source_record_id,
            }
            for case in cases
        ],
    }
    atomic_write_json(output / "input_validation_report.json", report)
    atomic_write_json(output / "input_manifest.json", manifest)
    return cases, report, manifest


def _selected_records(
    cases: list[Any], args: argparse.Namespace
) -> list[Any]:
    selected_ids = resolve_case_ids(args.case_id, args.case_id_file)
    selected = select_by_origin_limit(
        cases, selected_ids, args.max_cases_per_origin
    )
    if not selected:
        raise ValueError("No cases selected")
    return selected


def _merge_output(
    path: Path, new_rows: list[dict[str, Any]], *, origin: str | None = None
) -> dict[str, dict[str, Any]]:
    existing = list(by_case(path).values())
    rows = [
        row for row in new_rows
        if origin is None or row.get("case_origin") == origin
    ]
    merged = merge_by_case(existing, rows)
    atomic_write_jsonl(path, merged)
    return {str(row["case_id"]): row for row in merged}


def _update_current_quarantine(
    path: Path,
    selected_case_ids: list[str],
    current_records: list[dict[str, Any]],
) -> None:
    """Replace stale selected-case failures while retaining other batches."""
    current = by_case(path)
    for case_id in selected_case_ids:
        current.pop(case_id, None)
    for record in current_records:
        current[str(record["case_id"])] = record
    atomic_write_jsonl(path, current.values())


def _should_run(
    existing: dict[str, Any] | None,
    status_field: str,
    args: argparse.Namespace,
) -> bool:
    if existing is None or args.regenerate:
        return True
    status = str(existing.get(status_field) or "fail")
    if args.retry_failed and status == "fail":
        return True
    if args.retry_warnings and status == "warning":
        return True
    return False


def _run_stage(
    items: list[Any],
    worker: Callable[[Any], dict[str, Any]],
    concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not items:
        return [], []
    return run_parallel(items, worker, concurrency)


def _counts(
    records: dict[str, dict[str, Any]], field: str, manifest_count: int = 70
) -> dict[str, int]:
    normalized = []
    for row in records.values():
        value = str(row.get(field) or "missing")
        if value == "complete":
            value = "pass"
        elif value == "incomplete":
            value = "fail"
        normalized.append(value)
    values = Counter(normalized)
    recognized = {
        "pass": values["pass"],
        "warning": values["warning"],
        "fail": values["fail"],
    }
    recognized["missing"] = max(0, manifest_count - sum(recognized.values()))
    return recognized


def _regression_report(
    output: Path,
    case_ids: set[str],
    segments: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        tests.append({"name": name, "status": "pass" if passed else "fail", "details": details})

    add(
        "fixed_six_cases_present",
        set(REGRESSION_CASE_IDS) <= case_ids,
        sorted(set(REGRESSION_CASE_IDS) - case_ids),
    )
    normal_pairs = [
        ("다툼이 없었다", "It was undisputed"),
        ("3분의 1", "one-third"),
        ("8주", "8 weeks"),
        ("23세", "23 years old"),
        ("25%", "25 percent"),
        ("7.6 m였고", "7.6 m long"),
    ]
    for index, (ko, en) in enumerate(normal_pairs, 1):
        add(
            f"normal_translation_surface_{index}",
            canonical_quantity_tokens(ko) == canonical_quantity_tokens(en),
            {"ko": dict(canonical_quantity_tokens(ko)), "en": dict(canonical_quantity_tokens(en))},
        )

    base_master = {
        "case_origin": "KR",
        "master_neutral_text": "[PERSON_A]는 2 m 이동한 뒤 사망했다.",
        "fact_units": [{
            "fact_id": "F001",
            "master_text": "[PERSON_A]는 2 m 이동한 뒤 사망했다.",
            "relation_ids": ["R001"],
            "realized_relations": [{
                "subject_placeholder": "[PERSON_A]",
                "relation_type": "moved_toward",
                "object_placeholder": "[PERSON_B]",
            }],
        }],
    }
    corruptions = {
        "number_change": "[PERSON_A] moved 20 m and then died.",
        "harm_change": "[PERSON_A] moved 2 m and then suffered a minor injury.",
        "placeholder_change": "[PERSON_B] moved 2 m and then died.",
        "legal_insertion": "[PERSON_A] negligently moved 2 m and then died.",
        "jurisdiction_insertion": "[PERSON_A] moved 2 m in California and then died.",
    }
    for name, text in corruptions.items():
        payload = {
            "translated_neutral_text": text,
            "translated_fact_units": [{
                "fact_id": "F001",
                "translated_text": text,
                "relation_ids": ["R001"],
                "realized_relations": [{
                    "subject_placeholder": "[PERSON_A]" if name != "placeholder_change" else "[PERSON_B]",
                    "relation_type": "moved_toward",
                    "object_placeholder": "[PERSON_B]",
                }],
            }],
        }
        checked = translation_checks(base_master, payload, "en")
        add(f"corruption_{name}_hard_fail", checked["status"] == "fail", checked["errors"])

    truck_id = "CA_90588b6bc671dd08"
    truck_segment = segments.get(truck_id, {})
    segment_ids = {
        str(row.get("source_sentence_id"))
        for row in truck_segment.get("segments") or []
    }
    add(
        "truck_src0034_src0035_exist",
        {"SRC0034", "SRC0035"} <= segment_ids,
        sorted(segment_ids & {"SRC0034", "SRC0035"}),
    )
    if truck_id in evidence:
        anchored = {
            source_id
            for unit in evidence[truck_id].get("evidence_units") or []
            for source_id in unit.get("source_sentence_ids") or []
        }
        add(
            "truck_src0034_src0035_in_evidence",
            {"SRC0034", "SRC0035"} <= anchored,
            sorted(anchored & {"SRC0034", "SRC0035"}),
        )
    report = {
        "dataset_version": DATASET_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(row["status"] == "pass" for row in tests) else "fail",
        "tests": tests,
    }
    atomic_write_json(output / "regression_smoke_test_report.json", report)
    return report


def _merge_all(
    output: Path,
    manifest: dict[str, Any],
    masters: dict[str, dict[str, Any]],
    translations: dict[str, dict[str, Any]],
    grounding: dict[str, dict[str, Any]],
    translation_verification: dict[str, dict[str, Any]],
    source_checks_map: dict[str, dict[str, Any]],
    translation_checks_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_ids = list(manifest["kr_case_ids"]) + list(manifest["ca_case_ids"])
    origin_by_id = (
        {case_id: "KR" for case_id in manifest["kr_case_ids"]}
        | {case_id: "CA" for case_id in manifest["ca_case_ids"]}
    )
    subtype_by_id = {
        row["case_id"]: row.get("case_subtype")
        for row in manifest.get("case_inputs") or []
    }
    rows: list[dict[str, Any]] = []
    for case_id in all_ids:
        origin = origin_by_id[case_id]
        master = masters.get(case_id, {})
        translated = translations.get(case_id, {})
        ground = grounding.get(case_id, {})
        trans_verify = translation_verification.get(case_id, {})
        deterministic_source = source_checks_map.get(case_id, {})
        deterministic_translation = translation_checks_map.get(case_id, {})
        translated_by_id = {
            str(unit.get("fact_id")): unit
            for unit in translated.get("translated_fact_units") or []
        }
        canonical: list[dict[str, Any]] = []
        for unit in master.get("fact_units") or []:
            translated_unit = translated_by_id.get(str(unit.get("fact_id")), {})
            source_text = str(unit.get("master_text") or "")
            target_text = str(translated_unit.get("translated_text") or "")
            canonical.append({
                "fact_id": unit.get("fact_id"),
                "epistemic_status": unit.get("epistemic_status"),
                "ko": source_text if origin == "KR" else target_text,
                "en": target_text if origin == "KR" else source_text,
                "relation_ids": unit.get("relation_ids") or [],
                "ko_generation_type": "source_neutralized" if origin == "KR" else "translated",
                "en_generation_type": "translated" if origin == "KR" else "source_neutralized",
            })
        source_status = str(master.get("neutralization_status") or "missing")
        relation_status = (
            "fail" if any(
                "relation" in error
                for error in deterministic_source.get("errors") or []
            ) else "pass" if master else "missing"
        )
        ground_status = str(ground.get("validated_verifier_status") or "missing")
        trans_status = str(translated.get("translation_status") or "missing")
        trans_relation_status = str(
            translated.get("translation_relation_status") or "missing"
        )
        trans_verify_status = str(
            trans_verify.get("validated_verifier_status") or "missing"
        )
        statuses = {
            source_status, relation_status, ground_status, trans_status,
            trans_relation_status, trans_verify_status,
        }
        automatic = (
            "missing" if "missing" in statuses
            else "fail" if "fail" in statuses
            else "warning" if "warning" in statuses
            else "pass"
        )
        rows.append({
            "dataset_version": DATASET_VERSION,
            "case_id": case_id,
            "case_origin": origin,
            "case_subtype": master.get("case_subtype", subtype_by_id.get(case_id)),
            "master_language": "ko" if origin == "KR" else "en",
            "translation_direction": "ko_to_en" if origin == "KR" else "en_to_ko",
            "neutral_fact_ko": (
                master.get("master_neutral_text", "") if origin == "KR"
                else translated.get("translated_neutral_text", "")
            ),
            "neutral_fact_en": (
                translated.get("translated_neutral_text", "") if origin == "KR"
                else master.get("master_neutral_text", "")
            ),
            "canonical_facts": canonical,
            "coverage_status": deterministic_source.get("coverage_status", "missing"),
            "model_insufficient_factual_detail": master.get(
                "model_insufficient_factual_detail"
            ),
            "deterministic_factual_sufficiency": deterministic_source.get(
                "deterministic_factual_sufficiency", "missing"
            ),
            "case_is_usable_for_translation": master.get(
                "case_is_usable_for_translation"
            ),
            "source_neutral_status": source_status,
            "relation_consistency_status": relation_status,
            "grounding_verifier_status": ground_status,
            "translation_status": trans_status,
            "translation_relation_status": trans_relation_status,
            "translation_verifier_status": trans_verify_status,
            "automatic_quality_status": automatic,
            "case_is_finally_usable": None,
            "human_qc_status": "",
            "human_qc_notes": "",
            "source_text_sha256": master.get("source_text_sha256", ""),
            "source_output_sha256": stable_hash(master) if master else "",
            "translation_output_sha256": stable_hash(translated) if translated else "",
        })
    atomic_write_jsonl(output / "neutral_pairs_all.jsonl", rows)
    atomic_write_jsonl(
        output / "neutral_pairs_pass.jsonl",
        [row for row in rows if row["automatic_quality_status"] == "pass"],
    )
    qc_stream = StringIO(newline="")
    writer = csv.DictWriter(qc_stream, fieldnames=HUMAN_QC_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        source = source_checks_map.get(row["case_id"], {})
        translation = translation_checks_map.get(row["case_id"], {})
        values = {field: "" for field in HUMAN_QC_FIELDS}
        values.update({
            "case_id": row["case_id"],
            "case_origin": row["case_origin"],
            "case_subtype": row.get("case_subtype") or "",
            "source_coverage_complete": source.get("coverage_status") == "complete",
            "core_event_preserved": source.get("event_present"),
            "harm_preserved": source.get("harm_present"),
            "causal_sequence_preserved": source.get("event_harm_sequence_present"),
            "legal_conclusion_removed": not bool(source.get("legal_terms")),
            "entity_placeholders_consistent": (
                translation.get("placeholder_identity_match")
            ),
            "numbers_units_preserved": all(
                item.get("match")
                for item in (translation.get("number_unit_normalization") or {}).values()
            ) if translation else "",
            "temporal_order_preserved": translation.get("temporal_order_preserved"),
            "jurisdiction_leakage_absent": (
                "jurisdiction_term_reintroduced"
                not in (translation.get("errors") or [])
            ) if translation else "",
        })
        writer.writerow(values)
    atomic_write_text(output / "human_qc_template.csv", qc_stream.getvalue())
    counts = {
        "manifest_case_count": len(all_ids),
        "source_segment_record_count": sum(
            len(by_case(output / f"source_segments_{suffix}.jsonl"))
            for suffix in ("kr", "ca")
        ),
        "evidence_record_count": sum(
            len(by_case(output / f"factual_evidence_{suffix}.jsonl"))
            for suffix in ("kr", "ca")
        ),
        "entity_graph_record_count": sum(
            len(by_case(output / f"entity_relation_graphs_{suffix}.jsonl"))
            for suffix in ("kr", "ca")
        ),
        "master_record_count": len(masters),
        "grounding_verification_count": len(grounding),
        "translation_record_count": len(translations),
        "translation_verification_count": len(translation_verification),
        "complete_pair_count": sum(
            case_id in masters and case_id in grounding and case_id in translations
            and case_id in translation_verification for case_id in all_ids
        ),
        "missing_master_count": sum(case_id not in masters for case_id in all_ids),
        "missing_translation_count": sum(
            case_id not in translations for case_id in all_ids
        ),
        "automatic_pass_count": sum(
            row["automatic_quality_status"] == "pass" for row in rows
        ),
        "automatic_warning_count": sum(
            row["automatic_quality_status"] == "warning" for row in rows
        ),
        "automatic_fail_count": sum(
            row["automatic_quality_status"] == "fail" for row in rows
        ),
        "human_qc_pending_count": sum(
            row["automatic_quality_status"] != "missing" for row in rows
        ),
        "quarantined_count": len(by_case(output / "quarantine.jsonl")),
    }
    return rows, counts


def _calibration_report(
    output: Path,
    selected: list[Any],
    evidence: dict[str, dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
    masters: dict[str, dict[str, Any]],
    grounding: dict[str, dict[str, Any]],
    translations: dict[str, dict[str, Any]],
    trans_verify: dict[str, dict[str, Any]],
    source_check_map: dict[str, dict[str, Any]],
    translation_check_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    quarantine = by_case(output / "quarantine.jsonl")
    for case in selected:
        case_id = case.case_id
        ev = evidence.get(case_id, {})
        graph = graphs.get(case_id, {})
        master = masters.get(case_id, {})
        ground = grounding.get(case_id, {})
        translated = translations.get(case_id, {})
        verifier = trans_verify.get(case_id, {})
        source = source_check_map.get(case_id, {})
        translation = translation_check_map.get(case_id, {})
        provenances = (
            list(ev.get("chunk_model_provenance") or [])
            + [
                graph.get("model_provenance") or {},
                master.get("model_provenance") or {},
                ground.get("model_provenance") or {},
                translated.get("model_provenance") or {},
                verifier.get("model_provenance") or {},
            ]
        )
        provenance = (
            "mock" if any(item.get("mock") for item in provenances)
            else "real" if provenances and all(item and not item.get("mock") for item in provenances)
            else "missing"
        )
        cases.append({
            "case_id": case_id,
            "case_origin": case.case_origin,
            "case_subtype": case.case_subtype,
            "source_coverage_ratio": (ev.get("source_coverage") or {}).get(
                "segment_coverage_ratio"
            ),
            "processed_source_ranges": (ev.get("source_coverage") or {}).get(
                "processed_source_ranges"
            ),
            "extraction_call_count": (ev.get("source_coverage") or {}).get(
                "extraction_call_count"
            ),
            "core_event": source.get("event_present"),
            "harm": source.get("harm_present"),
            "event_harm_sequence": source.get("event_harm_sequence_present"),
            "fact_unit_count": len(master.get("fact_units") or []),
            "material_relations": source.get("material_relation_ids") or [],
            "source_neutral_status": master.get("neutralization_status"),
            "model_insufficient_flag": master.get(
                "model_insufficient_factual_detail"
            ),
            "deterministic_sufficiency": source.get(
                "deterministic_factual_sufficiency"
            ),
            "number_normalization": translation.get("number_unit_normalization"),
            "unit_normalization": translation.get("number_unit_normalization"),
            "placeholder_identity_comparison": translation.get(
                "placeholder_comparison"
            ),
            "subject_object_comparison": translation.get("relation_comparison"),
            "negation_warning": translation.get("negation_warning"),
            "target_language_residue": translation.get("target_language_residue"),
            "model_grounding_status": ground.get("model_verifier_status"),
            "validated_grounding_status": ground.get("validated_verifier_status"),
            "model_translation_status": verifier.get("model_verifier_status"),
            "validated_translation_status": verifier.get(
                "validated_verifier_status"
            ),
            "automatic_status": next(
                (
                    row.get("automatic_quality_status")
                    for row in by_case(output / "neutral_pairs_all.jsonl").values()
                    if row.get("case_id") == case_id
                ),
                "missing",
            ),
            "quarantine_reason": quarantine.get(case_id, {}).get("failure_reasons", []),
            "provenance": provenance,
        })
    report = {
        "dataset_version": DATASET_VERSION,
        "batch_name": "calibration",
        "case_count": len(cases),
        "cases": cases,
    }
    atomic_write_json(output / "calibration_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _snapshot_prompts(output)
    input_hashes_before = {
        "KR": sha256_file(args.kr_input),
        "CA": sha256_file(args.ca_input),
    }
    cases, _, manifest = _write_input_artifacts(
        args.kr_input, args.ca_input, output
    )
    selected = _selected_records(cases, args)
    case_by_id = {case.case_id: case for case in cases}
    selected_ids = [case.case_id for case in selected]

    if not (output / "run_manifest.json").exists():
        atomic_write_json(
            output / "run_manifest.json",
            {
                **manifest,
                "phases": {},
                "run_history": [],
            },
        )
    _load_local_api_key(args.api_key_env)
    mode = "dry_run" if args.dry_run else "mock" if args.mock_response_dir else "api"

    segment_rows: list[dict[str, Any]] = []
    for case in selected:
        segments = segment_source(case.source_text)
        chunks, metadata = select_candidate_chunks(
            case,
            segments,
            max_input_tokens=args.max_input_tokens,
            overlap_sentences=args.chunk_overlap_sentences,
        )
        segment_rows.append(segmentation_record(case, segments, chunks, metadata))
    segment_map = {row["case_id"]: row for row in segment_rows}
    segment_all: dict[str, dict[str, Any]] = {}
    for origin, suffix in (("KR", "kr"), ("CA", "ca")):
        segment_all.update(
            _merge_output(
                output / f"source_segments_{suffix}.jsonl",
                segment_rows,
                origin=origin,
            )
        )
    coverage_rows = [coverage_record(row) for row in segment_rows]
    coverage_all = _merge_output(output / "source_coverage.jsonl", coverage_rows)
    coverage_failures = [
        row for row in coverage_rows if row["coverage_status"] != "complete"
    ]
    regression = _regression_report(
        output, set(case_by_id), segment_all, {}
    )
    if regression["status"] == "fail":
        raise RuntimeError("Deterministic regression preflight failed")
    if args.dry_run:
        append_errors(output / "api_errors.jsonl", [])
        write_usage(output / "api_usage.csv", [])
        append_run_history(
            output,
            {
                "batch_name": args.batch_name,
                "selected_case_ids": selected_ids,
                "mode": mode,
                "new_api_calls": 0,
                "cache_hits": 0,
                "deterministic_rechecks": 0,
            },
            {
                "phase_0_input_validation": {
                    "execution_status": "completed",
                    "record_counts": {"pass": 70, "warning": 0, "fail": 0, "missing": 0},
                },
                "phase_1_source_segmentation": {
                    "execution_status": "completed_subset",
                    "record_counts": {
                        "pass": len(coverage_rows) - len(coverage_failures),
                        "warning": 0,
                        "fail": len(coverage_failures),
                        "missing": 70 - len(coverage_rows),
                    },
                },
            },
        )
        print(
            f"v3 dry-run complete: inputs=70 selected={len(selected)} "
            f"coverage_failures={len(coverage_failures)} api_calls=0"
        )
        return 2 if coverage_failures else 0

    client = LLMClient(
        output_dir=output,
        model=args.model,
        base_url=args.base_url,
        max_retries=args.max_retries,
        mock_response_dir=args.mock_response_dir,
        api_key_env=args.api_key_env,
        bypass_cache=bool(
            args.retry_failed or args.retry_warnings or args.regenerate
        ),
    )
    all_errors: list[dict[str, Any]] = []
    usage_records: list[dict[str, Any]] = []

    evidence_paths = {
        "KR": output / "factual_evidence_kr.jsonl",
        "CA": output / "factual_evidence_ca.jsonl",
    }
    evidence_existing = {
        origin: by_case(path) for origin, path in evidence_paths.items()
    }
    evidence_new: list[dict[str, Any]] = []
    if not args.recheck_deterministic:
        evidence_todo = [
            case for case in selected
            if coverage_all[case.case_id]["coverage_status"] == "complete"
            and _should_run(
                evidence_existing[case.case_origin].get(case.case_id),
                "extraction_status",
                args,
            )
        ]

        def evidence_worker(case: Any) -> dict[str, Any]:
            return extract_evidence(case, segment_map[case.case_id], client, ROOT)

        evidence_new, errors = _run_stage(
            evidence_todo, evidence_worker, args.concurrency
        )
        all_errors.extend(errors)
        usage_records.extend(evidence_new)
    evidence_all: dict[str, dict[str, Any]] = {}
    for origin in ("KR", "CA"):
        evidence_all.update(
            _merge_output(evidence_paths[origin], evidence_new, origin=origin)
        )

    graph_paths = {
        "KR": output / "entity_relation_graphs_kr.jsonl",
        "CA": output / "entity_relation_graphs_ca.jsonl",
    }
    graph_existing = {origin: by_case(path) for origin, path in graph_paths.items()}
    graph_new: list[dict[str, Any]] = []
    if args.recheck_deterministic:
        for case in selected:
            existing = graph_existing[case.case_origin].get(case.case_id)
            evidence = evidence_all.get(case.case_id)
            if existing and evidence:
                graph_new.append(normalize_entity_relation_graph(
                    existing,
                    case_id=case.case_id,
                    case_origin=case.case_origin,
                    evidence=evidence,
                    provenance=existing.get("model_provenance") or {},
                    segment_record=segment_map[case.case_id],
                ))
    else:
        graph_todo = [
            case for case in selected
            if evidence_all.get(case.case_id, {}).get("extraction_status") == "pass"
            and _should_run(
                graph_existing[case.case_origin].get(case.case_id),
                "graph_status",
                args,
            )
        ]

        def graph_worker(case: Any) -> dict[str, Any]:
            return build_entity_relation_graph(
                case,
                evidence_all[case.case_id],
                client,
                ROOT,
                segment_map[case.case_id],
            )

        graph_new, errors = _run_stage(graph_todo, graph_worker, args.concurrency)
        all_errors.extend(errors)
        usage_records.extend(graph_new)
    graph_all: dict[str, dict[str, Any]] = {}
    for origin in ("KR", "CA"):
        graph_all.update(_merge_output(graph_paths[origin], graph_new, origin=origin))

    master_paths = {
        "KR": output / "source_neutral_kr.jsonl",
        "CA": output / "source_neutral_ca.jsonl",
    }
    master_existing = {
        origin: by_case(path) for origin, path in master_paths.items()
    }
    grounding_prior = by_case(
        output / "source_grounding_role_verification.jsonl"
    )
    source_check_existing = by_case(output / "deterministic_source_checks.jsonl")
    master_new: list[dict[str, Any]] = []
    source_check_new: list[dict[str, Any]] = []
    if args.recheck_deterministic:
        for case in selected:
            master = master_existing[case.case_origin].get(case.case_id)
            evidence = evidence_all.get(case.case_id)
            graph = graph_all.get(case.case_id)
            if master and evidence and graph:
                payload = {
                    "master_neutral_text": master.get("master_neutral_text") or "",
                    "fact_units": master.get("fact_units") or [],
                }
                checks = source_checks(payload, evidence, graph, case.source_language)
                updated = dict(master)
                updated["deterministic_checks"] = checks
                updated["neutralization_status"] = checks["status"]
                updated["deterministic_factual_sufficiency"] = checks[
                    "deterministic_factual_sufficiency"
                ]
                updated["case_is_usable_for_translation"] = checks["status"] in {
                    "pass", "warning"
                }
                master_new.append(updated)
                source_check_new.append({
                    "case_id": case.case_id,
                    "case_origin": case.case_origin,
                    **checks,
                })
    else:
        master_todo = [
            case for case in selected
            if graph_all.get(case.case_id, {}).get("graph_status") in {"pass", "warning"}
            and (
                _should_run(
                    master_existing[case.case_origin].get(case.case_id),
                    "neutralization_status",
                    args,
                )
                or (
                    args.retry_failed
                    and grounding_prior.get(case.case_id, {}).get(
                        "validated_verifier_status"
                    ) == "fail"
                )
            )
        ]

        def master_worker(case: Any) -> dict[str, Any]:
            master, check = neutralize(
                case,
                evidence_all[case.case_id],
                graph_all[case.case_id],
                client,
                ROOT,
            )
            return {"master": master, "check": check, "case_id": case.case_id}

        bundled, errors = _run_stage(master_todo, master_worker, args.concurrency)
        all_errors.extend(errors)
        master_new = [row["master"] for row in bundled]
        source_check_new = [row["check"] for row in bundled]
        usage_records.extend(master_new)
    master_all: dict[str, dict[str, Any]] = {}
    for origin in ("KR", "CA"):
        master_all.update(_merge_output(master_paths[origin], master_new, origin=origin))
    source_check_all = _merge_output(
        output / "deterministic_source_checks.jsonl", source_check_new
    )

    grounding_existing = by_case(
        output / "source_grounding_role_verification.jsonl"
    )
    grounding_new: list[dict[str, Any]] = []
    if args.recheck_deterministic:
        for case in selected:
            existing = grounding_existing.get(case.case_id)
            master = master_all.get(case.case_id)
            if existing and master:
                grounding_new.append(validate_grounding_payload(existing, master))
    else:
        grounding_todo = [
            case for case in selected
            if master_all.get(case.case_id, {}).get("neutralization_status")
            in {"pass", "warning"}
            and _should_run(
                grounding_existing.get(case.case_id),
                "validated_verifier_status",
                args,
            )
        ]

        def grounding_worker(case: Any) -> dict[str, Any]:
            return verify_grounding(
                master_all[case.case_id],
                evidence_all[case.case_id],
                graph_all[case.case_id],
                client,
                ROOT,
            )

        grounding_new, errors = _run_stage(
            grounding_todo, grounding_worker, args.concurrency
        )
        all_errors.extend(errors)
        usage_records.extend(grounding_new)
    grounding_all = _merge_output(
        output / "source_grounding_role_verification.jsonl", grounding_new
    )
    atomic_write_jsonl(
        output / "source_grounding_verification.jsonl",
        grounding_all.values(),
    )

    translation_paths = {
        "KR": output / "translated_pairs_kr.jsonl",
        "CA": output / "translated_pairs_ca.jsonl",
    }
    translation_existing = {
        origin: by_case(path) for origin, path in translation_paths.items()
    }
    translation_verifier_prior = by_case(output / "translation_verification.jsonl")
    translation_check_existing = by_case(
        output / "deterministic_translation_checks.jsonl"
    )
    translation_new: list[dict[str, Any]] = []
    translation_check_new: list[dict[str, Any]] = []
    if args.recheck_deterministic:
        for case in selected:
            master = master_all.get(case.case_id)
            record = translation_existing[case.case_origin].get(case.case_id)
            if master and record:
                target = "en" if case.case_origin == "KR" else "ko"
                checks = translation_checks(master, record, target)
                updated = dict(record)
                updated["deterministic_checks"] = checks
                updated["translation_status"] = checks["status"]
                updated["translation_relation_status"] = checks[
                    "translation_relation_status"
                ]
                translation_new.append(updated)
                translation_check_new.append({
                    "case_id": case.case_id,
                    "case_origin": case.case_origin,
                    **checks,
                })
    else:
        translation_todo = [
            case for case in selected
            if grounding_all.get(case.case_id, {}).get("validated_verifier_status")
            in {"pass", "warning"}
            and (
                _should_run(
                    translation_existing[case.case_origin].get(case.case_id),
                    "translation_status",
                    args,
                )
                or (
                    args.retry_failed
                    and translation_verifier_prior.get(case.case_id, {}).get(
                        "validated_verifier_status"
                    ) == "fail"
                )
            )
        ]

        def translation_worker(case: Any) -> dict[str, Any]:
            translated, check = translate(master_all[case.case_id], client, ROOT)
            return {
                "translated": translated,
                "check": check,
                "case_id": case.case_id,
            }

        bundled, errors = _run_stage(
            translation_todo, translation_worker, args.concurrency
        )
        all_errors.extend(errors)
        translation_new = [row["translated"] for row in bundled]
        translation_check_new = [row["check"] for row in bundled]
        usage_records.extend(translation_new)
    translation_all: dict[str, dict[str, Any]] = {}
    for origin in ("KR", "CA"):
        translation_all.update(
            _merge_output(
                translation_paths[origin], translation_new, origin=origin
            )
        )
    translation_check_all = _merge_output(
        output / "deterministic_translation_checks.jsonl",
        translation_check_new,
    )

    trans_verify_existing = by_case(output / "translation_verification.jsonl")
    trans_verify_new: list[dict[str, Any]] = []
    if args.recheck_deterministic:
        for case in selected:
            existing = trans_verify_existing.get(case.case_id)
            translated = translation_all.get(case.case_id)
            if existing and translated:
                trans_verify_new.append(
                    validate_translation_verifier_payload(existing, translated)
                )
    else:
        trans_verify_todo = [
            case for case in selected
            if translation_all.get(case.case_id, {}).get("translation_status")
            in {"pass", "warning"}
            and _should_run(
                trans_verify_existing.get(case.case_id),
                "validated_verifier_status",
                args,
            )
        ]

        def trans_verify_worker(case: Any) -> dict[str, Any]:
            return verify_translation(
                master_all[case.case_id],
                translation_all[case.case_id],
                client,
                ROOT,
            )

        trans_verify_new, errors = _run_stage(
            trans_verify_todo, trans_verify_worker, args.concurrency
        )
        all_errors.extend(errors)
        usage_records.extend(trans_verify_new)
    trans_verify_all = _merge_output(
        output / "translation_verification.jsonl", trans_verify_new
    )

    selected_failures: list[tuple[Any, list[str], str]] = []
    for case in selected:
        case_id = case.case_id
        reasons: list[str] = []
        failed_stage = ""
        if coverage_all.get(case_id, {}).get("coverage_status") != "complete":
            reasons.append("source_coverage_incomplete")
            failed_stage = "source_segmentation"
        elif evidence_all.get(case_id, {}).get("extraction_status") != "pass":
            reasons.extend(
                evidence_all.get(case_id, {}).get("validation_errors")
                or ["evidence_extraction_missing_or_failed"]
            )
            failed_stage = "evidence_extraction"
        elif graph_all.get(case_id, {}).get("graph_status") == "fail" or case_id not in graph_all:
            reasons.extend(
                graph_all.get(case_id, {}).get("validation_errors")
                or ["entity_relation_graph_missing_or_failed"]
            )
            failed_stage = "entity_relation_graph"
        elif master_all.get(case_id, {}).get("neutralization_status") == "fail" or case_id not in master_all:
            reasons.extend(
                source_check_all.get(case_id, {}).get("errors")
                or ["source_neutral_missing_or_failed"]
            )
            failed_stage = "source_neutralization"
        elif grounding_all.get(case_id, {}).get("validated_verifier_status") == "fail" or case_id not in grounding_all:
            reasons.extend(
                grounding_all.get(case_id, {}).get("deterministic_verifier_reasons")
                or ["source_grounding_role_verification_missing_or_failed"]
            )
            failed_stage = "source_grounding_role_verification"
        elif translation_all.get(case_id, {}).get("translation_status") == "fail" or case_id not in translation_all:
            reasons.extend(
                translation_check_all.get(case_id, {}).get("errors")
                or ["translation_missing_or_failed"]
            )
            failed_stage = "translation"
        elif trans_verify_all.get(case_id, {}).get("validated_verifier_status") == "fail" or case_id not in trans_verify_all:
            reasons.extend(
                trans_verify_all.get(case_id, {}).get("deterministic_verifier_reasons")
                or ["translation_verification_missing_or_failed"]
            )
            failed_stage = "translation_verification"
        if (
            grounding_all.get(case_id, {}).get("verifier_consistency_violation")
            or trans_verify_all.get(case_id, {}).get("verifier_consistency_violation")
        ):
            reasons.append("verifier_consistency_violation")
            failed_stage = failed_stage or "verifier_consistency"
        if reasons:
            selected_failures.append((case, sorted(set(reasons)), failed_stage))
    quarantines = [
        quarantine_record(
            master_all.get(case.case_id)
            or graph_all.get(case.case_id)
            or evidence_all.get(case.case_id)
            or {"case_id": case.case_id, "case_origin": case.case_origin},
            failed_stage,
            reasons,
            regeneration=failed_stage in {
                "evidence_extraction",
                "entity_relation_graph",
                "source_neutralization",
                "source_grounding_role_verification",
                "translation",
                "translation_verification",
            },
            deterministic_recheck=failed_stage in {
                "source_segmentation",
                "verifier_consistency",
            },
        )
        for case, reasons, failed_stage in selected_failures
    ]
    _update_current_quarantine(
        output / "quarantine.jsonl",
        selected_ids,
        quarantines,
    )
    append_errors(output / "api_errors.jsonl", all_errors)
    write_usage(output / "api_usage.csv", usage_records)
    regression = _regression_report(
        output, set(case_by_id), segment_all, evidence_all
    )
    merged_rows, merge_counts = _merge_all(
        output,
        manifest,
        master_all,
        translation_all,
        grounding_all,
        trans_verify_all,
        source_check_all,
        translation_check_all,
    )
    _calibration_report(
        output,
        selected,
        evidence_all,
        graph_all,
        master_all,
        grounding_all,
        translation_all,
        trans_verify_all,
        source_check_all,
        translation_check_all,
    )

    input_hashes_after = {
        "KR": sha256_file(args.kr_input),
        "CA": sha256_file(args.ca_input),
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("Input selection hash changed during run")
    selected_count = len(selected)
    hard_rate = len(selected_failures) / max(1, selected_count)
    api_rate = len(all_errors) / max(1, selected_count)
    batch = {
        "batch_name": args.batch_name,
        "selected_case_ids": selected_ids,
        "mode": mode,
        "new_api_calls": client.new_api_calls,
        "cache_hits": client.cache_hits,
        "deterministic_rechecks": selected_count if args.recheck_deterministic else 0,
        "completed_cases": sum(
            row.get("automatic_quality_status") != "missing"
            for row in merged_rows if row["case_id"] in set(selected_ids)
        ),
        "pass_count": sum(
            row.get("automatic_quality_status") == "pass"
            for row in merged_rows if row["case_id"] in set(selected_ids)
        ),
        "warning_count": sum(
            row.get("automatic_quality_status") == "warning"
            for row in merged_rows if row["case_id"] in set(selected_ids)
        ),
        "fail_count": len(selected_failures),
        "quarantined_count": len(quarantines),
        "missing_count": sum(
            row.get("automatic_quality_status") == "missing"
            for row in merged_rows if row["case_id"] in set(selected_ids)
        ),
        "hard_failure_rate": hard_rate,
        "api_failure_rate": api_rate,
    }
    phase_records = {
        "phase_0_input_validation": {
            "execution_status": "completed",
            "record_counts": {"pass": 70, "warning": 0, "fail": 0, "missing": 0},
        },
        "phase_1_source_segmentation": {
            "execution_status": "completed_subset",
            "record_counts": _counts(coverage_all, "coverage_status"),
        },
        "phase_2_evidence_extraction": {
            "execution_status": "completed_subset",
            "record_counts": _counts(evidence_all, "extraction_status"),
        },
        "phase_3_evidence_merge_anchoring": {
            "execution_status": "completed_subset",
            "record_counts": _counts(evidence_all, "extraction_status"),
        },
        "phase_4_entity_relation_graph": {
            "execution_status": "completed_subset",
            "record_counts": _counts(graph_all, "graph_status"),
        },
        "phase_5_source_neutralization": {
            "execution_status": "completed_subset",
            "record_counts": _counts(master_all, "neutralization_status"),
        },
        "phase_6_deterministic_source_checks": {
            "execution_status": "completed_subset",
            "record_counts": _counts(source_check_all, "status"),
        },
        "phase_7_source_grounding_role_verifier": {
            "execution_status": "completed_subset",
            "record_counts": _counts(grounding_all, "validated_verifier_status"),
        },
        "phase_8_translation": {
            "execution_status": "completed_subset",
            "record_counts": _counts(translation_all, "translation_status"),
        },
        "phase_9_deterministic_translation_checks": {
            "execution_status": "completed_subset",
            "record_counts": _counts(translation_check_all, "status"),
        },
        "phase_10_translation_verifier": {
            "execution_status": "completed_subset",
            "record_counts": _counts(trans_verify_all, "validated_verifier_status"),
        },
        "phase_11_merge_quarantine_human_qc": {
            "execution_status": (
                "completed" if merge_counts["complete_pair_count"] == 70
                else "completed_subset" if merge_counts["complete_pair_count"]
                else "partial"
            ),
            "record_counts": {
                "pass": merge_counts["automatic_pass_count"],
                "warning": merge_counts["automatic_warning_count"],
                "fail": merge_counts["automatic_fail_count"],
                "missing": 70 - (
                    merge_counts["automatic_pass_count"]
                    + merge_counts["automatic_warning_count"]
                    + merge_counts["automatic_fail_count"]
                ),
            },
            **merge_counts,
        },
    }
    append_run_history(output, batch, phase_records)
    safe_batch = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.batch_name)
    atomic_write_json(output / f"batch_{safe_batch}_quality.json", batch)
    blocked = (
        bool(selected_failures)
        or hard_rate > args.max_hard_failure_rate
        or api_rate > args.max_api_failure_rate
        or regression["status"] == "fail"
    )
    print(
        f"v3 batch complete: selected={selected_count} "
        f"api_calls={client.new_api_calls} cache_hits={client.cache_hits} "
        f"hard_fail={len(selected_failures)} blocked={blocked}"
    )
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
