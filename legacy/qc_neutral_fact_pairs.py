from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl
from pipeline.llm_client import LLMClient
from pipeline.paired_qc import (
    BACK_TRANSLATION_SCHEMA,
    BATCH_TARGETS,
    QC_DATASET_VERSION,
    SOURCE_QC_PROMPT_VERSION,
    SOURCE_QC_SCHEMA,
    SOURCE_REVIEW_FIELDS,
    TRANSLATION_QC_PROMPT_VERSION,
    TRANSLATION_QC_SCHEMA,
    TRANSLATION_REVIEW_FIELDS,
    adjudicate_source_qc,
    adjudicate_translation_qc,
    aggregate_source_qc,
    append_audit_jsonl,
    artifact_indexes,
    deterministic_source_precheck,
    deterministic_translation_precheck,
    finalize_pairs,
    import_source_reviews,
    import_translation_reviews,
    load_json,
    master_sha256,
    merge_jsonl,
    read_records,
    resolve_raw_paths,
    sha256_text,
    source_review_rows,
    summarize,
    translation_readiness,
    translation_review_rows,
    translation_sha256,
    utc_now,
    validate_qc_inputs,
    validated_master_record,
    write_csv,
)
from pipeline.stage2_runtime import append_run_history, by_case
from pipeline.stage2_v3_pipeline import (
    configure_generation_profile,
    stable_hash,
    translate,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_GENERATION_OUTPUT = Path("outputs/neutral/stage2-neutral-35x35-v4")
DEFAULT_OUTPUT = Path("outputs/neutral/stage2-paired-qc-v1")
PROMPTS = [
    "qc_source_neutral_ko_v1_en.txt",
    "qc_source_neutral_en_v1_en.txt",
    "qc_translation_ko_to_en_v1_en.txt",
    "qc_translation_en_to_ko_v1_en.txt",
    "qc_back_translate_en_to_ko_v1_en.txt",
    "qc_back_translate_ko_to_en_v1_en.txt",
    "qc_disagreement_adjudication_v1_en.txt",
]
BATCH_CONFIGS = {
    "stage-a": Path("configs/stage2_calibration_a_6.txt"),
    "stage-b": Path("configs/stage2_stage_b_cumulative_20.txt"),
    "stage-c": Path("configs/stage2_stage_c_cumulative_40.txt"),
    "stage-d": Path("configs/stage2_stage_d_cumulative_70.txt"),
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Independent source-neutral and translation QC with mandatory "
            "human-review gates."
        )
    )
    value.add_argument(
        "--generation-output-dir", type=Path, default=DEFAULT_GENERATION_OUTPUT,
    )
    value.add_argument("--raw-kr-input", type=Path)
    value.add_argument("--raw-ca-input", type=Path)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument(
        "--batch-name", choices=sorted(BATCH_TARGETS), default="stage-a",
    )
    value.add_argument("--case-id", action="append")
    value.add_argument("--case-id-file", type=Path)
    value.add_argument("--source-qc-only", action="store_true")
    value.add_argument("--regenerate-stale-translations", action="store_true")
    value.add_argument("--translation-qc-only", action="store_true")
    value.add_argument("--export-human-source-review", action="store_true")
    value.add_argument("--human-source-review-output", type=Path)
    value.add_argument(
        "--import-human-source-review", type=Path, nargs="?",
        const=DEFAULT_OUTPUT / "human_source_review.csv",
    )
    value.add_argument("--export-human-translation-review", action="store_true")
    value.add_argument("--human-translation-review-output", type=Path)
    value.add_argument(
        "--import-human-translation-review", type=Path, nargs="?",
        const=DEFAULT_OUTPUT / "human_translation_review.csv",
    )
    value.add_argument("--run-back-translation", action="store_true")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--retry-failed", action="store_true")
    value.add_argument("--retry-warnings", action="store_true")
    value.add_argument("--regenerate", action="store_true")
    value.add_argument("--recheck-deterministic", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--mock-response-dir", type=Path)
    value.add_argument("--model", default="gpt-5.6-luna")
    value.add_argument("--base-url", default="https://gw.letsur.ai/v1")
    value.add_argument("--concurrency", type=int, default=2)
    value.add_argument("--max-retries", type=int, default=5)
    return value


def _load_local_api_key() -> None:
    if os.environ.get("LETSUR_API_KEY"):
        return
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        if key.strip() == "LETSUR_API_KEY":
            os.environ["LETSUR_API_KEY"] = raw_value.strip().strip("\"'")
            return


def _read_case_ids(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _selected_ids(args: argparse.Namespace, generation_dir: Path) -> list[str]:
    if args.case_id:
        return list(dict.fromkeys(args.case_id))
    if args.case_id_file:
        return _read_case_ids(args.case_id_file)
    configured = ROOT / BATCH_CONFIGS[args.batch_name]
    if configured.is_file():
        return _read_case_ids(configured)
    manifest = load_json(generation_dir / "input_manifest.json")
    ids = [
        *[str(value) for value in manifest.get("kr_case_ids") or []],
        *[str(value) for value in manifest.get("ca_case_ids") or []],
    ]
    limits = BATCH_TARGETS[args.batch_name]
    selected: list[str] = []
    for origin in ("KR", "CA"):
        selected.extend(
            [case_id for case_id in ids if case_id.startswith(f"{origin}_")][
                :limits[origin]
            ]
        )
    return selected


def _snapshot_prompts(output_dir: Path) -> None:
    for name in PROMPTS:
        source = ROOT / "prompts" / name
        destination = output_dir / "prompts" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        content = source.read_text(encoding="utf-8")
        if destination.exists():
            if destination.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"Immutable QC prompt snapshot differs: {destination}"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")


def _write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if not path.exists():
        atomic_write_json(path, manifest)
        return
    existing = load_json(path)
    compare_existing = {key: value for key, value in existing.items() if key != "created_at"}
    compare_new = {key: value for key, value in manifest.items() if key != "created_at"}
    if compare_existing != compare_new:
        raise FileExistsError(
            "Immutable QC input manifest differs; use a separate output directory."
        )


def _model_source_failure(case: dict[str, Any], reason: str) -> dict[str, Any]:
    finding = {
        "fact_id": "", "master_span": "", "source_sentence_ids": [],
        "source_excerpt": "", "error_type": "source_qc_parsing_failure",
        "severity": "hard", "explanation": reason,
    }
    return {
        "case_id": case["case_id"], "source_language": case["source_language"],
        "source_qc": {
            "factual_support_status": "fail",
            "unsupported_facts": [finding], "overstated_facts": [],
            "missing_material_facts": [], "entity_role_errors": [],
            "epistemic_status_errors": [], "legal_conclusion_leakage": [],
            "jurisdiction_leakage": [], "source_copy_risk": "low",
            "recognition_risk_reasons": ["source_qc_parsing_failure"],
            "factual_sufficiency": "insufficient",
            "model_source_qc_status": "fail", "model_confidence": "low",
            "recommended_human_action": "review_only",
        },
        "model_provenance": {
            "prompt_version": SOURCE_QC_PROMPT_VERSION[case["source_language"]],
            "error": reason, "mock": False,
        },
    }


def _model_translation_failure(case: dict[str, Any], reason: str) -> dict[str, Any]:
    finding = {
        "fact_id": "", "master_span": "", "translation_span": "",
        "error_type": "translation_qc_parsing_failure", "severity": "hard",
        "explanation": reason,
    }
    arrays = {
        field: [] for field in (
            "added_information", "omitted_information",
            "subject_object_shifts", "entity_role_shifts",
            "epistemic_status_shifts", "polarity_shifts",
            "temporal_relation_shifts", "causal_direction_shifts",
            "spatial_direction_shifts", "number_unit_shifts",
            "legal_terms_reintroduced", "jurisdiction_signals_reintroduced",
            "fact_structure_errors",
        )
    }
    arrays["fact_structure_errors"] = [finding]
    return {
        "case_id": case["case_id"],
        "translation_direction": case["translation_direction"],
        "translation_qc": {
            "semantic_equivalence": "fail", **arrays,
            "model_translation_qc_status": "fail", "model_confidence": "low",
            "recommended_human_action": "review_only",
        },
        "model_provenance": {
            "prompt_version": TRANSLATION_QC_PROMPT_VERSION[
                case["translation_direction"]
            ],
            "error": reason, "mock": False,
        },
    }


def _should_call(
    existing: dict[str, Any] | None, status_field: str,
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


def _source_payload(
    case: dict[str, Any], chunk: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    source_ids = set(chunk.get("source_sentence_ids") or [])
    segments = [
        segment for segment in case["segments"].get("segments") or []
        if segment.get("source_sentence_id") in source_ids
    ]
    evidence = [
        unit for unit in case["evidence"].get("evidence_units") or []
        if not source_ids or source_ids & set(unit.get("source_sentence_ids") or [])
    ]
    return {
        "case_id": case["case_id"],
        "source_language": case["source_language"],
        "instruction_language": "English",
        "source_chunk": {
            "chunk_id": chunk.get("chunk_id"),
            "source_sentence_ids": chunk.get("source_sentence_ids") or [],
            "ordered_source_text": chunk.get("text") or "",
        },
        "source_segments": segments,
        "source_coverage": case["master"].get("source_coverage") or {},
        "validated_evidence_units": evidence,
        "entity_map": case["graph"].get("entities") or [],
        "material_relation_graph": [
            relation for relation in case["graph"].get("relations") or []
            if relation.get("material_relation")
        ],
        "master_fact_units": case["master"].get("fact_units") or [],
        "master_neutral_text": case["master"].get("master_neutral_text") or "",
        "deterministic_source_flags": deterministic,
        "previous_grounding_verifier": case.get("source_verifier") or {},
        "scope_note": (
            "This is one ordered source chunk. Report only errors supported by "
            "this chunk; do not assume absent facts are missing unless full "
            "coverage metadata and all chunks establish the omission."
        ),
    }


def _run_source_qc(
    args: argparse.Namespace, cases: dict[str, dict[str, Any]],
    output_dir: Path, client: LLMClient,
) -> list[dict[str, Any]]:
    existing_models = {
        **by_case(output_dir / "automatic_source_qc_kr.jsonl"),
        **by_case(output_dir / "automatic_source_qc_ca.jsonl"),
    }
    existing_validated = by_case(output_dir / "validated_source_qc.jsonl")
    automatic: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    deterministic_records: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        source_ready = _source_qc_ready(case)
        if not source_ready:
            reason = "missing_required_source_qc_artifact"
            model = _model_source_failure(case, reason)
            deterministic = {
                "case_id": case_id, "deterministic_source_qc_status": "fail",
                "errors": [reason], "warnings": [], "source_coverage": {},
            }
        else:
            deterministic = deterministic_source_precheck(case)
            existing = existing_models.get(case_id)
            if not _should_call(
                existing, "model_source_qc_status", args,
            ) and not args.recheck_deterministic:
                model = existing
            elif args.recheck_deterministic and existing and not (
                args.regenerate or args.retry_failed or args.retry_warnings
            ):
                model = existing
            else:
                prompt_version = SOURCE_QC_PROMPT_VERSION[
                    case["source_language"]
                ]
                prompt = (ROOT / "prompts" / f"{prompt_version}.txt").read_text(
                    encoding="utf-8"
                )
                chunks = case["segments"].get("candidate_chunks") or [{
                    "chunk_id": "FULL",
                    "source_sentence_ids": [
                        item.get("source_sentence_id")
                        for item in case["segments"].get("segments") or []
                    ],
                    "text": "\n".join(
                        f"<{item.get('source_sentence_id')}>"
                        f"{item.get('text')}</{item.get('source_sentence_id')}>"
                        for item in case["segments"].get("segments") or []
                    ),
                }]
                payloads: list[dict[str, Any]] = []
                provenances: list[dict[str, Any]] = []
                try:
                    for chunk in chunks:
                        result = client.call(
                            case_id=case_id, stage="qc_source_neutral",
                            system_prompt=prompt,
                            user_payload=_source_payload(
                                case, chunk, deterministic,
                            ),
                            schema=SOURCE_QC_SCHEMA,
                            required_fields=(
                                "case_id", "source_language", "source_qc",
                            ),
                            prompt_version=prompt_version,
                            schema_version="stage2-paired-source-qc-v1",
                            context_hashes={
                                "source_master_sha256": master_sha256(
                                    case["master"]
                                ),
                                "source_chunk_sha256": sha256_text(
                                    str(chunk.get("text") or "")
                                ),
                            },
                        )
                        payloads.append(result.payload)
                        provenances.append(result.provenance)
                    model = aggregate_source_qc(case, payloads, provenances)
                except Exception as exc:
                    model = _model_source_failure(
                        case, f"{type(exc).__name__}: {exc}",
                    )
        deterministic_records.append(deterministic)
        automatic.append(model)
        record = adjudicate_source_qc(case, deterministic, model)
        validated.append(record)
        if record["validated_source_qc_status"] == "fail":
            quarantine.append({
                "case_id": case_id, "batch_name": args.batch_name,
                "failed_phase": "source_qc",
                "failure_reasons": record["validation_reasons"],
                "automatic_status": "fail", "human_status": "pending",
                "requires_master_edit": True,
                "requires_translation_regeneration": False,
                "requires_human_review": True,
                "raw_response_path": (
                    (model.get("model_provenance") or {})
                    .get("raw_response_paths") or [""]
                )[0],
            })
    merge_jsonl(output_dir / "deterministic_source_qc_prechecks.jsonl",
                deterministic_records)
    merge_jsonl(output_dir / "automatic_source_qc_kr.jsonl", [
        row for row in automatic if cases[str(row["case_id"])]["case_origin"] == "KR"
    ])
    merge_jsonl(output_dir / "automatic_source_qc_ca.jsonl", [
        row for row in automatic if cases[str(row["case_id"])]["case_origin"] == "CA"
    ])
    merge_jsonl(output_dir / "validated_source_qc.jsonl", validated)
    all_validated = list(by_case(output_dir / "validated_source_qc.jsonl").values())
    for status in ("pass", "warning", "fail"):
        atomic_write_jsonl(
            output_dir / f"source_qc_{status}.jsonl",
            [
                row for row in all_validated
                if row.get("validated_source_qc_status") == status
            ],
        )
    merge_jsonl(output_dir / "quarantine.jsonl", quarantine)
    if quarantine:
        append_audit_jsonl(output_dir / "quarantine_history.jsonl", quarantine)
    append_run_history(output_dir, {
        "batch_name": args.batch_name, "qc_phase": "source_qc",
        "selected_case_ids": list(cases),
        "new_api_calls": client.new_api_calls,
        "cache_hits": client.cache_hits,
        "deterministic_rechecks": (
            len(cases) if args.recheck_deterministic else 0
        ),
    }, {
        "source_qc": {
            "execution_status": "completed",
            "record_counts": {
                status: sum(
                    row["validated_source_qc_status"] == status
                    for row in validated
                )
                for status in ("pass", "warning", "fail")
            },
        },
    })
    return validated


def _source_qc_ready(case: dict[str, Any]) -> bool:
    # The prior generation verifier is useful context but not a prerequisite
    # for the independent paired-QC verifier.
    return all(
        case.get(field)
        for field in (
            "segments", "evidence", "graph", "master", "source_checks",
        )
    ) and bool(case.get("raw_source"))


def _run_translation_qc(
    args: argparse.Namespace, ready: list[dict[str, Any]],
    output_dir: Path, client: LLMClient,
) -> list[dict[str, Any]]:
    existing_models = {
        **by_case(output_dir / "automatic_translation_qc_kr.jsonl"),
        **by_case(output_dir / "automatic_translation_qc_ca.jsonl"),
    }
    automatic: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    deterministic_records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for item in ready:
        case = item["case"]
        master = item["master"]
        translation = item["translation"]
        deterministic = deterministic_translation_precheck(
            case, master, translation,
        )
        deterministic_records.append(deterministic)
        existing = existing_models.get(case["case_id"])
        if existing and (
            str(existing.get("source_master_sha256") or "")
            != master_sha256(master)
            or str(existing.get("translation_sha256") or "")
            != translation_sha256(translation)
        ):
            existing = None
        if not _should_call(
            existing, "model_translation_qc_status", args,
        ) and not args.recheck_deterministic:
            model = existing
        elif args.recheck_deterministic and existing and not (
            args.regenerate or args.retry_failed or args.retry_warnings
        ):
            model = existing
        else:
            prompt_version = TRANSLATION_QC_PROMPT_VERSION[
                case["translation_direction"]
            ]
            prompt = (ROOT / "prompts" / f"{prompt_version}.txt").read_text(
                encoding="utf-8"
            )
            # Deliberately excludes raw source and source evidence.
            payload = {
                "case_id": case["case_id"],
                "translation_direction": case["translation_direction"],
                "instruction_language": "English",
                "human_validated_master": {
                    "master_neutral_text": master.get("master_neutral_text"),
                    "fact_units": master.get("fact_units") or [],
                },
                "translation": {
                    "translated_neutral_text": translation.get(
                        "translated_neutral_text"
                    ),
                    "translated_fact_units": translation.get(
                        "translated_fact_units"
                    ) or [],
                },
                "entity_role_graph": {
                    "entities": case["graph"].get("entities") or [],
                    "material_relations": [
                        relation
                        for relation in case["graph"].get("relations") or []
                        if relation.get("material_relation")
                    ],
                },
                "deterministic_translation_flags": deterministic,
            }
            try:
                result = client.call(
                    case_id=case["case_id"], stage="qc_translation",
                    system_prompt=prompt, user_payload=payload,
                    schema=TRANSLATION_QC_SCHEMA,
                    required_fields=(
                        "case_id", "translation_direction", "translation_qc",
                    ),
                    prompt_version=prompt_version,
                    schema_version="stage2-paired-translation-qc-v1",
                    context_hashes={
                        "source_master_sha256": master_sha256(master),
                        "translation_sha256": stable_hash(
                            translation.get("translated_fact_units") or []
                        ),
                    },
                )
                model = {
                    **result.payload, "model_provenance": result.provenance,
                }
                forbidden = {
                    "edited_translation", "replacement_text", "translation_text",
                } & set(model.get("translation_qc") or {})
                if forbidden:
                    raise ValueError("GPT QC attempted translation repair")
            except Exception as exc:
                model = _model_translation_failure(
                    case, f"{type(exc).__name__}: {exc}",
                )
        automatic.append(model)
        record = adjudicate_translation_qc(
            case, master, translation, deterministic, model,
        )
        validated.append(record)
        if record["validated_translation_qc_status"] == "fail":
            quarantine.append({
                "case_id": case["case_id"], "batch_name": args.batch_name,
                "failed_phase": "translation_qc",
                "failure_reasons": record["validation_reasons"],
                "automatic_status": "fail", "human_status": "pending",
                "requires_master_edit": False,
                "requires_translation_regeneration": True,
                "requires_human_review": True,
                "raw_response_path": (
                    model.get("model_provenance") or {}
                ).get("raw_response_path", ""),
            })
    merge_jsonl(output_dir / "deterministic_translation_qc_prechecks.jsonl",
                deterministic_records)
    merge_jsonl(output_dir / "automatic_translation_qc_kr.jsonl", [
        row for row in automatic
        if str(row.get("translation_direction")) == "ko_to_en"
    ])
    merge_jsonl(output_dir / "automatic_translation_qc_ca.jsonl", [
        row for row in automatic
        if str(row.get("translation_direction")) == "en_to_ko"
    ])
    merge_jsonl(output_dir / "validated_translation_qc.jsonl", validated)
    all_validated = list(
        by_case(output_dir / "validated_translation_qc.jsonl").values()
    )
    for status in ("pass", "warning", "fail"):
        atomic_write_jsonl(
            output_dir / f"translation_qc_{status}.jsonl",
            [
                row for row in all_validated
                if row.get("validated_translation_qc_status") == status
            ],
        )
    merge_jsonl(output_dir / "quarantine.jsonl", quarantine)
    if quarantine:
        append_audit_jsonl(output_dir / "quarantine_history.jsonl", quarantine)
    append_run_history(output_dir, {
        "batch_name": args.batch_name, "qc_phase": "translation_qc",
        "selected_case_ids": [
            item["case"]["case_id"] for item in ready
        ],
        "new_api_calls": client.new_api_calls,
        "cache_hits": client.cache_hits,
        "deterministic_rechecks": (
            len(ready) if args.recheck_deterministic else 0
        ),
    }, {
        "translation_qc": {
            "execution_status": "completed",
            "record_counts": {
                status: sum(
                    row["validated_translation_qc_status"] == status
                    for row in validated
                )
                for status in ("pass", "warning", "fail")
            },
        },
    })
    return validated


def _overlay_regenerated_translations(
    cases: dict[str, dict[str, Any]],
    human_masters: dict[str, dict[str, Any]],
    output_dir: Path,
) -> int:
    applied = 0
    for case_id, translated in by_case(
        output_dir / "regenerated_translations.jsonl"
    ).items():
        case = cases.get(case_id)
        review = human_masters.get(case_id)
        if not case or not review:
            continue
        expected_hash = str(
            review.get("human_validated_master_sha256") or ""
        )
        if (
            not expected_hash
            or str(translated.get("parent_master_sha256") or "")
            != expected_hash
        ):
            continue
        case["translation"] = translated
        case["translation_checks"] = (
            translated.get("deterministic_checks") or {}
        )
        applied += 1
    return applied


def _unreviewed_records(
    records: Any,
    cases: dict[str, dict[str, Any]],
    carried_reviews: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record for record in records
        if str(record.get("case_id") or "") in cases
        and str(record.get("case_id") or "") not in carried_reviews
    ]


def _run_translation_regeneration(
    args: argparse.Namespace,
    cases: dict[str, dict[str, Any]],
    human_masters: dict[str, dict[str, Any]],
    output_dir: Path,
    client: LLMClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configure_generation_profile("english-v1")
    existing = by_case(output_dir / "regenerated_translations.jsonl")
    generated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    eligible = [
        case_id for case_id in cases
        if (human_masters.get(case_id) or {}).get("human_source_status")
        in {"accepted", "accepted_with_edits"}
    ]
    for case_id in eligible:
        case = cases[case_id]
        review = human_masters[case_id]
        master = validated_master_record(case, review)
        parent_hash = str(review["human_validated_master_sha256"])
        previous = existing.get(case_id)
        previous_is_current = bool(
            previous
            and str(previous.get("parent_master_sha256") or "") == parent_hash
        )
        should_retry = bool(
            previous
            and previous.get("translation_status") == "fail"
            and args.retry_failed
        )
        if previous_is_current and not args.regenerate and not should_retry:
            generated.append(previous)
            continue
        try:
            translated, checks = translate(master, client, ROOT)
            record = {
                **translated,
                "qc_dataset_version": QC_DATASET_VERSION,
                "translation_generation_type": (
                    "human_validated_master_regeneration"
                ),
                "parent_master_sha256": parent_hash,
                "human_validated_master_sha256": parent_hash,
                "translation_sha256": translation_sha256(translated),
                "deterministic_checks": checks,
            }
            generated.append(record)
        except Exception as exc:
            failures.append({
                "case_id": case_id,
                "batch_name": args.batch_name,
                "failed_phase": "translation_regeneration",
                "failure_reasons": [f"{type(exc).__name__}: {exc}"],
                "automatic_status": "fail",
                "human_status": "pending",
                "requires_master_edit": False,
                "requires_translation_regeneration": True,
                "requires_human_review": True,
                "raw_response_path": "",
            })
    merge_jsonl(output_dir / "regenerated_translations.jsonl", generated)
    unresolved = by_case(
        output_dir / "translations_requiring_regeneration.jsonl"
    )
    for record in generated:
        case_id = str(record["case_id"])
        review = human_masters.get(case_id) or {}
        if (
            str(record.get("parent_master_sha256") or "")
            == str(review.get("human_validated_master_sha256") or "")
        ):
            unresolved.pop(case_id, None)
    for failure in failures:
        unresolved[str(failure["case_id"])] = {
            "case_id": failure["case_id"],
            "translation_requires_regeneration": True,
            "reason": failure["failure_reasons"][0],
        }
    atomic_write_jsonl(
        output_dir / "translations_requiring_regeneration.jsonl",
        unresolved.values(),
    )
    if failures:
        merge_jsonl(output_dir / "quarantine.jsonl", failures)
        append_audit_jsonl(
            output_dir / "quarantine_history.jsonl", failures,
        )
    append_run_history(output_dir, {
        "batch_name": args.batch_name,
        "qc_phase": "translation_regeneration",
        "selected_case_ids": eligible,
        "new_api_calls": client.new_api_calls,
        "cache_hits": client.cache_hits,
        "deterministic_rechecks": 0,
    }, {
        "translation_regeneration": {
            "execution_status": (
                "completed" if not failures else "completed_with_failures"
            ),
            "record_counts": {
                "generated": len(generated),
                "failed": len(failures),
            },
        },
    })
    return generated, failures


def _run_back_translation(
    args: argparse.Namespace, ready_by_case: dict[str, dict[str, Any]],
    translation_qc: dict[str, dict[str, Any]], output_dir: Path,
    client: LLMClient,
) -> None:
    outputs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for case_id, item in ready_by_case.items():
        record = translation_qc.get(case_id) or {}
        reasons = record.get("validation_reasons") or []
        qc = record.get("translation_qc") or {}
        eligible = (
            record.get("validated_translation_qc_status") == "warning" or
            "deterministic_and_gpt_disagreement" in reasons or
            bool(qc.get("subject_object_shifts")) or
            bool(qc.get("epistemic_status_shifts")) or
            bool(qc.get("entity_role_shifts"))
        )
        if not eligible:
            continue
        case = item["case"]
        direction = case["translation_direction"]
        prompt_name = (
            "qc_back_translate_en_to_ko_v1_en.txt"
            if direction == "ko_to_en"
            else "qc_back_translate_ko_to_en_v1_en.txt"
        )
        prompt = (ROOT / "prompts" / prompt_name).read_text(encoding="utf-8")
        result = client.call(
            case_id=case_id, stage="qc_back_translation",
            system_prompt=prompt,
            user_payload={
                "case_id": case_id,
                "translated_fact_units": item["translation"].get(
                    "translated_fact_units"
                ) or [],
                "diagnostic_only": True,
            },
            schema=BACK_TRANSLATION_SCHEMA,
            required_fields=(
                "case_id", "back_translated_fact_units", "diagnostic_notes",
            ),
            prompt_version=prompt_name.removesuffix(".txt"),
            schema_version="stage2-paired-back-translation-v1",
            context_hashes={
                "translation_sha256": stable_hash(
                    item["translation"].get("translated_fact_units") or []
                )
            },
        )
        output = {
            **result.payload, "model_provenance": result.provenance,
            "diagnostic_only": True, "cannot_replace_final_text": True,
        }
        outputs.append(output)
        diagnostics.append({
            "case_id": case_id,
            "trigger_reasons": reasons,
            "diagnostic_notes": result.payload.get("diagnostic_notes") or [],
            "master_fact_ids": [
                unit.get("fact_id")
                for unit in item["master"].get("fact_units") or []
            ],
            "back_translation_fact_ids": [
                unit.get("fact_id")
                for unit in result.payload.get("back_translated_fact_units") or []
            ],
        })
    merge_jsonl(output_dir / "optional_back_translations.jsonl", outputs)
    merge_jsonl(output_dir / "back_translation_diagnostics.jsonl", diagnostics)


def _quarantine_input_failures(
    report: dict[str, Any], output_dir: Path, batch_name: str,
) -> None:
    rows = [
        {
            "case_id": item["case_id"], "batch_name": batch_name,
            "failed_phase": "input_validation",
            "failure_reasons": item["validation_errors"],
            "automatic_status": "fail", "human_status": "pending",
            "requires_master_edit": False,
            "requires_translation_regeneration": False,
            "requires_human_review": True,
            "raw_response_path": "",
        }
        for item in report["case_validation"] if item["status"] == "fail"
    ]
    if rows:
        merge_jsonl(output_dir / "quarantine.jsonl", rows)
        append_audit_jsonl(output_dir / "quarantine_history.jsonl", rows)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    generation_dir = args.generation_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_name != "stage-a":
        prior = {
            "stage-b": "stage_a", "stage-c": "stage_b", "stage-d": "stage_c",
        }[args.batch_name]
        prior_path = output_dir / "batch_reports" / f"{prior}.json"
        prior_report = load_json(prior_path) if prior_path.exists() else {}
        unlocked_field = {
            "stage-b": "stage_b_unlocked",
            "stage-c": "next_batch_unlocked",
            "stage-d": "next_batch_unlocked",
        }[args.batch_name]
        if not prior_report.get(unlocked_field):
            print(
                f"{args.batch_name} is locked until the prior batch has complete "
                "human source and translation acceptance with no hard failure, "
                "pending review, or stale translation.",
                file=sys.stderr,
            )
            return 4
    _snapshot_prompts(output_dir)
    selected = _selected_ids(args, generation_dir)
    raw_kr, raw_ca = resolve_raw_paths(
        generation_dir, args.raw_kr_input, args.raw_ca_input,
    )
    report, cases, immutable = validate_qc_inputs(
        generation_dir, raw_kr, raw_ca, selected,
    )
    atomic_write_json(output_dir / "input_validation_report.json", report)
    version_slug = re.sub(
        r"[^a-zA-Z0-9_.-]+", "_",
        str(report.get("generation_dataset_version") or "unknown"),
    )
    batch_manifest = (
        output_dir / "input_manifests" /
        f"{args.batch_name.replace('-', '_')}__{version_slug}.json"
    )
    _write_immutable_manifest(batch_manifest, immutable)
    atomic_write_json(output_dir / "input_manifest.json", {
        **immutable,
        "immutable_snapshot_path": str(batch_manifest),
    })
    run_manifest_path = output_dir / "run_manifest.json"
    if not run_manifest_path.exists():
        atomic_write_json(run_manifest_path, {
            "qc_dataset_version": QC_DATASET_VERSION,
            "generation_dataset_version": report["generation_dataset_version"],
            "phases": {}, "run_history": [],
        })
    else:
        run_manifest = load_json(run_manifest_path)
        previous_version = str(
            run_manifest.get("generation_dataset_version") or ""
        )
        current_version = str(report["generation_dataset_version"] or "")
        if previous_version != current_version:
            completed_nonvalidation = {
                key for key in (run_manifest.get("phases") or {})
                if key != "input_validation"
            }
            if completed_nonvalidation:
                raise ValueError(
                    "QC output already contains non-validation phases for a "
                    "different generation version; use a separate QC output."
                )
            history = list(run_manifest.get("generation_target_history") or [])
            if previous_version:
                history.append({
                    "generation_dataset_version": previous_version,
                    "superseded_at": utc_now(),
                    "reason": "failed_preflight_replaced_by_consistent_version",
                })
            atomic_write_json(run_manifest_path, {
                **run_manifest,
                "generation_dataset_version": current_version,
                "generation_target_history": history,
            })
    _quarantine_input_failures(report, output_dir, args.batch_name)
    append_run_history(output_dir, {
        "batch_name": args.batch_name, "qc_phase": "input_validation",
        "selected_case_ids": selected, "new_api_calls": 0, "cache_hits": 0,
        "deterministic_rechecks": 0,
    }, {
        "input_validation": {
            "execution_status": "completed",
            "quality_status": report["validation_status"],
            "generation_consistency_status": (
                report["generation_consistency"]["status"]
            ),
        },
    })
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["validation_status"] == "pass" else 2
    if report["generation_consistency"]["status"] != "pass":
        print(
            "QC stopped before Stage A: generation version is not a single "
            "English-instruction prompt/model/schema/policy version.",
            file=sys.stderr,
        )
        return 2

    client = LLMClient(
        output_dir=output_dir, model=args.model, base_url=args.base_url,
        max_retries=args.max_retries,
        mock_response_dir=args.mock_response_dir,
        bypass_cache=args.regenerate,
    )
    action_is_import = bool(
        args.import_human_source_review or
        args.import_human_translation_review
    )
    if not args.mock_response_dir and not action_is_import:
        _load_local_api_key()

    source_qc = by_case(output_dir / "validated_source_qc.jsonl")
    if args.import_human_source_review:
        import_source_reviews(
            args.import_human_source_review.resolve(), cases, output_dir,
        )
        print(
            "Human source reviews imported. Translation readiness was updated; "
            "translation QC was not started automatically."
        )
        summarize(output_dir, cases)
        return 0

    if (
        not args.translation_qc_only
        and not args.regenerate_stale_translations
        and not args.import_human_translation_review
        and not args.export_human_translation_review
    ):
        if not args.export_human_source_review or not source_qc:
            validated = _run_source_qc(args, cases, output_dir, client)
            source_qc = {str(row["case_id"]): row for row in validated}
        carried_source_reviews = by_case(
            output_dir / "human_validated_masters.jsonl"
        )
        rows = source_review_rows(
            args.batch_name,
            cases,
            _unreviewed_records(
                source_qc.values(), cases, carried_source_reviews,
            ),
        )
        source_review_path = (
            args.human_source_review_output.resolve()
            if args.human_source_review_output
            else output_dir / "human_source_review.csv"
        )
        write_csv(source_review_path, SOURCE_REVIEW_FIELDS, rows)
        summarize(output_dir, cases)
        print(
            f"Automatic source QC complete for {len(rows)} cases. "
            f"Stopped for human source review: "
            f"{source_review_path}"
        )
        return 0

    human_masters = by_case(output_dir / "human_validated_masters.jsonl")
    _overlay_regenerated_translations(cases, human_masters, output_dir)
    if args.regenerate_stale_translations:
        generated, failures = _run_translation_regeneration(
            args, cases, human_masters, output_dir, client,
        )
        _overlay_regenerated_translations(cases, human_masters, output_dir)
        summarize(output_dir, cases)
        print(
            f"Translation regeneration complete: {len(generated)} generated, "
            f"{len(failures)} failed. Translation QC was not started "
            "automatically."
        )
        return 0 if not failures and len(generated) == len(cases) else 3
    ready, stale = translation_readiness(cases, source_qc, human_masters)
    if stale:
        merge_jsonl(
            output_dir / "translations_requiring_regeneration.jsonl", stale,
        )
    ready_by_case = {
        str(item["case"]["case_id"]): item for item in ready
    }
    if args.import_human_translation_review:
        import_translation_reviews(
            args.import_human_translation_review.resolve(),
            ready_by_case, output_dir,
        )
        report_final = finalize_pairs(
            output_dir, cases, source_qc,
            by_case(output_dir / "validated_translation_qc.jsonl"),
            human_masters,
            by_case(output_dir / "human_validated_translations.jsonl"),
            ROOT / "outputs" / "experiments",
            ROOT / "outputs" / "manifests",
            args.batch_name,
        )
        summarize(output_dir, cases)
        print(json.dumps(report_final, ensure_ascii=False, indent=2))
        return 0
    if not ready:
        print(
            "No translations are ready: import accepted human source reviews "
            "and regenerate any stale translations first.",
            file=sys.stderr,
        )
        return 3

    translation_qc = by_case(output_dir / "validated_translation_qc.jsonl")
    if not args.export_human_translation_review or not translation_qc:
        validated_translation = _run_translation_qc(
            args, ready, output_dir, client,
        )
        translation_qc = {
            str(row["case_id"]): row for row in validated_translation
        }
    carried_translation_reviews = by_case(
        output_dir / "human_validated_translations.jsonl"
    )
    rows = translation_review_rows(
        args.batch_name,
        ready_by_case,
        _unreviewed_records(
            translation_qc.values(),
            ready_by_case,
            carried_translation_reviews,
        ),
    )
    translation_review_path = (
        args.human_translation_review_output.resolve()
        if args.human_translation_review_output
        else output_dir / "human_translation_review.csv"
    )
    write_csv(
        translation_review_path, TRANSLATION_REVIEW_FIELDS, rows,
    )
    if args.run_back_translation:
        _run_back_translation(
            args, ready_by_case, translation_qc, output_dir, client,
        )
    summarize(output_dir, cases)
    print(
        f"Automatic translation QC complete for {len(rows)} cases. "
        f"Stopped for human bilingual review: "
        f"{translation_review_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
