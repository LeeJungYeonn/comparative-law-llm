from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from pipeline.checkpoint import atomic_write_json, atomic_write_text
from pipeline.stage2_runtime import by_case, resolve_case_ids


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a detailed Stage 2 calibration report without API calls.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--case-id-file", type=Path, required=True)
    p.add_argument("--output-prefix", default="calibration_report")
    return p


def _status(master: dict[str, Any], translated: dict[str, Any], grounding: dict[str, Any], translation_verify: dict[str, Any]) -> str:
    statuses = {
        master.get("neutralization_status", "missing"), translated.get("translation_status", "missing"),
        grounding.get("validated_verifier_status", "missing"), translation_verify.get("validated_verifier_status", "missing"),
    }
    return "missing" if "missing" in statuses else "fail" if "fail" in statuses else "warning" if "warning" in statuses else "pass"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); root = args.input_dir
    masters: dict[str, Any] = {}; evidence: dict[str, Any] = {}; translations: dict[str, Any] = {}
    for suffix in ("kr", "ca"):
        masters.update(by_case(root / f"source_neutral_{suffix}.jsonl"))
        evidence.update(by_case(root / f"factual_evidence_{suffix}.jsonl"))
        translations.update(by_case(root / f"translated_pairs_{suffix}.jsonl"))
    grounding = by_case(root / "source_grounding_verification.jsonl"); translation_verify = by_case(root / "translation_verification.jsonl")
    case_ids = resolve_case_ids(None, args.case_id_file); rows = []
    for case_id in case_ids:
        master, ev, translated = masters.get(case_id, {}), evidence.get(case_id, {}), translations.get(case_id, {})
        ground, trans_verify = grounding.get(case_id, {}), translation_verify.get(case_id, {})
        checks, trans_checks = master.get("deterministic_checks") or {}, translated.get("deterministic_checks") or {}
        fact_units = master.get("fact_units") or []
        event_facts = [unit.get("master_text", "") for unit in fact_units if set(unit.get("fact_types") or []) & {"action", "event", "causation_relevant"}]
        harm_facts = [unit.get("master_text", "") for unit in fact_units if set(unit.get("fact_types") or []) & {"harm", "economic_harm"}]
        provenances = (ev.get("chunk_model_provenance") or []) + [master.get("model_provenance") or {}, translated.get("model_provenance") or {}, ground.get("model_provenance") or {}, trans_verify.get("model_provenance") or {}]
        provenance = "mock" if any(item.get("mock") for item in provenances) else "real" if provenances and all(item and not item.get("mock") for item in provenances) else "missing"
        number_details = trans_checks.get("number_normalization") or {}; unit_details = trans_checks.get("unit_normalization") or {}
        rows.append({
            "case_id": case_id, "origin": master.get("case_origin"), "subtype": master.get("case_subtype"),
            "source_coverage": ev.get("source_coverage") or master.get("source_coverage") or {},
            "core_event_facts": event_facts, "harm_facts": harm_facts,
            "event_present": checks.get("event_present"), "harm_present": checks.get("harm_present"), "causal_sequence_present": checks.get("causal_sequence_present"),
            "source_neutral_status": master.get("neutralization_status"), "model_insufficient_factual_detail": master.get("model_insufficient_factual_detail"),
            "deterministic_factual_sufficiency": master.get("deterministic_factual_sufficiency"),
            "number_normalization": {"all_values_match": all(item.get("match") for item in number_details.values()), "details": number_details},
            "unit_normalization": {"all_values_match": all(item.get("match") for item in unit_details.values()), "details": unit_details},
            "placeholder_set_comparison": {"match": trans_checks.get("placeholder_match"), "occurrence_match": trans_checks.get("placeholder_occurrence_match"), "fact_units": trans_checks.get("fact_placeholder_sets")},
            "negation_warning": trans_checks.get("negation_warning"), "language_residue": trans_checks.get("language_residue") or [],
            "translation_status": translated.get("translation_status"),
            "grounding_model_verifier_status": ground.get("model_verifier_status"), "grounding_validated_verifier_status": ground.get("validated_verifier_status"),
            "translation_model_verifier_status": trans_verify.get("model_verifier_status"), "translation_validated_verifier_status": trans_verify.get("validated_verifier_status"),
            "verifier_consistency_violation": bool(ground.get("verifier_consistency_violation") or trans_verify.get("verifier_consistency_violation")),
            "deterministic_warnings": sorted(set((checks.get("warnings") or []) + (trans_checks.get("warnings") or []))),
            "hard_failures": sorted(set((checks.get("errors") or []) + (trans_checks.get("errors") or []))),
            "automatic_status": _status(master, translated, ground, trans_verify), "human_qc_status": "pending", "provenance": provenance,
        })
    prefix = root / args.output_prefix; atomic_write_json(prefix.with_suffix(".json"), {"case_count": len(rows), "cases": rows})
    fields = ["case_id", "origin", "subtype", "segment_coverage", "character_coverage", "extraction_calls", "fact_units", "event_present", "harm_present", "causal_sequence_present", "source_neutral_status", "model_insufficient", "deterministic_sufficiency", "number_values_match", "unit_values_match", "placeholder_set_match", "negation_warning", "language_residue", "grounding_model_status", "grounding_validated_status", "translation_model_status", "translation_validated_status", "automatic_status", "provenance"]
    stream = StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
    for row in rows:
        coverage = row["source_coverage"]
        writer.writerow({"case_id": row["case_id"], "origin": row["origin"], "subtype": row["subtype"], "segment_coverage": coverage.get("segment_coverage_ratio"), "character_coverage": coverage.get("character_coverage_ratio"), "extraction_calls": coverage.get("extraction_call_count"), "fact_units": len(masters.get(row["case_id"], {}).get("fact_units") or []), "event_present": row["event_present"], "harm_present": row["harm_present"], "causal_sequence_present": row["causal_sequence_present"], "source_neutral_status": row["source_neutral_status"], "model_insufficient": row["model_insufficient_factual_detail"], "deterministic_sufficiency": row["deterministic_factual_sufficiency"], "number_values_match": row["number_normalization"]["all_values_match"], "unit_values_match": row["unit_normalization"]["all_values_match"], "placeholder_set_match": row["placeholder_set_comparison"]["match"], "negation_warning": row["negation_warning"], "language_residue": ";".join(row["language_residue"]), "grounding_model_status": row["grounding_model_verifier_status"], "grounding_validated_status": row["grounding_validated_verifier_status"], "translation_model_status": row["translation_model_verifier_status"], "translation_validated_status": row["translation_validated_verifier_status"], "automatic_status": row["automatic_status"], "provenance": row["provenance"]})
    atomic_write_text(prefix.with_suffix(".csv"), stream.getvalue())
    print(f"calibration report complete: cases={len(rows)} json={prefix.with_suffix('.json')} csv={prefix.with_suffix('.csv')}")
    return 0 if len(rows) == len(case_ids) and all(row["automatic_status"] != "fail" for row in rows) else 2


if __name__ == "__main__": raise SystemExit(main())
