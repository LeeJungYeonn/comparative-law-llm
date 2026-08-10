from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl, atomic_write_text
from pipeline.stage2_runtime import append_run_history, by_case, json_hash


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge sequential source masters and translations without dropping failed cases.")
    p.add_argument("--input-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path); p.add_argument("--batch-name", default="unnamed"); p.add_argument("--overwrite", action="store_true"); p.add_argument("--resume", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); source = args.input_dir; output = args.output_dir or source
    all_path, pass_path, qc_path = output / "neutral_pairs_all.jsonl", output / "neutral_pairs_pass.jsonl", output / "human_qc_template.csv"
    existing = [path for path in (all_path, pass_path, qc_path) if path.exists()]
    if existing and not (args.overwrite or args.resume): raise FileExistsError(f"Merged output exists; pass --resume or --overwrite: {existing[0]}")
    manifest = json.loads((source / "input_manifest.json").read_text(encoding="utf-8")); manifest_ids = manifest["kr_case_ids"] + manifest["ca_case_ids"]
    origin_by_id = {case_id: "KR" for case_id in manifest["kr_case_ids"]} | {case_id: "CA" for case_id in manifest["ca_case_ids"]}
    masters = {}; translations = {}
    for suffix in ("kr", "ca"):
        masters.update(by_case(source / f"source_neutral_{suffix}.jsonl")); translations.update(by_case(source / f"translated_pairs_{suffix}.jsonl"))
    grounding, translation_verify = by_case(source / "source_grounding_verification.jsonl"), by_case(source / "translation_verification.jsonl")
    rows = []
    for case_id in manifest_ids:
        origin = origin_by_id[case_id]; master = masters.get(case_id, {}); translated = translations.get(case_id, {}); ground = grounding.get(case_id, {}); verify = translation_verify.get(case_id, {})
        master_text, translated_text = str(master.get("master_neutral_text") or ""), str(translated.get("translated_neutral_text") or "")
        translated_units = {unit.get("fact_id"): unit.get("translated_text", "") for unit in translated.get("translated_fact_units", [])}; canonical = []
        for unit in master.get("fact_units", []):
            fact_id, master_unit, translated_unit = unit.get("fact_id"), unit.get("master_text", ""), translated_units.get(unit.get("fact_id"), "")
            canonical.append({"fact_id": fact_id, "epistemic_status": unit.get("epistemic_status"), "ko": master_unit if origin == "KR" else translated_unit, "en": translated_unit if origin == "KR" else master_unit, "ko_generation_type": "source_neutralized" if origin == "KR" else "translated", "en_generation_type": "translated" if origin == "KR" else "source_neutralized"})
        source_status = master.get("neutralization_status", "missing"); ground_status = ground.get("validated_verifier_status", ground.get("verifier_status", "missing")); trans_status = translated.get("translation_status", "missing"); trans_verify_status = verify.get("validated_verifier_status", verify.get("translation_status", "missing"))
        statuses = {source_status, ground_status, trans_status, trans_verify_status}
        automatic_status = "missing" if "missing" in statuses else "fail" if "fail" in statuses else "warning" if "warning" in statuses else "pass"
        automatic_pass = automatic_status == "pass" and bool(master.get("case_is_usable_for_translation", master.get("case_is_usable")))
        rows.append({"dataset_version": manifest["dataset_version"], "case_id": case_id, "case_origin": origin, "case_subtype": master.get("case_subtype"), "master_language": "ko" if origin == "KR" else "en", "translation_direction": "ko_to_en" if origin == "KR" else "en_to_ko", "neutral_fact_ko": master_text if origin == "KR" else translated_text, "neutral_fact_en": translated_text if origin == "KR" else master_text, "canonical_facts": canonical, "automatic_status": automatic_status, "case_is_usable": automatic_pass, "case_is_usable_for_translation": master.get("case_is_usable_for_translation"), "case_is_finally_usable": None, "source_neutral_status": source_status, "grounding_verifier_status": ground_status, "translation_status": trans_status, "translation_verifier_status": trans_verify_status, "human_qc_status": "", "human_qc_notes": "", "source_text_sha256": master.get("source_text_sha256", ""), "source_output_sha256": json_hash(master) if master else "", "translation_output_sha256": json_hash(translated) if translated else ""})
    atomic_write_jsonl(all_path, rows); atomic_write_jsonl(pass_path, [row for row in rows if row["case_is_usable"]])
    qc_fields = ["case_id", "case_origin", "case_subtype", "source_neutral_status", "grounding_verifier_status", "translation_status", "translation_verifier_status", "automatic_status", "human_qc_status", "human_qc_notes"]
    stream = StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=qc_fields, lineterminator="\n"); writer.writeheader(); writer.writerows({field: row.get(field, "") for field in qc_fields} for row in rows); atomic_write_text(qc_path, stream.getvalue())
    quality_fields = ["case_id", "origin", "subtype", "source_coverage", "fact_unit_count", "event_present", "harm_present", "causal_sequence_present", "source_neutral_status", "translation_status", "grounding_verifier_status", "translation_verifier_status", "deterministic_warnings", "hard_failures", "human_qc_status"]
    quality_stream = StringIO(newline=""); quality_writer = csv.DictWriter(quality_stream, fieldnames=quality_fields, lineterminator="\n"); quality_writer.writeheader()
    for row in rows:
        master = masters.get(row["case_id"], {}); translated = translations.get(row["case_id"], {}); checks = master.get("deterministic_checks") or {}; trans_checks = translated.get("deterministic_checks") or {}
        quality_writer.writerow({"case_id": row["case_id"], "origin": row["case_origin"], "subtype": row.get("case_subtype"), "source_coverage": (master.get("source_coverage") or {}).get("segment_coverage_ratio"), "fact_unit_count": len(master.get("fact_units") or []), "event_present": checks.get("event_present"), "harm_present": checks.get("harm_present"), "causal_sequence_present": checks.get("causal_sequence_present"), "source_neutral_status": row["source_neutral_status"], "translation_status": row["translation_status"], "grounding_verifier_status": row["grounding_verifier_status"], "translation_verifier_status": row["translation_verifier_status"], "deterministic_warnings": ";".join((checks.get("warnings") or []) + (trans_checks.get("warnings") or [])), "hard_failures": ";".join((checks.get("errors") or []) + (trans_checks.get("errors") or [])), "human_qc_status": ""})
    atomic_write_text(output / "quality_report.csv", quality_stream.getvalue())
    master_count, translation_count, grounding_count, translation_verification_count = len(masters), len(translations), len(grounding), len(translation_verify)
    complete_pair_count = sum(case_id in masters and case_id in translations and case_id in grounding and case_id in translation_verify for case_id in manifest_ids)
    counts = {"manifest_case_count": len(manifest_ids), "master_record_count": master_count, "translation_record_count": translation_count, "grounding_verification_count": grounding_count, "translation_verification_count": translation_verification_count, "complete_pair_count": complete_pair_count, "missing_master_count": len(manifest_ids) - master_count, "missing_translation_count": len(manifest_ids) - translation_count, "automatic_pass_count": sum(bool(row["case_is_usable"]) for row in rows), "human_qc_pending_count": complete_pair_count}
    execution = "completed" if complete_pair_count == len(manifest_ids) else "completed_subset" if complete_pair_count else "partial"
    phase = {"execution_status": execution, "record_counts": {"pass": counts["automatic_pass_count"], "warning": sum(row["automatic_status"] == "warning" for row in rows), "fail": sum(row["automatic_status"] == "fail" for row in rows), "missing": sum(row["automatic_status"] == "missing" for row in rows)}, **counts}
    append_run_history(output, {"batch_name": args.batch_name, "selected_case_ids": [row["case_id"] for row in rows if row["automatic_status"] != "missing"], "mode": "merge", "new_api_calls": 0, "cache_hits": 0, **counts}, {"phase_9_merge": phase})
    mismatch = any(case_id in translations and case_id not in masters for case_id in translations) or complete_pair_count > min(master_count, translation_count, grounding_count, translation_verification_count)
    print(f"merge complete: all={len(rows)} complete_pairs={complete_pair_count} pass={counts['automatic_pass_count']} status={execution} mismatch={mismatch}")
    return 2 if mismatch else 0


if __name__ == "__main__": raise SystemExit(main())
