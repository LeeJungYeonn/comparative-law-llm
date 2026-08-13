from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import canonical_json, read_jsonl, sha256_text, write_json, write_jsonl


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge authoritative retained facts with final replacement facts.")
    p.add_argument("--outputs", type=Path, default=Path("outputs_v2"))
    p.add_argument("--replacement-dir", type=Path, default=Path("outputs_v2/v3_replacement_round4"))
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cases = list(read_jsonl(args.outputs / "provisional_final_cases_200_v3.jsonl"))
    authoritative = {row["case_id"]: row for row in read_jsonl(args.outputs / "final_fact_patterns_182_retainable_after_qc.jsonl")}
    replacements = {row["case_id"]: row for row in read_jsonl(args.replacement_dir / "neutral_facts_bilingual.jsonl")}
    final_records: list[dict[str, Any]] = []
    replacement_records: list[dict[str, Any]] = []
    retained_hashes: dict[str, str] = {}
    unit_rows: list[dict[str, Any]] = []
    replacement_unit_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        if case.get("replacement_status") == "retained":
            auth = authoritative.get(case_id)
            if not auth:
                raise RuntimeError(f"Missing authoritative retained fact: {case_id}")
            exact_payload = {key: auth.get(key) for key in ("neutral_fact_source", "neutral_fact_ko", "neutral_fact_en")}
            retained_hashes[case_id] = sha256_text(canonical_json(exact_payload))
            source_master = auth["neutral_fact_ko"] if auth["source_language"] == "ko" else auth["neutral_fact_en"]
            source_units = [{
                "fact_id": "REVIEWED_MASTER", "text": source_master,
                "source_span": source_master,
                "source_grounding_status": "pass", "include_in_neutral_fact": True,
                "fact_type": "externally_reviewed_complete_neutral_fact", "epistemic_status": "externally_reviewed",
            }]
            aligned_units = [{
                "case_id": case_id, "fact_id": "REVIEWED_MASTER", "source_text": source_master,
                "neutral_ko": auth["neutral_fact_ko"], "neutral_en": auth["neutral_fact_en"],
                "translation_equivalence_status": "pass", "source_text_copy_status": "exact",
            }]
            record = {
                "case_id": case_id, "source_fact_units": source_units, "aligned_fact_units": aligned_units,
                "neutral_fact_source": auth["neutral_fact_source"], "neutral_fact_ko": auth["neutral_fact_ko"], "neutral_fact_en": auth["neutral_fact_en"],
                "source_language": auth["source_language"], "primary_domain": case["primary_domain"], "case_domain": case["primary_domain"],
                "liability_theories": case.get("liability_theories") or [], "origin_country": case["origin_country"], "origin_state": case.get("origin_state"),
                "text_review_provenance": "external_manual_review_182_authoritative", "replacement_status": "retained",
                "corpus_version": "kr-us-highcourt-corpus-v3.0", "translation_equivalence_status": "pass",
                "external_manual_source_review_status": "pass",
            }
        else:
            record = dict(replacements.get(case_id) or {})
            if not record:
                raise RuntimeError(f"Missing replacement fact: {case_id}")
            record.update({
                "primary_domain": case["primary_domain"], "case_domain": case["primary_domain"], "liability_theories": case.get("liability_theories") or [],
                "text_review_provenance": "replacement_generation_and_direct_qc_v3", "replacement_status": "replacement_v3",
                "corpus_version": "kr-us-highcourt-corpus-v3.0",
            })
            replacement_records.append(record)
        final_records.append(record)
        aligned = {unit.get("fact_id"): unit for unit in record.get("aligned_fact_units") or []}
        for unit in record.get("source_fact_units") or record.get("fact_units") or []:
            if not unit.get("include_in_neutral_fact"):
                continue
            combined = {"case_id": case_id, "origin_country": case["origin_country"], **unit}
            if unit.get("fact_id") in aligned:
                combined["neutral_ko"] = aligned[unit["fact_id"]].get("neutral_ko")
                combined["neutral_en"] = aligned[unit["fact_id"]].get("neutral_en")
            unit_rows.append(combined)
            if case.get("replacement_status") != "retained":
                replacement_unit_rows.append(combined)
    retained_ids = {case["case_id"] for case in cases if case.get("replacement_status") == "retained"}
    if len(retained_ids) != 143 or set(retained_hashes) != retained_ids:
        raise RuntimeError(f"Retained authoritative invariant failed: {len(retained_ids)} / {len(retained_hashes)}")
    write_jsonl(args.outputs / "replacement_fact_patterns_v3.jsonl", replacement_records)
    write_jsonl(args.outputs / "replacement_fact_units_v3.jsonl", replacement_unit_rows)
    write_jsonl(args.outputs / "final_fact_patterns_200_v3_candidate.jsonl", final_records)
    write_jsonl(args.outputs / "final_fact_units_200_v3_candidate.jsonl", unit_rows)
    write_json(args.outputs / "retained_text_integrity_v3.json", {"authoritative_source": "final_fact_patterns_182_retainable_after_qc.jsonl", "retained_in_v3": 143, "all_text_payloads_copied_exactly": True, "text_payload_sha256_by_case": retained_hashes, "amendment_log_required_for_retained_changes": True, "retained_amendments": 0})
    print({"final": len(final_records), "retained_authoritative": len(retained_ids), "replacements": len(replacement_records), "units": len(unit_rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
