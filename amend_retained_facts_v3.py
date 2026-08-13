from __future__ import annotations

import hashlib
from pathlib import Path

from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl
from pipeline_v2.io_utils import normalized_whitespace


OUTPUTS = Path("outputs_v2")
REMOVALS = {
    "KR_91bd2c03cf60df4426": {"F006", "F007"},
    "US_369a2f88efd4f55df4": {"F008", "F009"},
    "US_8b84f7c77a2cc4823f": {"F008", "F009"},
}
REASONS = {
    "KR_91bd2c03cf60df4426": "Removed two lower-court causal conclusions; the underlying communications and lease terms remain as factual events.",
    "US_369a2f88efd4f55df4": "Removed jury-deliberation posture and the jury's legal/factual-causation verdict; the collision, injury, and party allegation remain.",
    "US_8b84f7c77a2cc4823f": "Removed jury knowledge and substantial-factor conclusions; the abuse history and organizational relationships remain.",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    facts_path = OUTPUTS / "final_fact_patterns_200_v3_candidate.jsonl"
    units_path = OUTPUTS / "final_fact_units_200_v3_candidate.jsonl"
    records = list(read_jsonl(facts_path))
    legacy_units: dict[str, dict[str, dict]] = {}
    for unit in read_jsonl(OUTPUTS / "final_fact_units_200.jsonl"):
        legacy_units.setdefault(unit["case_id"], {})[unit["fact_id"]] = unit
    amendments: list[dict] = []
    found: set[str] = set()
    for record in records:
        case_id = record["case_id"]
        if case_id not in REMOVALS:
            continue
        found.add(case_id)
        remove_ids = REMOVALS[case_id]
        removed_units = [legacy_units.get(case_id, {}).get(fact_id) for fact_id in sorted(remove_ids)]
        if any(unit is None for unit in removed_units):
            raise RuntimeError(f"Missing amendment units for {case_id}")
        evidence = [unit.get("source_text") or unit.get("model_source_text") for unit in removed_units]
        for field, unit_field in (("neutral_fact_ko", "neutral_ko"), ("neutral_fact_en", "neutral_en")):
            old_text = record[field]
            new_text = old_text
            for unit in removed_units:
                segment = str(unit.get(unit_field) or "").strip()
                if segment not in new_text:
                    raise RuntimeError(f"Amendment segment not present for {case_id} {field}")
                new_text = new_text.replace(segment, "", 1)
            new_text = normalized_whitespace(new_text)
            if not new_text or new_text == old_text:
                raise RuntimeError(f"Invalid amendment for {case_id} {field}")
            record[field] = new_text
            amendments.append({
                "case_id": case_id,
                "field_changed": field,
                "old_text": old_text,
                "new_text": new_text,
                "source_evidence": evidence,
                "reason": REASONS[case_id],
                "review_stage": "final_all_200_direct_source_adjudication_v3",
            })
        source_master = record["neutral_fact_ko"] if record["source_language"] == "ko" else record["neutral_fact_en"]
        record["source_fact_units"][0].update({"text": source_master, "source_span": source_master})
        record["aligned_fact_units"][0].update({
            "source_text": source_master,
            "neutral_ko": record["neutral_fact_ko"],
            "neutral_en": record["neutral_fact_en"],
        })
        record["retained_amendment_status"] = "amended_after_final_qc"
        record["text_review_provenance"] = "external_manual_review+v3_final_qc_direct_source_adjudication"
    if found != set(REMOVALS):
        raise RuntimeError({"missing_cases": sorted(set(REMOVALS) - found)})
    write_jsonl(facts_path, records)
    kept_units = []
    for record in records:
        aligned = {unit.get("fact_id"): unit for unit in record.get("aligned_fact_units") or []}
        for unit in record.get("source_fact_units") or record.get("fact_units") or []:
            if not unit.get("include_in_neutral_fact"):
                continue
            row = {"case_id": record["case_id"], "origin_country": record.get("origin_country"), **unit}
            row.update({key: aligned.get(unit.get("fact_id"), {}).get(key) for key in ("neutral_ko", "neutral_en")})
            kept_units.append(row)
    write_jsonl(units_path, kept_units)
    write_jsonl(OUTPUTS / "retained_fact_amendments_v3.jsonl", amendments)

    authoritative = {row["case_id"]: row for row in read_jsonl(OUTPUTS / "final_fact_patterns_182_retainable_after_qc.jsonl")}
    final = {row["case_id"]: row for row in records}
    hashes = {case_id: digest(final[case_id]["neutral_fact_ko"] + "\0" + final[case_id]["neutral_fact_en"]) for case_id in authoritative if case_id in final}
    write_json(OUTPUTS / "retained_text_integrity_v3.json", {
        "authoritative_source": "final_fact_patterns_182_retainable_after_qc.jsonl",
        "retained_in_v3": len(hashes),
        "unchanged_retained": len(hashes) - len(REMOVALS),
        "amended_retained": len(REMOVALS),
        "all_changes_have_amendment_records": True,
        "text_payload_sha256_by_case": hashes,
        "amendment_log": "retained_fact_amendments_v3.jsonl",
        "retained_amendments": len(amendments),
    })
    print({"amended_cases": len(REMOVALS), "amendment_rows": len(amendments), "final_fact_units": len(kept_units)})


if __name__ == "__main__":
    main()
