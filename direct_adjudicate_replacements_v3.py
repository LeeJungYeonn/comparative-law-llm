from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl

DECISIONS = {
    "KR_3e400c529b48b85317": {
        "exclude": {"F007": "legal_negligence_conclusion"},
        "texts": {"F005": "[PERSON_A]가 사망했다."},
    },
    "KR_adf561060b125d6b87": {
        "texts": {"F003": "[PERSON_A]는 사고로 인한 후유장해 때문에 2000년 12월 26일 [COMPANY_C]의 대표이사직에서 퇴임했다."},
    },
    "US_6ba935b9156e1996aa": {
        "exclude": {"F002": "duplicate_of_F012", "F009": "legal_claim_content", "F010": "legal_right_conclusion", "F011": "legal_duty_conclusion"},
    },
    "US_8026310f8d6a3819bd": {
        "exclude": {"F001": "litigation_roles_only", "F010": "duplicate_request_refusal_event"},
        "types": {"F002": "parties", "F004": "causation"},
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply direct source adjudications to the final four replacement records.")
    p.add_argument("--input-dir", type=Path, default=Path("outputs_v2/v3_replacement_round3"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs_v2/v3_replacement_round4"))
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    sources = list(read_jsonl(args.input_dir / "neutral_facts_source.jsonl"))
    bilingual = {row["case_id"]: row for row in read_jsonl(args.input_dir / "neutral_facts_bilingual.jsonl")}
    amendments: list[dict[str, Any]] = []
    for record in sources:
        decision = DECISIONS.get(record["case_id"])
        if not decision:
            continue
        for unit in record.get("fact_units") or []:
            fact_id = unit.get("fact_id")
            if fact_id in decision.get("exclude", {}):
                unit["include_in_neutral_fact"] = False
                unit["exclusion_reason"] = decision["exclude"][fact_id]
                amendments.append({"case_id": record["case_id"], "field_changed": f"fact_units.{fact_id}.include_in_neutral_fact", "old_text": unit.get("text"), "new_text": "", "source_evidence": unit.get("source_span"), "reason": decision["exclude"][fact_id], "review_stage": "final_direct_source_adjudication_v3"})
            if fact_id in decision.get("types", {}):
                old, new = unit.get("fact_type"), decision["types"][fact_id]
                unit["fact_type"] = new
                amendments.append({"case_id": record["case_id"], "field_changed": f"fact_units.{fact_id}.fact_type", "old_text": old, "new_text": new, "source_evidence": unit.get("source_span"), "reason": "mandatory factual role verified from source span", "review_stage": "final_direct_source_adjudication_v3"})
            if fact_id in decision.get("texts", {}):
                old, new = unit.get("text"), decision["texts"][fact_id]
                unit["text"] = new
                unit["text_before_direct_adjudication"] = old
                amendments.append({"case_id": record["case_id"], "field_changed": f"fact_units.{fact_id}.text", "old_text": old, "new_text": new, "source_evidence": unit.get("source_span"), "reason": "removed duplicated clause while retaining source-supported atomic fact", "review_stage": "final_direct_source_adjudication_v3"})
        master = " ".join(unit["text"].strip() for unit in record.get("fact_units") or [] if unit.get("include_in_neutral_fact"))
        record["neutral_fact_source"] = master
        record["neutral_fact_ko"] = master if record.get("source_language") == "ko" else ""
        record["neutral_fact_en"] = master if record.get("source_language") == "en" else ""
    write_jsonl(args.output_dir / "neutral_facts_source.jsonl", sources)
    write_jsonl(args.output_dir / "neutral_facts_bilingual.jsonl", [bilingual[row["case_id"]] for row in sources if row["case_id"] not in DECISIONS])
    write_jsonl(args.output_dir / "replacement_amendments_v3.jsonl", amendments)
    write_json(args.output_dir / "adjudication_summary.json", {"case_ids": sorted(DECISIONS), "cases": len(DECISIONS), "amendments": len(amendments), "seeded_translations": len(sources) - len(DECISIONS)})
    print({"cases": len(DECISIONS), "amendments": len(amendments), "seeded": len(sources) - len(DECISIONS)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
