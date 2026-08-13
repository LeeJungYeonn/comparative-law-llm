from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import normalized_whitespace, read_jsonl, write_json, write_jsonl

CURRENCY_LITERAL = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion))?|"
    r"\b(?:millions?|billions?)\s+of\s+dollars\b|"
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:dollars?|won)\b|\d[\d,]*(?:\.\d+)?\s*원",
    re.I,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply logged, deterministic repairs to replacement source-language facts.")
    p.add_argument("--source", type=Path, default=Path("outputs_v2/v3_replacements/neutral_facts_source.jsonl"))
    p.add_argument("--bilingual", type=Path, default=Path("outputs_v2/v3_replacements/neutral_facts_bilingual.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs_v2/v3_replacements_repaired"))
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source_out = args.output_dir / "neutral_facts_source.jsonl"
    bilingual_out = args.output_dir / "neutral_facts_bilingual.jsonl"
    if not args.overwrite and (source_out.exists() or bilingual_out.exists()):
        raise FileExistsError("Repair outputs exist; pass --overwrite")
    sources = list(read_jsonl(args.source))
    prior_bilingual = {row["case_id"]: row for row in read_jsonl(args.bilingual)}
    amendments: list[dict[str, Any]] = []
    changed_ids: set[str] = set()
    repaired: list[dict[str, Any]] = []
    for original in sources:
        record = dict(original)
        units = [dict(unit) for unit in original.get("fact_units") or []]
        prior_texts: list[tuple[str, str]] = []
        for unit in units:
            if not unit.get("include_in_neutral_fact"):
                continue
            old = unit.get("text") or ""
            new = CURRENCY_LITERAL.sub("[CURRENCY_AMOUNT_A]", old)
            normalized = normalized_whitespace(new).casefold()
            duplicate_of = next((fact_id for fact_id, prior in prior_texts if len(normalized) >= 40 and normalized in prior), None)
            if duplicate_of:
                unit["include_in_neutral_fact"] = False
                unit["exclusion_reason"] = f"duplicate_content_of:{duplicate_of}"
                amendments.append({
                    "case_id": record["case_id"], "field_changed": f"fact_units.{unit.get('fact_id')}.include_in_neutral_fact",
                    "old_text": old, "new_text": "", "source_evidence": unit.get("source_span"),
                    "reason": f"duplicate factual content already included in {duplicate_of}", "review_stage": "replacement_deterministic_qc_v3",
                })
                changed_ids.add(record["case_id"])
                continue
            if new != old:
                unit["text_before_v3_qc_repair"] = old
                unit["text"] = new
                amendments.append({
                    "case_id": record["case_id"], "field_changed": f"fact_units.{unit.get('fact_id')}.text",
                    "old_text": old, "new_text": new, "source_evidence": unit.get("source_span"),
                    "reason": "neutralized literal currency identity", "review_stage": "replacement_deterministic_qc_v3",
                })
                changed_ids.add(record["case_id"])
                normalized = normalized_whitespace(new).casefold()
            prior_texts.append((unit.get("fact_id") or "", normalized))
        master = " ".join(unit["text"].strip() for unit in units if unit.get("include_in_neutral_fact"))
        record["fact_units"] = units
        record["neutral_fact_source"] = master
        record["neutral_fact_ko"] = master if record.get("source_language") == "ko" else ""
        record["neutral_fact_en"] = master if record.get("source_language") == "en" else ""
        repaired.append(record)
    seed_bilingual = [prior_bilingual[row["case_id"]] for row in repaired if row["case_id"] not in changed_ids and row["case_id"] in prior_bilingual]
    write_jsonl(source_out, repaired)
    write_jsonl(bilingual_out, seed_bilingual)
    write_jsonl(args.output_dir / "replacement_amendments_v3.jsonl", amendments)
    write_json(args.output_dir / "repair_summary.json", {"changed_case_ids": sorted(changed_ids), "changed_cases": len(changed_ids), "amendments": len(amendments), "seeded_unchanged_translations": len(seed_bilingual)})
    print({"changed_cases": len(changed_ids), "amendments": len(amendments), "seeded_unchanged_translations": len(seed_bilingual)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
