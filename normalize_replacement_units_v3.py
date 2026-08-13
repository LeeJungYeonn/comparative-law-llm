from __future__ import annotations

import argparse
import re
from pathlib import Path

from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl

POUNDS = re.compile(r"\b(\d+(?:\.\d+)?)\s+pounds?\b", re.I)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize residual customary units in replacement neutral text.")
    p.add_argument("--input-dir", type=Path, default=Path("outputs_v2/v3_replacement_final"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs_v2/v3_replacement_final_normalized"))
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    sources = list(read_jsonl(args.input_dir / "neutral_facts_source.jsonl"))
    bilingual = {row["case_id"]: row for row in read_jsonl(args.input_dir / "neutral_facts_bilingual.jsonl")}
    changed: set[str] = set()
    amendments = []
    for record in sources:
        for unit in record.get("fact_units") or []:
            old = unit.get("text") or ""
            def repl(match: re.Match[str]) -> str:
                kg = round(float(match.group(1)) * 0.45359237)
                return f"approximately {kg} kilograms"
            new = POUNDS.sub(repl, old)
            if new != old:
                unit["text_before_v3_unit_normalization"] = old
                unit["text"] = new
                changed.add(record["case_id"])
                amendments.append({"case_id": record["case_id"], "field_changed": f"fact_units.{unit.get('fact_id')}.text", "old_text": old, "new_text": new, "source_evidence": unit.get("source_span"), "reason": "normalized pounds to metric kilograms; original retained in source span", "review_stage": "replacement_deterministic_qc_v3"})
        master = " ".join(unit["text"].strip() for unit in record.get("fact_units") or [] if unit.get("include_in_neutral_fact"))
        record["neutral_fact_source"] = master
        record["neutral_fact_ko"] = master if record.get("source_language") == "ko" else ""
        record["neutral_fact_en"] = master if record.get("source_language") == "en" else ""
    write_jsonl(args.output_dir / "neutral_facts_source.jsonl", sources)
    write_jsonl(args.output_dir / "neutral_facts_bilingual.jsonl", [bilingual[row["case_id"]] for row in sources if row["case_id"] not in changed])
    write_jsonl(args.output_dir / "replacement_amendments_v3.jsonl", amendments)
    write_json(args.output_dir / "normalization_summary.json", {"changed_case_ids": sorted(changed), "changed_cases": len(changed), "seeded_translations": len(sources) - len(changed)})
    print({"changed_cases": len(changed), "seeded": len(sources) - len(changed)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
