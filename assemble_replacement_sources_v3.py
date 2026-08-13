from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import canonical_json, normalized_whitespace, read_jsonl, sha256_text, write_json, write_jsonl

CURRENCY_LITERAL = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion))?|\b(?:millions?|billions?)\s+of\s+dollars\b|"
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:dollars?|won)\b|\d[\d,]*(?:\.\d+)?\s*원",
    re.I,
)
POUNDS = re.compile(r"\b(\d+(?:\.\d+)?)\s+pounds?\b", re.I)
FALLBACKS = {
    "KR_6623c47ec5cdef48bc": {"exclude": {"F003": "duplicate_harm_already_in_F002"}, "types": {"F002": "harm"}},
    "US_60f81ce937468b4218": {"exclude": {"F001": "litigation_role_only"}, "types": {"F003": "parties"}},
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble source records after semantic repair and direct source adjudication.")
    p.add_argument("--outputs", type=Path, default=Path("outputs_v2"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs_v2/v3_replacement_final"))
    p.add_argument("--base-dir", type=Path)
    p.add_argument("--repair-dir", type=Path)
    p.add_argument("--semantic-file", type=Path)
    p.add_argument("--no-fallbacks", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def repair_record(record: dict[str, Any], amendments: list[dict[str, Any]], apply_fallbacks: bool = True) -> dict[str, Any]:
    row = dict(record)
    units = [dict(unit) for unit in row.get("fact_units") or []]
    fallback = FALLBACKS.get(row["case_id"], {}) if apply_fallbacks else {}
    prior: list[tuple[str, str]] = []
    for unit in units:
        fact_id = unit.get("fact_id") or ""
        if fact_id in fallback.get("exclude", {}):
            unit["include_in_neutral_fact"] = False
            unit["exclusion_reason"] = fallback["exclude"][fact_id]
            amendments.append({"case_id": row["case_id"], "field_changed": f"fact_units.{fact_id}.include_in_neutral_fact", "old_text": unit.get("text"), "new_text": "", "source_evidence": unit.get("source_span"), "reason": fallback["exclude"][fact_id], "review_stage": "direct_source_adjudication_v3"})
            continue
        if fact_id in fallback.get("types", {}):
            old_type, new_type = unit.get("fact_type"), fallback["types"][fact_id]
            unit["fact_type"] = new_type
            amendments.append({"case_id": row["case_id"], "field_changed": f"fact_units.{fact_id}.fact_type", "old_text": old_type, "new_text": new_type, "source_evidence": unit.get("source_span"), "reason": "direct source adjudication of mandatory factual role", "review_stage": "direct_source_adjudication_v3"})
        if not unit.get("include_in_neutral_fact"):
            continue
        old = unit.get("text") or ""
        new = CURRENCY_LITERAL.sub("[CURRENCY_AMOUNT_A]", old)
        new = POUNDS.sub(lambda match: f"approximately {round(float(match.group(1)) * 0.45359237)} kilograms", new)
        norm = normalized_whitespace(new).casefold()
        duplicate = next((prior_id for prior_id, prior_text in prior if len(norm) >= 40 and norm in prior_text), None)
        if duplicate:
            unit["include_in_neutral_fact"] = False
            unit["exclusion_reason"] = f"duplicate_content_of:{duplicate}"
            amendments.append({"case_id": row["case_id"], "field_changed": f"fact_units.{fact_id}.include_in_neutral_fact", "old_text": old, "new_text": "", "source_evidence": unit.get("source_span"), "reason": f"duplicate content of {duplicate}", "review_stage": "semantic_repair_deterministic_qc_v3"})
            continue
        if new != old:
            unit["text_before_v3_qc_repair"] = old
            unit["text"] = new
            reason = "normalized jurisdiction-sensitive currency or customary unit"
            amendments.append({"case_id": row["case_id"], "field_changed": f"fact_units.{fact_id}.text", "old_text": old, "new_text": new, "source_evidence": unit.get("source_span"), "reason": reason, "review_stage": "semantic_repair_deterministic_qc_v3"})
            norm = normalized_whitespace(new).casefold()
        prior.append((fact_id, norm))
    master = " ".join(unit["text"].strip() for unit in units if unit.get("include_in_neutral_fact"))
    row["fact_units"] = units
    row["neutral_fact_source"] = master
    row["neutral_fact_ko"] = master if row.get("source_language") == "ko" else ""
    row["neutral_fact_en"] = master if row.get("source_language") == "en" else ""
    return row


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source_out = args.output_dir / "neutral_facts_source.jsonl"
    bilingual_out = args.output_dir / "neutral_facts_bilingual.jsonl"
    if not args.overwrite and (source_out.exists() or bilingual_out.exists()):
        raise FileExistsError("Assembled outputs exist; pass --overwrite")
    base_dir = args.base_dir or args.outputs / "v3_replacements_repaired"
    repair_dir = args.repair_dir or args.outputs / "v3_semantic_repair"
    semantic_file = args.semantic_file or args.outputs / "replacement_neutral_fact_semantic_qc_v3.jsonl"
    base = {row["case_id"]: row for row in read_jsonl(base_dir / "neutral_facts_source.jsonl")}
    base_bilingual = {row["case_id"]: row for row in read_jsonl(base_dir / "neutral_facts_bilingual.jsonl")}
    repaired = {row["case_id"]: row for row in read_jsonl(repair_dir / "neutral_facts_source.jsonl")}
    semantic = {row["case_id"]: row for row in read_jsonl(semantic_file)}
    amendments: list[dict[str, Any]] = []
    assembled: list[dict[str, Any]] = []
    for case_id in sorted(base):
        failed = semantic[case_id].get("hard_fail") or semantic[case_id].get("manual_review_required")
        use_fallback = case_id in FALLBACKS and not args.no_fallbacks
        chosen = repaired.get(case_id) if failed and not use_fallback else base[case_id]
        if chosen is None:
            raise RuntimeError(f"Missing semantic repair for {case_id}")
        assembled.append(repair_record(chosen, amendments, apply_fallbacks=not args.no_fallbacks))
    seed = []
    for row in assembled:
        prior = base.get(row["case_id"])
        if prior and row["case_id"] in base_bilingual and sha256_text(canonical_json(row.get("fact_units"))) == sha256_text(canonical_json(prior.get("fact_units"))):
            seed.append(base_bilingual[row["case_id"]])
    write_jsonl(source_out, assembled)
    write_jsonl(bilingual_out, seed)
    write_jsonl(args.output_dir / "replacement_amendments_v3.jsonl", amendments)
    write_json(args.output_dir / "assembly_summary.json", {"records": len(assembled), "seeded_unchanged_translations": len(seed), "translations_required": len(assembled) - len(seed), "amendments": len(amendments), "direct_adjudications": sorted(FALLBACKS)})
    print({"records": len(assembled), "seeded": len(seed), "translate": len(assembled) - len(seed), "amendments": len(amendments)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
