from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import guard_outputs, normalized_whitespace, read_jsonl, stable_id, write_csv, write_jsonl
from pipeline_v2.rules import KR_CASE_RE, KR_LOWER_CODES, assess_fact_sufficiency


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Link lower-court records using exact identifiers only and supplement fact-poor high-court opinions.")
    result.add_argument("--input", type=Path, action="append", required=True, help="High-court/candidate JSONL; may be repeated.")
    result.add_argument("--lower-source", type=Path, action="append", default=[], help="Additional candidate JSONL containing possible lower-court records.")
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def identifiers(row: dict[str, Any]) -> set[str]:
    values = set()
    for key in ("case_id", "source_record_id", "case_number", "citation", "docket_number"):
        value = normalized_whitespace(row.get(key))
        if value:
            values.add(value.casefold())
    return values


def cited_identifiers(row: dict[str, Any]) -> set[str]:
    text = row.get("main_opinion_text") or ""
    values = set()
    if row.get("origin_country") == "KR":
        for match in KR_CASE_RE.finditer(text):
            if match.group(2) in KR_LOWER_CODES:
                values.add(normalized_whitespace(match.group(0)).casefold())
    for key in ("history", "cross_reference"):
        value = row.get(key)
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else normalized_whitespace(value)
        for token in re.findall(r"\b\d{3,12}\b|\b[A-Z][A-Z0-9.-]{2,20}\b", rendered):
            values.add(token.casefold())
    return values


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    links_path = args.output_dir / "case_family_links.jsonl"
    qc_path = args.output_dir / "lower_court_supplement_qc.csv"
    enriched_path = args.output_dir / "candidates_with_family_links.jsonl"
    guard_outputs((links_path, qc_path, enriched_path), overwrite=args.overwrite, resume=args.resume)
    if args.resume and links_path.exists() and qc_path.exists() and enriched_path.exists():
        print(f"resume: existing family-link outputs retained at {links_path}")
        return 0
    high = [row for path in args.input for row in read_jsonl(path) if row.get("court_level") == "supreme" and row.get("court_level_confidence") == "high"]
    if args.limit:
        high = high[:args.limit]
    lower = [row for path in [*args.input, *args.lower_source] for row in read_jsonl(path) if row.get("court_level") != "supreme"]
    index: dict[str, list[dict[str, Any]]] = {}
    for row in lower:
        for value in identifiers(row):
            index.setdefault(value, []).append(row)
    links = []
    qc_rows = []
    enriched = []
    for row in high:
        need = not row.get("factual_background_sufficient")
        cited = cited_identifiers(row)
        matches = []
        evidence = []
        for value in sorted(cited):
            candidates = index.get(value, [])
            if len(candidates) == 1:
                matches.append(candidates[0])
                evidence.append(f"exact_identifier:{value}")
        unique_matches = {match["case_id"]: match for match in matches}
        selected = list(unique_matches.values())[:1] if need else []
        supplemented = bool(selected)
        lower_text = selected[0].get("main_opinion_text") or selected[0].get("full_opinion_text") or "" if selected else ""
        combined_facts = assess_fact_sufficiency(f"{row.get('main_opinion_text', '')}\n{lower_text}") if supplemented else assess_fact_sufficiency(row.get("main_opinion_text", ""))
        status = "not_needed" if not need else "supplemented" if supplemented else "not_found"
        family_id = stable_id("FAM", row["case_id"], *(match["case_id"] for match in selected), length=16)
        updated = dict(row)
        updated.update({
            "case_family_id": family_id, "highest_court_case_id": row["case_id"],
            "lower_court_case_ids": [match["case_id"] for match in selected], "lower_court_supplemented": supplemented,
            "lower_court_supplementation_status": status, "lower_court_link_confidence": "high" if supplemented else "none",
            "lower_court_link_evidence": evidence, "lower_court_fact_text": lower_text if supplemented else "",
            "fact_sufficiency_after_supplementation": combined_facts["fact_sufficiency_score"],
        })
        if need and combined_facts["factual_background_sufficient"]:
            updated["exclusion_reasons"] = [reason for reason in updated.get("exclusion_reasons", []) if reason != "fact_insufficient_before_supplementation"]
            updated["strict_source_eligible"] = not updated["exclusion_reasons"]
        links.append({key: updated[key] for key in ("case_family_id", "highest_court_case_id", "lower_court_case_ids", "lower_court_supplemented", "lower_court_link_confidence", "lower_court_link_evidence", "lower_court_supplementation_status")})
        qc_rows.append({"case_id": row["case_id"], "origin_country": row.get("origin_country"), "supplementation_needed": need, "attempted": need, "candidate_identifiers": sorted(cited), "reliably_linked": supplemented, "successfully_supplemented": supplemented and combined_facts["factual_background_sufficient"], "still_fact_insufficient": not combined_facts["factual_background_sufficient"], "status": status, "link_evidence": evidence})
        enriched.append(updated)
    print(json.dumps({"high_court_cases": len(high), "supplementation_attempted": sum(row["attempted"] for row in qc_rows), "reliably_linked": sum(row["reliably_linked"] for row in qc_rows), "successfully_supplemented": sum(row["successfully_supplemented"] for row in qc_rows), "still_fact_insufficient": sum(row["still_fact_insufficient"] for row in qc_rows)}, ensure_ascii=False, indent=2))
    if not args.dry_run:
        write_jsonl(links_path, links)
        write_csv(qc_path, qc_rows)
        write_jsonl(enriched_path, enriched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

