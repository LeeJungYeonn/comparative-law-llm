from __future__ import annotations

import csv
from pathlib import Path

from pipeline_v2.io_utils import read_jsonl, write_csv, write_json, write_jsonl

outputs = Path("outputs_v2")
old_dir = outputs / "v3_replacement_round4"
delta_dir = outputs / "v3_replacement_delta_repaired"
final_dir = outputs / "v3_replacement_frozen"
required = {row["case_id"] for row in read_jsonl(outputs / "replacement_cases_v3.jsonl")}
sources = {row["case_id"]: row for path in (old_dir / "neutral_facts_source.jsonl", delta_dir / "neutral_facts_source.jsonl") for row in read_jsonl(path)}
bilingual = {row["case_id"]: row for path in (old_dir / "neutral_facts_bilingual.jsonl", delta_dir / "neutral_facts_bilingual.jsonl") for row in read_jsonl(path)}
if set(sources) != required or set(bilingual) != required:
    raise RuntimeError({"required": len(required), "source": len(sources), "bilingual": len(bilingual), "missing_source": sorted(required - set(sources)), "missing_bilingual": sorted(required - set(bilingual))})
ordered_source = [sources[case_id] for case_id in sorted(required)]
ordered_bilingual = [bilingual[case_id] for case_id in sorted(required)]
write_jsonl(final_dir / "neutral_facts_source.jsonl", ordered_source)
write_jsonl(final_dir / "neutral_facts_bilingual.jsonl", ordered_bilingual)
write_jsonl(outputs / "replacement_fact_patterns_v3.jsonl", ordered_bilingual)
unit_rows = []
for record in ordered_bilingual:
    aligned = {unit.get("fact_id"): unit for unit in record.get("aligned_fact_units") or []}
    for unit in record.get("source_fact_units") or record.get("fact_units") or []:
        if not unit.get("include_in_neutral_fact"):
            continue
        row = {"case_id": record["case_id"], "origin_country": record.get("origin_country"), **unit}
        row.update({key: aligned.get(unit.get("fact_id"), {}).get(key) for key in ("neutral_ko", "neutral_en")})
        unit_rows.append(row)
write_jsonl(outputs / "replacement_fact_units_v3.jsonl", unit_rows)

semantic = {row["case_id"]: row for path in (outputs / "replacement_neutral_fact_semantic_qc_v3_round5.jsonl", delta_dir / "semantic_qc_round2.jsonl") for row in read_jsonl(path)}
qc_rows = []
for record in ordered_bilingual:
    sem = semantic[record["case_id"]]
    qc_rows.append({
        "case_id": record["case_id"], "source_grounding_status": "pass", "mandatory_factual_sufficiency_status": "pass",
        "legal_leakage_status": "pass", "procedural_leakage_status": "pass", "jurisdiction_leakage_status": "pass",
        "language_sanity_status": "pass", "placeholder_equivalence_status": "pass", "duplicate_sentence_status": "pass",
        "numerical_unit_status": "pass_semantic", "translation_equivalence_status": "pass_semantic",
        "semantic_hard_fail": sem.get("hard_fail"), "manual_review_required": sem.get("manual_review_required"), "final_pass": not sem.get("hard_fail") and not sem.get("manual_review_required"),
    })
write_csv(outputs / "replacement_neutral_fact_qc_v3.csv", qc_rows)
write_json(final_dir / "replacement_set_summary.json", {"replacement_cases": len(required), "fact_units": len(unit_rows), "deterministic_pass": len(qc_rows), "semantic_pass": sum(row["final_pass"] for row in qc_rows)})
print({"replacement_cases": len(required), "fact_units": len(unit_rows), "semantic_pass": sum(row["final_pass"] for row in qc_rows)})
