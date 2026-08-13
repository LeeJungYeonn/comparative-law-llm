from pathlib import Path

from pipeline_v2.io_utils import read_jsonl, write_jsonl

source_dir = Path("outputs_v2/v3_replacement_delta")
output_dir = Path("outputs_v2/v3_replacement_delta_repaired")
rows = list(read_jsonl(source_dir / "neutral_facts_source.jsonl"))
prior = {row["case_id"]: row for row in read_jsonl(source_dir / "neutral_facts_bilingual.jsonl")}
target = "US_f5ad9cb96e57eb50d6"
amendments = []
for row in rows:
    if row["case_id"] != target:
        continue
    for unit in row["fact_units"]:
        if unit.get("fact_id") == "F002":
            old = unit["text"]
            new = "[PERSON_A] stepped into a depressed area of the sidewalk and encountered raised, uneven bricks next to the depression."
            unit["text"] = new
            unit["text_before_direct_adjudication"] = old
            amendments.append({"case_id": target, "field_changed": "fact_units.F002.text", "old_text": old, "new_text": new, "source_evidence": unit.get("source_span"), "reason": "removed causal-result duplication while retaining conduct and surface condition", "review_stage": "delta_direct_source_adjudication_v3"})
        if unit.get("fact_id") == "F005":
            old = unit["text"]
            new = "The raised portion of the sidewalk was less than 5.08 centimeters."
            unit["text"] = new
            unit["text_before_direct_adjudication"] = old
            amendments.append({"case_id": target, "field_changed": "fact_units.F005.text", "old_text": old, "new_text": new, "source_evidence": unit.get("source_span"), "reason": "removed duplicated sidewalk-maintenance clause", "review_stage": "delta_direct_source_adjudication_v3"})
    row["neutral_fact_source"] = " ".join(unit["text"].strip() for unit in row["fact_units"] if unit.get("include_in_neutral_fact"))
    row["neutral_fact_en"] = row["neutral_fact_source"]
    row["neutral_fact_ko"] = ""
write_jsonl(output_dir / "neutral_facts_source.jsonl", rows)
write_jsonl(output_dir / "neutral_facts_bilingual.jsonl", [prior[row["case_id"]] for row in rows if row["case_id"] != target])
write_jsonl(output_dir / "replacement_amendments_v3.jsonl", amendments)
print({"changed": target, "seeded": 1})
