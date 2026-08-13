from pathlib import Path

from pipeline_v2.io_utils import read_jsonl, write_jsonl

prior = {row["case_id"] for row in read_jsonl(Path("outputs_v2/replacement_cases_v3_prior.jsonl"))}
current = list(read_jsonl(Path("outputs_v2/replacement_cases_v3.jsonl")))
delta = [row for row in current if row["case_id"] not in prior]
write_jsonl(Path("outputs_v2/replacement_cases_delta_v3.jsonl"), delta)
print({"prior": len(prior), "current": len(current), "delta": len(delta), "case_ids": [row["case_id"] for row in delta]})
