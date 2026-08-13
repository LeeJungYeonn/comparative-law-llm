from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from finalize_case_sample_v2 import choose_allocation, dedupe_families
from pipeline_v2 import SEED, VERSION
from pipeline_v2.io_utils import guard_outputs, read_jsonl, write_json
from pipeline_v2.schema import PRIMARY_DOMAINS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess pre-LLM 100+100 feasibility from strict source candidates.")
    parser.add_argument("--kr-input", type=Path, default=Path("outputs_v2/kr_supreme_candidates.jsonl"))
    parser.add_argument("--us-input", type=Path, default=Path("outputs_v2/us_state_highcourt_candidates.jsonl"))
    parser.add_argument("--states-from", type=Path, default=Path("outputs_v2/us_state_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs_v2/pre_llm_feasibility.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    guard_outputs((args.output,), overwrite=args.overwrite)
    kr_all = list(read_jsonl(args.kr_input))
    us_all = list(read_jsonl(args.us_input))
    kr = dedupe_families([row for row in kr_all if row.get("strict_source_eligible") is True], args.seed)
    us = dedupe_families([row for row in us_all if row.get("strict_source_eligible") is True], args.seed)
    states = [entry["state"] for entry in json.loads(args.states_from.read_text(encoding="utf-8")).get("selected_states", [])]
    allocation, state_domain_flow, report = choose_allocation(kr, us, states)
    state_counts = Counter(row.get("origin_state") for row in us)
    payload = {
        "collection_version": VERSION, "assessment_stage": "after deterministic source eligibility; before LLM extraction",
        "hard_constraints_checked": ["KR structured Supreme Court/date", "US court_type=S/date", "substantive centrality", "controlling opinion", "core_fact_sufficient", "deduplicated case_family"],
        "hard_constraints_not_yet_checked": ["source-grounded fact-unit extraction", "legal/jurisdiction leakage", "KO/EN translation equivalence"],
        "candidate_counts": {"KR_all": len(kr_all), "US_all": len(us_all), "KR_strict_deduped": len(kr), "US_strict_deduped": len(us)},
        "eligible_by_primary_domain": {
            "KR": {domain: sum((row.get("primary_domain") or row.get("case_domain")) == domain for row in kr) for domain in PRIMARY_DOMAINS},
            "US": {domain: sum((row.get("primary_domain") or row.get("case_domain")) == domain for row in us) for domain in PRIMARY_DOMAINS},
        },
        "US_eligible_by_state": {state: state_counts[state] for state in states},
        "matched_allocation": allocation, "US_state_domain_flow": {
            state: {domain: state_domain_flow.get((state, domain), 0) for domain in PRIMARY_DOMAINS} for state in states
        },
        "allocation_report": report,
        "pre_llm_100_plus_100_source_feasible": len(kr) >= 100 and len(us) >= 100 and allocation is not None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    write_json(args.output, payload)
    return 0 if payload["pre_llm_100_plus_100_source_feasible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
