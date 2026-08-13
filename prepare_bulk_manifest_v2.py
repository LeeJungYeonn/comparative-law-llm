from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from finalize_case_sample_v2 import DOMAINS, choose_allocation, dedupe_families, deterministic_order
from pipeline_v2 import SEED, VERSION
from pipeline_v2.io_utils import guard_outputs, read_jsonl, write_json, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the deterministic pre-LLM 100+100 extraction manifest.")
    parser.add_argument("--input", type=Path, default=Path("outputs_v2/candidates_with_family_links.jsonl"))
    parser.add_argument("--states-from", type=Path, default=Path("outputs_v2/us_state_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs_v2/bulk_extraction_manifest_200.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs_v2/bulk_extraction_manifest_summary.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    guard_outputs((args.output, args.summary), overwrite=args.overwrite)

    all_rows = list(read_jsonl(args.input))
    strict = [row for row in all_rows if row.get("strict_source_eligible") is True]
    kr = dedupe_families([row for row in strict if row.get("origin_country") == "KR"], args.seed)
    us = dedupe_families([row for row in strict if row.get("origin_country") == "US"], args.seed)
    state_payload = json.loads(args.states_from.read_text(encoding="utf-8"))
    states = [entry["state"] for entry in state_payload.get("selected_states", [])]
    allocation, state_domain_counts, report = choose_allocation(kr, us, states)
    if allocation is None or len(states) != 5:
        raise RuntimeError("No feasible frozen five-state matched allocation")

    chosen_kr: list[dict[str, Any]] = []
    for domain in DOMAINS:
        pool = [row for row in kr if (row.get("primary_domain") or row.get("case_domain")) == domain]
        chosen_kr.extend(deterministic_order(pool, args.seed)[: allocation[domain]])
    chosen_us: list[dict[str, Any]] = []
    for state in states:
        for domain in DOMAINS:
            pool = [
                row for row in us
                if row.get("origin_state") == state and (row.get("primary_domain") or row.get("case_domain")) == domain
            ]
            chosen_us.extend(deterministic_order(pool, args.seed)[: state_domain_counts[state, domain]])
    combined = chosen_kr + chosen_us
    if len(chosen_kr) != 100 or len(chosen_us) != 100 or len({row["case_id"] for row in combined}) != 200:
        raise RuntimeError("Frozen manifest invariant failed")
    summary = {
        "collection_version": VERSION,
        "stage": "pre_llm_manifest_frozen_after_full_source_audit",
        "seed": args.seed,
        "candidate_pool": {"KR": len(kr), "US": len(us)},
        "matched_allocation": allocation,
        "states": states,
        "US_state_domain_flow": {
            state: {domain: state_domain_counts[state, domain] for domain in DOMAINS} for state in states
        },
        "manifest_counts": {"KR": len(chosen_kr), "US": len(chosen_us)},
        "manifest_state_counts": dict(Counter(row.get("origin_state") for row in chosen_us)),
        "manifest_domain_counts": {
            "KR": dict(Counter(row.get("primary_domain") or row.get("case_domain") for row in chosen_kr)),
            "US": dict(Counter(row.get("primary_domain") or row.get("case_domain") for row in chosen_us)),
        },
        "allocation_report": report,
    }
    write_jsonl(args.output, combined)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
