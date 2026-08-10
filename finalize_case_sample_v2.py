from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from pipeline_v2 import SEED, VERSION
from pipeline_v2.io_utils import guard_outputs, read_jsonl, sha256_text, write_csv, write_json, write_jsonl
from pipeline_v2.schema import DOMAIN_TARGET

DOMAINS = [*DOMAIN_TARGET, "other_civil_liability"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Select matched KR 100 + five-state U.S. 100 only after neutral-fact QC.")
    result.add_argument("--kr-input", type=Path, default=Path("outputs_v2/kr_supreme_candidates.jsonl"))
    result.add_argument("--us-input", type=Path, default=Path("outputs_v2/us_state_highcourt_candidates.jsonl"))
    result.add_argument("--facts-input", type=Path, default=Path("outputs_v2/neutral_facts_bilingual.jsonl"))
    result.add_argument("--qc-input", type=Path, default=Path("outputs_v2/neutral_fact_qc.csv"))
    result.add_argument("--states-from", type=Path, default=Path("outputs_v2/us_state_selection.json"))
    result.add_argument("--kr-target", type=int, default=100)
    result.add_argument("--us-target", type=int, default=100)
    result.add_argument("--seed", type=int, default=SEED)
    result.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def read_qc(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def deterministic_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: sha256_text(f"{seed}|{row['case_id']}"))


def dedupe_families(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for row in deterministic_order(rows, seed):
        family = row.get("case_family_id") or row["case_id"]
        if family not in seen:
            seen.add(family)
            result.append(row)
    return result


def max_flow_counts(states: list[str], availability: dict[tuple[str, str], int], domain_caps: dict[str, int]) -> tuple[int, dict[tuple[str, str], int]]:
    source, sink = "@source", "@sink"
    capacity: dict[tuple[str, str], int] = {}
    adjacency: dict[str, list[str]] = defaultdict(list)
    def edge(a: str, b: str, cap: int) -> None:
        capacity[a, b] = cap
        capacity.setdefault((b, a), 0)
        adjacency[a].append(b); adjacency[b].append(a)
    for state in states:
        edge(source, f"S:{state}", 20)
        for domain in DOMAINS:
            edge(f"S:{state}", f"D:{domain}", availability.get((state, domain), 0))
    for domain in DOMAINS:
        edge(f"D:{domain}", sink, domain_caps.get(domain, 0))
    flow: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt not in parent and capacity[node, nxt] - flow[node, nxt] > 0:
                    parent[nxt] = node; queue.append(nxt)
        if sink not in parent:
            break
        amount = 10**9
        node = sink
        while parent[node] is not None:
            amount = min(amount, capacity[parent[node], node] - flow[parent[node], node]); node = parent[node]
        node = sink
        while parent[node] is not None:
            flow[parent[node], node] += amount; flow[node, parent[node]] -= amount; node = parent[node]
        total += amount
    counts = {(state, domain): flow[f"S:{state}", f"D:{domain}"] for state in states for domain in DOMAINS}
    return total, counts


def choose_allocation(kr: list[dict[str, Any]], us: list[dict[str, Any]], states: list[str]) -> tuple[dict[str, int] | None, dict[tuple[str, str], int], dict[str, Any]]:
    kr_available = Counter(row["case_domain"] for row in kr)
    us_available = Counter(row["case_domain"] for row in us)
    availability = Counter((row["origin_state"], row["case_domain"]) for row in us)
    common_caps = {domain: min(kr_available[domain], us_available[domain]) for domain in DOMAINS}
    desired = {**DOMAIN_TARGET, "other_civil_liability": 0}
    flow, counts = max_flow_counts(states, availability, desired)
    deviation = False
    if flow < 100:
        flow, counts = max_flow_counts(states, availability, common_caps)
        deviation = True
    allocation = {domain: sum(counts[state, domain] for state in states) for domain in DOMAINS}
    report = {"kr_available": dict(kr_available), "us_available": dict(us_available), "shared_caps": common_caps, "initial_target": desired, "revised_allocation_used": deviation, "max_flow": flow, "allocation": allocation}
    return (allocation if flow == 100 else None), counts, report


def assign_splits(kr: list[dict[str, Any]], us: list[dict[str, Any]], seed: int) -> None:
    kr_by_domain = defaultdict(list)
    for row in kr: kr_by_domain[row["case_domain"]].append(row)
    dev_counts = {domain: len(rows) // 5 for domain, rows in kr_by_domain.items()}
    while sum(dev_counts.values()) < 20:
        domain = max(DOMAINS, key=lambda value: (len(kr_by_domain[value]) / 5 - dev_counts.get(value, 0), -DOMAINS.index(value)))
        dev_counts[domain] = dev_counts.get(domain, 0) + 1
    for domain, rows in kr_by_domain.items():
        ordered = deterministic_order(rows, seed + 1)
        dev_ids = {row["case_id"] for row in ordered[:dev_counts.get(domain, 0)]}
        for row in rows: row["analysis_split"] = "development" if row["case_id"] in dev_ids else "confirmatory"
    us_by_state = defaultdict(list)
    for row in us: us_by_state[row["origin_state"]].append(row)
    for state, rows in us_by_state.items():
        ordered = deterministic_order(rows, seed + 2)
        # Four per state preserves exact five-state balance in development.
        dev_ids = {row["case_id"] for row in ordered[:4]}
        for row in rows: row["analysis_split"] = "development" if row["case_id"] in dev_ids else "confirmatory"


def manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    facts = row["neutral_facts"]
    qc = row["neutral_qc"]
    source = facts["neutral_fact_source"]; ko = facts["neutral_fact_ko"]; en = facts["neutral_fact_en"]
    return {
        "case_id": row["case_id"], "case_family_id": row["case_family_id"], "origin_country": row["origin_country"], "origin_state": row.get("origin_state"),
        "court_name": row.get("court_name"), "court_level": row.get("court_level"), "court_level_confidence": row.get("court_level_confidence"),
        "decision_date": row.get("decision_date"), "case_number_or_citation": row.get("case_number") or row.get("citation"), "case_domain": row["case_domain"], "analysis_split": row["analysis_split"],
        "main_source_case_id": row["case_id"], "lower_court_supplemented": row.get("lower_court_supplemented", False), "lower_court_case_ids": row.get("lower_court_case_ids", []),
        "fact_sufficiency_score": row.get("fact_sufficiency_after_supplementation", row.get("fact_sufficiency_score")),
        "neutral_fact_source_language": facts["source_language"], "neutral_fact_source_chars": len(source), "neutral_fact_ko_chars": len(ko), "neutral_fact_en_chars": len(en),
        "jurisdiction_salience_level": "low", "jurisdiction_salience_tags": [], "remedy_salience_level": "low",
        "legal_leakage_status": qc["legal_leakage_status"], "jurisdiction_leakage_status": qc["jurisdiction_leakage_status"],
        "translation_equivalence_status": qc["translation_equivalence_status"], "source_grounding_status": qc["source_grounding_status"], "final_eligible": True,
        "raw_text_sha256": row["raw_text_sha256"], "neutral_fact_source_sha256": sha256_text(source), "neutral_fact_ko_sha256": sha256_text(ko), "neutral_fact_en_sha256": sha256_text(en),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    names = ("final_kr_cases_100.jsonl", "final_us_cases_100.jsonl", "final_cases_200.jsonl", "final_fact_patterns_200.jsonl", "final_fact_units_200.jsonl", "final_manifest.csv", "collection_summary.json")
    paths = [args.output_dir / name for name in names]
    guard_outputs(paths, overwrite=args.overwrite)
    if (args.kr_target, args.us_target) != (100, 100):
        raise ValueError("The v2 design requires exactly --kr-target 100 --us-target 100")
    facts = {row["case_id"]: row for row in read_jsonl(args.facts_input)}
    qc = read_qc(args.qc_input)
    def eligible(path: Path, country: str) -> list[dict[str, Any]]:
        output = []
        for row in read_jsonl(path):
            q = qc.get(row["case_id"]); fact = facts.get(row["case_id"])
            if row.get("origin_country") == country and row.get("strict_source_eligible") is True and q and q.get("final_eligible", "").lower() == "true" and fact:
                row["case_family_id"] = fact.get("case_family_id") or row.get("case_family_id") or row["case_id"]
                for key in ("highest_court_case_id", "lower_court_supplemented", "lower_court_case_ids", "lower_court_link_confidence"):
                    if key in fact:
                        row[key] = fact[key]
                row["neutral_facts"] = fact; row["neutral_qc"] = q; output.append(row)
        return dedupe_families(output, args.seed)
    kr, us = eligible(args.kr_input, "KR"), eligible(args.us_input, "US")
    selection = json.loads(args.states_from.read_text(encoding="utf-8"))
    states = [entry["state"] for entry in selection.get("selected_states", [])]
    allocation, state_domain_counts, allocation_report = choose_allocation(kr, us, states) if len(states) == 5 else (None, {}, {"error": "not_five_frozen_states"})
    summary: dict[str, Any] = {"collection_version": VERSION, "status": "shortfall", "seed": args.seed, "eligible_pool": {"KR": len(kr), "US": len(us)}, "allocation_report": allocation_report, "frozen_states": states}
    deterministic = {}
    for country, name in (("KR", "kr_collection_deterministic_summary.json"), ("US", "us_collection_deterministic_summary.json")):
        path = args.output_dir / name
        if path.exists():
            deterministic[country] = json.loads(path.read_text(encoding="utf-8"))
    summary["deterministic_collection"] = deterministic
    if allocation is None or len(kr) < 100 or len(us) < 100:
        summary["shortfall"] = {"KR": max(0, 100 - len(kr)), "US": max(0, 100 - len(us)), "matched_state_domain_flow_shortfall": 100 - allocation_report.get("max_flow", 0)}
        summary["sanity_checks"] = {
            "len_final_KR_100": False, "len_final_US_100": False,
            "all_KR_supreme": False, "all_US_court_type_S": False,
            "all_dates_in_window": False, "unique_case_id": False,
            "unique_family_within_country": False, "all_qc_pass": False,
            "all_bilingual_nonempty": False, "us_state_counts_20_each": False,
            "matched_domain_counts": False, "split_20_80_each": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not args.dry_run:
            write_json(args.output_dir / "collection_summary.json", summary)
        return 2
    chosen_us = []
    for state in states:
        for domain in DOMAINS:
            pool = [row for row in us if row["origin_state"] == state and row["case_domain"] == domain]
            chosen_us.extend(deterministic_order(pool, args.seed)[:state_domain_counts[state, domain]])
    chosen_kr = []
    for domain in DOMAINS:
        pool = [row for row in kr if row["case_domain"] == domain]
        chosen_kr.extend(deterministic_order(pool, args.seed)[:allocation[domain]])
    assign_splits(chosen_kr, chosen_us, args.seed)
    combined = chosen_kr + chosen_us
    manifest = [manifest_row(row) for row in combined]
    checks = {
        "len_final_KR_100": len(chosen_kr) == 100, "len_final_US_100": len(chosen_us) == 100,
        "all_KR_supreme": all(row["court_level"] == "supreme" and row["court_level_confidence"] == "high" for row in chosen_kr),
        "all_US_court_type_S": all(row.get("court_type") == "S" for row in chosen_us),
        "all_dates_in_window": all("2000-01-01" <= row["decision_date"] <= "2025-12-31" for row in combined),
        "unique_case_id": len({row["case_id"] for row in combined}) == 200,
        "unique_family_within_country": all(len({row["case_family_id"] for row in group}) == 100 for group in (chosen_kr, chosen_us)),
        "all_qc_pass": all(all(item[key] == "pass" for key in ("source_grounding_status", "legal_leakage_status", "jurisdiction_leakage_status", "translation_equivalence_status")) for item in manifest),
        "all_bilingual_nonempty": all(row["neutral_facts"]["neutral_fact_ko"] and row["neutral_facts"]["neutral_fact_en"] for row in combined),
        "us_state_counts_20_each": Counter(row["origin_state"] for row in chosen_us) == Counter({state: 20 for state in states}),
        "matched_domain_counts": Counter(row["case_domain"] for row in chosen_kr) == Counter(row["case_domain"] for row in chosen_us),
        "split_20_80_each": all(Counter(row["analysis_split"] for row in group) == Counter({"development": 20, "confirmatory": 80}) for group in (chosen_kr, chosen_us)),
    }
    summary.update({"status": "complete" if all(checks.values()) else "invariant_failure", "final_counts": {"KR": len(chosen_kr), "US": len(chosen_us)}, "state_counts": dict(Counter(row["origin_state"] for row in chosen_us)), "domain_counts": {"KR": dict(Counter(row["case_domain"] for row in chosen_kr)), "US": dict(Counter(row["case_domain"] for row in chosen_us))}, "analysis_splits": {"KR": dict(Counter(row["analysis_split"] for row in chosen_kr)), "US": dict(Counter(row["analysis_split"] for row in chosen_us))}, "sanity_checks": checks})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0 if all(checks.values()) else 2
    public_case_rows = [{key: value for key, value in row.items() if key not in {"neutral_facts", "neutral_qc"}} for row in combined]
    fact_patterns = [{"case_id": row["case_id"], "origin_country": row["origin_country"], "origin_state": row.get("origin_state"), "case_domain": row["case_domain"], "analysis_split": row["analysis_split"], **{key: row["neutral_facts"][key] for key in ("source_language", "neutral_fact_source", "neutral_fact_ko", "neutral_fact_en")}} for row in combined]
    fact_units = [{"case_id": row["case_id"], "origin_country": row["origin_country"], **unit} for row in combined for unit in row["neutral_facts"]["aligned_fact_units"]]
    write_jsonl(args.output_dir / "final_kr_cases_100.jsonl", public_case_rows[:100])
    write_jsonl(args.output_dir / "final_us_cases_100.jsonl", public_case_rows[100:])
    write_jsonl(args.output_dir / "final_cases_200.jsonl", public_case_rows)
    write_jsonl(args.output_dir / "final_fact_patterns_200.jsonl", fact_patterns)
    write_jsonl(args.output_dir / "final_fact_units_200.jsonl", fact_units)
    write_csv(args.output_dir / "final_manifest.csv", manifest)
    write_json(args.output_dir / "collection_summary.json", summary)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
