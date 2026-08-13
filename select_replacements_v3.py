from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl
from pipeline_v2.v3_rules import DOMAINS

SEED = 20260810
STATES = ("Pennsylvania", "Michigan", "Louisiana", "Nevada", "West Virginia")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select the minimum-change v3 roster with matched domains and bounded states.")
    p.add_argument("--outputs", type=Path, default=Path("outputs_v2"))
    p.add_argument("--overwrite", action="store_true")
    return p


def stable_tie(case_id: str) -> float:
    value = int(hashlib.sha256(f"{SEED}:{case_id}".encode()).hexdigest()[:8], 16)
    return value / 2**32


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    selection_path = args.outputs / "replacement_selection_v3.json"
    provisional_path = args.outputs / "provisional_final_cases_200_v3.jsonl"
    replacement_cases_path = args.outputs / "replacement_cases_v3.jsonl"
    if not args.overwrite and (selection_path.exists() or provisional_path.exists()):
        raise FileExistsError("v3 selection outputs exist; pass --overwrite")

    original = list(read_jsonl(args.outputs / "final_cases_200.jsonl"))
    original_by_id = {row["case_id"]: row for row in original}
    review = {row["case_id"]: row for row in read_jsonl(args.outputs / "domain_reclassification_v3.jsonl")}
    audit_bad = {row["case_id"] for row in read_jsonl(args.outputs / "neutral_fact_qc_audit_200.jsonl") if row.get("case_level_status") == "replacement_required"}

    case_sources: dict[str, dict[str, Any]] = {}
    candidate_reviews: dict[str, dict[str, Any]] = {}
    candidate_pairs = (
        (args.outputs / "kr_replacement_candidates_review_v3.jsonl", args.outputs / "kr_candidate_domain_reclassification_v3.jsonl"),
        (args.outputs / "us_replacement_candidates_review_v3.jsonl", args.outputs / "us_candidate_domain_reclassification_v3.jsonl"),
        (args.outputs / "us_replacement_candidates_review_extra_v3.jsonl", args.outputs / "us_candidate_domain_reclassification_extra_v3.jsonl"),
        (args.outputs / "us_replacement_candidates_review_remaining_v3.jsonl", args.outputs / "us_candidate_domain_reclassification_remaining_v3.jsonl"),
    )
    for source_path, review_path in candidate_pairs:
        case_sources.update({row["case_id"]: row for row in read_jsonl(source_path)})
        candidate_reviews.update({row["case_id"]: row for row in read_jsonl(review_path)})

    choices: list[dict[str, Any]] = []
    for row in original:
        decision = review[row["case_id"]]
        if decision.get("eligible_main_corpus") is True and row["case_id"] not in audit_bad:
            choices.append({"kind": "existing", "source": row, "review": decision})
    for case_id, decision in candidate_reviews.items():
        source = case_sources[case_id]
        if decision.get("eligible_main_corpus") is True and source.get("core_fact_sufficient") is True:
            choices.append({"kind": "candidate", "source": source, "review": decision})

    n = len(choices)
    objective = np.zeros(n)
    for i, item in enumerate(choices):
        if item["kind"] == "existing":
            objective[i] = -1_000_000 + stable_tie(item["source"]["case_id"])
        else:
            score = int(item["source"].get("fact_sufficiency_score") or 0)
            confidence_cost = 0 if item["review"].get("confidence") == "high" else 10_000
            objective[i] = confidence_cost + (7 - score) * 1_000 + stable_tie(item["source"]["case_id"])

    rows: list[np.ndarray] = []
    lows: list[float] = []
    highs: list[float] = []

    def constraint(predicate, low: float, high: float, sign=lambda _: 1.0) -> None:
        vector = np.array([sign(item) if predicate(item) else 0.0 for item in choices])
        rows.append(vector); lows.append(low); highs.append(high)

    for country in ("KR", "US"):
        constraint(lambda item, c=country: item["source"].get("origin_country") == c, 100, 100)
    for domain in DOMAINS:
        vector = np.array([
            (1.0 if item["source"].get("origin_country") == "KR" else -1.0)
            if item["review"].get("primary_domain") == domain else 0.0 for item in choices
        ])
        rows.append(vector); lows.append(0); highs.append(0)
    for state in STATES:
        constraint(lambda item, s=state: item["source"].get("origin_country") == "US" and item["source"].get("origin_state") == s, 10, 30)
    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, item in enumerate(choices):
        key = (item["source"].get("origin_country") or "", item["source"].get("case_family_id") or item["source"]["case_id"])
        families[key].append(i)
    for indices in families.values():
        if len(indices) > 1:
            vector = np.zeros(n); vector[indices] = 1
            rows.append(vector); lows.append(0); highs.append(1)

    result = milp(
        c=objective, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lows), np.array(highs)),
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        write_json(selection_path, {"status": "infeasible", "message": result.message, "seed": SEED})
        print({"status": "infeasible", "message": result.message})
        return 2
    selected = [item for item, value in zip(choices, result.x) if value >= 0.5]
    selected_ids = {item["source"]["case_id"] for item in selected}
    kept_existing = [item for item in selected if item["kind"] == "existing"]
    selected_candidates = [item for item in selected if item["kind"] == "candidate"]
    invalid_original = [row for row in original if row["case_id"] in audit_bad or not review[row["case_id"]].get("eligible_main_corpus")]
    balancing_removed = [row for row in original if review[row["case_id"]].get("eligible_main_corpus") and row["case_id"] not in audit_bad and row["case_id"] not in selected_ids]

    final_rows: list[dict[str, Any]] = []
    for item in sorted(selected, key=lambda x: (x["source"].get("origin_country") or "", x["source"]["case_id"])):
        row = dict(item["source"])
        decision = item["review"]
        row.update({
            "corpus_version": "kr-us-highcourt-corpus-v3.0", "primary_domain": decision["primary_domain"],
            "case_domain": decision["primary_domain"], "liability_theories": decision.get("liability_theories") or [],
            "eligible_main_corpus": True, "domain_review_status": decision["domain_review_status"],
            "domain_evidence": decision.get("domain_evidence_spans") or [],
            "eligibility_evidence": decision.get("eligibility_evidence_spans") or [],
            "replacement_status": "retained" if item["kind"] == "existing" else "replacement_v3",
            "analysis_split": None,
        })
        final_rows.append(row)

    counts = lambda rows_, key: dict(sorted(Counter(row.get(key) for row in rows_).items(), key=lambda pair: str(pair[0])))
    kr_domains = counts([row for row in final_rows if row["origin_country"] == "KR"], "primary_domain")
    us_domains = counts([row for row in final_rows if row["origin_country"] == "US"], "primary_domain")
    state_counts = counts([row for row in final_rows if row["origin_country"] == "US"], "origin_state")
    invariants = {
        "total_200": len(final_rows) == 200,
        "kr_100": sum(row["origin_country"] == "KR" for row in final_rows) == 100,
        "us_100": sum(row["origin_country"] == "US" for row in final_rows) == 100,
        "unique_case_id": len({row["case_id"] for row in final_rows}) == 200,
        "unique_family_within_country": len({(row["origin_country"], row.get("case_family_id")) for row in final_rows}) == 200,
        "domain_match": kr_domains == us_domains,
        "states_bounded": set(state_counts) == set(STATES) and all(10 <= value <= 30 for value in state_counts.values()),
        "all_domain_resolved": all(row.get("domain_review_status") != "unresolved" for row in final_rows),
        "all_source_eligible": all(row.get("eligible_main_corpus") is True for row in final_rows),
        "all_dates_eligible": all("2000-01-01" <= (row.get("decision_date") or "") <= "2025-12-31" for row in final_rows),
        "all_candidate_core_fact_sufficient": all(item["source"].get("core_fact_sufficient") is True for item in selected_candidates),
    }
    if not all(invariants.values()):
        write_json(selection_path, {"status": "not_frozen", "invariants": invariants, "seed": SEED})
        print({"status": "not_frozen", "invariants": invariants})
        return 2

    removed = [*invalid_original, *balancing_removed]
    candidate_by_country = defaultdict(list)
    for item in selected_candidates:
        candidate_by_country[item["source"]["origin_country"]].append(item["source"])
    removed_by_country = defaultdict(list)
    for row in removed:
        removed_by_country[row["origin_country"]].append(row)
    mappings = []
    for country in ("KR", "US"):
        old_rows = sorted(removed_by_country[country], key=lambda row: (row.get("origin_state") or "", row["case_id"]))
        new_rows = sorted(candidate_by_country[country], key=lambda row: (row.get("origin_state") or "", row["case_id"]))
        for old, new in zip(old_rows, new_rows):
            mappings.append({
                "origin_country": country, "old_case_id": old["case_id"], "old_state": old.get("origin_state"),
                "new_case_id": new["case_id"], "new_state": new.get("origin_state"),
                "reason_category": "original_audit_flag" if old["case_id"] in audit_bad else ("additional_source_ineligible" if not review[old["case_id"]].get("eligible_main_corpus") else "additional_balancing_swap"),
                "reason": ("confirmed replacement required by external review and source recheck" if old["case_id"] in audit_bad else review[old["case_id"]].get("exclusion_reason")) or "minimum balancing swap required for exact KR-US domain equality",
            })
    selection = {
        "status": "provisional_roster_frozen", "corpus_version": "kr-us-highcourt-corpus-v3.0", "seed": SEED,
        "original_audit_flagged_replacements": sorted(row["case_id"] for row in invalid_original if row["case_id"] in audit_bad),
        "additional_source_ineligible_replacements": sorted(row["case_id"] for row in invalid_original if row["case_id"] not in audit_bad),
        "additional_balancing_swaps": sorted(row["case_id"] for row in balancing_removed),
        "replacement_set_v3": sorted(item["source"]["case_id"] for item in selected_candidates),
        "replacement_count": len(selected_candidates), "mappings": mappings,
        "domain_counts": {"KR": kr_domains, "US": us_domains}, "state_counts": state_counts,
        "infeasibility_proof_without_balancing_swaps": {
            "valid_kr_product_cases": sum(item["review"]["primary_domain"] == "product_liability" for item in choices if item["kind"] == "existing" and item["source"]["origin_country"] == "KR"),
            "valid_us_product_cases": sum(item["review"]["primary_domain"] == "product_liability" for item in choices if item["kind"] == "existing" and item["source"]["origin_country"] == "US"),
            "mandatory_sufficient_us_product_candidates": sum(item["review"]["primary_domain"] == "product_liability" for item in choices if item["kind"] == "candidate" and item["source"]["origin_country"] == "US"),
            "minimum_balancing_swaps": len(balancing_removed),
        },
        "invariants": invariants,
    }
    write_json(selection_path, selection)
    write_jsonl(provisional_path, final_rows)
    write_jsonl(replacement_cases_path, [row for row in final_rows if row.get("replacement_status") == "replacement_v3"])
    print({"status": selection["status"], "replacement_count": len(selected_candidates), "balancing_swaps": len(balancing_removed), "domains": kr_domains, "states": state_counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
