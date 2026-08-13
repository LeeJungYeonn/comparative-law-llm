from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_jsonl
from pipeline_v2.v3_rules import PRODUCT_DEFECT

SEED = 20260810


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare deterministic blinded-review candidate batches without recollection.")
    p.add_argument("--outputs", type=Path, default=Path("outputs_v2"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--extra-us", action="store_true", help="Prepare an additional unreviewed US batch after the first candidate review.")
    p.add_argument("--product-rescue", action="store_true", help="Prepare remaining product-signal records for feasibility proof.")
    p.add_argument("--remaining-preferred", action="store_true", help="Prepare every still-unreviewed preferred US candidate.")
    return p


def rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (-int(row.get("fact_sufficiency_score") or 0), -int(row.get("raw_text_chars") or 0), row["case_id"])


def take(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return sorted(rows, key=rank)[:count]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    kr_out, us_out = args.outputs / "kr_replacement_candidates_review_v3.jsonl", args.outputs / "us_replacement_candidates_review_v3.jsonl"
    if args.extra_us:
        reviewed = {row["case_id"] for row in read_jsonl(us_out)}
        final = list(read_jsonl(args.outputs / "final_cases_200.jsonl"))
        ids = {row["case_id"] for row in final} | reviewed
        families = {(row["origin_country"], row.get("case_family_id")) for row in final}
        pool = [row for row in read_jsonl(args.outputs / "us_state_highcourt_candidates.jsonl") if row["case_id"] not in ids and ("US", row.get("case_family_id")) not in families and row.get("preferred_fact_sufficiency") is True]
        product_signal = [row for row in pool if PRODUCT_DEFECT.search(" ".join(str(row.get(key) or "") for key in ("case_name", "nature_of_suit", "main_opinion_text")))]
        chosen = take(product_signal, 30)
        chosen_ids = {row["case_id"] for row in chosen}
        remainder = [row for row in pool if row["case_id"] not in chosen_ids]
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(remainder, key=rank):
            by_state[row.get("origin_state") or ""].append(row)
        states = sorted(by_state)
        while len(chosen) < min(80, len(pool)) and any(by_state.values()):
            for state in states:
                if by_state[state] and len(chosen) < min(80, len(pool)):
                    chosen.append(by_state[state].pop(0))
        extra_out = args.outputs / "us_replacement_candidates_review_extra_v3.jsonl"
        if extra_out.exists() and not args.overwrite:
            raise FileExistsError(f"{extra_out} exists; pass --overwrite")
        write_jsonl(extra_out, sorted(chosen, key=lambda row: row["case_id"]))
        print({"seed": SEED, "remaining_pool": len(pool), "product_signal": len(product_signal), "us_extra_review": len(chosen)})
        return 0
    if args.product_rescue:
        final = list(read_jsonl(args.outputs / "final_cases_200.jsonl"))
        used = {row["case_id"] for row in final}
        for path in (args.outputs / "us_replacement_candidates_review_v3.jsonl", args.outputs / "us_replacement_candidates_review_extra_v3.jsonl"):
            used.update(row["case_id"] for row in read_jsonl(path))
        pool = [row for row in read_jsonl(args.outputs / "us_state_highcourt_candidates.jsonl") if row["case_id"] not in used]
        chosen = [row for row in pool if row.get("primary_domain") == "product_liability" or PRODUCT_DEFECT.search(" ".join(str(row.get(key) or "") for key in ("case_name", "nature_of_suit", "main_opinion_text")))]
        output = args.outputs / "us_product_rescue_candidates_review_v3.jsonl"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        write_jsonl(output, sorted(chosen, key=lambda row: row["case_id"]))
        print({"seed": SEED, "product_rescue_review": len(chosen), "preclassified_core_sufficient": sum(bool(row.get("core_fact_sufficient")) for row in chosen)})
        return 0
    if args.remaining_preferred:
        final = list(read_jsonl(args.outputs / "final_cases_200.jsonl"))
        used = {row["case_id"] for row in final}
        for path in (args.outputs / "us_replacement_candidates_review_v3.jsonl", args.outputs / "us_replacement_candidates_review_extra_v3.jsonl", args.outputs / "us_product_rescue_candidates_review_v3.jsonl"):
            used.update(row["case_id"] for row in read_jsonl(path))
        chosen = [row for row in read_jsonl(args.outputs / "us_state_highcourt_candidates.jsonl") if row["case_id"] not in used and row.get("preferred_fact_sufficiency") is True]
        output = args.outputs / "us_replacement_candidates_review_remaining_v3.jsonl"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        write_jsonl(output, sorted(chosen, key=lambda row: row["case_id"]))
        print({"seed": SEED, "remaining_preferred_review": len(chosen)})
        return 0
    if not args.overwrite and (kr_out.exists() or us_out.exists()):
        raise FileExistsError("Candidate review input exists; pass --overwrite")
    final = list(read_jsonl(args.outputs / "final_cases_200.jsonl"))
    ids = {row["case_id"] for row in final}
    families = {(row["origin_country"], row.get("case_family_id")) for row in final}

    kr_pool = [row for row in read_jsonl(args.outputs / "kr_supreme_candidates.jsonl") if row["case_id"] not in ids and ("KR", row.get("case_family_id")) not in families and row.get("strict_source_eligible") is True]
    kr_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kr_pool:
        kr_groups[row.get("primary_domain") or ""] .append(row)
    kr = [
        *take(kr_groups["medical_professional_liability"], 100),
        *take(kr_groups["general_negligence_personal_injury"], 10),
        *take(kr_groups["other_civil_liability"], 10),
    ]

    us_pool = [row for row in read_jsonl(args.outputs / "us_state_highcourt_candidates.jsonl") if row["case_id"] not in ids and ("US", row.get("case_family_id")) not in families and row.get("preferred_fact_sufficiency") is True]
    us_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in us_pool:
        us_groups[row.get("primary_domain") or ""].append(row)
    us: list[dict[str, Any]] = []
    quotas = {"other_civil_liability": 34, "product_liability": 8, "general_negligence_personal_injury": 60, "medical_professional_liability": 38}
    for domain, count in quotas.items():
        # State round-robin prevents one state from consuming a domain batch.
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(us_groups[domain], key=rank):
            by_state[row.get("origin_state") or ""].append(row)
        states = sorted(by_state)
        selected: list[dict[str, Any]] = []
        while len(selected) < count and any(by_state.values()):
            for state in states:
                if by_state[state] and len(selected) < count:
                    selected.append(by_state[state].pop(0))
        us.extend(selected)
    write_jsonl(kr_out, sorted(kr, key=lambda row: row["case_id"]))
    write_jsonl(us_out, sorted(us, key=lambda row: row["case_id"]))
    print({"seed": SEED, "kr_pool": len(kr_pool), "kr_review": len(kr), "us_pool": len(us_pool), "us_review": len(us)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
