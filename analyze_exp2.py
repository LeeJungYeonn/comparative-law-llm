"""Compare Exp 2 evaluations with matched Exp 1 evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exp1.common import ONTOLOGY_PATH, read_jsonl, write_csv, write_json
from exp1.design import EXP2_EXPERIMENT_ID, TARGET_JURISDICTIONS

LABELS = [
    "issue_identification", "governing_rule", "duty_or_protected_interest",
    "breach_or_wrongfulness", "fault_or_intent", "factual_causation",
    "legal_or_proximate_causation", "injury_or_damage", "plaintiff_fault_or_defense",
    "vicarious_or_organizational_liability", "multiple_tortfeasors", "damages_scope",
    "evidentiary_uncertainty", "procedural_reasoning", "policy_reasoning", "conclusion", "other",
]


def mean(values: list[float | int | bool]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def reasoning_vector(evaluation: dict[str, Any]) -> dict[str, float]:
    units = evaluation["reasoning_units"]
    counts = Counter(label for unit in units for label in unit["labels"])
    return {label: counts[label] / len(units) if units else 0.0 for label in LABELS}


def js_divergence(first: dict[str, float], second: dict[str, float]) -> float:
    a, b = [first[label] for label in LABELS], [second[label] for label in LABELS]
    a_total, b_total = sum(a), sum(b)
    if not a_total and not b_total:
        return 0.0
    a = [value / a_total for value in a] if a_total else [1 / len(a)] * len(a)
    b = [value / b_total for value in b] if b_total else [1 / len(b)] * len(b)
    midpoint = [(x + y) / 2 for x, y in zip(a, b)]
    def kl(values: list[float]) -> float:
        return sum(value * math.log2(value / middle) for value, middle in zip(values, midpoint) if value)
    return (kl(a) + kl(b)) / 2


def _index(records: list[dict[str, Any]], expected_experiment: str | None = None) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        if not record.get("evaluation"):
            continue
        if expected_experiment and record.get("experiment_id") != expected_experiment:
            raise ValueError(f"Unexpected experiment_id: {record.get('experiment_id')}")
        key = (record["case_id"], record["condition"], int(record["replicate_id"]))
        if key in result:
            raise ValueError(f"Duplicate evaluation key: {key}")
        result[key] = record
    return result


def _party_conclusions(evaluation: dict[str, Any]) -> dict[str, str]:
    return {party["party_id"]: party["conclusion"] for party in evaluation["parties"]}


def _damages(evaluation: dict[str, Any]) -> set[str]:
    return {item["damage_id"] for item in evaluation["damages"] if item["present"]}


def analyze(exp1_records: list[dict[str, Any]], exp2_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exp1 = _index(exp1_records)
    exp2 = _index(exp2_records, EXP2_EXPERIMENT_ID)
    matched = sorted(exp1.keys() & exp2.keys())
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    systems = {concept["concept_id"]: concept["system"] for concept in ontology["concepts"]}
    rows: list[dict[str, Any]] = []
    conclusion_pairs: list[tuple[str, str]] = []
    added_damage_counts: Counter[str] = Counter()
    removed_damage_counts: Counter[str] = Counter()
    for key in matched:
        before, after = exp1[key], exp2[key]
        origin, language = after["case_origin"], after["condition"]
        target = after.get("target_jurisdiction") or TARGET_JURISDICTIONS[origin]
        target_system = "KR" if target == "KR" else "US_COMMON_LAW"
        wrong_system = "US_COMMON_LAW" if target_system == "KR" else "KR"
        evaluation_before, evaluation_after = before["evaluation"], after["evaluation"]
        parties_before, parties_after = _party_conclusions(evaluation_before), _party_conclusions(evaluation_after)
        shared_parties = sorted(parties_before.keys() & parties_after.keys())
        local_pairs = [(parties_before[party], parties_after[party]) for party in shared_parties]
        conclusion_pairs.extend(local_pairs)
        concepts = {item["concept_id"] for item in evaluation_after["concepts"] if item["present"]}
        target_terms = {concept for concept in concepts if systems.get(concept) == target_system}
        wrong_terms = {concept for concept in concepts if systems.get(concept) == wrong_system}
        explicit = evaluation_after["jurisdiction_signals"]["explicit_jurisdiction"]
        explicit_target = explicit == target
        explicit_wrong = explicit not in {target, "NONE", "AMBIGUOUS", "OTHER"}
        damages_before, damages_after = _damages(evaluation_before), _damages(evaluation_after)
        added, removed = damages_after - damages_before, damages_before - damages_after
        added_damage_counts.update(added)
        removed_damage_counts.update(removed)
        union = damages_before | damages_after
        row = {
            "case_id": key[0], "case_origin": origin, "input_language": language,
            "replicate_number": key[2], "target_jurisdiction": target,
            "matched_party_count": len(shared_parties),
            "conclusion_agreement_rate": mean([first == second for first, second in local_pairs]),
            "any_conclusion_shift": any(first != second for first, second in local_pairs),
            "explicit_target_jurisdiction": explicit_target,
            "target_jurisdiction_term_count": len(target_terms),
            "wrong_jurisdiction_term_count": len(wrong_terms),
            "has_wrong_jurisdiction_terms": bool(wrong_terms),
            "explicit_wrong_jurisdiction": explicit_wrong,
            "instruction_aligned": bool((explicit_target or target_terms) and not explicit_wrong and not wrong_terms),
            "remedy_categories_added": "|".join(sorted(added)),
            "remedy_categories_removed": "|".join(sorted(removed)),
            "any_remedy_category_shift": bool(added or removed),
            "remedy_jaccard_distance": 1 - len(damages_before & damages_after) / len(union) if union else 0.0,
            "reasoning_unit_js_distance": js_divergence(
                reasoning_vector(evaluation_before), reasoning_vector(evaluation_after),
            ),
        }
        rows.append(row)

    def grouped(metric: str, *, boolean: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for origin in ("KR", "CA"):
            for language in ("ko", "en"):
                values = [row[metric] for row in rows if row["case_origin"] == origin and row["input_language"] == language]
                values = [value for value in values if value is not None]
                output[f"{origin}_{language}"] = mean(values)
        return output

    cross_language_pairs: list[tuple[str, str]] = []
    by_case_rep: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, record in exp2.items():
        by_case_rep[(key[0], key[2])][key[1]] = record
    for paired in by_case_rep.values():
        if set(paired) != {"ko", "en"}:
            continue
        ko, en = _party_conclusions(paired["ko"]["evaluation"]), _party_conclusions(paired["en"]["evaluation"])
        cross_language_pairs.extend((ko[party], en[party]) for party in ko.keys() & en.keys())

    summary = {
        "experiment_id": EXP2_EXPERIMENT_ID,
        "matched_response_count": len(rows),
        "matched_case_count": len({row["case_id"] for row in rows}),
        "conclusion_stability": {
            "exp1_to_exp2_party_agreement_rate": mean([first == second for first, second in conclusion_pairs]),
            "exp1_to_exp2_party_shift_rate": mean([first != second for first, second in conclusion_pairs]),
            "response_level_any_shift_rate": mean([row["any_conclusion_shift"] for row in rows]),
            "exp2_cross_language_party_agreement_rate": mean([first == second for first, second in cross_language_pairs]),
        },
        "instruction_jurisdiction_alignment": {
            "definition": "target explicit jurisdiction or target-system ontology marker, with no wrong explicit jurisdiction or wrong-system marker",
            "alignment_rate": mean([row["instruction_aligned"] for row in rows]),
            "explicit_target_rate": mean([row["explicit_target_jurisdiction"] for row in rows]),
            "target_term_rate": mean([row["target_jurisdiction_term_count"] > 0 for row in rows]),
            "alignment_rate_by_condition": grouped("instruction_aligned"),
        },
        "wrong_jurisdiction_terms": {
            "response_prevalence": mean([row["wrong_jurisdiction_term_count"] > 0 for row in rows]),
            "mean_term_count": mean([row["wrong_jurisdiction_term_count"] for row in rows]),
            "explicit_wrong_jurisdiction_rate": mean([row["explicit_wrong_jurisdiction"] for row in rows]),
            "prevalence_by_condition": grouped("has_wrong_jurisdiction_terms"),
        },
        "remedy_category_shift": {
            "response_shift_rate": mean([row["any_remedy_category_shift"] for row in rows]),
            "mean_jaccard_distance": mean([row["remedy_jaccard_distance"] for row in rows]),
            "added_category_counts": dict(added_damage_counts),
            "removed_category_counts": dict(removed_damage_counts),
        },
        "reasoning_unit_distribution_distance": {
            "mean_js_distance": mean([row["reasoning_unit_js_distance"] for row in rows]),
            "mean_js_distance_by_condition": grouped("reasoning_unit_js_distance"),
        },
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Exp 2 evaluator output with Exp 1.")
    parser.add_argument("--exp1-evaluations", type=Path, default=Path("outputs/exp1/evaluations.jsonl"))
    parser.add_argument("--exp2-evaluations", type=Path, default=Path("outputs/exp2/evaluations.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp2"))
    args = parser.parse_args()
    rows, summary = analyze(read_jsonl(args.exp1_evaluations), read_jsonl(args.exp2_evaluations))
    fields = list(rows[0]) if rows else ["case_id"]
    write_csv(args.output_dir / "exp1_exp2_comparison_metrics.csv", rows, fields)
    write_json(args.output_dir / "exp2_summary.json", summary)
    print(f"matched_responses={len(rows)} matched_cases={summary['matched_case_count']}")


if __name__ == "__main__":
    main()
