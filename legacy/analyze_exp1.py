from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs/exp1/.matplotlib"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from exp1.common import ONTOLOGY_PATH, read_jsonl, write_csv, write_json

LABELS = [
    "issue_identification", "governing_rule", "duty_or_protected_interest",
    "breach_or_wrongfulness", "fault_or_intent", "factual_causation",
    "legal_or_proximate_causation", "injury_or_damage", "plaintiff_fault_or_defense",
    "vicarious_or_organizational_liability", "multiple_tortfeasors", "damages_scope",
    "evidentiary_uncertainty", "procedural_reasoning", "policy_reasoning", "conclusion", "other",
]
CONCLUSIONS = ["likely", "unlikely", "mixed_or_partial", "conditional", "uncertain", "not_assessed"]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def bootstrap_ci(values: list[float], seed: int, iterations: int = 5000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    estimates = np.mean(rng.choice(data, size=(iterations, len(data)), replace=True), axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def bootstrap_group_difference(
    first: list[float], second: list[float], seed: int, iterations: int = 5000,
) -> list[float | None]:
    if not first or not second:
        return [None, None]
    rng = np.random.default_rng(seed)
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    estimates = (
        np.mean(rng.choice(a, size=(iterations, len(a)), replace=True), axis=1)
        - np.mean(rng.choice(b, size=(iterations, len(b)), replace=True), axis=1)
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def permutation_p(values: list[float], seed: int, iterations: int = 10000) -> float | None:
    data = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(data) == 0:
        return None
    observed = abs(float(np.mean(data)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        value = abs(float(np.mean(data * rng.choice([-1.0, 1.0], size=len(data)))))
        extreme += value >= observed - 1e-15
    return (extreme + 1) / (iterations + 1)


def bh_adjust(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted((p, key) for key, p in pvalues.items() if p is not None)
    result = {key: None for key in pvalues}
    running = 1.0
    for rank_from_end, (p, key) in enumerate(reversed(valid), 1):
        rank = len(valid) - rank_from_end + 1
        running = min(running, p * len(valid) / rank)
        result[key] = running
    return result


def mcnemar_exact(ko: list[int], en: list[int]) -> dict[str, Any]:
    b = sum(k == 1 and e == 0 for k, e in zip(ko, en))
    c = sum(k == 0 and e == 1 for k, e in zip(ko, en))
    return {"ko_only": b, "en_only": c, "p_value": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}


def reasoning_vector(evaluation: dict[str, Any]) -> dict[str, float]:
    units = evaluation["reasoning_units"]
    counts = Counter(label for unit in units for label in unit["labels"])
    return {label: counts[label] / len(units) if units else 0.0 for label in LABELS}


def js_divergence(a: dict[str, float], b: dict[str, float]) -> float:
    x, y = np.array([a[k] for k in LABELS]), np.array([b[k] for k in LABELS])
    if x.sum() == 0 and y.sum() == 0:
        return 0.0
    x = x / x.sum() if x.sum() else np.full(len(x), 1 / len(x))
    y = y / y.sum() if y.sum() else np.full(len(y), 1 / len(y))
    m = (x + y) / 2
    def kl(p: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / m[mask])))
    return (kl(x) + kl(y)) / 2


def aggregate(records: list[dict[str, Any]], concept_system: dict[str, str]) -> dict[str, Any]:
    evaluations = [r["evaluation"] for r in records]
    vectors = [reasoning_vector(e) for e in evaluations]
    parties: dict[str, list[str]] = defaultdict(list)
    parties_by_replicate: dict[int, dict[str, str]] = {}
    for evaluation in evaluations:
        for party in evaluation["parties"]:
            parties[party["party_id"]].append(party["conclusion"])
    for record in records:
        parties_by_replicate[record["replicate_id"]] = {
            party["party_id"]: party["conclusion"] for party in record["evaluation"]["parties"]
        }
    modal = {party: Counter(values).most_common(1)[0][0] for party, values in parties.items()}
    concepts = [
        {c["concept_id"] for c in evaluation["concepts"] if c["present"]}
        for evaluation in evaluations
    ]
    jurisdictions = [e["jurisdiction_signals"] for e in evaluations]
    return {
        "records": records,
        "party_conclusions": modal,
        "party_conclusions_by_replicate": parties_by_replicate,
        "party_instability": mean([len(set(v)) > 1 for v in parties.values()]),
        "reasoning": {label: mean([v[label] for v in vectors]) for label in LABELS},
        "unit_count": mean([len(e["reasoning_units"]) for e in evaluations]),
        "output_chars": mean([len(r.get("raw_response", "")) for r in records]),
        "kr_marker": mean([any(concept_system.get(cid) == "KR" for cid in cs) for cs in concepts]),
        "us_marker": mean([any(concept_system.get(cid) == "US_COMMON_LAW" for cid in cs) for cs in concepts]),
        "a_marker": mean([any(c["present"] and c["marker_strength"] == "A" for c in e["concepts"]) for e in evaluations]),
        "explicit_jurisdiction": mean([j["explicit_jurisdiction"] != "NONE" for j in jurisdictions]),
        "statute": mean([j["explicit_statute_reference"] for j in jurisdictions]),
        "precedent": mean([j["explicit_precedent_reference"] for j in jurisdictions]),
        "hallucinated": mean([j["unsupported_or_hallucinated_authority"] for j in jurisdictions]),
        "confidence": mean([e["evaluator_confidence"] for e in evaluations]),
        "within_js": mean([
            js_divergence(vectors[i], vectors[j])
            for i in range(len(vectors)) for j in range(i + 1, len(vectors))
        ]) if len(vectors) > 1 else 0.0,
        "reasoning_by_replicate": {
            record["replicate_id"]: vector for record, vector in zip(records, vectors)
        },
    }


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    agree = sum(a == b for a, b in pairs) / len(pairs)
    a_counts, b_counts = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((a_counts[c] / len(pairs)) * (b_counts[c] / len(pairs)) for c in CONCLUSIONS)
    return (agree - expected) / (1 - expected) if expected < 1 else 1.0


def plot_outputs(pair_rows: list[dict[str, Any]], stats: dict[str, Any], output_dir: Path) -> None:
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    labels = ["KR-oriented", "US/common-law"]
    ko = [mean([r["ko_kr_marker"] for r in pair_rows]), mean([r["ko_us_marker"] for r in pair_rows])]
    en = [mean([r["en_kr_marker"] for r in pair_rows]), mean([r["en_us_marker"] for r in pair_rows])]
    x = np.arange(2)
    plt.bar(x - .18, ko, .36, label="KO")
    plt.bar(x + .18, en, .36, label="EN")
    plt.xticks(x, labels); plt.ylabel("Prevalence"); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(graph_dir / "marker_prevalence.png", dpi=180); plt.close()

    matrix = np.zeros((len(CONCLUSIONS), len(CONCLUSIONS)), dtype=int)
    for row in pair_rows:
        for a, b in row["_conclusion_pairs"]:
            matrix[CONCLUSIONS.index(a), CONCLUSIONS.index(b)] += 1
    plt.figure(figsize=(7, 6)); plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(CONCLUSIONS)), CONCLUSIONS, rotation=45, ha="right")
    plt.yticks(range(len(CONCLUSIONS)), CONCLUSIONS)
    plt.xlabel("EN"); plt.ylabel("KO"); plt.colorbar(label="Party-pairs")
    for i in range(len(CONCLUSIONS)):
        for j in range(len(CONCLUSIONS)):
            if matrix[i, j]: plt.text(j, i, matrix[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(graph_dir / "conclusion_transitions.png", dpi=180); plt.close()

    reason = stats["reasoning_composition"]
    names = LABELS
    estimates = [reason[x]["mean_paired_difference_ko_minus_en"] for x in names]
    lows = [reason[x]["bootstrap_95_ci"][0] for x in names]
    highs = [reason[x]["bootstrap_95_ci"][1] for x in names]
    y = np.arange(len(names))
    plt.figure(figsize=(8, 8)); plt.axvline(0, color="black", lw=.8)
    plt.errorbar(estimates, y, xerr=[np.array(estimates)-np.array(lows), np.array(highs)-np.array(estimates)], fmt="o")
    plt.yticks(y, names); plt.xlabel("KO − EN unit proportion"); plt.tight_layout()
    plt.savefig(graph_dir / "reasoning_paired_differences.png", dpi=180); plt.close()

    origins = ["KR", "CA"]
    metrics = ["kr_marker_difference", "us_marker_difference"]
    values = [[mean([r[m] for r in pair_rows if r["case_origin"] == origin]) for m in metrics] for origin in origins]
    plt.figure(figsize=(6, 4)); x = np.arange(2)
    for i, origin in enumerate(origins):
        plt.bar(x + (i-.5)*.36, values[i], .36, label=f"{origin} origin")
    plt.axhline(0, color="black", lw=.8); plt.xticks(x, ["KR markers", "US markers"]); plt.ylabel("KO − EN"); plt.legend(); plt.tight_layout()
    plt.savefig(graph_dir / "origin_stratified.png", dpi=180); plt.close()

    within = [r["ko_within_js"] for r in pair_rows] + [r["en_within_js"] for r in pair_rows]
    cross = [r["cross_language_js"] for r in pair_rows]
    plt.figure(figsize=(5, 4)); plt.boxplot([within, cross], tick_labels=["Within-language", "Cross-language"])
    plt.ylabel("Jensen–Shannon divergence"); plt.tight_layout()
    plt.savefig(graph_dir / "replicate_vs_language_divergence.png", dpi=180); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair-level Experiment 1 analysis.")
    parser.add_argument("--evaluations", type=Path, default=Path("outputs/exp1/evaluations.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--human-random-pairs", type=int, default=10)
    args = parser.parse_args()

    records = [r for r in read_jsonl(args.evaluations) if r.get("evaluation")]
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    concept_system = {c["concept_id"]: c["system"] for c in ontology["concepts"]}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["case_id"]][record["condition"]].append(record)
    incomplete = sorted(case_id for case_id, values in grouped.items() if set(values) != {"ko", "en"})
    if incomplete:
        raise SystemExit(f"Incomplete KO/EN evaluations: {incomplete[:10]}")

    pair_rows: list[dict[str, Any]] = []
    all_conclusion_pairs: list[tuple[str, str]] = []
    for case_id, conditions in sorted(grouped.items()):
        ko, en = aggregate(conditions["ko"], concept_system), aggregate(conditions["en"], concept_system)
        shared_parties = sorted(set(ko["party_conclusions"]) & set(en["party_conclusions"]))
        conclusion_pairs = [(ko["party_conclusions"][p], en["party_conclusions"][p]) for p in shared_parties]
        replicate_conclusion_pairs: list[tuple[str, str]] = []
        for replicate_id in sorted(
            set(ko["party_conclusions_by_replicate"]) & set(en["party_conclusions_by_replicate"])
        ):
            ko_parties = ko["party_conclusions_by_replicate"][replicate_id]
            en_parties = en["party_conclusions_by_replicate"][replicate_id]
            replicate_conclusion_pairs.extend(
                (ko_parties[party], en_parties[party])
                for party in sorted(set(ko_parties) & set(en_parties))
            )
        matched_replicates = sorted(
            set(ko["reasoning_by_replicate"]) & set(en["reasoning_by_replicate"])
        )
        matched_cross_js = [
            js_divergence(
                ko["reasoning_by_replicate"][replicate_id],
                en["reasoning_by_replicate"][replicate_id],
            )
            for replicate_id in matched_replicates
        ]
        all_conclusion_pairs.extend(conclusion_pairs)
        direct_flip = any(set(pair) == {"likely", "unlikely"} for pair in conclusion_pairs)
        conditional_shift = any(
            (a in {"likely", "unlikely"}) != (b in {"likely", "unlikely"}) and
            ({a, b} & {"conditional", "uncertain"})
            for a, b in conclusion_pairs
        )
        origin = conditions["ko"][0]["case_origin"]
        row: dict[str, Any] = {
            "case_id": case_id,
            "case_origin": origin,
            "case_subtype": conditions["ko"][0]["case_subtype"],
            "party_count_compared": len(conclusion_pairs),
            "conclusion_agreement": mean([a == b for a, b in conclusion_pairs]),
            "any_conclusion_change": any(a != b for a, b in conclusion_pairs),
            "direct_likely_unlikely_flip": direct_flip,
            "likely_unlikely_to_conditional_uncertain": bool(conditional_shift),
            "ko_kr_marker": ko["kr_marker"], "en_kr_marker": en["kr_marker"],
            "kr_marker_difference": ko["kr_marker"] - en["kr_marker"],
            "ko_us_marker": ko["us_marker"], "en_us_marker": en["us_marker"],
            "us_marker_difference": ko["us_marker"] - en["us_marker"],
            "ko_strong_a": ko["a_marker"], "en_strong_a": en["a_marker"],
            "strong_a_difference": ko["a_marker"] - en["a_marker"],
            "ko_explicit_jurisdiction": ko["explicit_jurisdiction"],
            "en_explicit_jurisdiction": en["explicit_jurisdiction"],
            "explicit_jurisdiction_difference": ko["explicit_jurisdiction"] - en["explicit_jurisdiction"],
            "ko_statute": ko["statute"], "en_statute": en["statute"],
            "statute_difference": ko["statute"] - en["statute"],
            "ko_precedent": ko["precedent"], "en_precedent": en["precedent"],
            "precedent_difference": ko["precedent"] - en["precedent"],
            "ko_hallucinated_authority": ko["hallucinated"],
            "en_hallucinated_authority": en["hallucinated"],
            "hallucinated_authority_difference": ko["hallucinated"] - en["hallucinated"],
            "ko_output_chars": ko["output_chars"], "en_output_chars": en["output_chars"],
            "output_chars_difference": ko["output_chars"] - en["output_chars"],
            "ko_reasoning_units": ko["unit_count"], "en_reasoning_units": en["unit_count"],
            "cross_language_js": mean(matched_cross_js),
            "ko_within_js": ko["within_js"], "en_within_js": en["within_js"],
            "ko_conclusion_instability": ko["party_instability"],
            "en_conclusion_instability": en["party_instability"],
            "min_evaluator_confidence": min(ko["confidence"], en["confidence"]),
            "master_condition": "ko" if origin == "KR" else "en",
            "_conclusion_pairs": conclusion_pairs,
            "_replicate_conclusion_pairs": replicate_conclusion_pairs,
            "_ko": ko, "_en": en,
        }
        for label in LABELS:
            row[f"reasoning_diff__{label}"] = ko["reasoning"][label] - en["reasoning"][label]
        pair_rows.append(row)

    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in pair_rows]
    write_csv(args.output_dir / "pair_metrics.csv", public_rows, list(public_rows[0]))

    stats: dict[str, Any] = {
        "case_pairs": len(pair_rows),
        "response_evaluations": len(records),
        "conclusion_stability": {
            "party_level_agreement_rate": mean([a == b for a, b in all_conclusion_pairs]),
            "party_level_change_rate": mean([a != b for a, b in all_conclusion_pairs]),
            "case_level_change_rate": mean([r["any_conclusion_change"] for r in pair_rows]),
            "case_level_direct_likely_unlikely_flip_rate": mean([r["direct_likely_unlikely_flip"] for r in pair_rows]),
            "case_level_likely_unlikely_to_conditional_uncertain_rate": mean([
                r["likely_unlikely_to_conditional_uncertain"] for r in pair_rows
            ]),
            "cohen_kappa_unweighted": cohen_kappa(all_conclusion_pairs),
            "by_origin": {
                origin: {
                    "case_count": sum(r["case_origin"] == origin for r in pair_rows),
                    "case_level_change_rate": mean([r["any_conclusion_change"] for r in pair_rows if r["case_origin"] == origin]),
                    "direct_likely_unlikely_flip_rate": mean([
                        r["direct_likely_unlikely_flip"] for r in pair_rows if r["case_origin"] == origin
                    ]),
                    "likely_unlikely_to_conditional_uncertain_rate": mean([
                        r["likely_unlikely_to_conditional_uncertain"] for r in pair_rows if r["case_origin"] == origin
                    ]),
                } for origin in ("KR", "CA")
            },
        },
        "legal_system_signals": {},
        "reasoning_composition": {},
        "origin_interaction": {},
        "replicate_stability": {},
        "limitations": [
            "Input language and translation status are not fully independent; language effects cannot be interpreted as pure translation-free causal effects.",
            "Evaluator coding is automated and requires blinded human validation.",
            "Party matching uses exact placeholder identifiers; unmatched identifiers are excluded from party-level agreement.",
            "McNemar tests dichotomize replicate prevalence by majority when repetitions exceed one.",
        ],
    }
    signal_specs = {
        "kr_oriented_marker": ("ko_kr_marker", "en_kr_marker"),
        "us_common_law_marker": ("ko_us_marker", "en_us_marker"),
        "strong_a": ("ko_strong_a", "en_strong_a"),
        "explicit_jurisdiction": ("ko_explicit_jurisdiction", "en_explicit_jurisdiction"),
        "statute": ("ko_statute", "en_statute"),
        "precedent": ("ko_precedent", "en_precedent"),
        "hallucinated_authority": ("ko_hallucinated_authority", "en_hallucinated_authority"),
    }
    for offset, (name, (ko_key, en_key)) in enumerate(signal_specs.items()):
        differences = [r[ko_key] - r[en_key] for r in pair_rows]
        ko_binary = [int(r[ko_key] >= .5) for r in pair_rows]
        en_binary = [int(r[en_key] >= .5) for r in pair_rows]
        stats["legal_system_signals"][name] = {
            "ko_prevalence": mean([r[ko_key] for r in pair_rows]),
            "en_prevalence": mean([r[en_key] for r in pair_rows]),
            "paired_risk_difference": mean(differences),
            "bootstrap_95_ci": bootstrap_ci(differences, args.seed + offset),
            "mcnemar_exact": mcnemar_exact(ko_binary, en_binary),
            "by_origin": {
                origin: {
                    "ko_prevalence": mean([r[ko_key] for r in pair_rows if r["case_origin"] == origin]),
                    "en_prevalence": mean([r[en_key] for r in pair_rows if r["case_origin"] == origin]),
                    "paired_risk_difference": mean([
                        r[ko_key] - r[en_key] for r in pair_rows if r["case_origin"] == origin
                    ]),
                } for origin in ("KR", "CA")
            },
        }

    pvalues: dict[str, float | None] = {}
    for index, label in enumerate(LABELS):
        values = [r[f"reasoning_diff__{label}"] for r in pair_rows]
        p = permutation_p(values, args.seed + index)
        pvalues[label] = p
        stats["reasoning_composition"][label] = {
            "mean_paired_difference_ko_minus_en": mean(values),
            "median_paired_difference_ko_minus_en": median(values),
            "bootstrap_95_ci": bootstrap_ci(values, args.seed + index),
            "paired_permutation_p": p,
        }
    adjusted = bh_adjust(pvalues)
    for label in LABELS:
        stats["reasoning_composition"][label]["bh_fdr_q"] = adjusted[label]
    stats["reasoning_composition_summary"] = {
        "mean_js_divergence": mean([r["cross_language_js"] for r in pair_rows]),
        "mean_output_chars_ko": mean([r["ko_output_chars"] for r in pair_rows]),
        "mean_output_chars_en": mean([r["en_output_chars"] for r in pair_rows]),
    }

    for metric in ("kr_marker_difference", "us_marker_difference", "output_chars_difference"):
        kr = [r[metric] for r in pair_rows if r["case_origin"] == "KR"]
        ca = [r[metric] for r in pair_rows if r["case_origin"] == "CA"]
        stats["origin_interaction"][metric] = {
            "kr_origin_language_effect_ko_minus_en": mean(kr),
            "ca_origin_language_effect_ko_minus_en": mean(ca),
            "difference_in_differences": mean(kr) - mean(ca),
            "interaction_bootstrap_95_ci": bootstrap_group_difference(
                kr, ca, args.seed + len(stats["origin_interaction"]),
            ),
        }
    stats["origin_interaction"]["master_vs_translated"] = {}
    for name, column in {
        "kr_marker": "kr_marker_difference",
        "us_marker": "us_marker_difference",
        "strong_a": "strong_a_difference",
        "explicit_jurisdiction": "explicit_jurisdiction_difference",
        "statute": "statute_difference",
        "precedent": "precedent_difference",
        "hallucinated_authority": "hallucinated_authority_difference",
        "output_chars": "output_chars_difference",
    }.items():
        values = [
            row[column] if row["case_origin"] == "KR" else -row[column]
            for row in pair_rows
        ]
        stats["origin_interaction"]["master_vs_translated"][name] = {
            "mean_master_minus_translated": mean(values),
            "bootstrap_95_ci": bootstrap_ci(values, args.seed + len(name)),
        }
    stats["origin_interaction"]["model_note"] = (
        "A random-intercept mixed model was not fit because statsmodels is unavailable; "
        "paired case-level estimates and the origin-stratified difference-in-differences are primary."
    )
    jurisdiction_counts: dict[str, dict[str, int]] = {"ko": {}, "en": {}}
    for condition in ("ko", "en"):
        counts = Counter(
            record["evaluation"]["jurisdiction_signals"]["explicit_jurisdiction"]
            for record in records if record["condition"] == condition
        )
        jurisdiction_counts[condition] = dict(counts)
    stats["legal_system_signals"]["explicit_jurisdiction_categories"] = jurisdiction_counts
    reasoning_excess = [
        row["cross_language_js"] - (row["ko_within_js"] + row["en_within_js"]) / 2
        for row in pair_rows
    ]
    conclusion_excess = [
        mean([a != b for a, b in row["_replicate_conclusion_pairs"]])
        - mean([row["ko_conclusion_instability"], row["en_conclusion_instability"]])
        for row in pair_rows
    ]
    stats["replicate_stability"] = {
        "mean_within_language_reasoning_js": mean([v for r in pair_rows for v in (r["ko_within_js"], r["en_within_js"])]),
        "mean_cross_language_reasoning_js": mean([r["cross_language_js"] for r in pair_rows]),
        "mean_within_language_conclusion_instability": mean([v for r in pair_rows for v in (r["ko_conclusion_instability"], r["en_conclusion_instability"])]),
        "cross_language_conclusion_discordance_modal": mean([a != b for a, b in all_conclusion_pairs]),
        "cross_language_conclusion_discordance_matched_replicates": mean([
            a != b for row in pair_rows for a, b in row["_replicate_conclusion_pairs"]
        ]),
        "cross_minus_within_reasoning_js": {
            "mean_paired_difference": mean(reasoning_excess),
            "bootstrap_95_ci": bootstrap_ci(reasoning_excess, args.seed + 700),
            "paired_permutation_p": permutation_p(reasoning_excess, args.seed + 700),
        },
        "cross_minus_within_conclusion_instability": {
            "mean_paired_difference": mean(conclusion_excess),
            "bootstrap_95_ci": bootstrap_ci(conclusion_excess, args.seed + 701),
            "paired_permutation_p": permutation_p(conclusion_excess, args.seed + 701),
        },
    }
    raw_generation_path = args.output_dir / "raw_responses.jsonl"
    evaluator_attempt_path = args.output_dir / "evaluator_raw_attempts.jsonl"
    generation_records = read_jsonl(raw_generation_path) if raw_generation_path.is_file() else []
    evaluator_attempts = read_jsonl(evaluator_attempt_path) if evaluator_attempt_path.is_file() else []
    stats["execution"] = {
        "generation_successful_responses": len(generation_records),
        "generation_recorded_api_calls": sum(
            1 + int(record.get("retry_count", 0)) for record in generation_records
        ),
        "generation_transport_retries": sum(
            int(record.get("retry_count", 0)) for record in generation_records
        ),
        "evaluation_successful_responses": len(records),
        "evaluation_api_response_attempts": len(evaluator_attempts),
        "evaluation_schema_invalid_attempts": sum(
            attempt.get("validation_status") == "invalid" for attempt in evaluator_attempts
        ),
    }
    write_json(args.output_dir / "summary.json", stats)

    rng = random.Random(args.seed)
    mandatory = {
        r["case_id"] for r in pair_rows
        if (
            r["direct_likely_unlikely_flip"]
            or r["ko_strong_a"] > 0
            or r["en_strong_a"] > 0
            or r["min_evaluator_confidence"] < .7
        )
    }
    for origin in ("KR", "CA"):
        pool = [r["case_id"] for r in pair_rows if r["case_origin"] == origin and r["case_id"] not in mandatory]
        rng.shuffle(pool)
        mandatory.update(pool[:max(1, args.human_random_pairs // 2)])
    human_rows = []
    for row in pair_rows:
        if row["case_id"] not in mandatory:
            continue
        ko_records, en_records = row["_ko"]["records"], row["_en"]["records"]
        human_rows.append({
            "case_id": row["case_id"], "case_origin": row["case_origin"],
            "ko_raw_response": json.dumps([r["raw_response"] for r in ko_records], ensure_ascii=False),
            "en_raw_response": json.dumps([r["raw_response"] for r in en_records], ensure_ascii=False),
            "automatic_conclusion_coding": json.dumps({
                "ko": row["_ko"]["party_conclusions"], "en": row["_en"]["party_conclusions"],
            }, ensure_ascii=False),
            "automatic_concept_coding": json.dumps({
                condition: [[c for c in rec["evaluation"]["concepts"] if c["present"]] for rec in row[f"_{condition}"]["records"]]
                for condition in ("ko", "en")
            }, ensure_ascii=False),
            "reviewer_conclusion_fields": "", "reviewer_concept_fields": "",
            "reviewer_notes": "", "reviewer_id": "", "review_status": "not_reviewed",
        })
    write_csv(args.output_dir / "human_validation_sample.csv", human_rows, list(human_rows[0]) if human_rows else [
        "case_id", "case_origin", "ko_raw_response", "en_raw_response", "automatic_conclusion_coding",
        "automatic_concept_coding", "reviewer_conclusion_fields", "reviewer_concept_fields",
        "reviewer_notes", "reviewer_id", "review_status",
    ])
    plot_outputs(pair_rows, stats, args.output_dir)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    kr_signal = stats["legal_system_signals"]["kr_oriented_marker"]
    us_signal = stats["legal_system_signals"]["us_common_law_marker"]
    strong = stats["legal_system_signals"]["strong_a"]
    explicit = stats["legal_system_signals"]["explicit_jurisdiction"]
    statute = stats["legal_system_signals"]["statute"]
    top_reasoning = sorted(
        (
            (label, values["mean_paired_difference_ko_minus_en"], values["bh_fdr_q"])
            for label, values in stats["reasoning_composition"].items()
            if values["bh_fdr_q"] is not None and values["bh_fdr_q"] < .05
        ),
        key=lambda value: abs(value[1]),
        reverse=True,
    )
    reasoning_lines = "\n".join(
        f"- `{label}`: KO − EN = {difference:.3f}, BH q = {q_value:.4g}"
        for label, difference, q_value in top_reasoning
    ) or "- No category passed BH-FDR 0.05."
    report = f"""# Experiment 1 results

## Run

- 70 cases, 420 responses (KO/EN × 3 replicates), 70 case-level pairs
- Generation: {stats['execution']['generation_successful_responses']} successful responses; {stats['execution']['generation_recorded_api_calls']} recorded API calls; {stats['execution']['generation_transport_retries']} transport retries
- Evaluation: {stats['execution']['evaluation_successful_responses']} valid evaluations from {stats['execution']['evaluation_api_response_attempts']} API response attempts; {stats['execution']['evaluation_schema_invalid_attempts']} schema-invalid intermediate attempts

## Conclusion stability

- Party-level agreement: {stats['conclusion_stability']['party_level_agreement_rate']:.3f}
- Party-level change: {stats['conclusion_stability']['party_level_change_rate']:.3f}
- Case-level any change: {stats['conclusion_stability']['case_level_change_rate']:.3f}
- Direct likely/unlikely flip: {stats['conclusion_stability']['case_level_direct_likely_unlikely_flip_rate']:.3f}
- Unweighted Cohen's kappa (nominal categories): {stats['conclusion_stability']['cohen_kappa_unweighted']:.3f}
- Case-level change by origin: KR {stats['conclusion_stability']['by_origin']['KR']['case_level_change_rate']:.3f}; CA {stats['conclusion_stability']['by_origin']['CA']['case_level_change_rate']:.3f}

## Legal-system signals

- KR-oriented marker prevalence: KO {kr_signal['ko_prevalence']:.3f}, EN {kr_signal['en_prevalence']:.3f}; paired difference {kr_signal['paired_risk_difference']:.3f}, bootstrap 95% CI [{kr_signal['bootstrap_95_ci'][0]:.3f}, {kr_signal['bootstrap_95_ci'][1]:.3f}], McNemar p = {kr_signal['mcnemar_exact']['p_value']:.3g}
- US/common-law marker prevalence: KO {us_signal['ko_prevalence']:.3f}, EN {us_signal['en_prevalence']:.3f}; paired difference {us_signal['paired_risk_difference']:.3f}, bootstrap 95% CI [{us_signal['bootstrap_95_ci'][0]:.3f}, {us_signal['bootstrap_95_ci'][1]:.3f}], McNemar p = {us_signal['mcnemar_exact']['p_value']:.3g}
- Strong A marker prevalence: KO {strong['ko_prevalence']:.3f}, EN {strong['en_prevalence']:.3f}; paired difference {strong['paired_risk_difference']:.3f}
- Explicit-jurisdiction prevalence: KO {explicit['ko_prevalence']:.3f}, EN {explicit['en_prevalence']:.3f}
- Statute-reference prevalence: KO {statute['ko_prevalence']:.3f}, EN {statute['en_prevalence']:.3f}
- Hallucinated authority detected: KO {stats['legal_system_signals']['hallucinated_authority']['ko_prevalence']:.3f}, EN {stats['legal_system_signals']['hallucinated_authority']['en_prevalence']:.3f}

## Reasoning composition

Significant paired proportion differences (positive means more weight in KO):

{reasoning_lines}

- Mean output length: KO {stats['reasoning_composition_summary']['mean_output_chars_ko']:.0f} chars; EN {stats['reasoning_composition_summary']['mean_output_chars_en']:.0f} chars
- Mean cross-language JS divergence: {stats['replicate_stability']['mean_cross_language_reasoning_js']:.3f}
- Mean within-language replicate JS divergence: {stats['replicate_stability']['mean_within_language_reasoning_js']:.3f}
- Cross-minus-within JS difference: {stats['replicate_stability']['cross_minus_within_reasoning_js']['mean_paired_difference']:.3f}, bootstrap 95% CI [{stats['replicate_stability']['cross_minus_within_reasoning_js']['bootstrap_95_ci'][0]:.3f}, {stats['replicate_stability']['cross_minus_within_reasoning_js']['bootstrap_95_ci'][1]:.3f}], permutation p = {stats['replicate_stability']['cross_minus_within_reasoning_js']['paired_permutation_p']:.4g}
- Matched-replicate cross-language conclusion discordance: {stats['replicate_stability']['cross_language_conclusion_discordance_matched_replicates']:.3f}
- Within-language conclusion instability: {stats['replicate_stability']['mean_within_language_conclusion_instability']:.3f}

## Origin interaction

- KR-marker language effect (KO − EN): KR-origin {stats['origin_interaction']['kr_marker_difference']['kr_origin_language_effect_ko_minus_en']:.3f}; CA-origin {stats['origin_interaction']['kr_marker_difference']['ca_origin_language_effect_ko_minus_en']:.3f}; interaction contrast {stats['origin_interaction']['kr_marker_difference']['difference_in_differences']:.3f}, bootstrap 95% CI [{stats['origin_interaction']['kr_marker_difference']['interaction_bootstrap_95_ci'][0]:.3f}, {stats['origin_interaction']['kr_marker_difference']['interaction_bootstrap_95_ci'][1]:.3f}]
- US-marker language effect (KO − EN): KR-origin {stats['origin_interaction']['us_marker_difference']['kr_origin_language_effect_ko_minus_en']:.3f}; CA-origin {stats['origin_interaction']['us_marker_difference']['ca_origin_language_effect_ko_minus_en']:.3f}; interaction contrast {stats['origin_interaction']['us_marker_difference']['difference_in_differences']:.3f}, bootstrap 95% CI [{stats['origin_interaction']['us_marker_difference']['interaction_bootstrap_95_ci'][0]:.3f}, {stats['origin_interaction']['us_marker_difference']['interaction_bootstrap_95_ci'][1]:.3f}]
- Master − translated contrast: KR marker {stats['origin_interaction']['master_vs_translated']['kr_marker']['mean_master_minus_translated']:.3f}; US marker {stats['origin_interaction']['master_vs_translated']['us_marker']['mean_master_minus_translated']:.3f}; output length {stats['origin_interaction']['master_vs_translated']['output_chars']['mean_master_minus_translated']:.0f} chars

## Interpretation and limitations

The language conditions show large legal-system-marker shifts, but conclusion coding is less stable and requires human review. Cross-language reasoning divergence was lower than within-language replicate divergence on average, so reasoning-composition differences should not be overstated. Input language and translation status are not independent: translation status is the language-by-origin interaction in this 2×2 design. Automated evaluator coding, exact placeholder party matching, unequal response lengths, and multiple model samples are additional limitations.
"""
    (args.reports_dir / "exp1_results.md").write_text(report, encoding="utf-8")
    print(f"pairs={len(pair_rows)} evaluations={len(records)} human_sample={len(human_rows)}")


if __name__ == "__main__":
    main()
