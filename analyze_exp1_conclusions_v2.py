from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs/exp1/.matplotlib"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp1.common import read_jsonl, sha256_file, write_csv, write_json
from exp1.conclusion_v2 import (
    CONCLUSIONS, VERSION, aggregate_replicates, canonical_ids,
)

PRIMARY_CATEGORIES = ["likely", "unlikely", "mixed_or_partial", "conditional", "uncertain"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def bootstrap_ci(values: list[float], seed: int, iterations: int = 10000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    estimates = np.mean(rng.choice(data, size=(iterations, len(data)), replace=True), axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def permutation_p(values: list[float], seed: int, iterations: int = 20000) -> float | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(data)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        estimate = abs(float(np.mean(data * rng.choice([-1.0, 1.0], size=len(data)))))
        extreme += estimate >= observed - 1e-15
    return (extreme + 1) / (iterations + 1)


def cohen_kappa(pairs: list[tuple[str, str]], categories: list[str]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    a_counts, b_counts = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(
        (a_counts[category] / len(pairs)) * (b_counts[category] / len(pairs))
        for category in categories
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def transition_matrix(
    pairs: list[tuple[str, str]], categories: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {ko: {en: 0 for en in categories} for ko in categories}
    for ko, en in pairs:
        matrix[ko][en] += 1
    return matrix


def pairwise_instability(labels: dict[int, str]) -> float:
    pairs = [
        labels[a] != labels[b]
        for a, b in ((1, 2), (1, 3), (2, 3))
    ]
    return sum(pairs) / len(pairs)


def validate_flat_rows(
    rows: list[dict[str, Any]], registry: list[dict[str, str]],
) -> None:
    allowed = {
        (row["case_id"], row["canonical_party_id"]) for row in registry
    }
    seen: set[tuple[str, str]] = set()
    response_parties: dict[str, set[str]] = defaultdict(set)
    registry_by_case: dict[str, set[str]] = defaultdict(set)
    for row in registry:
        registry_by_case[row["case_id"]].add(row["canonical_party_id"])
    for row in rows:
        if row["conclusion"] not in CONCLUSIONS:
            raise ValueError(f"invalid_conclusion:{row['conclusion']}")
        if (row["case_id"], row["canonical_party_id"]) not in allowed:
            raise ValueError(f"unregistered_party:{row['case_id']}:{row['canonical_party_id']}")
        key = (row["response_id"], row["canonical_party_id"])
        if key in seen:
            raise ValueError(f"duplicate_response_party:{key}")
        seen.add(key)
        response_parties[row["response_id"]].add(row["canonical_party_id"])
    response_cases = {}
    for row in rows:
        response_cases[row["response_id"]] = row["case_id"]
    for response, parties in response_parties.items():
        expected = registry_by_case[response_cases[response]]
        if parties != expected:
            raise ValueError(f"response_party_set_mismatch:{response}")


def build_consensus(
    flat_rows: list[dict[str, Any]], registry: list[dict[str, str]],
) -> list[dict[str, Any]]:
    registry_map = {
        (row["case_id"], row["canonical_party_id"]): row for row in registry
    }
    grouped: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    evidence: dict[tuple[str, str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in flat_rows:
        key = (row["case_id"], row["language"], row["canonical_party_id"])
        replicate = int(row["replicate_id"])
        if replicate in grouped[key]:
            raise ValueError(f"duplicate_replicate_observation:{key}:{replicate}")
        grouped[key][replicate] = row["conclusion"]
        evidence[key][replicate] = {
            "response_id": row["response_id"],
            "supporting_text": row["supporting_text"],
            "aggregation_note": row["aggregation_note"],
        }
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        case_id, language, canonical = key
        aggregation = aggregate_replicates(grouped[key])
        source = registry_map[(case_id, canonical)]
        result.append({
            "version": VERSION,
            "case_id": case_id,
            "case_origin": source["case_origin"],
            "case_subtype": source["case_subtype"],
            "language": language,
            "canonical_party_id": canonical,
            "aggregation_status": aggregation["aggregation_status"],
            "consensus_conclusion": aggregation["consensus_conclusion"],
            "consensus_count": aggregation["consensus_count"],
            "replicate_1": grouped[key].get(1),
            "replicate_2": grouped[key].get(2),
            "replicate_3": grouped[key].get(3),
            "replicate_labels_json": json.dumps(aggregation["replicate_labels"], sort_keys=True),
            "evidence_json": json.dumps(evidence[key], ensure_ascii=False, sort_keys=True),
            "source_party_set_mismatch": source["source_party_set_mismatch"],
            "unresolved_source_issue": source["unresolved_source_issue"],
            "audit_flags": source["audit_flags"],
        })
    return result


def build_pair_rows(
    consensus: list[dict[str, Any]], registry: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["case_id"], row["language"], row["canonical_party_id"]): row
        for row in consensus
    }
    rows: list[dict[str, Any]] = []
    for source in sorted(registry, key=lambda row: (row["case_id"], row["canonical_party_id"])):
        case_id, party = source["case_id"], source["canonical_party_id"]
        ko, en = by_key.get((case_id, "ko", party)), by_key.get((case_id, "en", party))
        reasons: list[str] = []
        if str(source["unresolved_source_issue"]).casefold() == "true":
            reasons.append("unresolved_source_issue")
        if ko is None or ko["aggregation_status"] == "incomplete_replicates":
            reasons.append("ko_incomplete_replicates")
        elif ko["aggregation_status"] == "replicate_disagreement":
            reasons.append("ko_replicate_disagreement")
        if en is None or en["aggregation_status"] == "incomplete_replicates":
            reasons.append("en_incomplete_replicates")
        elif en["aggregation_status"] == "replicate_disagreement":
            reasons.append("en_replicate_disagreement")
        ko_label = ko["consensus_conclusion"] if ko else None
        en_label = en["consensus_conclusion"] if en else None
        if ko_label == "not_assessed":
            reasons.append("ko_not_assessed_consensus")
        if en_label == "not_assessed":
            reasons.append("en_not_assessed_consensus")
        primary = not reasons
        sensitivity = (
            ko is not None and en is not None
            and ko["aggregation_status"] == en["aggregation_status"] == "consensus"
            and str(source["unresolved_source_issue"]).casefold() != "true"
        )
        transition = f"{ko_label}->{en_label}" if ko_label and en_label else ""
        rows.append({
            "version": VERSION,
            "case_id": case_id,
            "case_origin": source["case_origin"],
            "case_subtype": source["case_subtype"],
            "canonical_party_id": party,
            "audit_flags": source["audit_flags"],
            "source_party_set_mismatch": source["source_party_set_mismatch"],
            "ko_aggregation_status": ko["aggregation_status"] if ko else "missing",
            "en_aggregation_status": en["aggregation_status"] if en else "missing",
            "ko_consensus": ko_label,
            "en_consensus": en_label,
            "ko_consensus_count": ko["consensus_count"] if ko else None,
            "en_consensus_count": en["consensus_count"] if en else None,
            "ko_replicate_labels": ko["replicate_labels_json"] if ko else "{}",
            "en_replicate_labels": en["replicate_labels_json"] if en else "{}",
            "ko_evidence": ko["evidence_json"] if ko else "{}",
            "en_evidence": en["evidence_json"] if en else "{}",
            "primary_eligible": primary,
            "sensitivity_eligible_including_not_assessed": sensitivity,
            "exclusion_reasons": ";".join(reasons),
            "agreement": primary and ko_label == en_label,
            "disagreement": primary and ko_label != en_label,
            "transition": transition,
            "direct_likely_unlikely_flip": primary and {ko_label, en_label} == {"likely", "unlikely"},
            "likely_unlikely_to_conditional_uncertain": primary and (
                (ko_label in {"likely", "unlikely"} and en_label in {"conditional", "uncertain"})
                or (en_label in {"likely", "unlikely"} and ko_label in {"conditional", "uncertain"})
            ),
        })
    return rows


def plot_heatmap(
    matrix: dict[str, dict[str, int]], categories: list[str], path: Path,
) -> None:
    values = np.array([[matrix[ko][en] for en in categories] for ko in categories])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(values, cmap="Blues")
    plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
    plt.yticks(range(len(categories)), categories)
    plt.xlabel("EN consensus")
    plt.ylabel("KO consensus")
    plt.colorbar(label="Canonical party comparisons")
    for i in range(len(categories)):
        for j in range(len(categories)):
            plt.text(j, i, int(values[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze canonical-party conclusion recoding v2.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    registry = read_csv(args.output_dir / "party_registry_v2.csv")
    flat = read_jsonl(args.output_dir / "party_conclusions_response_v2.jsonl")
    validate_flat_rows(flat, registry)
    consensus = build_consensus(flat, registry)
    consensus_fields = list(consensus[0])
    write_csv(args.output_dir / "party_conclusions_consensus_v2.csv", consensus, consensus_fields)
    pair_rows = build_pair_rows(consensus, registry)
    write_csv(args.output_dir / "conclusion_pair_metrics_v2.csv", pair_rows, list(pair_rows[0]))

    eligible = [row for row in pair_rows if row["primary_eligible"]]
    primary_pairs = [(row["ko_consensus"], row["en_consensus"]) for row in eligible]
    sensitivity_rows = [
        row for row in pair_rows if row["sensitivity_eligible_including_not_assessed"]
    ]
    sensitivity_pairs = [
        (row["ko_consensus"], row["en_consensus"]) for row in sensitivity_rows
    ]
    matrix = transition_matrix(primary_pairs, PRIMARY_CATEGORIES)
    sensitivity_matrix = transition_matrix(sensitivity_pairs, list(CONCLUSIONS))
    plot_heatmap(
        matrix, PRIMARY_CATEGORIES,
        args.output_dir / "graphs_v2/conclusion_transition_heatmap.png",
    )

    language_party_units = len(registry) * 2
    aggregation_counts = Counter(
        (row["language"], row["aggregation_status"]) for row in consensus
    )
    not_assessed_counts = Counter(
        row["language"] for row in consensus
        if row["aggregation_status"] == "consensus"
        and row["consensus_conclusion"] == "not_assessed"
    )
    exclusion_counts = Counter(
        reason
        for row in pair_rows
        for reason in str(row["exclusion_reasons"]).split(";")
        if reason
    )
    category_distributions = {
        language: dict(Counter(
            row[f"{language}_consensus"] for row in eligible
        ))
        for language in ("ko", "en")
    }
    agreement_count = sum(ko == en for ko, en in primary_pairs)
    disagreement_count = len(primary_pairs) - agreement_count

    eligible_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        eligible_by_case[row["case_id"]].append(row)
    case_any_change = {
        case_id: any(row["ko_consensus"] != row["en_consensus"] for row in rows)
        for case_id, rows in eligible_by_case.items()
    }
    case_direct_flip = {
        case_id: any(row["direct_likely_unlikely_flip"] for row in rows)
        for case_id, rows in eligible_by_case.items()
    }
    case_conditional_shift = {
        case_id: any(row["likely_unlikely_to_conditional_uncertain"] for row in rows)
        for case_id, rows in eligible_by_case.items()
    }

    per_party_divergence: list[dict[str, Any]] = []
    for row in eligible:
        ko_labels = {int(k): v for k, v in json.loads(row["ko_replicate_labels"]).items()}
        en_labels = {int(k): v for k, v in json.loads(row["en_replicate_labels"]).items()}
        within = (pairwise_instability(ko_labels) + pairwise_instability(en_labels)) / 2
        cross = sum(ko_labels[r] != en_labels[r] for r in (1, 2, 3)) / 3
        per_party_divergence.append({
            "case_id": row["case_id"],
            "case_origin": row["case_origin"],
            "canonical_party_id": row["canonical_party_id"],
            "within_language_instability": within,
            "cross_language_discordance": cross,
            "cross_minus_within": cross - within,
        })
    by_case_difference: dict[str, list[float]] = defaultdict(list)
    for row in per_party_divergence:
        by_case_difference[row["case_id"]].append(row["cross_minus_within"])
    case_level_differences = [float(np.mean(values)) for values in by_case_difference.values()]

    origins = {}
    for origin in ("KR", "CA"):
        origin_parties = [row for row in eligible if row["case_origin"] == origin]
        origin_cases = {
            row["case_id"] for row in origin_parties
        }
        origin_pairs = [(row["ko_consensus"], row["en_consensus"]) for row in origin_parties]
        origins[origin] = {
            "eligible_party_n": len(origin_parties),
            "party_denominator_total_registry": sum(row["case_origin"] == origin for row in registry),
            "agreement_n": sum(a == b for a, b in origin_pairs),
            "agreement_rate": mean([a == b for a, b in origin_pairs]),
            "kappa": cohen_kappa(origin_pairs, PRIMARY_CATEGORIES),
            "eligible_case_n": len(origin_cases),
            "case_any_change_n": sum(case_any_change[case_id] for case_id in origin_cases),
            "case_any_change_rate": mean([case_any_change[case_id] for case_id in origin_cases]),
            "direct_flip_case_n": sum(case_direct_flip[case_id] for case_id in origin_cases),
            "conditional_transition_case_n": sum(
                case_conditional_shift[case_id] for case_id in origin_cases
            ),
        }

    legacy = json.loads(
        (args.output_dir / "legacy_conclusion_audit_v2.json").read_text(encoding="utf-8")
    )
    legacy_comparisons = read_csv(args.output_dir / "legacy_conclusion_comparisons_v2.csv")
    legacy_by_canonical: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in legacy_comparisons:
        ids = canonical_ids(row["exact_party_string"])
        if len(ids) == 1:
            legacy_by_canonical[(row["case_id"], ids[0])].append(row)
    change_log = []
    for row in pair_rows:
        old = legacy_by_canonical.get((row["case_id"], row["canonical_party_id"]), [])
        old_ko = sorted({entry["ko_modal"] for entry in old})
        old_en = sorted({entry["en_modal"] for entry in old})
        tie = any(
            str(entry["ko_modal_tie"]).casefold() == "true"
            or str(entry["en_modal_tie"]).casefold() == "true"
            for entry in old
        )
        reasons = []
        if not old:
            reasons.append("no_single_canonical_legacy_match")
        if len(old) > 1:
            reasons.append("multiple_legacy_strings_map_to_canonical_party")
        if tie:
            reasons.append("legacy_modal_tie")
        if row["primary_eligible"] and (
            old_ko != [row["ko_consensus"]] or old_en != [row["en_consensus"]]
        ):
            reasons.append("conclusion_changed_after_canonical_recoding")
        if not row["primary_eligible"]:
            reasons.append("excluded_under_v2_primary_rules")
        change_log.append({
            "case_id": row["case_id"],
            "case_origin": row["case_origin"],
            "canonical_party_id": row["canonical_party_id"],
            "legacy_exact_party_strings": json.dumps(
                [entry["exact_party_string"] for entry in old], ensure_ascii=False
            ),
            "legacy_ko_modal_labels": json.dumps(old_ko),
            "legacy_en_modal_labels": json.dumps(old_en),
            "legacy_modal_tie": tie,
            "v2_ko_consensus": row["ko_consensus"],
            "v2_en_consensus": row["en_consensus"],
            "v2_primary_eligible": row["primary_eligible"],
            "v2_transition": row["transition"],
            "change_reasons": ";".join(reasons),
        })
    write_csv(
        args.output_dir / "conclusion_change_log_v2.csv",
        change_log,
        list(change_log[0]),
    )

    raw_records = read_jsonl(args.output_dir / "raw_responses.jsonl")
    raw_by_case_language: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        raw_by_case_language[(record["case_id"], record["condition"])].append(record)
    direct_rows = []
    for row in pair_rows:
        if not row["direct_likely_unlikely_flip"]:
            continue
        direct_rows.append({
            "case_id": row["case_id"],
            "case_origin": row["case_origin"],
            "canonical_party_id": row["canonical_party_id"],
            "automatic_ko_consensus": row["ko_consensus"],
            "automatic_en_consensus": row["en_consensus"],
            "ko_replicate_labels": row["ko_replicate_labels"],
            "en_replicate_labels": row["en_replicate_labels"],
            "ko_evaluator_evidence": row["ko_evidence"],
            "en_evaluator_evidence": row["en_evidence"],
            "ko_raw_responses_json": json.dumps(
                [record["raw_response"] for record in sorted(
                    raw_by_case_language[(row["case_id"], "ko")],
                    key=lambda value: value["replicate_id"],
                )],
                ensure_ascii=False,
            ),
            "en_raw_responses_json": json.dumps(
                [record["raw_response"] for record in sorted(
                    raw_by_case_language[(row["case_id"], "en")],
                    key=lambda value: value["replicate_id"],
                )],
                ensure_ascii=False,
            ),
            "human_flip_status": "",
            "reviewer_notes": "",
            "reviewer_id": "",
            "review_status": "",
        })
    direct_fields = list(direct_rows[0]) if direct_rows else [
        "case_id", "case_origin", "canonical_party_id", "automatic_ko_consensus",
        "automatic_en_consensus", "ko_replicate_labels", "en_replicate_labels",
        "ko_evaluator_evidence", "en_evaluator_evidence", "ko_raw_responses_json",
        "en_raw_responses_json", "human_flip_status", "reviewer_notes", "reviewer_id",
        "review_status",
    ]
    write_csv(args.output_dir / "direct_flip_review_v2.csv", direct_rows, direct_fields)

    mandatory: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in pair_rows:
        key = (row["case_id"], row["canonical_party_id"])
        if row["direct_likely_unlikely_flip"]:
            mandatory[key].add("direct_flip_candidate")
        if "replicate_disagreement" in row["ko_aggregation_status"] or "replicate_disagreement" in row["en_aggregation_status"]:
            mandatory[key].add("replicate_disagreement")
        if str(row["source_party_set_mismatch"]).casefold() == "true":
            mandatory[key].add("source_party_set_mismatch")
        audit = str(row["audit_flags"])
        for flag in (
            "existing_attached_legal_role",
            "existing_grouped_party_string",
            "existing_duplicate_party_mapping",
        ):
            if flag in audit:
                mandatory[key].add(flag)
        if row["primary_eligible"] and row["ko_consensus"] != row["en_consensus"]:
            mandatory[key].add("language_conclusion_difference")
    agreements = [
        row for row in pair_rows
        if row["primary_eligible"] and row["ko_consensus"] == row["en_consensus"]
        and (row["case_id"], row["canonical_party_id"]) not in mandatory
    ]
    rng = random.Random(args.seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in agreements:
        strata[(row["case_origin"], row["ko_consensus"])].append(row)
    for stratum, candidates in sorted(strata.items()):
        rng.shuffle(candidates)
        for row in candidates[:2]:
            mandatory[(row["case_id"], row["canonical_party_id"])].add(
                f"stratified_agreement_sample:{stratum[0]}:{stratum[1]}"
            )
    pair_map = {(row["case_id"], row["canonical_party_id"]): row for row in pair_rows}
    human_rows = []
    for key in sorted(mandatory):
        row = pair_map[key]
        human_rows.append({
            "case_id": row["case_id"],
            "case_origin": row["case_origin"],
            "case_subtype": row["case_subtype"],
            "canonical_party_id": row["canonical_party_id"],
            "selection_reasons": ";".join(sorted(mandatory[key])),
            "audit_flags": row["audit_flags"],
            "automatic_conclusion_ko": row["ko_consensus"],
            "automatic_conclusion_en": row["en_consensus"],
            "ko_aggregation_status": row["ko_aggregation_status"],
            "en_aggregation_status": row["en_aggregation_status"],
            "ko_replicate_labels": row["ko_replicate_labels"],
            "en_replicate_labels": row["en_replicate_labels"],
            "ko_evidence": row["ko_evidence"],
            "en_evidence": row["en_evidence"],
            "primary_eligible": row["primary_eligible"],
            "exclusion_reasons": row["exclusion_reasons"],
            "human_conclusion_ko": "",
            "human_conclusion_en": "",
            "human_party_match_status": "",
            "human_flip_status": "",
            "reviewer_notes": "",
            "reviewer_id": "",
            "review_status": "",
        })
    write_csv(
        args.output_dir / "human_conclusion_validation_v2.csv",
        human_rows,
        list(human_rows[0]) if human_rows else [],
    )

    broad_summary_path = args.output_dir / "summary.json"
    broad_summary = json.loads(broad_summary_path.read_text(encoding="utf-8"))
    broad_marker_snapshot = {
        "summary_sha256": sha256_file(broad_summary_path),
        "kr_oriented_marker": broad_summary["legal_system_signals"]["kr_oriented_marker"],
        "us_common_law_marker": broad_summary["legal_system_signals"]["us_common_law_marker"],
        "strong_a": broad_summary["legal_system_signals"]["strong_a"],
    }
    summary = {
        "version": VERSION,
        "legacy_provisional": legacy,
        "canonical_registry": {
            "case_party_n": len(registry),
            "language_party_unit_denominator": language_party_units,
            "source_party_set_mismatch_case_n": len({
                row["case_id"] for row in registry
                if str(row["source_party_set_mismatch"]).casefold() == "true"
            }),
        },
        "primary": {
            "eligible_party_n": len(eligible),
            "total_canonical_party_n": len(registry),
            "agreement_n": agreement_count,
            "disagreement_n": disagreement_count,
            "agreement_rate": agreement_count / len(eligible) if eligible else None,
            "disagreement_rate": disagreement_count / len(eligible) if eligible else None,
            "cohen_kappa_unweighted": cohen_kappa(primary_pairs, PRIMARY_CATEGORIES),
            "transition_matrix": matrix,
            "language_category_distributions": category_distributions,
            "eligible_case_n": len(eligible_by_case),
            "case_any_change_n": sum(case_any_change.values()),
            "case_any_change_rate": mean(list(case_any_change.values())),
            "direct_likely_unlikely_flip_party_n": sum(
                row["direct_likely_unlikely_flip"] for row in eligible
            ),
            "direct_likely_unlikely_flip_case_n": sum(case_direct_flip.values()),
            "likely_unlikely_conditional_uncertain_transition_party_n": sum(
                row["likely_unlikely_to_conditional_uncertain"] for row in eligible
            ),
            "likely_unlikely_conditional_uncertain_transition_case_n": sum(
                case_conditional_shift.values()
            ),
        },
        "replicate_aggregation": {
            "language_party_unit_denominator": language_party_units,
            "aggregation_status_by_language": {
                language: {
                    status: aggregation_counts[(language, status)]
                    for status in ("consensus", "replicate_disagreement", "incomplete_replicates")
                } for language in ("ko", "en")
            },
            "replicate_disagreement_n": sum(
                status == "replicate_disagreement"
                for _, status in aggregation_counts.elements()
            ),
            "replicate_disagreement_rate": (
                sum(
                    count for (language, status), count in aggregation_counts.items()
                    if status == "replicate_disagreement"
                ) / language_party_units
            ),
            "not_assessed_consensus_by_language": dict(not_assessed_counts),
            "not_assessed_consensus_n": sum(not_assessed_counts.values()),
            "not_assessed_consensus_rate": sum(not_assessed_counts.values()) / language_party_units,
            "mean_within_language_replicate_instability": mean([
                row["within_language_instability"] for row in per_party_divergence
            ]),
            "mean_cross_language_discordance": mean([
                row["cross_language_discordance"] for row in per_party_divergence
            ]),
            "cross_minus_within_case_clustered": {
                "case_denominator": len(case_level_differences),
                "mean_difference": mean(case_level_differences),
                "bootstrap_95_ci": bootstrap_ci(case_level_differences, args.seed),
                "permutation_p": permutation_p(case_level_differences, args.seed),
            },
        },
        "exclusions": {
            "excluded_party_n": len(pair_rows) - len(eligible),
            "reason_counts_nonexclusive": dict(exclusion_counts),
        },
        "sensitivity_including_not_assessed": {
            "eligible_party_n": len(sensitivity_rows),
            "agreement_n": sum(a == b for a, b in sensitivity_pairs),
            "agreement_rate": mean([a == b for a, b in sensitivity_pairs]),
            "cohen_kappa_unweighted": cohen_kappa(sensitivity_pairs, list(CONCLUSIONS)),
            "transition_matrix": sensitivity_matrix,
        },
        "by_origin": origins,
        "human_validation": {
            "rows": len(human_rows),
            "direct_flip_candidates": len(direct_rows),
            "status": "pending_human_review",
        },
        "broad_marker_snapshot_unchanged": broad_marker_snapshot,
    }
    write_json(args.output_dir / "conclusion_summary_v2.json", summary)

    manifest_path = args.output_dir / "conclusion_reanalysis_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = manifest["protected_file_hashes_before"]
    after = {
        path: sha256_file(Path(path))
        for path in before
    }
    manifest["protected_file_hashes_after"] = after
    manifest["protected_files_preserved"] = {
        path: before[path] == after[path] for path in before
    }
    manifest["all_protected_files_preserved"] = all(
        manifest["protected_files_preserved"].values()
    )
    write_json(manifest_path, manifest)

    report = f"""# Experiment 1 conclusion reanalysis v2

## Why the legacy conclusion results were provisional

The legacy analysis matched evaluator-generated party strings exactly and selected `Counter.most_common()` when three replicate labels tied. It reproduced {legacy['exact_string_matched_comparisons']} matched comparisons, {legacy['agreement']:.1%} agreement, κ={legacy['cohen_kappa_unweighted']:.3f}, {legacy['any_conclusion_change_cases']}/{legacy['case_denominator']} cases with any change, {legacy['direct_likely_unlikely_flip_cases']} direct-flip cases, and {legacy['modal_tie_matched_comparisons']} matched comparisons with a modal tie. The tie result depended on JSONL row order.

## Canonical-party primary result

- Canonical case-party registry: {len(registry)}
- Primary eligible parties: {len(eligible)} / {len(registry)}
- Agreement: {agreement_count}/{len(eligible)} = {agreement_count / len(eligible):.1%}
- Disagreement: {disagreement_count}/{len(eligible)} = {disagreement_count / len(eligible):.1%}
- Unweighted Cohen's κ: {summary['primary']['cohen_kappa_unweighted']:.3f}
- Eligible cases: {len(eligible_by_case)}
- Cases with any eligible conclusion change: {sum(case_any_change.values())}/{len(eligible_by_case)} = {mean(list(case_any_change.values())):.1%}
- Direct likely↔unlikely candidates: {summary['primary']['direct_likely_unlikely_flip_party_n']} parties in {summary['primary']['direct_likely_unlikely_flip_case_n']} cases
- Likely/unlikely↔conditional/uncertain: {summary['primary']['likely_unlikely_conditional_uncertain_transition_party_n']} parties in {summary['primary']['likely_unlikely_conditional_uncertain_transition_case_n']} cases

## Replicate stability and exclusions

- Replicate-disagreement language-party units: {summary['replicate_aggregation']['replicate_disagreement_n']}/{language_party_units} = {summary['replicate_aggregation']['replicate_disagreement_rate']:.1%}
- `not_assessed` consensus units: {summary['replicate_aggregation']['not_assessed_consensus_n']}/{language_party_units} = {summary['replicate_aggregation']['not_assessed_consensus_rate']:.1%}
- Excluded canonical parties: {summary['exclusions']['excluded_party_n']}/{len(registry)}
- Within-language instability: {summary['replicate_aggregation']['mean_within_language_replicate_instability']:.3f}
- Cross-language discordance: {summary['replicate_aggregation']['mean_cross_language_discordance']:.3f}
- Cross-minus-within: {summary['replicate_aggregation']['cross_minus_within_case_clustered']['mean_difference']:.3f}, 95% CI [{summary['replicate_aggregation']['cross_minus_within_case_clustered']['bootstrap_95_ci'][0]:.3f}, {summary['replicate_aggregation']['cross_minus_within_case_clustered']['bootstrap_95_ci'][1]:.3f}], permutation p={summary['replicate_aggregation']['cross_minus_within_case_clustered']['permutation_p']:.4g}

## Status

The broad legal-system-marker files and statistics were not modified. All v2 conclusions remain automatically coded and provisional until review of `human_conclusion_validation_v2.csv`, especially every direct-flip candidate, replicate disagreement, grouped/estate/duplicate mapping flag, and language-discordant conclusion.
"""
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "exp1_conclusion_reanalysis_v2.md").write_text(
        report, encoding="utf-8",
    )
    patch_proposal = """# Proposed patch to the paper / legacy report

Do not use the legacy party-level conclusion paragraph as a confirmed result. Replace it with the canonical-party v2 estimates in `reports/exp1_conclusion_reanalysis_v2.md`, explicitly identifying the former 57.9%, κ=0.424, 63/70, and two direct-flip cases as provisional exact-string/modal-tie results. Keep the broad legal-system-marker paragraph unchanged. Direct flips and all other v2 conclusion values remain provisional until the accompanying human-validation sheet is reviewed.
"""
    (args.reports_dir / "exp1_results_conclusion_patch_proposal_v2.md").write_text(
        patch_proposal, encoding="utf-8",
    )
    print(
        f"registry={len(registry)} eligible={len(eligible)} agreement={agreement_count} "
        f"cases={len(eligible_by_case)} direct_flip_parties={len(direct_rows)} "
        f"human_rows={len(human_rows)} protected={manifest['all_protected_files_preserved']}"
    )


if __name__ == "__main__":
    main()
