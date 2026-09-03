#!/usr/bin/env python3
"""Apply frozen measurement and analysis to the completed Experiment 1 generation."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from scripts.exp1.country_orientation_v1 import score_country_orientation
from scripts.exp1.run_implicit_jurisdiction_full import REPO_ROOT, read_jsonl, sha_file, write_csv, write_json
from scripts.lexicon.confirmatory_measurement_v1 import extract_features, precedent_citations, statute_citations


OUTPUT_DIR = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_full_v1"
RESPONSES_PATH = OUTPUT_DIR / "raw_responses.jsonl"
GENERATION_SUMMARY = OUTPUT_DIR / "generation_summary.json"
CASE_QC_PATH = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_final_qc_v6/final_case_qc_manifest.csv"
INVENTORY_PATH = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_final_qc_v6/country_orientation_marker_inventory.csv"
BASE_LEXICON = REPO_ROOT / "legal_lexicon/legal_lexicon_v1_0"
ORIENTATION_LEXICON = REPO_ROOT / "legal_lexicon/legal_lexicon_orientation_v1_1"
PCA_PATH = REPO_ROOT / "outputs_v2/robustness_gate2/multivariate_full_6d_no_procedure.json"
FACTS_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_fact_patterns_200_v4_0_2.jsonl"

PRIMARY_FEATURES = [
    "avg_chars_per_sentence", "statute_density", "precedent_density",
    "doctrine_density", "remedy_density", "party_arg_density",
]
UNCERTAINTY_RE = re.compile(
    r"(?i)insufficient (?:information|evidence)|record (?:does not|is insufficient)|cannot determine|"
    r"정보가 부족|기록만으로|판단하기 어렵|확정할 수 없|알 수 없"
)


def load_matcher(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load matcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_stats(frame: pd.DataFrame, outcome: str, positive_direction: str) -> dict:
    pivot = frame.pivot(index="case_id", columns="input_language", values=outcome).dropna()
    delta = (pivot["EN"] - pivot["KO"]).astype(float)
    n = len(delta)
    mean = float(delta.mean()) if n else math.nan
    sd = float(delta.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
    t_result = stats.ttest_1samp(delta, 0.0) if n > 1 else None
    try:
        wilcoxon_p = float(stats.wilcoxon(delta).pvalue) if n and np.any(delta != 0) else 1.0
    except ValueError:
        wilcoxon_p = math.nan
    return {
        "outcome": outcome, "n_cases": n, "difference_definition": "EN_minus_KO",
        "positive_direction": positive_direction, "mean_difference": mean,
        "sd_difference": sd, "se_difference": se,
        "ci95_low": mean - critical * se if n > 1 else math.nan,
        "ci95_high": mean + critical * se if n > 1 else math.nan,
        "paired_t": float(t_result.statistic) if t_result else math.nan,
        "paired_t_p_two_sided": float(t_result.pvalue) if t_result else math.nan,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "median_difference": float(delta.median()) if n else math.nan,
        "positive_case_proportion": float((delta > 0).mean()) if n else math.nan,
        "zero_case_proportion": float((delta == 0).mean()) if n else math.nan,
    }


def fit_mixed_model(data: pd.DataFrame) -> list[dict]:
    formula = (
        "country_orientation_score ~ C(input_language, Treatment(reference='KO')) * "
        "C(reasoning_effort, Treatment(reference='low')) + C(origin_country) + C(case_domain) + "
        "C(jurisdiction_salience_level) + C(lower_court_supplemented)"
    )
    rows = []
    try:
        fit = smf.mixedlm(formula, data=data, groups=data["case_id"]).fit(reml=False, method="lbfgs", maxiter=2000)
        for term in fit.params.index:
            rows.append({
                "analysis": "primary_mixed_model", "formula": formula, "term": term,
                "estimate": float(fit.params[term]), "std_error": float(fit.bse[term]),
                "z": float(fit.tvalues[term]), "p_value": float(fit.pvalues[term]),
                "converged": bool(fit.converged), "n_observations": int(fit.nobs),
            })
    except BaseException as exc:
        rows.append({"analysis": "primary_mixed_model", "formula": formula, "term": "FIT_ERROR", "estimate": "", "std_error": "", "z": "", "p_value": "", "converged": False, "n_observations": len(data), "error": f"{type(exc).__name__}: {exc}"})

    us = data[data["origin_country"] == "US"].copy()
    state_formula = (
        "country_orientation_score ~ C(input_language, Treatment(reference='KO')) * "
        "C(reasoning_effort, Treatment(reference='low')) + C(origin_state) + C(case_domain) + "
        "C(jurisdiction_salience_level) + C(lower_court_supplemented)"
    )
    try:
        fit = smf.mixedlm(state_formula, data=us, groups=us["case_id"]).fit(reml=False, method="lbfgs", maxiter=2000)
        for term in fit.params.index:
            rows.append({
                "analysis": "US_origin_state_fixed_effect_sensitivity", "formula": state_formula, "term": term,
                "estimate": float(fit.params[term]), "std_error": float(fit.bse[term]),
                "z": float(fit.tvalues[term]), "p_value": float(fit.pvalues[term]),
                "converged": bool(fit.converged), "n_observations": int(fit.nobs),
            })
    except BaseException as exc:
        rows.append({"analysis": "US_origin_state_fixed_effect_sensitivity", "formula": state_formula, "term": "FIT_ERROR", "estimate": "", "std_error": "", "z": "", "p_value": "", "converged": False, "n_observations": len(us), "error": f"{type(exc).__name__}: {exc}"})
    return rows


def main() -> None:
    generation = json.loads(GENERATION_SUMMARY.read_text(encoding="utf-8"))
    responses = read_jsonl(RESPONSES_PATH)
    if len(responses) != 1920 or generation["response_records"] != 1920:
        raise RuntimeError("Analysis requires the complete 1,920-record generation checkpoint")
    facts = {row["case_id"]: row for row in read_jsonl(FACTS_PATH)}
    case_qc = {row["case_id"]: row for row in read_csv(CASE_QC_PATH)}
    inventory = {row["concept_id"]: row for row in read_csv(INVENTORY_PATH)}
    base_matcher = load_matcher(BASE_LEXICON / "lexicon_matcher_v1_0.py", "base_lexicon_matcher")
    orientation_matcher = load_matcher(ORIENTATION_LEXICON / "lexicon_matcher_v1_1.py", "orientation_lexicon_matcher")

    feature_rows, occurrence_rows, marker_rows, authority_rows = [], [], [], []
    for response in sorted(responses, key=lambda row: int(row["randomized_order"])):
        text = response.get("response_text") or ""
        language = response["input_language"].lower()
        parser_country = "KR" if language == "ko" else "US"
        base_features, base_hits = extract_features(
            text, parser_country, base_matcher,
            BASE_LEXICON / "surface_forms_v1_0.csv", BASE_LEXICON / "concepts_v1_0.csv",
        )
        orientation_hits = orientation_matcher.match_text(
            text, ORIENTATION_LEXICON / "surface_forms_v1_1.csv",
            ORIENTATION_LEXICON / "concepts_v1_1.csv", language,
        )
        orientation_hits = [hit for hit in orientation_hits if int(hit.get("orientation_include", 0)) == 1]
        orientation = score_country_orientation(orientation_hits, inventory)
        statutes = statute_citations(text, parser_country)
        precedents = precedent_citations(text, parser_country)
        unsupported_authority = bool(statutes or precedents)
        metadata = {
            "request_id": response["request_id"], "case_id": response["case_id"],
            "origin_country": response["origin_country"], "origin_state": response.get("origin_state") or "",
            "case_domain": response["case_domain"], "input_language": response["input_language"],
            "reasoning_effort": response["reasoning_effort"], "replicate": int(response["replicate"]),
            "model_id": response["model_id"], "model_returned": response.get("model_returned") or "",
        }
        diagnostics = {
            "output_character_count": len(text), "output_word_count": len(text.split()),
            "uncertainty_disclosure": int(bool(UNCERTAINTY_RE.search(text))),
            "authority_mention_count": len(statutes) + len(precedents),
            "authority_hallucination_status": "requires_human_validation" if unsupported_authority else "none_detected",
        }
        feature_rows.append({**metadata, **base_features, **orientation, **diagnostics})
        occurrence_rows.append({
            **metadata,
            "lexicon_hit_count": len(base_hits),
            "orientation_hit_occurrences": len(orientation_hits),
            "KR_orientation_hit_occurrences": sum(inventory[h["concept_id"]]["jurisdiction_family"] == "KR_specific" for h in orientation_hits),
            "US_orientation_hit_occurrences": sum(inventory[h["concept_id"]]["jurisdiction_family"] in {"generic_US", "state_specific_US"} for h in orientation_hits),
            "orientation_occurrence_score_sensitivity": (
                (sum(inventory[h["concept_id"]]["jurisdiction_family"] in {"generic_US", "state_specific_US"} for h in orientation_hits)
                 - sum(inventory[h["concept_id"]]["jurisdiction_family"] == "KR_specific" for h in orientation_hits))
                / (len(orientation_hits) + 1)
            ),
        })
        for hit in orientation_hits:
            marker_rows.append({
                **metadata, "concept_id": hit["concept_id"],
                "jurisdiction_family": inventory[hit["concept_id"]]["jurisdiction_family"],
                "matched_text": hit["matched_text"], "start": hit["start"], "end": hit["end"],
            })
        for kind, matches in (("statute", statutes), ("precedent", precedents)):
            for start, end, mention in matches:
                authority_rows.append({
                    **metadata, "authority_type": kind, "mention": mention, "start": start, "end": end,
                    "hallucination_status": "requires_human_validation",
                })

    write_csv(OUTPUT_DIR / "generated_features_primary.csv", feature_rows, list(feature_rows[0]))
    write_csv(OUTPUT_DIR / "generated_features_occurrence.csv", occurrence_rows, list(occurrence_rows[0]))
    write_csv(OUTPUT_DIR / "jurisdiction_markers.csv", marker_rows, list(marker_rows[0]) if marker_rows else [
        "request_id", "case_id", "origin_country", "origin_state", "case_domain", "input_language",
        "reasoning_effort", "replicate", "model_id", "model_returned", "concept_id",
        "jurisdiction_family", "matched_text", "start", "end",
    ])
    write_csv(OUTPUT_DIR / "authority_mentions.csv", authority_rows, list(authority_rows[0]) if authority_rows else [
        "request_id", "case_id", "origin_country", "origin_state", "case_domain", "input_language",
        "reasoning_effort", "replicate", "model_id", "model_returned", "authority_type", "mention",
        "start", "end", "hallucination_status",
    ])

    data = pd.DataFrame(feature_rows)
    numeric = [
        "country_orientation_score", "N_KR", "N_US", "statute_density", "precedent_density",
        "doctrine_density", "remedy_density", "party_arg_density", "output_character_count",
        "output_word_count", "uncertainty_disclosure", "authority_mention_count",
    ]
    aggregate_columns = list(dict.fromkeys(numeric + PRIMARY_FEATURES))
    aggregate = data.groupby([
        "case_id", "origin_country", "origin_state", "case_domain", "input_language", "reasoning_effort"
    ], as_index=False)[aggregate_columns].mean()
    qc_frame = pd.DataFrame(case_qc.values())[[
        "case_id", "jurisdiction_salience_level", "jurisdiction_salience_tags", "lower_court_supplemented"
    ]]
    aggregate = aggregate.merge(qc_frame, on="case_id", how="left", validate="many_to_one")

    paired_rows = []
    overall_by_language = aggregate.groupby(["case_id", "input_language"], as_index=False)[numeric].mean()
    paired_rows.append({"stratum": "overall", **paired_stats(overall_by_language, "country_orientation_score", "more_US_oriented")})
    for effort in ("low", "medium", "high"):
        subset = aggregate[aggregate["reasoning_effort"] == effort]
        paired_rows.append({"stratum": f"effort_{effort}", **paired_stats(subset, "country_orientation_score", "more_US_oriented")})
    write_csv(OUTPUT_DIR / "orientation_paired_effects.csv", paired_rows, list(paired_rows[0]))

    effort_rows = []
    outcomes = {
        "country_orientation_score": "more_US_oriented",
        "N_KR": "more_KR_markers_in_EN",
        "N_US": "more_US_markers_in_EN",
        "statute_density": "more_statute_use_in_EN",
        "precedent_density": "more_precedent_use_in_EN",
        "authority_mention_count": "more_authority_mentions_in_EN",
        "uncertainty_disclosure": "more_uncertainty_disclosure_in_EN",
        "output_character_count": "longer_EN_output",
    }
    for effort in ("low", "medium", "high"):
        subset = aggregate[aggregate["reasoning_effort"] == effort]
        for outcome, direction in outcomes.items():
            effort_rows.append({"reasoning_effort": effort, **paired_stats(subset, outcome, direction)})
    write_csv(OUTPUT_DIR / "effort_results.csv", effort_rows, list(effort_rows[0]))

    origin_rows = []
    for origin in ("KR", "US"):
        for effort in ("all", "low", "medium", "high"):
            subset = aggregate[aggregate["origin_country"] == origin]
            if effort != "all":
                subset = subset[subset["reasoning_effort"] == effort]
            else:
                subset = subset.groupby(["case_id", "input_language"], as_index=False)[numeric].mean()
            origin_rows.append({"origin_country": origin, "reasoning_effort": effort, **paired_stats(subset, "country_orientation_score", "more_US_oriented")})
    write_csv(OUTPUT_DIR / "origin_stratified_results.csv", origin_rows, list(origin_rows[0]))

    salience_rows = []
    for level in sorted(aggregate["jurisdiction_salience_level"].dropna().unique()):
        subset = aggregate[aggregate["jurisdiction_salience_level"] == level]
        subset = subset.groupby(["case_id", "input_language"], as_index=False)[numeric].mean()
        salience_rows.append({"jurisdiction_salience_level": level, **paired_stats(subset, "country_orientation_score", "more_US_oriented")})
    write_csv(OUTPUT_DIR / "jurisdiction_salience_sensitivity.csv", salience_rows, list(salience_rows[0]))

    mixed_rows = fit_mixed_model(aggregate)
    mixed_columns = ["analysis", "formula", "term", "estimate", "std_error", "z", "p_value", "converged", "n_observations", "error"]
    write_csv(OUTPUT_DIR / "mixed_model_results.csv", mixed_rows, mixed_columns)

    pca = json.loads(PCA_PATH.read_text(encoding="utf-8"))
    matrix = data[PRIMARY_FEATURES].astype(float).to_numpy()
    standardized = (matrix - np.array(pca["scaler"]["mean"])) / np.array(pca["scaler"]["scale"])
    scores = standardized @ np.array(pca["pca"]["components"]).T
    pca_rows = []
    for index, row in data.reset_index(drop=True).iterrows():
        pca_rows.append({
            "request_id": row["request_id"], "case_id": row["case_id"], "origin_country": row["origin_country"],
            "input_language": row["input_language"], "reasoning_effort": row["reasoning_effort"],
            "replicate": row["replicate"], **{f"PC{component + 1}": float(scores[index, component]) for component in range(scores.shape[1])},
        })
    write_csv(OUTPUT_DIR / "llm_pca_scores.csv", pca_rows, list(pca_rows[0]))
    pca_frame = pd.DataFrame(pca_rows)
    pc_columns = [column for column in pca_frame if column.startswith("PC")]
    centroids = pca_frame.groupby(["input_language", "reasoning_effort"], as_index=False)[pc_columns].mean().to_dict("records")
    write_csv(OUTPUT_DIR / "llm_pca_centroids.csv", centroids, list(centroids[0]))

    response_by_id = {row["request_id"]: row for row in responses}
    sample_rows = []
    data["sample_score"] = data["request_id"].map(lambda value: hashlib.sha256(f"20260730|human-validation|{value}".encode()).hexdigest())
    for _, stratum in data.groupby(["origin_country", "input_language", "reasoning_effort"]):
        for _, selected in stratum.nsmallest(4, "sample_score").iterrows():
            response = response_by_id[selected["request_id"]]
            sample_rows.append({
                "request_id": selected["request_id"], "case_id": selected["case_id"],
                "origin_country": selected["origin_country"], "input_language": selected["input_language"],
                "reasoning_effort": selected["reasoning_effort"], "replicate": selected["replicate"],
                "response_text": response["response_text"], "human_review_status": "pending",
                "human_accepted": "", "human_reviewer_id": "", "human_notes": "",
            })
    write_csv(OUTPUT_DIR / "human_validation_sample.csv", sample_rows, list(sample_rows[0]))

    overall = paired_rows[0]
    p_value = float(overall["paired_t_p_two_sided"])
    effort_means = [float(row["mean_difference"]) for row in paired_rows[1:]]
    if float(overall["mean_difference"]) > 0 and p_value < 0.05 and all(value > 0 for value in effort_means):
        conclusion = "A. clear language-associated jurisdictional orientation shift"
    elif p_value < 0.05 and len({value > 0 for value in effort_means}) > 1:
        conclusion = "C. effort-dependent or subset-specific effect"
    elif p_value < 0.05:
        conclusion = "B. mainly lexical/style shift with weak jurisdictional-doctrine shift"
    else:
        conclusion = "D. little/no systematic language-associated orientation effect"

    artifact_names = [
        "raw_responses.jsonl", "request_attempt_log.jsonl", "generation_qc.csv", "failed_requests.csv",
        "generated_features_primary.csv", "generated_features_occurrence.csv", "jurisdiction_markers.csv",
        "authority_mentions.csv", "orientation_paired_effects.csv", "effort_results.csv",
        "origin_stratified_results.csv", "jurisdiction_salience_sensitivity.csv", "mixed_model_results.csv",
        "llm_pca_scores.csv", "llm_pca_centroids.csv", "human_validation_sample.csv",
    ]
    analysis_summary = {
        "status": "ANALYSIS_COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tested_model_scope": "gpt-5.6-luna only", "n_generated_documents": len(data),
        "primary_delta_definition": "orientation_EN - orientation_KO",
        "primary_mean_delta": overall["mean_difference"], "primary_p_two_sided": overall["paired_t_p_two_sided"],
        "classification": conclusion, "human_validation_sample_size": len(sample_rows),
        "multi_model_replication_status": "deferred_to_future_robustness",
    }
    write_json(OUTPUT_DIR / "analysis_summary.json", analysis_summary)
    report = [
        "# Experiment 1 full-run report", "", f"Status: **{generation['status']} / ANALYSIS_COMPLETE**", "",
        "All findings are scoped to the tested gpt-5.6-luna conditions.", "",
        f"- Planned/generated documents: 1,920/{len(data)}",
        f"- Generation QC pass/fail: {generation['generation_qc_pass']}/{generation['generation_qc_fail']}",
        f"- Technical retry attempts: {generation['technical_retry_attempts']}",
        f"- Overall mean Delta (EN - KO): {float(overall['mean_difference']):.6f}",
        f"- Paired t-test two-sided p: {float(overall['paired_t_p_two_sided']):.6g}",
        f"- Effort-specific mean Delta: low={effort_means[0]:.6f}, medium={effort_means[1]:.6f}, high={effort_means[2]:.6f}",
        f"- Result classification: **{conclusion}**",
        f"- Authority mentions requiring human hallucination validation: {len(authority_rows)}",
        f"- Human-validation sample prepared: {len(sample_rows)}; no row was automatically human-accepted.",
        "- Gate 2 projection: frozen restricted 6D PCA; procedure excluded; PCA was not refit.",
        "- Cross-model replication: deferred_to_future_robustness.", "",
    ]
    (OUTPUT_DIR / "EXP1_FULL_RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    artifact_names += ["analysis_summary.json", "EXP1_FULL_RUN_REPORT.md"]
    final_manifest = json.loads((OUTPUT_DIR / "pre_generation_experiment_manifest.json").read_text(encoding="utf-8"))
    final_manifest.update({
        "status": "FULL_RUN_AND_ANALYSIS_COMPLETE", "generation_summary": generation,
        "analysis_summary": analysis_summary, "generation_start_timestamp": min(row["request_started_at"] for row in responses),
        "generation_completion_timestamp": generation["completed_at_utc"],
        "final_artifact_sha256": {name: sha_file(OUTPUT_DIR / name) for name in artifact_names},
    })
    write_json(OUTPUT_DIR / "final_experiment_manifest.json", final_manifest)
    checksum_paths = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name != "SHA256SUMS_FINAL.txt")
    (OUTPUT_DIR / "SHA256SUMS_FINAL.txt").write_text(
        "".join(f"{sha_file(path)}  {path.name}\n" for path in checksum_paths), encoding="ascii", newline="\n"
    )
    print(json.dumps(analysis_summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
