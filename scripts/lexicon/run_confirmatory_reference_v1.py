#!/usr/bin/env python3
"""Freeze lexicon v1.0 and build the confirmatory real-reference feature/PCA space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_MODULE_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_MODULE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_REPO_ROOT))

from scripts.lexicon.confirmatory_measurement_v1 import (
    CATEGORIES,
    PRIMARY_FEATURES,
    SENSITIVITY_FEATURES,
    extract_features,
)


REPO_ROOT = _MODULE_REPO_ROOT
PREVALIDATION_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_tuning_review_v0_3"
PREVALIDATION_SEAL = PREVALIDATION_DIR / "PREVALIDATION_SHA256.json"
PREVALIDATION_RUN_MANIFEST = PREVALIDATION_DIR / "run_manifest.json"
VALIDATION_LOCK = PREVALIDATION_DIR / "validation_case_manifest_LOCKED.csv"
LEXICON_OUTPUT = REPO_ROOT / "outputs_v2/legal_lexicon_v1_0"
ANALYSIS_OUTPUT = REPO_ROOT / "outputs_v2/real_reference_confirmatory_v1"
CORPUS_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
FREEZE_MANIFEST_PATH = REPO_ROOT / "outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json"
MEASUREMENT_PATH = Path(__file__).with_name("confirmatory_measurement_v1.py")
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_RUNS = 10_000
PERMDISP_SEED = 20260827
PERMDISP_PERMUTATIONS = 10_000

LEXICON_FILES = {
    "concepts_prevalidation_v0_3.csv": "concepts_v1_0.csv",
    "surface_forms_prevalidation_v0_3.csv": "surface_forms_v1_0.csv",
    "lexicon_active_primary_prevalidation_v0_3.csv": "lexicon_active_primary_v1_0.csv",
    "lexicon_combined_prevalidation_v0_3.csv": "lexicon_combined_v1_0.csv",
    "lexicon_matcher_prevalidation_v0_3.py": "lexicon_matcher_v1_0.py",
}

FEATURE_COLUMNS = [
    "case_id", "origin_country", "origin_state", "primary_domain",
    "doc_length_sentences", "doc_length_chars", "avg_chars_per_sentence",
    "statute_occurrence_count", "unique_statute_count", "statute_density",
    "precedent_occurrence_count", "unique_precedent_count", "precedent_density",
]
for _category in CATEGORIES:
    FEATURE_COLUMNS.extend([
        f"{_category}_hit_count", f"{_category}_unique_concepts",
        f"{_category}_unique_families", f"{_category}_density",
    ])
FEATURE_COLUMNS.append("log1p_doc_length_sentences")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str], bom: bool = True) -> None:
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: _csv_value(row.get(column, "")) for column in columns} for row in rows)


def _csv_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return value


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_source() -> tuple[str, dict]:
    manifest = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN"
    expected = [value for key, value in manifest["artifact_sha256"].items() if Path(key).name == CORPUS_PATH.name]
    assert len(expected) == 1
    actual = sha256_file(CORPUS_PATH)
    assert actual == expected[0]
    return actual, manifest


def verify_prevalidation_seal() -> dict[str, str]:
    seal = json.loads(PREVALIDATION_SEAL.read_text(encoding="utf-8"))
    assert seal["status"] == "PRE_VALIDATION_HASHED_CANDIDATE"
    actual = {}
    for name, expected in sorted(seal["files"].items()):
        path = PREVALIDATION_DIR / name
        actual[name] = sha256_file(path)
        assert actual[name] == expected, f"Pre-validation seal mismatch: {name}"
    assert set(LEXICON_FILES) <= set(actual)
    return actual


def freeze_lexicon(output_dir: Path = LEXICON_OUTPUT) -> dict[str, str]:
    sealed = verify_prevalidation_seal()
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in LEXICON_FILES.items():
        source, target = PREVALIDATION_DIR / source_name, output_dir / target_name
        shutil.copyfile(source, target)
        assert source.read_bytes() == target.read_bytes()
    copied_hashes = {target: sha256_file(output_dir / target) for target in LEXICON_FILES.values()}
    manifest = {
        "lexicon_version": "v1.0",
        "status": "FROZEN_AFTER_HELD_OUT_VALIDATION_NO_CHANGES",
        "source_prevalidation_version": "v0.3",
        "source_prevalidation_seal_sha256": sha256_file(PREVALIDATION_SEAL),
        "source_hashes": sealed,
        "copied_file_hashes": copied_hashes,
        "content_byte_equivalent_to_prevalidation": True,
        "validation_derived_modifications": 0,
        "validation_used_for_lexicon_modification": False,
        "confirmatory_used_for_lexicon_modification": False,
        "generation_timestamp_utc": json.loads(PREVALIDATION_RUN_MANIFEST.read_text(encoding="utf-8"))["generated_at_utc"],
        "timestamp_policy": "deterministic pre-validation candidate timestamp",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "README.md").write_text(
        "# Legal lexicon v1.0\n\nFrozen byte-equivalent copy of sealed pre-validation v0.3. "
        "Held-out validation produced zero lexicon changes. Confirmatory results may not modify this artifact.\n",
        encoding="utf-8", newline="\n",
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii", newline="\n"
    )
    return {path.name: sha256_file(path) for path in output_dir.iterdir() if path.is_file()}


def load_matcher(path: Path):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("frozen_lexicon_matcher_v1_0", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def load_confirmatory_documents() -> tuple[list[dict], set[str]]:
    documents, development_ids, seen = [], set(), set()
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case_id = str(record["case_id"])
            assert case_id not in seen
            seen.add(case_id)
            if record["analysis_split"] != "confirmatory":
                development_ids.add(case_id)
                # main_opinion_text is not accessed for development cases.
                continue
            text = record.get("main_opinion_text")
            assert isinstance(text, str) and text.strip()
            documents.append({
                "case_id": case_id, "origin_country": record["origin_country"],
                "origin_state": str(record.get("origin_state") or ""),
                "primary_domain": str(record.get("primary_domain") or ""), "text": text,
            })
    documents.sort(key=lambda row: row["case_id"])
    assert len(documents) == 160 and len(development_ids) == 40
    assert Counter(row["origin_country"] for row in documents) == Counter({"KR": 80, "US": 80})
    assert not ({row["case_id"] for row in documents} & development_ids)
    return documents, development_ids


def extract_reference_features(documents: Sequence[dict], matcher) -> tuple[list[dict], list[dict]]:
    rows, hit_rows = [], []
    for document in documents:
        features, hits = extract_features(
            document["text"], document["origin_country"], matcher,
            LEXICON_OUTPUT / "surface_forms_v1_0.csv", LEXICON_OUTPUT / "concepts_v1_0.csv",
        )
        rows.append({key: document[key] for key in ("case_id", "origin_country", "origin_state", "primary_domain")} | features)
        for index, hit in enumerate(hits, 1):
            start, end = int(hit["start"]), int(hit["end"])
            assert document["text"][start:end] == hit["matched_text"]
            hit_rows.append({
                "case_id": document["case_id"], "hit_id": f"{document['case_id']}::C{index:05d}",
                "origin_country": document["origin_country"], "category": hit["category"],
                "concept_id": hit["concept_id"], "concept_family": hit["concept_family"],
                "surface_id": hit["surface_id"], "matched_text": hit["matched_text"],
                "start": start, "end": end,
            })
    assert len(rows) == 160 and all(float(row["doc_length_sentences"]) > 0 for row in rows)
    assert all(math.isfinite(float(row[feature])) for row in rows for feature in PRIMARY_FEATURES)
    return rows, hit_rows


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return float(q1), float(median), float(q3)


def descriptive_rows(feature_rows: Sequence[dict]) -> list[dict]:
    output = []
    for feature in PRIMARY_FEATURES:
        for country in ("KR", "US"):
            values = np.array([float(row[feature]) for row in feature_rows if row["origin_country"] == country])
            q1, median, q3 = quantiles(values)
            output.append({
                "feature": feature, "country": country, "n": len(values),
                "mean": np.mean(values), "sd": np.std(values, ddof=1), "median": median,
                "q1": q1, "q3": q3, "iqr": q3 - q1, "min": np.min(values), "max": np.max(values),
            })
    return output


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    sorted_b = np.sort(b)
    greater = np.searchsorted(sorted_b, a, side="left").sum()
    less = (len(sorted_b) - np.searchsorted(sorted_b, a, side="right")).sum()
    return float((greater - less) / (len(a) * len(b)))


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    deltas = np.empty(BOOTSTRAP_RUNS, dtype=float)
    for index in range(BOOTSTRAP_RUNS):
        sample_a = a[rng.integers(0, len(a), len(a))]
        sample_b = b[rng.integers(0, len(b), len(b))]
        deltas[index] = cliffs_delta(sample_a, sample_b)
    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    size = len(p_values)
    order = np.argsort(np.asarray(p_values), kind="stable")
    adjusted = np.empty(size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (size - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def inferential_rows(feature_rows: Sequence[dict]) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output, p_values = [], []
    for feature in PRIMARY_FEATURES:
        kr = np.array([float(row[feature]) for row in feature_rows if row["origin_country"] == "KR"])
        us = np.array([float(row[feature]) for row in feature_rows if row["origin_country"] == "US"])
        test = mannwhitneyu(kr, us, alternative="two-sided", method="asymptotic")
        kr_q1, kr_median, kr_q3 = quantiles(kr)
        us_q1, us_median, us_q3 = quantiles(us)
        ci_low, ci_high = bootstrap_delta_ci(kr, us, rng)
        p_values.append(float(test.pvalue))
        output.append({
            "feature": feature, "KR_n": len(kr), "US_n": len(us),
            "KR_median": kr_median, "KR_Q1": kr_q1, "KR_Q3": kr_q3,
            "US_median": us_median, "US_Q1": us_q1, "US_Q3": us_q3,
            "mannwhitney_U": float(test.statistic), "p_raw": float(test.pvalue), "p_holm": 0.0,
            "cliffs_delta": cliffs_delta(kr, us),
            "cliffs_delta_ci_low": ci_low, "cliffs_delta_ci_high": ci_high,
        })
    adjusted = holm_adjust(p_values)
    assert len(adjusted) == len(PRIMARY_FEATURES) == 7
    for row, value in zip(output, adjusted):
        row["p_holm"] = value
    return output


def fit_pca(feature_rows: Sequence[dict], features: Sequence[str]) -> tuple[StandardScaler, PCA, np.ndarray]:
    matrix = np.array([[float(row[feature]) for feature in features] for row in feature_rows], dtype=float)
    assert matrix.shape == (160, len(features)) and np.isfinite(matrix).all()
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    pca = PCA().fit(standardized)
    scores = pca.transform(standardized)
    return scaler, pca, scores


def scaler_json(scaler: StandardScaler, features: Sequence[str]) -> dict:
    return {
        "feature_order": list(features), "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(),
        "variances": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_),
        "library": "scikit-learn", "library_version": sklearn.__version__,
        "fit_population": "real-reference confirmatory 160 only",
    }


def pca_json(pca: PCA, features: Sequence[str]) -> dict:
    return {
        "feature_order": list(features), "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(), "mean": pca.mean_.tolist(),
        "n_samples": int(pca.n_samples_), "library": "scikit-learn",
        "library_version": sklearn.__version__, "fit_population": "real-reference confirmatory 160 only",
    }


def pca_tables(feature_rows: Sequence[dict], features: Sequence[str], pca: PCA, scores: np.ndarray):
    pc_names = [f"PC{index + 1}" for index in range(len(features))]
    score_rows = []
    for row, values in zip(feature_rows, scores):
        score_rows.append({"case_id": row["case_id"], "origin_country": row["origin_country"]} |
                          {name: value for name, value in zip(pc_names, values)})
    loading_rows = [
        {"feature": feature} | {name: pca.components_[index, feature_index]
                                for index, name in enumerate(pc_names)}
        for feature_index, feature in enumerate(features)
    ]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    explained_rows = [{
        "component": name, "explained_variance": pca.explained_variance_[index],
        "explained_variance_ratio": pca.explained_variance_ratio_[index],
        "cumulative_explained_variance_ratio": cumulative[index],
        "singular_value": pca.singular_values_[index],
    } for index, name in enumerate(pc_names)]
    return score_rows, loading_rows, explained_rows


def save_scatter(feature_rows: Sequence[dict], scores: np.ndarray, pca: PCA, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    countries = np.array([row["origin_country"] for row in feature_rows])
    colors = {"KR": "#2f6b9a", "US": "#c25b3f"}
    for country in ("KR", "US"):
        selected = scores[countries == country]
        axis.scatter(selected[:, 0], selected[:, 1], s=24, alpha=0.65, label=country, color=colors[country])
        centroid = selected[:, :2].mean(axis=0)
        axis.scatter(centroid[0], centroid[1], s=150, marker="X", color=colors[country], edgecolor="black")
    axis.axhline(0, color="#999999", linewidth=0.6)
    axis.axvline(0, color="#999999", linewidth=0.6)
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% explained)")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% explained)")
    axis.set_title("Confirmatory real-reference PCA (visualization, not a significance test)")
    axis.legend()
    fig.savefig(path, dpi=180, metadata={"Software": "comparative-law-llm frozen reference pipeline v1"})
    plt.close(fig)


def pseudo_f(distances: np.ndarray, labels: np.ndarray) -> float:
    groups = [distances[labels == country] for country in ("KR", "US")]
    grand = distances.mean()
    between = sum(len(group) * (group.mean() - grand) ** 2 for group in groups)
    within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    return float((between / 1) / (within / (len(distances) - 2))) if within > 0 else float("inf")


def permdisp(feature_rows: Sequence[dict], standardized: np.ndarray):
    labels = np.array([row["origin_country"] for row in feature_rows])
    distances = np.empty(len(labels), dtype=float)
    for country in ("KR", "US"):
        mask = labels == country
        centroid = standardized[mask].mean(axis=0)
        distances[mask] = np.linalg.norm(standardized[mask] - centroid, axis=1)
    observed = pseudo_f(distances, labels)
    rng = np.random.default_rng(PERMDISP_SEED)
    exceed = 0
    for _ in range(PERMDISP_PERMUTATIONS):
        permuted = rng.permutation(labels)
        perm_distances = np.empty(len(labels), dtype=float)
        for country in ("KR", "US"):
            mask = permuted == country
            centroid = standardized[mask].mean(axis=0)
            perm_distances[mask] = np.linalg.norm(standardized[mask] - centroid, axis=1)
        exceed += pseudo_f(perm_distances, permuted) >= observed
    p_value = (exceed + 1) / (PERMDISP_PERMUTATIONS + 1)
    summary = []
    for country in ("KR", "US"):
        values = distances[labels == country]
        summary.append({
            "origin_country": country, "n": len(values), "mean_distance": values.mean(),
            "median_distance": np.median(values), "pseudo_F": observed,
            "permutation_p": p_value, "permutations": PERMDISP_PERMUTATIONS,
        })
    detail = [{
        "case_id": row["case_id"], "origin_country": row["origin_country"],
        "distance_to_group_centroid": distance,
    } for row, distance in zip(feature_rows, distances)]
    return summary, detail


def limitations_rows() -> list[dict]:
    return [
        {"limitation_id": "L001", "stage": "measurement", "description": "Rule-based sentence and citation parsers are frozen but not exhaustively gold-validated.", "action": "Report as limitation; do not revise from confirmatory results."},
        {"limitation_id": "L002", "stage": "validation", "description": "No exhaustive gold-span annotation is available, so recall and F1 are not claimed.", "action": "Future independent annotation only; v1.0 remains unchanged."},
        {"limitation_id": "L003", "stage": "PCA", "description": "PCA visualization is descriptive and is not a test of group separation.", "action": "Use inferential feature tests and corrected p-values for feature-wise comparisons."},
    ]


def readme_text(pca7: PCA) -> str:
    loadings = pca7.components_[:2]
    summaries = []
    for component_index in range(2):
        order = np.argsort(loadings[component_index])
        summaries.append(
            f"PC{component_index + 1}: strongest negative {PRIMARY_FEATURES[order[0]]} "
            f"({loadings[component_index, order[0]]:.4f}); strongest positive "
            f"{PRIMARY_FEATURES[order[-1]]} ({loadings[component_index, order[-1]]:.4f})"
        )
    return f"""# Confirmatory real-reference analysis v1

This analysis uses exactly 160 frozen confirmatory real judgments (KR 80, US 80). Development cases and
LLM outputs are excluded. The byte-equivalent legal lexicon v1.0 and measurement-code hashes were frozen
before confirmatory extraction and remained unchanged. Confirmatory observations did not modify any rule.

Seven primary features are reported. Densities use unique citations/concepts per frozen sentence count,
multiplied by 100. Medians and IQRs are primary descriptive summaries. Mann-Whitney tests are two-sided;
Holm-adjusted p-values, not raw p-values alone, govern multiplicity-aware interpretation. Cliff's delta is
KR relative to US with deterministic {BOOTSTRAP_RUNS:,}-bootstrap 95% intervals (seed {BOOTSTRAP_SEED}).

The 7D StandardScaler and PCA were fitted only on the confirmatory real-reference 160. Future LLM judgments
must use the saved scaler/PCA transform without refitting. The separate 8D sensitivity PCA adds log sentence
count and does not replace the primary 7D model. PCA separation is not itself a significance test.

- PC1 explained variance: {pca7.explained_variance_ratio_[0] * 100:.4f}%
- PC2 explained variance: {pca7.explained_variance_ratio_[1] * 100:.4f}%
- PC1+PC2 cumulative: {(pca7.explained_variance_ratio_[:2].sum()) * 100:.4f}%
- {summaries[0]}
- {summaries[1]}

PERMDISP is a secondary permutation analysis of within-country dispersion in standardized 7D space, using
{PERMDISP_PERMUTATIONS:,} deterministic permutations (seed {PERMDISP_SEED}); it does not test feature means.
"""


def write_analysis(
    output_dir: Path, feature_rows: list[dict], hit_rows: list[dict], source_hash: str,
    source_manifest: dict, lexicon_hashes: dict[str, str], code_hashes: dict[str, str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = descriptive_rows(feature_rows)
    inferences = inferential_rows(feature_rows)
    scaler7, pca7, scores7 = fit_pca(feature_rows, PRIMARY_FEATURES)
    scaler8, pca8, scores8 = fit_pca(feature_rows, SENSITIVITY_FEATURES)
    score7, load7, explained7 = pca_tables(feature_rows, PRIMARY_FEATURES, pca7, scores7)
    score8, load8, explained8 = pca_tables(feature_rows, SENSITIVITY_FEATURES, pca8, scores8)

    write_csv(output_dir / "reference_features_160.csv", feature_rows, FEATURE_COLUMNS)
    write_csv(output_dir / "descriptive_stats_primary.csv", descriptions,
              ["feature", "country", "n", "mean", "sd", "median", "q1", "q3", "iqr", "min", "max"])
    write_csv(output_dir / "mannwhitney_cliffs_primary.csv", inferences,
              ["feature", "KR_n", "US_n", "KR_median", "KR_Q1", "KR_Q3", "US_median", "US_Q1", "US_Q3",
               "mannwhitney_U", "p_raw", "p_holm", "cliffs_delta", "cliffs_delta_ci_low", "cliffs_delta_ci_high"])
    write_csv(output_dir / "reference_lexicon_hits_160.csv", hit_rows,
              ["case_id", "hit_id", "origin_country", "category", "concept_id", "concept_family", "surface_id", "matched_text", "start", "end"])
    write_csv(output_dir / "pca_scores_7d.csv", score7, ["case_id", "origin_country"] + [f"PC{i}" for i in range(1, 8)])
    write_csv(output_dir / "pca_loadings_7d.csv", load7, ["feature"] + [f"PC{i}" for i in range(1, 8)])
    write_csv(output_dir / "pca_explained_variance_7d.csv", explained7,
              ["component", "explained_variance", "explained_variance_ratio", "cumulative_explained_variance_ratio", "singular_value"])
    write_csv(output_dir / "pca_scores_8d_sensitivity.csv", score8, ["case_id", "origin_country"] + [f"PC{i}" for i in range(1, 9)])
    write_csv(output_dir / "pca_loadings_8d_sensitivity.csv", load8, ["feature"] + [f"PC{i}" for i in range(1, 9)])
    write_csv(output_dir / "pca_explained_variance_8d_sensitivity.csv", explained8,
              ["component", "explained_variance", "explained_variance_ratio", "cumulative_explained_variance_ratio", "singular_value"])
    (output_dir / "reference_scaler_7d.json").write_text(json.dumps(scaler_json(scaler7, PRIMARY_FEATURES), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "reference_pca_7d.json").write_text(json.dumps(pca_json(pca7, PRIMARY_FEATURES), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "reference_scaler_8d_sensitivity.json").write_text(json.dumps(scaler_json(scaler8, SENSITIVITY_FEATURES), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "reference_pca_8d_sensitivity.json").write_text(json.dumps(pca_json(pca8, SENSITIVITY_FEATURES), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    save_scatter(feature_rows, scores7, pca7, output_dir / "pca_7d_pc1_pc2.png")
    standardized7 = scaler7.transform(np.array([[float(row[f]) for f in PRIMARY_FEATURES] for row in feature_rows]))
    permdisp_summary, permdisp_detail = permdisp(feature_rows, standardized7)
    write_csv(output_dir / "permdisp_7d_sensitivity.csv", permdisp_summary,
              ["origin_country", "n", "mean_distance", "median_distance", "pseudo_F", "permutation_p", "permutations"])
    write_csv(output_dir / "permdisp_7d_distances.csv", permdisp_detail,
              ["case_id", "origin_country", "distance_to_group_centroid"])
    write_csv(output_dir / "confirmatory_measurement_limitations.csv", limitations_rows(),
              ["limitation_id", "stage", "description", "action"])
    (output_dir / "README.md").write_text(readme_text(pca7), encoding="utf-8", newline="\n")

    manifest = {
        "analysis_version": "real_reference_confirmatory_v1",
        "source_corpus_version": source_manifest["corpus_version"],
        "source_corpus_sha256": source_hash,
        "analysis_split": "confirmatory", "confirmatory_cases": 160, "KR": 80, "US": 80,
        "development_cases_used": 0, "LLM_outputs_used": 0,
        "primary_feature_order": PRIMARY_FEATURES, "sensitivity_feature_order": SENSITIVITY_FEATURES,
        "density_denominator": "doc_length_sentences; unique count / sentences * 100",
        "lexicon_v1_0_hashes_before": lexicon_hashes,
        "measurement_code_hashes_before": code_hashes,
        "lexicon_validation_derived_modifications": 0,
        "confirmatory_data_used_to_modify_measurement_pipeline": False,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_runs": BOOTSTRAP_RUNS,
        "permdisp_seed": PERMDISP_SEED, "permdisp_permutations": PERMDISP_PERMUTATIONS,
        "mannwhitney_tests": 7, "holm_correction_tests": 7, "cliffs_delta_tests": 7,
        "pca_7d_fit_population": "real-reference confirmatory 160 only",
        "pca_8d_fit_population": "real-reference confirmatory 160 only",
        "scipy_version": scipy.__version__, "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "PC1_explained_variance_ratio": float(pca7.explained_variance_ratio_[0]),
        "PC2_explained_variance_ratio": float(pca7.explained_variance_ratio_[1]),
        "PC1_PC2_cumulative_ratio": float(pca7.explained_variance_ratio_[:2].sum()),
        "randomness_limited_to_fixed_seed_resampling": True,
        "generation_timestamp_utc": json.loads(PREVALIDATION_RUN_MANIFEST.read_text(encoding="utf-8"))["generated_at_utc"],
        "timestamp_policy": "deterministic sealed pre-validation timestamp",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii", newline="\n"
    )
    return manifest


def run(lexicon_output: Path = LEXICON_OUTPUT, analysis_output: Path = ANALYSIS_OUTPUT) -> dict:
    source_hash, source_manifest = verify_source()
    lexicon_hashes = freeze_lexicon(lexicon_output)
    code_hashes = {
        "measurement": sha256_file(MEASUREMENT_PATH),
        "runner": sha256_file(Path(__file__).resolve()),
    }
    # Only after lexicon and full measurement code are hashed may confirmatory text be read.
    documents, _ = load_confirmatory_documents()
    matcher = load_matcher(lexicon_output / "lexicon_matcher_v1_0.py")
    feature_rows, hit_rows = extract_reference_features(documents, matcher)
    manifest = write_analysis(analysis_output, feature_rows, hit_rows, source_hash, source_manifest, lexicon_hashes, code_hashes)
    assert {path.name: sha256_file(path) for path in lexicon_output.iterdir() if path.is_file()} == lexicon_hashes
    assert code_hashes == {"measurement": sha256_file(MEASUREMENT_PATH), "runner": sha256_file(Path(__file__).resolve())}
    assert sha256_file(CORPUS_PATH) == source_hash
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon-output", type=Path, default=LEXICON_OUTPUT)
    parser.add_argument("--analysis-output", type=Path, default=ANALYSIS_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.lexicon_output.resolve(), args.analysis_output.resolve())
    print("lexicon_v1.0 frozen: yes")
    print("lexicon validation-derived changes: 0")
    print(f"confirmatory cases: {result['confirmatory_cases']}")
    print(f"KR: {result['KR']}")
    print(f"US: {result['US']}")
    print("primary features extracted: 7")
    print("missing/nonfinite feature values: 0")
    print("Mann-Whitney tests completed: 7")
    print("Holm correction completed: yes")
    print("Cliff's delta completed: 7")
    print(f"PC1 explained variance: {result['PC1_explained_variance_ratio']:.8f}")
    print(f"PC2 explained variance: {result['PC2_explained_variance_ratio']:.8f}")
    print(f"PC1+PC2 cumulative variance: {result['PC1_PC2_cumulative_ratio']:.8f}")
    print("7D PCA fitted on real-reference 160 only: yes")
    print("8D sensitivity PCA completed: yes")
    print("PERMDISP completed: yes")
    print("confirmatory data used to modify measurement pipeline: no")
    print("LLM outputs used in PCA fitting: no")
    print(f"output directory: {args.analysis_output.resolve()}")


if __name__ == "__main__":
    main()
