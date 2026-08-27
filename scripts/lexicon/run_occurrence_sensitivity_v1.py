#!/usr/bin/env python3
"""Run the post-hoc supplementary occurrence-based confirmatory sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = REPO_ROOT / ".cache_v2/matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REQUESTED_INPUT = REPO_ROOT / "outputs_v2/real_reference_confirmatory_v1/reference_features_160.csv"
AVAILABLE_INPUT = REPO_ROOT / "outputs_v2/real_reference_confirmatory/reference_features_160.csv"
REQUESTED_PRIMARY = REPO_ROOT / "outputs_v2/real_reference_confirmatory_v1/mannwhitney_cliffs_primary.csv"
AVAILABLE_PRIMARY = REPO_ROOT / "outputs_v2/real_reference_confirmatory/mannwhitney_cliffs_primary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs_v2/real_reference_occurrence_sensitivity"
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_RUNS = 10_000
ALPHA = 0.05

FEATURE_SPECS = (
    ("statute", "statute_occurrence_count", "statute_mention_density", "statute_density"),
    ("precedent", "precedent_occurrence_count", "precedent_mention_density", "precedent_density"),
    ("doctrine", "doctrine_hit_count", "doctrine_mention_density", "doctrine_density"),
    ("remedy", "remedy_hit_count", "remedy_mention_density", "remedy_density"),
    ("procedure", "procedure_hit_count", "procedure_mention_density", "procedure_density"),
    ("party_arg", "party_arg_hit_count", "party_arg_mention_density", "party_arg_density"),
)
OCCURRENCE_FEATURES = tuple(spec[2] for spec in FEATURE_SPECS)
PCA_FEATURES = ("avg_chars_per_sentence",) + OCCURRENCE_FEATURES
REQUIRED_INPUT_COLUMNS = {
    "case_id", "origin_country", "origin_state", "primary_domain", "doc_length_sentences",
    "avg_chars_per_sentence",
    *(spec[1] for spec in FEATURE_SPECS),
}
FEATURE_COLUMNS = [
    "case_id", "origin_country", "origin_state", "primary_domain",
    "doc_length_sentences", "avg_chars_per_sentence",
]
for _, count_column, density_column, _ in FEATURE_SPECS:
    FEATURE_COLUMNS.extend([count_column, density_column])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def protected_hashes(input_path: Path, primary_path: Path) -> dict[str, str]:
    paths = {
        input_path.resolve(), primary_path.resolve(),
        (REPO_ROOT / "scripts/lexicon/confirmatory_measurement_v1.py").resolve(),
        (REPO_ROOT / "scripts/lexicon/run_confirmatory_reference_v1.py").resolve(),
        (REPO_ROOT / "legal_lexicon/legal_lexicon_v1_0/lexicon_matcher_v1_0.py").resolve(),
    }
    hashes = {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}
    for primary_directory in {input_path.resolve().parent, primary_path.resolve().parent}:
        hashes.update(tree_hashes(primary_directory))
    hashes.update(tree_hashes(REPO_ROOT / "legal_lexicon/legal_lexicon_v1_0"))
    return dict(sorted(hashes.items()))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return value


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: csv_value(row.get(column, "")) for column in columns} for row in rows)


def validate_input(rows: Sequence[dict]) -> None:
    assert len(rows) == 160, f"Expected 160 cases, found {len(rows)}"
    assert len({row["case_id"] for row in rows}) == 160
    assert sum(row["origin_country"] == "KR" for row in rows) == 80
    assert sum(row["origin_country"] == "US" for row in rows) == 80
    assert all(float(row["doc_length_sentences"]) > 0 for row in rows)
    assert REQUIRED_INPUT_COLUMNS <= set(rows[0])


def derive_occurrence_features(rows: Sequence[dict]) -> list[dict]:
    validate_input(rows)
    output = []
    for row in rows:
        denominator = float(row["doc_length_sentences"])
        derived = {column: row[column] for column in FEATURE_COLUMNS if column in row}
        for _, count_column, density_column, _ in FEATURE_SPECS:
            count = float(row[count_column])
            assert math.isfinite(count) and count >= 0
            derived[density_column] = count / denominator * 100
        output.append(derived)
    assert all(
        math.isfinite(float(row[feature])) and float(row[feature]) >= 0
        for row in output for feature in OCCURRENCE_FEATURES
    )
    return output


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return float(q1), float(median), float(q3)


def descriptive_rows(rows: Sequence[dict]) -> list[dict]:
    output = []
    for feature in OCCURRENCE_FEATURES:
        for country in ("KR", "US"):
            values = np.array([float(row[feature]) for row in rows if row["origin_country"] == country])
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


def bootstrap_delta_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    deltas = np.empty(BOOTSTRAP_RUNS, dtype=float)
    for index in range(BOOTSTRAP_RUNS):
        sample_a = a[rng.integers(0, len(a), len(a))]
        sample_b = b[rng.integers(0, len(b), len(b))]
        deltas[index] = cliffs_delta(sample_a, sample_b)
    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    assert len(p_values) == 6
    order = np.argsort(np.asarray(p_values), kind="stable")
    adjusted = np.empty(6, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (6 - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def inferential_rows(rows: Sequence[dict]) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output, p_values = [], []
    for feature in OCCURRENCE_FEATURES:
        kr = np.array([float(row[feature]) for row in rows if row["origin_country"] == "KR"])
        us = np.array([float(row[feature]) for row in rows if row["origin_country"] == "US"])
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
    assert len(output) == len(p_values) == 6
    adjusted = holm_adjust(p_values)
    assert len(adjusted) == 6
    for row, value in zip(output, adjusted):
        row["p_holm"] = value
    return output


def fit_pca(rows: Sequence[dict]) -> tuple[StandardScaler, PCA, np.ndarray, np.ndarray]:
    matrix = np.array([[float(row[feature]) for feature in PCA_FEATURES] for row in rows], dtype=float)
    assert matrix.shape == (160, 7) and np.isfinite(matrix).all()
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    pca = PCA().fit(standardized)
    scores = pca.transform(standardized)
    assert scores.shape == (160, 7) and np.isfinite(scores).all()
    return scaler, pca, standardized, scores


def scaler_json(scaler: StandardScaler) -> dict:
    return {
        "feature_order": list(PCA_FEATURES),
        "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(),
        "variances": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_),
        "library": "scikit-learn", "library_version": sklearn.__version__,
        "fit_population": "real-reference confirmatory 160; occurrence-based exploratory PCA only",
        "primary_reference_scaler_modified": False,
    }


def pca_json(pca: PCA) -> dict:
    return {
        "feature_order": list(PCA_FEATURES), "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(), "mean": pca.mean_.tolist(),
        "n_samples": int(pca.n_samples_), "library": "scikit-learn",
        "library_version": sklearn.__version__,
        "fit_population": "real-reference confirmatory 160; occurrence-based exploratory PCA only",
        "primary_reference_pca_modified": False,
    }


def pca_tables(rows: Sequence[dict], pca: PCA, scores: np.ndarray):
    component_names = [f"PC{index + 1}" for index in range(7)]
    score_rows = [
        {"case_id": row["case_id"], "origin_country": row["origin_country"]}
        | {name: value for name, value in zip(component_names, values)}
        for row, values in zip(rows, scores)
    ]
    loading_rows = [
        {"feature": feature}
        | {name: pca.components_[index, feature_index] for index, name in enumerate(component_names)}
        for feature_index, feature in enumerate(PCA_FEATURES)
    ]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    explained_rows = [{
        "component": name,
        "explained_variance": pca.explained_variance_[index],
        "explained_variance_ratio": pca.explained_variance_ratio_[index],
        "cumulative_explained_variance_ratio": cumulative[index],
        "singular_value": pca.singular_values_[index],
    } for index, name in enumerate(component_names)]
    return score_rows, loading_rows, explained_rows


def save_pca_scatter(rows: Sequence[dict], scores: np.ndarray, pca: PCA, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    countries = np.array([row["origin_country"] for row in rows])
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
    axis.set_title("Exploratory occurrence-based real-reference PCA")
    axis.legend()
    figure.savefig(
        path, dpi=180,
        metadata={"Software": "comparative-law-llm occurrence sensitivity pipeline v1"},
    )
    plt.close(figure)


def direction(delta: float) -> str:
    if delta > 0:
        return "KR_higher"
    if delta < 0:
        return "US_higher"
    return "tie"


def comparison_rows(primary_rows: Sequence[dict], occurrence_rows: Sequence[dict]) -> list[dict]:
    primary = {row["feature"]: row for row in primary_rows}
    occurrence = {row["feature"]: row for row in occurrence_rows}
    output = []
    for dimension, _, occurrence_feature, primary_feature in FEATURE_SPECS:
        primary_row, occurrence_row = primary[primary_feature], occurrence[occurrence_feature]
        primary_delta, occurrence_delta = float(primary_row["cliffs_delta"]), float(occurrence_row["cliffs_delta"])
        primary_significant = float(primary_row["p_holm"]) < ALPHA
        occurrence_significant = float(occurrence_row["p_holm"]) < ALPHA
        output.append({
            "dimension": dimension,
            "primary_feature": primary_feature,
            "primary_direction": direction(primary_delta),
            "primary_p_holm": float(primary_row["p_holm"]),
            "primary_cliffs_delta": primary_delta,
            "occurrence_feature": occurrence_feature,
            "occurrence_direction": direction(occurrence_delta),
            "occurrence_p_holm": float(occurrence_row["p_holm"]),
            "occurrence_cliffs_delta": occurrence_delta,
            "direction_consistent": direction(primary_delta) == direction(occurrence_delta),
            "significance_consistent": primary_significant == occurrence_significant,
        })
    return output


def classification(row: dict) -> str:
    breadth = float(row["primary_p_holm"]) < ALPHA
    intensity = float(row["occurrence_p_holm"]) < ALPHA
    if breadth and intensity:
        return "breadth + intensity differences"
    if breadth:
        return "breadth-only difference"
    if intensity:
        return "intensity-only difference"
    return "neither"


def readme_text(comparisons: Sequence[dict], inference: Sequence[dict], pca: PCA) -> str:
    grouped = {
        label: [row["dimension"] for row in comparisons if classification(row) == label]
        for label in (
            "breadth + intensity differences", "breadth-only difference",
            "intensity-only difference", "neither",
        )
    }
    inference_by_feature = {row["feature"]: row for row in inference}
    lines = [
        "# Post-hoc supplementary occurrence-based sensitivity analysis", "",
        "This analysis is supplementary and does not replace or modify the official primary confirmatory analysis.", "",
        "## Breadth/diversity", "",
        "The primary features measure unique authorities or concept IDs per 100 frozen-parser sentences.", "",
        "## Mention intensity", "",
        "The supplementary features measure all saved citation or lexicon-hit occurrences per 100 frozen-parser sentences.",
        "No lexicon, matcher, citation parser, sentence segmentation, primary analysis, or primary PCA/scaler was rerun or modified. Only saved counts and the saved primary comparison results were read; no LLM output was used.", "",
        "## Results", "",
    ]
    for label, dimensions in grouped.items():
        lines.append(f"- {label}: {', '.join(dimensions) if dimensions else 'none'}")
    lines.extend(["", "Primary reporting: median [Q1–Q3]; two-sided Mann–Whitney U; Holm correction across six tests; Cliff's delta is KR relative to US.", ""])
    for _, _, feature, _ in FEATURE_SPECS:
        row = inference_by_feature[feature]
        lines.append(
            f"- {feature}: KR {row['KR_median']:.3f} [{row['KR_Q1']:.3f}–{row['KR_Q3']:.3f}]; "
            f"US {row['US_median']:.3f} [{row['US_Q1']:.3f}–{row['US_Q3']:.3f}]; "
            f"Holm p={row['p_holm']:.4g}; delta={row['cliffs_delta']:.3f}."
        )
    absolute_pc1 = np.abs(pca.components_[0])
    absolute_pc2 = np.abs(pca.components_[1])
    pc1_feature = PCA_FEATURES[int(np.argmax(absolute_pc1))]
    pc2_feature = PCA_FEATURES[int(np.argmax(absolute_pc2))]
    lines.extend([
        "", "## Exploratory occurrence-based 7D PCA", "",
        "This PCA uses avg_chars_per_sentence plus the six occurrence mention densities. A new occurrence-only StandardScaler and PCA were fitted on the 160 real-reference cases. The official primary 7D scaler/PCA was not modified or refitted.", "",
        f"- PC1 explained variance: {pca.explained_variance_ratio_[0] * 100:.4f}%",
        f"- PC2 explained variance: {pca.explained_variance_ratio_[1] * 100:.4f}%",
        f"- PC1+PC2 cumulative: {pca.explained_variance_ratio_[:2].sum() * 100:.4f}%",
        f"- Largest absolute PC1 loading: {pc1_feature} ({pca.components_[0, int(np.argmax(absolute_pc1))]:.4f})",
        f"- Largest absolute PC2 loading: {pc2_feature} ({pca.components_[1, int(np.argmax(absolute_pc2))]:.4f})",
        "", "This occurrence PCA is exploratory and is not a replacement for the official primary confirmatory PCA or a significance test.",
    ])
    return "\n".join(lines) + "\n"


def run(input_path: Path, primary_path: Path, output_dir: Path) -> dict:
    assert input_path.exists(), f"Missing input: {input_path}"
    assert primary_path.exists(), f"Missing primary results: {primary_path}"
    protected_before = protected_hashes(input_path, primary_path)
    input_rows = read_csv(input_path)
    occurrence_features = derive_occurrence_features(input_rows)
    descriptions = descriptive_rows(occurrence_features)
    inference = inferential_rows(occurrence_features)
    comparisons = comparison_rows(read_csv(primary_path), inference)
    scaler, pca, _, scores = fit_pca(occurrence_features)
    score_rows, loading_rows, explained_rows = pca_tables(occurrence_features, pca, scores)
    assert len(inference) == len(comparisons) == 6

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "occurrence_features_160.csv", occurrence_features, FEATURE_COLUMNS)
    write_csv(
        output_dir / "descriptive_stats_occurrence.csv", descriptions,
        ["feature", "country", "n", "mean", "sd", "median", "q1", "q3", "iqr", "min", "max"],
    )
    write_csv(
        output_dir / "mannwhitney_cliffs_occurrence.csv", inference,
        ["feature", "KR_n", "US_n", "KR_median", "KR_Q1", "KR_Q3", "US_median", "US_Q1", "US_Q3",
         "mannwhitney_U", "p_raw", "p_holm", "cliffs_delta", "cliffs_delta_ci_low", "cliffs_delta_ci_high"],
    )
    write_csv(
        output_dir / "primary_vs_occurrence_comparison.csv", comparisons,
        ["dimension", "primary_feature", "primary_direction", "primary_p_holm", "primary_cliffs_delta",
         "occurrence_feature", "occurrence_direction", "occurrence_p_holm", "occurrence_cliffs_delta",
         "direction_consistent", "significance_consistent"],
    )
    component_columns = [f"PC{index}" for index in range(1, 8)]
    write_csv(output_dir / "occurrence_pca_scores_7d.csv", score_rows, ["case_id", "origin_country", *component_columns])
    write_csv(output_dir / "occurrence_pca_loadings_7d.csv", loading_rows, ["feature", *component_columns])
    write_csv(
        output_dir / "occurrence_pca_explained_variance_7d.csv", explained_rows,
        ["component", "explained_variance", "explained_variance_ratio",
         "cumulative_explained_variance_ratio", "singular_value"],
    )
    (output_dir / "occurrence_scaler_7d.json").write_text(
        json.dumps(scaler_json(scaler), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    (output_dir / "occurrence_pca_7d.json").write_text(
        json.dumps(pca_json(pca), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    save_pca_scatter(occurrence_features, scores, pca, output_dir / "occurrence_pca_7d_pc1_pc2.png")
    (output_dir / "README.md").write_text(readme_text(comparisons, inference, pca), encoding="utf-8", newline="\n")

    protected_after = protected_hashes(input_path, primary_path)
    assert protected_before == protected_after
    artifact_names = [
        "occurrence_features_160.csv", "descriptive_stats_occurrence.csv",
        "mannwhitney_cliffs_occurrence.csv", "primary_vs_occurrence_comparison.csv",
        "occurrence_pca_scores_7d.csv", "occurrence_pca_loadings_7d.csv",
        "occurrence_pca_explained_variance_7d.csv", "occurrence_scaler_7d.json",
        "occurrence_pca_7d.json", "occurrence_pca_7d_pc1_pc2.png", "README.md",
    ]
    manifest = {
        "analysis_label": "post-hoc supplementary occurrence-based sensitivity analysis",
        "analysis_version": "real_reference_occurrence_sensitivity_v1",
        "input_path": input_path.relative_to(REPO_ROOT).as_posix(),
        "input_sha256": sha256_file(input_path),
        "primary_results_path": primary_path.relative_to(REPO_ROOT).as_posix(),
        "primary_results_sha256": sha256_file(primary_path),
        "confirmatory_cases_used": 160, "KR": 80, "US": 80,
        "supplementary_occurrence_features": 6, "mannwhitney_tests": 6,
        "holm_correction_tests": 6, "cliffs_delta_tests": 6,
        "bootstrap_runs": BOOTSTRAP_RUNS, "bootstrap_seed": BOOTSTRAP_SEED,
        "density_denominator": "doc_length_sentences; total saved occurrence count / sentences * 100",
        "primary_analysis_modified": False, "lexicon_modified": False,
        "parser_or_matcher_modified": False,
        "primary_pca_modified_or_refit": False,
        "exploratory_occurrence_pca_fitted": True,
        "occurrence_pca_feature_order": list(PCA_FEATURES),
        "occurrence_pca_fit_population": "real-reference confirmatory 160 only",
        "occurrence_PC1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "occurrence_PC2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "occurrence_PC1_PC2_cumulative_ratio": float(pca.explained_variance_ratio_[:2].sum()),
        "LLM_outputs_used": False, "protected_hashes_before_and_after_equal": True,
        "protected_sha256": protected_before,
        "artifact_sha256": {name: sha256_file(output_dir / name) for name in artifact_names},
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    all_names = artifact_names + ["analysis_manifest.json"]
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(output_dir / name)}  {name}\n" for name in sorted(all_names)),
        encoding="ascii", newline="\n",
    )
    return {"manifest": manifest, "comparisons": comparisons}


def default_existing(preferred: Path, fallback: Path) -> Path:
    return preferred if preferred.exists() else fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_existing(REQUESTED_INPUT, AVAILABLE_INPUT))
    parser.add_argument("--primary-results", type=Path, default=default_existing(REQUESTED_PRIMARY, AVAILABLE_PRIMARY))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.input.resolve(), args.primary_results.resolve(), args.output_dir.resolve())
    manifest = result["manifest"]
    print(f"confirmatory cases used: {manifest['confirmatory_cases_used']}")
    print(f"KR: {manifest['KR']}")
    print(f"US: {manifest['US']}")
    print(f"supplementary occurrence features: {manifest['supplementary_occurrence_features']}")
    print(f"Mann-Whitney tests: {manifest['mannwhitney_tests']}")
    print(f"Holm correction: {manifest['holm_correction_tests']} tests")
    print(f"Cliff's delta: {manifest['cliffs_delta_tests']} tests")
    print(f"bootstrap runs: {manifest['bootstrap_runs']}")
    print("primary analysis modified: no")
    print("lexicon modified: no")
    print("primary PCA modified/refit: no")
    print("exploratory occurrence PCA fitted: yes")
    print(f"occurrence PC1 explained variance: {manifest['occurrence_PC1_explained_variance_ratio'] * 100:.4f}%")
    print(f"occurrence PC2 explained variance: {manifest['occurrence_PC2_explained_variance_ratio'] * 100:.4f}%")
    print("LLM outputs used: no")
    print(f"output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
