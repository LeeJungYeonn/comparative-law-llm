#!/usr/bin/env python3
"""Gate 2 robustness analysis for the frozen real-reference corpus.

This module never changes the frozen corpus, lexicon, parser, primary feature
definitions, or existing primary outputs.  Every artifact is written below a
new robustness directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
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

from scripts.lexicon.confirmatory_measurement_v1 import (
    CATEGORIES,
    PRIMARY_FEATURES,
    extract_features,
    sentence_spans,
)
from scripts.lexicon.run_confirmatory_reference_v1 import cliffs_delta, holm_adjust


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
CORPUS_MANIFEST = REPO_ROOT / "outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json"
ROSTER = REPO_ROOT / "outputs_v2/v4.0.2/final_roster_manifest_200_v4_0_2.csv"
PRIMARY_DISCOVERED = REPO_ROOT / "outputs_v2/real_reference_confirmatory/reference_features_160.csv"
PRIMARY_MANIFEST_DISCOVERED = REPO_ROOT / "outputs_v2/real_reference_confirmatory/analysis_manifest.json"
OCCURRENCE_DISCOVERED = REPO_ROOT / "outputs_v2/real_reference_occurrence_sensitivity/occurrence_features_160.csv"
OCCURRENCE_MANIFEST_DISCOVERED = REPO_ROOT / "outputs_v2/real_reference_occurrence_sensitivity/analysis_manifest.json"
LEXICON_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_v1_0"
MATCHER_PATH = LEXICON_DIR / "lexicon_matcher_v1_0.py"
SURFACE_PATH = LEXICON_DIR / "surface_forms_v1_0.csv"
CONCEPT_PATH = LEXICON_DIR / "concepts_v1_0.csv"
MEASUREMENT_PATH = REPO_ROOT / "scripts/lexicon/confirmatory_measurement_v1.py"
DEFAULT_OUTPUT = REPO_ROOT / "outputs_v2/robustness_gate2"

PROCEDURE = "procedure_density"
NO_PROCEDURE_FEATURES = tuple(feature for feature in PRIMARY_FEATURES if feature != PROCEDURE)
OCCURRENCE_FEATURES = (
    "avg_chars_per_sentence",
    "statute_mention_density",
    "precedent_mention_density",
    "doctrine_mention_density",
    "remedy_mention_density",
    "procedure_mention_density",
    "party_arg_mention_density",
)
OCCURRENCE_NO_PROCEDURE = tuple(f for f in OCCURRENCE_FEATURES if f != "procedure_mention_density")
SEED = 20260903
BOOTSTRAP_RUNS = 10_000

# Values read from the pre-existing primary manifest during the repository audit.
# They are used only to check reproduction, never to construct results.
AUDITED_PRIMARY_BASELINE = {
    "analysis_version": "real_reference_confirmatory_v1",
    "n": 160,
    "n_KR": 80,
    "n_US": 80,
    "PC1_explained_variance_ratio": 0.35616224693414517,
    "PC2_explained_variance_ratio": 0.1637753527076169,
    "PC1_PC2_cumulative_ratio": 0.5199375996417621,
}

INITIAL_OUTPUT_AUDIT = {
    "primary_feature_matrix": {
        "was_present": True,
        "sha256": "887f996d74df258db7ac73f94cb55fa860076549cc0c2a993fc98b5828b1d03f",
    },
    "primary_analysis_manifest": {
        "was_present": True,
        "sha256": "36def7fe32af81009aa7ad28796224cf7c377cd3ac8801924938dc05c6b2ea1c",
    },
    "occurrence_feature_matrix": {
        "was_present": True,
        "sha256": "fda72c164b059472f8347154e0b2fab5b49866143031a23cb2a22f8901bc1a2d",
    },
    "occurrence_analysis_manifest": {"was_present": True, "sha256": None},
}

# Frozen before confirmatory application.  The same paragraph/sentence criteria
# are used in both languages; only literal language equivalents differ.
REASONING_RULE = {
    "version": "gate2_reasoning_v1_dev_frozen",
    "unit": "frozen-parser sentence",
    "default": "retain",
    "exclude_if": "procedural_score>=1 and substantive_score==0, or boilerplate_score>=1 and substantive_score==0",
    "procedural_patterns": {
        "ko": [
            r"원심(?:판결|법원|은|이)", r"상고(?:를|이유|심)", r"항소(?:를|심|법원)",
            r"소송비용", r"기록에 (?:의하면|비추어)", r"청구를 (?:기각|각하)",
            r"파기환송", r"환송 후", r"변론을 거쳐", r"관할(?:권|법원)",
        ],
        "en": [
            r"procedural history", r"standard of review", r"we review .{0,30} de novo",
            r"filed (?:a |this )?(?:complaint|action|appeal)", r"trial court", r"district court",
            r"court of appeals", r"summary judgment", r"motion to (?:dismiss|strike)",
            r"remand(?:ed)?", r"appellate jurisdiction", r"notice of appeal",
        ],
    },
    "substantive_patterns": {
        "ko": [
            r"손해", r"배상", r"과실", r"주의의무", r"인과관계", r"책임", r"위법",
            r"불법행위", r"채무불이행", r"의무", r"법리", r"상당인과", r"손실",
        ],
        "en": [
            r"negligen", r"duty", r"breach", r"caus", r"damage", r"liabil", r"tort",
            r"injur", r"foresee", r"standard of care", r"wrongful", r"remed", r"proximate",
        ],
    },
    "boilerplate_patterns": {
        "ko": [r"^주\s*문", r"^대법관", r"주문과 같이 판결", r"^원고[, ]", r"^피고[, ]"],
        "en": [r"^counsel for", r"^attorneys? for", r"^before: ", r"^appeal from ", r"^case no\."],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g") if math.isfinite(float(value)) else ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "|".join(map(str, value))
    return value


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: csv_value(row.get(key, "")) for key in columns} for row in rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def package_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "scikit-learn", "matplotlib")
    return {name: importlib.metadata.version(name) for name in names}


def load_matcher():
    spec = importlib.util.spec_from_file_location("frozen_lexicon_matcher_v1_0_gate2", MATCHER_PATH)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load frozen matcher: {MATCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_and_verify_corpus() -> tuple[list[dict], dict]:
    required = (CORPUS, CORPUS_MANIFEST, ROSTER, MATCHER_PATH, SURFACE_PATH, CONCEPT_PATH, MEASUREMENT_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen inputs missing: " + ", ".join(missing))
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise AssertionError("Corpus manifest is not FROZEN")
    expected = next(
        value for key, value in manifest["artifact_sha256"].items() if Path(key).name == CORPUS.name
    )
    actual = sha256_file(CORPUS)
    if actual != expected:
        raise AssertionError(f"Frozen corpus SHA256 mismatch: {actual} != {expected}")
    records = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    if len(records) != 200 or len({r["case_id"] for r in records}) != 200:
        raise AssertionError("Expected 200 unique frozen corpus records")
    return records, manifest


def input_audit(records: list[dict], corpus_manifest: dict) -> dict:
    confirmatory = [r for r in records if r["analysis_split"] == "confirmatory"]
    development = [r for r in records if r["analysis_split"] == "development"]
    countries = Counter(r["origin_country"] for r in confirmatory)
    states = Counter(str(r.get("origin_state") or "") for r in confirmatory if r["origin_country"] == "US")
    hashes = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in (CORPUS, CORPUS_MANIFEST, ROSTER, MATCHER_PATH, SURFACE_PATH, CONCEPT_PATH, MEASUREMENT_PATH)
    }
    discovered = {}
    for label, path in (
        ("primary_feature_matrix", PRIMARY_DISCOVERED),
        ("primary_analysis_manifest", PRIMARY_MANIFEST_DISCOVERED),
        ("occurrence_feature_matrix", OCCURRENCE_DISCOVERED),
        ("occurrence_analysis_manifest", OCCURRENCE_MANIFEST_DISCOVERED),
    ):
        discovered[label] = {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "present_during_initial_repository_audit": INITIAL_OUTPUT_AUDIT[label]["was_present"],
            "sha256_recorded_during_initial_audit": INITIAL_OUTPUT_AUDIT[label]["sha256"],
            "present_at_execution": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    lexicon_manifest = json.loads((LEXICON_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    reason_keys = sorted({key for r in records for key in r if re.search(r"reason|substantive|section|span", key, re.I)})
    approved_reason_keys = [key for key in reason_keys if key in {"substantive_reasoning", "reasoning_text", "substantive_analysis_span"}]
    return {
        "git_commit": git_commit(),
        "input_files": discovered | {
            "frozen_corpus": {"path": CORPUS.relative_to(REPO_ROOT).as_posix(), "sha256": hashes[CORPUS.relative_to(REPO_ROOT).as_posix()]},
            "frozen_roster": {"path": ROSTER.relative_to(REPO_ROOT).as_posix(), "sha256": hashes[ROSTER.relative_to(REPO_ROOT).as_posix()]},
        },
        "all_protected_input_hashes": hashes,
        "n_total": len(confirmatory),
        "n_KR": countries["KR"],
        "n_US": countries["US"],
        "US_state_counts": dict(sorted(states.items())),
        "split_counts": dict(sorted(Counter(r["analysis_split"] for r in records).items())),
        "country_split_counts": dict(sorted(Counter(f"{r['origin_country']}_{r['analysis_split']}" for r in records).items())),
        "exact_primary_feature_columns": list(PRIMARY_FEATURES),
        "exact_occurrence_feature_columns": list(OCCURRENCE_FEATURES),
        "metadata_column_mapping": {
            "origin_country": "origin_country", "state": "origin_state", "case_id": "case_id",
            "case_domain": "primary_domain", "decision_year": "decision_date[0:4]",
        },
        "full_opinion_text": all(isinstance(r.get("main_opinion_text"), str) and r["main_opinion_text"].strip() for r in confirmatory),
        "full_opinion_source_column": "main_opinion_text",
        "existing_reasoning_only_keys": approved_reason_keys,
        "reasoning_only_status": "not found; deterministic development-only-frozen extraction used",
        "working_tree_sync_note": (
            "The four discovered ignored-output paths were present during the initial audit, then the outputs_v2 "
            "working-tree directory disappeared during external OneDrive synchronization. Tracked v4.0.2 inputs "
            "were restored byte-exact from HEAD (LF preserved); ignored primary outputs were not recreated in place."
        ),
        "pca_transform_metadata": {
            "discovered_scaler": "outputs_v2/real_reference_confirmatory/reference_scaler_7d.json",
            "discovered_pca": "outputs_v2/real_reference_confirmatory/reference_pca_7d.json",
            "standardization": "sklearn StandardScaler population variance (ddof=0)",
            "fit_population": "confirmatory real-reference 160",
        },
        "frozen_corpus_version": corpus_manifest["corpus_version"],
        "frozen_lexicon_version": lexicon_manifest.get("lexicon_version", "v1.0"),
        "frozen_lexicon_status": lexicon_manifest.get("status"),
        "frozen_lexicon_hashes": {p.name: sha256_file(p) for p in sorted(LEXICON_DIR.iterdir()) if p.is_file()},
        "frozen_citation_parser": {"version": "confirmatory_measurement_v1", "sha256": sha256_file(MEASUREMENT_PATH)},
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
    }


def confirmatory_documents(records: Sequence[dict]) -> list[dict]:
    output = []
    for record in records:
        if record["analysis_split"] != "confirmatory":
            continue
        text = record.get("main_opinion_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing main_opinion_text: {record['case_id']}")
        output.append({
            "case_id": record["case_id"], "origin_country": record["origin_country"],
            "origin_state": str(record.get("origin_state") or ""),
            "primary_domain": str(record.get("primary_domain") or record.get("case_domain") or ""),
            "decision_year": int(str(record["decision_date"])[:4]), "text": text,
        })
    output.sort(key=lambda row: row["case_id"])
    counts = Counter(row["origin_country"] for row in output)
    if len(output) != 160 or counts != Counter({"KR": 80, "US": 80}):
        raise AssertionError(f"Unexpected confirmatory population: n={len(output)}, groups={counts}")
    return output


FEATURE_OUTPUT_COLUMNS = [
    "case_id", "origin_country", "origin_state", "primary_domain", "decision_year",
    "doc_length_sentences", "doc_length_chars", "avg_chars_per_sentence",
    "statute_occurrence_count", "unique_statute_count", "statute_density",
    "precedent_occurrence_count", "unique_precedent_count", "precedent_density",
]
for _category in CATEGORIES:
    FEATURE_OUTPUT_COLUMNS.extend([
        f"{_category}_hit_count", f"{_category}_unique_concepts",
        f"{_category}_unique_families", f"{_category}_density",
    ])
FEATURE_OUTPUT_COLUMNS.append("log1p_doc_length_sentences")


def measure_documents(documents: Sequence[dict], matcher) -> list[dict]:
    output = []
    for document in documents:
        features, _ = extract_features(document["text"], document["origin_country"], matcher, SURFACE_PATH, CONCEPT_PATH)
        row = {key: document[key] for key in ("case_id", "origin_country", "origin_state", "primary_domain", "decision_year")}
        row.update(features)
        output.append(row)
    if any(not math.isfinite(float(row[f])) for row in output for f in PRIMARY_FEATURES):
        raise ValueError("Missing/nonfinite primary feature; no row was dropped")
    return output


def occurrence_rows(primary: Sequence[dict]) -> list[dict]:
    output = []
    for row in primary:
        denominator = float(row["doc_length_sentences"])
        derived = dict(row)
        for category, count in (
            ("statute", "statute_occurrence_count"), ("precedent", "precedent_occurrence_count"),
            ("doctrine", "doctrine_hit_count"), ("remedy", "remedy_hit_count"),
            ("procedure", "procedure_hit_count"), ("party_arg", "party_arg_hit_count"),
        ):
            derived[f"{category}_mention_density"] = float(row[count]) / denominator * 100
        output.append(derived)
    return output


def compile_reasoning_rule() -> dict[str, dict[str, list[re.Pattern]]]:
    return {
        group: {language: [re.compile(pattern, re.I | re.S) for pattern in patterns] for language, patterns in languages.items()}
        for group, languages in REASONING_RULE.items() if group.endswith("_patterns")
    }


def merge_spans(spans: Sequence[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def extract_reasoning_text(text: str, country: str, compiled: dict) -> tuple[str, list[list[int]], list[list[int]], list[str]]:
    language = "ko" if country == "KR" else "en"
    procedural = compiled["procedural_patterns"][language]
    substantive = compiled["substantive_patterns"][language]
    boilerplate = compiled["boilerplate_patterns"][language]
    included, excluded, flags = [], [], []
    for start, end in sentence_spans(text):
        unit = text[start:end]
        p_score = sum(bool(pattern.search(unit)) for pattern in procedural)
        s_score = sum(bool(pattern.search(unit)) for pattern in substantive)
        b_score = sum(bool(pattern.search(unit.strip())) for pattern in boilerplate)
        if s_score == 0 and (p_score >= 1 or b_score >= 1):
            excluded.append((start, end))
        else:
            included.append((start, end))
    if not included:
        raise ValueError("Reasoning extraction removed the entire opinion")
    ranges = merge_spans(included)
    excluded_ranges = merge_spans(excluded)
    reasoning = "\n".join(text[start:end].strip() for start, end in ranges if text[start:end].strip())
    retained = len(reasoning) / len(text)
    if retained < 0.50:
        flags.append("LOW_RETAINED_RATIO_LT_0_50")
    if retained > 0.98:
        flags.append("HIGH_RETAINED_RATIO_GT_0_98")
    return reasoning, ranges, excluded_ranges, flags


def build_reasoning_corpus(records: Sequence[dict], output_dir: Path) -> tuple[list[dict], list[dict], dict]:
    compiled = compile_reasoning_rule()
    rule_hash = json_hash(REASONING_RULE)
    dev_summary = []
    for country in ("KR", "US"):
        ratios = []
        for record in records:
            if record["analysis_split"] == "development" and record["origin_country"] == country:
                reasoning, _, _, _ = extract_reasoning_text(record["main_opinion_text"], country, compiled)
                ratios.append(len(reasoning) / len(record["main_opinion_text"]))
        dev_summary.append({
            "origin_country": country, "n": len(ratios), "mean_retained_ratio": float(np.mean(ratios)),
            "median_retained_ratio": float(np.median(ratios)), "min_retained_ratio": float(np.min(ratios)),
            "max_retained_ratio": float(np.max(ratios)),
        })
    if Counter(row["n"] for row in dev_summary) != Counter({20: 2}):
        raise AssertionError("Reasoning rule must be frozen on 20 KR + 20 US development cases")
    write_json(output_dir / "reasoning_extraction_rule.json", REASONING_RULE | {
        "sha256": rule_hash, "development_QC": dev_summary,
        "freeze_statement": "Rule fixed after development-set application and before confirmatory-set application.",
    })

    corpus, manifest = [], []
    for document in confirmatory_documents(records):
        reasoning, included, excluded, flags = extract_reasoning_text(document["text"], document["origin_country"], compiled)
        corpus.append(document | {"text": reasoning})
        manifest.append({
            "case_id": document["case_id"], "origin_country": document["origin_country"],
            "original_char_count": len(document["text"]), "reasoning_char_count": len(reasoning),
            "retained_ratio": len(reasoning) / len(document["text"]),
            "extraction_method": REASONING_RULE["version"], "rule_sha256": rule_hash,
            "included_span_offsets": included, "excluded_span_offsets": excluded,
            "excluded_section_labels": [], "QC_flags": flags,
        })
    write_json(output_dir / "reasoning_extraction_manifest.json", {
        "rule_version": REASONING_RULE["version"], "rule_sha256": rule_hash,
        "development_QC": dev_summary, "confirmatory_cases": manifest,
    })
    qc = {}
    for country in ("KR", "US"):
        values = [row["retained_ratio"] for row in manifest if row["origin_country"] == country]
        qc[country] = {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "min": min(values), "max": max(values)}
    return corpus, manifest, qc


@dataclass
class SpaceResult:
    name: str
    features: tuple[str, ...]
    rows: Sequence[dict]
    scaler: StandardScaler
    pca: PCA
    standardized: np.ndarray
    scores: np.ndarray
    permanova: dict
    permdisp: dict


def labels_for(rows: Sequence[dict]) -> np.ndarray:
    labels = np.array([row["origin_country"] for row in rows])
    if set(labels) != {"KR", "US"}:
        raise ValueError("Exactly KR and US labels are required")
    return labels


def fit_space(name: str, rows: Sequence[dict], features: Sequence[str], permutations: int, seed: int) -> SpaceResult:
    matrix = np.asarray([[float(row[f]) for f in features] for row in rows], dtype=float)
    if matrix.shape != (len(rows), len(features)) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid feature matrix for {name}; no row was dropped")
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    pca = PCA().fit(standardized)
    scores = pca.transform(standardized)
    labels = labels_for(rows)
    return SpaceResult(
        name, tuple(features), rows, scaler, pca, standardized, scores,
        permanova_test(standardized, labels, permutations, seed),
        permdisp_test(standardized, labels, permutations, seed + 1),
    )


def permanova_components(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float, float]:
    grand = matrix.mean(axis=0)
    total = float(np.square(matrix - grand).sum())
    within = 0.0
    for group in np.unique(labels):
        selected = matrix[labels == group]
        within += float(np.square(selected - selected.mean(axis=0)).sum())
    between = max(0.0, total - within)
    df_between = len(np.unique(labels)) - 1
    df_within = len(labels) - len(np.unique(labels))
    pseudo_f = (between / df_between) / (within / df_within) if within > 0 else math.inf
    return pseudo_f, between / total if total else 0.0, between, within, total


def permanova_test(matrix: np.ndarray, labels: np.ndarray, permutations: int, seed: int) -> dict:
    observed, r2, between, within, total = permanova_components(matrix, labels)
    rng = np.random.default_rng(seed)
    exceed = sum(permanova_components(matrix, rng.permutation(labels))[0] >= observed for _ in range(permutations))
    return {
        "pseudo_F": observed, "R2": r2, "permutation_p": (exceed + 1) / (permutations + 1),
        "SS_between": between, "SS_within": within, "SS_total": total,
        "n_KR": int((labels == "KR").sum()), "n_US": int((labels == "US").sum()),
        "permutations": permutations, "seed": seed,
        "p_convention": "(1 + permuted pseudo-F >= observed pseudo-F) / (1 + permutations); observed included",
    }


def one_way_f(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    groups = [values[labels == group] for group in np.unique(labels)]
    between = sum(len(group) * (group.mean() - grand) ** 2 for group in groups)
    within = sum(float(np.square(group - group.mean()).sum()) for group in groups)
    return float((between / (len(groups) - 1)) / (within / (len(values) - len(groups)))) if within else math.inf


def dispersion_distances(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = np.empty(len(labels), dtype=float)
    for group in np.unique(labels):
        mask = labels == group
        centroid = matrix[mask].mean(axis=0)
        distances[mask] = np.linalg.norm(matrix[mask] - centroid, axis=1)
    return distances


def permdisp_test(matrix: np.ndarray, labels: np.ndarray, permutations: int, seed: int) -> dict:
    distances = dispersion_distances(matrix, labels)
    observed = one_way_f(distances, labels)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        permuted_distances = dispersion_distances(matrix, permuted)
        exceed += one_way_f(permuted_distances, permuted) >= observed
    kr, us = distances[labels == "KR"], distances[labels == "US"]
    return {
        "pseudo_F": observed, "permutation_p": (exceed + 1) / (permutations + 1),
        "KR_mean_distance": float(kr.mean()), "US_mean_distance": float(us.mean()),
        "KR_median_distance": float(np.median(kr)), "US_median_distance": float(np.median(us)),
        "KR_US_mean_distance_ratio": float(kr.mean() / us.mean()) if us.mean() else math.inf,
        "permutations": permutations, "seed": seed,
        "centroid_convention": "arithmetic group centroid in Euclidean standardized feature space; mean and median distances reported",
    }


def centroid_distance(result: SpaceResult) -> float:
    labels = labels_for(result.rows)
    return float(np.linalg.norm(result.standardized[labels == "KR"].mean(axis=0) - result.standardized[labels == "US"].mean(axis=0)))


def feature_inference(rows: Sequence[dict], features: Sequence[str], bootstrap_runs: int, seed: int) -> tuple[list[dict], list[dict]]:
    descriptive, tests, p_values = [], [], []
    rng = np.random.default_rng(seed)
    for feature in features:
        groups = {country: np.asarray([float(row[feature]) for row in rows if row["origin_country"] == country]) for country in ("KR", "US")}
        for country, values in groups.items():
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            descriptive.append({
                "feature": feature, "origin_country": country, "n": len(values), "mean": values.mean(),
                "sd": values.std(ddof=1), "median": median, "q1": q1, "q3": q3,
                "min": values.min(), "max": values.max(),
            })
        kr, us = groups["KR"], groups["US"]
        mw = mannwhitneyu(kr, us, alternative="two-sided", method="asymptotic")
        delta = cliffs_delta(kr, us)
        boots = np.empty(bootstrap_runs)
        for index in range(bootstrap_runs):
            boots[index] = cliffs_delta(rng.choice(kr, len(kr), replace=True), rng.choice(us, len(us), replace=True))
        low, high = np.quantile(boots, [0.025, 0.975])
        direction = "KR>US" if delta > 0 else "KR<US" if delta < 0 else "tie"
        tests.append({
            "feature": feature, "KR_n": len(kr), "US_n": len(us), "mannwhitney_U": mw.statistic,
            "p_raw": mw.pvalue, "cliffs_delta_KR_minus_US": delta, "cliffs_delta_CI_low": low,
            "cliffs_delta_CI_high": high, "direction": direction, "bootstrap_runs": bootstrap_runs,
        })
        p_values.append(float(mw.pvalue))
    for row, adjusted in zip(tests, holm_adjust(p_values)):
        row["p_holm"] = adjusted
    return descriptive, tests


def save_space_tables(result: SpaceResult, output_dir: Path) -> None:
    stem = result.name
    pc_names = [f"PC{i}" for i in range(1, len(result.features) + 1)]
    score_rows = [
        {"case_id": row["case_id"], "origin_country": row["origin_country"]}
        | {pc: score for pc, score in zip(pc_names, values)}
        for row, values in zip(result.rows, result.scores)
    ]
    loading_rows = [
        {"feature": feature} | {pc: result.pca.components_[i, j] for i, pc in enumerate(pc_names)}
        for j, feature in enumerate(result.features)
    ]
    cumulative = np.cumsum(result.pca.explained_variance_ratio_)
    explained_rows = [
        {"component": pc, "explained_variance": result.pca.explained_variance_[i],
         "explained_variance_ratio": result.pca.explained_variance_ratio_[i],
         "cumulative_explained_variance_ratio": cumulative[i]}
        for i, pc in enumerate(pc_names)
    ]
    write_csv(output_dir / f"pca_scores_{stem}.csv", score_rows, ["case_id", "origin_country", *pc_names])
    write_csv(output_dir / f"pca_loadings_{stem}.csv", loading_rows, ["feature", *pc_names])
    write_csv(output_dir / f"pca_explained_variance_{stem}.csv", explained_rows,
              ["component", "explained_variance", "explained_variance_ratio", "cumulative_explained_variance_ratio"])
    write_json(output_dir / f"multivariate_{stem}.json", {
        "space": stem, "features": list(result.features), "n": len(result.rows),
        "scaler": {"mean": result.scaler.mean_.tolist(), "scale": result.scaler.scale_.tolist(), "ddof": 0},
        "pca": {"components": result.pca.components_.tolist(), "explained_variance_ratio": result.pca.explained_variance_ratio_.tolist()},
        "standardized_centroid_distance": centroid_distance(result),
        "PERMANOVA": result.permanova, "PERMDISP": result.permdisp,
    })


def save_pca(result: SpaceResult, path: Path, title: str) -> None:
    labels = labels_for(result.rows)
    colors = {"KR": "#2f6b9a", "US": "#c25b3f"}
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for country in ("KR", "US"):
        selected = result.scores[labels == country]
        axis.scatter(selected[:, 0], selected[:, 1], s=24, alpha=0.62, label=country, color=colors[country])
        centre = selected[:, :2].mean(axis=0)
        axis.scatter(*centre, s=155, marker="X", color=colors[country], edgecolor="black", linewidth=0.8)
    axis.axhline(0, color="#999999", linewidth=0.6)
    axis.axvline(0, color="#999999", linewidth=0.6)
    axis.set_xlabel(f"PC1 ({result.pca.explained_variance_ratio_[0] * 100:.1f}% explained)")
    axis.set_ylabel(f"PC2 ({result.pca.explained_variance_ratio_[1] * 100:.1f}% explained)")
    axis.set_title(title + "\n(visualization, not a significance test)")
    axis.legend()
    fig.savefig(path, dpi=180, metadata={"Software": "comparative-law-llm Gate 2"})
    plt.close(fig)


def space_summary(result: SpaceResult) -> dict:
    labels = labels_for(result.rows)
    return {
        "space": result.name, "n_KR": int((labels == "KR").sum()), "n_US": int((labels == "US").sum()),
        "n_features": len(result.features), "features": list(result.features),
        "PC1_variance": result.pca.explained_variance_ratio_[0], "PC2_variance": result.pca.explained_variance_ratio_[1],
        "cumulative_PC1_PC2": result.pca.explained_variance_ratio_[:2].sum(),
        "standardized_centroid_distance": centroid_distance(result),
        "PERMANOVA_F": result.permanova["pseudo_F"], "PERMANOVA_R2": result.permanova["R2"],
        "PERMANOVA_p": result.permanova["permutation_p"], "PERMDISP_F": result.permdisp["pseudo_F"],
        "PERMDISP_p": result.permdisp["permutation_p"],
        "KR_dispersion_mean": result.permdisp["KR_mean_distance"], "US_dispersion_mean": result.permdisp["US_mean_distance"],
    }


def loso_analysis(primary: Sequence[dict], permutations: int, seed: int) -> tuple[list[dict], list[dict]]:
    states = sorted({row["origin_state"] for row in primary if row["origin_country"] == "US"})
    summaries, feature_rows = [], []
    for state_index, state in enumerate(states):
        subset = [row for row in primary if row["origin_country"] == "KR" or row["origin_state"] != state]
        for space_index, features in enumerate((tuple(PRIMARY_FEATURES), NO_PROCEDURE_FEATURES)):
            name = "primary_7d" if PROCEDURE in features else "primary_6d_no_procedure"
            result = fit_space(f"loso_{name}_{state}", subset, features, permutations, seed + state_index * 20 + space_index * 2)
            summary = space_summary(result) | {"omitted_state": state, "space": name}
            summaries.append(summary)
            p_values, pending = [], []
            for feature in features:
                kr = np.asarray([float(row[feature]) for row in subset if row["origin_country"] == "KR"])
                us = np.asarray([float(row[feature]) for row in subset if row["origin_country"] == "US"])
                test = mannwhitneyu(kr, us, alternative="two-sided", method="asymptotic")
                delta = cliffs_delta(kr, us)
                pending.append({
                    "omitted_state": state, "space": name, "feature": feature,
                    "n_KR": len(kr), "n_US": len(us), "cliffs_delta_KR_minus_US": delta,
                    "direction": "KR>US" if delta > 0 else "KR<US" if delta < 0 else "tie", "p_raw": test.pvalue,
                })
                p_values.append(float(test.pvalue))
            for row, adjusted in zip(pending, holm_adjust(p_values)):
                row["p_holm"] = adjusted
                feature_rows.append(row)
    return summaries, feature_rows


def balanced_loso(primary: Sequence[dict], repeats: int, seed: int) -> list[dict]:
    states = sorted({row["origin_state"] for row in primary if row["origin_country"] == "US"})
    kr_rows = [row for row in primary if row["origin_country"] == "KR"]
    output = []
    master = np.random.SeedSequence(seed)
    children = iter(master.spawn(len(states) * 2))
    for state in states:
        us_rows = [row for row in primary if row["origin_country"] == "US" and row["origin_state"] != state]
        m = len(us_rows)
        for features in (tuple(PRIMARY_FEATURES), NO_PROCEDURE_FEATURES):
            name = "primary_7d" if PROCEDURE in features else "primary_6d_no_procedure"
            rng = np.random.default_rng(next(children))
            metrics = {key: [] for key in ("PERMANOVA_R2", "PERMANOVA_F", "PERMDISP_F", "centroid_distance")}
            for _ in range(repeats):
                chosen = rng.choice(len(kr_rows), m, replace=False)
                subset = [kr_rows[index] for index in chosen] + us_rows
                matrix = np.asarray([[float(row[f]) for f in features] for row in subset])
                standardized = StandardScaler().fit_transform(matrix)
                labels = labels_for(subset)
                p_f, r2, *_ = permanova_components(standardized, labels)
                d_f = one_way_f(dispersion_distances(standardized, labels), labels)
                distance = float(np.linalg.norm(standardized[labels == "KR"].mean(0) - standardized[labels == "US"].mean(0)))
                for key, value in zip(metrics, (r2, p_f, d_f, distance)):
                    metrics[key].append(value)
            row = {"omitted_state": state, "space": name, "n_each_group": m, "repeats": repeats}
            for key, values in metrics.items():
                values = np.asarray(values)
                row |= {f"{key}_mean": values.mean(), f"{key}_sd": values.std(ddof=1),
                        f"{key}_q025": np.quantile(values, .025), f"{key}_median": np.median(values),
                        f"{key}_q975": np.quantile(values, .975)}
            output.append(row)
    return output


def save_loso_figures(summary: Sequence[dict], feature_rows: Sequence[dict], output_dir: Path) -> None:
    states = sorted({row["omitted_state"] for row in summary})
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(states)); width = .36
    for offset, space, label in ((-.18, "primary_7d", "7D"), (.18, "primary_6d_no_procedure", "6D no procedure")):
        values = [next(row["PERMANOVA_R2"] for row in summary if row["omitted_state"] == state and row["space"] == space) for state in states]
        axis.bar(x + offset, values, width, label=label)
    axis.set_xticks(x, states, rotation=25, ha="right"); axis.set_ylabel("PERMANOVA R²")
    axis.set_title("Leave-one-U.S.-state-out multivariate effect size"); axis.legend()
    fig.savefig(output_dir / "leave_one_state_out_permanova_r2.png", dpi=180); plt.close(fig)

    feature_order = [f for f in PRIMARY_FEATURES]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True, sharex=True)
    for axis, feature in zip(axes.flat, feature_order):
        selected = [row for row in feature_rows if row["space"] == "primary_7d" and row["feature"] == feature]
        axis.plot(states, [next(r["cliffs_delta_KR_minus_US"] for r in selected if r["omitted_state"] == state) for state in states], marker="o")
        axis.axhline(0, color="#777", linewidth=.7); axis.set_title(feature); axis.tick_params(axis="x", rotation=45)
    axes.flat[-1].axis("off"); fig.suptitle("Leave-one-U.S.-state-out Cliff's δ (KR − US)")
    fig.savefig(output_dir / "leave_one_state_out_cliffs_delta.png", dpi=180); plt.close(fig)


def report_text(summaries: dict[str, dict], reason_qc: dict, loso: Sequence[dict], full_tests: Sequence[dict], reasoning_tests: Sequence[dict], reproduction: dict) -> str:
    f7, f6 = summaries["full_7d"], summaries["full_6d_no_procedure"]
    r7, r6 = summaries["reasoning_7d"], summaries["reasoning_6d_no_procedure"]
    o7, o6 = summaries["occurrence_7d"], summaries["occurrence_6d_no_procedure"]
    distance_change = (f6["standardized_centroid_distance"] / f7["standardized_centroid_distance"] - 1) * 100
    r2_change = (f6["PERMANOVA_R2"] / f7["PERMANOVA_R2"] - 1) * 100
    loso7 = [row for row in loso if row["space"] == "primary_7d"]
    loso6 = [row for row in loso if row["space"] == "primary_6d_no_procedure"]
    substantive = {"doctrine_density", "party_arg_density", "precedent_density", "statute_density", "remedy_density"}
    full_by = {row["feature"]: row for row in full_tests}
    reason_by = {row["feature"]: row for row in reasoning_tests}
    agreements = sum(np.sign(full_by[f]["cliffs_delta_KR_minus_US"]) == np.sign(reason_by[f]["cliffs_delta_KR_minus_US"]) for f in substantive)
    dispersion_warning = any(row["PERMDISP_p"] < .05 for row in summaries.values())
    if distance_change > -25 and r2_change > -25 and agreements >= 4:
        grade = "A. Suitable for primary LLM projection"
    elif f6["PERMANOVA_R2"] >= .02 and agreements >= 3:
        grade = "B. Suitable only with restricted feature space"
    else:
        grade = "C. Not suitable as a primary jurisdictional reference space"
    recommendation = "Use the six frozen features excluding procedure for downstream LLM projection; preserve the original 7D confirmatory result unchanged." if grade.startswith("B") else "Use the frozen space with the limitations below; do not alter the confirmatory pipeline."
    lines = [
        "# Gate 2 robustness report", "",
        "All statements concern only the sampled confirmatory Korean and U.S. state-court corpus.", "",
        "## Input and reproduction", "",
        f"The frozen corpus contains 160 confirmatory cases (80 KR, 80 US). Primary 7D reproduction: **{'passed' if reproduction['passed'] else 'failed'}**; PC1={f7['PC1_variance']:.6f}, PC2={f7['PC2_variance']:.6f}.", "",
        "## Q1. Dependence on procedure", "",
        f"Removing procedure changed standardized centroid distance from {f7['standardized_centroid_distance']:.4f} to {f6['standardized_centroid_distance']:.4f} ({distance_change:+.1f}%) and PERMANOVA R² from {f7['PERMANOVA_R2']:.4f} to {f6['PERMANOVA_R2']:.4f} ({r2_change:+.1f}%). PC1+PC2 variance changed from {f7['cumulative_PC1_PC2']:.4f} to {f6['cumulative_PC1_PC2']:.4f}. These quantities, rather than the PCA picture alone, determine the interpretation.", "",
        "## Q2. Substantive-reasoning-only text", "",
        f"The frozen development-derived deterministic rule retained mean text ratios KR={reason_qc['KR']['mean']:.3f}, US={reason_qc['US']['mean']:.3f}. Direction agreed with the full-opinion result for {agreements}/5 specified substantive features. Reasoning-only 6D PERMANOVA R²={r6['PERMANOVA_R2']:.4f}, p={r6['PERMANOVA_p']:.5g}. This is a rule-based sensitivity, not a manually validated gold reasoning corpus.", "",
        "## Q3. Multivariate evidence and dispersion", "",
        f"Full 7D: PERMANOVA F={f7['PERMANOVA_F']:.4f}, R²={f7['PERMANOVA_R2']:.4f}, p={f7['PERMANOVA_p']:.5g}; PERMDISP F={f7['PERMDISP_F']:.4f}, p={f7['PERMDISP_p']:.5g}. Full 6D: PERMANOVA F={f6['PERMANOVA_F']:.4f}, R²={f6['PERMANOVA_R2']:.4f}, p={f6['PERMANOVA_p']:.5g}; PERMDISP F={f6['PERMDISP_F']:.4f}, p={f6['PERMDISP_p']:.5g}.", "",
        ("Significant dispersion heterogeneity may contribute to one or more PERMANOVA results; centroid separation must therefore not be treated as definitive from PERMANOVA alone." if dispersion_warning else "No PERMDISP test crossed .05, which is not evidence that dispersions are exactly equal."), "",
        "## Q4. Leave-one-state-out", "",
        f"Across omitted states, 7D PERMANOVA R² ranged {min(r['PERMANOVA_R2'] for r in loso7):.4f}–{max(r['PERMANOVA_R2'] for r in loso7):.4f}; 6D ranged {min(r['PERMANOVA_R2'] for r in loso6):.4f}–{max(r['PERMANOVA_R2'] for r in loso6):.4f}. Standardized centroid distance ranged {min(r['standardized_centroid_distance'] for r in loso6):.4f}–{max(r['standardized_centroid_distance'] for r in loso6):.4f} in 6D. Feature-level directions and exact omissions are in `leave_one_state_out_feature_effects.csv`.", "",
        "## Occurrence sensitivity", "",
        f"Exploratory occurrence-based sensitivity: 7D PERMANOVA R²={o7['PERMANOVA_R2']:.4f}, p={o7['PERMANOVA_p']:.5g}; 6D no-procedure R²={o6['PERMANOVA_R2']:.4f}, p={o6['PERMANOVA_p']:.5g}. This does not replace the primary breadth-based measure.", "",
        "## Q5. Gate 2 decision", "", f"**{grade}.** {recommendation}", "",
        "Before an LLM-generation experiment, independently annotate and validate a bilingual sample of reasoning-only spans; then pre-register the downstream projection feature space and scoring rule. No post-confirmatory feature tuning is warranted here.", "",
        "## Statistical conventions", "",
        "PERMANOVA uses Euclidean distances on independently standardized features. For Euclidean space, sums of squares are computed by the equivalent centroid decomposition. Each p-value is `(1 + count(permuted F >= observed F)) / (1 + permutations)`, so the observed statistic is included. PERMDISP uses arithmetic group centroids and Euclidean distances in the same standardized space; both group mean and median distances are reported.", "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, permutations: int = 10_000, balanced_repeats: int = 1_000, bootstrap_runs: int = BOOTSTRAP_RUNS) -> dict:
    if output_dir.resolve() in {PRIMARY_DISCOVERED.parent.resolve(), OCCURRENCE_DISCOVERED.parent.resolve()}:
        raise ValueError("Refusing to overwrite an existing primary/sensitivity output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, corpus_manifest = load_and_verify_corpus()
    audit = input_audit(records, corpus_manifest)
    write_json(output_dir / "input_audit.json", audit)
    print(f"input audit: {output_dir / 'input_audit.json'}", flush=True)

    matcher = load_matcher()
    documents = confirmatory_documents(records)
    primary = measure_documents(documents, matcher)
    write_csv(output_dir / "primary_features_reproduced.csv", primary, FEATURE_OUTPUT_COLUMNS)
    occurrence = occurrence_rows(primary)
    occurrence_columns = ["case_id", "origin_country", "origin_state", "primary_domain", "decision_year", "doc_length_sentences", *OCCURRENCE_FEATURES]
    write_csv(output_dir / "occurrence_features_reproduced.csv", occurrence, occurrence_columns)

    reasoning_documents, reasoning_manifest, reasoning_qc = build_reasoning_corpus(records, output_dir)
    reasoning = measure_documents(reasoning_documents, matcher)
    write_csv(output_dir / "reasoning_features_160.csv", reasoning, FEATURE_OUTPUT_COLUMNS)
    print("feature spaces prepared", flush=True)

    specs = (
        ("full_7d", primary, tuple(PRIMARY_FEATURES)), ("full_6d_no_procedure", primary, NO_PROCEDURE_FEATURES),
        ("reasoning_7d", reasoning, tuple(PRIMARY_FEATURES)), ("reasoning_6d_no_procedure", reasoning, NO_PROCEDURE_FEATURES),
        ("occurrence_7d", occurrence, OCCURRENCE_FEATURES), ("occurrence_6d_no_procedure", occurrence, OCCURRENCE_NO_PROCEDURE),
    )
    results = {}
    for index, (name, rows, features) in enumerate(specs):
        results[name] = fit_space(name, rows, features, permutations, SEED + index * 10)
        save_space_tables(results[name], output_dir)
    save_pca(results["full_7d"], output_dir / "pca_primary_7d_reproduced.png", "Confirmatory real-reference PCA: primary 7D reproduced")
    save_pca(results["full_6d_no_procedure"], output_dir / "pca_primary_6d_no_procedure.png", "Confirmatory real-reference PCA without procedure feature")
    save_pca(results["reasoning_7d"], output_dir / "pca_reasoning_only_7d.png", "Confirmatory substantive-reasoning-only PCA: 7D")
    save_pca(results["reasoning_6d_no_procedure"], output_dir / "pca_reasoning_only_6d_no_procedure.png", "Confirmatory substantive-reasoning-only PCA without procedure")
    save_pca(results["occurrence_6d_no_procedure"], output_dir / "pca_occurrence_6d_no_procedure.png", "Exploratory occurrence-based PCA without procedure")

    full_desc, full_tests = feature_inference(primary, PRIMARY_FEATURES, bootstrap_runs, SEED + 100)
    reason_desc, reason_tests = feature_inference(reasoning, PRIMARY_FEATURES, bootstrap_runs, SEED + 101)
    write_csv(output_dir / "full_feature_descriptive.csv", full_desc, ["feature", "origin_country", "n", "mean", "sd", "median", "q1", "q3", "min", "max"])
    write_csv(output_dir / "full_feature_tests.csv", full_tests, ["feature", "KR_n", "US_n", "mannwhitney_U", "p_raw", "p_holm", "cliffs_delta_KR_minus_US", "cliffs_delta_CI_low", "cliffs_delta_CI_high", "direction", "bootstrap_runs"])
    write_csv(output_dir / "reasoning_feature_descriptive.csv", reason_desc, ["feature", "origin_country", "n", "mean", "sd", "median", "q1", "q3", "min", "max"])
    write_csv(output_dir / "reasoning_feature_tests.csv", reason_tests, ["feature", "KR_n", "US_n", "mannwhitney_U", "p_raw", "p_holm", "cliffs_delta_KR_minus_US", "cliffs_delta_CI_low", "cliffs_delta_CI_high", "direction", "bootstrap_runs"])

    summary_rows = [space_summary(results[name]) for name, *_ in specs]
    summary_columns = ["space", "n_KR", "n_US", "n_features", "features", "PC1_variance", "PC2_variance", "cumulative_PC1_PC2", "standardized_centroid_distance", "PERMANOVA_F", "PERMANOVA_R2", "PERMANOVA_p", "PERMDISP_F", "PERMDISP_p", "KR_dispersion_mean", "US_dispersion_mean"]
    write_csv(output_dir / "gate2_multivariate_summary.csv", summary_rows, summary_columns)
    write_csv(output_dir / "procedure_ablation_summary.csv", [row for row in summary_rows if row["space"].startswith("full_")], summary_columns)

    reproduction = {
        "audited_baseline": AUDITED_PRIMARY_BASELINE,
        "reproduced": {"n": len(primary), "n_KR": 80, "n_US": 80,
                       "PC1_explained_variance_ratio": results["full_7d"].pca.explained_variance_ratio_[0],
                       "PC2_explained_variance_ratio": results["full_7d"].pca.explained_variance_ratio_[1],
                       "PC1_PC2_cumulative_ratio": results["full_7d"].pca.explained_variance_ratio_[:2].sum()},
    }
    reproduction["passed"] = all(
        math.isclose(float(reproduction["reproduced"][key]), float(AUDITED_PRIMARY_BASELINE[key]), rel_tol=0, abs_tol=1e-12)
        for key in ("n", "n_KR", "n_US", "PC1_explained_variance_ratio", "PC2_explained_variance_ratio", "PC1_PC2_cumulative_ratio")
    )
    write_json(output_dir / "primary_reproduction_check.json", reproduction)

    loso_summary, loso_features = loso_analysis(primary, permutations, SEED + 1000)
    write_csv(output_dir / "leave_one_state_out_summary.csv", loso_summary, summary_columns + ["omitted_state"])
    write_csv(output_dir / "leave_one_state_out_feature_effects.csv", loso_features,
              ["omitted_state", "space", "feature", "n_KR", "n_US", "cliffs_delta_KR_minus_US", "direction", "p_raw", "p_holm"])
    balanced = balanced_loso(primary, balanced_repeats, SEED + 2000)
    balanced_columns = list(balanced[0])
    write_csv(output_dir / "leave_one_state_out_balanced_sensitivity.csv", balanced, balanced_columns)
    save_loso_figures(loso_summary, loso_features, output_dir)

    summaries = {row["space"]: row for row in summary_rows}
    report = report_text(summaries, reasoning_qc, loso_summary, full_tests, reason_tests, reproduction)
    (output_dir / "GATE2_ROBUSTNESS_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    readme = (
        "# Gate 2 robustness outputs\n\n"
        "This sibling directory was selected because the audited primary analyses were under `outputs_v2/real_reference_confirmatory` and `outputs_v2/real_reference_occurrence_sensitivity`. Existing outputs were not intentionally overwritten.\n\n"
        "Run: `python -m scripts.lexicon.run_gate2_robustness`\n\n"
        "The frozen corpus, lexicon, citation parser, marker rules, denominators, case membership, and original analyses are unchanged. The reasoning-only result is a development-set-frozen deterministic sensitivity and is not a manually validated gold segmentation. PCA panels are visualizations, not significance tests. Occurrence spaces are exploratory occurrence-based sensitivity and do not replace primary breadth-based measures. Statistical conventions are stated in `GATE2_ROBUSTNESS_REPORT.md`.\n"
        "\nDuring the initial audit, the ignored primary and occurrence output directories were present and inspected. They disappeared from the working tree during external OneDrive synchronization. The tracked v4.0.2 corpus was restored byte-for-byte from Git HEAD with LF preserved; the ignored prior outputs were not recreated or overwritten in place. Their initially recorded paths/hashes and execution-time availability are documented in `input_audit.json`.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    protected_after = {path: sha256_file(REPO_ROOT / path) for path in audit["all_protected_input_hashes"]}
    if protected_after != audit["all_protected_input_hashes"]:
        raise AssertionError("A protected frozen input changed during Gate 2")
    artifact_hashes = {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file() and path.name not in {"SHA256SUMS.txt", "analysis_manifest.json"}}
    manifest = {
        "analysis": "Gate 2 robustness", "git_commit": git_commit(), "seed": SEED,
        "permutations": permutations, "balanced_repeats": balanced_repeats, "bootstrap_runs": bootstrap_runs,
        "protected_inputs_unchanged": True, "primary_result_modified": False,
        "reasoning_rule_hash": reasoning_manifest[0]["rule_sha256"], "spaces": list(results),
        "artifact_sha256": artifact_hashes,
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    checksums = {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "SHA256SUMS.txt"}
    (output_dir / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="ascii", newline="\n")
    print(f"completed: {output_dir}", flush=True)
    return {"summary": summaries, "reproduction": reproduction, "reasoning_qc": reasoning_qc, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--balanced-repeats", type=int, default=1_000)
    parser.add_argument("--bootstrap-runs", type=int, default=BOOTSTRAP_RUNS)
    args = parser.parse_args()
    if min(args.permutations, args.balanced_repeats, args.bootstrap_runs) < 1:
        parser.error("all iteration counts must be positive")
    run(args.output_dir, args.permutations, args.balanced_repeats, args.bootstrap_runs)


if __name__ == "__main__":
    main()
