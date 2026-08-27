import hashlib
from pathlib import Path

import numpy as np

from scripts.lexicon.run_occurrence_sensitivity_v1 import (
    AVAILABLE_INPUT,
    AVAILABLE_PRIMARY,
    BOOTSTRAP_RUNS,
    OCCURRENCE_FEATURES,
    PCA_FEATURES,
    derive_occurrence_features,
    fit_pca,
    holm_adjust,
    protected_hashes,
    read_csv,
    run,
)


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*")) if file.is_file()
    }


def test_occurrence_feature_guards():
    rows = derive_occurrence_features(read_csv(AVAILABLE_INPUT))
    assert len(rows) == 160
    assert sum(row["origin_country"] == "KR" for row in rows) == 80
    assert sum(row["origin_country"] == "US" for row in rows) == 80
    assert all(float(row["doc_length_sentences"]) > 0 for row in rows)
    assert all(
        np.isfinite(float(row[feature])) and float(row[feature]) >= 0
        for row in rows for feature in OCCURRENCE_FEATURES
    )
    assert PCA_FEATURES == ("avg_chars_per_sentence", *OCCURRENCE_FEATURES)
    scaler, pca, standardized, scores = fit_pca(rows)
    assert standardized.shape == scores.shape == (160, 7)
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scores, pca.transform(scaler.transform(
        np.array([[float(row[feature]) for feature in PCA_FEATURES] for row in rows])
    )))


def test_holm_is_exactly_six_tests():
    adjusted = holm_adjust([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert len(adjusted) == 6


def test_run_is_deterministic_and_protected_files_do_not_change(tmp_path):
    protected_before = protected_hashes(AVAILABLE_INPUT, AVAILABLE_PRIMARY)
    output = tmp_path / "occurrence"
    first = run(AVAILABLE_INPUT, AVAILABLE_PRIMARY, output)
    first_hashes = directory_hashes(output)
    second = run(AVAILABLE_INPUT, AVAILABLE_PRIMARY, output)
    second_hashes = directory_hashes(output)
    protected_after = protected_hashes(AVAILABLE_INPUT, AVAILABLE_PRIMARY)

    assert first["manifest"] == second["manifest"]
    assert first_hashes == second_hashes
    assert protected_before == protected_after
    assert first["manifest"]["mannwhitney_tests"] == 6
    assert first["manifest"]["holm_correction_tests"] == 6
    assert first["manifest"]["bootstrap_runs"] == BOOTSTRAP_RUNS
    assert first["manifest"]["primary_analysis_modified"] is False
    assert first["manifest"]["lexicon_modified"] is False
    assert first["manifest"]["primary_pca_modified_or_refit"] is False
    assert first["manifest"]["exploratory_occurrence_pca_fitted"] is True
    assert first["manifest"]["occurrence_pca_feature_order"] == list(PCA_FEATURES)
    assert first["manifest"]["LLM_outputs_used"] is False
    assert (output / "occurrence_pca_7d.json").exists()
    assert (output / "occurrence_scaler_7d.json").exists()
    assert (output / "occurrence_pca_7d_pc1_pc2.png").exists()
