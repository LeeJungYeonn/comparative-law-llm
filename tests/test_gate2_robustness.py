import numpy as np

from scripts.lexicon.run_gate2_robustness import (
    NO_PROCEDURE_FEATURES,
    PRIMARY_FEATURES,
    PROCEDURE,
    dispersion_distances,
    extract_reasoning_text,
    compile_reasoning_rule,
    permanova_components,
)


def test_procedure_ablation_removes_only_procedure():
    assert PROCEDURE in PRIMARY_FEATURES
    assert PROCEDURE not in NO_PROCEDURE_FEATURES
    assert len(NO_PROCEDURE_FEATURES) == 6
    assert set(NO_PROCEDURE_FEATURES) == set(PRIMARY_FEATURES) - {PROCEDURE}


def test_permanova_euclidean_partition():
    matrix = np.array([[0.0, 0.0], [0.0, 1.0], [4.0, 0.0], [4.0, 1.0]])
    labels = np.array(["KR", "KR", "US", "US"])
    pseudo_f, r2, between, within, total = permanova_components(matrix, labels)
    assert np.isclose(total, between + within)
    assert np.isclose(r2, between / total)
    assert pseudo_f > 0


def test_dispersion_distances_are_group_centroid_distances():
    matrix = np.array([[0.0], [2.0], [10.0], [14.0]])
    labels = np.array(["KR", "KR", "US", "US"])
    assert np.allclose(dispersion_distances(matrix, labels), [1.0, 1.0, 2.0, 2.0])


def test_reasoning_rule_retains_substance_and_excludes_pure_procedure():
    text = (
        "The plaintiff filed a complaint in the district court and filed an appeal. "
        "Negligence requires duty, breach, causation, and damages."
    )
    reasoning, included, excluded, flags = extract_reasoning_text(text, "US", compile_reasoning_rule())
    assert "Negligence requires duty" in reasoning
    assert "filed a complaint" not in reasoning
    assert included and excluded
    assert isinstance(flags, list)
