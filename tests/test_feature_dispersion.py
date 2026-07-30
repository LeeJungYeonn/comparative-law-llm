import numpy as np
import pandas as pd

from analyze_kr_ca_features import (
    CA_DOCTRINE_TERMS,
    CA_PCA_DOCTRINE_TERMS,
    CA_REMEDY_TERMS,
    KR_DOCTRINE_TERMS,
    KR_PCA_DOCTRINE_TERMS,
    KR_REMEDY_TERMS,
    PCA_FEATURES,
    dispersion_analysis,
    extract_features,
)


def test_pca_feature_space_is_log1p_and_doctrine_remedy_exclusive() -> None:
    assert len(PCA_FEATURES) == 8
    assert all(feature.startswith("log1p(") for feature in PCA_FEATURES)
    assert len(KR_PCA_DOCTRINE_TERMS) < len(KR_DOCTRINE_TERMS)
    assert len(CA_PCA_DOCTRINE_TERMS) == len(CA_DOCTRINE_TERMS)
    assert not any(
        doctrine in remedy or remedy in doctrine
        for doctrine in KR_PCA_DOCTRINE_TERMS
        for remedy in KR_REMEDY_TERMS
    )
    assert not any(
        doctrine in remedy or remedy in doctrine
        for doctrine in CA_PCA_DOCTRINE_TERMS
        for remedy in CA_REMEDY_TERMS
    )

    features = extract_features(
        {"case_id": "KR_test", "raw_text": "손해배상 위자료 상당인과관계."},
        "KR",
    )
    assert features["remedy_exclusive_term_count"] >= 2
    assert features["doctrine_exclusive_term_count"] >= 1
    assert features["doctrine_exclusive_term_count"] < features["doctrine_term_count"]
    assert features["log1p(doctrine_exclusive_per_1k)"] == np.log1p(
        features["doctrine_exclusive_per_1k"]
    )


def test_dispersion_analysis_detects_larger_kr_spread() -> None:
    rows = []
    coordinates = []
    for jurisdiction, radius in [("KR", 3.0), ("CA", 1.0)]:
        for dimension in range(len(PCA_FEATURES)):
            for sign in (-1.0, 1.0):
                vector = np.zeros(len(PCA_FEATURES))
                vector[dimension] = sign * radius
                coordinates.append(vector)
                rows.append(
                    {
                        "case_id": f"{jurisdiction}_{dimension}_{sign:+.0f}",
                        "jurisdiction": jurisdiction,
                    }
                )

    features = pd.DataFrame(rows)
    distances, summary = dispersion_analysis(
        features,
        np.asarray(coordinates),
        permutations=999,
        seed=42,
    )
    result = summary.iloc[0]

    assert len(distances) == 32
    assert np.allclose(
        distances.loc[
            distances["jurisdiction"] == "KR",
            "distance_to_group_centroid_8d",
        ],
        3.0,
    )
    assert np.allclose(
        distances.loc[
            distances["jurisdiction"] == "CA",
            "distance_to_group_centroid_8d",
        ],
        1.0,
    )
    assert result.mean_difference_KR_minus_CA == 2.0
    assert result.mean_ratio_KR_over_CA == 3.0
    assert result.p_permdisp_two_sided <= 0.01
    assert result.p_directional_KR_greater <= 0.01


def test_dispersion_analysis_rejects_zero_permutations() -> None:
    features = pd.DataFrame(
        {
            "case_id": ["KR_1", "CA_1"],
            "jurisdiction": ["KR", "CA"],
        }
    )
    scaled = np.zeros((2, len(PCA_FEATURES)))

    try:
        dispersion_analysis(features, scaled, permutations=0, seed=42)
    except ValueError as error:
        assert "at least 1" in str(error)
    else:
        raise AssertionError("Expected ValueError for zero permutations")
