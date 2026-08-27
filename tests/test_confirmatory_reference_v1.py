import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.lexicon.confirmatory_measurement_v1 import (
    PRIMARY_FEATURES,
    SENSITIVITY_FEATURES,
    citation_counts,
    extract_features,
    precedent_citations,
    sentence_spans,
    statute_citations,
)
from scripts.lexicon.run_confirmatory_reference_v1 import (
    ANALYSIS_OUTPUT,
    BOOTSTRAP_RUNS,
    CORPUS_PATH,
    LEXICON_FILES,
    LEXICON_OUTPUT,
    MEASUREMENT_PATH,
    PREVALIDATION_DIR,
    PREVALIDATION_SEAL,
    cliffs_delta,
    fit_pca,
    holm_adjust,
    read_csv,
    sha256_file,
    verify_prevalidation_seal,
)


class SyntheticMatcher:
    @staticmethod
    def match_text(text, surface_path, concept_path, language):
        hits = []
        for category, word in (("doctrine", "duty"), ("remedy", "damages"),
                               ("procedure", "appeal"), ("party_arg", "argues")):
            start = text.find(word)
            if start >= 0:
                hits.append({
                    "category": category, "concept_id": category.upper(),
                    "concept_family": category.upper() + "_FAMILY", "surface_id": "SYN",
                    "matched_text": word, "start": start, "end": start + len(word),
                })
        return hits


def test_frozen_sentence_rule_on_synthetic_text():
    text = "First sentence. Second sentence!\n셋째 문장이다."
    spans = sentence_spans(text)
    assert len(spans) == 3
    assert [text[start:end] for start, end in spans] == ["First sentence.", "Second sentence!", "셋째 문장이다."]


def test_jurisdiction_citation_rules_on_synthetic_text():
    kr = "민법 제750조에 따른다. 대법원 2018. 1. 1. 선고 2017다12345 판결을 참조한다."
    us = "See 42 U.S.C. § 1983 and Smith, 123 U.S. 456; also 2020 WL 12345."
    assert citation_counts(statute_citations(kr, "KR")) == (1, 1)
    assert citation_counts(precedent_citations(kr, "KR"))[0] >= 1
    assert citation_counts(statute_citations(us, "US")) == (1, 1)
    assert citation_counts(precedent_citations(us, "US")) == (2, 2)


def test_synthetic_feature_definitions_are_finite(tmp_path):
    text = "The plaintiff argues duty and seeks damages. The appeal follows 42 U.S.C. § 1983."
    features, hits = extract_features(text, "US", SyntheticMatcher(), tmp_path / "s.csv", tmp_path / "c.csv")
    assert features["doc_length_sentences"] == 2
    assert len(hits) == 4
    assert all(math.isfinite(float(features[name])) for name in SENSITIVITY_FEATURES)
    assert all(features[f"{category}_unique_concepts"] == 1 for category in ("doctrine", "remedy", "procedure", "party_arg"))


def test_statistical_primitives_are_deterministic():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 1.0])
    assert cliffs_delta(a, b) == pytest.approx(7 / 9)
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def _require_outputs():
    if not ANALYSIS_OUTPUT.exists() or not LEXICON_OUTPUT.exists():
        pytest.skip("Run run_confirmatory_reference_v1.py before artifact tests")


def test_frozen_lexicon_is_byte_equivalent_and_hash_stable():
    _require_outputs()
    sealed = verify_prevalidation_seal()
    for source_name, target_name in LEXICON_FILES.items():
        source, target = PREVALIDATION_DIR / source_name, LEXICON_OUTPUT / target_name
        assert source.read_bytes() == target.read_bytes()
        assert sha256_file(target) == sealed[source_name]
    manifest = json.loads((LEXICON_OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_derived_modifications"] == 0
    assert manifest["validation_used_for_lexicon_modification"] is False


def test_reference_feature_boundaries_and_finite_values():
    _require_outputs()
    rows = read_csv(ANALYSIS_OUTPUT / "reference_features_160.csv")
    assert len(rows) == 160 and len({row["case_id"] for row in rows}) == 160
    assert {row["origin_country"] for row in rows} == {"KR", "US"}
    assert sum(row["origin_country"] == "KR" for row in rows) == 80
    assert sum(row["origin_country"] == "US" for row in rows) == 80
    assert all(float(row["doc_length_sentences"]) > 0 for row in rows)
    assert all(math.isfinite(float(row[feature])) for row in rows for feature in PRIMARY_FEATURES)
    tests = read_csv(ANALYSIS_OUTPUT / "mannwhitney_cliffs_primary.csv")
    assert len(tests) == 7 and {row["feature"] for row in tests} == set(PRIMARY_FEATURES)


def test_feature_order_and_pca_fit_population_are_frozen():
    _require_outputs()
    scaler = json.loads((ANALYSIS_OUTPUT / "reference_scaler_7d.json").read_text(encoding="utf-8"))
    pca = json.loads((ANALYSIS_OUTPUT / "reference_pca_7d.json").read_text(encoding="utf-8"))
    manifest = json.loads((ANALYSIS_OUTPUT / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert scaler["feature_order"] == pca["feature_order"] == PRIMARY_FEATURES
    assert scaler["n_samples_seen"] == 160 and pca["n_samples"] == 160
    assert scaler["fit_population"] == pca["fit_population"] == "real-reference confirmatory 160 only"
    assert manifest["development_cases_used"] == 0 and manifest["LLM_outputs_used"] == 0
    assert manifest["confirmatory_data_used_to_modify_measurement_pipeline"] is False


def test_saved_numerical_transform_matches_deterministic_refit():
    _require_outputs()
    rows = read_csv(ANALYSIS_OUTPUT / "reference_features_160.csv")
    scaler_a, pca_a, scores_a = fit_pca(rows, PRIMARY_FEATURES)
    scaler_b, pca_b, scores_b = fit_pca(rows, PRIMARY_FEATURES)
    assert np.allclose(scaler_a.mean_, scaler_b.mean_)
    assert np.allclose(pca_a.components_, pca_b.components_)
    assert np.allclose(scores_a, scores_b)
    saved = json.loads((ANALYSIS_OUTPUT / "reference_pca_7d.json").read_text(encoding="utf-8"))
    assert np.allclose(pca_a.components_, np.asarray(saved["components"]))


def test_hit_offsets_hashes_and_bom():
    _require_outputs()
    # Load source text only for confirmatory IDs present in the hit artifact.
    hits = read_csv(ANALYSIS_OUTPUT / "reference_lexicon_hits_160.csv")
    needed = {row["case_id"] for row in hits}
    source = {}
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["case_id"] in needed:
                assert record["analysis_split"] == "confirmatory"
                source[record["case_id"]] = record["main_opinion_text"]
    for row in hits:
        assert source[row["case_id"]][int(row["start"]):int(row["end"])] == row["matched_text"]
    for line in (ANALYSIS_OUTPUT / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        assert sha256_file(ANALYSIS_OUTPUT / name) == expected
    assert (ANALYSIS_OUTPUT / "reference_features_160.csv").read_bytes().startswith(b"\xef\xbb\xbf")
