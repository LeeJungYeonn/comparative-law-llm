import csv
import hashlib
import json
import math
from collections import Counter

import pytest

from scripts.lexicon.build_dev_candidates import (
    DEFAULT_CORPUS,
    DEFAULT_MANIFEST,
    Document,
    build_rows,
    collect_contexts,
    collect_statistics,
    load_and_verify_corpus,
    ranked_rows,
    tokenize,
    weighted_log_odds,
    write_csv,
    ALL_COLUMNS,
)


@pytest.fixture(scope="module")
def verified():
    return load_and_verify_corpus(DEFAULT_CORPUS, DEFAULT_MANIFEST)


@pytest.fixture(scope="module")
def analysis(verified):
    documents, _, _, _ = verified
    candidates, token_counts = collect_statistics(documents)
    rows = build_rows(candidates, documents, {})
    ranked = ranked_rows(rows)
    contexts = collect_contexts(documents, ranked)
    return candidates, token_counts, rows, ranked, contexts


def test_frozen_development_roster_and_nonempty_text(verified):
    documents, inventory, manifest, actual_hash = verified
    assert len(documents) == len(inventory) == 40
    assert Counter(doc.origin_country for doc in documents) == {"KR": 20, "US": 20}
    assert all(doc.analysis_split == "development" and doc.text.strip() for doc in documents)
    expected = next(value for key, value in manifest["artifact_sha256"].items() if key.endswith(DEFAULT_CORPUS.name))
    assert actual_hash == expected


def test_confirmatory_text_cannot_reach_counts():
    docs = [
        Document(f"KR_{i}", "development", "KR", "", "d", "d", "2000-01-01", "c", "court_opinion", "공동 불법행위")
        for i in range(20)
    ] + [
        Document(f"US_{i}", "development", "US", "S", "d", "d", "2000-01-01", "c", "court_opinion", "summary judgment")
        for i in range(20)
    ]
    poison = Document("US_POISON", "confirmatory", "US", "S", "d", "d", "", "", "", "uniqueconfirmatorypoison")
    with pytest.raises(AssertionError, match="Non-development"):
        collect_statistics(docs + [poison])
    candidates, _ = collect_statistics(docs)
    assert all("uniqueconfirmatorypoison" not in key[1] for key in candidates)


def test_document_frequency_is_binary(analysis):
    candidates, _, _, _, _ = analysis
    repeated = next(stats for stats in candidates.values() if max(stats.counts.values()) > max(map(len, stats.documents.values())))
    for country in ("KR", "US"):
        assert len(repeated.documents[country]) <= 20


def test_context_offsets_and_no_confirmatory_ids(verified, analysis):
    documents, _, _, _ = verified
    _, _, _, ranked, contexts = analysis
    source = {doc.case_id: doc.text for doc in documents}
    development_ids = set(source)
    assert contexts and ranked
    assert all(item["case_id"] in development_ids for item in contexts)
    assert all(source[item["case_id"]][item["start"]:item["end"]] == item["matched_text"] for item in contexts)


def test_one_sided_log_odds_is_finite():
    delta, z_score = weighted_log_odds(7, 0, 100, 80, 7, 180)
    assert math.isfinite(delta) and math.isfinite(z_score)


def test_df1_retained_and_rank_order_deterministic(tmp_path, analysis):
    _, _, rows, ranked, _ = analysis
    assert any(row["kr_df"] + row["us_df"] == 1 for row in rows)
    assert ranked == ranked_rows(list(reversed(rows)))
    path_a, path_b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(path_a, ranked, ALL_COLUMNS)
    write_csv(path_b, ranked_rows(list(reversed(rows))), ALL_COLUMNS)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert hashlib.sha256(path_a.read_bytes()).digest() == hashlib.sha256(path_b.read_bytes()).digest()


def test_identical_analysis_runs_have_byte_identical_candidate_output(tmp_path):
    docs = [
        Document(f"KR_{i:02}", "development", "KR", "", "d", "d", "2000-01-01", "c", "court_opinion", "손해배상 공동 불법행위")
        for i in range(20)
    ] + [
        Document(f"US_{i:02}", "development", "US", "S", "d", "d", "2000-01-01", "c", "court_opinion", "proximate cause and damages")
        for i in range(20)
    ]
    first, _ = collect_statistics(docs)
    second, _ = collect_statistics(list(reversed(docs)))
    rows_a = ranked_rows(build_rows(first, docs, {}))
    rows_b = ranked_rows(build_rows(second, docs, {}))
    path_a, path_b = tmp_path / "run_a.csv", tmp_path / "run_b.csv"
    write_csv(path_a, rows_a, ALL_COLUMNS)
    write_csv(path_b, rows_b, ALL_COLUMNS)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_df2_threshold_is_country_specific():
    docs = [
        Document(f"KR_{i:02}", "development", "KR", "", "d", "d", "", "c", "court_opinion",
                 "shared 고유" if i == 0 else f"한국어{i}")
        for i in range(20)
    ] + [
        Document(f"US_{i:02}", "development", "US", "S", "d", "d", "", "c", "court_opinion",
                 "shared term" if i == 0 else f"uniqueword{i}")
        for i in range(20)
    ]
    candidates, _ = collect_statistics(docs)
    row = next(row for row in build_rows(candidates, docs, {}) if row["normalized_ngram"] == "shared")
    assert (row["kr_df"], row["us_df"]) == (1, 1)
    assert row["meets_df2_threshold"] is False


def test_expected_smoke_candidates_if_present(verified, analysis):
    candidates, _, _, _, _ = analysis
    normalized = {key[1] for key in candidates}
    documents, _, _, _ = verified
    corpus_text = "\n".join(doc.text for doc in documents).lower()
    for phrase in ("proximate cause", "negligence", "summary judgment", "damages", "손해배상", "과실", "원고", "피고", "상고"):
        if phrase in corpus_text:
            assert phrase in normalized


def test_exactly_required_development_inventory_can_be_serialized(tmp_path, verified):
    _, inventory, _, _ = verified
    path = tmp_path / "inventory.csv"
    columns = list(inventory[0])
    write_csv(path, inventory, columns)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
