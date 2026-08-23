import csv
import hashlib
import json
from collections import Counter

import pytest

from scripts.lexicon.build_nested_discovery_v1 import (
    CORPUS_PATH,
    FREEZE_MANIFEST_PATH,
    OUTPUT_DIR,
    SPLIT_COLUMNS,
    SPLITS,
    assign_nested_split,
    assert_artifact_case_ids,
    collect_candidate_stats,
    load_discovery_documents,
    read_verified_metadata,
    sha256_file,
    split_rows,
    validate_revision_evidence,
    write_csv,
)


@pytest.fixture(scope="module")
def nested():
    development, confirmatory_ids, manifest, source_hash = read_verified_metadata(
        CORPUS_PATH, FREEZE_MANIFEST_PATH
    )
    split_map = assign_nested_split(development)
    return development, confirmatory_ids, manifest, source_hash, split_map


def test_development_40_is_exact_nested_partition(nested):
    development, confirmatory_ids, _, _, split_map = nested
    assert len(development) == len(split_map) == 40
    assert Counter(split_map.values()) == {"discovery": 20, "tuning": 10, "validation": 10}
    subsets = [{case_id for case_id, value in split_map.items() if value == split} for split in SPLITS]
    assert set.union(*subsets) == {item.case_id for item in development}
    assert all(not (left & right) for index, left in enumerate(subsets) for right in subsets[index + 1:])
    assert not (set(split_map) & confirmatory_ids)


def test_exact_country_counts(nested):
    development, _, _, _, split_map = nested
    countries = {item.case_id: item.origin_country for item in development}
    assert Counter((split_map[case_id], countries[case_id]) for case_id in split_map) == {
        ("discovery", "KR"): 10, ("discovery", "US"): 10,
        ("tuning", "KR"): 5, ("tuning", "US"): 5,
        ("validation", "KR"): 5, ("validation", "US"): 5,
    }


def test_identical_runs_make_byte_identical_split_manifest(tmp_path, nested):
    development, _, _, _, first = nested
    second = assign_nested_split(list(reversed(development)))
    assert first == second
    path_a, path_b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(path_a, split_rows(development, first), SPLIT_COLUMNS)
    write_csv(path_b, split_rows(list(reversed(development)), second), SPLIT_COLUMNS)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_discovery_statistics_reject_non_discovery_documents(nested):
    development, confirmatory_ids, _, _, split_map = nested
    documents = load_discovery_documents(CORPUS_PATH, split_map)
    discovery_ids = {item.metadata.case_id for item in documents}
    assert len(documents) == 20
    assert Counter(item.metadata.origin_country for item in documents) == {"KR": 10, "US": 10}
    assert discovery_ids == {case_id for case_id, split in split_map.items() if split == "discovery"}
    assert not (discovery_ids & confirmatory_ids)
    candidates = collect_candidate_stats(documents)
    assert candidates and all(stats.documents["KR"] | stats.documents["US"] <= discovery_ids for stats in candidates.values())
    with pytest.raises(AssertionError):
        collect_candidate_stats(documents[:-1])


def test_future_tuning_and_validation_artifact_guards(nested):
    _, _, _, _, split_map = nested
    tuning = [case_id for case_id, split in split_map.items() if split == "tuning"]
    validation = [case_id for case_id, split in split_map.items() if split == "validation"]
    assert_artifact_case_ids(tuning, "tuning", split_map)
    assert_artifact_case_ids(validation, "validation", split_map)
    with pytest.raises(AssertionError, match="Wrong nested split"):
        assert_artifact_case_ids(validation[:1], "tuning", split_map)
    with pytest.raises(AssertionError, match="Wrong nested split"):
        assert_artifact_case_ids(tuning[:1], "validation", split_map)


def test_revision_evidence_forbids_validation_and_confirmatory(nested):
    _, confirmatory_ids, _, _, split_map = nested
    discovery_id = next(case_id for case_id, split in split_map.items() if split == "discovery")
    tuning_id = next(case_id for case_id, split in split_map.items() if split == "tuning")
    validate_revision_evidence([
        {"evidence_split": "official_resource", "case_id": ""},
        {"evidence_split": "discovery", "case_id": discovery_id},
        {"evidence_split": "tuning", "case_id": tuning_id},
    ], split_map, confirmatory_ids)
    validation_id = next(case_id for case_id, split in split_map.items() if split == "validation")
    with pytest.raises(AssertionError):
        validate_revision_evidence([{"evidence_split": "validation", "case_id": validation_id}], split_map, confirmatory_ids)
    with pytest.raises(AssertionError):
        validate_revision_evidence([{"evidence_split": "tuning", "case_id": next(iter(confirmatory_ids))}], split_map, confirmatory_ids)


def test_source_hash_still_matches_frozen_manifest(nested):
    _, _, manifest, source_hash, _ = nested
    expected = next(value for key, value in manifest["artifact_sha256"].items() if key.endswith(CORPUS_PATH.name))
    assert source_hash == expected == sha256_file(CORPUS_PATH)


def test_generated_hashes_and_evidence_boundaries_when_outputs_exist(nested):
    assert OUTPUT_DIR.exists(), "Run build_nested_discovery_v1.py before the complete test suite"
    for line in (OUTPUT_DIR / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        assert sha256_file(OUTPUT_DIR / relative) == expected
    run_manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["confirmatory_text_used_for_lexicon_development"] is False
    assert run_manifest["validation_text_used_for_lexicon_modification"] is False
    split_manifest_hash = sha256_file(OUTPUT_DIR / "dev_nested_split_manifest.csv")
    assert run_manifest["nested_split_manifest_sha256"] == split_manifest_hash

    development, confirmatory_ids, _, _, split_map = nested
    discovery_ids = {case_id for case_id, split in split_map.items() if split == "discovery"}
    context_path = OUTPUT_DIR / "discovery_candidates/discovery_candidate_contexts.jsonl"
    with context_path.open(encoding="utf-8") as handle:
        for line in handle:
            context = json.loads(line)
            assert context["nested_split"] == "discovery"
            assert context["case_id"] in discovery_ids
            assert context["case_id"] not in confirmatory_ids

    revision_path = OUTPUT_DIR / "tuning_revision_log_template.csv"
    with revision_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
