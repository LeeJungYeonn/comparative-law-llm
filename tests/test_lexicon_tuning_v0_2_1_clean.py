import csv
import hashlib
import json
from collections import Counter

import pytest

from scripts.lexicon.run_tuning_v0_2_1_clean import (
    CONCEPTS_PATH,
    CORPUS_PATH,
    INPUT_REVISION_LOG_PATH,
    MATCHER_PATH,
    NESTED_SPLIT_PATH,
    OUTPUT_DIR,
    SURFACES_PATH,
    TUNING_MANIFEST_PATH,
    VALIDATION_LOCK_PATH,
    load_tuning_documents,
    read_csv,
    run,
    sha256_file,
    verify_draft_inputs,
    verify_tuning_manifests,
)


@pytest.fixture(scope="module")
def routing():
    tuning, validation, nested_hash = verify_tuning_manifests()
    return tuning, validation, nested_hash


def test_exact_tuning_roster(routing):
    tuning, validation, _ = routing
    assert len(tuning) == 10
    assert Counter(row["origin_country"] for row in tuning.values()) == {"KR": 5, "US": 5}
    assert all(row["analysis_split"] == "development" and row["nested_split"] == "tuning" for row in tuning.values())
    assert not (set(tuning) & validation)


def test_only_tuning_documents_are_loaded(routing):
    tuning, validation, _ = routing
    documents = load_tuning_documents(tuning, validation)
    assert {document.case_id for document in documents} == set(tuning)
    assert not ({document.case_id for document in documents} & validation)


def test_output_offsets_and_case_boundaries(routing):
    tuning, validation, _ = routing
    documents = {document.case_id: document.text for document in load_tuning_documents(tuning, validation)}
    hit_rows = read_csv(OUTPUT_DIR / "tuning_matcher_hits_for_review.csv")
    fn_rows = read_csv(OUTPUT_DIR / "tuning_false_negative_audit.csv")
    summary_rows = read_csv(OUTPUT_DIR / "tuning_matcher_summary.csv")
    assert {row["case_id"] for row in summary_rows} == set(tuning)
    for row in hit_rows:
        assert row["case_id"] in tuning and row["case_id"] not in validation
        assert documents[row["case_id"]][int(row["start"]):int(row["end"])] == row["matched_text"]
    for row in fn_rows:
        assert row["case_id"] in tuning and row["case_id"] not in validation
        assert documents[row["case_id"]][int(row["start"]):int(row["end"])] == row["text_span"]
        assert row["existing_hit_count"] == "0"


def test_manifest_hard_boundaries_and_hashes(routing):
    tuning, _, nested_hash = routing
    manifest = json.loads((OUTPUT_DIR / "tuning_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tuning_cases_used"] == 10
    assert set(manifest["tuning_case_ids"]) == set(tuning)
    assert manifest["discovery_text_used_in_this_run"] is False
    assert manifest["validation_text_used"] is False
    assert manifest["confirmatory_text_used"] is False
    assert manifest["lexicon_modified"] is False
    assert manifest["nested_split_manifest_sha256"] == nested_hash == sha256_file(NESTED_SPLIT_PATH)
    for name, digest in manifest["input_hashes"].items():
        path = {
            CONCEPTS_PATH.name: CONCEPTS_PATH,
            SURFACES_PATH.name: SURFACES_PATH,
            MATCHER_PATH.name: MATCHER_PATH,
            TUNING_MANIFEST_PATH.name: TUNING_MANIFEST_PATH,
            VALIDATION_LOCK_PATH.name: VALIDATION_LOCK_PATH,
            INPUT_REVISION_LOG_PATH.name: INPUT_REVISION_LOG_PATH,
        }[name]
        assert sha256_file(path) == digest


def test_draft_hashes_unchanged_and_revision_log_empty():
    before = verify_draft_inputs()
    assert read_csv(INPUT_REVISION_LOG_PATH) == []
    assert read_csv(OUTPUT_DIR / "tuning_revision_log.csv") == []
    assert verify_draft_inputs() == before


def test_repeated_runs_are_byte_identical(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run(first)
    run(second)
    first_files = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
    second_files = {path.name: path.read_bytes() for path in second.iterdir() if path.is_file()}
    assert first_files == second_files


def test_output_sha256s():
    for line in (OUTPUT_DIR / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        assert sha256_file(OUTPUT_DIR / name) == expected
