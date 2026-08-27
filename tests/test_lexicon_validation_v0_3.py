import json
from collections import Counter

import pytest

from scripts.lexicon.run_validation_v0_3 import (
    CONCEPTS_PATH,
    CORPUS_PATH,
    HUMAN_CSV_NAMES,
    MATCHER_PATH,
    OUTPUT_DIR,
    PREVALIDATION_HASH_PATH,
    SURFACES_PATH,
    VALIDATION_MANIFEST_PATH,
    load_validation_documents,
    read_csv,
    run,
    sha256_file,
    verify_prevalidation_hashes,
    verify_validation_roster,
)


@pytest.fixture(scope="module")
def roster():
    return verify_validation_roster()


def test_exact_locked_validation_roster(roster):
    validation, discovery, tuning, validation_ids = roster
    assert len(validation) == len(validation_ids) == 10
    assert Counter(row["origin_country"] for row in validation.values()) == {"KR": 5, "US": 5}
    assert all(row["analysis_split"] == "development" and row["nested_split"] == "validation" for row in validation.values())
    assert not (validation_ids & discovery or validation_ids & tuning)


def test_only_validation_documents_are_loaded(roster):
    validation, discovery, tuning, validation_ids = roster
    documents, confirmatory = load_validation_documents(validation)
    processed = {document.case_id for document in documents}
    assert processed == validation_ids
    assert not (processed & discovery or processed & tuning or processed & confirmatory)
    assert len(confirmatory) == 160


def test_output_ids_and_exact_offsets(roster):
    validation, _, _, validation_ids = roster
    documents, _ = load_validation_documents(validation)
    source = {document.case_id: document.text for document in documents}
    hits = read_csv(OUTPUT_DIR / "validation_matcher_hits_for_review.csv")
    fn_rows = read_csv(OUTPUT_DIR / "validation_false_negative_audit.csv")
    summaries = read_csv(OUTPUT_DIR / "validation_matcher_summary.csv")
    assert {row["case_id"] for row in summaries} == validation_ids
    assert {row["case_id"] for row in fn_rows} == validation_ids
    for row in hits:
        assert row["case_id"] in validation_ids
        assert source[row["case_id"]][int(row["start"]):int(row["end"])] == row["matched_text"]
    for row in fn_rows:
        assert row["case_id"] in validation_ids
        assert source[row["case_id"]][int(row["start"]):int(row["end"])] == row["text_span"]
        assert row["existing_hit_count"] == "0"


def test_prevalidation_hashes_and_manifest_are_unchanged():
    before = verify_prevalidation_hashes()
    manifest = json.loads((OUTPUT_DIR / "validation_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prevalidation_version"] == "v0.3"
    assert manifest["validation_cases_used"] == 10
    assert manifest["discovery_cases_used"] == 0
    assert manifest["tuning_cases_used"] == 0
    assert manifest["confirmatory_cases_used"] == 0
    assert manifest["lexicon_modified"] is False
    assert manifest["validation_used_for_lexicon_modification"] is False
    assert manifest["reviewed_hit_precision"] is None
    assert manifest["recall"] is None and manifest["f1"] is None
    assert manifest["input_hashes"]["concepts"] == sha256_file(CONCEPTS_PATH)
    assert manifest["input_hashes"]["surface_forms"] == sha256_file(SURFACES_PATH)
    assert manifest["input_hashes"]["matcher"] == sha256_file(MATCHER_PATH)
    assert manifest["input_hashes"]["PREVALIDATION_SHA256.json"] == sha256_file(PREVALIDATION_HASH_PATH)
    assert manifest["input_hashes"]["source_corpus"] == sha256_file(CORPUS_PATH)
    assert manifest["input_hashes"]["validation_manifest"] == sha256_file(VALIDATION_MANIFEST_PATH)
    assert verify_prevalidation_hashes() == before


def test_human_fields_blank_and_human_csvs_have_bom():
    hit_rows = read_csv(OUTPUT_DIR / "validation_matcher_hits_for_review.csv")
    fn_rows = read_csv(OUTPUT_DIR / "validation_false_negative_audit.csv")
    for row in hit_rows:
        assert not any(row[column] for column in (
            "human_tp_fp", "human_category_correct", "human_concept_correct",
            "human_surface_too_broad", "human_notes",
        ))
    for row in fn_rows:
        assert not any(row[column] for column in (
            "human_missed_concept", "human_expected_category", "human_expected_concept_family",
            "human_expected_surface", "human_notes",
        ))
    for name in HUMAN_CSV_NAMES:
        assert (OUTPUT_DIR / name).read_bytes().startswith(b"\xef\xbb\xbf")


def test_repeated_validation_execution_is_byte_identical(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run(first)
    run(second)
    first_files = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
    second_files = {path.name: path.read_bytes() for path in second.iterdir() if path.is_file()}
    assert first_files == second_files


def test_output_checksums():
    for line in (OUTPUT_DIR / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        assert sha256_file(OUTPUT_DIR / name) == expected
