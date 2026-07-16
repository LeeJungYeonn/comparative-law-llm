from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from collect_ca_raw_cases import evaluate_row as evaluate_ca_row
from collect_kr_raw_cases import evaluate_row as evaluate_kr_row
from pipeline.stage1_raw import apply_duplicate_qc, make_raw_record, require_outputs, sample_records, stratified_sample_with_fallback


def ca_args(**overrides):
    values = {"dataset": "harvard-lil/cold-cases", "min_text_chars": 20, "max_text_chars": 0, "year_min": 0, "year_max": 0}
    values.update(overrides)
    return Namespace(**values)


def kr_args(**overrides):
    values = {"dataset": "lbox/lbox_open", "config": "precedent_corpus", "min_text_chars": 20, "max_text_chars": 0, "year_min": 0, "year_max": 0, "text_col": ""}
    values.update(overrides)
    return Namespace(**values)


def test_kr_criminal_case_excluded():
    row = {"id": "kr-crim", "precedent": "서울중앙지방법원 2021고단123 피고인에게 징역을 선고한다. 손해배상 관련 배상명령."}
    record = evaluate_kr_row(row, kr_args())
    assert record["collection_status"] == "fail"
    assert "criminal_case" in record["exclude_signals"]


def test_kr_supreme_case_excluded():
    row = {"id": "kr-supreme", "precedent": "대법원 2020다12345 손해배상 인정사실 원고는 사고로 상해를 입었다."}
    record = evaluate_kr_row(row, kr_args())
    assert record["collection_status"] == "fail"
    assert "supreme_court_excluded" in record["exclude_signals"]


def test_ca_federal_case_excluded():
    row = {
        "id": "ca-fed",
        "case_name": "Smith v. Widget Co.",
        "court_full_name": "United States District Court, N.D. Cal.",
        "court_jurisdiction": "California",
        "court_type": "FD",
        "date_filed": "2020-01-02",
        "opinions": [{"type": "majority", "opinion_text": "FACTS Plaintiff alleges negligence and damages after an accident."}],
    }
    record = evaluate_ca_row(row, ca_args())
    assert record["collection_status"] == "fail"
    assert "not_california_state_court_or_federal" in record["exclude_signals"]


def test_ca_procedural_only_excluded():
    row = {
        "id": "ca-proc",
        "case_name": "Smith v. Jones",
        "court_full_name": "California Court of Appeal",
        "court_jurisdiction": "California",
        "court_type": "SA",
        "date_filed": "2021-02-03",
        "opinions": [{"type": "majority", "opinion_text": "This damages appeal concerns only the statute of limitations and jurisdiction."}],
    }
    record = evaluate_ca_row(row, ca_args())
    assert record["collection_status"] == "fail"
    assert "procedural_only_or_no_factual_background" in record["exclude_signals"]


def test_duplicate_exact_hash_excludes_second_record():
    first = make_raw_record(
        case_origin="CA",
        jurisdiction="California",
        source_dataset="source",
        source_record_id="1",
        source_url_or_citation="1 Cal. 1",
        case_name="A v. B",
        case_number_or_citation="1 Cal. 1",
        court_name="California Court of Appeal",
        court_level="appellate",
        decision_date="2020-01-01",
        opinion_type="majority",
        procedural_posture="appeal",
        case_subtype="personal_injury",
        raw_text="FACTS A was injured in an accident.",
        include_signals=["damages"],
        exclude_signals=[],
        quality_flags=[],
        collection_status="pass",
    )
    second = dict(first)
    second["case_id"] = "CA_other"
    second["source_record_id"] = "2"
    apply_duplicate_qc([first, second])
    assert second["collection_status"] == "fail"
    assert "duplicate_exact_hash" in second["quality_flags"]
    assert second["related_case_group_id"]


def test_stable_case_id_and_hash_reproducible():
    kwargs = dict(
        case_origin="KR",
        jurisdiction="Korea",
        source_dataset="lbox/lbox_open::precedent_corpus",
        source_record_id="abc",
        source_url_or_citation="2020가단1",
        case_name="손해배상",
        case_number_or_citation="2020가단1",
        court_name="서울중앙지방법원",
        court_level="trial",
        decision_date="2020-01-01",
        opinion_type="main",
        procedural_posture="unknown",
        case_subtype="personal_injury",
        raw_text="인정사실 원고는 사고로 상해를 입었다.",
        include_signals=["손해배상"],
        exclude_signals=[],
        quality_flags=[],
        collection_status="pass",
    )
    left = make_raw_record(**kwargs)
    right = make_raw_record(**kwargs)
    assert left["case_id"] == right["case_id"]
    assert left["raw_text_sha256"] == right["raw_text_sha256"]


def test_overwrite_protection(tmp_path: Path):
    output = tmp_path / "existing.jsonl"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        require_outputs([output], overwrite=False)
    require_outputs([output], overwrite=True)


def test_fixed_seed_sampling_reproducible():
    records = [{"case_id": f"CA_{idx}", "court_level": "appellate", "decision_year": 2020, "case_subtype": "personal_injury"} for idx in range(10)]
    assert [row["case_id"] for row in sample_records(records, 4, 123)] == [row["case_id"] for row in sample_records(records, 4, 123)]


def test_stratified_sampling_prefers_primary_period_before_fallback():
    primary = [
        {"case_id": f"CA_primary_{idx}", "court_level": "appellate", "decision_year": 2015, "case_subtype": "auto_accident"}
        for idx in range(3)
    ]
    fallback = [
        {"case_id": f"CA_fallback_{idx}", "court_level": "appellate", "decision_year": 2005, "case_subtype": "auto_accident"}
        for idx in range(10)
    ]
    selected, meta = stratified_sample_with_fallback(primary + fallback, target_count=3, seed=42)
    assert {row["case_id"] for row in selected} == {row["case_id"] for row in primary}
    assert meta["fallback_used_for_total_shortage"] is False


def test_stratified_sampling_uses_fallback_only_for_total_shortage():
    primary = [{"case_id": "CA_primary", "court_level": "appellate", "decision_year": 2015, "case_subtype": "auto_accident"}]
    fallback = [{"case_id": "CA_fallback", "court_level": "appellate", "decision_year": 2005, "case_subtype": "auto_accident"}]
    selected, meta = stratified_sample_with_fallback(primary + fallback, target_count=2, seed=42)
    assert {row["case_id"] for row in selected} == {"CA_primary", "CA_fallback"}
    assert meta["fallback_used_for_total_shortage"] is True


def test_stratified_sampling_excludes_unknown_court_and_pre_2000():
    records = [
        {"case_id": "CA_ok", "court_level": "appellate", "decision_year": 2015, "case_subtype": "auto_accident"},
        {"case_id": "CA_trial", "court_level": "trial", "decision_year": 2015, "case_subtype": "auto_accident"},
        {"case_id": "CA_old", "court_level": "appellate", "decision_year": 1999, "case_subtype": "auto_accident"},
    ]
    selected, _ = stratified_sample_with_fallback(records, target_count=3, seed=42)
    assert [row["case_id"] for row in selected] == ["CA_ok"]
