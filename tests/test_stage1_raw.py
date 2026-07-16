from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from collect_ca_raw_cases import evaluate_row as evaluate_ca_row
from collect_kr_raw_cases import evaluate_row as evaluate_kr_row
from collect_kr_raw_cases import mark_duplicate_candidates, select_final_sample, split_pools
from pipeline.stage1_raw import apply_duplicate_qc, make_raw_record, require_outputs, sample_records, stratified_sample_with_fallback


def ca_args(**overrides):
    values = {"dataset": "harvard-lil/cold-cases", "min_text_chars": 20, "max_text_chars": 0, "year_min": 0, "year_max": 0}
    values.update(overrides)
    return Namespace(**values)


def kr_args(**overrides):
    values = {
        "dataset": "lbox/lbox_open",
        "config": "precedent_corpus",
        "min_text_chars": 20,
        "max_text_chars": 0,
        "year_min": 0,
        "year_max": 9999,
        "text_col": "",
        "court_level": "appellate",
        "strict_tort_only": True,
        "target_count": 20,
        "seed": 42,
        "allow_year_fallback": False,
        "relax_subtype_quota": False,
    }
    values.update(overrides)
    return Namespace(**values)


def kr_case_text(body: str) -> str:
    return (
        "서울고등법원 2019나12345 손해배상\n"
        "기초사실\n"
        f"{body}\n"
        "판단\n"
        "민법 제750조의 불법행위 책임이 문제 된다."
    )


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


def test_kr_case_number_na_classifies_appellate():
    row = {
        "id": "kr-app",
        "case_number": "2019나12345",
        "court_name": "서울고등법원",
        "decision_date": "2019. 5. 1.",
        "precedent": kr_case_text("2018. 1. 1. 피고가 차량을 운전하다 원고 차량을 충돌하여 원고가 상해를 입고 치료비 손해가 발생하였다."),
    }
    record = evaluate_kr_row(row, kr_args(year_min=2010, year_max=2021))
    assert record["court_level"] == "appellate"
    assert record["court_level_confidence"] == "high"
    assert record["strict_eligible"] is True


def test_kr_case_number_da_classifies_supreme():
    row = {
        "id": "kr-sup-da",
        "case_number": "2020다12345",
        "court_name": "대법원",
        "decision_date": "2020. 1. 1.",
        "precedent": "대법원 2020다12345 상고이유 손해배상 관련 법리를 판단한다.",
    }
    record = evaluate_kr_row(row, kr_args())
    assert record["court_level"] == "supreme"


def test_kr_high_court_name_is_appellate_evidence():
    row = {
        "id": "kr-high-court",
        "case_number": "2019나12345",
        "court_name": "서울고등법원",
        "decision_date": "2019. 1. 1.",
        "precedent": kr_case_text("2018. 1. 1. 피고가 시설을 방치하여 원고가 추락 사고로 상해와 치료비 손해를 입었다."),
    }
    record = evaluate_kr_row(row, kr_args())
    assert record["court_level"] == "appellate"
    assert any("court_name: 서울고등법원" == item for item in record["court_level_evidence"])


def test_kr_supreme_roles_with_supreme_court_are_supreme():
    row = {
        "id": "kr-sup-role",
        "court_name": "대법원",
        "case_number": "2020다12345",
        "precedent": "대법원 상고이유 피상고인 원심판결 손해배상",
    }
    record = evaluate_kr_row(row, kr_args())
    assert record["court_level"] == "supreme"


def test_kr_metadata_case_number_beats_cited_case_number():
    row = {
        "id": "kr-citation",
        "case_number": "2019나12345",
        "court_name": "서울고등법원",
        "decision_date": "2019. 5. 1.",
        "precedent": kr_case_text("대법원 2020다99999 판결을 인용한다. 2018. 1. 1. 피고가 차량을 충돌하여 원고가 상해와 치료비 손해를 입었다."),
    }
    record = evaluate_kr_row(row, kr_args())
    assert record["case_number"] == "2019나12345"
    assert record["court_level"] == "appellate"


def test_kr_contract_damages_classifies_contract_only():
    row = {
        "id": "kr-contract",
        "case_number": "2019나22222",
        "court_name": "서울고등법원",
        "decision_date": "2019. 5. 1.",
        "precedent": "손해배상(기) 기초사실 원고와 피고는 매매계약을 체결하였다. 피고가 대금 지급 계약상 의무를 이행하지 않아 계약 위반 손해배상이 문제 되었다.",
    }
    record = evaluate_kr_row(row, kr_args())
    assert record["liability_basis"] == "contract_only"
    assert record["strict_eligible"] is False


def test_kr_traffic_personal_injury_classifies_non_contractual_tort():
    row = {
        "id": "kr-traffic",
        "case_number": "2019나33333",
        "court_name": "서울고등법원",
        "decision_date": "2019. 5. 1.",
        "precedent": kr_case_text("2018. 1. 1. 피고가 자동차를 운전하다 원고를 추돌하였다. 원고는 상해를 입고 치료비와 위자료 손해가 발생하였다."),
    }
    record = evaluate_kr_row(row, kr_args(year_min=2010, year_max=2021))
    assert record["liability_basis"] == "non_contractual_tort"
    assert record["case_subtype"] == "traffic_accident"


def test_kr_insurance_policy_only_classifies_insurance_only():
    row = {
        "id": "kr-insurance",
        "case_number": "2019나44444",
        "court_name": "서울고등법원",
        "decision_date": "2019. 5. 1.",
        "precedent": "손해배상 보험금 청구 기초사실 원고와 피고는 보험계약을 체결하였다. 이 사건은 보험약관의 면책 조항과 보험금 지급 범위만 문제 된다.",
    }
    record = evaluate_kr_row(row, kr_args())
    assert record["liability_basis"] == "insurance_only"
    assert record["strict_eligible"] is False


def test_kr_legal_principles_only_is_factually_insufficient():
    row = {
        "id": "kr-legal-only",
        "case_number": "2019나55555",
        "court_name": "서울고등법원",
        "decision_date": "2019. 5. 1.",
        "precedent": "손해배상 관련 법리 대법원은 불법행위 손해배상책임의 요건과 과실 및 인과관계에 관한 법리는 다음과 같다고 판시하였다.",
    }
    record = evaluate_kr_row(row, kr_args())
    assert record["factual_background_sufficient"] is False


def test_kr_strict_pool_and_selected_count_are_separate():
    first = evaluate_kr_row(
        {
            "id": "kr-s1",
            "case_number": "2019나10001",
            "court_name": "서울고등법원",
            "decision_date": "2019. 1. 1.",
            "precedent": kr_case_text("2018. 1. 1. 피고가 차량을 충돌하여 원고가 상해와 치료비 손해를 입었다."),
        },
        kr_args(year_min=2010, year_max=2021, target_count=1),
    )
    second = evaluate_kr_row(
        {
            "id": "kr-s2",
            "case_number": "2018나10002",
            "court_name": "서울고등법원",
            "decision_date": "2018. 1. 1.",
            "precedent": kr_case_text("2017. 1. 1. 피고가 시설을 방치하여 원고가 추락하고 치료비 손해를 입었다."),
        },
        kr_args(year_min=2010, year_max=2021, target_count=1),
    )
    args = kr_args(year_min=2010, year_max=2021, target_count=1)
    pools = split_pools([first, second], args)
    selected, _ = select_final_sample(pools["strict_eligible"], args)
    assert len(pools["strict_eligible"]) == 2
    assert len(selected) == 1


def test_kr_target_count_does_not_relax_strict_shortage():
    record = evaluate_kr_row(
        {
            "id": "kr-shortage",
            "case_number": "2019나77777",
            "court_name": "서울고등법원",
            "decision_date": "2019. 1. 1.",
            "precedent": kr_case_text("2018. 1. 1. 피고가 자동차를 충돌하여 원고가 상해와 치료비 손해를 입었다."),
        },
        kr_args(year_min=2010, year_max=2021),
    )
    selected, meta = select_final_sample([record], kr_args(year_min=2010, year_max=2021, target_count=20))
    assert len(selected) == 1
    assert meta["shortage"] == 19


def test_kr_related_duplicate_marked():
    first = evaluate_kr_row(
        {
            "id": "kr-dupe1",
            "case_number": "2019나88888",
            "court_name": "서울고등법원",
            "decision_date": "2019. 1. 1.",
            "precedent": kr_case_text("2018. 1. 1. 피고가 차량을 충돌하여 원고가 상해와 치료비 손해를 입었다."),
        },
        kr_args(),
    )
    second = dict(first)
    second["case_id"] = "KR_other"
    second["source_record_id"] = "kr-dupe2"
    counts = mark_duplicate_candidates([first, second])
    assert counts["duplicate_exact_hash"] == 1
    assert second["duplicate_of_case_id"] == first["case_id"]


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
