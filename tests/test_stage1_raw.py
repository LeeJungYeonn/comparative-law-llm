from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from collect_ca_raw_cases import evaluate_row as evaluate_ca_v3_row
from collect_ca_raw_cases import mark_duplicates as mark_ca_v3_duplicates
from collect_ca_raw_cases import select_final_sample as select_ca_v3_final_sample
from collect_ca_raw_cases import split_pools as split_ca_v3_pools
from collect_kr_raw_cases import evaluate_row as evaluate_kr_row
from collect_kr_raw_cases import mark_duplicate_candidates, select_final_sample, split_pools
from pipeline.stage1_raw import apply_duplicate_qc, make_raw_record, require_outputs, sample_records, stratified_sample_with_fallback


def ca_v3_args(**overrides):
    values = {
        "dataset": "harvard-lil/cold-cases",
        "year_min": 2010,
        "year_max": 2021,
        "court_system": "california-state",
        "court_level": "intermediate-appellate",
        "strict_tort_only": True,
        "publication_status": "any",
        "min_text_chars": 20,
        "max_text_chars": 0,
        "target_count": 20,
        "seed": 42,
        "reference_kr_selected": "",
        "match_kr_subtypes": True,
        "match_kr_years": True,
        "match_kr_lengths": True,
    }
    values.update(overrides)
    return Namespace(**values)


def ca_v3_row(**overrides):
    text = overrides.pop(
        "text",
        (
            "FACTUAL BACKGROUND Plaintiff was a pedestrian. In 2020 defendant drove a vehicle and struck plaintiff in a collision. "
            "Plaintiff suffered bodily injury, medical expenses, emotional distress, and other damages. "
            "Defendant denied negligence and argued comparative fault after the accident. "
            "DISCUSSION The negligence claim required duty, breach, causation, and damages."
        ),
    )
    row = {
        "id": overrides.pop("id", "ca-v3"),
        "case_name": overrides.pop("case_name", "Smith v. Jones"),
        "court_full_name": overrides.pop("court_full_name", "California Court of Appeal, Second Appellate District, Division Seven"),
        "court_jurisdiction": overrides.pop("court_jurisdiction", "California"),
        "court_type": overrides.pop("court_type", "SA"),
        "date_filed": overrides.pop("date_filed", "2020-01-01"),
        "citations": overrides.pop("citations", [{"cite": "1 Cal.App.5th 1"}]),
        "opinions": overrides.pop("opinions", [{"type": "majority", "opinion_text": text}]),
    }
    row.update(overrides)
    return row


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


def test_ca_v3_court_of_appeal_included():
    record = evaluate_ca_v3_row(ca_v3_row(), ca_v3_args())
    assert record["court_system"] == "california_state"
    assert record["court_level"] == "intermediate_appellate"
    assert record["court_level_confidence"] == "high"
    assert record["strict_eligible"] is True


def test_ca_v3_supreme_court_excluded():
    record = evaluate_ca_v3_row(
        ca_v3_row(court_full_name="Supreme Court of California", court_type="ST"),
        ca_v3_args(),
    )
    assert record["court_level"] == "supreme"
    assert "california_supreme_excluded" in record["exclusion_reasons"]


def test_ca_v3_federal_courts_excluded():
    ninth = evaluate_ca_v3_row(ca_v3_row(court_full_name="United States Court of Appeals for the Ninth Circuit", court_jurisdiction="Federal", court_type="F"), ca_v3_args())
    district = evaluate_ca_v3_row(ca_v3_row(court_full_name="United States District Court, Central District of California", court_jurisdiction="Federal", court_type="FD"), ca_v3_args())
    assert ninth["court_level"] == "federal"
    assert district["court_level"] == "federal"
    assert "federal_court_excluded" in ninth["exclusion_reasons"]
    assert "federal_court_excluded" in district["exclusion_reasons"]


def test_ca_v3_superior_appellate_division_excluded():
    record = evaluate_ca_v3_row(ca_v3_row(court_full_name="Superior Court Appellate Division of the Superior Court of California"), ca_v3_args())
    assert record["court_level"] == "trial_or_other"
    assert any(reason.startswith("non_target_court_level") for reason in record["exclusion_reasons"])


def test_ca_v3_other_state_mentions_california_excluded():
    record = evaluate_ca_v3_row(
        ca_v3_row(court_full_name="Supreme Court of Nevada", court_jurisdiction="Nevada", court_type="ST", text="FACTUAL BACKGROUND The California defendant drove a vehicle in 2020 and struck plaintiff, causing bodily injury and damages."),
        ca_v3_args(),
    )
    assert record["court_system"] != "california_state"
    assert record["strict_eligible"] is False


def test_ca_v3_majority_selected_over_dissent():
    row = ca_v3_row(
        opinions=[
            {"type": "dissent", "opinion_text": "I dissent."},
            {"type": "majority", "opinion_text": "FACTUAL BACKGROUND Plaintiff was injured in 2020 when defendant failed to maintain stairs. Plaintiff suffered bodily injury and damages. Defendant disputed causation. DISCUSSION negligence damages."},
        ]
    )
    record = evaluate_ca_v3_row(row, ca_v3_args())
    assert record["main_opinion_type"] == "majority"
    assert record["separate_opinion_count"] == 1


def test_ca_v3_dissent_only_excluded():
    row = ca_v3_row(opinions=[{"type": "dissent", "opinion_text": "Plaintiff was injured and damages are discussed only in dissent."}])
    record = evaluate_ca_v3_row(row, ca_v3_args())
    assert record["main_opinion_type"] == "dissent"
    assert record["strict_eligible"] is False


def test_ca_v3_automobile_negligence_is_non_contractual_tort():
    record = evaluate_ca_v3_row(ca_v3_row(), ca_v3_args())
    assert record["liability_basis"] == "non_contractual_tort"
    assert record["case_subtype"] == "traffic_accident"


def test_ca_v3_premises_liability_is_non_contractual_tort():
    record = evaluate_ca_v3_row(
        ca_v3_row(text="FACTUAL BACKGROUND Plaintiff was a customer. In 2019 defendant failed to maintain a stairway and plaintiff fell in an accident. Plaintiff suffered bodily injury and medical damages. Defendant disputed notice and causation. DISCUSSION premises liability negligence."),
        ca_v3_args(),
    )
    assert record["liability_basis"] == "non_contractual_tort"
    assert record["case_subtype"] == "premises_facility_safety"


def test_ca_v3_medical_malpractice_is_non_contractual_tort():
    record = evaluate_ca_v3_row(
        ca_v3_row(text="FACTUAL BACKGROUND Plaintiff was a patient. In 2018 defendant physician performed surgery and failed to diagnose a complication. Plaintiff suffered bodily injury, medical expenses, and damages. Defendant disputed causation. DISCUSSION medical malpractice professional negligence."),
        ca_v3_args(),
    )
    assert record["liability_basis"] == "non_contractual_tort"
    assert record["case_subtype"] == "medical_professional"


def test_ca_v3_contract_insurance_procedural_and_criminal_classification():
    contract = evaluate_ca_v3_row(ca_v3_row(text="FACTUAL BACKGROUND Plaintiff and defendant signed a purchase agreement in 2020. Defendant breached the contract and failed to pay contract damages. No bodily injury occurred."), ca_v3_args())
    insurance = evaluate_ca_v3_row(ca_v3_row(text="FACTUAL BACKGROUND The parties disputed insurance coverage and policy interpretation after an accident. The only issue was the insurer duty to defend and payment obligation."), ca_v3_args())
    procedural = evaluate_ca_v3_row(ca_v3_row(text="FACTUAL BACKGROUND Plaintiff filed late. The appeal concerns only the statute of limitations and jurisdiction, with no underlying accident facts."), ca_v3_args())
    criminal = evaluate_ca_v3_row(ca_v3_row(case_name="People v. Smith", text="FACTUAL BACKGROUND Defendant was convicted of felony assault and sentenced to prison. This criminal appeal concerns Penal Code instructions."), ca_v3_args())
    assert contract["liability_basis"] == "contract_only"
    assert insurance["liability_basis"] == "insurance_only"
    assert procedural["liability_basis"] == "procedural_only"
    assert criminal["criminal_case_likely"] is True


def test_ca_v3_probate_in_re_classifies_family_or_probate():
    record = evaluate_ca_v3_row(
        ca_v3_row(case_name="In re Estate of Smith", text="FACTUAL BACKGROUND This probate trust dispute involved an estate, beneficiaries, and distribution of property. The appeal did not involve independent negligence injury damages."),
        ca_v3_args(),
    )
    assert record["liability_basis"] == "family_or_probate"


def test_ca_v3_underlying_crime_wrongful_death_stays_civil_tort_candidate():
    record = evaluate_ca_v3_row(
        ca_v3_row(case_name="Smith v. Security Co.", text="FACTUAL BACKGROUND Plaintiff brought a civil wrongful death negligence action after a criminal assault in 2020. Defendant security company failed to supervise the premises, and decedent was killed. Plaintiff suffered damages and defendant disputed causation. DISCUSSION wrongful death negligence."),
        ca_v3_args(),
    )
    assert record["criminal_case_likely"] is False
    assert record["liability_basis"] == "non_contractual_tort"


def test_ca_v3_legal_rules_only_factually_insufficient():
    record = evaluate_ca_v3_row(
        ca_v3_row(text="DISCUSSION We review the standard of review and legal principles for negligence, causation, duty, breach, and damages. Prior precedent governs the issue."),
        ca_v3_args(),
    )
    assert record["factual_background_sufficient"] is False


def test_ca_v3_demurrer_fact_status_is_assumed_true():
    record = evaluate_ca_v3_row(
        ca_v3_row(text="FACTUAL BACKGROUND On demurrer, plaintiff alleged that defendant failed to warn in 2020 and plaintiff suffered bodily injury and damages. Defendant disputed causation. DISCUSSION The demurrer assumes pleaded facts are true."),
        ca_v3_args(),
    )
    assert record["procedural_posture"] == "demurrer_or_motion_to_dismiss"
    assert record["fact_status"] == "assumed_true_at_pleading_stage"


def test_ca_v3_strict_pool_and_selected_count_are_separate():
    records = []
    for idx in range(2):
        records.append(evaluate_ca_v3_row(ca_v3_row(id=f"ca-pool-{idx}", date_filed=f"202{idx}-01-01"), ca_v3_args()))
    pools = split_ca_v3_pools(records, [])
    selected, _, _ = select_ca_v3_final_sample(pools["strict_eligible"], ca_v3_args(target_count=1, match_kr_subtypes=False, match_kr_years=False, match_kr_lengths=False))
    assert len(pools["strict_eligible"]) == 2
    assert len(selected) == 1


def test_ca_v3_fixed_seed_sampling_reproducible_and_no_subtype_relaxation():
    records = [evaluate_ca_v3_row(ca_v3_row(id="ca-only", text="FACTUAL BACKGROUND Plaintiff was a pedestrian. In 2020 defendant drove a car and struck plaintiff in a collision. Plaintiff suffered bodily injury and damages. Defendant disputed causation. DISCUSSION negligence damages."), ca_v3_args())]
    args = ca_v3_args(target_count=20, match_kr_subtypes=False, match_kr_years=False, match_kr_lengths=False)
    left, _, meta_left = select_ca_v3_final_sample(records, args)
    right, _, meta_right = select_ca_v3_final_sample(records, args)
    assert [row["case_id"] for row in left] == [row["case_id"] for row in right]
    assert len(left) == 1
    assert meta_left["quota_shortage_report"]["traffic_accident"]["shortage"] > 0
    assert meta_right["quota_shortage_report"]["traffic_accident"]["shortage"] > 0


def test_ca_v3_related_duplicate_marked():
    first = evaluate_ca_v3_row(ca_v3_row(id="ca-dupe-1"), ca_v3_args())
    second = dict(first)
    second["case_id"] = "CA_other"
    second["source_record_id"] = "ca-dupe-2"
    counts = mark_ca_v3_duplicates([first, second])
    assert counts["duplicate_citation"] == 1
    assert second["duplicate_or_related_reason"] == "duplicate_citation"


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
