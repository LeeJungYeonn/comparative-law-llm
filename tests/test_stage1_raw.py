from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from collect_ca_raw_cases import build_shortlist as build_ca_shortlist
from collect_ca_raw_cases import evaluate_row as evaluate_ca_v4_row
from collect_ca_raw_cases import mark_duplicates as mark_ca_v4_duplicates
from collect_ca_raw_cases import minimum_targets, read_kr_reference
from collect_ca_raw_cases import split_pools as split_ca_v4_pools
from collect_kr_raw_cases import evaluate_row as evaluate_kr_row
from collect_kr_raw_cases import mark_duplicate_candidates, select_final_sample, split_pools
from pipeline.stage1_raw import apply_duplicate_qc, make_raw_record, require_outputs, sample_records, stratified_sample_with_fallback


def ca_v4_args(**overrides):
    values = {
        "dataset": "harvard-lil/cold-cases",
        "court_system": "california-state",
        "court_level": "intermediate-appellate",
        "strict_direct_tort_only": True,
        "publication_status": "any",
        "min_opinion_chars": 20,
        "seed": 42,
    }
    values.update(overrides)
    return Namespace(**values)


def ca_v4_row(**overrides):
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
        "id": overrides.pop("id", "ca-v4"),
        "case_name": overrides.pop("case_name", "Smith v. Jones"),
        "court_full_name": overrides.pop("court_full_name", "California Court of Appeal, Second Appellate District, Division Seven"),
        "court_jurisdiction": overrides.pop("court_jurisdiction", "California, CA"),
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
        "주문 피고의 항소를 기각한다. 항소비용은 피고가 부담한다.\n"
        "항소취지 제1심판결 중 피고 패소 부분을 취소한다.\n"
        "기초사실\n"
        "원고는 피해자이고 피고는 사고 당시 행위자이다. 사건 현장에서 다음 사고가 발생하였다. "
        f"{body} 그 후 원고에게 손해가 발생하였다. "
        "피고는 원고의 과실 때문에 사고가 발생하였고 자신은 주의의무를 다하였다고 주장한다.\n"
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
    first["strict_eligible"] = True
    second["strict_eligible"] = True
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
    record["strict_eligible"] = True
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


def test_ca_v4_target_court_and_main_opinion_codes():
    row = ca_v4_row(opinions=[
        {"type": "040dissent", "opinion_text": "I dissent."},
        {"type": "010combined", "opinion_text": ca_v4_row()["opinions"][0]["opinion_text"]},
    ])
    record = evaluate_ca_v4_row(row, ca_v4_args())
    assert record["court_system"] == "california_state"
    assert record["court_level"] == "intermediate_appellate"
    assert record["main_opinion_type"] == "majority"
    assert len(record["separate_opinions"]) == 1


@pytest.mark.parametrize("court,jurisdiction", [
    ("California Supreme Court", "California, CA"),
    ("U.S. Court of Appeals for the Ninth Circuit", "Federal"),
    ("United States District Court, Central District of California", "Federal"),
    ("Superior Court Appellate Division of the Superior Court of California", "California, CA"),
    ("Supreme Court of Nevada", "Nevada"),
])
def test_ca_v4_non_target_courts_excluded(court, jurisdiction):
    record = evaluate_ca_v4_row(ca_v4_row(court_full_name=court, court_jurisdiction=jurisdiction), ca_v4_args())
    assert record["strict_eligible"] is False
    assert "not_california_court_of_appeal" in record["exclusion_reasons"]


def test_ca_v4_dissent_only_and_headnote_only_excluded():
    dissent = evaluate_ca_v4_row(ca_v4_row(opinions=[{"type": "040dissent", "opinion_text": "I dissent."}]), ca_v4_args())
    headnote = evaluate_ca_v4_row(ca_v4_row(opinions=[], summary="negligence duty damages"), ca_v4_args())
    assert dissent["full_main_opinion_available"] is False
    assert headnote["full_main_opinion_available"] is False
    assert not dissent["strict_eligible"] and not headnote["strict_eligible"]


@pytest.mark.parametrize("text,posture", [
    ("Plaintiff sued the liability insurer directly under Insurance Code section 11580 for payment of the tort judgment.", "direct_action_against_liability_insurer"),
    ("Plaintiff insurer paid its insured and as subrogee sued for reimbursement after the collision.", "insurer_subrogation"),
    ("The cross-complaint sought equitable indemnity and contribution among joint tortfeasors.", "joint_tortfeasor_contribution"),
    ("The dispute concerned insurance coverage, a policy exclusion, and the duty to defend.", "insurance_coverage"),
    ("Plaintiff sought damages for breach of contract under a purchase agreement.", "contract_or_payment"),
    ("The judgment creditor pursued postjudgment collection and a writ of execution.", "judgment_enforcement"),
])
def test_ca_v4_excluded_claim_postures(text, posture):
    record = evaluate_ca_v4_row(ca_v4_row(text=text), ca_v4_args())
    assert record["claim_posture"] == posture
    assert record["strict_eligible"] is False


def test_ca_v4_direct_traffic_premises_and_medical_claims():
    traffic = evaluate_ca_v4_row(ca_v4_row(), ca_v4_args())
    premises_text = "FACTUAL BACKGROUND Plaintiff was a customer of defendant. In 2019 defendant failed to maintain a stairway, and plaintiff fell in an accident. Plaintiff suffered physical injury and medical expenses because of the dangerous condition. Defendant denied notice and disputed causation. Plaintiff sued for premises liability and negligence damages."
    medical_text = "FACTUAL BACKGROUND Plaintiff was a patient of defendant physician. In 2018 defendant performed surgery and failed to diagnose a complication. Plaintiff suffered physical injury and medical expenses as a result. Defendant denied negligence and disputed causation. Plaintiff sued for medical malpractice and professional negligence damages."
    premises = evaluate_ca_v4_row(ca_v4_row(id="premises", text=premises_text), ca_v4_args())
    medical = evaluate_ca_v4_row(ca_v4_row(id="medical", text=medical_text), ca_v4_args())
    assert traffic["claim_posture"] == "direct_tort_claim" and traffic["strict_eligible"]
    assert premises["case_subtype"] == "premises_facility_safety" and premises["strict_eligible"]
    assert medical["case_subtype"] == "medical_professional" and medical["strict_eligible"]


def test_ca_v4_crime_in_wrongful_death_does_not_make_current_case_criminal():
    text = "FACTUAL BACKGROUND Plaintiff sued defendant security company for civil wrongful death negligence. In 2020 an attacker assaulted and killed decedent on defendant's premises. Defendant failed to supervise security, causing the death and damages. Defendant denied notice and disputed causation. Plaintiff brought this civil action for damages."
    record = evaluate_ca_v4_row(ca_v4_row(case_name="Smith v. Security Co.", text=text), ca_v4_args())
    assert record["civil_candidate"] is True
    assert record["death_involved"] is True


def test_ca_v4_legal_only_is_factually_insufficient_and_demurrer_status_preserved():
    legal = evaluate_ca_v4_row(ca_v4_row(text="DISCUSSION The standard of review and legal principles concern negligence, duty, causation, and damages. Prior precedent controls."), ca_v4_args())
    pleading_text = "FACTUAL BACKGROUND On demurrer, plaintiff alleged that defendant failed to warn in 2020. The complaint alleges plaintiff was injured in an accident and suffered bodily injury because of the omission. Defendant disputed causation. Plaintiff sued for negligence damages. We assume the pleaded facts are true."
    pleading = evaluate_ca_v4_row(ca_v4_row(id="pleading", text=pleading_text), ca_v4_args())
    assert legal["factual_background_sufficient"] is False
    assert pleading["procedural_posture"] == "demurrer_or_motion_to_dismiss"
    assert pleading["fact_epistemic_status"] == "assumed_true_for_pleading"


def test_ca_v4_duplicate_removed_from_strict_pool():
    first = evaluate_ca_v4_row(ca_v4_row(id="ca-dupe-1"), ca_v4_args())
    second = evaluate_ca_v4_row(ca_v4_row(id="ca-dupe-2"), ca_v4_args())
    counts = mark_ca_v4_duplicates([first, second])
    pools = split_ca_v4_pools([first, second])
    assert sum(counts.values()) == 1
    assert len(pools["strict"]) == 1


def test_ca_v4_shortlist_is_deterministic_strict_only_and_reports_shortage():
    records = []
    for idx in range(3):
        record = evaluate_ca_v4_row(ca_v4_row(id=f"ca-{idx}", citations=[{"cite": f"{idx} Cal.App.5th 1"}]), ca_v4_args())
        records.append(record)
    kr = {"traffic_accident": 10, "medical_professional": 9}
    left, meta_left = build_ca_shortlist(records, shortlist_count=100, kr_distribution=kr, seed=42)
    right, meta_right = build_ca_shortlist(records, shortlist_count=100, kr_distribution=kr, seed=42)
    assert [row["case_id"] for row in left] == [row["case_id"] for row in right]
    assert all(row["strict_eligible"] and row["claim_posture"] == "direct_tort_claim" for row in left)
    assert len(left) == 3
    assert meta_left["shortage_report"]["medical_professional"]["shortage"] == 14
    assert meta_left == meta_right


def test_ca_v4_minimum_targets_formula():
    assert minimum_targets({"traffic_accident": 10, "general_personal_injury": 1}) == {"traffic_accident": 15, "general_personal_injury": 3}


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
