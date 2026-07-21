from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from collect_kr_raw_cases import evaluate_row, load_sampling_config, select_final_sample, split_pools


ROOT = Path(__file__).resolve().parents[1]
V3_SELECTED = ROOT / "outputs/raw/kr_v3/kr_cases_selected_50.jsonl"


def args(**overrides) -> Namespace:
    values = {
        "dataset": "lbox/lbox_open",
        "config": "precedent_corpus",
        "text_col": "",
        "court_level": "appellate",
        "require_verified_decision_year": False,
        "min_text_chars": 20,
        "max_text_chars": 0,
        "target_count": 50,
        "seed": 42,
        "relax_subtype_quota": False,
        "require_human_accept": False,
        "do_not_use_year_for_sampling": True,
        "sampling_config": str(ROOT / "configs/tort_n50_sampling.yaml"),
    }
    values.update(overrides)
    return Namespace(**values)


def direct_traffic_text(*, incorporated: bool = False) -> str:
    incorporation = "민사소송법 제420조 본문에 따라 이를 그대로 인용한다. " if incorporated else ""
    facts = (
        "기초사실 원고는 보행자이고 피고는 승용차 운전자이다. "
        "2020. 5. 20. 서울시 도로의 횡단보도에서 피고가 전방주시를 하지 않고 자동차를 운전하다 원고를 충돌하였다. "
        "이 사고로 인하여 원고는 다리 골절의 상해를 입고 병원 치료비, 일실수입과 위자료 손해가 발생하였다. "
        "그 후 원고는 계속 치료를 받았다. 피고는 원고가 신호를 위반하였으므로 원고의 과실이 크고 자신은 주의의무를 다하였다고 주장한다. "
    )
    if incorporated:
        facts = "기초사실 이 부분은 제1심판결 이유와 같다. "
    return (
        "주문 피고의 항소를 기각한다. 항소비용은 피고가 부담한다. "
        "항소취지 제1심판결 중 피고 패소 부분을 취소한다. "
        f"이유 {incorporation}{facts}"
        "판단 피고는 민법 제750조 불법행위로 인한 손해를 원고에게 배상할 의무가 있다."
    )


def direct_row(**overrides):
    row = {
        "id": "direct-traffic",
        "case_number": "2021나12345",
        "court_name": "서울고등법원",
        "precedent": direct_traffic_text(),
    }
    row.update(overrides)
    return row


def test_incident_date_is_not_decision_date():
    record = evaluate_row(direct_row(), args())
    assert record["decision_date"] is None
    assert record["decision_year"] is None
    assert record["decision_date_verified"] is False
    assert record["incident_date"] == "2020-05-20"


def test_cited_case_number_is_not_current_case_number():
    text = direct_traffic_text() + " 관련 법리로 대법원 2020다99999 판결을 인용한다."
    record = evaluate_row({"id": "cited", "precedent": text}, args())
    assert record["current_case_number"] is None
    assert "2020다99999" in record["cited_case_numbers"]


def test_current_appellate_signals_required_and_recorded():
    record = evaluate_row(direct_row(), args())
    assert record["court_level"] == "appellate"
    assert record["court_level_confidence"] == "high"
    assert record["appellate_evidence_count"] >= 2
    assert record["current_case_number"] == "2021나12345"


def test_insurer_recovery_can_be_traffic_but_is_not_strict():
    text = (
        "주문 원고 보험회사의 항소를 기각한다. 항소비용은 원고가 부담한다. 항소취지 제1심판결을 취소한다. "
        "기초사실 자동차 충돌 교통사고가 발생하자 원고 보험회사는 피해자에게 보험금을 지급하였다. "
        "원고는 보험자대위에 따라 피고 보험회사에 구상금을 청구한다."
    )
    record = evaluate_row({"id": "subrogation", "case_number": "2021나22222", "court_name": "서울고등법원", "precedent": text}, args())
    assert record["case_subtype"] == "traffic_accident"
    assert record["claim_posture"] == "insurer_subrogation"
    assert record["strict_eligible"] is False


def test_direct_traffic_claim_is_strict_without_verified_year():
    record = evaluate_row(direct_row(), args())
    assert record["claim_posture"] == "direct_tort_claim"
    assert record["liability_basis"] == "non_contractual_tort"
    assert record["facts_independently_reconstructable"] is True
    assert record["strict_eligible"] is True


def test_substantive_product_and_premises_signals_outrank_treatment_mentions():
    product = evaluate_row(
        direct_row(id="product", precedent=direct_traffic_text().replace("자동차를 운전하다 원고를 충돌", "결함 있는 제품이 폭발하여 원고를 상해").replace("승용차 운전자", "제품 제조업자")),
        args(),
    )
    premises = evaluate_row(
        direct_row(id="premises", precedent=direct_traffic_text().replace("자동차를 운전하다 원고를 충돌", "안전관리를 하지 않은 계단에서 원고가 추락")),
        args(),
    )
    assert product["case_subtype"] == "product_safety"
    assert premises["case_subtype"] == "premises_facility_safety"


def test_mostly_incorporated_document_is_factually_insufficient():
    record = evaluate_row(direct_row(id="incorporated", precedent=direct_traffic_text(incorporated=True)), args())
    assert record["facts_independently_reconstructable"] is False
    assert record["fact_source_quality"] == "mostly_incorporated"
    assert record["strict_eligible"] is False


def test_v3_regression_fixtures_have_expected_non_direct_postures():
    expected = {
        "KR_61c61d33ca3d73dc": "contract_or_payment",
        "KR_8726dc96a20ac82c": "judgment_enforcement",
        "KR_acdc0358fe600040": "wage_or_compensation",
        "KR_ae4b2d017d3e6e24": "contract_or_payment",
        "KR_0efbffc2ab4fa14f": "wage_or_compensation",
        "KR_26ec13fbc2ae1680": "joint_tortfeasor_contribution",
    }
    found = {}
    with V3_SELECTED.open(encoding="utf-8") as handle:
        for line in handle:
            old = json.loads(line)
            if old["case_id"] not in expected:
                continue
            record = evaluate_row({"id": old["source_record_id"], "precedent": old["raw_text"]}, args())
            found[old["case_id"]] = record["claim_posture"]
            assert record["strict_eligible"] is False
    assert found == expected


def synthetic(case_id: str, subtype: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "strict_eligible": True,
        "duplicate_or_related_reason": None,
        "case_subtype": subtype,
        "court_level_confidence": "high",
        "factual_sufficiency_score": 90,
        "human_qc_status": "",
        "sampling_rank": None,
        "sampling_reasons": [],
    }


def balanced_pool() -> list[dict[str, object]]:
    counts = {
        "traffic_accident": 30,
        "medical_professional": 12,
        "premises_facility_safety": 12,
        "product_safety": 10,
        "employer_vicarious_liability": 8,
        "privacy_reputation": 8,
        "general_personal_injury": 8,
        "property_damage": 8,
    }
    return [synthetic(f"KR_{subtype}_{idx:02d}", subtype) for subtype, count in counts.items() for idx in range(count)]


def test_strict_pool_is_not_truncated_by_target_count():
    records = balanced_pool()
    pools = split_pools(records, args())
    selected, _ = select_final_sample(pools["strict_eligible"], args(target_count=50))
    assert len(pools["strict_eligible"]) == len(records)
    assert len(selected) == 50


def test_quota_relaxation_default_off_and_traffic_cap_ten():
    traffic_only = [synthetic(f"KR_t_{idx}", "traffic_accident") for idx in range(30)]
    selected, meta = select_final_sample(traffic_only, args())
    assert meta["relax_subtype_quota"] is False
    assert len(selected) == 10
    selected_balanced, _ = select_final_sample(balanced_pool(), args())
    assert sum(row["case_subtype"] == "traffic_accident" for row in selected_balanced) == 10


def test_fixed_seed_reproducible():
    left, _ = select_final_sample(balanced_pool(), args(seed=42))
    right, _ = select_final_sample(balanced_pool(), args(seed=42))
    assert [row["case_id"] for row in left] == [row["case_id"] for row in right]


def test_sampling_config_has_explicit_n50_quotas():
    config = load_sampling_config(ROOT / "configs/tort_n50_sampling.yaml")
    assert sum(config["quotas"].values()) == 50
    assert config["quotas"]["traffic_accident"] == 10
