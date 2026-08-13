from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from extract_neutral_facts_v2 import select_smoke_cases
from finalize_case_sample_v2 import flexible_state_domain_flow, main as finalize_main, max_flow_counts
from pipeline_v2.io_utils import write_json, write_jsonl
from pipeline_v2.legalize_kr import parse_markdown_record
from pipeline_v2.rules import (
    assess_fact_sufficiency, classify_domain, classify_kr_court, date_in_window, is_us_state_highcourt,
    leakage_checks, neutralize_jurisdiction_signals, select_main_opinion, source_span_grounding, strip_legal_citations,
    translation_equivalence_checks,
)


def test_kr_supreme_civil_case_number_requires_reinforcing_signal() -> None:
    text = "대법원 판결\n사건 2020다12345 손해배상\n주문 원심판결을 파기한다.\n이유 상고이유를 판단한다."
    result = classify_kr_court(text)
    assert (result["court_level"], result["court_level_confidence"], result["case_code"]) == ("supreme", "high", "다")
    assert len(result["court_level_evidence"]) >= 2


def test_kr_lower_court_is_excluded_even_with_supreme_vocabulary_in_body() -> None:
    text = "사건 2020나123 손해배상\n대법원 판례를 참조한다."
    result = classify_kr_court(text, structured_court="서울고등법원", structured_case_number="2020나123")
    assert result["court_level"] == "lower"
    assert result["court_level_confidence"] == "high"


def test_us_court_type_s_is_required() -> None:
    ok, evidence = is_us_state_highcourt({"court_type": "S", "court_jurisdiction": "Oregon, OR", "court_full_name": "Supreme Court of Oregon"})
    assert ok and "court_type=S" in evidence


def test_us_sa_st_and_federal_are_excluded() -> None:
    for court_type in ("SA", "ST"):
        assert not is_us_state_highcourt({"court_type": court_type, "court_jurisdiction": "Oregon, OR", "court_full_name": "Oregon Court"})[0]
    assert not is_us_state_highcourt({"court_type": "S", "court_jurisdiction": "Federal", "court_full_name": "United States Supreme Court"})[0]


def test_main_opinion_selection_prefers_combined_and_separates_dissent() -> None:
    row = {"opinions": [
        {"opinion_id": "d", "type": "040dissent", "opinion_text": "d" * 4000},
        {"opinion_id": "l", "type": "020lead", "opinion_text": "l" * 3000},
        {"opinion_id": "c", "type": "010combined", "opinion_text": "c" * 2000},
    ]}
    result = select_main_opinion(row)
    assert result["main_opinion_type"] == "combined"
    assert result["main_opinion_text"] == "c" * 2000
    assert result["has_dissent"] is True


def test_date_range_boundaries_and_invalid_dates() -> None:
    assert date_in_window("2000-01-01") and date_in_window("2025-12-31")
    assert not date_in_window("1999-12-31") and not date_in_window("2026-01-01")
    assert not date_in_window("2025-02-30")


def test_product_domain_outranks_generic_injury_signal() -> None:
    result = classify_domain("A defective product lacked a warning and caused personal injury through negligence.")
    assert result["case_domain"] == "product_liability"


def test_employer_theory_is_secondary_to_general_injury_domain() -> None:
    result = classify_domain("An employee negligently drove within the scope of employment and injured a pedestrian.")
    assert result["primary_domain"] == "general_negligence_personal_injury"
    assert "vicarious_liability" in result["liability_theories"]


def test_core_fact_sufficiency_does_not_require_defense_or_location() -> None:
    result = assess_fact_sufficiency("[PERSON_A] drove and caused an injury to [PERSON_B].")
    assert result["core_fact_sufficient"] is True
    assert result["fact_sufficiency_score"] == 4
    assert result["preferred_fact_sufficiency"] is False


def test_translation_equivalence_normalizes_number_words_months_and_ordinals() -> None:
    checks = translation_equivalence_checks(
        "On December 12, 2007, [PERSON_A] visited the first-floor office with three experts.",
        "2007년 12월 12일 [PERSON_A]는 전문가 3명과 1층 사무실을 방문했다.",
        "en",
    )
    assert checks["translation_equivalence_status"] == "pass"


def test_translation_equivalence_normalizes_repeated_months_korean_counters_and_total_others() -> None:
    assert translation_equivalence_checks(
        "2005년 5월 17일과 2005년 5월 29일", "May 17, 2005 and May 29, 2005", "ko"
    )["translation_equivalence_status"] == "pass"
    assert translation_equivalence_checks(
        "[COMPANY_A] 등 7인과 한 대의 차량", "[COMPANY_A] and six others with one vehicle", "ko"
    )["translation_equivalence_status"] == "pass"
    assert translation_equivalence_checks(
        "2008년 8월 20일경 안전에 관한 대책을 제출했다.",
        "Around August 20, 2008, safety measures were submitted.",
        "ko",
    )["translation_equivalence_status"] == "pass"


def test_translation_equivalence_accepts_semantic_negation_and_placeholder_repetition() -> None:
    checks = translation_equivalence_checks(
        "[PERSON_A]는 [PERSON_B]가 더 이상 일을 하지 않았다고 주장했다.",
        "[PERSON_A] claimed that [PERSON_B] stopped working.",
        "ko",
    )
    assert checks["translation_equivalence_status"] == "pass"


def test_leakage_checks_do_not_match_state_names_inside_words_or_dates_as_citations() -> None:
    checks = leakage_checks("The condition remained unchanged from 19 March 1993 until 28 June 2003.")
    assert checks["jurisdiction_leakage_status"] == "pass"
    assert checks["legal_leakage_status"] == "pass"
    assert leakage_checks("The event occurred in Maine.")["jurisdiction_leakage_status"] == "fail"
    neutralized, evidence = neutralize_jurisdiction_signals("The product was in the Texas area.")
    assert neutralized == "The product was in the [LOCATION_JURISDICTION] area."
    assert evidence == ["Texas"]
    assert leakage_checks("A follow-up brain CT scan confirmed the injury.")["jurisdiction_leakage_status"] == "pass"


def test_smoke_selection_prioritizes_fact_score_and_domain_diversity() -> None:
    cases = [
        {"case_id": "KR_A", "origin_country": "KR", "fact_sufficiency_score": 7, "primary_domain": "general"},
        {"case_id": "KR_B", "origin_country": "KR", "fact_sufficiency_score": 7, "primary_domain": "general"},
        {"case_id": "KR_C", "origin_country": "KR", "fact_sufficiency_score": 6, "primary_domain": "medical"},
        {"case_id": "US_A", "origin_country": "US", "fact_sufficiency_score": 7, "primary_domain": "general"},
        {"case_id": "US_B", "origin_country": "US", "fact_sufficiency_score": 6, "primary_domain": "product"},
    ]
    selected = select_smoke_cases(cases, 2)
    assert [row["case_id"] for row in selected] == ["KR_A", "KR_C", "US_A", "US_B"]


def test_legalize_markdown_structured_fields_and_opinion(tmp_path: Path) -> None:
    path = tmp_path / "case.md"
    path.write_text("---\n판례일련번호: '1'\n사건종류: 민사\n법원등급: 대법원\n선고일자: 2020-01-01\n---\n# 제목\n\n## 판례내용\n\n사실 본문\n", encoding="utf-8")
    metadata, _, opinion = parse_markdown_record(path)
    assert metadata["사건종류"] == "민사" and metadata["법원등급"] == "대법원"
    assert opinion == "사실 본문"


def test_placeholder_and_number_consistency_detects_missing_entity() -> None:
    ok = translation_equivalence_checks("[PERSON_A] drove 97 kilometers.", "[PERSON_A]은 97킬로미터를 운전했다.", "en")
    bad = translation_equivalence_checks("[PERSON_A] drove 97 kilometers.", "그 사람은 79킬로미터를 운전했다.", "en")
    assert ok["translation_equivalence_status"] == "pass"
    assert bad["translation_equivalence_issues"] == ["placeholder_mismatch"]
    assert bad["translation_equivalence_warnings"] == ["number_mismatch"]


def test_legal_citation_stripping_preserves_factual_sentence() -> None:
    text = "민법 제750조에 따라 [PERSON_A]는 차량을 운전했다. 123 P.3d 456 (Or. 2005)."
    stripped = strip_legal_citations(text)
    assert "민법" not in stripped and "123 P.3d 456" not in stripped
    assert "[PERSON_A]는 차량을 운전했다" in stripped


def test_jurisdiction_leakage_detects_state_and_court() -> None:
    result = leakage_checks("The California Supreme Court reviewed the event.")
    assert result["jurisdiction_leakage_status"] == "fail"
    assert result["legal_leakage_status"] == "fail"


def test_source_span_grounding_whitespace_normalization_and_failure() -> None:
    assert source_span_grounding("A  driver\nwarned B.", "A driver warned B.")[0] == "pass"
    assert source_span_grounding("A warned B.", "A never warned B.")[0] == "fail"


def test_numeric_translation_consistency() -> None:
    assert translation_equivalence_checks("approximately 3.05 meters", "약 3.05미터", "en")["translation_equivalence_status"] == "pass"
    assert "number_mismatch" in translation_equivalence_checks("approximately 3.05 meters", "약 3.5미터", "en")["translation_equivalence_warnings"]


def test_duplicate_family_capacity_cannot_fill_state_quota() -> None:
    states = ["A", "B", "C", "D", "E"]
    availability = {(state, "general_negligence_personal_injury"): (19 if state == "A" else 20) for state in states}
    flow, _ = max_flow_counts(states, availability, {"general_negligence_personal_injury": 100})
    assert flow == 99


def test_flexible_state_bounds_are_enforced() -> None:
    states = ["A", "B", "C", "D", "E"]
    targets = {"general_negligence_personal_injury": 100, "medical_professional_liability": 0, "product_liability": 0, "other_civil_liability": 0}
    enough = {(state, "general_negligence_personal_injury"): 20 for state in states}
    feasible, counts = flexible_state_domain_flow(states, enough, targets)
    assert feasible and all(sum(counts[state, domain] for domain in targets) == 20 for state in states)
    enough["A", "general_negligence_personal_injury"] = 9
    assert flexible_state_domain_flow(states, enough, targets)[0] is False


def _fact(case_id: str, country: str, domain: str) -> dict:
    source_language = "ko" if country == "KR" else "en"
    source = "[PERSON_A]가 1개의 경고를 받았다." if country == "KR" else "[PERSON_A] received 1 warning."
    return {"case_id": case_id, "case_family_id": f"F_{case_id}", "origin_country": country, "origin_state": None, "case_domain": domain, "source_language": source_language, "neutral_fact_source": source, "neutral_fact_ko": "[PERSON_A]가 1개의 경고를 받았다.", "neutral_fact_en": "[PERSON_A] received 1 warning.", "aligned_fact_units": [{"fact_id": "F001", "source_text": source, "neutral_ko": "[PERSON_A]가 1개의 경고를 받았다.", "neutral_en": "[PERSON_A] received 1 warning.", "translation_status": "aligned", "translation_equivalence_status": "pass"}]}


def test_finalizer_enforces_100_100_state_domain_and_split_invariants(tmp_path: Path) -> None:
    domains = (["general_negligence_personal_injury"] * 45 + ["medical_professional_liability"] * 25 + ["product_liability"] * 12 + ["other_civil_liability"] * 18)
    states = ["Oregon", "Texas", "Ohio", "Maine", "Nevada"]
    kr, us, facts = [], [], []
    for index, domain in enumerate(domains):
        kid, uid = f"KR_{index:03d}", f"US_{index:03d}"
        kr.append({"case_id": kid, "case_family_id": f"KF_{index}", "origin_country": "KR", "origin_state": None, "court_name": "대법원", "court_level": "supreme", "court_level_confidence": "high", "decision_date": "2020-01-01", "case_number": f"2020다{index}", "case_domain": domain, "strict_source_eligible": True, "raw_text_sha256": f"k{index}", "full_opinion_text": "raw", "main_opinion_text": "main"})
        state = states[index % 5]
        us.append({"case_id": uid, "case_family_id": f"UF_{index}", "origin_country": "US", "origin_state": state, "court_name": f"{state} high court", "court_level": "supreme", "court_level_confidence": "high", "court_type": "S", "decision_date": "2020-01-01", "citation": f"{index} X 1", "case_domain": domain, "strict_source_eligible": True, "raw_text_sha256": f"u{index}", "full_opinion_text": "raw", "main_opinion_text": "main"})
        facts.extend((_fact(kid, "KR", domain), _fact(uid, "US", domain)))
    write_jsonl(tmp_path / "kr.jsonl", kr); write_jsonl(tmp_path / "us.jsonl", us); write_jsonl(tmp_path / "facts.jsonl", facts)
    with (tmp_path / "qc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "final_eligible", "source_grounding_status", "legal_leakage_status", "jurisdiction_leakage_status", "translation_equivalence_status"]); writer.writeheader()
        for row in kr + us: writer.writerow({"case_id": row["case_id"], "final_eligible": "True", "source_grounding_status": "pass", "legal_leakage_status": "pass", "jurisdiction_leakage_status": "pass", "translation_equivalence_status": "pass"})
    write_json(tmp_path / "states.json", {"selected_states": [{"state": state} for state in states]})
    out = tmp_path / "out"
    code = finalize_main(["--kr-input", str(tmp_path / "kr.jsonl"), "--us-input", str(tmp_path / "us.jsonl"), "--facts-input", str(tmp_path / "facts.jsonl"), "--qc-input", str(tmp_path / "qc.csv"), "--states-from", str(tmp_path / "states.json"), "--output-dir", str(out)])
    assert code == 0
    summary = json.loads((out / "collection_summary.json").read_text(encoding="utf-8"))
    assert all(summary["sanity_checks"].values())
    assert summary["state_counts"] == {state: 20 for state in states}
    assert summary["primary_domain_counts"]["KR"] == summary["primary_domain_counts"]["US"]
