from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from finalize_case_sample_v2 import main as finalize_main, max_flow_counts
from pipeline_v2.io_utils import write_json, write_jsonl
from pipeline_v2.rules import (
    classify_domain, classify_kr_court, date_in_window, is_us_state_highcourt,
    leakage_checks, select_main_opinion, source_span_grounding, strip_legal_citations,
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


def test_placeholder_and_number_consistency_detects_missing_entity() -> None:
    ok = translation_equivalence_checks("[PERSON_A] drove 97 kilometers.", "[PERSON_A]은 97킬로미터를 운전했다.", "en")
    bad = translation_equivalence_checks("[PERSON_A] drove 97 kilometers.", "그 사람은 79킬로미터를 운전했다.", "en")
    assert ok["translation_equivalence_status"] == "pass"
    assert set(bad["translation_equivalence_issues"]) == {"placeholder_mismatch", "number_mismatch"}


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
    assert "number_mismatch" in translation_equivalence_checks("approximately 3.05 meters", "약 3.5미터", "en")["translation_equivalence_issues"]


def test_duplicate_family_capacity_cannot_fill_state_quota() -> None:
    states = ["A", "B", "C", "D", "E"]
    availability = {(state, "general_negligence_personal_injury"): (19 if state == "A" else 20) for state in states}
    flow, _ = max_flow_counts(states, availability, {"general_negligence_personal_injury": 100})
    assert flow == 99


def _fact(case_id: str, country: str, domain: str) -> dict:
    source_language = "ko" if country == "KR" else "en"
    source = "[PERSON_A]가 1개의 경고를 받았다." if country == "KR" else "[PERSON_A] received 1 warning."
    return {"case_id": case_id, "case_family_id": f"F_{case_id}", "origin_country": country, "origin_state": None, "case_domain": domain, "source_language": source_language, "neutral_fact_source": source, "neutral_fact_ko": "[PERSON_A]가 1개의 경고를 받았다.", "neutral_fact_en": "[PERSON_A] received 1 warning.", "aligned_fact_units": [{"fact_id": "F001", "source_text": source, "neutral_ko": "[PERSON_A]가 1개의 경고를 받았다.", "neutral_en": "[PERSON_A] received 1 warning.", "translation_status": "aligned", "translation_equivalence_status": "pass"}]}


def test_finalizer_enforces_100_100_state_domain_and_split_invariants(tmp_path: Path) -> None:
    domains = (["general_negligence_personal_injury"] * 40 + ["medical_professional_liability"] * 20 + ["product_liability"] * 20 + ["employer_supervisory_vicarious_liability"] * 20)
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
    assert summary["domain_counts"]["KR"] == summary["domain_counts"]["US"]

