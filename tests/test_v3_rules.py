from pipeline_v2.v3_rules import (
    bilingual_deterministic_qc, deterministic_domain_guard, evidence_spans_exist,
    obvious_source_exclusion, retained_text_changes_have_amendments,
    script_language_sanity, strict_leakage_checks,
)


def _pair(ko: str, en: str) -> dict:
    return {"origin_country": "KR", "source_language": "ko", "neutral_fact_ko": ko, "neutral_fact_en": en}


def test_korean_master_in_english_fails() -> None:
    assert script_language_sanity("This is an English master fact with enough words.", "ko")["status"] == "fail"


def test_placeholder_mismatch_fails() -> None:
    row = _pair("[PERSON_A]는 다쳤다.", "[PERSON_B] was injured in the incident.")
    assert bilingual_deterministic_qc(row)["placeholder_equivalence_status"] == "fail"


def test_duplicate_sentence_fails() -> None:
    row = _pair("[PERSON_A]는 현장에서 다쳤다. [PERSON_A]는 현장에서 다쳤다.", "[PERSON_A] was injured at the site. [PERSON_A] was injured at the site.")
    assert bilingual_deterministic_qc(row)["duplicate_sentence_status"] == "fail"


def test_lower_court_holding_is_flagged() -> None:
    assert strict_leakage_checks("원심은 피고에게 책임이 있다고 판단하였다.")["procedural_leakage_status"] == "fail"


def test_jury_or_court_conclusion_is_flagged() -> None:
    assert strict_leakage_checks("The jury found that the defendant breached a duty.")["procedural_leakage_status"] == "fail"


def test_legal_causation_is_flagged() -> None:
    assert strict_leakage_checks("Proximate cause was established.")["legal_leakage_status"] == "fail"


def test_customary_units_are_flagged() -> None:
    assert strict_leakage_checks("The vehicle traveled five miles at 30 mph.")["jurisdiction_leakage_status"] == "fail"


def test_currency_literals_are_flagged() -> None:
    assert strict_leakage_checks("The payment was 5,000 dollars.")["jurisdiction_leakage_status"] == "fail"
    assert strict_leakage_checks("지급액은 5,000원이었다.")["jurisdiction_leakage_status"] == "fail"


def test_korean_institution_cue_is_flagged() -> None:
    assert strict_leakage_checks("국토교통부장관이 문서를 발급했다.")["jurisdiction_leakage_status"] == "fail"


def test_jurisdiction_placeholder_is_flagged() -> None:
    assert strict_leakage_checks("[BOROUGH_A] maintained the road.")["jurisdiction_leakage_status"] == "fail"


def test_attorney_disciplinary_exclusion() -> None:
    assert obvious_source_exclusion({"main_opinion_text": "ATTORNEY DISCIPLINARY PROCEEDINGS"}) == "attorney_disciplinary"


def test_workers_compensation_exclusion() -> None:
    assert obvious_source_exclusion({"main_opinion_text": "This workers' compensation appeal concerns benefits."}) == "workers_compensation"


def test_insurance_coverage_exclusion() -> None:
    assert obvious_source_exclusion({"main_opinion_text": "A declaratory judgment action concerns insurance coverage."}) == "insurance_coverage_only"


def test_patent_does_not_imply_product_liability() -> None:
    assert deterministic_domain_guard("product_liability", "A patent royalty dispute over technology licensing.") is not None


def test_treatment_alone_does_not_imply_medical_malpractice() -> None:
    assert deterministic_domain_guard("medical_professional_liability", "A doctor described treatment and medical records as an expert witness.") is not None


def test_source_span_must_exist() -> None:
    assert evidence_spans_exist("The driver stopped.", ["The driver stopped."])
    assert not evidence_spans_exist("The driver stopped.", ["The driver accelerated."])


def test_human_foot_is_not_a_customary_unit() -> None:
    assert strict_leakage_checks("[PERSON_A] stood with one foot on the step.")["jurisdiction_leakage_status"] == "pass"


def test_retained_text_change_requires_amendment_record() -> None:
    original = {"CASE_A": {"neutral_fact_ko": "원문", "neutral_fact_en": "Original."}}
    final = {"CASE_A": {"neutral_fact_ko": "수정", "neutral_fact_en": "Original."}}
    assert not retained_text_changes_have_amendments(original, final, [])
    assert retained_text_changes_have_amendments(original, final, [{"case_id": "CASE_A", "field_changed": "neutral_fact_ko"}])
