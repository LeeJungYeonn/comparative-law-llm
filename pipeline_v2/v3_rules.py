from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .io_utils import normalized_whitespace
from .rules import PLACEHOLDER_RE, leakage_checks, numeric_concepts, source_span_grounding

DOMAINS = (
    "general_negligence_personal_injury",
    "medical_professional_liability",
    "product_liability",
    "other_civil_liability",
)

PROCEDURAL_LEAK = re.compile(
    r"원심(?:은|이).{0,80}판단하|대법원(?:은|이).{0,80}판단하|배심(?:원|은|이).{0,80}(?:판단|평결)|"
    r"\b(?:the\s+)?(?:court|lower court|jury)\s+(?:held|found|concluded|determined)|"
    r"\b(?:affirmed|reversed|remanded|summary judgment was appropriate)\b",
    re.I,
)
LEGAL_CAUSATION_LEAK = re.compile(
    r"상당인과관계(?:가|는)?\s*인정|책임(?:이|을)\s*인정|주의의무(?:가|를)\s*(?:있|부담)|"
    r"\bproximate cause was (?:established|proven)|\bforeseeable as a matter of law|"
    r"\b(?:duty|breach|liability) (?:existed|was established|was proven)",
    re.I,
)
CUSTOMARY_UNIT_LEAK = re.compile(
    r"\b(?:miles?|feet|pounds?|mph)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:foot|ft)\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+foot\s+(?:high|long|wide|deep|away|from)\b|"
    r"\bfoot-(?:high|long|wide|deep)\b",
    re.I,
)
CURRENCY_LEAK = re.compile(r"(?:\$|₩|\b(?:won|dollars?)\b|\d[\d,]*(?:\s*)원(?:\b|을|의|이|은|는|만|씩))", re.I)
KR_INSTITUTION_LEAK = re.compile(r"주민등록증|서울특별시장|도지사|국토교통부장관")
JURISDICTION_PLACEHOLDER_LEAK = re.compile(r"\[(?:BOROUGH|COUNTY|STATE_AGENCY)_[A-Z]+\]")

ATTORNEY_DISCIPLINE = re.compile(r"ATTORNEY DISCIPLINARY|disciplinary (?:matter|proceeding)|Office of Disciplinary Counsel", re.I)
WORKERS_COMP = re.compile(r"workers['’]? compensation|workmen['’]?s compensation|compensation appeal board", re.I)
INSURANCE_COVERAGE_ONLY = re.compile(
    r"declaratory (?:judgment|action).{0,100}(?:coverage|insurance)|insurance coverage|duty to (?:defend|indemnify)|"
    r"pollution exclusion|coverage exclusion|insurance-contract indemnity",
    re.I | re.S,
)
PATENT_ROYALTY = re.compile(r"\b(?:patent|royalt(?:y|ies)|technology licens(?:e|ing))\b", re.I)
PRODUCT_DEFECT = re.compile(r"\b(?:design defect|manufacturing defect|failure to warn|defective product|strict products? liability)\b", re.I)
MEDICAL_TREATMENT = re.compile(r"의료과실|의료사고|진료상 과실|malpractice|failure to diagnose|negligent professional service", re.I)
MEDICAL_INCIDENTAL = re.compile(r"\b(?:doctor|physician|medical records?|treatment|expert witness)\b|의사|진료기록|치료|감정인", re.I)


def script_language_sanity(text: str, expected: str) -> dict[str, Any]:
    hangul = len(re.findall(r"[가-힣]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    letters = hangul + latin
    if not (text or "").strip() or letters == 0:
        return {"status": "fail", "hangul": hangul, "latin": latin, "reason": "empty_or_no_letters"}
    if expected == "ko":
        passed = hangul >= 10 and hangul / letters >= 0.35
    elif expected == "en":
        passed = latin >= 20 and latin / letters >= 0.70
    else:
        raise ValueError(f"Unsupported language: {expected}")
    return {"status": "pass" if passed else "fail", "hangul": hangul, "latin": latin, "reason": None if passed else f"not_{expected}"}


def placeholder_equivalence(ko: str, en: str) -> dict[str, Any]:
    ko_set, en_set = set(PLACEHOLDER_RE.findall(ko or "")), set(PLACEHOLDER_RE.findall(en or ""))
    return {"status": "pass" if ko_set == en_set else "fail", "ko": sorted(ko_set), "en": sorted(en_set)}


def _sentences(text: str) -> list[str]:
    return [
        normalized_whitespace(part).casefold()
        for part in re.split(r"(?<=[.!?])\s+(?!\d)|\n+", text or "")
        if len(normalized_whitespace(part)) >= 15
    ]


def duplicate_sentences(text: str) -> dict[str, Any]:
    counts = Counter(_sentences(text))
    duplicates = sorted(sentence for sentence, count in counts.items() if count > 1)
    return {"status": "pass" if not duplicates else "fail", "duplicates": duplicates}


def strict_leakage_checks(text: str) -> dict[str, Any]:
    base = leakage_checks(text or "")
    groups = {
        "procedural": [m.group(0) for m in PROCEDURAL_LEAK.finditer(text or "")],
        "legal_conclusion": [m.group(0) for m in LEGAL_CAUSATION_LEAK.finditer(text or "")],
        "customary_unit": [m.group(0) for m in CUSTOMARY_UNIT_LEAK.finditer(text or "")],
        "currency": [m.group(0) for m in CURRENCY_LEAK.finditer(text or "")],
        "kr_institution": [m.group(0) for m in KR_INSTITUTION_LEAK.finditer(text or "")],
        "jurisdiction_placeholder": [m.group(0) for m in JURISDICTION_PLACEHOLDER_LEAK.finditer(text or "")],
    }
    return {
        **base,
        "procedural_leakage_status": "fail" if groups["procedural"] else "pass",
        "legal_leakage_status": "fail" if base["legal_leakage_status"] == "fail" or groups["legal_conclusion"] else "pass",
        "jurisdiction_leakage_status": "fail" if base["jurisdiction_leakage_status"] == "fail" or any(groups[key] for key in ("customary_unit", "currency", "kr_institution", "jurisdiction_placeholder")) else "pass",
        "strict_leakage_evidence": groups,
    }


def bilingual_deterministic_qc(record: dict[str, Any]) -> dict[str, Any]:
    ko, en = record.get("neutral_fact_ko") or "", record.get("neutral_fact_en") or ""
    ko_leak, en_leak = strict_leakage_checks(ko), strict_leakage_checks(en)
    placeholders = placeholder_equivalence(ko, en)
    ko_dup, en_dup = duplicate_sentences(ko), duplicate_sentences(en)
    source_language = record.get("source_language") or ("ko" if record.get("origin_country") == "KR" else "en")
    source_master = ko if source_language == "ko" else en
    target = en if source_language == "ko" else ko
    source_numbers, target_numbers = numeric_concepts(source_master), numeric_concepts(target)
    return {
        "language_sanity_status": "pass" if script_language_sanity(ko, "ko")["status"] == script_language_sanity(en, "en")["status"] == "pass" else "fail",
        "language_sanity_detail": {"ko": script_language_sanity(ko, "ko"), "en": script_language_sanity(en, "en")},
        "placeholder_equivalence_status": placeholders["status"], "placeholder_detail": placeholders,
        "duplicate_sentence_status": "pass" if ko_dup["status"] == en_dup["status"] == "pass" else "fail",
        "duplicate_sentence_detail": {"ko": ko_dup, "en": en_dup},
        "legal_leakage_status": "pass" if ko_leak["legal_leakage_status"] == en_leak["legal_leakage_status"] == "pass" else "fail",
        "procedural_leakage_status": "pass" if ko_leak["procedural_leakage_status"] == en_leak["procedural_leakage_status"] == "pass" else "fail",
        "jurisdiction_leakage_status": "pass" if ko_leak["jurisdiction_leakage_status"] == en_leak["jurisdiction_leakage_status"] == "pass" else "fail",
        "leakage_detail": {"ko": ko_leak, "en": en_leak},
        "numerical_unit_status": "pass" if source_numbers == target_numbers else "fail",
        "number_detail": {"source": dict(source_numbers), "target": dict(target_numbers)},
    }


def retained_text_changes_have_amendments(
    authoritative: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    amendments: list[dict[str, Any]],
) -> bool:
    """Require an explicit amendment row for every changed retained language field."""
    logged = {(row.get("case_id"), row.get("field_changed")) for row in amendments}
    for case_id, original in authoritative.items():
        current = final.get(case_id)
        if current is None:
            continue
        for field in ("neutral_fact_ko", "neutral_fact_en"):
            if original.get(field) != current.get(field) and (case_id, field) not in logged:
                return False
    return True


def deterministic_source_signals(case: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(case.get(key) or "") for key in ("case_name", "nature_of_suit", "main_opinion_text"))
    return {
        "attorney_discipline": bool(ATTORNEY_DISCIPLINE.search(text)),
        "workers_compensation": bool(WORKERS_COMP.search(text)),
        "insurance_coverage": bool(INSURANCE_COVERAGE_ONLY.search(text)),
        "patent_or_royalty": bool(PATENT_ROYALTY.search(text)),
        "product_defect": bool(PRODUCT_DEFECT.search(text)),
        "medical_malpractice": bool(MEDICAL_TREATMENT.search(text)),
        "medical_incidental": bool(MEDICAL_INCIDENTAL.search(text)),
    }


def evidence_spans_exist(source: str, spans: list[str]) -> bool:
    return bool(spans) and all(source_span_grounding(source, span)[0] == "pass" for span in spans)


def obvious_source_exclusion(case: dict[str, Any]) -> str | None:
    signals = deterministic_source_signals(case)
    if signals["attorney_discipline"]:
        return "attorney_disciplinary"
    if signals["workers_compensation"]:
        return "workers_compensation"
    if signals["insurance_coverage"]:
        return "insurance_coverage_only"
    return None


def deterministic_domain_guard(label: str, source: str) -> str | None:
    if label == "product_liability" and PATENT_ROYALTY.search(source) and not PRODUCT_DEFECT.search(source):
        return "patent_or_royalty_without_product_defect"
    if label == "medical_professional_liability" and MEDICAL_INCIDENTAL.search(source) and not MEDICAL_TREATMENT.search(source):
        return "incidental_medical_reference_without_malpractice"
    return None
