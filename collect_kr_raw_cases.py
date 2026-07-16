from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset

from pipeline.stage1_raw import (
    compact,
    normalized_text_for_hash,
    parse_year,
    require_outputs,
    sha256_text,
    short_hash,
    unique,
    write_summary,
)
from pipeline.text_utils import extract_fact_section_with_metadata, normalize_whitespace


LOGGER = logging.getLogger(__name__)

COLLECTION_VERSION = "stage1-kr-tort-appellate-v3"
DEFAULT_OUTPUT_DIR = Path("outputs/raw/kr_v3")
DEFAULT_MANIFEST_OUTPUT = Path("outputs/manifests/kr_v3_case_manifest.csv")

TEXT_COLUMNS = ("precedent", "raw_text", "text", "facts", "reason", "ruling")
CASE_NUMBER_COLUMNS = ("case_number_or_citation", "case_number", "case_no", "caseno", "사건번호")
COURT_COLUMNS = ("court_name", "court", "court_full_name", "법원명")
DATE_COLUMNS = ("decision_date", "date", "선고일자", "판결선고일")
CASE_NAME_COLUMNS = ("case_name", "casename", "사건명")
CASE_TYPE_COLUMNS = ("case_type", "case_type_keyword", "사건종류", "사건명")

BROAD_KEYWORD_PATTERNS = [
    r"손해배상",
    r"불법행위",
    r"민법\s*제?\s*750\s*조",
    r"민법\s*제?\s*751\s*조",
    r"민법\s*제?\s*756\s*조",
    r"민법\s*제?\s*758\s*조",
    r"과실",
    r"주의의무",
    r"안전배려의무",
    r"사용자책임",
    r"공작물책임",
    r"제조물책임",
    r"의료과오",
    r"의료사고",
    r"교통사고",
    r"추돌",
    r"충돌",
    r"상해",
    r"사망",
    r"후유장해",
    r"치료비",
    r"일실수입",
    r"위자료",
    r"재산상\s*손해",
    r"정신적\s*손해",
    r"명예훼손",
    r"개인정보\s*침해",
    r"구상금",
    r"보험자대위",
    r"국가배상",
]

CASE_NUMBER_RE = re.compile(r"\b((?:18|19|20)\d{2})\s*([가-힣]{1,5})\s*(\d{1,7})\b")
HEADER_CASE_NUMBER_RE = re.compile(
    r"(?:사건번호|사\s*건|선고|판결|결정|법원)\s*[:：]?\s*((?:18|19|20)\d{2}\s*[가-힣]{1,5}\s*\d{1,7})"
)
DATE_RE = re.compile(r"\b((?:18|19|20)\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?")
COURT_RE = re.compile(
    r"([가-힣]{0,12}(?:대법원|고등법원|특허법원|지방법원|가정법원|행정법원)(?:\s*[가-힣]+지원)?(?:\s*(?:민사)?항소부)?)"
)

TRIAL_CASE_CODES = ("가합", "가단", "가소")
APPELLATE_CASE_CODES = ("나",)
SUPREME_CASE_CODES = ("다",)

APPELLATE_ROLE_PATTERNS = [r"항소인", r"피항소인", r"항소취지", r"제1심판결", r"원고의\s*항소", r"피고의\s*항소", r"제1심\s*법원"]
SUPREME_ROLE_PATTERNS = [r"상고인", r"피상고인", r"상고이유", r"상고를\s*기각", r"파기환송", r"파기\s*환송"]
CRIMINAL_PATTERNS = [r"피고인", r"징역", r"집행유예", r"공소사실", r"범죄사실", r"\d{4}\s*고(?:단|합|정)\s*\d+"]
FAMILY_PATTERNS = [r"가정법원", r"이혼", r"친권", r"양육", r"재산분할", r"\d{4}\s*(?:드단|드합|르|므|브|푸)\s*\d+"]
IP_PATTERNS = [r"특허", r"상표", r"디자인권", r"저작권"]

CONTRACT_PATTERNS = [
    r"계약\s*위반",
    r"채무불이행",
    r"하자담보",
    r"매매계약",
    r"임대차",
    r"도급계약",
    r"공사대금",
    r"대금",
    r"계약해제",
    r"계약상\s*의무",
]
INSURANCE_PATTERNS = [r"보험금", r"보험계약", r"보험약관", r"면책", r"보험자"]
PROCEDURAL_PATTERNS = [
    r"\d{4}\s*(?:카단|카합|카기|카확|머|차)\s*\d+",
    r"소송비용",
    r"관할",
    r"이송",
    r"각하",
    r"항소기간",
    r"재심",
    r"소멸시효",
    r"제척기간",
    r"강제집행",
    r"배당",
]
ADMIN_PATTERNS = [r"국가배상", r"행정처분", r"처분취소", r"거부처분", r"영업정지"]
TORT_STRONG_PATTERNS = [
    r"불법행위",
    r"민법\s*제?\s*750\s*조",
    r"민법\s*제?\s*751\s*조",
    r"민법\s*제?\s*756\s*조",
    r"민법\s*제?\s*758\s*조",
    r"사용자책임",
    r"공작물책임",
    r"제조물책임",
    r"의료과오",
    r"의료사고",
    r"교통사고",
    r"추돌",
    r"충돌",
    r"상해",
    r"사망",
    r"후유장해",
    r"치료비",
    r"일실수입",
    r"위자료",
    r"명예훼손",
    r"개인정보\s*침해",
]
ACTION_PATTERNS = [r"사고", r"충돌", r"추돌", r"추락", r"수술", r"진료", r"투약", r"운전", r"폭행", r"게시", r"누출", r"설치", r"관리", r"방치", r"위반", r"침해"]
DAMAGE_PATTERNS = [r"상해", r"부상", r"사망", r"치료비", r"후유장해", r"손해", r"위자료", r"일실수입", r"재산", r"정신적", r"명예", r"개인정보"]
FACT_CONTEXT_PATTERNS = [r"원고", r"피고", r"발생", r"당시", r"그 후", r"인하여", r"때문", r"결과", r"관계", r"경위"]
FACT_START_PATTERNS = [r"기초\s*사실", r"인정\s*사실", r"사실\s*관계", r"전제\s*사실", r"다툼\s*없는\s*사실", r"인정되는\s*사실", r"사건의\s*경위"]

SUBTYPE_PATTERNS = [
    ("traffic_accident", [r"교통사고", r"자동차", r"차량", r"운전", r"추돌", r"충돌", r"횡단보도"]),
    ("medical_professional", [r"의료", r"병원", r"의사", r"수술", r"진료", r"투약", r"간호", r"전문가"]),
    ("premises_facility_safety", [r"시설", r"공작물", r"건물", r"계단", r"추락", r"넘어", r"안전관리", r"하자"]),
    ("product_safety", [r"제품", r"제조물", r"제조", r"결함", r"식품", r"기계"]),
    ("employer_vicarious_liability", [r"사용자책임", r"피용자", r"근로자", r"직원", r"업무집행"]),
    ("privacy_reputation", [r"명예훼손", r"모욕", r"사생활", r"개인정보", r"초상권"]),
    ("wrongful_death", [r"사망", r"유족", r"망인", r"장례"]),
    ("property_damage", [r"재산상\s*손해", r"건물", r"차량\s*파손", r"영업손실", r"화재", r"침수"]),
    ("general_personal_injury", [r"상해", r"부상", r"치료비", r"후유장해", r"폭행"]),
]

SAMPLE_QUOTAS = {
    "traffic_general_personal_injury": 5,
    "medical_professional": 4,
    "premises_facility_safety": 4,
    "product_property_damage": 3,
    "employer_other_negligence": 4,
}

QC_FIELDS = [
    "case_id",
    "case_number",
    "court_name",
    "decision_date",
    "decision_year",
    "case_type",
    "court_level",
    "court_level_confidence",
    "civil_case_likely",
    "liability_basis",
    "tort_confidence",
    "case_subtype",
    "factual_background_sufficient",
    "strict_eligible",
    "exclusion_reasons",
    "court_level_evidence",
    "tort_evidence",
    "factual_sufficiency_reasons",
    "duplicate_reason",
    "duplicate_of_case_id",
    "raw_text_sha256",
]

MANIFEST_FIELDS = [
    "case_id",
    "collection_version",
    "source_dataset",
    "source_record_id",
    "case_number",
    "court_name",
    "decision_date",
    "decision_year",
    "court_level",
    "court_level_confidence",
    "liability_basis",
    "case_subtype",
    "strict_eligible",
    "selected",
    "duplicate_reason",
    "duplicate_of_case_id",
    "related_case_group_id",
    "raw_text_sha256",
    "raw_length_chars",
    "raw_path",
]


def regex_hits(patterns: Iterable[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def first_existing(row: dict[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        if column in row and compact(row.get(column, "")):
            return compact(row.get(column, ""))
    return ""


def extract_case_number(text: str) -> str:
    header = text[:1500]
    header_match = HEADER_CASE_NUMBER_RE.search(header)
    if header_match:
        return compact(header_match.group(1))
    first_header_case = CASE_NUMBER_RE.search(header)
    if first_header_case:
        return compact(first_header_case.group(0))
    first_case = CASE_NUMBER_RE.search(text)
    return compact(first_case.group(0)) if first_case else ""


def extract_case_code(case_number: str) -> str:
    match = CASE_NUMBER_RE.search(case_number)
    return match.group(2) if match else ""


def extract_decision_date(text: str) -> str:
    match = DATE_RE.search(text[:2000])
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def infer_court(text: str) -> str:
    match = COURT_RE.search(text[:1500])
    return compact(match.group(1)) if match else ""


def classify_court_level(court_name: str, case_number: str, text: str) -> tuple[str, str, list[str]]:
    preview = f"{court_name}\n{case_number}\n{text[:2500]}"
    evidence: list[str] = []
    code = extract_case_code(case_number)
    supreme_score = appellate_score = trial_score = 0

    if code in SUPREME_CASE_CODES:
        supreme_score += 3
        evidence.append(f"case_number_pattern: {case_number}")
    elif code in APPELLATE_CASE_CODES:
        appellate_score += 3
        evidence.append(f"case_number_pattern: {case_number}")
    elif code in TRIAL_CASE_CODES:
        trial_score += 3
        evidence.append(f"case_number_pattern: {case_number}")

    if "대법원" in court_name:
        supreme_score += 3
        evidence.append(f"court_name: {court_name}")
    if "특허법원" in court_name:
        evidence.append(f"court_name_excluded: {court_name}")
    elif "고등법원" in court_name or re.search(r"지방법원.*(?:민사)?항소부", court_name):
        appellate_score += 3
        evidence.append(f"court_name: {court_name}")
    elif "지방법원" in court_name:
        trial_score += 1
        evidence.append(f"court_name: {court_name}")

    supreme_roles = regex_hits(SUPREME_ROLE_PATTERNS, preview)
    appellate_roles = regex_hits(APPELLATE_ROLE_PATTERNS, preview)
    if supreme_roles:
        supreme_score += min(2, len(supreme_roles))
        evidence.extend(f"supreme_role: {hit}" for hit in supreme_roles[:3])
    if appellate_roles:
        appellate_score += min(2, len(appellate_roles))
        evidence.extend(f"appellate_role: {hit}" for hit in appellate_roles[:3])

    scores = {"supreme": supreme_score, "appellate": appellate_score, "trial": trial_score}
    top_level, top_score = max(scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(scores.values(), reverse=True)

    if top_score == 0:
        return "unknown", "low", evidence or ["no_court_level_signal"]
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        evidence.append("conflicting_court_level_signals")
        return "unknown", "low", evidence
    if top_level == "appellate" and code in APPELLATE_CASE_CODES and ("고등법원" in court_name or re.search(r"지방법원.*(?:민사)?항소부", court_name)):
        return "appellate", "high", evidence
    if top_level == "supreme" and (code in SUPREME_CASE_CODES or "대법원" in court_name):
        return "supreme", "high", evidence
    if top_score >= 4:
        return top_level, "high", evidence
    if top_score >= 2:
        return top_level, "medium", evidence
    return top_level, "low", evidence


def classify_civil_case(haystack: str) -> tuple[bool, list[str]]:
    reasons = []
    if regex_hits(CRIMINAL_PATTERNS, haystack):
        reasons.append("criminal_case")
    if regex_hits(FAMILY_PATTERNS, haystack):
        reasons.append("family_case")
    if regex_hits(IP_PATTERNS, haystack) and not regex_hits(TORT_STRONG_PATTERNS, haystack):
        reasons.append("ip_only")
    return not reasons, reasons


def classify_liability_basis(haystack: str) -> tuple[str, str, list[str]]:
    tort_hits = regex_hits(TORT_STRONG_PATTERNS, haystack)
    contract_hits = regex_hits(CONTRACT_PATTERNS, haystack)
    insurance_hits = regex_hits(INSURANCE_PATTERNS, haystack)
    procedural_hits = regex_hits(PROCEDURAL_PATTERNS, haystack)
    admin_hits = regex_hits(ADMIN_PATTERNS, haystack)
    action_hits = regex_hits(ACTION_PATTERNS, haystack)
    damage_hits = regex_hits(DAMAGE_PATTERNS, haystack)
    evidence = [f"tort_signal: {hit}" for hit in tort_hits[:8]]

    procedural_case_number_hits = regex_hits([r"\d{4}\s*(?:카단|카합|카기|카확|머|차)\s*\d+"], haystack)
    if procedural_case_number_hits:
        return "procedural_only", "high", [f"procedural_case_number: {hit}" for hit in procedural_case_number_hits]
    if procedural_hits and not (tort_hits and action_hits and damage_hits):
        return "procedural_only", "high", [f"procedural_signal: {hit}" for hit in procedural_hits[:5]]
    if insurance_hits and not (tort_hits and action_hits and damage_hits):
        return "insurance_only", "high", [f"insurance_signal: {hit}" for hit in insurance_hits[:5]]
    if admin_hits:
        return "administrative_or_state_liability", "medium", [f"admin_signal: {hit}" for hit in admin_hits[:5]] + evidence
    if tort_hits and action_hits and damage_hits:
        if contract_hits:
            return "mixed_tort_contract", "medium", evidence + [f"contract_signal: {hit}" for hit in contract_hits[:5]]
        return "non_contractual_tort", "high", evidence + [f"action_signal: {hit}" for hit in action_hits[:5]] + [f"damage_signal: {hit}" for hit in damage_hits[:5]]
    if contract_hits and not tort_hits:
        return "contract_only", "high", [f"contract_signal: {hit}" for hit in contract_hits[:5]]
    if insurance_hits:
        return "insurance_only", "medium", [f"insurance_signal: {hit}" for hit in insurance_hits[:5]]
    if tort_hits:
        return "unclear", "low", evidence
    return "unclear", "low", ["no_substantive_tort_signal"]


def classify_subtype(haystack: str) -> str:
    for subtype, patterns in SUBTYPE_PATTERNS:
        if regex_hits(patterns, haystack):
            return subtype
    if regex_hits(TORT_STRONG_PATTERNS, haystack):
        return "other_tort"
    return "unclear"


def assess_factual_sufficiency(raw_text: str, haystack: str) -> tuple[bool, list[str]]:
    fact_text, metadata = extract_fact_section_with_metadata(raw_text, "KR")
    fact_probe = fact_text if len(fact_text) >= 120 else haystack[:6000]
    reasons: list[str] = []

    if metadata.get("has_fact_heading"):
        reasons.append("fact_heading_detected")
    elif regex_hits(FACT_START_PATTERNS, haystack):
        reasons.append("inline_fact_heading_detected")
    else:
        reasons.append("no_explicit_fact_heading")

    categories = {
        "party_or_context": regex_hits(FACT_CONTEXT_PATTERNS, fact_probe),
        "specific_conduct": regex_hits(ACTION_PATTERNS, fact_probe),
        "damage": regex_hits(DAMAGE_PATTERNS, fact_probe),
        "time_sequence": regex_hits([r"\d{4}\.\s*\d{1,2}\.\s*\d{1,2}", r"당시", r"그 후", r"이후", r"전후"], fact_probe),
    }
    present = [name for name, hits in categories.items() if hits]
    reasons.extend(f"{name}: {hits[0]}" for name, hits in categories.items() if hits)

    legal_only = bool(regex_hits([r"관련\s*법리", r"대법원.*판시", r"법리는.*같다"], fact_probe)) and len(present) < 3
    if legal_only:
        reasons.append("mostly_legal_principles")
    min_fact_chars = 80 if metadata.get("has_fact_heading") and len(present) >= 3 else 180
    if len(fact_probe) < min_fact_chars:
        reasons.append("fact_section_too_short")
    if len(present) < 3:
        reasons.append("insufficient_fact_categories")

    return len(fact_probe) >= min_fact_chars and len(present) >= 3 and not legal_only, unique(reasons)


def stable_case_id(source_dataset: str, source_record_id: str, case_number: str, decision_date: str, raw_text: str) -> str:
    stable_source = compact(source_record_id) or compact(case_number)
    digest = short_hash(source_dataset, stable_source, decision_date, raw_text[:1000], length=16)
    return f"KR_{digest}"


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, object] | None:
    raw_text = first_existing(row, [args.text_col] if getattr(args, "text_col", "") else TEXT_COLUMNS)
    if not raw_text:
        return None
    raw_text = normalize_whitespace(raw_text)
    source_dataset = f"{getattr(args, 'dataset', 'lbox/lbox_open')}::{getattr(args, 'config', 'precedent_corpus')}"
    source_record_id = compact(row.get("id", ""))
    case_number = first_existing(row, CASE_NUMBER_COLUMNS) or extract_case_number(raw_text)
    court_name = first_existing(row, COURT_COLUMNS) or infer_court(raw_text)
    decision_date = first_existing(row, DATE_COLUMNS) or extract_decision_date(raw_text)
    decision_year = parse_year(decision_date)
    case_name = first_existing(row, CASE_NAME_COLUMNS)
    case_type = first_existing(row, CASE_TYPE_COLUMNS) or case_name
    haystack = f"{case_name}\n{case_type}\n{case_number}\n{court_name}\n{raw_text[:12000]}"
    keyword_hits = regex_hits(BROAD_KEYWORD_PATTERNS, haystack)
    civil_case_likely, civil_exclusions = classify_civil_case(haystack)
    court_level, court_confidence, court_evidence = classify_court_level(court_name, case_number, raw_text)
    liability_basis, tort_confidence, tort_evidence = classify_liability_basis(haystack)
    subtype = classify_subtype(haystack)
    factual_sufficient, factual_reasons = assess_factual_sufficiency(raw_text, haystack)

    exclusion_reasons: list[str] = []
    exclusion_reasons.extend(civil_exclusions)
    if not keyword_hits:
        exclusion_reasons.append("no_broad_keyword_signal")
    if court_level == "supreme":
        exclusion_reasons.append("supreme_court_excluded")
    if court_level != getattr(args, "court_level", "appellate"):
        exclusion_reasons.append("non_appellate_or_unknown_court_level")
    if decision_year is None:
        exclusion_reasons.append("decision_year_unknown")
    elif decision_year < getattr(args, "year_min", 2010) or decision_year > getattr(args, "year_max", 2021):
        exclusion_reasons.append("decision_year_out_of_range")
    if liability_basis != "non_contractual_tort" and getattr(args, "strict_tort_only", False):
        exclusion_reasons.append(f"not_strict_non_contractual_tort:{liability_basis}")
    if not factual_sufficient:
        exclusion_reasons.append("factual_background_insufficient")
    if len(raw_text) < getattr(args, "min_text_chars", 0):
        exclusion_reasons.append("too_short_or_no_full_opinion_text")
    max_text_chars = getattr(args, "max_text_chars", 0)
    if max_text_chars and len(raw_text) > max_text_chars:
        exclusion_reasons.append("too_long")

    strict_eligible = (
        bool(keyword_hits)
        and civil_case_likely
        and court_level == getattr(args, "court_level", "appellate")
        and liability_basis == "non_contractual_tort"
        and factual_sufficient
        and isinstance(decision_year, int)
        and getattr(args, "year_min", 2010) <= decision_year <= getattr(args, "year_max", 2021)
        and len(raw_text) >= getattr(args, "min_text_chars", 0)
        and not (max_text_chars and len(raw_text) > max_text_chars)
    )

    collection_status = "pass" if strict_eligible else "fail"
    return {
        "case_id": stable_case_id(source_dataset, source_record_id, case_number, decision_date, raw_text),
        "collection_version": COLLECTION_VERSION,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "raw_text": raw_text,
        "case_number": case_number,
        "case_number_or_citation": case_number,
        "case_name": case_name,
        "court_name": court_name or "unknown",
        "decision_date": decision_date,
        "decision_year": decision_year,
        "case_type": case_type,
        "court_level": court_level,
        "court_level_confidence": court_confidence,
        "court_level_evidence": court_evidence,
        "civil_case_likely": civil_case_likely,
        "liability_basis": liability_basis,
        "tort_confidence": tort_confidence,
        "tort_evidence": tort_evidence,
        "case_subtype": subtype,
        "factual_background_sufficient": factual_sufficient,
        "factual_sufficiency_reasons": factual_reasons,
        "strict_eligible": strict_eligible,
        "exclusion_reasons": unique(exclusion_reasons),
        "include_signals": keyword_hits,
        "exclude_signals": unique(exclusion_reasons),
        "quality_flags": [],
        "collection_status": collection_status,
        "raw_text_sha256": sha256_text(raw_text),
        "normalized_text_sha256": sha256_text(normalized_text_for_hash(raw_text)),
        "raw_length_chars": len(raw_text),
        "related_case_group_id": "",
        "duplicate_reason": "",
        "duplicate_of_case_id": "",
        "selected": False,
    }


def keyword_gate(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    raw_text = first_existing(row, [args.text_col] if getattr(args, "text_col", "") else TEXT_COLUMNS)
    case_number = first_existing(row, CASE_NUMBER_COLUMNS)
    court_name = first_existing(row, COURT_COLUMNS)
    case_name = first_existing(row, CASE_NAME_COLUMNS)
    haystack = f"{case_name}\n{case_number}\n{court_name}\n{raw_text[:12000]}"
    hits = regex_hits(BROAD_KEYWORD_PATTERNS, haystack)
    return bool(hits), hits


def mark_duplicate_candidates(records: list[dict[str, object]]) -> dict[str, int]:
    counters: Counter[str] = Counter()
    seen_exact: dict[str, str] = {}
    seen_norm: dict[str, str] = {}
    seen_case_number: dict[str, str] = {}
    for record in records:
        duplicate_of = ""
        duplicate_reason = ""
        exact = str(record.get("raw_text_sha256") or "")
        norm = str(record.get("normalized_text_sha256") or "")
        case_number = compact(record.get("case_number", "")).lower()
        if exact and exact in seen_exact:
            duplicate_of = seen_exact[exact]
            duplicate_reason = "duplicate_exact_hash"
        elif norm and norm in seen_norm:
            duplicate_of = seen_norm[norm]
            duplicate_reason = "duplicate_normalized_text_hash"
        elif case_number and case_number in seen_case_number:
            duplicate_of = seen_case_number[case_number]
            duplicate_reason = "duplicate_case_number"
        seen_exact.setdefault(exact, str(record["case_id"]))
        seen_norm.setdefault(norm, str(record["case_id"]))
        if case_number:
            seen_case_number.setdefault(case_number, str(record["case_id"]))
        if duplicate_of:
            group_id = f"grp_{short_hash(duplicate_of, record['case_id'], length=12)}"
            record["duplicate_of_case_id"] = duplicate_of
            record["duplicate_reason"] = duplicate_reason
            record["related_case_group_id"] = group_id
            counters[duplicate_reason] += 1
    return dict(counters)


def split_pools(records: list[dict[str, object]], args: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    return {
        "keyword_hits": records,
        "civil_candidates": [row for row in records if row.get("civil_case_likely") is True],
        "appellate_candidates": [row for row in records if row.get("civil_case_likely") is True and row.get("court_level") == args.court_level],
        "tort_candidates": [
            row
            for row in records
            if row.get("civil_case_likely") is True
            and row.get("court_level") == args.court_level
            and row.get("liability_basis") in {"non_contractual_tort", "mixed_tort_contract", "administrative_or_state_liability"}
        ],
        "strict_eligible": [row for row in records if row.get("strict_eligible") is True],
    }


def sample_quota_group(subtype: object) -> str:
    value = compact(subtype)
    if value in {"traffic_accident", "general_personal_injury", "wrongful_death"}:
        return "traffic_general_personal_injury"
    if value == "medical_professional":
        return "medical_professional"
    if value == "premises_facility_safety":
        return "premises_facility_safety"
    if value in {"product_safety", "property_damage", "privacy_reputation"}:
        return "product_property_damage"
    return "employer_other_negligence"


def candidate_sort_key(record: dict[str, object]) -> tuple[int, int, str]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(str(record.get("court_level_confidence")), 3)
    year = record.get("decision_year")
    year_rank = -int(year) if isinstance(year, int) else 0
    return (year_rank, confidence_rank, str(record.get("case_id", "")))


def select_final_sample(records: list[dict[str, object]], args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    year_min = args.year_min if not args.allow_year_fallback else min(args.year_min, 2000)
    preferred = [row for row in records if isinstance(row.get("decision_year"), int) and args.year_min <= int(row["decision_year"]) <= args.year_max]
    fallback = [row for row in records if isinstance(row.get("decision_year"), int) and year_min <= int(row["decision_year"]) < args.year_min]
    pool = preferred + fallback if args.allow_year_fallback else preferred
    pool = [row for row in pool if not row.get("duplicate_of_case_id")]

    rng = random.Random(args.seed)
    by_group: dict[str, list[dict[str, object]]] = {group: [] for group in SAMPLE_QUOTAS}
    for row in pool:
        by_group.setdefault(sample_quota_group(row.get("case_subtype")), []).append(row)
    for rows in by_group.values():
        rows.sort(key=lambda row: (candidate_sort_key(row), str(row.get("case_id", ""))))
        rng.shuffle(rows)
        rows.sort(key=candidate_sort_key)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    shortage_by_quota: dict[str, dict[str, int]] = {}
    for group, quota in SAMPLE_QUOTAS.items():
        rows = by_group.get(group, [])
        take = min(quota, len(rows), max(0, args.target_count - len(selected)))
        selected.extend(rows[:take])
        selected_ids.update(str(row["case_id"]) for row in rows[:take])
        shortage_by_quota[group] = {"quota": quota, "available": len(rows), "selected": take, "shortage": max(0, quota - take)}

    if args.relax_subtype_quota and len(selected) < args.target_count:
        leftovers = [row for row in pool if str(row.get("case_id")) not in selected_ids]
        rng.shuffle(leftovers)
        leftovers.sort(key=candidate_sort_key)
        for row in leftovers:
            if len(selected) >= args.target_count:
                break
            selected.append(row)
            selected_ids.add(str(row["case_id"]))

    selected = selected[: args.target_count]
    for row in records:
        row["selected"] = str(row.get("case_id")) in selected_ids

    fallback_period_available = {
        "2000-2009": sum(1 for row in records if isinstance(row.get("decision_year"), int) and 2000 <= int(row["decision_year"]) <= 2009),
        "1990-1999": sum(1 for row in records if isinstance(row.get("decision_year"), int) and 1990 <= int(row["decision_year"]) <= 1999),
    }
    return sorted(selected, key=lambda row: str(row.get("case_id", ""))), {
        "sampling_method": "strict_pool_subtype_year_quota",
        "seed": args.seed,
        "subtype_quotas": SAMPLE_QUOTAS,
        "quota_shortage_report": shortage_by_quota,
        "preferred_period": f"{args.year_min}-{args.year_max}",
        "preferred_period_eligible": len(preferred),
        "shortage": max(0, args.target_count - len(selected)),
        "fallback_period_available": fallback_period_available,
        "allow_year_fallback": args.allow_year_fallback,
        "relax_subtype_quota": args.relax_subtype_quota,
    }


def count_by(rows: Iterable[dict[str, object]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "") or "unknown") for row in rows))


def count_evidence(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in row.get("court_level_evidence") or []:
            counter[str(item).split(":", 1)[0]] += 1
    return dict(counter)


def summarize(
    *,
    scanned: int,
    keyword_skipped: int,
    pools: dict[str, list[dict[str, object]]],
    selected: list[dict[str, object]],
    duplicate_counts: dict[str, int],
    args: argparse.Namespace,
    sampling_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    keyword_hits = pools["keyword_hits"]
    strict = pools["strict_eligible"]
    liability_counts = Counter(str(row.get("liability_basis") or "unclear") for row in keyword_hits)
    factual_sufficient = [row for row in keyword_hits if row.get("factual_background_sufficient") is True]
    court_counts = Counter(str(row.get("court_level") or "unknown") for row in keyword_hits)
    confidence_counts = Counter(str(row.get("court_level_confidence") or "unknown") for row in pools["appellate_candidates"])
    exclusion_counts = Counter(reason for row in keyword_hits for reason in row.get("exclusion_reasons", []))
    factual_reason_counts = Counter(reason.split(":", 1)[0] for row in keyword_hits for reason in row.get("factual_sufficiency_reasons", []))
    summary = {
        "collection_version": COLLECTION_VERSION,
        "total_scanned": scanned,
        "keyword_hit_count": len(keyword_hits),
        "keyword_gate_skipped": keyword_skipped,
        "civil_candidate_count": len(pools["civil_candidates"]),
        "court_level_appellate_count": court_counts.get("appellate", 0),
        "court_level_supreme_count": court_counts.get("supreme", 0),
        "court_level_trial_count": court_counts.get("trial", 0),
        "court_level_unknown_count": court_counts.get("unknown", 0),
        "appellate_high_confidence": confidence_counts.get("high", 0),
        "appellate_medium_confidence": confidence_counts.get("medium", 0),
        "appellate_low_confidence": confidence_counts.get("low", 0),
        "tort_likely_count": liability_counts.get("non_contractual_tort", 0),
        "non_contractual_tort": liability_counts.get("non_contractual_tort", 0),
        "mixed_tort_contract_count": liability_counts.get("mixed_tort_contract", 0),
        "contract_only_count": liability_counts.get("contract_only", 0),
        "insurance_only_count": liability_counts.get("insurance_only", 0),
        "procedural_only_count": liability_counts.get("procedural_only", 0),
        "administrative_or_state_liability_count": liability_counts.get("administrative_or_state_liability", 0),
        "factually_sufficient_count": len(factual_sufficient),
        "strict_eligible_pool_count": len(strict),
        "selected_count": len(selected),
        "target_count": args.target_count,
        "candidate_by_year": count_by(keyword_hits, "decision_year"),
        "strict_eligible_by_year": count_by(strict, "decision_year"),
        "candidate_by_subtype": count_by(keyword_hits, "case_subtype"),
        "strict_eligible_by_subtype": count_by(strict, "case_subtype"),
        "court_level_evidence_counts": count_evidence(keyword_hits),
        "exclusion_reason_counts": dict(exclusion_counts),
        "factual_insufficiency_reason_counts": dict(factual_reason_counts),
        "duplicate_removed_counts": duplicate_counts,
        "final_selected_subtype_distribution": count_by(selected, "case_subtype"),
        "final_selected_year_distribution": count_by(selected, "decision_year"),
        "scan_limit": args.scan_limit,
        "year_min": args.year_min,
        "year_max": args.year_max,
        "court_level": args.court_level,
        "strict_tort_only": args.strict_tort_only,
    }
    if sampling_meta:
        summary.update(sampling_meta)
    return summary


def iter_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.local_arrow_dir:
        arrow_paths = sorted(Path(args.local_arrow_dir).glob("*.arrow"))
        if not arrow_paths:
            raise FileNotFoundError(f"No .arrow files found in {args.local_arrow_dir}")
        for arrow_path in arrow_paths:
            dataset = Dataset.from_file(str(arrow_path))
            for row in dataset:
                yield row
        return
    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    yield from dataset


def collect(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    scanned = 0
    keyword_skipped = 0
    for scanned, row in enumerate(iter_rows(args), start=1):
        if args.scan_limit and scanned > args.scan_limit:
            scanned -= 1
            break
        gate_keep, _ = keyword_gate(row, args)
        if not gate_keep:
            keyword_skipped += 1
            if args.progress_every and scanned % args.progress_every == 0:
                LOGGER.info("scanned=%s keyword_hits=%s qc_candidates=%s gate_skipped=%s", scanned, len(records), len(records), keyword_skipped)
            continue
        record = evaluate_row(row, args)
        if record:
            records.append(record)
        if args.preview_only and len(records) >= args.preview_count:
            break
        if args.progress_every and scanned % args.progress_every == 0:
            LOGGER.info("scanned=%s keyword_hits=%s qc_candidates=%s gate_skipped=%s", scanned, len(records), len(records), keyword_skipped)

    duplicate_counts = mark_duplicate_candidates(records)
    pools = split_pools(records, args)
    selected: list[dict[str, object]] = []
    sampling_meta: dict[str, object] = {
        "preferred_period": f"{args.year_min}-{args.year_max}",
        "preferred_period_eligible": sum(
            1 for row in pools["strict_eligible"] if isinstance(row.get("decision_year"), int) and args.year_min <= int(row["decision_year"]) <= args.year_max
        ),
        "shortage": args.target_count,
        "fallback_period_available": {
            "2000-2009": sum(1 for row in pools["strict_eligible"] if isinstance(row.get("decision_year"), int) and 2000 <= int(row["decision_year"]) <= 2009),
            "1990-1999": sum(1 for row in pools["strict_eligible"] if isinstance(row.get("decision_year"), int) and 1990 <= int(row["decision_year"]) <= 1999),
        },
    }
    if args.select_final_sample:
        selected, sampling_meta = select_final_sample(pools["strict_eligible"], args)
    summary = summarize(
        scanned=scanned,
        keyword_skipped=keyword_skipped,
        pools=pools,
        selected=selected,
        duplicate_counts=duplicate_counts,
        args=args,
        sampling_meta=sampling_meta,
    )
    return pools, selected, summary


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: object) -> object:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return value


def write_qc_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in QC_FIELDS})


def write_manifest(path: Path, selected: list[dict[str, object]], raw_path: Path) -> None:
    selected_ids = {str(row.get("case_id")) for row in selected}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            out = {field: row.get(field, "") for field in MANIFEST_FIELDS}
            out["selected"] = str(row.get("case_id")) in selected_ids
            out["raw_path"] = str(raw_path)
            writer.writerow(out)


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    return {
        "keyword_hits": output_dir / "kr_keyword_hits_all.jsonl",
        "civil_candidates": output_dir / "kr_civil_candidates_all.jsonl",
        "appellate_candidates": output_dir / "kr_appellate_candidates_all.jsonl",
        "tort_candidates": output_dir / "kr_tort_candidates_all.jsonl",
        "strict_eligible": output_dir / "kr_strict_eligible_all.jsonl",
        "selected": output_dir / f"kr_cases_selected_{args.target_count}.jsonl",
        "qc": output_dir / "kr_cases_qc.csv",
        "summary": output_dir / "kr_cases_summary.json",
        "manifest": Path(args.manifest_output),
    }


def print_summary_lines(summary: dict[str, object]) -> None:
    keys = [
        "total_scanned",
        "keyword_hit_count",
        "civil_candidate_count",
        "appellate_high_confidence",
        "appellate_medium_confidence",
        "appellate_low_confidence",
        "court_level_unknown_count",
        "non_contractual_tort",
        "mixed_tort_contract_count",
        "contract_only_count",
        "insurance_only_count",
        "procedural_only_count",
        "factually_sufficient_count",
        "strict_eligible_pool_count",
        "selected_count",
        "target_count",
    ]
    for key in keys:
        print(f"{key}={summary.get(key, 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Korean tort appellate case collector v3.")
    parser.add_argument("--dataset", default="lbox/lbox_open")
    parser.add_argument("--config", default="precedent_corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-col", default="")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--scan-limit", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2021)
    parser.add_argument("--court-level", choices=["trial", "appellate", "supreme", "unknown"], default="appellate")
    parser.add_argument("--strict-tort-only", action="store_true")
    parser.add_argument("--allow-year-fallback", action="store_true")
    parser.add_argument("--relax-subtype-quota", action="store_true")
    parser.add_argument("--export-all-candidates", action="store_true")
    parser.add_argument("--select-final-sample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--min-text-chars", type=int, default=1200)
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST_OUTPUT))
    parser.add_argument("--local-arrow-dir", default="")
    args = parser.parse_args()
    if not args.export_all_candidates and not args.select_final_sample:
        args.export_all_candidates = True
        args.select_final_sample = True
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    pools, selected, summary = collect(args)
    print_summary_lines(summary)
    if args.preview_only or args.dry_run:
        return

    paths = output_paths(args)
    write_targets = [paths["summary"], paths["qc"], paths["manifest"]]
    if args.export_all_candidates:
        write_targets.extend([paths["keyword_hits"], paths["civil_candidates"], paths["appellate_candidates"], paths["tort_candidates"], paths["strict_eligible"]])
    if args.select_final_sample:
        write_targets.append(paths["selected"])
    require_outputs(write_targets, args.overwrite)

    if args.export_all_candidates:
        write_jsonl(paths["keyword_hits"], pools["keyword_hits"])
        write_jsonl(paths["civil_candidates"], pools["civil_candidates"])
        write_jsonl(paths["appellate_candidates"], pools["appellate_candidates"])
        write_jsonl(paths["tort_candidates"], pools["tort_candidates"])
        write_jsonl(paths["strict_eligible"], pools["strict_eligible"])
    if args.select_final_sample:
        write_jsonl(paths["selected"], selected)
    write_qc_csv(paths["qc"], pools["keyword_hits"])
    write_summary(paths["summary"], summary)
    write_manifest(paths["manifest"], selected, paths["selected"])
    print(f"raw={paths['selected']}")
    print(f"qc={paths['qc']}")
    print(f"summary={paths['summary']}")
    print(f"manifest={paths['manifest']}")


if __name__ == "__main__":
    main()
