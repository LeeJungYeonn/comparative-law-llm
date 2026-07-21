from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset

from pipeline.stage1_raw import compact, normalized_text_for_hash, parse_year, require_outputs, sha256_text, short_hash, unique, write_summary
from pipeline.text_utils import normalize_whitespace, split_sentences


LOGGER = logging.getLogger(__name__)

COLLECTION_VERSION = "stage1-ca-tort-appellate-v3"
DEFAULT_OUTPUT_DIR = Path("outputs/raw/ca_v3")
DEFAULT_MANIFEST_OUTPUT = Path("outputs/manifests/ca_v3_case_manifest.csv")
DEFAULT_ALIGNMENT_OUTPUT = Path("outputs/manifests/kr_ca_sampling_alignment.csv")

TEXT_KEYS = ("opinion_text", "text", "raw_text")
CASE_NAME_KEYS = ("case_name", "case_name_full", "case_name_short", "name")
COURT_KEYS = ("court_full_name", "court_short_name", "court_name", "court")
DATE_KEYS = ("date_filed", "decision_date", "date")
DOCKET_KEYS = ("docket_number", "docket", "docket_numbers")

BROAD_TORT_PATTERNS = [
    r"\btort\b",
    r"\bnegligence\b",
    r"\bnegligent\b",
    r"duty of care",
    r"breach of duty",
    r"\bdamages\b",
    r"personal injur",
    r"bodily injur",
    r"wrongful death",
    r"premises liability",
    r"dangerous condition",
    r"product liability",
    r"strict liability",
    r"defective product",
    r"medical malpractice",
    r"professional negligence",
    r"motor vehicle accident",
    r"automobile accident",
    r"\bcollision\b",
    r"\bpedestrian\b",
    r"property damage",
    r"emotional distress",
    r"defamation",
    r"\blibel\b",
    r"\bslander\b",
    r"invasion of privacy",
    r"vicarious liability",
    r"respondeat superior",
    r"employer liability",
    r"failure to warn",
    r"failure to maintain",
    r"\bcausation\b",
    r"proximate cause",
    r"comparative fault",
    r"contributory negligence",
    r"intentional tort",
    r"\bbattery\b",
    r"\bassault\b",
    r"\bfraud\b",
    r"misrepresentation",
    r"\bnuisance\b",
    r"\btrespass\b",
    r"\baccident\b",
]

CA_APPELLATE_PATTERNS = [
    r"California Court of Appeal",
    r"Court of Appeal of the State of California",
    r"Court of Appeal,? (?:First|Second|Third|Fourth|Fifth|Sixth) Appellate District",
    r"Cal\. Ct\. App\.",
    r"California Courts of Appeal",
]
CA_SUPREME_PATTERNS = [r"Supreme Court of California", r"California Supreme Court"]
FEDERAL_PATTERNS = [
    r"United States Supreme Court",
    r"U\.S\. Supreme Court",
    r"United States Court of Appeals",
    r"Ninth Circuit",
    r"9th Cir\.",
    r"United States District Court",
    r"Central District of California",
    r"Northern District of California",
    r"Southern District of California",
    r"Eastern District of California",
    r"\bC\.D\. Cal\.",
    r"\bN\.D\. Cal\.",
    r"\bS\.D\. Cal\.",
    r"\bE\.D\. Cal\.",
    r"\bF\. ?Supp\.?\b",
    r"\bF\. ?\d+d\b",
]
TRIAL_OR_OTHER_PATTERNS = [
    r"California Superior Court",
    r"Superior Court Appellate Division",
    r"Appellate Division of the Superior Court",
    r"Workers'? Compensation Appeals Board",
    r"Tax Appeals",
    r"Board of Immigration Appeals",
]
ADMIN_PATTERNS = [r"administrative review", r"agency decision", r"writ of mandate", r"mandamus", r"Public Utilities Commission", r"Workers'? Compensation"]
CRIMINAL_PATTERNS = [
    r"\bPeople v\.",
    r"criminal conviction",
    r"defendant was convicted",
    r"\bsentence\b",
    r"imprisonment",
    r"\bfelony\b",
    r"misdemeanor",
    r"prosecution",
    r"Penal Code",
    r"habeas corpus",
    r"\bwarden\b",
    r"\bparole\b",
    r"probation revocation",
    r"criminal appeal",
]
FAMILY_PROBATE_PATTERNS = [r"\bprobate\b", r"\btrust\b", r"\bestate\b", r"child custody", r"\bdivorce\b", r"\bmarital\b", r"\bspousal\b", r"guardianship"]
CONTRACT_PATTERNS = [r"breach of contract", r"contract interpretation", r"contract damages", r"promissory note", r"specific performance", r"purchase agreement", r"lease agreement", r"employment contract"]
INSURANCE_PATTERNS = [r"insurance coverage", r"policy interpretation", r"duty to defend", r"\binsurer\b", r"\binsured\b", r"coverage dispute"]
PROCEDURAL_PATTERNS = [r"statute of limitations", r"\blimitations\b", r"jurisdiction", r"\bvenue\b", r"arbitration", r"attorney fees?", r"costs only", r"judgment enforcement", r"appealability", r"standard of review", r"\bdemurrer\b", r"anti-SLAPP"]
PUBLIC_LAW_PATTERNS = [r"constitutional", r"civil rights", r"42 U\.S\.C", r"section 1983", r"administrative", r"mandamus", r"tax", r"immigration"]
TORT_STRONG_PATTERNS = [
    r"\bnegligence\b",
    r"\bnegligent\b",
    r"premises liability",
    r"product liability",
    r"strict liability",
    r"medical malpractice",
    r"professional negligence",
    r"wrongful death",
    r"personal injur",
    r"bodily injur",
    r"emotional distress",
    r"defamation",
    r"vicarious liability",
    r"respondeat superior",
    r"\bfraud\b",
    r"misrepresentation",
    r"\bnuisance\b",
    r"\btrespass\b",
    r"\bbattery\b",
    r"\bassault\b",
]
CONDUCT_PATTERNS = [r"failed to", r"failure to", r"struck", r"collided", r"operated", r"drove", r"treated", r"diagnosed", r"performed", r"warn", r"maintain", r"supervise", r"published", r"represented", r"exposed"]
DAMAGE_PATTERNS = [r"injur", r"death", r"killed", r"damage", r"harm", r"loss", r"medical expenses", r"emotional distress", r"property", r"pain and suffering"]
CHRONOLOGY_PATTERNS = [r"\b(?:19|20)\d{2}\b", r"after", r"before", r"when", r"while", r"following", r"subsequently", r"then"]
FACT_HEADING_PATTERNS = [r"FACTS", r"BACKGROUND", r"FACTUAL BACKGROUND", r"FACTUAL AND PROCEDURAL BACKGROUND", r"FACTUAL HISTORY", r"STATEMENT OF FACTS", r"THE ACCIDENT", r"UNDERLYING FACTS", r"EVIDENCE AT TRIAL"]
ANALYSIS_HEADING_RE = re.compile(r"(?im)^\s*(?:DISCUSSION|ANALYSIS|STANDARD OF REVIEW|LEGAL PRINCIPLES|CONTENTIONS|DISPOSITION|CONCLUSION)\s*$")

SUBTYPE_PATTERNS = [
    ("traffic_accident", [r"motor vehicle", r"automobile", r"\bcar\b", r"\btruck\b", r"collision", r"pedestrian", r"driver", r"traffic accident"]),
    ("medical_professional", [r"medical malpractice", r"professional negligence", r"hospital", r"physician", r"doctor", r"surgery", r"diagnos", r"treatment"]),
    ("premises_facility_safety", [r"premises liability", r"dangerous condition", r"slip and fall", r"fell", r"sidewalk", r"stairs", r"property owner"]),
    ("product_safety", [r"product liability", r"defective product", r"failure to warn", r"manufacturer", r"design defect", r"strict liability"]),
    ("employer_vicarious_liability", [r"respondeat superior", r"vicarious liability", r"employer", r"employee", r"scope of employment"]),
    ("privacy_reputation", [r"defamation", r"libel", r"slander", r"invasion of privacy", r"privacy", r"reputation"]),
    ("wrongful_death", [r"wrongful death", r"survival action", r"decedent", r"killed", r"fatal"]),
    ("property_damage", [r"property damage", r"trespass", r"nuisance", r"fire damage", r"flood", r"real property"]),
    ("general_personal_injury", [r"personal injur", r"bodily injur", r"assault", r"battery", r"was injured", r"were injured"]),
]

OPINION_PRIORITY = {"majority": 0, "lead": 1, "per_curiam": 2, "per curiam": 2, "plurality": 3, "main": 4, "unanimous": 4, "unknown": 5}
KR_REFERENCE_SUBTYPE_DEFAULT = {
    "traffic_accident": 4,
    "premises_facility_safety": 4,
    "medical_professional": 4,
    "other_tort": 1,
    "product_safety": 1,
    "employer_vicarious_liability": 3,
    "privacy_reputation": 2,
    "general_personal_injury": 1,
}
KR_REFERENCE_YEAR_DEFAULT = {2019: 8, 2020: 8, 2021: 4}

QC_FIELDS = [
    "case_id",
    "source_record_id",
    "case_name",
    "docket_number",
    "citation",
    "decision_date",
    "decision_year",
    "court_name",
    "court_system",
    "court_level",
    "appellate_district",
    "division",
    "court_level_confidence",
    "main_opinion_type",
    "main_opinion_confidence",
    "publication_status",
    "civil_case_likely",
    "criminal_case_likely",
    "liability_basis",
    "case_subtype",
    "procedural_posture",
    "factual_background_sufficient",
    "factual_sufficiency_score",
    "strict_eligible",
    "exclusion_reasons",
    "court_evidence",
    "tort_evidence",
    "factual_sufficiency_reasons",
    "duplicate_or_related_reason",
    "related_case_group_id",
    "raw_text_sha256",
]

MANIFEST_FIELDS = [
    "case_id",
    "collection_version",
    "source_dataset",
    "source_record_id",
    "case_name",
    "docket_number",
    "citation",
    "decision_date",
    "decision_year",
    "court_name",
    "court_system",
    "court_level",
    "appellate_district",
    "division",
    "publication_status",
    "liability_basis",
    "case_subtype",
    "procedural_posture",
    "strict_eligible",
    "selected",
    "matched_kr_case_id",
    "year_match_level",
    "duplicate_or_related_reason",
    "related_case_group_id",
    "raw_text_sha256",
    "raw_length_chars",
    "raw_path",
]


def regex_hits(patterns: Iterable[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def first_existing(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and compact(row.get(key, "")):
            value = row.get(key)
            if isinstance(value, list):
                return "; ".join(compact(item) for item in value if compact(item))
            return compact(value)
    return ""


def citations_text(row: dict[str, Any]) -> str:
    citations = row.get("citations") or row.get("citation") or []
    if isinstance(citations, str):
        return compact(citations)
    if not isinstance(citations, list):
        return compact(citations)
    values = []
    for citation in citations:
        if isinstance(citation, dict):
            values.append(compact(citation.get("cite") or citation.get("citation") or ""))
        else:
            values.append(compact(citation))
    return "; ".join(value for value in values if value)


def normalize_opinion_type(value: object) -> str:
    text = compact(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"majority", "lead", "plurality", "main", "unanimous"}:
        return text
    if text in {"per_curiam", "percuriam"}:
        return "per_curiam"
    if "dissent" in text:
        return "dissent"
    if "concur" in text:
        return "concurrence"
    return "unknown"


def opinion_text_from_dict(opinion: dict[str, Any]) -> str:
    return normalize_whitespace(first_existing(opinion, TEXT_KEYS))


def select_main_opinion(row: dict[str, Any]) -> tuple[str, str, str, list[dict[str, str]], str]:
    opinions = row.get("opinions") or []
    if not isinstance(opinions, list) or not opinions:
        text = normalize_whitespace(first_existing(row, TEXT_KEYS))
        confidence = "medium" if len(text) >= 1000 else "low"
        return text, "unknown", confidence, [], ""

    candidates: list[tuple[int, str, str, dict[str, Any]]] = []
    separate: list[dict[str, str]] = []
    source_url = ""
    for idx, opinion in enumerate(opinions):
        if not isinstance(opinion, dict):
            continue
        text = opinion_text_from_dict(opinion)
        op_type = normalize_opinion_type(opinion.get("type") or opinion.get("opinion_type") or "")
        if opinion.get("download_url") and not source_url:
            source_url = compact(opinion.get("download_url"))
        if op_type in {"dissent", "concurrence"}:
            separate.append({"opinion_type": op_type, "text": text})
            continue
        priority = OPINION_PRIORITY.get(op_type, OPINION_PRIORITY["unknown"])
        if text:
            candidates.append((priority, op_type, text, opinion))

    if not candidates:
        for opinion in opinions:
            if not isinstance(opinion, dict):
                continue
            text = opinion_text_from_dict(opinion)
            op_type = normalize_opinion_type(opinion.get("type") or opinion.get("opinion_type") or "")
            if text:
                return text, op_type, "low", separate, source_url
        return "", "unknown", "low", separate, source_url

    candidates.sort(key=lambda item: (item[0], -len(item[2])))
    _, op_type, text, _ = candidates[0]
    confidence = "high" if op_type in {"majority", "lead", "per_curiam", "plurality"} else "medium"
    return text, op_type, confidence, separate, source_url


def classify_court(row: dict[str, Any], text: str) -> dict[str, object]:
    court_name = first_existing(row, COURT_KEYS)
    jurisdiction = compact(row.get("court_jurisdiction", ""))
    court_type = compact(row.get("court_type", "")).upper()
    haystack = f"{court_name}\n{jurisdiction}\n{court_type}\n{citations_text(row)}\n{text[:1200]}"
    evidence: list[str] = []
    if court_name:
        evidence.append(f"court_metadata: {court_name}")
    if jurisdiction:
        evidence.append(f"jurisdiction_metadata: {jurisdiction}")
    if court_type:
        evidence.append(f"court_type_metadata: {court_type}")

    court_system = "unknown"
    court_level = "unknown"
    if regex_hits(FEDERAL_PATTERNS, haystack):
        court_system = "federal"
        court_level = "federal"
    elif regex_hits(CA_SUPREME_PATTERNS, haystack):
        court_system = "california_state"
        court_level = "supreme"
    elif regex_hits(TRIAL_OR_OTHER_PATTERNS, haystack):
        court_system = "california_state" if "California" in haystack else "unknown"
        court_level = "trial_or_other"
    elif regex_hits(CA_APPELLATE_PATTERNS, haystack) or (court_type == "SA" and jurisdiction.lower() in {"california", "ca", "cal."}):
        court_system = "california_state"
        court_level = "intermediate_appellate"
    elif jurisdiction.lower() in {"california", "ca", "cal."}:
        court_system = "california_state"
        court_level = "unknown"

    district_match = re.search(r"\b(First|Second|Third|Fourth|Fifth|Sixth)\s+Appellate District\b", haystack, flags=re.IGNORECASE)
    division_match = re.search(r"\bDivision\s+(One|Two|Three|Four|Five|Six|Seven|Eight|\d+)\b", haystack, flags=re.IGNORECASE)
    confidence = "high" if court_level in {"intermediate_appellate", "supreme", "federal"} and court_name else "medium" if court_level != "unknown" else "low"
    return {
        "court_name": court_name or "unknown",
        "court_system": court_system,
        "court_level": court_level,
        "appellate_district": district_match.group(1).title() if district_match else "",
        "division": division_match.group(1).title() if division_match else "",
        "court_level_confidence": confidence,
        "jurisdiction_confidence": confidence if court_system == "california_state" else "low",
        "court_evidence": evidence or ["no_court_metadata"],
    }


def classify_publication_status(row: dict[str, Any], citation: str, text: str) -> str:
    raw = compact(row.get("status") or row.get("publication_status") or row.get("published") or "")
    lowered = raw.lower()
    if "unpublished" in lowered or "not published" in lowered:
        return "unpublished"
    if "published" in lowered or lowered in {"true", "yes"}:
        return "published"
    if re.search(r"\b\d+\s+Cal\.(?:App\.)?\s*(?:\d+d|\d+th|\d+)?\s+\d+\b", citation):
        return "published"
    if "not certified for publication" in text[:2000].lower() or "not to be published" in text[:2000].lower():
        return "unpublished"
    return "unknown"


def classify_civil_criminal(case_name: str, haystack: str) -> tuple[bool, bool, list[str]]:
    criminal_hits = regex_hits(CRIMINAL_PATTERNS, f"{case_name}\n{haystack[:4000]}")
    wrongful_death_or_civil = regex_hits([r"wrongful death", r"civil action", r"negligence", r"damages"], haystack)
    criminal = bool(criminal_hits) and not wrongful_death_or_civil
    reasons = [f"criminal_signal: {hit}" for hit in criminal_hits[:5]] if criminal else []
    return not criminal, criminal, reasons


def classify_governing_law(court_system: str, haystack: str) -> tuple[str, str, bool, bool, list[str]]:
    federal_hits = regex_hits([r"42 U\.S\.C", r"section 1983", r"Title VII", r"federal claim", r"constitutional"], haystack)
    other_state_hits = regex_hits([r"New York law", r"Nevada law", r"Arizona law", r"Oregon law", r"Texas law"], haystack)
    if court_system == "california_state" and not federal_hits and not other_state_hits:
        return "california_state_law", "high", False, False, []
    if court_system == "california_state" and federal_hits:
        return "mixed_or_federal_law", "medium", True, bool(other_state_hits), [f"federal_law_signal: {hit}" for hit in federal_hits[:4]]
    return "unclear", "low", bool(federal_hits), bool(other_state_hits), []


def classify_liability_basis(haystack: str) -> tuple[str, str, list[str]]:
    tort_hits = regex_hits(TORT_STRONG_PATTERNS, haystack)
    broad_hits = regex_hits(BROAD_TORT_PATTERNS, haystack)
    contract_hits = regex_hits(CONTRACT_PATTERNS, haystack)
    insurance_hits = regex_hits(INSURANCE_PATTERNS, haystack)
    procedural_hits = regex_hits(PROCEDURAL_PATTERNS, haystack)
    admin_hits = regex_hits(ADMIN_PATTERNS, haystack)
    family_hits = regex_hits(FAMILY_PROBATE_PATTERNS, haystack)
    public_law_hits = regex_hits(PUBLIC_LAW_PATTERNS, haystack)
    conduct_hits = regex_hits(CONDUCT_PATTERNS, haystack)
    damage_hits = regex_hits(DAMAGE_PATTERNS, haystack)
    evidence = [f"tort_signal: {hit}" for hit in (tort_hits or broad_hits)[:8]]

    if family_hits and (not tort_hits or regex_hits([r"did not involve independent", r"no independent"], haystack)):
        return "family_or_probate", "high", [f"family_or_probate_signal: {hit}" for hit in family_hits[:5]]
    if contract_hits and regex_hits([r"no bodily injury", r"no personal injury", r"no property damage", r"only contract", r"contract damages"], haystack):
        return "contract_only", "high", [f"contract_signal: {hit}" for hit in contract_hits[:5]]
    if insurance_hits and not (tort_hits and conduct_hits and damage_hits):
        return "insurance_only", "high", [f"insurance_signal: {hit}" for hit in insurance_hits[:5]]
    if procedural_hits and not (tort_hits and conduct_hits and damage_hits):
        return "procedural_only", "high", [f"procedural_signal: {hit}" for hit in procedural_hits[:5]]
    if admin_hits:
        return "administrative_or_public_law", "medium", [f"admin_signal: {hit}" for hit in admin_hits[:5]]
    if public_law_hits and not tort_hits:
        return "civil_rights_only", "medium", [f"public_law_signal: {hit}" for hit in public_law_hits[:5]]
    if tort_hits and conduct_hits and damage_hits:
        if contract_hits:
            return "mixed_tort_contract", "medium", evidence + [f"contract_signal: {hit}" for hit in contract_hits[:5]]
        return "non_contractual_tort", "high", evidence + [f"conduct_signal: {hit}" for hit in conduct_hits[:5]] + [f"damage_signal: {hit}" for hit in damage_hits[:5]]
    if contract_hits and not tort_hits:
        return "contract_only", "high", [f"contract_signal: {hit}" for hit in contract_hits[:5]]
    if tort_hits or broad_hits:
        return "unclear", "low", evidence
    return "unclear", "low", ["no_tort_signal"]


def classify_subtype(haystack: str) -> str:
    for subtype, patterns in SUBTYPE_PATTERNS:
        if regex_hits(patterns, haystack):
            return subtype
    if regex_hits(TORT_STRONG_PATTERNS, haystack):
        return "other_tort"
    return "unclear"


def classify_procedural_posture(haystack: str) -> str:
    posture_patterns = [
        ("summary_judgment", [r"summary judgment"]),
        ("demurrer_or_motion_to_dismiss", [r"\bdemurrer\b", r"motion to dismiss"]),
        ("anti_slapp", [r"anti-SLAPP", r"special motion to strike"]),
        ("judgment_notwithstanding_verdict", [r"judgment notwithstanding the verdict", r"\bJNOV\b"]),
        ("new_trial_motion", [r"new trial motion", r"motion for new trial"]),
        ("post_trial_appeal", [r"jury verdict", r"bench trial", r"judgment after trial", r"trial court entered judgment"]),
        ("interlocutory_or_procedural", [r"writ petition", r"appealability", r"interlocutory"]),
    ]
    for label, patterns in posture_patterns:
        if regex_hits(patterns, haystack):
            return label
    return "unknown"


def factual_status(posture: str) -> str:
    if posture == "summary_judgment":
        return "record_evidence"
    if posture == "demurrer_or_motion_to_dismiss":
        return "assumed_true_at_pleading_stage"
    if posture == "post_trial_appeal":
        return "jury_found_fact"
    return "unclear"


def extract_fact_probe(text: str) -> tuple[str, bool]:
    value = normalize_whitespace(text)
    heading_re = re.compile(
        r"(?im)^\s*(?:I+\.\s*)?(?:FACTS|BACKGROUND|FACTUAL BACKGROUND|FACTUAL AND PROCEDURAL BACKGROUND|FACTUAL HISTORY|STATEMENT OF FACTS|THE ACCIDENT|UNDERLYING FACTS|EVIDENCE AT TRIAL)\s*$"
    )
    match = heading_re.search(value)
    if match:
        tail = value[match.end() :]
        end_match = ANALYSIS_HEADING_RE.search(tail)
        return normalize_whitespace(tail[: end_match.start()] if end_match else tail), True
    sentences = split_sentences(value)
    return " ".join(sentences[: min(18, len(sentences))]), False


def assess_factual_sufficiency(text: str, haystack: str, posture: str) -> tuple[bool, int, list[str], str]:
    fact_probe, has_heading = extract_fact_probe(text)
    if len(fact_probe) < 250:
        fact_probe = haystack[:6000]
    reasons: list[str] = ["fact_heading_detected" if has_heading else "no_explicit_fact_heading"]
    categories = {
        "party_or_context": regex_hits([r"plaintiff", r"defendant", r"appellant", r"respondent", r"patient", r"driver", r"employee", r"customer"], fact_probe),
        "specific_conduct": regex_hits(CONDUCT_PATTERNS, fact_probe),
        "injury_event": regex_hits([r"accident", r"collision", r"fall", r"surgery", r"publication", r"exposure", r"incident"], fact_probe),
        "harm": regex_hits(DAMAGE_PATTERNS, fact_probe),
        "chronology": regex_hits(CHRONOLOGY_PATTERNS, fact_probe),
        "defense_or_dispute": regex_hits([r"denied", r"disputed", r"contended", r"argued", r"defense", r"comparative", r"assumption of risk"], fact_probe),
    }
    score = sum(1 for hits in categories.values() if hits)
    reasons.extend(f"{name}: {hits[0]}" for name, hits in categories.items() if hits)
    min_fact_chars = 200 if score >= 5 else 500
    if len(fact_probe) < min_fact_chars:
        reasons.append("fact_background_too_short")
    if score < 4:
        reasons.append("insufficient_fact_categories")
    procedural_heavy = posture in {"interlocutory_or_procedural", "anti_slapp"} and score < 5
    if procedural_heavy:
        reasons.append("procedural_posture_with_weak_underlying_facts")
    legal_only = bool(regex_hits([r"standard of review", r"legal principles", r"we review", r"precedent"], fact_probe)) and score < 4
    if legal_only:
        reasons.append("mostly_legal_or_procedural_discussion")
    return len(fact_probe) >= min_fact_chars and score >= 4 and not procedural_heavy and not legal_only, score, unique(reasons), factual_status(posture)


def stable_case_id(source_dataset: str, source_record_id: str, citation: str, docket_number: str, decision_date: str, text: str) -> str:
    stable = compact(source_record_id) or compact(citation) or compact(docket_number)
    return f"CA_{short_hash(source_dataset, stable, decision_date, text[:1000], length=16)}"


def scanned_manifest_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, object]:
    court = first_existing(row, COURT_KEYS)
    return {
        "source_dataset": args.dataset,
        "source_record_id": compact(row.get("id", "")),
        "case_name": first_existing(row, CASE_NAME_KEYS),
        "court_name": court,
        "court_jurisdiction": compact(row.get("court_jurisdiction", "")),
        "court_type": compact(row.get("court_type", "")),
        "decision_date": first_existing(row, DATE_KEYS),
        "citation": citations_text(row),
        "main_opinion_type": "",
        "main_opinion_confidence": "",
        "raw_length_chars": 0,
    }


def row_maybe_california_metadata(row: dict[str, Any]) -> bool:
    court = first_existing(row, COURT_KEYS)
    jurisdiction = compact(row.get("court_jurisdiction", ""))
    court_type = compact(row.get("court_type", ""))
    citation = citations_text(row)
    haystack = f"{court}\n{jurisdiction}\n{court_type}\n{citation}"
    return bool(
        re.search(r"\bCalifornia\b|\bCal\.?\b|Cal\. Ct\. App\.|Ninth Circuit|Central District of California|Northern District of California|Southern District of California|Eastern District of California", haystack, flags=re.IGNORECASE)
        or jurisdiction.lower() in {"california", "ca", "cal."}
    )


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, object]:
    main_text, main_type, main_confidence, separate_opinions, source_url = select_main_opinion(row)
    main_text = normalize_whitespace(main_text)
    case_name = first_existing(row, CASE_NAME_KEYS)
    citation = citations_text(row)
    docket_number = first_existing(row, DOCKET_KEYS)
    decision_date = first_existing(row, DATE_KEYS)
    decision_year = parse_year(decision_date)
    court = classify_court(row, main_text)
    publication_status = classify_publication_status(row, citation, main_text)
    metadata = "\n".join(
        [
            case_name,
            docket_number,
            citation,
            decision_date,
            str(court["court_name"]),
            compact(row.get("headmatter", "")),
            compact(row.get("summary", "")),
            compact(row.get("syllabus", "")),
            compact(row.get("disposition", "")),
        ]
    )
    haystack = f"{metadata}\n{main_text[:16000]}"
    civil_likely, criminal_likely, criminal_reasons = classify_civil_criminal(case_name, haystack)
    primary_law, law_confidence, federal_claim, other_state_law, law_reasons = classify_governing_law(str(court["court_system"]), haystack)
    liability_basis, tort_confidence, tort_evidence = classify_liability_basis(haystack)
    subtype = classify_subtype(haystack)
    posture = classify_procedural_posture(haystack)
    fact_sufficient, fact_score, fact_reasons, fact_status = assess_factual_sufficiency(main_text, haystack, posture)
    broad_hits = regex_hits(BROAD_TORT_PATTERNS, haystack)
    raw_hash = sha256_text(main_text)

    exclusion_reasons: list[str] = []
    exclusion_reasons.extend(criminal_reasons)
    exclusion_reasons.extend(law_reasons)
    if court["court_system"] != args.court_system.replace("-", "_"):
        exclusion_reasons.append(f"non_target_court_system:{court['court_system']}")
    if court["court_level"] == "supreme":
        exclusion_reasons.append("california_supreme_excluded")
    if court["court_level"] == "federal":
        exclusion_reasons.append("federal_court_excluded")
    if court["court_level"] in {"trial_or_other", "unknown"}:
        exclusion_reasons.append(f"non_target_court_level:{court['court_level']}")
    if court["court_level"] != args.court_level.replace("-", "_"):
        exclusion_reasons.append(f"not_{args.court_level}:{court['court_level']}")
    if decision_year is None:
        exclusion_reasons.append("decision_year_unknown")
    elif decision_year < args.year_min or decision_year > args.year_max:
        exclusion_reasons.append("decision_year_out_of_range")
    if not civil_likely:
        exclusion_reasons.append("not_civil_case")
    if not broad_hits:
        exclusion_reasons.append("no_broad_tort_keyword")
    if args.strict_tort_only and liability_basis != "non_contractual_tort":
        exclusion_reasons.append(f"not_strict_non_contractual_tort:{liability_basis}")
    if primary_law != "california_state_law":
        exclusion_reasons.append(f"non_primary_california_state_law:{primary_law}")
    if not fact_sufficient:
        exclusion_reasons.append("factual_background_insufficient")
    if not main_text:
        exclusion_reasons.append("main_opinion_missing")
    if main_type in {"dissent", "concurrence"} or (main_confidence == "low" and main_type == "unknown"):
        exclusion_reasons.append(f"main_opinion_not_strict_eligible:{main_type}")
    if len(main_text) < args.min_text_chars:
        exclusion_reasons.append("too_short_or_no_full_opinion_text")
    if args.max_text_chars and len(main_text) > args.max_text_chars:
        exclusion_reasons.append("too_long")
    if args.publication_status != "any" and publication_status != args.publication_status:
        exclusion_reasons.append(f"publication_status_mismatch:{publication_status}")

    strict_eligible = (
        str(court["court_system"]) == args.court_system.replace("-", "_")
        and str(court["court_level"]) == args.court_level.replace("-", "_")
        and civil_likely
        and not criminal_likely
        and bool(broad_hits)
        and liability_basis == "non_contractual_tort"
        and primary_law == "california_state_law"
        and fact_sufficient
        and isinstance(decision_year, int)
        and args.year_min <= decision_year <= args.year_max
        and bool(main_text)
        and main_type not in {"dissent", "concurrence"}
        and not (main_confidence == "low" and main_type == "unknown")
        and len(main_text) >= args.min_text_chars
        and not (args.max_text_chars and len(main_text) > args.max_text_chars)
        and (args.publication_status == "any" or publication_status == args.publication_status)
    )
    source_record_id = compact(row.get("id", ""))
    case_id = stable_case_id(args.dataset, source_record_id, citation, docket_number, decision_date, main_text)
    return {
        "case_id": case_id,
        "collection_version": COLLECTION_VERSION,
        "source_dataset": args.dataset,
        "source_record_id": source_record_id,
        "source_url_or_citation": source_url or citation,
        "case_name": case_name,
        "docket_number": docket_number,
        "citation": citation,
        "decision_date": decision_date,
        "decision_year": decision_year,
        **court,
        "main_opinion_text": main_text,
        "raw_text": main_text,
        "main_opinion_type": main_type,
        "main_opinion_confidence": main_confidence,
        "separate_opinions": separate_opinions[:5],
        "separate_opinion_count": len(separate_opinions),
        "publication_status": publication_status,
        "civil_case_likely": civil_likely,
        "criminal_case_likely": criminal_likely,
        "primary_governing_law": primary_law,
        "federal_claim_present": federal_claim,
        "other_state_law_present": other_state_law,
        "governing_law_confidence": law_confidence,
        "liability_basis": liability_basis,
        "tort_confidence": tort_confidence,
        "tort_evidence": tort_evidence,
        "case_subtype": subtype,
        "procedural_posture": posture,
        "fact_status": fact_status,
        "factual_background_sufficient": fact_sufficient,
        "factual_sufficiency_score": fact_score,
        "factual_sufficiency_reasons": fact_reasons,
        "strict_eligible": strict_eligible,
        "exclusion_reasons": unique(exclusion_reasons),
        "include_signals": broad_hits,
        "raw_text_sha256": raw_hash,
        "normalized_text_sha256": sha256_text(normalized_text_for_hash(main_text)),
        "raw_length_chars": len(main_text),
        "related_case_group_id": "",
        "related_case_ids": [],
        "duplicate_or_related_reason": "",
        "selected": False,
        "sampling_rank": "",
        "matched_kr_case_id": "",
        "subtype_match": "",
        "year_match_distance": "",
        "year_match_level": "",
        "factual_length_ratio": "",
        "sampling_reason": [],
    }


def iter_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.local_arrow_dir:
        paths = sorted(Path(args.local_arrow_dir).glob("*.arrow"))
        if not paths:
            raise FileNotFoundError(f"No .arrow files found in {args.local_arrow_dir}")
        for path in paths:
            dataset = Dataset.from_file(str(path))
            for row in dataset:
                yield row
        return
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    yield from dataset


def is_california_candidate(record: dict[str, object]) -> bool:
    return str(record.get("court_system")) == "california_state" or "California" in f"{record.get('court_name')} {record.get('case_name')} {record.get('citation')}"


def split_pools(records: list[dict[str, object]], scanned_manifest: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    california = [row for row in records if is_california_candidate(row)]
    state = [row for row in california if row.get("court_system") == "california_state"]
    coa = [row for row in state if row.get("court_level") == "intermediate_appellate"]
    civil = [row for row in coa if row.get("civil_case_likely") is True and row.get("criminal_case_likely") is False]
    tort = [row for row in civil if row.get("include_signals")]
    strict = [row for row in records if row.get("strict_eligible") is True]
    return {
        "all_scanned_manifest": scanned_manifest,
        "california_candidates": california,
        "state_court_candidates": state,
        "court_of_appeal_candidates": coa,
        "civil_candidates": civil,
        "tort_candidates": tort,
        "strict_eligible": strict,
        "strict_eligible_published": [row for row in strict if row.get("publication_status") == "published"],
    }


def mark_duplicates(records: list[dict[str, object]]) -> dict[str, int]:
    counters: Counter[str] = Counter()
    seen_source: dict[str, str] = {}
    seen_citation: dict[str, str] = {}
    seen_docket: dict[str, str] = {}
    seen_exact: dict[str, str] = {}
    seen_norm: dict[str, str] = {}
    for record in records:
        duplicate_of = ""
        reason = ""
        source_id = compact(record.get("source_record_id", ""))
        citation = compact(record.get("citation", "")).lower()
        docket = compact(record.get("docket_number", "")).lower()
        exact = str(record.get("raw_text_sha256") or "")
        norm = str(record.get("normalized_text_sha256") or "")
        if source_id and source_id in seen_source:
            duplicate_of, reason = seen_source[source_id], "duplicate_source_case_id"
        elif citation and citation in seen_citation:
            duplicate_of, reason = seen_citation[citation], "duplicate_citation"
        elif docket and docket in seen_docket:
            duplicate_of, reason = seen_docket[docket], "duplicate_docket_number"
        elif exact and exact in seen_exact:
            duplicate_of, reason = seen_exact[exact], "duplicate_exact_opinion_hash"
        elif norm and norm in seen_norm:
            duplicate_of, reason = seen_norm[norm], "duplicate_normalized_opinion_hash"
        seen_source.setdefault(source_id, str(record["case_id"]))
        if citation:
            seen_citation.setdefault(citation, str(record["case_id"]))
        if docket:
            seen_docket.setdefault(docket, str(record["case_id"]))
        if exact:
            seen_exact.setdefault(exact, str(record["case_id"]))
        if norm:
            seen_norm.setdefault(norm, str(record["case_id"]))
        if duplicate_of:
            record["related_case_group_id"] = f"grp_{short_hash(duplicate_of, record['case_id'], length=12)}"
            record["related_case_ids"] = [duplicate_of]
            record["duplicate_or_related_reason"] = reason
            counters[reason] += 1
    return dict(counters)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reference_distribution(args: argparse.Namespace) -> tuple[dict[str, int], dict[int, int], list[dict[str, object]]]:
    rows = read_jsonl(Path(args.reference_kr_selected)) if args.reference_kr_selected else []
    if not rows:
        return dict(KR_REFERENCE_SUBTYPE_DEFAULT), dict(KR_REFERENCE_YEAR_DEFAULT), []
    subtype_counts = Counter(str(row.get("case_subtype") or "unclear") for row in rows)
    year_counts = Counter(int(row["decision_year"]) for row in rows if isinstance(row.get("decision_year"), int))
    return dict(subtype_counts), dict(year_counts), rows


def year_match_level(distance: int) -> str:
    if distance == 0:
        return "exact"
    if distance == 1:
        return "plus_minus_one_year"
    if distance <= 2:
        return "same_three_year_band"
    return "same_candidate_period"


def candidate_rank(row: dict[str, object], ref_year: int | None, ref_length: int | None) -> tuple[int, int, int, int, str]:
    year = row.get("decision_year")
    year_distance = abs(int(year) - ref_year) if isinstance(year, int) and ref_year is not None else 99
    fact_score = int(row.get("factual_sufficiency_score") or 0)
    opinion_conf = {"high": 0, "medium": 1, "low": 2}.get(str(row.get("main_opinion_confidence")), 3)
    court_conf = {"high": 0, "medium": 1, "low": 2}.get(str(row.get("jurisdiction_confidence")), 3)
    length_distance = abs(int(row.get("raw_length_chars") or 0) - ref_length) if ref_length else 0
    return (year_distance, -fact_score, opinion_conf + court_conf, length_distance, str(row.get("case_id", "")))


def select_final_sample(records: list[dict[str, object]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    subtype_quota, year_quota, kr_rows = reference_distribution(args)
    pool = [row for row in records if not row.get("duplicate_or_related_reason")]
    rng = random.Random(args.seed)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    alignment_rows: list[dict[str, object]] = []
    shortage: dict[str, dict[str, int]] = {}

    if args.match_kr_subtypes and kr_rows:
        refs = kr_rows[: args.target_count]
    else:
        refs = []
        for subtype, count in subtype_quota.items():
            for _ in range(count):
                refs.append({"case_id": "", "case_subtype": subtype, "decision_year": None, "raw_text": ""})
        refs = refs[: args.target_count]

    for rank, ref in enumerate(refs, start=1):
        subtype = str(ref.get("case_subtype") or "unclear")
        ref_year = ref.get("decision_year") if args.match_kr_years else None
        ref_year_int = int(ref_year) if isinstance(ref_year, int) else None
        ref_length = len(str(ref.get("raw_text") or "")) if args.match_kr_lengths else None
        candidates = [row for row in pool if str(row.get("case_id")) not in selected_ids and row.get("case_subtype") == subtype]
        if args.match_kr_years and ref_year_int is not None:
            same_year = [row for row in candidates if row.get("decision_year") == ref_year_int]
            if same_year:
                candidates = same_year
            else:
                candidates = [row for row in candidates if isinstance(row.get("decision_year"), int) and abs(int(row["decision_year"]) - ref_year_int) <= 1] or candidates
        rng.shuffle(candidates)
        candidates.sort(key=lambda row: candidate_rank(row, ref_year_int, ref_length))
        if not candidates:
            shortage.setdefault(subtype, {"quota": 0, "selected": 0, "shortage": 0})
            shortage[subtype]["quota"] += 1
            shortage[subtype]["shortage"] += 1
            continue
        row = candidates[0]
        selected_ids.add(str(row["case_id"]))
        ca_year = row.get("decision_year")
        distance = abs(int(ca_year) - ref_year_int) if isinstance(ca_year, int) and ref_year_int is not None else ""
        length_ratio = round((int(row.get("raw_length_chars") or 0) / ref_length), 4) if ref_length else ""
        row["selected"] = True
        row["sampling_rank"] = rank
        row["matched_kr_case_id"] = str(ref.get("case_id") or "")
        row["subtype_match"] = row.get("case_subtype") == subtype
        row["year_match_distance"] = distance
        row["year_match_level"] = year_match_level(int(distance)) if isinstance(distance, int) else ""
        row["factual_length_ratio"] = length_ratio
        row["sampling_reason"] = ["matched_reference_subtype", f"year_match:{row['year_match_level']}" if row["year_match_level"] else "year_not_matched"]
        selected.append(row)
        shortage.setdefault(subtype, {"quota": 0, "selected": 0, "shortage": 0})
        shortage[subtype]["quota"] += 1
        shortage[subtype]["selected"] += 1
        alignment_rows.append(
            {
                "sampling_rank": rank,
                "kr_case_id": ref.get("case_id", ""),
                "kr_subtype": subtype,
                "kr_year": ref.get("decision_year", ""),
                "kr_raw_length_chars": len(str(ref.get("raw_text") or "")),
                "ca_case_id": row.get("case_id", ""),
                "ca_subtype": row.get("case_subtype", ""),
                "ca_year": row.get("decision_year", ""),
                "ca_raw_length_chars": row.get("raw_length_chars", ""),
                "year_match_distance": distance,
                "year_match_level": row.get("year_match_level", ""),
                "subtype_match": row.get("subtype_match", ""),
            }
        )

    selected = selected[: args.target_count]
    for subtype, quota in subtype_quota.items():
        info = shortage.setdefault(subtype, {"quota": quota, "selected": 0, "shortage": 0})
        info["quota"] = quota
        info["available"] = sum(1 for row in pool if row.get("case_subtype") == subtype)
        info["shortage"] = max(info.get("shortage", 0), max(0, min(quota, args.target_count) - info.get("selected", 0)) if subtype in subtype_quota else info.get("shortage", 0))

    return sorted(selected, key=lambda row: int(row.get("sampling_rank") or 999)), alignment_rows, {
        "sampling_method": "korea_reference_subtype_year_length_matching",
        "seed": args.seed,
        "reference_kr_selected": args.reference_kr_selected,
        "reference_subtype_distribution": subtype_quota,
        "reference_year_distribution": {str(key): value for key, value in year_quota.items()},
        "quota_shortage_report": shortage,
    }


def count_by(rows: Iterable[dict[str, object]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "") or "unknown") for row in rows))


def summarize(
    *,
    scanned: int,
    pools: dict[str, list[dict[str, object]]],
    selected: list[dict[str, object]],
    duplicate_counts: dict[str, int],
    sampling_meta: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    records = pools["california_candidates"]
    coa = pools["court_of_appeal_candidates"]
    strict = pools["strict_eligible"]
    liability_counts = Counter(str(row.get("liability_basis") or "unclear") for row in records)
    exclusion_counts = Counter(reason for row in records for reason in row.get("exclusion_reasons", []))
    factual_reason_counts = Counter(reason.split(":", 1)[0] for row in records for reason in row.get("factual_sufficiency_reasons", []))
    summary = {
        "collection_version": COLLECTION_VERSION,
        "total_scanned": scanned,
        "california_keyword_or_metadata_hits": len(pools["california_candidates"]),
        "california_state_court_candidates": len(pools["state_court_candidates"]),
        "california_court_of_appeal_candidates": len(coa),
        "california_supreme_count": sum(1 for row in records if row.get("court_level") == "supreme"),
        "federal_court_count": sum(1 for row in records if row.get("court_level") == "federal"),
        "trial_or_other_court_count": sum(1 for row in records if row.get("court_level") == "trial_or_other"),
        "court_unknown_count": sum(1 for row in records if row.get("court_level") == "unknown"),
        "civil_candidate_count": len(pools["civil_candidates"]),
        "broad_tort_candidate_count": len(pools["tort_candidates"]),
        "non_contractual_tort_count": liability_counts.get("non_contractual_tort", 0),
        "mixed_tort_contract_count": liability_counts.get("mixed_tort_contract", 0),
        "contract_only_count": liability_counts.get("contract_only", 0),
        "insurance_only_count": liability_counts.get("insurance_only", 0),
        "procedural_only_count": liability_counts.get("procedural_only", 0),
        "administrative_or_public_law_count": liability_counts.get("administrative_or_public_law", 0),
        "civil_rights_only_count": liability_counts.get("civil_rights_only", 0),
        "family_or_probate_count": liability_counts.get("family_or_probate", 0),
        "unclear_count": liability_counts.get("unclear", 0),
        "factually_sufficient_count": sum(1 for row in records if row.get("factual_background_sufficient") is True),
        "criminal_excluded": sum(1 for row in records if row.get("criminal_case_likely") is True),
        "full_main_opinion_available": sum(1 for row in records if int(row.get("raw_length_chars") or 0) >= args.min_text_chars),
        "main_opinion_unknown": sum(1 for row in records if row.get("main_opinion_type") == "unknown"),
        "strict_eligible_pool_count": len(strict),
        "published_strict_eligible": sum(1 for row in strict if row.get("publication_status") == "published"),
        "unpublished_strict_eligible": sum(1 for row in strict if row.get("publication_status") == "unpublished"),
        "publication_unknown_strict_eligible": sum(1 for row in strict if row.get("publication_status") == "unknown"),
        "selected_count": len(selected),
        "target_count": args.target_count,
        "court_of_appeal_candidates_by_year": count_by(coa, "decision_year"),
        "strict_eligible_by_year": count_by(strict, "decision_year"),
        "candidate_by_subtype": count_by(records, "case_subtype"),
        "strict_eligible_by_subtype": count_by(strict, "case_subtype"),
        "appellate_district_counts": count_by(coa, "appellate_district"),
        "publication_status_counts": count_by(records, "publication_status"),
        "strict_publication_status_counts": count_by(strict, "publication_status"),
        "procedural_posture_counts": count_by(records, "procedural_posture"),
        "liability_basis_counts": count_by(records, "liability_basis"),
        "exclusion_reason_counts": dict(exclusion_counts),
        "factual_insufficiency_reason_counts": dict(factual_reason_counts),
        "duplicate_removed_counts": duplicate_counts,
        "final_selected_subtype_distribution": count_by(selected, "case_subtype"),
        "final_selected_year_distribution": count_by(selected, "decision_year"),
        "final_selected_publication_status_distribution": count_by(selected, "publication_status"),
        "year_min": args.year_min,
        "year_max": args.year_max,
        "scan_limit": args.scan_limit,
        "publication_status": args.publication_status,
    }
    summary.update(sampling_meta)
    return summary


def collect(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    scanned_manifest: list[dict[str, object]] = []
    scanned = 0
    for scanned, row in enumerate(iter_rows(args), start=1):
        if args.scan_limit and scanned > args.scan_limit:
            scanned -= 1
            break
        scanned_manifest.append(scanned_manifest_row(row, args))
        if not row_maybe_california_metadata(row):
            if args.progress_every and scanned % args.progress_every == 0:
                LOGGER.info("scanned=%s california_candidates=%s strict=%s", scanned, len(records), sum(1 for item in records if item.get("strict_eligible")))
            continue
        record = evaluate_row(row, args)
        if is_california_candidate(record):
            records.append(record)
        if args.preview_only and scanned >= args.preview_count:
            break
        if args.progress_every and scanned % args.progress_every == 0:
            LOGGER.info("scanned=%s california_candidates=%s strict=%s", scanned, sum(1 for item in records if is_california_candidate(item)), sum(1 for item in records if item.get("strict_eligible")))

    duplicate_counts = mark_duplicates(records)
    pools = split_pools(records, scanned_manifest)
    selected: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    sampling_meta: dict[str, object] = {}
    if args.select_final_sample:
        selected, alignment_rows, sampling_meta = select_final_sample(pools["strict_eligible"], args)
    summary = summarize(scanned=scanned, pools=pools, selected=selected, duplicate_counts=duplicate_counts, sampling_meta=sampling_meta, args=args)
    return pools, selected, alignment_rows, summary


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    return {
        "all_scanned_manifest": output_dir / "ca_all_scanned_manifest.jsonl",
        "california_candidates": output_dir / "ca_california_candidates_all.jsonl",
        "state_court_candidates": output_dir / "ca_state_court_candidates_all.jsonl",
        "court_of_appeal_candidates": output_dir / "ca_court_of_appeal_candidates_all.jsonl",
        "civil_candidates": output_dir / "ca_civil_candidates_all.jsonl",
        "tort_candidates": output_dir / "ca_tort_candidates_all.jsonl",
        "strict_eligible": output_dir / "ca_strict_eligible_all.jsonl",
        "strict_eligible_published": output_dir / "ca_strict_eligible_published.jsonl",
        "selected": output_dir / f"ca_cases_selected_{args.target_count}.jsonl",
        "qc": output_dir / "ca_cases_qc.csv",
        "summary": output_dir / "ca_cases_summary.json",
        "manifest": Path(args.manifest_output),
        "alignment": Path(args.alignment_output),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: object) -> object:
    if isinstance(value, list):
        return "; ".join(json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item) for item in value)
    return value


def write_qc_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in QC_FIELDS})


def write_manifest(path: Path, selected: list[dict[str, object]], raw_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            out = {field: row.get(field, "") for field in MANIFEST_FIELDS}
            out["raw_path"] = str(raw_path)
            writer.writerow(out)


def write_alignment(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sampling_rank",
        "kr_case_id",
        "kr_subtype",
        "kr_year",
        "kr_raw_length_chars",
        "ca_case_id",
        "ca_subtype",
        "ca_year",
        "ca_raw_length_chars",
        "year_match_distance",
        "year_match_level",
        "subtype_match",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, object]) -> None:
    keys = [
        ("total_scanned", "total_scanned"),
        ("california_candidates", "california_keyword_or_metadata_hits"),
        ("california_state_court_candidates", "california_state_court_candidates"),
        ("court_of_appeal_candidates", "california_court_of_appeal_candidates"),
        ("california_supreme_excluded", "california_supreme_count"),
        ("federal_court_excluded", "federal_court_count"),
        ("other_court_excluded", "trial_or_other_court_count"),
        ("civil_candidates", "civil_candidate_count"),
        ("criminal_excluded", "criminal_excluded"),
        ("broad_tort_candidates", "broad_tort_candidate_count"),
        ("non_contractual_tort", "non_contractual_tort_count"),
        ("mixed_tort_contract", "mixed_tort_contract_count"),
        ("contract_only", "contract_only_count"),
        ("insurance_only", "insurance_only_count"),
        ("procedural_only", "procedural_only_count"),
        ("administrative_or_public_law", "administrative_or_public_law_count"),
        ("unclear", "unclear_count"),
        ("full_main_opinion_available", "full_main_opinion_available"),
        ("main_opinion_unknown", "main_opinion_unknown"),
        ("factually_sufficient", "factually_sufficient_count"),
        ("strict_eligible_pool", "strict_eligible_pool_count"),
        ("published_strict_eligible", "published_strict_eligible"),
        ("unpublished_strict_eligible", "unpublished_strict_eligible"),
        ("publication_unknown_strict_eligible", "publication_unknown_strict_eligible"),
        ("selected", "selected_count"),
        ("target_count", "target_count"),
    ]
    for label, key in keys:
        print(f"{label}={summary.get(key, 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 California state Court of Appeal tort collector v3.")
    parser.add_argument("--dataset", default="harvard-lil/cold-cases")
    parser.add_argument("--split", default="train")
    parser.add_argument("--export-all-candidates", action="store_true")
    parser.add_argument("--select-final-sample", action="store_true")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2021)
    parser.add_argument("--court-system", choices=["california-state"], default="california-state")
    parser.add_argument("--court-level", choices=["intermediate-appellate"], default="intermediate-appellate")
    parser.add_argument("--strict-tort-only", action="store_true")
    parser.add_argument("--publication-status", choices=["any", "published", "unpublished"], default="any")
    parser.add_argument("--reference-kr-selected", default="outputs/raw/kr_v3/kr_cases_selected_20.jsonl")
    parser.add_argument("--match-kr-subtypes", action="store_true")
    parser.add_argument("--match-kr-years", action="store_true")
    parser.add_argument("--match-kr-lengths", action="store_true")
    parser.add_argument("--allow-year-fallback", action="store_true")
    parser.add_argument("--relax-subtype-quota", action="store_true")
    parser.add_argument("--allow-publication-status-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-limit", type=int, default=750000)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--min-text-chars", type=int, default=3000)
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST_OUTPUT))
    parser.add_argument("--alignment-output", default=str(DEFAULT_ALIGNMENT_OUTPUT))
    parser.add_argument("--local-arrow-dir", default="")
    args = parser.parse_args()
    if not args.export_all_candidates and not args.select_final_sample:
        args.export_all_candidates = True
        args.select_final_sample = True
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    pools, selected, alignment_rows, summary = collect(args)
    print_summary(summary)
    if args.preview_only or args.dry_run:
        return
    paths = output_paths(args)
    targets = [paths["summary"], paths["qc"], paths["manifest"], paths["alignment"]]
    if args.export_all_candidates:
        targets.extend(
            [
                paths["all_scanned_manifest"],
                paths["california_candidates"],
                paths["state_court_candidates"],
                paths["court_of_appeal_candidates"],
                paths["civil_candidates"],
                paths["tort_candidates"],
                paths["strict_eligible"],
                paths["strict_eligible_published"],
            ]
        )
    if args.select_final_sample:
        targets.append(paths["selected"])
    require_outputs(targets, args.overwrite)
    if args.export_all_candidates:
        for key in [
            "all_scanned_manifest",
            "california_candidates",
            "state_court_candidates",
            "court_of_appeal_candidates",
            "civil_candidates",
            "tort_candidates",
            "strict_eligible",
            "strict_eligible_published",
        ]:
            write_jsonl(paths[key], pools[key])
    if args.select_final_sample:
        write_jsonl(paths["selected"], selected)
    write_qc_csv(paths["qc"], pools["california_candidates"])
    write_summary(paths["summary"], summary)
    write_manifest(paths["manifest"], selected, paths["selected"])
    write_alignment(paths["alignment"], alignment_rows)
    print(f"raw={paths['selected']}")
    print(f"qc={paths['qc']}")
    print(f"summary={paths['summary']}")
    print(f"manifest={paths['manifest']}")
    print(f"alignment={paths['alignment']}")


if __name__ == "__main__":
    main()
