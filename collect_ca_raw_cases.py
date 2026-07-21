from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import random
import re
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests
from datasets import Dataset, load_dataset

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from pipeline.claim_posture import classify_claim_posture, classify_liability_basis
from pipeline.factual_sufficiency import assess_factual_sufficiency, classify_fact_status, classify_procedural_posture
from pipeline.stage1_raw import compact, normalized_text_for_hash, require_outputs, sha256_text, short_hash, unique, write_summary
from pipeline.text_utils import normalize_whitespace
from pipeline.tort_taxonomy import TORT_SUBTYPES, classify_harm_flags, classify_tort_subtype


LOGGER = logging.getLogger(__name__)
COLLECTION_VERSION = "stage1-ca-direct-tort-appellate-v4-shortlist"
SOURCE_DATASET = "harvard-lil/cold-cases"
DEFAULT_OUTPUT_DIR = Path("outputs/raw/ca_v4_shortlist")
DEFAULT_MANIFEST = Path("outputs/manifests/ca_v4_shortlist_case_manifest.csv")
DEFAULT_ALIGNMENT = Path("outputs/manifests/kr_ca_v4_shortlist_alignment.csv")
DEFAULT_KR_REFERENCE = Path("outputs/raw/kr_v4/kr_cases_selected_50_final.jsonl")
DEFAULT_DECISION_DATE_FROM = "2000-01-01"

EXPECTED_KR_DISTRIBUTION = {
    "traffic_accident": 10,
    "medical_professional": 9,
    "employer_vicarious_liability": 7,
    "premises_facility_safety": 6,
    "privacy_reputation": 5,
    "intentional_tort": 5,
    "product_safety": 4,
    "property_damage": 2,
    "general_personal_injury": 1,
    "other_tort": 1,
}

OUTPUT_NAMES = {
    "california": "ca_california_candidates_all.jsonl",
    "state": "ca_state_court_candidates_all.jsonl",
    "appeal": "ca_court_of_appeal_candidates_all.jsonl",
    "civil": "ca_civil_candidates_all.jsonl",
    "direct": "ca_direct_tort_candidates_all.jsonl",
    "strict": "ca_strict_eligible_all.jsonl",
    "published": "ca_strict_eligible_published.jsonl",
    "unpublished": "ca_strict_eligible_unpublished.jsonl",
    "excluded": "ca_excluded_non_direct_claims.jsonl",
    "shortlist": "ca_direct_tort_shortlist_100.jsonl",
    "qc": "ca_direct_tort_shortlist_100_qc.csv",
    "summary": "ca_shortlist_100_summary.json",
}

BROAD_TORT_PATTERNS = [
    r"\btort(?:ious)?\b", r"\bneglig(?:ence|ent)\b", r"duty of care", r"personal injur",
    r"wrongful death", r"premises liability", r"dangerous condition", r"product liability",
    r"medical malpractice", r"professional negligence", r"motor vehicle", r"automobile accident",
    r"\bcollision\b", r"property damage", r"emotional distress", r"defamation", r"\blibel\b",
    r"\bslander\b", r"invasion of privacy", r"vicarious liability", r"respondeat superior",
    r"failure to warn", r"proximate cause", r"comparative fault", r"\bbattery\b", r"\bassault\b",
    r"\bnuisance\b", r"\btrespass\b",
]
POSTURE_SCREEN_PATTERNS = [
    r"liability insurer", r"Insurance Code section 11580", r"\bsubrogat(?:e|ed|ion)\b", r"\bsubrogee\b",
    r"equitable indemnity", r"joint tortfeasor", r"insurance coverage", r"duty to defend",
    r"breach of contract", r"purchase agreement", r"judgment creditor", r"writ of execution",
    r"\bdemurrer\b", r"motion to dismiss",
]

CA_APPEAL_NAMES = [
    r"California Court of Appeal", r"Court of Appeal of California",
    r"Court of Appeal of the State of California",
    r"Court of Appeal,? (?:First|Second|Third|Fourth|Fifth|Sixth) Appellate District",
    r"Cal\. Ct\. App\.",
]
EXCLUDED_COURTS = [
    r"California Supreme Court", r"Supreme Court of California", r"United States Supreme Court",
    r"Ninth Circuit", r"U\.S\. Court of Appeals", r"United States Court of Appeals",
    r"United States District Court", r"District of California", r"California Superior Court",
    r"Superior Court Appellate Division", r"Appellate Division of the Superior Court",
    r"Workers['’] Compensation Appeals Board",
]
CRIMINAL_PATTERNS = [
    r"^People v\.", r"\bconvicted\b", r"criminal conviction", r"\bsentence\b", r"\bfelony\b",
    r"misdemeanor", r"\bprosecution\b", r"habeas corpus", r"\bwarden\b", r"\bparole\b", r"\bprobation\b",
]
CIVIL_TORT_OVERRIDE = [r"wrongful death", r"premises liability", r"medical malpractice", r"personal injury action", r"civil action for"]

OPINION_TYPE_MAP = {
    "010combined": "majority", "015unamimous": "majority", "015unanimous": "majority",
    "020lead": "lead", "025plurality": "plurality", "030concurrence": "concurrence",
    "035concurrenceinpart": "concurrence", "040dissent": "dissent",
    "majority": "majority", "lead": "lead", "plurality": "plurality", "percuriam": "per_curiam",
    "per_curiam": "per_curiam", "concurrence": "concurrence", "dissent": "dissent",
}
MAIN_PRIORITY = {"majority": 0, "lead": 1, "per_curiam": 2, "plurality": 3, "unknown": 4}

QC_COLUMNS = [
    "case_id", "case_name", "citation", "docket_number", "decision_date", "decision_year",
    "court_name", "appellate_district", "division", "publication_status", "main_opinion_type",
    "main_opinion_confidence", "claim_posture", "claim_posture_confidence", "liability_basis",
    "governing_law_confidence", "case_subtype", "procedural_posture", "death_involved",
    "physical_injury_involved", "property_damage_involved", "emotional_harm_involved",
    "facts_independently_reconstructable", "fact_source_quality", "factual_sufficiency_score",
    "reference_kr_subtype_count", "shortlist_minimum_target", "shortlist_subtype_rank",
    "shortlist_overflow_candidate", "shortlist_selection_score", "shortlist_selection_reasons",
    "quality_flags", "exclusion_reasons", "human_qc_status", "human_qc_corrected_subtype",
    "human_qc_corrected_claim_posture", "human_qc_notes",
]


def _matches(text: str, patterns: Iterable[str]) -> list[str]:
    found = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found.append(match.group(0)[:180])
    return unique(found)


def _first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return compact(value)
    return ""


def _metadata_text(row: dict[str, Any]) -> str:
    return "\n".join(compact(row.get(key)) for key in (
        "case_name", "case_name_full", "nature_of_suit", "posture", "summary", "syllabus", "headnotes", "headmatter", "history"
    ) if row.get(key))


def normalize_opinion_type(opinion: dict[str, Any]) -> str:
    if opinion.get("per_curiam") is True:
        return "per_curiam"
    raw = re.sub(r"[^a-z0-9_]", "", compact(opinion.get("type")).lower())
    if raw in OPINION_TYPE_MAP:
        return OPINION_TYPE_MAP[raw]
    if "dissent" in raw:
        return "dissent"
    if "concurr" in raw:
        return "concurrence"
    if "plural" in raw:
        return "plurality"
    if "lead" in raw or "merits" in raw:
        return "lead"
    if "major" in raw or "combined" in raw or "unanim" in raw:
        return "majority"
    return "unknown"


def _opinion_text(opinion: dict[str, Any]) -> str:
    return normalize_whitespace(opinion.get("opinion_text") or opinion.get("text") or opinion.get("ocr") or "")


def select_main_opinion(row: dict[str, Any], *, minimum_chars: int = 1800) -> dict[str, Any]:
    opinions = row.get("opinions") if isinstance(row.get("opinions"), list) else []
    candidates = []
    separate = []
    for opinion in opinions:
        if not isinstance(opinion, dict):
            continue
        kind = normalize_opinion_type(opinion)
        text = _opinion_text(opinion)
        item = {
            "opinion_id": compact(opinion.get("opinion_id")), "opinion_type": kind,
            "author": compact(opinion.get("author_str")), "text_length": len(text),
            "raw_text_sha256": sha256_text(text) if text else "",
        }
        raw_kind = compact(opinion.get("type")).lower()
        if any(raw_kind.startswith(prefix) for prefix in ("050", "060", "070", "090")):
            separate.append(item)
            continue
        if kind in {"concurrence", "dissent"}:
            separate.append(item)
        elif text:
            candidates.append((MAIN_PRIORITY.get(kind, 4), -len(text), kind, text, item))
    if not candidates:
        return {
            "main_opinion_text": "", "main_opinion_type": "unknown", "main_opinion_confidence": "low",
            "full_main_opinion_available": False, "separate_opinions": separate,
            "opinion_exclusion_reason": "separate_opinion_or_summary_only" if separate else "no_opinion_text",
        }
    _, _, kind, text, _ = sorted(candidates, key=lambda item: (item[0], item[1], item[4]["opinion_id"]))[0]
    full = len(text) >= minimum_chars and not re.match(r"(?is)^\s*(?:syllabus|headnote|summary)\b", text)
    confidence = "high" if kind in {"majority", "lead", "per_curiam"} else "medium" if kind in {"plurality", "unknown"} and full else "low"
    return {
        "main_opinion_text": text, "main_opinion_type": kind, "main_opinion_confidence": confidence,
        "full_main_opinion_available": full, "separate_opinions": separate,
        "opinion_exclusion_reason": "" if full else "full_main_opinion_unavailable",
    }


def classify_court(row: dict[str, Any]) -> dict[str, Any]:
    court_name = _first(row, "court_full_name", "court_short_name", "court_name", "court")
    jurisdiction = compact(row.get("court_jurisdiction"))
    court_type = compact(row.get("court_type")).upper()
    value = f"{court_name}\n{jurisdiction}"
    california = bool(re.search(r"California(?:, CA)?", jurisdiction, re.I) or re.search(r"California|Cal\. Ct\.", court_name, re.I))
    excluded = _matches(value, EXCLUDED_COURTS)
    appeal_evidence = _matches(court_name, CA_APPEAL_NAMES)
    metadata_evidence = []
    if jurisdiction.lower() in {"california", "california, ca", "ca"}:
        metadata_evidence.append(f"court_jurisdiction={jurisdiction}")
    if court_type == "SA":
        metadata_evidence.append("court_type=SA")
    is_appeal = bool(appeal_evidence) and not excluded
    state = california and not _matches(value, [r"United States|Federal|Ninth Circuit|District Court"])
    confidence = "high" if is_appeal and appeal_evidence and metadata_evidence else "medium" if is_appeal and len(appeal_evidence + metadata_evidence) >= 2 else "low"
    return {
        "california_candidate": california, "california_state_candidate": state,
        "court_name": court_name or None, "court_system": "california_state" if state else "other",
        "court_level": "intermediate_appellate" if is_appeal else "other",
        "court_level_confidence": confidence,
        "court_evidence": unique(appeal_evidence + metadata_evidence), "court_exclusion_evidence": excluded,
    }


def extract_district_division(row: dict[str, Any], text: str) -> tuple[str | None, str | None]:
    value = f"{_metadata_text(row)[:4000]}\n{text[:1500]}"
    suffix = re.search(r"\bCA([1-6])/(\d+)\b", value, re.I)
    district = re.search(r"\b(First|Second|Third|Fourth|Fifth|Sixth) Appellate District\b", value, re.I)
    division = re.search(r"\bDivision\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|\d+)\b", value, re.I)
    number_words = {"1": "First", "2": "Second", "3": "Third", "4": "Fourth", "5": "Fifth", "6": "Sixth"}
    if suffix:
        return f"{number_words[suffix.group(1)]} Appellate District", f"Division {suffix.group(2)}"
    return (district.group(0).title() if district else None, division.group(0).title() if division else None)


def classify_publication_status(value: Any) -> str:
    raw = compact(value).lower()
    if raw in {"published", "precedential"} or ("published" in raw and "unpublished" not in raw):
        return "published"
    if "unpublished" in raw or "non-precedential" in raw or "nonprecedential" in raw:
        return "unpublished"
    return "unknown"


def classify_civil(case_name: str, text: str, claim_posture: str) -> tuple[bool, list[str]]:
    sample = f"{case_name}\n{text[:15000]}"
    criminal = _matches(sample, CRIMINAL_PATTERNS)
    civil_override = _matches(sample, CIVIL_TORT_OVERRIDE)
    if claim_posture == "direct_tort_claim":
        return True, civil_override
    return not bool(criminal), criminal


def classify_governing_law(text: str, court: dict[str, Any], claim_posture: str) -> dict[str, Any]:
    sample = text[:30000]
    federal = _matches(sample, [r"42 U\.?S\.?C\.? ?§? ?1983", r"federal civil rights", r"federal constitutional claim"])
    other_state = _matches(sample, [r"law of (?!California)[A-Z][a-z]+", r"under (?!California)[A-Z][a-z]+ law"])
    ca = _matches(sample, [r"California (?:Civil|Evidence|Government|Code)", r"California law", r"Cal\. Civ\. Code", r"Civil Code section", r"California common law"])
    state_core = court["court_system"] == "california_state" and claim_posture != "civil_rights_only" and not other_state
    confidence = "high" if state_core and (ca or court["court_level_confidence"] == "high") else "medium" if state_core else "low"
    return {
        "primary_governing_law": "california_state_law" if state_core else "federal_or_other_law",
        "federal_claim_present": bool(federal), "other_state_law_present": bool(other_state),
        "governing_law_confidence": confidence, "governing_law_evidence": unique(ca + federal + other_state),
    }


def _citation(row: dict[str, Any]) -> str | None:
    values = row.get("citations")
    if isinstance(values, list):
        rendered = []
        for value in values:
            if isinstance(value, dict):
                rendered.append(compact(value.get("cite") or value.get("citation") or value.get("volume") or value))
            else:
                rendered.append(compact(value))
        return "; ".join(item for item in rendered if item) or None
    return compact(values) or None


def _docket(row: dict[str, Any], text: str) -> str | None:
    structured = _first(row, "docket_number", "docket", "docket_numbers")
    if structured:
        return structured
    match = re.search(r"(?im)^\s*(?:No\.|Case No\.)\s*([A-Z0-9][A-Z0-9.\-/ ]{2,30})\s*$", text[:2500])
    return compact(match.group(1)) if match else None


def _date(value: Any) -> tuple[str | None, int | None]:
    raw = compact(value)
    match = re.match(r"((?:18|19|20)\d{2})-(\d{2})-(\d{2})", raw)
    if match:
        return match.group(0), int(match.group(1))
    year = re.search(r"\b(18|19|20)\d{2}\b", raw)
    return (raw or None, int(year.group(0)) if year else None)


def event_era(year: Any) -> str:
    try: value = int(year)
    except (TypeError, ValueError): return "unknown"
    if value < 1960: return "before_1960"
    if value <= 1979: return "1960_1979"
    if value <= 1999: return "1980_1999"
    if value <= 2009: return "2000_2009"
    if value <= 2019: return "2010_2019"
    return "2020_or_later"


def assign_primary_exclusion(record: dict[str, Any]) -> None:
    reasons = list(record.get("exclusion_reasons") or [])
    if not reasons:
        record["primary_exclusion_stage"] = None
        record["primary_exclusion_reason"] = None
        return
    reason = reasons[0]
    if reason.startswith("claim_posture:"):
        stage, reason = "claim_posture", reason.split(":", 1)[1]
    elif reason.startswith("liability_basis:"):
        stage = "liability_basis"
    elif "court" in reason or reason.startswith("not_california"):
        stage = "court"
    elif reason == "not_civil_current_appeal":
        stage, reason = "civil", "criminal_current_proceeding"
    elif reason == "not_broad_tort_candidate":
        stage, reason = "broad_tort", "not_broad_tort"
    elif "governing" in reason or "primary_law" in reason:
        stage, reason = "governing_law", "not_california_state_law"
    elif "opinion" in reason or "principal" in reason:
        stage, reason = "main_opinion", "no_full_main_opinion"
    elif "fact" in reason:
        stage, reason = "factual_sufficiency", "insufficient_factual_background"
    elif "duplicate" in reason or "related" in reason:
        stage = "duplicate_related"
    else:
        stage = "unclear"
    record["primary_exclusion_stage"] = stage
    record["primary_exclusion_reason"] = reason


def broad_tort_gate(text: str) -> tuple[bool, list[str]]:
    evidence = _matches(text[:30000], BROAD_TORT_PATTERNS)
    screening = _matches(text[:30000], POSTURE_SCREEN_PATTERNS)
    return len(evidence) >= 2 or bool(screening), unique(evidence + screening)


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    court = classify_court(row)
    opinion = select_main_opinion(row, minimum_chars=getattr(args, "min_opinion_chars", 1800))
    text = opinion["main_opinion_text"]
    metadata = _metadata_text(row)
    case_name = _first(row, "case_name", "case_name_full", "case_name_short")
    source_id = compact(row.get("id") or row.get("source_record_id"))
    decision_date, decision_year = _date(row.get("date_filed") or row.get("decision_date"))
    docket = _docket(row, text or metadata)
    citation = _citation(row)
    district, division = extract_district_division(row, text)
    publication = classify_publication_status(row.get("precedential_status") or row.get("publication_status"))

    classification_text = f"{metadata}\n{text}"
    broad, broad_evidence = broad_tort_gate(classification_text)
    if broad:
        posture = classify_claim_posture(text or metadata, case_name=case_name, metadata=metadata)
        liability_basis, liability_evidence = classify_liability_basis(classification_text, posture["claim_posture"])
        subtype, subtype_evidence = classify_tort_subtype(classification_text)
        harms = classify_harm_flags(classification_text)
        procedural, procedural_evidence = classify_procedural_posture(text)
        fact_status, fact_status_evidence = classify_fact_status(text, procedural)
        facts = assess_factual_sufficiency(text, full_main_opinion_available=opinion["full_main_opinion_available"])
    else:
        posture = {"claim_posture": "unclear", "confidence": "low", "evidence": []}
        liability_basis, liability_evidence = "unclear", []
        subtype, subtype_evidence = "unclear", []
        harms = {key: False for key in ("death_involved", "physical_injury_involved", "property_damage_involved", "emotional_harm_involved")}
        procedural, procedural_evidence = "unknown", []
        fact_status, fact_status_evidence = "unclear", []
        facts = {
            "factual_background_sufficient": False, "facts_independently_reconstructable": False,
            "fact_source_quality": "insufficient", "factual_sufficiency_score": 0,
            "factual_sufficiency_evidence": [], "factual_insufficiency_reasons": ["not_broad_tort_candidate"],
        }
    civil, civil_evidence = classify_civil(case_name, classification_text, posture["claim_posture"])
    governing = classify_governing_law(classification_text, court, posture["claim_posture"])

    exclusions = []
    if not court["california_candidate"]: exclusions.append("not_california_jurisdiction")
    if not court["california_state_candidate"]: exclusions.append("not_california_state_court")
    if court["court_level"] != "intermediate_appellate": exclusions.append("not_california_court_of_appeal")
    if court["court_level_confidence"] not in {"high", "medium"}: exclusions.append("court_level_not_verified")
    if not civil: exclusions.append("not_civil_current_appeal")
    if not broad: exclusions.append("not_broad_tort_candidate")
    if posture["claim_posture"] != "direct_tort_claim": exclusions.append(f"claim_posture:{posture['claim_posture']}")
    if liability_basis != "non_contractual_tort": exclusions.append(f"liability_basis:{liability_basis}")
    if governing["primary_governing_law"] != "california_state_law": exclusions.append("primary_law_not_california_state_law")
    if governing["governing_law_confidence"] not in {"high", "medium"}: exclusions.append("governing_law_not_verified")
    if not opinion["full_main_opinion_available"]: exclusions.append(opinion["opinion_exclusion_reason"] or "full_main_opinion_unavailable")
    if opinion["main_opinion_type"] in {"concurrence", "dissent"} or opinion["main_opinion_confidence"] == "low": exclusions.append("no_valid_principal_opinion")
    if not facts["factual_background_sufficient"]: exclusions.append("factual_background_insufficient")
    if not facts["facts_independently_reconstructable"]: exclusions.append("facts_not_independently_reconstructable")
    if facts["fact_source_quality"] not in {"self_contained", "partially_incorporated"}: exclusions.append("fact_source_quality_insufficient")

    case_id = f"CA_{short_hash(SOURCE_DATASET, source_id or docket or citation or case_name, length=16)}"
    quality_flags = []
    if publication == "unknown": quality_flags.append("publication_status_unknown")
    if not district: quality_flags.append("appellate_district_unknown")
    if opinion["main_opinion_confidence"] == "medium": quality_flags.append("main_opinion_medium_confidence")
    if procedural == "demurrer_or_motion_to_dismiss": quality_flags.append("pleading_facts_not_adjudicated")
    record = {
        "case_id": case_id, "source_dataset": SOURCE_DATASET, "source_record_id": source_id,
        "case_name": case_name or None, "docket_number": docket, "citation": citation,
        "decision_date": decision_date, "decision_year": decision_year,
        "incident_date": None, "incident_year": None, "event_era": "unknown",
        **{key: court[key] for key in ("court_name", "court_system", "court_level", "court_level_confidence", "court_evidence")},
        "appellate_district": district, "division": division,
        **{key: opinion[key] for key in ("main_opinion_text", "main_opinion_type", "main_opinion_confidence", "full_main_opinion_available", "separate_opinions")},
        "publication_status": publication, **governing,
        "liability_basis": liability_basis, "liability_basis_evidence": liability_evidence,
        "claim_posture": posture["claim_posture"], "claim_posture_confidence": posture["confidence"],
        "direct_tort_evidence": posture["evidence"] if posture["claim_posture"] == "direct_tort_claim" else [],
        "claim_posture_evidence": posture["evidence"], "broad_tort_candidate": broad, "broad_tort_evidence": broad_evidence,
        "civil_candidate": civil, "civil_classification_evidence": civil_evidence,
        "case_subtype": subtype, "case_subtype_evidence": subtype_evidence,
        "procedural_posture": procedural, "procedural_posture_evidence": procedural_evidence,
        "fact_epistemic_status": fact_status, "fact_status_evidence": fact_status_evidence,
        **harms, **facts,
        "strict_eligible": not exclusions, "exclusion_reasons": unique(exclusions), "quality_flags": unique(quality_flags),
        "related_case_group_id": None, "related_case_ids": [], "underlying_incident_fingerprint": None,
        "duplicate_or_related_reason": None, "raw_text_sha256": sha256_text(text),
        "normalized_main_opinion_sha256": sha256_text(normalized_text_for_hash(text)),
        "reference_kr_subtype_count": None, "shortlist_minimum_target": None, "shortlist_subtype_rank": None,
        "shortlist_overflow_candidate": False, "shortlist_selection_score": None, "shortlist_selection_reasons": [],
        "human_qc_status": None, "human_qc_corrected_subtype": None,
        "human_qc_corrected_claim_posture": None, "human_qc_notes": None,
        "collection_version": COLLECTION_VERSION,
    }
    assign_primary_exclusion(record)
    return record


def underlying_incident_fingerprint(record: dict[str, Any]) -> str | None:
    name = compact(record.get("case_name")).lower()
    tokens = sorted(set(re.findall(r"[a-z]{3,}", name)) - {"super", "superior", "court", "appeal", "county", "state", "california", "estate"})
    if len(tokens) < 2:
        return None
    text = compact(record.get("main_opinion_text"))[:5000]
    incident_date = re.search(r"\b(?:18|19|20)\d{2}[-/, ]\d{1,2}[-/, ]\d{1,2}\b", text)
    event = _matches(text, [r"motor vehicle|collision|medical procedure|surgery|premises|publication|shooting|assault|product"])
    if not incident_date and not event:
        return None
    return short_hash(" ".join(tokens[:8]), incident_date.group(0) if incident_date else "", "|".join(event[:2]), length=20)


def mark_duplicates(records: list[dict[str, Any]]) -> dict[str, int]:
    reason_counts: Counter[str] = Counter()
    keys = ["source_record_id", "docket_number", "citation", "raw_text_sha256"]
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    normalized_seen: dict[str, dict[str, Any]] = {}
    incident_seen: dict[str, dict[str, Any]] = {}
    ordered = sorted(records, key=lambda row: (
        row.get("claim_posture") != "direct_tort_claim", -int(row.get("factual_sufficiency_score") or 0),
        row.get("main_opinion_confidence") != "high", str(row.get("case_id")),
    ))
    for record in ordered:
        duplicate_of = None
        reason = None
        for key in keys:
            value = compact(record.get(key)).lower()
            if value and (key, value) in seen:
                duplicate_of, reason = seen[(key, value)], f"duplicate_{key}"
                break
        normalized = sha256_text(normalized_text_for_hash(str(record.get("main_opinion_text") or "")))
        if not duplicate_of and normalized in normalized_seen:
            duplicate_of, reason = normalized_seen[normalized], "duplicate_normalized_main_opinion"
        fingerprint = underlying_incident_fingerprint(record)
        record["underlying_incident_fingerprint"] = fingerprint
        if not duplicate_of and fingerprint and fingerprint in incident_seen:
            duplicate_of, reason = incident_seen[fingerprint], "related_underlying_incident"
        if duplicate_of:
            group = duplicate_of.get("related_case_group_id") or f"CAREL_{short_hash(duplicate_of['case_id'], record['case_id'], length=14)}"
            duplicate_of["related_case_group_id"] = group
            record["related_case_group_id"] = group
            record["related_case_ids"] = unique(list(record.get("related_case_ids") or []) + [str(duplicate_of["case_id"])])
            duplicate_of["related_case_ids"] = unique(list(duplicate_of.get("related_case_ids") or []) + [str(record["case_id"])])
            record["duplicate_or_related_reason"] = reason
            record["strict_eligible"] = False
            record["exclusion_reasons"] = unique(list(record.get("exclusion_reasons") or []) + [reason])
            assign_primary_exclusion(record)
            reason_counts[reason] += 1
        else:
            for key in keys:
                value = compact(record.get(key)).lower()
                if value: seen[(key, value)] = record
            normalized_seen[normalized] = record
            if fingerprint: incident_seen[fingerprint] = record
    return dict(reason_counts)


def split_pools(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    california = [row for row in records if classify_bool(row, "california_candidate", default=True)]
    state = [row for row in california if row.get("court_system") == "california_state"]
    appeal = [row for row in state if row.get("court_level") == "intermediate_appellate"]
    civil = [row for row in appeal if row.get("civil_candidate")]
    direct = [row for row in civil if row.get("claim_posture") == "direct_tort_claim"]
    strict = [row for row in direct if row.get("strict_eligible")]
    excluded = [row for row in civil if row.get("broad_tort_candidate") and row.get("claim_posture") != "direct_tort_claim"]
    return {"california": california, "state": state, "appeal": appeal, "civil": civil, "direct": direct, "strict": strict, "excluded": excluded}


def classify_bool(row: dict[str, Any], key: str, default: bool = False) -> bool:
    if key in row: return bool(row[key])
    return default


def iter_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    local_dir = getattr(args, "local_arrow_dir", None)
    if local_dir:
        dataset = Dataset.load_from_disk(str(local_dir))
        yield from dataset
        return
    if getattr(args, "source_loader", "datasets-server") == "datasets-server":
        yield from iter_hf_filtered_rows(args)
        return
    columns = [
        "id", "case_name", "case_name_full", "case_name_short", "citations", "court_full_name",
        "court_short_name", "court_jurisdiction", "court_type", "date_filed", "headmatter", "headnotes",
        "history", "nature_of_suit", "opinions", "posture", "precedential_status", "slug", "summary", "syllabus",
    ]
    # Parquet predicate pushdown scans the complete California jurisdiction range; no first-N limit is used.
    year_min = getattr(args, "year_min", None)
    year_max = getattr(args, "year_max", None)
    decision_date_from = date(int(year_min), 1, 1) if year_min is not None else date.fromisoformat(getattr(args, "decision_date_from", DEFAULT_DECISION_DATE_FROM))
    filters: list[tuple[str, str, Any]] = [
        ("court_jurisdiction", "==", "California, CA"), ("court_type", "==", "SA"),
        ("date_filed", ">=", decision_date_from),
    ]
    if year_max is not None:
        filters.append(("date_filed", "<=", date(int(year_max), 12, 31)))
    dataset = load_dataset(
        getattr(args, "dataset", SOURCE_DATASET), split=getattr(args, "split", "train"), streaming=True,
        columns=columns,
        filters=filters,
        batch_size=getattr(args, "loader_batch_size", 32),
    )
    yield from dataset


def iter_hf_filtered_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    endpoint = "https://datasets-server.huggingface.co/filter"
    year_min = getattr(args, "year_min", None)
    year_max = getattr(args, "year_max", None)
    decision_date_from = f"{int(year_min):04d}-01-01" if year_min is not None else getattr(args, "decision_date_from", DEFAULT_DECISION_DATE_FROM)
    decision_date_to = f"{int(year_max):04d}-12-31" if year_max is not None else None
    where = (
        '"court_jurisdiction"=\'California, CA\' AND '
        '"court_type"=\'SA\' AND '
        f'"date_filed">=\'{decision_date_from}\''
    )
    if decision_date_to:
        where += f' AND "date_filed"<=\'{decision_date_to}\''
    page_size = max(1, min(100, int(getattr(args, "loader_page_size", 10))))
    offset = max(0, int(getattr(args, "source_start_offset", 0)))
    total_rows: int | None = None
    while total_rows is None or offset < total_rows:
        params = {
            "dataset": getattr(args, "dataset", SOURCE_DATASET), "config": "default", "split": getattr(args, "split", "train"),
            "where": where, "offset": offset, "length": page_size,
        }
        last_error: Exception | None = None
        for attempt in range(8):
            try:
                response = requests.get(endpoint, params=params, timeout=(30, 180))
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt == 7:
                    raise RuntimeError(f"Hugging Face filtered-page request failed at offset {offset}") from error
                time.sleep(min(30, 2 ** attempt))
        else:  # pragma: no cover
            raise RuntimeError("unreachable filtered-page retry state") from last_error
        reported_total = int(payload.get("num_rows_total") or 0)
        if total_rows is None:
            total_rows = reported_total
            setattr(args, "metadata_scope_row_count", total_rows)
            setattr(args, "source_filter_partial_flag", bool(payload.get("partial")))
            LOGGER.info("Hugging Face filtered scope rows: %s", total_rows)
        elif reported_total != total_rows:
            raise RuntimeError(f"Filtered source row count changed during scan: {total_rows} -> {reported_total}")
        rows = payload.get("rows") or []
        if not rows and offset < total_rows:
            raise RuntimeError(f"Filtered source returned an empty page before completion at offset {offset}")
        for relative_index, item in enumerate(rows):
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                row["_source_index"] = offset + relative_index
                yield row
        offset += len(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_kr_reference(path: Path) -> tuple[dict[str, int], list[str]]:
    rows = read_jsonl(path)
    distribution = dict(Counter(compact(row.get("case_subtype")) or "unclear" for row in rows))
    warnings = []
    if len(rows) != 50:
        warnings.append(f"reference_kr_record_count_is_{len(rows)}_not_50")
    if sum(distribution.values()) != 50:
        warnings.append("reference_kr_subtype_total_is_not_50")
    if distribution != EXPECTED_KR_DISTRIBUTION:
        warnings.append("reference_kr_distribution_differs_from_documented_distribution;file_distribution_used")
    return distribution, warnings


def minimum_targets(kr_distribution: dict[str, int]) -> dict[str, int]:
    return {subtype: max(count + 2, math.ceil(count * 1.5)) for subtype, count in kr_distribution.items() if subtype != "unclear"}


def _confidence_points(value: Any) -> int:
    return {"high": 12, "medium": 6, "low": 0}.get(compact(value).lower(), 0)


def base_shortlist_score(row: dict[str, Any], available: Counter[str], kr: dict[str, int]) -> float:
    subtype = str(row.get("case_subtype"))
    scarcity = 15 * (kr.get(subtype, 0) / max(1, available.get(subtype, 0)))
    harm_diversity = 2 * sum(bool(row.get(key)) for key in (
        "death_involved", "physical_injury_involved", "property_damage_involved", "emotional_harm_involved"
    ))
    return round(float(row.get("factual_sufficiency_score") or 0) + _confidence_points(row.get("main_opinion_confidence")) +
                 _confidence_points(row.get("governing_law_confidence")) + _confidence_points(row.get("claim_posture_confidence")) +
                 scarcity + harm_diversity, 3)


def build_shortlist(
    strict_pool: list[dict[str, Any]], *, shortlist_count: int, kr_distribution: dict[str, int], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [dict(row) for row in strict_pool if row.get("strict_eligible") and row.get("case_subtype") != "unclear"]
    available = Counter(str(row.get("case_subtype")) for row in eligible)
    targets = minimum_targets(kr_distribution)
    rng = random.Random(seed)
    tie = {str(row["case_id"]): rng.random() for row in sorted(eligible, key=lambda item: str(item["case_id"]))}
    for row in eligible:
        row["reference_kr_subtype_count"] = kr_distribution.get(str(row.get("case_subtype")), 0)
        row["shortlist_minimum_target"] = targets.get(str(row.get("case_subtype")), 0)
        row["shortlist_selection_score"] = base_shortlist_score(row, available, kr_distribution)

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (-float(row["shortlist_selection_score"]), tie[str(row["case_id"])], str(row["case_id"]))

    by_subtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_subtype[str(row["case_subtype"])].append(row)
    for values in by_subtype.values(): values.sort(key=rank_key)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for subtype, target in targets.items():
        for rank, row in enumerate(by_subtype.get(subtype, [])[:target], start=1):
            row["shortlist_subtype_rank"] = rank
            row["shortlist_overflow_candidate"] = False
            row["shortlist_selection_reasons"] = ["kr_reference_minimum_target", "strict_eligible", "subtype_rank"]
            selected.append(row); selected_ids.add(str(row["case_id"]))

    counts = Counter(str(row["case_subtype"]) for row in selected)
    procedure = Counter(str(row.get("procedural_posture")) for row in selected)
    district = Counter(str(row.get("appellate_district") or "unknown") for row in selected)
    decade = Counter(decade_bucket(row.get("decision_year")) for row in selected)
    publication = Counter(str(row.get("publication_status")) for row in selected)
    leftovers = [row for row in eligible if str(row["case_id"]) not in selected_ids]
    while len(selected) < shortlist_count and leftovers:
        scored = []
        for row in leftovers:
            subtype = str(row["case_subtype"])
            if counts[subtype] >= 25: continue
            diversity = (
                6 / (1 + procedure[str(row.get("procedural_posture"))]) +
                6 / (1 + district[str(row.get("appellate_district") or "unknown")]) +
                4 / (1 + decade[decade_bucket(row.get("decision_year"))]) +
                3 / (1 + publication[str(row.get("publication_status"))]) +
                8 / (1 + counts[subtype])
            )
            scored.append((-(float(row["shortlist_selection_score"]) + diversity), tie[str(row["case_id"])], str(row["case_id"]), row))
        if not scored: break
        row = min(scored)[3]
        subtype = str(row["case_subtype"])
        row["shortlist_subtype_rank"] = counts[subtype] + 1
        row["shortlist_overflow_candidate"] = True
        row["shortlist_selection_reasons"] = ["strict_eligible_overflow", "quality_score", "distribution_diversity"]
        selected.append(row); selected_ids.add(str(row["case_id"])); leftovers.remove(row)
        counts[subtype] += 1
        procedure[str(row.get("procedural_posture"))] += 1
        district[str(row.get("appellate_district") or "unknown")] += 1
        decade[decade_bucket(row.get("decision_year"))] += 1
        publication[str(row.get("publication_status"))] += 1

    selected.sort(key=lambda row: (str(row.get("case_subtype")), int(row.get("shortlist_subtype_rank") or 999), str(row["case_id"])))
    shortage = {subtype: {"minimum_target": target, "strict_available": available.get(subtype, 0), "shortage": max(0, target - available.get(subtype, 0))} for subtype, target in targets.items()}
    return selected, {"minimum_targets": targets, "shortage_report": shortage, "strict_available_by_subtype": dict(available)}


def decade_bucket(year: Any) -> str:
    try: value = int(year)
    except (TypeError, ValueError): return "unknown"
    return f"{value // 10 * 10}s"


def distribution(rows: Iterable[dict[str, Any]], key: str, *, transform=None) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        value = row.get(key)
        if transform: value = transform(value)
        counter[str(value if value not in (None, "") else "unknown")] += 1
    return dict(sorted(counter.items()))


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / name for key, name in OUTPUT_NAMES.items()}


def qc_value(value: Any) -> Any:
    if isinstance(value, (list, dict)): return json.dumps(value, ensure_ascii=False)
    if value is None: return ""
    return value


def write_qc(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows: writer.writerow({key: qc_value(row.get(key)) for key in QC_COLUMNS})


def write_manifest(path: Path, records: list[dict[str, Any]], shortlist_ids: set[str]) -> None:
    fields = ["case_id", "source_record_id", "case_name", "citation", "docket_number", "decision_date", "court_name",
              "claim_posture", "liability_basis", "case_subtype", "strict_eligible", "in_shortlist", "publication_status",
              "related_case_group_id", "raw_text_sha256", "collection_version"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in records:
            item = dict(row); item["in_shortlist"] = str(row.get("case_id")) in shortlist_ids; writer.writerow(item)


def write_alignment(path: Path, kr: dict[str, int], targets: dict[str, int], strict: list[dict[str, Any]], shortlist: list[dict[str, Any]]) -> None:
    strict_counts = Counter(str(row.get("case_subtype")) for row in strict)
    shortlist_counts = Counter(str(row.get("case_subtype")) for row in shortlist)
    fields = ["case_subtype", "kr_reference_count", "ca_shortlist_minimum_target", "ca_strict_available", "ca_shortlist_count", "shortage"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for subtype in kr:
            writer.writerow({"case_subtype": subtype, "kr_reference_count": kr[subtype],
                             "ca_shortlist_minimum_target": targets.get(subtype, 0), "ca_strict_available": strict_counts[subtype],
                             "ca_shortlist_count": shortlist_counts[subtype], "shortage": max(0, targets.get(subtype, 0) - strict_counts[subtype])})


def build_summary(
    strict: list[dict[str, Any]], shortlist: list[dict[str, Any]], scan: dict[str, Any], stage_counts: dict[str, int],
    kr: dict[str, int], shortlist_meta: dict[str, Any], duplicate_counts: dict[str, int], warnings: list[str], shortlist_count: int,
) -> dict[str, Any]:
    counts = {
        "total_scanned": scan["total_scanned"], "california_candidates": stage_counts["california_candidates"],
        "california_state_court_candidates": stage_counts["california_state_court_candidates"],
        "court_of_appeal_candidates": stage_counts["court_of_appeal_candidates"],
        "civil_candidates": stage_counts["civil_candidates"], "broad_tort_candidates": stage_counts["broad_tort_candidates"],
        "direct_tort_candidates": stage_counts["direct_tort_candidates"],
        "factually_sufficient_direct_tort_count": stage_counts["factually_sufficient_direct_tort_count"],
        "strict_eligible_pool_count": len(strict), "shortlist_requested_count": shortlist_count,
        "shortlist_actual_count": len(shortlist),
    }
    summary = {
        "collection_version": COLLECTION_VERSION, "source_dataset": SOURCE_DATASET,
        "source_schema": scan.get("source_schema", []), "opinion_structure": scan.get("opinion_structure", []),
        "scan": scan, "counts": counts,
        "excluded_claim_posture_counts": dict(sorted(stage_counts.get("excluded_claim_posture_counts", {}).items())),
        "duplicate_and_related_removal_counts": duplicate_counts,
        "strict_pool_distributions": {
            "case_subtype": distribution(strict, "case_subtype"), "publication_status": distribution(strict, "publication_status"),
            "procedural_posture": distribution(strict, "procedural_posture"), "appellate_district": distribution(strict, "appellate_district"),
            "decision_year": distribution(strict, "decision_year"), "decision_decade": distribution(strict, "decision_year", transform=decade_bucket),
        },
        "reference_kr_subtype_distribution": kr, "shortlist_minimum_targets": shortlist_meta.get("minimum_targets", {}),
        "shortage_report": shortlist_meta.get("shortage_report", {}),
        "shortlist_distributions": {
            "case_subtype": distribution(shortlist, "case_subtype"), "publication_status": distribution(shortlist, "publication_status"),
            "procedural_posture": distribution(shortlist, "procedural_posture"), "appellate_district": distribution(shortlist, "appellate_district"),
            "decision_year": distribution(shortlist, "decision_year"), "decision_decade": distribution(shortlist, "decision_year", transform=decade_bucket),
        },
        "warnings": list(warnings),
    }
    decades = Counter(decade_bucket(row.get("decision_year")) for row in shortlist)
    if shortlist and any(count / len(shortlist) >= 0.5 for decade, count in decades.items() if decade != "unknown"):
        summary["warnings"].append("one_decade_is_at_least_50_percent_of_shortlist")
    return summary


def chunk_paths(output_dir: Path, chunk_name: str) -> dict[str, Path]:
    base = output_dir / "chunks"
    prefix = f"ca_{chunk_name}"
    return {
        "court": base / f"{prefix}_court_candidates.jsonl",
        "direct": base / f"{prefix}_direct_tort_candidates.jsonl",
        "strict": base / f"{prefix}_strict_eligible.jsonl",
        "excluded": base / f"{prefix}_excluded.jsonl",
        "summary": base / f"{prefix}_summary.json",
        "checkpoint": base / f"{prefix}_checkpoint.json",
        "strict_pre": base / f".{prefix}_strict_pre.jsonl.incomplete",
    }


def lightweight_court_candidate(row: dict[str, Any], court: dict[str, Any]) -> dict[str, Any]:
    opinions = row.get("opinions") if isinstance(row.get("opinions"), list) else []
    opinion_inventory = []
    for opinion in opinions:
        if not isinstance(opinion, dict):
            continue
        text = _opinion_text(opinion)
        opinion_inventory.append({
            "opinion_id": compact(opinion.get("opinion_id")), "type": compact(opinion.get("type")),
            "normalized_type": normalize_opinion_type(opinion), "per_curiam": bool(opinion.get("per_curiam")),
            "text_available": bool(text), "text_length": len(text),
        })
    decision_date, decision_year = _date(row.get("date_filed"))
    return {
        "source_record_id": compact(row.get("id")), "case_name": _first(row, "case_name", "case_name_full", "case_name_short"),
        "decision_date": decision_date, "decision_year": decision_year, "court_name": court.get("court_name"),
        "court_system": court.get("court_system"), "court_level": court.get("court_level"),
        "court_level_confidence": court.get("court_level_confidence"), "court_evidence": court.get("court_evidence"),
        "court_jurisdiction": compact(row.get("court_jurisdiction")), "court_type": compact(row.get("court_type")),
        "citation": _citation(row), "docket_number": _docket(row, _metadata_text(row)),
        "opinion_count": len(opinions), "opinion_inventory": opinion_inventory,
        "collection_version": COLLECTION_VERSION,
    }


def _rss_bytes() -> int:
    if psutil is None:
        return 0
    return int(psutil.Process().memory_info().rss)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _config_signature(args: argparse.Namespace) -> str:
    return short_hash(
        COLLECTION_VERSION, getattr(args, "year_min", ""), getattr(args, "year_max", ""),
        getattr(args, "chunk_name", ""), getattr(args, "court_system", ""), getattr(args, "court_level", ""),
        getattr(args, "publication_status", ""), length=24,
    )


def _checkpoint_payload(
    args: argparse.Namespace, *, phase: str, next_source_index: int, counters: Counter[str],
    diagnostics: dict[str, Counter[str]], handles: dict[str, Any], completed: bool,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    offsets = {}
    paths = {}
    for key, handle in handles.items():
        handle.flush()
        offsets[key] = handle.tell()
        paths[key] = str(Path(handle.name))
    return {
        "collection_version": COLLECTION_VERSION, "config_signature": _config_signature(args),
        "year_min": args.year_min, "year_max": args.year_max, "chunk_name": args.chunk_name,
        "phase": phase, "last_source_index": next_source_index,
        "last_source_record_id": getattr(args, "last_source_record_id", None),
        "total_scanned": int(counters.get("total_scanned", 0)), "counters": dict(counters),
        "diagnostics": {key: dict(value) for key, value in diagnostics.items()},
        "partial_output_paths": paths, "partial_output_offsets": offsets,
        "peak_rss_bytes": peak_rss_bytes, "completed": completed,
    }


def _load_checkpoint(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("collection_version") != COLLECTION_VERSION or payload.get("config_signature") != _config_signature(args):
        raise RuntimeError("Checkpoint collection version or configuration does not match this run")
    return payload


def _truncate_to_checkpoint(payload: dict[str, Any]) -> None:
    for key, raw_path in (payload.get("partial_output_paths") or {}).items():
        path = Path(raw_path)
        offset = int((payload.get("partial_output_offsets") or {}).get(key, 0))
        if not path.exists():
            if offset:
                raise RuntimeError(f"Checkpoint output is missing: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        if path.stat().st_size < offset:
            raise RuntimeError(f"Checkpoint offset exceeds output size: {path}")
        with path.open("r+b") as handle:
            handle.truncate(offset)
        if offset:
            with path.open("rb") as handle:
                handle.seek(offset - 1)
                if handle.read(1) != b"\n":
                    raise RuntimeError(f"Checkpoint does not end at a complete JSONL record: {path}")


def _chunk_funnel_rates(funnel: Counter[str]) -> dict[str, float]:
    def rate(numerator: str, denominator: str) -> float:
        value = int(funnel.get(denominator, 0))
        return round(int(funnel.get(numerator, 0)) / value, 6) if value else 0.0
    return {
        "court_of_appeal_retention_rate": rate("court_of_appeal_candidates", "date_range_candidates"),
        "civil_retention_rate": rate("civil_candidates", "court_of_appeal_candidates"),
        "direct_tort_retention_rate": rate("direct_tort_candidates", "broad_tort_candidates"),
        "full_opinion_retention_rate": rate("full_main_opinion_candidates", "california_state_law_candidates"),
        "factual_sufficiency_retention_rate": rate("factually_sufficient_candidates", "full_main_opinion_candidates"),
    }


def _funnel_warnings(funnel: Counter[str], rates: dict[str, float]) -> list[str]:
    warnings = []
    labels = {
        "court_of_appeal_retention_rate": "court validation", "civil_retention_rate": "civil classification",
        "direct_tort_retention_rate": "direct tort classification", "full_opinion_retention_rate": "main opinion validation",
        "factual_sufficiency_retention_rate": "factual sufficiency",
    }
    for key, label in labels.items():
        if rates[key] < 0.1:
            warnings.append(f"Sharp funnel drop at {label}: retention={rates[key]:.1%}; inspect rules and source metadata")
    return warnings


def _strict_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "case_subtype": distribution(rows, "case_subtype"), "publication_status": distribution(rows, "publication_status"),
        "procedural_posture": distribution(rows, "procedural_posture"), "appellate_district": distribution(rows, "appellate_district"),
        "decision_year": distribution(rows, "decision_year"), "decision_decade": distribution(rows, "decision_year", transform=decade_bucket),
        "event_era": distribution(rows, "event_era"),
    }


def collect_chunk(args: argparse.Namespace) -> int:
    if args.year_min is None or args.year_max is None or not args.chunk_name:
        raise ValueError("Chunk collection requires --year-min, --year-max, and --chunk-name")
    if args.year_min > args.year_max:
        raise ValueError("--year-min must not exceed --year-max")
    paths = chunk_paths(args.output_dir, args.chunk_name)
    if args.dry_run:
        print(json.dumps({
            "action": "collect_chunk", "year_min": args.year_min, "year_max": args.year_max,
            "chunk_name": args.chunk_name, "streaming": True,
            "outputs": {key: str(value) for key, value in paths.items() if key != "strict_pre"},
        }, ensure_ascii=False, indent=2))
        return 0
    checkpoint_path = args.resume_from_checkpoint or paths["checkpoint"]
    resume = bool(args.resume or args.resume_from_checkpoint)
    required = [paths[key] for key in ("court", "direct", "strict", "excluded", "summary", "checkpoint")]
    checkpoint: dict[str, Any] | None = None
    if resume:
        checkpoint = _load_checkpoint(checkpoint_path, args)
        if checkpoint.get("completed"):
            LOGGER.info("Chunk %s is already complete; nothing to resume", args.chunk_name)
            return 0
        _truncate_to_checkpoint(checkpoint)
    else:
        require_outputs(required, args.overwrite)
        paths["court"].parent.mkdir(parents=True, exist_ok=True)
        if paths["strict_pre"].exists():
            paths["strict_pre"].unlink()

    phase = str(checkpoint.get("phase")) if checkpoint else "metadata"
    counters = Counter(checkpoint.get("counters") or {}) if checkpoint else Counter()
    diagnostics = {
        key: Counter((checkpoint.get("diagnostics") or {}).get(key, {}) if checkpoint else {})
        for key in ("court_name", "jurisdiction", "opinion_type", "primary_exclusion_reason")
    }
    peak_rss = max(int((checkpoint or {}).get("peak_rss_bytes") or 0), _rss_bytes())
    checkpoint_every = max(1, int(args.checkpoint_every))

    if phase == "metadata":
        start = int((checkpoint or {}).get("last_source_index") or 0)
        args.source_start_offset = start
        mode = "a" if resume and start else "w"
        with paths["court"].open(mode, encoding="utf-8", newline="\n") as court_handle:
            handles = {"court": court_handle}
            processed_since_checkpoint = 0
            for row in iter_rows(args):
                source_index = int(row.get("_source_index", start))
                args.last_source_record_id = compact(row.get("id"))
                counters["metadata_total_scanned"] += 1
                diagnostics["court_name"][_first(row, "court_full_name", "court_short_name") or "unknown"] += 1
                diagnostics["jurisdiction"][compact(row.get("court_jurisdiction")) or "unknown"] += 1
                for opinion in row.get("opinions") or []:
                    if isinstance(opinion, dict): diagnostics["opinion_type"][compact(opinion.get("type")) or "unknown"] += 1
                court = classify_court(row)
                if court["court_system"] == "california_state" and court["court_level"] == "intermediate_appellate":
                    court_handle.write(json.dumps(lightweight_court_candidate(row, court), ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
                    counters["metadata_court_candidates"] += 1
                processed_since_checkpoint += 1
                peak_rss = max(peak_rss, _rss_bytes())
                if processed_since_checkpoint >= checkpoint_every:
                    payload = _checkpoint_payload(args, phase="metadata", next_source_index=source_index + 1, counters=counters, diagnostics=diagnostics, handles=handles, completed=False, peak_rss_bytes=peak_rss)
                    _atomic_json(checkpoint_path, payload); processed_since_checkpoint = 0; gc.collect()
                del row, court
            payload = _checkpoint_payload(args, phase="full_opinion", next_source_index=0, counters=counters, diagnostics=diagnostics, handles=handles, completed=False, peak_rss_bytes=peak_rss)
            _atomic_json(checkpoint_path, payload)
        phase = "full_opinion"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    if phase == "full_opinion":
        start = int((checkpoint or {}).get("last_source_index") or 0)
        args.source_start_offset = start
        resume_full = resume and start > 0
        direct_mode = "a" if resume_full else "w"
        excluded_mode = "a" if resume_full else "w"
        pre_mode = "a" if resume_full else "w"
        with ExitStack() as stack:
            direct_handle = stack.enter_context(paths["direct"].open(direct_mode, encoding="utf-8", newline="\n"))
            excluded_handle = stack.enter_context(paths["excluded"].open(excluded_mode, encoding="utf-8", newline="\n"))
            pre_handle = stack.enter_context(paths["strict_pre"].open(pre_mode, encoding="utf-8", newline="\n"))
            handles = {"direct": direct_handle, "excluded": excluded_handle, "strict_pre": pre_handle}
            processed_since_checkpoint = 0
            for row in iter_rows(args):
                source_index = int(row.get("_source_index", start))
                args.last_source_record_id = compact(row.get("id"))
                counters["total_scanned"] += 1; counters["date_range_candidates"] += 1
                record = evaluate_row(dict(row), args)
                record["collection_chunk"] = args.chunk_name
                record["underlying_incident_fingerprint"] = underlying_incident_fingerprint(record)
                rendered = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")) + "\n"
                if classify_court(row)["california_candidate"]: counters["california_candidates"] += 1
                if record.get("court_system") == "california_state": counters["california_state_court_candidates"] += 1
                if record.get("court_level") == "intermediate_appellate": counters["court_of_appeal_candidates"] += 1
                if record.get("civil_candidate"): counters["civil_candidates"] += 1
                if record.get("civil_candidate") and record.get("broad_tort_candidate"): counters["broad_tort_candidates"] += 1
                if record.get("claim_posture") == "direct_tort_claim":
                    counters["direct_tort_candidates"] += 1; direct_handle.write(rendered)
                    if record.get("primary_governing_law") == "california_state_law": counters["california_state_law_candidates"] += 1
                    if record.get("primary_governing_law") == "california_state_law" and record.get("full_main_opinion_available"): counters["full_main_opinion_candidates"] += 1
                    if record.get("primary_governing_law") == "california_state_law" and record.get("full_main_opinion_available") and record.get("facts_independently_reconstructable"): counters["factually_sufficient_candidates"] += 1
                if record.get("strict_eligible"):
                    counters["strict_eligible_pre_dedup"] += 1; pre_handle.write(rendered)
                else:
                    excluded_handle.write(rendered)
                    diagnostics["primary_exclusion_reason"][str(record.get("primary_exclusion_reason") or "unclear")] += 1
                processed_since_checkpoint += 1
                peak_rss = max(peak_rss, _rss_bytes())
                if processed_since_checkpoint >= checkpoint_every:
                    payload = _checkpoint_payload(args, phase="full_opinion", next_source_index=source_index + 1, counters=counters, diagnostics=diagnostics, handles=handles, completed=False, peak_rss_bytes=peak_rss)
                    _atomic_json(checkpoint_path, payload); processed_since_checkpoint = 0; gc.collect()
                del row, record, rendered
            payload = _checkpoint_payload(args, phase="dedup_merge", next_source_index=int(getattr(args, "metadata_scope_row_count", counters["total_scanned"])), counters=counters, diagnostics=diagnostics, handles=handles, completed=False, peak_rss_bytes=peak_rss)
            _atomic_json(checkpoint_path, payload)

    pre_strict = read_jsonl(paths["strict_pre"]) if paths["strict_pre"].exists() else []
    duplicate_counts = mark_duplicates(pre_strict)
    strict = [row for row in pre_strict if row.get("strict_eligible")]
    removed = [row for row in pre_strict if not row.get("strict_eligible")]
    if removed:
        with paths["excluded"].open("a", encoding="utf-8", newline="\n") as handle:
            for row in removed:
                handle.write(json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
                diagnostics["primary_exclusion_reason"][str(row.get("primary_exclusion_reason") or "duplicate")] += 1
    counters["strict_eligible_candidates"] = len(strict)
    write_jsonl(paths["strict"], strict)
    rates = _chunk_funnel_rates(counters)
    summary = {
        "collection_version": COLLECTION_VERSION, "period": args.chunk_name,
        "year_min": args.year_min, "year_max": args.year_max, "source_dataset": SOURCE_DATASET,
        "scan_complete": True, "streaming": True, "funnel_counts": dict(counters),
        "retention_rates": rates, "warnings": _funnel_warnings(counters, rates),
        "exclusion_reason_counts": dict(diagnostics["primary_exclusion_reason"].most_common()),
        "court_metadata_top_100": diagnostics["court_name"].most_common(100),
        "jurisdiction_top_100": diagnostics["jurisdiction"].most_common(100),
        "opinion_type_frequencies": dict(diagnostics["opinion_type"].most_common()),
        "strict_pool_distributions": _strict_distributions(strict),
        "duplicate_and_related_removal_counts": duplicate_counts,
        "peak_memory_bytes": peak_rss, "peak_memory_mb": round(peak_rss / 1024 / 1024, 2),
        "checkpoint_path": str(checkpoint_path),
    }
    write_summary(paths["summary"], summary)
    completed_payload = _checkpoint_payload(args, phase="complete", next_source_index=int(counters.get("total_scanned", 0)), counters=counters, diagnostics=diagnostics, handles={}, completed=True, peak_rss_bytes=peak_rss)
    _atomic_json(checkpoint_path, completed_payload)
    if paths["strict_pre"].exists():
        paths["strict_pre"].unlink()
    LOGGER.info("chunk=%s scanned=%s strict=%s peak_mb=%.2f", args.chunk_name, counters["total_scanned"], len(strict), summary["peak_memory_mb"])
    return 0


def _completed_chunk_sets(output_dir: Path) -> list[dict[str, Path]]:
    chunk_dir = output_dir / "chunks"
    groups = []
    for checkpoint_path in sorted(chunk_dir.glob("ca_*_checkpoint.json")):
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not payload.get("completed"):
            continue
        name = str(payload.get("chunk_name"))
        paths = chunk_paths(output_dir, name)
        if all(paths[key].exists() for key in ("court", "direct", "strict", "excluded", "summary")):
            groups.append(paths)
    return groups


def _compact_for_merge(row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    if not row.get("normalized_main_opinion_sha256") and row.get("main_opinion_text"):
        row = dict(row)
        row["normalized_main_opinion_sha256"] = sha256_text(normalized_text_for_hash(str(row["main_opinion_text"])))
    compact_row = {key: value for key, value in row.items() if key != "main_opinion_text"}
    compact_row["_source_path"] = str(source_path)
    return compact_row


def _resolve_merged_strict(chunk_sets: list[dict[str, Path]]) -> tuple[list[dict[str, Any]], set[tuple[str, str]], dict[str, int]]:
    compact_rows = []
    for paths in chunk_sets:
        for row in read_jsonl(paths["strict"]):
            compact_rows.append(_compact_for_merge(row, paths["strict"]))
    compact_rows.sort(key=lambda row: (
        -int(row.get("factual_sufficiency_score") or 0), row.get("main_opinion_confidence") != "high",
        row.get("claim_posture_confidence") != "high", str(row.get("case_id")),
    ))
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    winners: list[dict[str, Any]] = []
    losers: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for row in compact_rows:
        duplicate_of = None
        reason = None
        keys = [
            ("source_record_id", compact(row.get("source_record_id")).lower()),
            ("docket_number", compact(row.get("docket_number")).lower()),
            ("citation", compact(row.get("citation")).lower()),
            ("exact_main_opinion_hash", compact(row.get("raw_text_sha256")).lower()),
            ("normalized_main_opinion_hash", compact(row.get("normalized_main_opinion_sha256")).lower()),
            ("related_underlying_incident", compact(row.get("underlying_incident_fingerprint")).lower()),
        ]
        for label, value in keys:
            if value and (label, value) in seen:
                duplicate_of, reason = seen[(label, value)], label
                break
        if duplicate_of:
            losers.add((str(row["_source_path"]), str(row["case_id"]))); counts[reason or "duplicate"] += 1
            row["strict_eligible"] = False
            row["duplicate_or_related_reason"] = reason
            row["exclusion_reasons"] = unique(list(row.get("exclusion_reasons") or []) + [reason or "duplicate"])
            assign_primary_exclusion(row)
            continue
        winners.append(row)
        for label, value in keys:
            if value: seen[(label, value)] = row
    return winners, losers, dict(counts)


def _stream_concat(inputs: list[Path], output: Path, *, extra_rows: Iterable[dict[str, Any]] = ()) -> None:
    temp = output.with_suffix(output.suffix + ".incomplete")
    if temp.exists(): temp.unlink()
    with temp.open("w", encoding="utf-8", newline="\n") as target:
        for path in inputs:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip(): target.write(line if line.endswith("\n") else line + "\n")
        for row in extra_rows:
            target.write(json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
    temp.replace(output)


def merge_chunks(args: argparse.Namespace) -> int:
    chunk_sets = _completed_chunk_sets(args.output_dir)
    if not chunk_sets:
        raise RuntimeError("No completed chunk checkpoints were found")
    if args.dry_run:
        print(json.dumps({
            "action": "merge_chunks", "completed_chunk_count": len(chunk_sets),
            "chunk_summaries": [str(group["summary"]) for group in chunk_sets],
            "output_dir": str(args.output_dir),
        }, ensure_ascii=False, indent=2))
        return 0
    paths = output_paths(args.output_dir)
    outputs = [paths[key] for key in ("appeal", "direct", "strict", "published", "unpublished", "excluded", "shortlist", "qc", "summary")]
    outputs += [args.manifest_output, args.alignment_output]
    require_outputs(outputs, args.overwrite)

    winner_meta, loser_ids, duplicate_counts = _resolve_merged_strict(chunk_sets)
    winner_occurrences = {(str(row["_source_path"]), str(row["case_id"])) for row in winner_meta}
    strict_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    for group in chunk_sets:
        for row in read_jsonl(group["strict"]):
            if not row.get("normalized_main_opinion_sha256"):
                row["normalized_main_opinion_sha256"] = sha256_text(normalized_text_for_hash(str(row.get("main_opinion_text") or "")))
            occurrence = (str(group["strict"]), str(row.get("case_id")))
            if occurrence in winner_occurrences:
                strict_rows.append(row)
            elif occurrence in loser_ids:
                row["strict_eligible"] = False
                row["duplicate_or_related_reason"] = "cross_chunk_duplicate_or_related"
                row["exclusion_reasons"] = unique(list(row.get("exclusion_reasons") or []) + ["cross_chunk_duplicate_or_related"])
                assign_primary_exclusion(row); removed_rows.append(row)

    _stream_concat([group["court"] for group in chunk_sets], paths["appeal"])
    _stream_concat([group["direct"] for group in chunk_sets], paths["direct"])
    _stream_concat([group["excluded"] for group in chunk_sets], paths["excluded"], extra_rows=removed_rows)
    write_jsonl(paths["strict"], strict_rows)
    write_jsonl(paths["published"], [row for row in strict_rows if row.get("publication_status") == "published"])
    write_jsonl(paths["unpublished"], [row for row in strict_rows if row.get("publication_status") == "unpublished"])

    kr_distribution, reference_warnings = read_kr_reference(args.reference_kr_final)
    shortlist_compact, shortlist_meta = build_shortlist(
        [_compact_for_merge(row, paths["strict"]) for row in strict_rows], shortlist_count=args.shortlist_count,
        kr_distribution=kr_distribution, seed=args.seed,
    )
    selection_by_id = {str(row["case_id"]): row for row in shortlist_compact}
    shortlist = []
    for row in strict_rows:
        selection = selection_by_id.get(str(row["case_id"]))
        if not selection: continue
        for key in (
            "reference_kr_subtype_count", "shortlist_minimum_target", "shortlist_subtype_rank",
            "shortlist_overflow_candidate", "shortlist_selection_score", "shortlist_selection_reasons",
        ):
            row[key] = selection.get(key)
        shortlist.append(row)
    shortlist.sort(key=lambda row: (str(row.get("case_subtype")), int(row.get("shortlist_subtype_rank") or 999), str(row["case_id"])))
    write_jsonl(paths["shortlist"], shortlist); write_qc(paths["qc"], shortlist)
    write_manifest(args.manifest_output, strict_rows, {str(row["case_id"]) for row in shortlist})
    write_alignment(args.alignment_output, kr_distribution, shortlist_meta["minimum_targets"], strict_rows, shortlist)

    chunk_summaries = [json.loads(group["summary"].read_text(encoding="utf-8")) for group in chunk_sets]
    merged_funnel: Counter[str] = Counter()
    merged_exclusions: Counter[str] = Counter()
    merged_courts: Counter[str] = Counter()
    merged_jurisdictions: Counter[str] = Counter()
    merged_opinion_types: Counter[str] = Counter()
    peak_memory = 0
    for summary in chunk_summaries:
        merged_funnel.update(summary.get("funnel_counts") or {})
        merged_exclusions.update(summary.get("exclusion_reason_counts") or {})
        merged_courts.update(dict(summary.get("court_metadata_top_100") or []))
        merged_jurisdictions.update(dict(summary.get("jurisdiction_top_100") or []))
        merged_opinion_types.update(summary.get("opinion_type_frequencies") or {})
        peak_memory = max(peak_memory, int(summary.get("peak_memory_bytes") or 0))
    summary = {
        "collection_version": COLLECTION_VERSION, "source_dataset": SOURCE_DATASET,
        "merged_chunks": [summary.get("period") for summary in chunk_summaries],
        "legacy_non_merged_files": [
            "ca_california_candidates_all.jsonl", "ca_state_court_candidates_all.jsonl", "ca_civil_candidates_all.jsonl"
        ],
        "funnel_counts": dict(merged_funnel), "retention_rates": _chunk_funnel_rates(merged_funnel),
        "warnings": unique(reference_warnings + _funnel_warnings(merged_funnel, _chunk_funnel_rates(merged_funnel))),
        "exclusion_reason_counts": dict(merged_exclusions.most_common()),
        "court_metadata_top_100": merged_courts.most_common(100),
        "jurisdiction_top_100": merged_jurisdictions.most_common(100),
        "opinion_type_frequencies": dict(merged_opinion_types.most_common()),
        "strict_eligible_pool_count": len(strict_rows), "strict_pool_distributions": _strict_distributions(strict_rows),
        "duplicate_and_related_removal_counts": duplicate_counts,
        "reference_kr_subtype_distribution": kr_distribution,
        "shortlist_requested_count": args.shortlist_count, "shortlist_actual_count": len(shortlist),
        "shortlist_minimum_targets": shortlist_meta["minimum_targets"], "shortage_report": shortlist_meta["shortage_report"],
        "shortlist_distributions": _strict_distributions(shortlist),
        "decision_year_distribution": distribution(shortlist, "decision_year"),
        "decision_decade_distribution": distribution(shortlist, "decision_year", transform=decade_bucket),
        "event_era_distribution": distribution(shortlist, "event_era"),
        "peak_chunk_memory_bytes": peak_memory, "peak_chunk_memory_mb": round(peak_memory / 1024 / 1024, 2),
    }
    decades = Counter(decade_bucket(row.get("decision_year")) for row in shortlist)
    if shortlist and any(count / len(shortlist) > 0.5 for decade, count in decades.items() if decade != "unknown"):
        summary["warnings"].append("More than 50% of shortlist records fall in the same decision decade")
    write_summary(paths["summary"], summary)
    LOGGER.info("merged_chunks=%s strict=%s shortlist=%s", len(chunk_sets), len(strict_rows), len(shortlist))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build a deterministic California direct-tort Court of Appeal pool and human-review shortlist.")
    value.add_argument("--export-all-candidates", action="store_true")
    value.add_argument("--build-shortlist", action="store_true")
    value.add_argument("--merge-chunks", action="store_true")
    value.add_argument("--shortlist-count", type=int, default=100)
    value.add_argument("--reference-kr-final", type=Path, default=DEFAULT_KR_REFERENCE)
    value.add_argument("--court-system", choices=["california-state"], default="california-state")
    value.add_argument("--court-level", choices=["intermediate-appellate"], default="intermediate-appellate")
    value.add_argument("--strict-direct-tort-only", action="store_true", default=True)
    value.add_argument("--publication-status", choices=["any", "published", "unpublished"], default="any")
    value.add_argument("--decision-date-from", default=DEFAULT_DECISION_DATE_FROM)
    value.add_argument("--year-min", type=int)
    value.add_argument("--year-max", type=int)
    value.add_argument("--chunk-name")
    value.add_argument("--streaming", action="store_true", default=True)
    value.add_argument("--checkpoint-every", type=int, default=10000)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--resume-from-checkpoint", type=Path)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=argparse.SUPPRESS)
    value.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST, help=argparse.SUPPRESS)
    value.add_argument("--alignment-output", type=Path, default=DEFAULT_ALIGNMENT, help=argparse.SUPPRESS)
    value.add_argument("--dataset", default=SOURCE_DATASET, help=argparse.SUPPRESS)
    value.add_argument("--split", default="train", help=argparse.SUPPRESS)
    value.add_argument("--local-arrow-dir", type=Path, help=argparse.SUPPRESS)
    value.add_argument("--min-opinion-chars", type=int, default=1800, help=argparse.SUPPRESS)
    value.add_argument("--loader-batch-size", type=int, default=32, help=argparse.SUPPRESS)
    value.add_argument("--loader-page-size", type=int, default=10, help=argparse.SUPPRESS)
    value.add_argument("--source-loader", choices=["datasets-server", "datasets"], default="datasets-server", help=argparse.SUPPRESS)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.merge_chunks:
        return merge_chunks(args)
    if args.year_min is not None or args.year_max is not None or args.chunk_name:
        return collect_chunk(args)
    if not args.export_all_candidates and not args.build_shortlist:
        args.export_all_candidates = True; args.build_shortlist = True
    paths = output_paths(args.output_dir)
    shortlist_outputs = [paths["shortlist"], paths["qc"], paths["summary"], args.alignment_output]
    export_outputs = [paths[key] for key in ("california", "state", "appeal", "civil", "direct", "strict", "published", "unpublished", "excluded")] + [args.manifest_output]
    if args.export_all_candidates:
        require_outputs(export_outputs, args.overwrite)
    if args.build_shortlist:
        require_outputs(shortlist_outputs, args.overwrite)
    if args.build_shortlist and not args.reference_kr_final.exists():
        raise FileNotFoundError(f"Korean reference file not found: {args.reference_kr_final}")

    total = 0
    stage_counts: dict[str, Any] = {
        "california_candidates": 0, "california_state_court_candidates": 0,
        "court_of_appeal_candidates": 0, "civil_candidates": 0, "broad_tort_candidates": 0,
        "direct_tort_candidates": 0, "factually_sufficient_direct_tort_count": 0,
        "excluded_claim_posture_counts": Counter(),
    }
    pre_strict: list[dict[str, Any]] = []
    streamed_temp_paths: dict[str, Path] = {}
    # Large candidate stages are streamed directly to disk. Only the much smaller
    # pre-dedup strict pool is retained for related-case resolution and shortlisting.
    with ExitStack() as stack:
        handles: dict[str, Any] = {}
        if args.export_all_candidates and not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for key in ("california", "state", "appeal", "civil", "direct", "excluded"):
                temp_path = paths[key].with_suffix(paths[key].suffix + ".incomplete")
                if temp_path.exists():
                    temp_path.unlink()
                streamed_temp_paths[key] = temp_path
                handles[key] = stack.enter_context(temp_path.open("w", encoding="utf-8", newline="\n"))

        def emit(key: str, rendered: str) -> None:
            if key in handles:
                handles[key].write(rendered)

        for row in iter_rows(args):
            total += 1
            record = evaluate_row(dict(row), args)
            rendered = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")) + "\n"
            stage_counts["california_candidates"] += 1
            emit("california", rendered)
            if total % 1000 == 0:
                for handle in handles.values():
                    handle.flush()
                LOGGER.info("California metadata rows scanned: %s; pre-strict retained: %s", total, len(pre_strict))
            if record.get("court_system") != "california_state":
                continue
            stage_counts["california_state_court_candidates"] += 1; emit("state", rendered)
            if record.get("court_level") != "intermediate_appellate":
                continue
            stage_counts["court_of_appeal_candidates"] += 1; emit("appeal", rendered)
            if not record.get("civil_candidate"):
                continue
            stage_counts["civil_candidates"] += 1; emit("civil", rendered)
            if record.get("broad_tort_candidate"):
                stage_counts["broad_tort_candidates"] += 1
            if record.get("broad_tort_candidate") and record.get("claim_posture") != "direct_tort_claim":
                stage_counts["excluded_claim_posture_counts"][str(record.get("claim_posture"))] += 1
                emit("excluded", rendered)
            if record.get("claim_posture") != "direct_tort_claim":
                continue
            stage_counts["direct_tort_candidates"] += 1; emit("direct", rendered)
            if record.get("factual_background_sufficient") and record.get("facts_independently_reconstructable"):
                stage_counts["factually_sufficient_direct_tort_count"] += 1
            if record.get("strict_eligible"):
                pre_strict.append(record)

    duplicate_counts = mark_duplicates(pre_strict)
    strict = [row for row in pre_strict if row.get("strict_eligible")]
    if args.publication_status != "any":
        strict = [row for row in strict if row.get("publication_status") == args.publication_status]

    kr_distribution: dict[str, int] = {}
    warnings: list[str] = []
    shortlist: list[dict[str, Any]] = []
    shortlist_meta: dict[str, Any] = {"minimum_targets": {}, "shortage_report": {}}
    if args.build_shortlist:
        kr_distribution, warnings = read_kr_reference(args.reference_kr_final)
        shortlist, shortlist_meta = build_shortlist(strict, shortlist_count=args.shortlist_count, kr_distribution=kr_distribution, seed=args.seed)
        for warning in warnings: LOGGER.warning(warning)

    scan = {
        "strategy": "bounded_page_filtered_index_scan" if args.source_loader == "datasets-server" else "bounded_batch_parquet_streaming",
        "scope": f"complete California Court of Appeal source range with date_filed >= {args.decision_date_from}",
        "full_source_scan": False,
        "date_filter_applied": True,
        "decision_date_from": args.decision_date_from,
        "metadata_scope_row_count_reported_by_hf_filter_api": getattr(args, "metadata_scope_row_count", None),
        "source_filter_partial_flag": getattr(args, "source_filter_partial_flag", None),
        "terminated_early": False,
        "first_n_or_scan_limit_used": False, "total_scanned": total,
        "loader_batch_size": args.loader_batch_size if args.source_loader == "datasets" else None,
        "loader_page_size": args.loader_page_size if args.source_loader == "datasets-server" else None,
        "loader_parallel_workers": 1,
        "memory_policy": "candidate stages streamed to disk; only pre-dedup strict candidates retained in memory",
        "source_total_records_reported_by_hf_size_api": 410807,
        "source_size_api_partial_flag": True,
        "source_readme_record_count_claim": 8300000,
        "source_size_metadata_warning": "live Hugging Face size API reports 410,807 rows while the dataset card describes 8.3 million decisions",
        "source_schema": ["arguments", "attorneys", "case_name", "case_name_full", "case_name_short", "citation_count", "citations", "correction", "court_full_name", "court_jurisdiction", "court_short_name", "court_type", "cross_reference", "date_filed", "date_filed_is_approximate", "disposition", "headmatter", "headnotes", "history", "id", "judges", "nature_of_suit", "opinions", "other_dates", "posture", "precedential_status", "slug", "summary", "syllabus"],
        "opinion_structure": ["author_id", "author_str", "download_url", "ocr", "opinion_id", "opinion_text", "page_count", "per_curiam", "type"],
    }
    summary = build_summary(strict, shortlist, scan, stage_counts, kr_distribution, shortlist_meta, duplicate_counts, warnings, args.shortlist_count)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.export_all_candidates:
        write_jsonl(paths["strict"], strict)
        write_jsonl(paths["published"], [row for row in strict if row.get("publication_status") == "published"])
        write_jsonl(paths["unpublished"], [row for row in strict if row.get("publication_status") == "unpublished"])
        write_manifest(args.manifest_output, strict, {str(row["case_id"]) for row in shortlist})
        for key, temp_path in streamed_temp_paths.items():
            temp_path.replace(paths[key])
    if args.build_shortlist:
        write_jsonl(paths["shortlist"], shortlist); write_qc(paths["qc"], shortlist)
        write_summary(paths["summary"], summary)
        write_alignment(args.alignment_output, kr_distribution, shortlist_meta["minimum_targets"], strict, shortlist)
    LOGGER.info("strict=%s shortlist=%s", len(strict), len(shortlist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
