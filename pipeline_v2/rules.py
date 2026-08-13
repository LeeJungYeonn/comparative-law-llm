from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterable

from .io_utils import normalize_for_grounding, normalized_whitespace, sha256_text, stable_id


DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})(?:일)?(?!\d)")
KR_CASE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*([가-힣]{1,5})\s*(\d{1,8})(?!\d)")
KR_SUPREME_CIVIL_CODES = {"다"}
KR_LOWER_CODES = {"가합", "가단", "가소", "나", "라"}
DATE_START, DATE_END = date(2000, 1, 1), date(2025, 12, 31)

DOMAIN_SIGNALS: dict[str, tuple[str, ...]] = {
    "medical_professional_liability": (
        r"손해배상\(의\)|의료(?:과오|사고)|의사|병원|수술|진료|투약|간호|medical malpractice|professional negligence|physician|hospital|surgery",
    ),
    "product_liability": (
        r"손해배상\(제\)|제조물책임|제품.{0,20}(?:결함|위험)|결함.{0,20}제품|product(?:s)? liability|strict products liability|defective product|failure to warn",
    ),
    "general_negligence_personal_injury": (
        r"손해배상\((?:자|산)\)|과실|주의의무|안전배려의무|교통사고|추락|상해|사망|자동차|추돌|충돌|보행자|negligen(?:ce|t|tly)|personal injury|wrongful death|premises liability|duty of care",
    ),
    "other_civil_liability": (
        r"손해배상|불법행위|위자료|명예훼손|프라이버시|일조권|부당이득|방해|구상금|"
        r"civil liability|civil damages|tort|defamation|privacy|nuisance|intentional infliction|conversion",
    ),
}

SECONDARY_TAG_SIGNALS: dict[str, str] = {
    "vicarious_liability": r"사용자책임|피용자.{0,40}업무집행|vicarious liability|scope of employment",
    "respondeat_superior": r"respondeat superior",
    "negligent_supervision": r"감독.{0,20}(?:과실|책임)|negligent (?:hiring|retention|supervision)",
    "premises_liability": r"공작물.{0,30}책임|영조물|점유자.{0,30}책임|premises liability",
    "wrongful_death": r"사망.{0,30}손해배상|유족|wrongful death",
    "comparative_fault": r"과실상계|과실비율|비교과실|comparative (?:fault|negligence)|contributory negligence",
    "intentional_misconduct": r"고의.{0,20}(?:불법행위|가해)|intentional (?:tort|misconduct)|willful misconduct",
    "punitive_or_multiple_damages_salient": r"징벌적 손해배상|배액배상|punitive damages|treble damages|multiple damages",
}

CORE_LIABILITY_RE = re.compile(
    r"손해배상|불법행위|제조물책임|사용자책임|의료(?:과오|사고)|과실|주의의무|상해|사망|"
    r"negligence|personal injury|wrongful death|medical malpractice|professional negligence|product(?:s)? liability|"
    r"defective product|failure to warn|premises liability|vicarious liability|respondeat superior|negligent supervision|compensatory damages",
    re.I,
)
EXCLUSION_PATTERNS: dict[str, re.Pattern[str]] = {
    "criminal_case": re.compile(r"피고인|형법|징역|벌금|공소사실|habeas|criminal defendant|conviction|sentence of imprisonment", re.I),
    "administrative_only": re.compile(r"행정처분|처분취소|영업정지|허가취소|administrative agency|judicial review of agency", re.I),
    "insurance_only": re.compile(r"보험금 청구|보험계약.{0,80}(?:면책|보상)|insurance coverage|duty to defend|policy exclusion", re.I),
    "contract_only": re.compile(r"대금청구|매매대금|공사대금|대여금|breach of contract|contract payment", re.I),
}

OPINION_TYPE_MAP = {
    "010combined": "combined",
    "015unamimous": "unanimous",  # spelling used in COLD Cases
    "015unanimous": "unanimous",
    "020lead": "lead",
    "025plurality": "plurality",
    "030concurrence": "concurrence",
    "040dissent": "dissent",
    "080onthemerits": "on_the_merits",
}
MAIN_PRIORITY = {"combined": 0, "unanimous": 1, "lead": 2, "on_the_merits": 3, "plurality": 8, "unknown": 9}

US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
REGION = {
    **{s: "Northeast" for s in ("Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania")},
    **{s: "Midwest" for s in ("Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota")},
    **{s: "South" for s in ("Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas")},
    **{s: "West" for s in ("Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington")},
}


def unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(value for value in values if value not in (None, "")))


def parse_date(value: object) -> tuple[str | None, str, list[str]]:
    raw = normalized_whitespace(value)
    match = DATE_RE.search(raw)
    if not match:
        iso = re.search(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", raw)
        match = iso
    if not match:
        return None, "low", []
    try:
        parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None, "low", [f"invalid_date:{match.group(0)}"]
    return parsed.isoformat(), "high", [f"parsed:{match.group(0)}"]


def date_in_window(value: object, start: str = "2000-01-01", end: str = "2025-12-31") -> bool:
    parsed, confidence, _ = parse_date(value)
    return bool(parsed and confidence == "high" and start <= parsed <= end)


def extract_kr_case_number(text: str, structured: object = None) -> dict[str, Any]:
    structured_text = normalized_whitespace(structured)
    if structured_text and (match := KR_CASE_RE.search(structured_text)):
        return {"case_number": normalized_whitespace(match.group(0)), "case_code": match.group(2), "case_number_confidence": "high", "case_number_evidence": ["structured_metadata"]}
    header = text[:3000]
    labelled = re.search(r"(?:사건|사건번호)\s*[:：]?\s*((?:19|20)\d{2}\s*[가-힣]{1,5}\s*\d{1,8})", header)
    if labelled and (match := KR_CASE_RE.search(labelled.group(1))):
        return {"case_number": normalized_whitespace(match.group(0)), "case_code": match.group(2), "case_number_confidence": "high", "case_number_evidence": ["labelled_header"]}
    return {"case_number": None, "case_code": None, "case_number_confidence": "low", "case_number_evidence": []}


def classify_kr_court(text: str, structured_court: object = None, structured_case_number: object = None) -> dict[str, Any]:
    case = extract_kr_case_number(text, structured_case_number)
    header = text[:5000]
    court = normalized_whitespace(structured_court)
    evidence: list[str] = []
    if court == "대법원" or "대한민국 대법원" in court:
        evidence.append(f"structured_court:{court}")
    if re.search(r"(?:^|\n)\s*대법원\s*(?:판결|결정)", header):
        evidence.append("explicit_supreme_header")
    if case["case_code"] in KR_SUPREME_CIVIL_CODES and case["case_number_confidence"] == "high":
        evidence.append(f"current_case_code:{case['case_code']}")
    disposition = []
    if re.search(r"(?:상고|재상고)를 (?:기각|각하)한다|원심판결을 (?:파기|일부 파기)", header):
        disposition.append("supreme_disposition")
    if re.search(r"상고이유(?:를|에 대하여)|원심판결 이유에 의하면", header):
        disposition.append("supreme_reasoning_marker")
    evidence.extend(disposition)
    lower = []
    if court and any(token in court for token in ("고등법원", "지방법원", "가정법원")):
        lower.append(f"structured_lower_court:{court}")
    if case["case_code"] in KR_LOWER_CODES and case["case_number_confidence"] == "high":
        lower.append(f"current_lower_code:{case['case_code']}")
    if lower:
        level, confidence = "lower", "high"
    elif ("structured_court:" in " ".join(evidence) or "explicit_supreme_header" in evidence or any(x.startswith("current_case_code") for x in evidence)) and len(evidence) >= 2:
        level, confidence = "supreme", "high"
    elif len(disposition) >= 2:
        level, confidence = "supreme", "medium"
    else:
        level, confidence = "unknown", "low"
    return {"court_level": level, "court_level_confidence": confidence, "court_level_evidence": unique(evidence + lower), **case}


def extract_kr_decision_date(text: str, structured: object = None) -> dict[str, Any]:
    if normalized_whitespace(structured):
        parsed, confidence, evidence = parse_date(structured)
        if parsed:
            return {"decision_date": parsed, "decision_date_confidence": confidence, "decision_date_evidence": ["structured_metadata", *evidence]}
    header = text[:3000]
    match = re.search(r"(?:판결)?선고\s*[:：]?\s*((?:19|20)\d{2}[.\-/년\s]+\d{1,2}[.\-/월\s]+\d{1,2}(?:일)?)", header)
    if match:
        parsed, confidence, evidence = parse_date(match.group(1))
        return {"decision_date": parsed, "decision_date_confidence": confidence, "decision_date_evidence": ["explicit_decision_header", *evidence]}
    return {"decision_date": None, "decision_date_confidence": "low", "decision_date_evidence": []}


def classify_domain(text: str) -> dict[str, Any]:
    scores: Counter[str] = Counter()
    evidence: dict[str, list[str]] = defaultdict(list)
    sample = text[:60000]
    for domain, patterns in DOMAIN_SIGNALS.items():
        for pattern in patterns:
            hits = list(re.finditer(pattern, sample, re.I))
            if hits:
                # Medical and product signals outrank generic injury vocabulary.
                weight = 2 if domain in {"medical_professional_liability", "product_liability"} else 1
                scores[domain] += min(len(hits), 5) * weight
                evidence[domain].extend(normalized_whitespace(hit.group(0)) for hit in hits[:4])
    if not scores:
        domain = "other_civil_liability"
        return {"primary_domain": domain, "case_domain": domain, "case_domain_confidence": "low", "case_domain_evidence": [], "liability_theories": [], "secondary_tags": []}
    domain, score = sorted(scores.items(), key=lambda item: (-item[1], list(DOMAIN_SIGNALS).index(item[0])))[0]
    second = scores.most_common(2)[1][1] if len(scores) > 1 else 0
    confidence = "high" if score >= 3 and score >= second + 2 else "medium" if score >= 2 else "low"
    tags = [tag for tag, pattern in SECONDARY_TAG_SIGNALS.items() if re.search(pattern, sample, re.I)]
    return {
        "primary_domain": domain, "case_domain": domain, "case_domain_confidence": confidence,
        "case_domain_evidence": unique(evidence[domain])[:8], "liability_theories": tags,
        "secondary_tags": tags,
    }


FACT_PATTERNS = {
    "fact_has_parties": (r"\[(?:PERSON|COMPANY|ORGANIZATION|INSTITUTION)_[A-Z]+\]|원고|피고|피해자|가해자|환자|의사|회사|employee|employer|plaintiff|defendant|patient|manufacturer",),
    "fact_has_conduct": (r"하였다|하지 아니|운전|수술|제조|경고|설치|관리|failed to|drove|performed|manufactured|warned|supervised",),
    "fact_has_context": (r"\[(?:LOCATION|PROPERTY|INSTITUTION)_[A-Z]+\]|병원|도로|건물|사업장|현장|장소|hospital|road|premises|workplace|facility|at the",),
    "fact_has_timeline": (r"그 후|이후|당시|먼저|다음|\b(?:19|20)\d{2}\b|before|after|then|subsequently|when",),
    "fact_has_harm": (r"상해|사망|손상|장해|치료|손해|injur|death|died|damage|harm|loss",),
    "fact_has_causation": (r"인하여|때문에|결과|발생|초래|caused|resulted|because|leading to",),
    "fact_has_defense_context": (r"주장|다투|경고|동의|거부|알고|위험|alleg|disput|warn|consent|assum|comparative|defen",),
}


def assess_fact_sufficiency(text: str) -> dict[str, Any]:
    sample = text[:80000]
    values = {key: any(re.search(pattern, sample, re.I) for pattern in patterns) for key, patterns in FACT_PATTERNS.items()}
    score = sum(values.values())
    mandatory = ("fact_has_parties", "fact_has_conduct", "fact_has_harm", "fact_has_causation")
    missing = [key for key in mandatory if not values[key]]
    core = not missing
    return {
        **values,
        "fact_has_causal_sequence": values["fact_has_causation"],
        "mandatory_fact_dimensions": {key: values[key] for key in mandatory},
        "missing_mandatory_fact_dimensions": missing,
        "core_fact_sufficient": core,
        "fact_sufficiency_score": score,
        "preferred_fact_sufficiency": core and score >= 5,
        # Backward-compatible name; eligibility now means the mandatory four, not 5/7.
        "factual_background_sufficient": core,
    }


def normalize_opinion_type(value: object, per_curiam: bool = False) -> str:
    raw = re.sub(r"[^a-z0-9]", "", normalized_whitespace(value).lower())
    if per_curiam:
        return "on_the_merits"
    if raw in OPINION_TYPE_MAP:
        return OPINION_TYPE_MAP[raw]
    if "dissent" in raw:
        return "dissent"
    if "concurr" in raw:
        return "concurrence"
    if "plural" in raw:
        return "plurality"
    if "combined" in raw or "major" in raw:
        return "combined"
    if "unanim" in raw:
        return "unanimous"
    if "lead" in raw:
        return "lead"
    if "merit" in raw:
        return "on_the_merits"
    return "unknown"


def select_main_opinion(row: dict[str, Any], minimum_chars: int = 1200) -> dict[str, Any]:
    opinions = row.get("opinions") if isinstance(row.get("opinions"), list) else []
    candidates: list[tuple[int, int, str, str, str]] = []
    separate: list[dict[str, Any]] = []
    has_concurrence = has_dissent = False
    for opinion in opinions:
        if not isinstance(opinion, dict):
            continue
        kind = normalize_opinion_type(opinion.get("type"), bool(opinion.get("per_curiam")))
        text = normalized_whitespace(opinion.get("opinion_text") or opinion.get("text") or opinion.get("ocr"))
        opinion_id = normalized_whitespace(opinion.get("opinion_id"))
        if kind == "concurrence":
            has_concurrence = True
        if kind == "dissent":
            has_dissent = True
        if kind in {"concurrence", "dissent"}:
            separate.append({"opinion_id": opinion_id, "opinion_type": kind, "text_chars": len(text), "text_sha256": sha256_text(text) if text else ""})
        elif text:
            candidates.append((MAIN_PRIORITY.get(kind, 9), -len(text), kind, text, opinion_id))
    if not candidates:
        return {"main_opinion_text": "", "main_opinion_type": "unknown", "opinion_selection_reason": "no_nonseparate_opinion", "has_concurrence": has_concurrence, "has_dissent": has_dissent, "separate_opinions": separate, "main_opinion_usable": False}
    priority, _, kind, text, opinion_id = sorted(candidates)[0]
    plurality = kind == "plurality"
    usable = len(text) >= minimum_chars and not plurality and kind != "unknown"
    reason = f"priority={priority};type={kind};opinion_id={opinion_id};longest_within_priority"
    if plurality:
        reason += ";plurality_requires_human_review"
    return {"main_opinion_text": text, "main_opinion_type": kind, "opinion_selection_reason": reason, "has_concurrence": has_concurrence, "has_dissent": has_dissent, "separate_opinions": separate, "main_opinion_usable": usable}


def is_us_state_highcourt(row: dict[str, Any]) -> tuple[bool, list[str]]:
    court_type = normalized_whitespace(row.get("court_type")).upper()
    jurisdiction = normalized_whitespace(row.get("court_jurisdiction"))
    court_name = normalized_whitespace(row.get("court_full_name") or row.get("court_short_name"))
    evidence = [f"court_type={court_type}", f"court_jurisdiction={jurisdiction}"]
    federal = bool(re.search(r"United States|U\.S\.|Federal|Circuit|District Court", f"{jurisdiction} {court_name}", re.I))
    return court_type == "S" and not federal, evidence + (["federal_name_signal"] if federal else [])


def state_from_jurisdiction(value: object) -> str | None:
    raw = normalized_whitespace(value)
    for state, abbreviation in US_STATES.items():
        if raw.casefold() in {state.casefold(), f"{state}, {abbreviation}".casefold(), abbreviation.casefold()} or raw.casefold().startswith(state.casefold() + ","):
            return state
    return None


def civil_liability_candidate(text: str) -> tuple[bool, list[str], list[str]]:
    include = unique(match.group(0) for match in CORE_LIABILITY_RE.finditer(text[:70000]))
    exclusions = [label for label, pattern in EXCLUSION_PATTERNS.items() if pattern.search(text[:30000])]
    # A direct liability signal may override incidental contract/insurance vocabulary,
    # but criminal and administrative-only records remain excluded.
    fatal = [label for label in exclusions if label in {"criminal_case", "administrative_only"}]
    return bool(include) and not fatal, include[:12], exclusions


def duplicate_family_id(row: dict[str, Any]) -> str:
    citation = normalized_whitespace(row.get("case_number") or row.get("citation") or row.get("docket_number"))
    source = normalized_whitespace(row.get("source_record_id") or row.get("id"))
    if citation:
        return stable_id("FAM", citation.casefold(), length=16)
    return stable_id("FAM", source, row.get("raw_text_sha256", ""), length=16)


def source_span_grounding(source: str, span: str) -> tuple[str, int | None, int | None]:
    if not span.strip():
        return "fail", None, None
    exact = source.find(span)
    if exact >= 0:
        return "pass", exact, exact + len(span)
    flexible = r"\s+".join(re.escape(part) for part in span.split())
    if flexible and (match := re.search(flexible, source, re.I)):
        return "pass", match.start(), match.end()
    normalized_source = normalize_for_grounding(source)
    normalized_span = normalize_for_grounding(span)
    if normalized_span and normalized_span in normalized_source:
        return "pass", None, None
    return "fail", None, None


PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|GROUP|COMPANY|ORGANIZATION|PRODUCT|VEHICLE|LOCATION|PROPERTY|INSTITUTION|CURRENCY_AMOUNT)_[A-Z]+\]")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?\d+(?:[.,]\d+)?")
NEGATION_KO = re.compile(r"않|아니|없|못")
NEGATION_EN = re.compile(
    r"\b(?:not|no|never|without|didn't|did not|couldn't|could not|stop(?:ped|ping)?|ceas(?:e|ed|ing)|"
    r"fail(?:ed|ing)?|lack(?:ed|ing)?|omit(?:ted|ting)?|mis(?:read(?:ing)?|diagnos(?:e|ed|is)|interpret(?:ed|ation)?))\b",
    re.I,
)
EN_NUMBER_WORDS = {
    "zero": "0", "one": "1", "first": "1", "two": "2", "second": "2", "three": "3", "third": "3",
    "four": "4", "fourth": "4", "five": "5", "fifth": "5", "six": "6", "sixth": "6",
    "seven": "7", "seventh": "7", "eight": "8", "eighth": "8", "nine": "9", "ninth": "9",
    "ten": "10", "tenth": "10", "eleven": "11", "eleventh": "11", "twelve": "12", "twelfth": "12",
}
EN_MONTHS = {
    "January": "1", "February": "2", "March": "3", "April": "4", "May": "5", "June": "6",
    "July": "7", "August": "8", "September": "9", "October": "10", "November": "11", "December": "12",
}


def numeric_concepts(text: str) -> Counter[str]:
    """Normalize Arabic numerals, English number words, ordinals, and month names."""
    values = list(NUMBER_RE.findall(text))
    word_matches = re.findall(
        r"\b(?:" + "|".join(EN_NUMBER_WORDS) + r")\b", text, re.I
    )
    values.extend(EN_NUMBER_WORDS[word.lower()] for word in word_matches)
    for month, value in EN_MONTHS.items():
        values.extend(value for _ in re.finditer(rf"\b{month}\b", text))
    # Korean counters often spell one as 한 while English renders it as one.
    korean_one_counter = re.compile(
        r"한쪽|한\s+(?:방향|대|명|개|곳|차례|번)(?:에는?|으로|에서|은|을|를|이|가|의|와|과|로|만|도)?(?=\s|$|[.,])"
    )
    values.extend("1" for _ in korean_one_counter.finditer(text))
    korean_counter_words = {
        "1": r"(?<![가-힣])한\s+(?:명|개|대|곳|차례|번)(?:에는?|으로|에서|은|을|를|이|가|의|와|과|로|만|도)?(?=\s|$|[.,])",
        "2": r"(?<![가-힣])두\s+(?:명|개|대|곳|차례|번)(?:에는?|으로|에서|은|을|를|이|가|의|와|과|로|만|도)?(?=\s|$|[.,])",
        "3": r"(?<![가-힣])세\s+(?:명|개|대|곳|차례|번)(?:에는?|으로|에서|은|을|를|이|가|의|와|과|로|만|도)?(?=\s|$|[.,])",
        "4": r"(?<![가-힣])네\s+(?:명|개|대|곳|차례|번)(?:에는?|으로|에서|은|을|를|이|가|의|와|과|로|만|도)?(?=\s|$|[.,])",
    }
    for value, pattern in korean_counter_words.items():
        values.extend(value for _ in re.finditer(pattern, text))
    # Avoid double-counting the legacy one-counter rule above.
    if any(re.finditer(korean_counter_words["1"], text)):
        legacy_ones = len(korean_one_counter.findall(text))
        for _ in range(min(legacy_ones, values.count("1"))):
            values.remove("1")
    # "A and six others" denotes seven people/entities, not the number six.
    for match in re.finditer(r"\band\s+(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+others\b", text, re.I):
        other = EN_NUMBER_WORDS[match.group(1).lower()]
        try:
            values.remove(other)
        except ValueError:
            pass
        values.append(str(int(other) + 1))
    return Counter(values)


def translation_equivalence_checks(source: str, translated: str, source_language: str) -> dict[str, Any]:
    # Repetition can legitimately change with pronouns or possessives; entity identity may not.
    source_placeholders = set(PLACEHOLDER_RE.findall(source))
    translated_placeholders = set(PLACEHOLDER_RE.findall(translated))
    source_numbers = numeric_concepts(source)
    translated_numbers = numeric_concepts(translated)
    source_neg = len((NEGATION_KO if source_language == "ko" else NEGATION_EN).findall(source))
    translated_neg = len((NEGATION_EN if source_language == "ko" else NEGATION_KO).findall(translated))
    unit_patterns = {
        "km": r"킬로미터|kilomet(?:er|re)s?|\bkm\b", "m": r"(?<!킬로)미터|(?<!kilo)met(?:er|re)s?|(?<!k)\bm\b",
        "kg": r"킬로그램|kilograms?|\bkg\b", "celsius": r"섭씨|degrees? celsius|°\s*c\b",
    }
    source_units = Counter(key for key, pattern in unit_patterns.items() for _ in re.finditer(pattern, source, re.I))
    translated_units = Counter(key for key, pattern in unit_patterns.items() for _ in re.finditer(pattern, translated, re.I))
    issues = []
    warnings = []
    if source_placeholders != translated_placeholders:
        issues.append("placeholder_mismatch")
    if source_numbers != translated_numbers:
        warnings.append("number_mismatch")
    if source_units != translated_units:
        warnings.append("unit_mismatch")
    if bool(source_neg) != bool(translated_neg):
        warnings.append("negation_presence_mismatch")
    return {
        "translation_equivalence_status": "pass" if not issues else "fail",
        "translation_equivalence_issues": issues,
        "translation_equivalence_warnings": warnings,
    }


KR_LEGAL_LEAK = re.compile(r"민법\s*제?\d+조|대법원|원심판결|상고|판결|불법행위책임이 성립|책임을 인정|과실비율", re.I)
EN_LEGAL_LEAK = re.compile(
    r"Cal\. Civ\. Code|Restatement|Supreme Court|Court of Appeals|held that|as a matter of law|summary judgment|"
    r"\baffirmed\b|\breversed\b|\b(?:default\s+|final\s+)?judgment\b|"
    r"\b\d+\s+(?:U\.S\.|S\.\s?Ct\.|F\.\s?(?:2d|3d|4th)|A\.\s?(?:2d|3d)|N\.E\.\s?(?:2d|3d)|N\.W\.\s?(?:2d|3d)|S\.E\.\s?(?:2d|3d)|S\.W\.\s?(?:2d|3d)|P\.\s?(?:2d|3d))\s+\d+\b",
    re.I,
)
JURISDICTION_LEAK = re.compile(
    r"대한민국|한국|미국|(?<![A-Za-z])(?:Korea|Korean|United States|U\.S\.|"
    + "|".join(re.escape(state) for state in sorted(US_STATES, key=len, reverse=True))
    + r")(?![A-Za-z])",
    re.I,
)
STATE_ABBR_LEAK = re.compile(r"\b(?:" + "|".join(sorted(set(US_STATES.values()))) + r")\b")
GROUP_IDENTITY = re.compile(r"\b(?:African[- ]American|American)s?\b|아프리카계\s*미국인", re.I)


def state_abbreviation_matches(text: str) -> list[re.Match[str]]:
    return [
        match for match in STATE_ABBR_LEAK.finditer(text)
        if not (
            match.group(0) == "CT" and (
                re.match(r"\s+(?:scan|촬영|검사)\b", text[match.end():], re.I)
                or re.search(r"(?:brain|뇌)\s*$", text[max(0, match.start() - 12):match.start()], re.I)
            )
        )
    ]


def neutralize_jurisdiction_signals(text: str) -> tuple[str, list[str]]:
    """Replace explicit jurisdiction identity in neutral text while preserving source spans."""
    group_evidence = [match.group(0) for match in GROUP_IDENTITY.finditer(text)]
    text = GROUP_IDENTITY.sub("[GROUP_A]", text)
    abbreviation_matches = state_abbreviation_matches(text)
    evidence = unique([
        *group_evidence,
        *(match.group(0) for match in JURISDICTION_LEAK.finditer(text)),
        *(match.group(0) for match in abbreviation_matches),
    ])
    neutralized = text
    for match in reversed(abbreviation_matches):
        neutralized = neutralized[:match.start()] + "[LOCATION_JURISDICTION]" + neutralized[match.end():]
    neutralized = JURISDICTION_LEAK.sub("[LOCATION_JURISDICTION]", neutralized)
    return neutralized, evidence


def leakage_checks(text: str) -> dict[str, Any]:
    legal = unique(match.group(0) for pattern in (KR_LEGAL_LEAK, EN_LEGAL_LEAK) for match in pattern.finditer(text))
    jurisdiction = unique([*(match.group(0) for match in JURISDICTION_LEAK.finditer(text)), *(match.group(0) for match in state_abbreviation_matches(text))])
    return {
        "legal_leakage_status": "pass" if not legal else "fail",
        "legal_leakage_evidence": legal,
        "jurisdiction_leakage_status": "pass" if not jurisdiction else "fail",
        "jurisdiction_leakage_evidence": jurisdiction,
    }


def strip_legal_citations(text: str) -> str:
    """Conservatively remove citation strings, not surrounding factual prose."""
    patterns = (
        r"민법\s*제?\s*\d+\s*조(?:의\s*\d+)?",
        r"대법원\s*(?:19|20)\d{2}[.\s/-]+\d{1,2}[.\s/-]+\d{1,2}[.]?\s*선고\s*(?:19|20)\d{2}[가-힣]+\d+\s*판결",
        r"\b\d+\s+[A-Z][A-Za-z0-9. ]{0,18}\s+\d+(?:\s*\([^)]+\))?",
        r"\b(?:19|20)\d{2}\s+[A-Z][A-Za-z.]+\s+LEXIS\s+\d+\b",
    )
    result = text
    for pattern in patterns:
        result = re.sub(pattern, " ", result, flags=re.I)
    return normalized_whitespace(result)
