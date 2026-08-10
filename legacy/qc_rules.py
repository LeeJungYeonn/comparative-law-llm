"""Deterministic Korean direct-tort QC rules.

The rules deliberately prefer false negatives to silently admitting contract,
insurance-recovery, enforcement, or fact-poor appellate documents.  They do not
call an LLM or any external API.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Iterable

from pipeline.stage1_raw import compact, normalized_text_for_hash, sha256_text, unique
from pipeline.text_utils import normalize_whitespace


TEXT_COLUMNS = ("precedent", "raw_text", "text", "facts", "reason", "ruling")
CASE_NUMBER_COLUMNS = ("case_number_or_citation", "case_number", "case_no", "caseno", "사건번호")
COURT_COLUMNS = ("court_name", "court", "court_full_name", "법원명")
DATE_COLUMNS = ("decision_date", "선고일자", "판결선고일", "date_filed")
CASE_NAME_COLUMNS = ("case_name", "casename", "사건명")
CASE_TYPE_COLUMNS = ("case_type", "case_type_keyword", "사건종류", "사건명")

CASE_NUMBER_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})\s*([가-힣]{1,6})\s*(\d{1,8})(?!\d)")
DATE_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})(?:일)?\.?(?!\d)")
COURT_RE = re.compile(r"([가-힣]{0,14}(?:대법원|고등법원|특허법원|지방법원|가정법원|행정법원)(?:\s*[가-힣]+지원)?(?:\s*(?:민사)?항소부)?)")

BROAD_PATTERNS = (
    r"손해배상", r"불법행위", r"민법\s*제?\s*(?:750|751|756|758)\s*조", r"과실",
    r"주의의무", r"안전배려의무", r"사용자책임", r"공작물책임", r"제조물책임",
    r"의료(?:과오|사고)", r"교통사고", r"추돌|충돌", r"상해|사망|후유장해",
    r"치료비|일실수입|위자료", r"재산상\s*손해|정신적\s*손해", r"명예훼손|개인정보\s*침해",
    r"구상금|보험자대위|국가배상",
)

CURRENT_APPELLATE_STRONG = (
    ("appeal_purpose", r"(?:청구취지\s*및\s*)?항소취지"),
    ("cross_appeal_purpose", r"부대항소취지"),
    ("plaintiff_appeal", r"원고(?:의|가 한)?\s*항소"),
    ("defendant_appeal", r"피고(?:의|가 한)?\s*항소"),
    ("cancel_trial_part", r"제1심\s*판결(?:\s*중)?[^.\n]{0,80}(?:취소|변경)한다"),
    ("change_trial", r"제1심\s*판결을[^.\n]{0,50}(?:변경|취소)한다"),
    ("dismiss_appeal", r"(?:항소|부대항소)를\s*(?:모두\s*)?기각한다"),
    ("appeal_costs", r"항소비용"),
    ("cross_appeal", r"부대항소"),
)
CURRENT_APPELLATE_AUX = (
    ("appellant", r"항소인"), ("appellee", r"피항소인"),
    ("trial_judgment", r"제1심\s*판결"), ("trial_court", r"제1심\s*법원"),
)
SUPREME_PATTERNS = (
    ("appellant_supreme", r"상고인"), ("appellee_supreme", r"피상고인"),
    ("grounds_supreme", r"상고이유"), ("reverse_remand", r"원심판결을\s*파기|파기환송|파기\s*환송"),
)

INCORPORATION_PATTERNS = (
    r"이 법원이 이 사건에 관하여 적을 이유는 제1심판결 이유와 같으므로 이를 그대로 인용",
    r"제1심판결의 해당 부분을 그대로 인용",
    r"민사소송법 제420조 본문에 (?:따라|의하여) 이를 (?:그대로 )?인용",
    r"(?:일부 내용을 .* 외에는|아래와 같이 .* 외에는) 제1심판결의? 이유[^.]{0,100}(?:그대로 )?인용",
)

POSTURE_PATTERNS: dict[str, tuple[str, ...]] = {
    "judgment_enforcement": (r"청구이의", r"강제집행(?:을|의)?\s*불허", r"강제집행정지", r"간접강제", r"집행문", r"배당이의", r"확정판결.*집행"),
    "wage_or_compensation": (r"미지급\s*임금", r"임금\s*(?:지급|체불|청구)", r"퇴직금", r"성과급", r"조합(?:장|원|의)?.{0,20}(?:보수|급여)", r"약정\s*보수", r"총무.{0,20}보수", r"근로기준법"),
    "insurer_subrogation": (r"보험자대위", r"보험금[^.]{0,80}지급[^.]{0,80}(?:구상|대위)", r"(?:보험회사|보험자|공제조합)[^\n.]{0,100}구상금", r"구상금[^\n.]{0,100}(?:보험회사|보험자|공제조합)"),
    "joint_tortfeasor_contribution": (r"공동불법행위자[^.]{0,100}(?:구상|부담)", r"내부\s*부담(?:비율|부분)", r"부담부분을\s*초과", r"손해배상금[^.]{0,100}(?:지급|변제)[^.]{0,100}구상", r"구상금[^.]{0,120}공동"),
    "insurance_coverage": (r"보험약관", r"보험금\s*지급의무", r"면책(?:사유|조항)", r"보험계약[^.]{0,100}(?:보상|담보|범위)"),
    "contract_or_payment": (r"대여금", r"증여(?:받|하였|계약)", r"매매대금|공사대금|용역대금|계약금|보증금|투자금|사업자금|약정금", r"매매계약|도급계약|임대차|대여계약", r"채무불이행|계약(?:의)?\s*(?:위반|해제|해지)", r"동업관계|정산금"),
    "property_or_title_dispute": (r"소유권이전등기", r"소유권확인", r"명의신탁", r"경계확정", r"점유취득시효"),
    "family_or_domestic_dispute": (r"이혼|친권|양육비|재산분할|가정법원"),
    "administrative_or_state_liability": (r"국가배상", r"행정처분|처분취소|거부처분|영업정지"),
    "procedural_only": (r"소송비용액확정", r"소송비용만", r"관할위반|이송결정|항소기간|재심의 소"),
}

DIRECT_TORT_PATTERNS = {
    "tort_basis": (r"불법행위", r"민법\s*제?\s*(?:750|751|756|758)\s*조", r"사용자책임|공작물책임|제조물책임"),
    "direct_damages_claim": (r"피고[^.]{0,100}원고[^.]{0,100}(?:배상|지급)할 의무", r"원고[^.]{0,120}피고[^.]{0,120}손해배상", r"피해자|유족"),
    "conduct": (r"사고|충돌|추돌|추락|폭행|상해를 가|수술|진료|투약|누출|게시|훼손|침해|방치|설치|관리상\s*하자|결함"),
    "harm": (r"상해|부상|사망|치료비|후유장해|일실수입|위자료|재산상\s*손해|정신적\s*손해|명예훼손|개인정보"),
    "causation": (r"인하여|때문에|그 결과|상당인과관계|원인으로|이로써|이로 인한|손해가\s*발생"),
}

SUBTYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("product_safety", (r"제조물책임|제조물|제품[^.]{0,30}결함|결함[^.]{0,30}제품|제품[^.]{0,30}안전|식품[^.]{0,30}(?:이물|결함)")),
    ("privacy_reputation", (r"명예훼손|모욕|사생활|개인정보|초상권|게시글|기사\s*게재")),
    ("employer_vicarious_liability", (r"사용자책임|피용자|직원[^.]{0,60}업무|업무집행[^.]{0,40}불법행위")),
    ("premises_facility_safety", (r"공작물책임|시설[^.]{0,30}(?:하자|안전)|건물[^.]{0,30}하자|계단|추락|미끄러|넘어져|안전관리")),
    ("traffic_accident", (r"교통사고|자동차|차량|운전|추돌|충돌|횡단보도|오토바이")),
    ("medical_professional", (r"의료과오|의료사고|의료진|병원|의사|간호사|수술|진료|투약")),
    ("intentional_tort", (r"고의로|폭행|감금|협박|기망하여|무고|재물손괴")),
    ("property_damage", (r"재산상\s*손해|차량\s*파손|건물\s*훼손|화재|침수|수목[^.]{0,20}벌목|영업손실")),
    ("general_personal_injury", (r"상해|부상|치료비|후유장해|신체")),
)


def first_existing(row: dict[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        if column in row and compact(row.get(column, "")):
            return compact(row.get(column, ""))
    return ""


def regex_hits(patterns: Iterable[str], text: str) -> list[str]:
    if isinstance(patterns, str):
        patterns = (patterns,)
    return [pattern for pattern in patterns if re.search(pattern, text, re.I)]


def normalize_date_match(match: re.Match[str]) -> str:
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def normalize_structured_date(value: object) -> str | None:
    match = DATE_RE.search(compact(value))
    return normalize_date_match(match) if match else None


def _context(text: str, start: int, end: int, radius: int = 90) -> str:
    return normalize_whitespace(text[max(0, start - radius): min(len(text), end + radius)])


def classify_date_role(context: str) -> str:
    if re.search(r"사고|발생|충돌|추락|폭행|수술|진료|침해|게시|누출|벌목", context):
        return "incident"
    if re.search(r"지급|송금|납부|변제|보험금|급여|임금|대금", context):
        return "payment"
    if re.search(r"계약|약정|임대차|매매|도급|체결", context):
        return "contract"
    if re.search(r"소장|송달|항소|판결|선고|변론|결정|제소", context):
        return "procedure"
    return "unknown"


def extract_dates(row: dict[str, Any], raw_text: str) -> dict[str, object]:
    decision_date: str | None = None
    verified = False
    evidence: list[str] = []
    for column in DATE_COLUMNS:
        if compact(row.get(column, "")):
            decision_date = normalize_structured_date(row[column])
            if decision_date:
                verified = True
                evidence.append(f"source_metadata:{column}")
                break

    header = raw_text[:1600]
    if not decision_date:
        header_patterns = (
            r"(?:판결\s*)?선고(?:일)?\s*[:：]?\s*" + DATE_RE.pattern,
            DATE_RE.pattern + r"\s*(?:선고|판결선고)",
        )
        for pattern in header_patterns:
            match = re.search(pattern, header)
            if match:
                date_match = DATE_RE.search(match.group(0))
                if date_match:
                    decision_date = normalize_date_match(date_match)
                    verified = True
                    evidence.append("header_explicit_decision_date")
                    break

    mentions: list[dict[str, str]] = []
    incident_date: str | None = None
    for match in DATE_RE.finditer(raw_text):
        date = normalize_date_match(match)
        context = _context(raw_text, match.start(), match.end())
        role = classify_date_role(context)
        if verified and date == decision_date and re.search(r"선고", context):
            continue
        mentions.append({"date": date, "context": context, "date_role": role})
        if incident_date is None and role == "incident":
            incident_date = date
    return {
        "decision_date": decision_date,
        "decision_year": int(decision_date[:4]) if decision_date else None,
        "decision_date_verified": verified,
        "decision_date_evidence": evidence,
        "incident_date": incident_date,
        "incident_year": int(incident_date[:4]) if incident_date else None,
        "other_date_mentions": mentions,
    }


def extract_case_numbers(row: dict[str, Any], raw_text: str) -> dict[str, object]:
    all_numbers = [compact(match.group(0)) for match in CASE_NUMBER_RE.finditer(raw_text)]
    current: str | None = None
    verified = False
    evidence: list[str] = []
    metadata_number = first_existing(row, CASE_NUMBER_COLUMNS)
    if metadata_number and CASE_NUMBER_RE.search(metadata_number):
        current = compact(CASE_NUMBER_RE.search(metadata_number).group(0))
        verified = True
        evidence.append("source_metadata")
    if not current:
        header = raw_text[:1600]
        explicit = re.search(r"(?:^|\n|\s)(?:사건번호|사\s*건)\s*[:：]?\s*(" + CASE_NUMBER_RE.pattern + r")", header)
        if explicit:
            number_match = CASE_NUMBER_RE.search(explicit.group(0))
            if number_match:
                current = compact(number_match.group(0))
                verified = True
                evidence.append("header_case_label")
    if not current:
        header = raw_text[:800]
        first_order = min([pos for pos in (header.find("주문"), header.find("당사자"), header.find("청구취지")) if pos >= 0] or [len(header)])
        header_zone = header[:first_order]
        match = CASE_NUMBER_RE.search(header_zone)
        court = COURT_RE.search(header_zone)
        if match and court and abs(match.start() - court.start()) < 250:
            current = compact(match.group(0))
            verified = True
            evidence.append("top_header_court_case_block")
    cited = []
    removed_current = False
    for number in all_numbers:
        if current and number == current and not removed_current:
            removed_current = True
            continue
        cited.append(number)
    return {
        "current_case_number": current,
        "current_case_number_verified": verified,
        "current_case_number_evidence": evidence,
        "cited_case_numbers": unique(cited),
    }


def extract_court(row: dict[str, Any], raw_text: str) -> tuple[str | None, bool]:
    metadata = first_existing(row, COURT_COLUMNS)
    if metadata:
        return metadata, True
    match = COURT_RE.search(raw_text[:1000])
    return (compact(match.group(1)), True) if match else (None, False)


def classify_court_level(raw_text: str, court_name: str | None, case_number: str | None, verified_number: bool) -> dict[str, object]:
    current_zone = raw_text[:5000]
    strong = [f"{name}:{match.group(0)}" for name, pattern in CURRENT_APPELLATE_STRONG if (match := re.search(pattern, current_zone))]
    aux = [f"{name}:{match.group(0)}" for name, pattern in CURRENT_APPELLATE_AUX if (match := re.search(pattern, current_zone))]
    supreme = [f"{name}:{match.group(0)}" for name, pattern in SUPREME_PATTERNS if (match := re.search(pattern, current_zone))]
    conflicts: list[str] = []
    code = ""
    if case_number and (number_match := CASE_NUMBER_RE.search(case_number)):
        code = number_match.group(2)
    if supreme:
        conflicts.extend(supreme)
    if court_name and "대법원" in court_name:
        return {"court_level": "supreme", "court_level_confidence": "high", "current_case_appellate_evidence": strong + aux, "appellate_evidence_count": len(strong), "conflicting_court_level_signals": conflicts + [f"court:{court_name}"]}
    if verified_number and code == "다" and not strong:
        return {"court_level": "supreme", "court_level_confidence": "high", "current_case_appellate_evidence": [], "appellate_evidence_count": 0, "conflicting_court_level_signals": conflicts}
    court_appellate = bool(court_name and ("고등법원" in court_name or re.search(r"지방법원.*항소부", court_name)))
    number_appellate = verified_number and code == "나"
    if court_appellate and number_appellate and not supreme:
        confidence = "high"
    elif len(strong) >= 3 and not supreme:
        confidence = "high"
    elif len(strong) >= 2:
        confidence = "medium"
    elif (court_appellate or number_appellate) and len(strong) >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    if confidence in {"high", "medium"}:
        level = "appellate"
    elif verified_number and code in {"가합", "가단", "가소"}:
        level = "trial"
    else:
        level = "unknown"
    evidence = strong + aux
    if court_appellate:
        evidence.append(f"current_court:{court_name}")
        evidence.append(f"court_name: {court_name}")
    if number_appellate:
        evidence.append(f"current_case_number:{case_number}")
    return {"court_level": level, "court_level_confidence": confidence, "current_case_appellate_evidence": unique(evidence), "appellate_evidence_count": len(strong), "conflicting_court_level_signals": unique(conflicts)}


def broad_candidate(text: str) -> tuple[bool, list[str]]:
    hits = regex_hits(BROAD_PATTERNS, text)
    criminal = regex_hits((r"피고인", r"공소사실", r"범죄사실", r"징역\s*\d", r"\d{4}\s*고(?:단|합|정)\s*\d+"), text[:5000])
    return bool(hits) and not criminal, hits


def classify_claim_posture(text: str) -> tuple[str, str, list[str]]:
    hits = {label: regex_hits(patterns, text) for label, patterns in POSTURE_PATTERNS.items()}
    precedence = (
        "judgment_enforcement", "wage_or_compensation", "insurer_subrogation",
        "joint_tortfeasor_contribution", "insurance_coverage", "family_or_domestic_dispute",
        "administrative_or_state_liability", "property_or_title_dispute", "contract_or_payment", "procedural_only",
    )
    for label in precedence:
        if hits[label]:
            return label, "high" if len(hits[label]) >= 2 else "medium", [f"{label}:{item}" for item in hits[label][:8]]
    direct_hits = {name: regex_hits(patterns, text) for name, patterns in DIRECT_TORT_PATTERNS.items()}
    present = [name for name, values in direct_hits.items() if values]
    evidence = [f"{name}:{values[0]}" for name, values in direct_hits.items() if values]
    if all(direct_hits[name] for name in ("tort_basis", "conduct", "harm")) and len(present) >= 4:
        return "direct_tort_claim", "high" if len(present) == 5 else "medium", evidence
    return "unclear", "low", evidence or ["no_current_claim_posture_signal"]


def classify_liability_basis(claim_posture: str, text: str) -> tuple[str, list[str]]:
    if claim_posture == "direct_tort_claim":
        contract = regex_hits(POSTURE_PATTERNS["contract_or_payment"], text)
        if contract:
            return "mixed_tort_contract", [f"contract_overlay:{item}" for item in contract]
        return "non_contractual_tort", ["direct_tort_elements_present"]
    if claim_posture in {"insurance_coverage", "insurer_subrogation"}:
        return "insurance_only", [f"posture:{claim_posture}"]
    if claim_posture in {"contract_or_payment", "wage_or_compensation", "property_or_title_dispute"}:
        return "contract_only", [f"posture:{claim_posture}"]
    if claim_posture == "administrative_or_state_liability":
        return "administrative_or_state_liability", [f"posture:{claim_posture}"]
    if claim_posture in {"judgment_enforcement", "procedural_only"}:
        return "procedural_only", [f"posture:{claim_posture}"]
    if claim_posture == "joint_tortfeasor_contribution":
        return "non_contractual_tort", ["underlying_tort_but_internal_contribution"]
    return "unclear", ["claim_posture_unclear"]


def classify_subtype(text: str) -> str:
    # Taxonomy-specific signals outrank generic injury/damage terms.  Counting
    # every occurrence caused long medical/premises opinions to collapse into
    # general_personal_injury merely because "상해" was repeated.
    for subtype, patterns in SUBTYPE_PATTERNS:
        if isinstance(patterns, str):
            patterns = (patterns,)
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            return subtype
    return "other_tort" if re.search(r"불법행위", text) else "unclear"


def harm_flags(text: str) -> dict[str, bool]:
    return {
        "death_involved": bool(re.search(r"사망|유족|망인", text)),
        "physical_injury_involved": bool(re.search(r"상해|부상|치료|후유장해|신체", text)),
        "property_damage_involved": bool(re.search(r"재산상\s*손해|파손|훼손|화재|침수|벌목|영업손실", text)),
        "emotional_harm_involved": bool(re.search(r"정신적\s*손해|위자료|명예훼손|모욕", text)),
    }


FACT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "party_relationship": (r"원고.{0,100}피고|피고.{0,100}원고|사용자|소유자|운영자|의사|환자|피해자"),
    "specific_conduct": DIRECT_TORT_PATTERNS["conduct"],
    "event_circumstances": (r"발생|당시|경위|장소|현장|도로|병원|건물|시설"),
    "time_sequence": (DATE_RE.pattern, r"그 후|이후|당시|다음날|경부터|무렵"),
    "harm_type": DIRECT_TORT_PATTERNS["harm"],
    "causation_facts": DIRECT_TORT_PATTERNS["causation"],
    "defense_facts": (r"피고는[^.]{0,200}(?:주장|항변|부인|다툰)", r"과실상계|원고의\s*과실|책임제한|인과관계가 없다|주의의무를 다"),
}


def assess_factual_sufficiency(raw_text: str) -> dict[str, object]:
    incorporation = regex_hits(INCORPORATION_PATTERNS, raw_text)
    current_fact_section = ""
    fact_match = re.search(r"(?:기초\s*사실|인정\s*사실|사실\s*관계|사건의\s*경위)(.*?)(?=\n?\s*\d+\.\s*(?:당사자|판단|쟁점|청구)|판\s*단|$)", raw_text, re.S)
    if fact_match:
        current_fact_section = fact_match.group(1)
    # The neutral pattern also needs causation and defense facts, which Korean
    # opinions commonly state in the following "당사자의 주장" section rather
    # than inside "기초사실".  Evaluate the current document, while retaining a
    # separate substantial-fact-section requirement below.
    probe = raw_text[:12000]
    category_hits = {name: regex_hits(patterns, probe) for name, patterns in FACT_CATEGORIES.items()}
    present = [name for name, hits in category_hits.items() if hits]
    score = min(100, len(present) * 13 + (9 if len(current_fact_section) >= 500 else 0))
    evidence = [f"{name}:{hits[0]}" for name, hits in category_hits.items() if hits]
    reasons: list[str] = []
    missing = [name for name in FACT_CATEGORIES if name not in present]
    reasons.extend(f"missing:{name}" for name in missing)
    if incorporation:
        reasons.append("first_instance_reason_incorporated")
    if len(current_fact_section) < 250:
        reasons.append("no_substantial_current_fact_section")
    if incorporation and len(current_fact_section) < 500:
        quality = "mostly_incorporated"
        independent = False
    elif incorporation:
        quality = "partially_incorporated"
        independent = len(present) >= 6 and score >= 78
    elif (len(current_fact_section) >= 180 or (len(raw_text) >= 500 and len(present) == len(FACT_CATEGORIES))) and len(present) >= 6 and score >= 78:
        quality = "self_contained"
        independent = True
    elif len(present) >= 4:
        quality = "insufficient"
        independent = False
    else:
        quality = "insufficient"
        independent = False
    sufficient = independent and quality in {"self_contained", "partially_incorporated"}
    return {
        "factual_background_sufficient": sufficient,
        "facts_independently_reconstructable": independent,
        "fact_source_quality": quality,
        "factual_sufficiency_score": score,
        "factual_sufficiency_evidence": evidence,
        "factual_insufficiency_reasons": unique(reasons),
    }


def incident_fingerprint(record: dict[str, object]) -> str | None:
    text = str(record.get("raw_text") or "")
    incident_date = record.get("incident_date")
    if not incident_date:
        return None
    date_forms = (
        str(incident_date),
        str(incident_date).replace("-", ". ") + ".",
        str(incident_date).replace("-", "."),
    )
    position = next((text.find(value) for value in date_forms if text.find(value) >= 0), -1)
    if position < 0:
        return None
    # Related opinions often repeat the same incident sentence.  Hashing its
    # local wording is conservative and avoids grouping thousands of unrelated
    # anonymized A/B cases that merely share a date and subtype.
    context = normalize_whitespace(text[max(0, position - 100): position + 320])
    if len(context) < 100:
        return None
    subtype = str(record.get("case_subtype") or "")
    normalized = f"{incident_date}|{subtype}|{context}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def apply_duplicate_and_related_qc(records: list[dict[str, object]]) -> dict[str, int]:
    counters: Counter[str] = Counter()
    seen_exact: dict[str, dict[str, object]] = {}
    seen_normalized: dict[str, dict[str, object]] = {}
    seen_case: dict[str, dict[str, object]] = {}
    seen_incident: dict[str, dict[str, object]] = {}
    priority = {"direct_tort_claim": 0, "unclear": 5, "insurer_subrogation": 8, "joint_tortfeasor_contribution": 8, "judgment_enforcement": 9}
    ordered = sorted(records, key=lambda row: (priority.get(str(row.get("claim_posture")), 4), str(row.get("case_id"))))
    for record in ordered:
        exact = str(record.get("raw_text_sha256") or "")
        normalized = str(record.get("normalized_text_sha256") or "")
        number = compact(record.get("current_case_number") or "").lower()
        fingerprint = incident_fingerprint(record)
        record["underlying_incident_fingerprint"] = fingerprint
        duplicate_of: dict[str, object] | None = None
        reason: str | None = None
        if exact and exact in seen_exact:
            duplicate_of, reason = seen_exact[exact], "exact_duplicate"
        elif normalized and normalized in seen_normalized:
            duplicate_of, reason = seen_normalized[normalized], "normalized_text_duplicate"
        elif number and number in seen_case:
            duplicate_of, reason = seen_case[number], "same_current_case_number"
        elif fingerprint and fingerprint in seen_incident:
            duplicate_of, reason = seen_incident[fingerprint], "same_underlying_incident"
        if duplicate_of:
            group = "rel_" + hashlib.sha256((str(duplicate_of["case_id"]) + str(record["case_id"])).encode()).hexdigest()[:12]
            record["related_case_group_id"] = group
            duplicate_of["related_case_group_id"] = duplicate_of.get("related_case_group_id") or group
            record["related_case_ids"] = unique([str(duplicate_of["case_id"])])
            record["duplicate_of_case_id"] = str(duplicate_of["case_id"])
            record["duplicate_reason"] = reason
            duplicate_of["related_case_ids"] = unique(list(duplicate_of.get("related_case_ids") or []) + [str(record["case_id"])])
            record["duplicate_or_related_reason"] = reason
            counters[reason] += 1
        else:
            record.setdefault("related_case_group_id", None)
            record.setdefault("related_case_ids", [])
            record.setdefault("duplicate_or_related_reason", None)
            record.setdefault("duplicate_of_case_id", "")
            record.setdefault("duplicate_reason", "")
        if exact:
            seen_exact.setdefault(exact, record)
        if normalized:
            seen_normalized.setdefault(normalized, record)
        if number:
            seen_case.setdefault(number, record)
        if fingerprint:
            seen_incident.setdefault(fingerprint, record)
    return dict(counters)


def record_hash_fields(raw_text: str) -> dict[str, str]:
    return {"raw_text_sha256": sha256_text(raw_text), "normalized_text_sha256": sha256_text(normalized_text_for_hash(raw_text))}
