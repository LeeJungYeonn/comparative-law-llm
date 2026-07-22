from __future__ import annotations

import re
import unicodedata
from collections import Counter
from fractions import Fraction
from typing import Any, Iterable

from pipeline.stage2_schema import EPISTEMIC_STATUSES


LEGAL_TERMS = {
    "en": ("negligence", "negligent", "breach", "duty of care", "proximate cause", "unlawful", "unreasonable", "liable", "strict liability", "premises liability", "product liability", "malpractice", "punitive damages", "compensatory damages", "tort"),
    "ko": ("과실", "위법", "책임", "상당인과관계", "주의의무", "주의의무 위반", "불법행위", "채무불이행", "사용자책임", "공작물책임", "의료과실", "위자료", "보증", "담보"),
}
DISPOSITION_TERMS = {"en": ("affirmed", "reversed", "remanded", "appeal", "judgment for", "summary judgment"), "ko": ("항소", "파기", "환송", "기각", "인용", "승소", "패소", "주문")}
JURISDICTION_TERMS = {"en": ("california", "los angeles", "san francisco", "court of appeal", "superior court", "jury", "deposition", "discovery"), "ko": ("대한민국", "서울", "부산", "대법원", "고등법원", "지방법원")}
PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|COMPANY|MEDICAL_INSTITUTION|PUBLIC_AGENCY|EDUCATIONAL_INSTITUTION|PROPERTY|PRODUCT|LOCATION|ADDRESS|AMOUNT)_[A-Z]+\]|\[AMOUNT\]")
ANY_BRACKET_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
CURRENCY_RE = re.compile(r"[$€£¥₩]|\b(?:USD|KRW|dollars?|won)\b|\d\s*원", re.IGNORECASE)
CASE_NUMBER_RE = re.compile(r"\b\d{2,4}[- ]?(?:cv|ca|app|가합|가단|나|다)[- ]?\d+\b", re.IGNORECASE)

EN_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-three": 23, "twenty-five": 25}
KO_COUNTER_WORDS = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "열한": 11, "열두": 12}
UNIT_ALIASES = {
    "km/h": (r"km\s*/\s*h", r"kilometers?\s+per\s+hour", r"킬로미터\s*/\s*시", r"킬로미터\s*퍼\s*시"),
    "m": (r"m(?![A-Za-z])", r"meters?", r"metres?", r"미터"),
    "km": (r"km(?![A-Za-z/])", r"kilometers?", r"kilometres?", r"킬로미터"),
    "kg": (r"kg(?![A-Za-z])", r"kilograms?", r"킬로그램"),
    "ton": (r"tons?", r"tonnes?", r"톤"),
    "week": (r"weeks?", r"주"),
    "month": (r"months?", r"개월"),
    "year": (r"years?(?:\s+old)?", r"세", r"년"),
    "percent": (r"percent", r"%", r"퍼센트"),
    "degree": (r"degrees?", r"°", r"도"),
    "knot": (r"knots?", r"노트"),
}


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.casefold()
    return [term for term in terms if term.casefold() in lowered]


def normalize_units(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Normalize explicit measurements while preserving source precision."""
    conversions: list[dict[str, Any]] = []
    numeric = r"\d[\d,]*(?:\.\d+)?"
    patterns = [
        (re.compile(rf"\b({numeric})\s*(?:mph|miles per hour)\b", re.I), "km/h", 1.60934),
        (re.compile(rf"\b({numeric})\s*(?:feet|foot|ft)\b", re.I), "m", 0.3048),
        (re.compile(rf"\b({numeric})\s*miles?\b", re.I), "km", 1.60934),
        (re.compile(rf"\b({numeric})\s*(?:pounds?|lbs?)\b", re.I), "kg", 0.453592),
        (re.compile(rf"\b(-?{numeric})\s*°?F\b", re.I), "°C", None),
    ]
    result = text
    for pattern, unit, multiplier in patterns:
        def replace(match: re.Match[str]) -> str:
            numeric_value = float(match.group(1).replace(",", ""))
            value = (numeric_value - 32) * 5 / 9 if multiplier is None else numeric_value * multiplier
            rounded = round(value, 1 if abs(value) < 100 else 0)
            rendered = f"{rounded:g} {unit}"
            conversions.append({"original": match.group(0), "normalized": rendered})
            return rendered
        result = pattern.sub(replace, result)
    word_pattern = re.compile(r"\b(" + "|".join(map(re.escape, EN_NUMBER_WORDS)) + r")(?:\s+or\s+(" + "|".join(map(re.escape, EN_NUMBER_WORDS)) + r"))?\s+(feet|foot|miles?|pounds?|lbs?)\b", re.I)
    def replace_words(match: re.Match[str]) -> str:
        values = [EN_NUMBER_WORDS[match.group(1).casefold()]] + ([EN_NUMBER_WORDS[match.group(2).casefold()]] if match.group(2) else [])
        source_unit = match.group(3).casefold()
        target, multiplier = ("m", 0.3048) if source_unit in {"foot", "feet"} else ("km", 1.60934) if source_unit.startswith("mile") else ("kg", 0.453592)
        rendered = " or ".join(f"{round(value * multiplier, 1):g} {target}" for value in values)
        conversions.append({"original": match.group(0), "normalized": rendered})
        return rendered
    result = word_pattern.sub(replace_words, result)
    korean_patterns = [
        (re.compile(r"시속\s*(\d+(?:\s*[~～-]\s*\d+)?)\s*킬로미터"), "km/h"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*킬로미터"), "km"), (re.compile(r"(\d+(?:\.\d+)?)\s*미터"), "m"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*킬로그램"), "kg"), (re.compile(r"(\d+(?:\.\d+)?)\s*톤"), "ton"),
        (re.compile(r"섭씨\s*(-?\d+(?:\.\d+)?)\s*도"), "°C"),
    ]
    for pattern, unit in korean_patterns:
        def replace_korean(match: re.Match[str]) -> str:
            rendered = f"{''.join(match.group(1).split())} {unit}"
            conversions.append({"original": match.group(0), "normalized": rendered})
            return rendered
        result = pattern.sub(replace_korean, result)
    return result, conversions


def validate_evidence(payload: dict[str, Any], case_id: str, segment_record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("case_id") != case_id: errors.append("case_id_mismatch")
    segment_map = {row["source_sentence_id"]: row["text"] for row in segment_record.get("segments", [])}
    seen: set[str] = set()
    for unit in payload.get("evidence_units", []):
        evidence_id = str(unit.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen: errors.append("invalid_or_duplicate_evidence_id")
        seen.add(evidence_id)
        source_ids = unit.get("source_sentence_ids") or []
        if not source_ids or any(source_id not in segment_map for source_id in source_ids): errors.append(f"{evidence_id}:unknown_source_sentence_id")
        exact = unit.get("exact_excerpts") or []
        if exact and exact != [segment_map[source_id] for source_id in source_ids if source_id in segment_map]: errors.append(f"{evidence_id}:non_deterministic_exact_excerpt")
        if unit.get("epistemic_status") not in EPISTEMIC_STATUSES: errors.append(f"{evidence_id}:invalid_epistemic_status")
    return sorted(set(errors))


def _tokens(text: str) -> list[str]: return re.findall(r"[A-Za-z0-9가-힣]+", text.casefold())
def _ngrams(tokens: list[str], size: int) -> Counter[tuple[str, ...]]: return Counter(tuple(tokens[i:i + size]) for i in range(max(0, len(tokens) - size + 1)))


def overlap_metrics(source: str, neutral: str) -> dict[str, float | int]:
    source_tokens, neutral_tokens = _tokens(source), _tokens(neutral)
    source_8grams, copied_positions = set(_ngrams(source_tokens, 8)), set()
    for index in range(max(0, len(neutral_tokens) - 7)):
        if tuple(neutral_tokens[index:index + 8]) in source_8grams: copied_positions.update(range(index, index + 8))
    longest = 0
    for size in range(1, min(30, len(neutral_tokens)) + 1):
        if set(_ngrams(source_tokens, size)) & set(_ngrams(neutral_tokens, size)): longest = size
        else: break
    return {"longest_shared_ngram": longest, "shared_8gram_count": len(source_8grams & set(_ngrams(neutral_tokens, 8))), "shared_12gram_count": len(set(_ngrams(source_tokens, 12)) & set(_ngrams(neutral_tokens, 12))), "verbatim_overlap_ratio": round(len(copied_positions) / max(1, len(neutral_tokens)), 4)}


def source_neutral_checks(payload: dict[str, Any], evidence: dict[str, Any], source_text: str, language: str) -> dict[str, Any]:
    warnings: list[str] = []; errors: list[str] = []
    fact_units = payload.get("fact_units") or []; fact_ids = [str(unit.get("fact_id") or "") for unit in fact_units]
    evidence_ids = {str(unit.get("evidence_id")) for unit in evidence.get("evidence_units", [])}
    if not fact_ids or len(fact_ids) != len(set(fact_ids)) or any(not fact_id for fact_id in fact_ids): errors.append("fact_ids_not_unique_or_missing")
    valid_links = True
    for unit in fact_units:
        linked = unit.get("source_evidence_ids") or []
        if not linked or any(item not in evidence_ids for item in linked): errors.append(f"{unit.get('fact_id')}:invalid_source_evidence"); valid_links = False
        if unit.get("epistemic_status") not in EPISTEMIC_STATUSES: errors.append(f"{unit.get('fact_id')}:invalid_epistemic_status")
    text = str(payload.get("master_neutral_text") or "")
    legal, disposition, jurisdiction = find_terms(text, LEGAL_TERMS[language]), find_terms(text, DISPOSITION_TERMS[language]), find_terms(text, JURISDICTION_TERMS[language])
    if legal: errors.append("legal_term_leakage")
    if disposition: errors.append("disposition_leakage")
    if jurisdiction: errors.append("jurisdiction_leakage")
    if CURRENCY_RE.search(text): errors.append("currency_leakage")
    if CASE_NUMBER_RE.search(text): errors.append("case_number_leakage")
    placeholders = ANY_BRACKET_PLACEHOLDER_RE.findall(text)
    if any(not PLACEHOLDER_RE.fullmatch(value) for value in placeholders): errors.append("invalid_placeholder")
    unit_placeholders = [value for unit in fact_units for value in ANY_BRACKET_PLACEHOLDER_RE.findall(str(unit.get("master_text") or ""))]
    if set(placeholders) != set(unit_placeholders): errors.append("placeholder_inconsistent_between_master_and_units")
    fact_types = {value for unit in fact_units for value in unit.get("fact_types") or []}
    event_present = bool(fact_types & {"action", "event"}); harm_present = bool(fact_types & {"harm", "economic_harm"})
    event_positions = [index for index, unit in enumerate(fact_units) if set(unit.get("fact_types") or []) & {"action", "event"}]
    harm_positions = [index for index, unit in enumerate(fact_units) if set(unit.get("fact_types") or []) & {"harm", "economic_harm"}]
    causal_sequence_present = bool(event_positions and harm_positions and min(event_positions) <= max(harm_positions))
    sufficiency_errors = []
    if not fact_units: sufficiency_errors.append("no_fact_units")
    if not event_present: sufficiency_errors.append("missing_action_or_event")
    if not harm_present: sufficiency_errors.append("missing_harm")
    if not valid_links: sufficiency_errors.append("invalid_source_evidence_links")
    if not causal_sequence_present: warnings.append("missing_event_to_harm_sequence")
    errors.extend(sufficiency_errors)
    metrics = overlap_metrics(source_text, text)
    if metrics["longest_shared_ngram"] >= 12 or metrics["verbatim_overlap_ratio"] > 0.5: warnings.append("high_source_overlap")
    fatal = bool(errors)
    deterministic_sufficiency = "fail" if sufficiency_errors else "warning" if not causal_sequence_present else "pass"
    risk = "high" if fatal or metrics["longest_shared_ngram"] >= 20 else "medium" if warnings or metrics["longest_shared_ngram"] >= 12 else "low"
    return {"status": "fail" if fatal else "warning" if warnings else "pass", "errors": sorted(set(errors)), "warnings": sorted(set(warnings)), "legal_terms": legal, "disposition_terms": disposition, "jurisdiction_terms": jurisdiction, "overlap": metrics, "memorization_risk": risk, "deterministic_factual_sufficiency": deterministic_sufficiency, "event_present": event_present, "harm_present": harm_present, "causal_sequence_present": causal_sequence_present, "valid_source_evidence_links": valid_links}


def _number_ready(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"(\d+)\s*분의\s*(\d+)", lambda m: f" FRACTION_{m.group(2)}_{m.group(1)} ", value)
    value = re.sub(r"(?<![가-힣])(?:절반|반)(?=(?:으로|을|이|의|만|과|와|\s|$))", " FRACTION_1_2 ", value)
    value = re.sub(r"\b(one|a)[ -]?third\b", " FRACTION_1_3 ", value)
    value = re.sub(r"\b(one|a)[ -]?half\b", " FRACTION_1_2 ", value)
    value = re.sub(r"\b(\d+)\s*/\s*(\d+)\b", lambda m: f" FRACTION_{m.group(1)}_{m.group(2)} ", value)
    value = re.sub(r"\btwice\b|두\s*차례|두\s*번", " 2 ", value)
    value = re.sub(r"\bage\s+(" + "|".join(map(re.escape, EN_NUMBER_WORDS)) + r")\b", lambda m: f" {EN_NUMBER_WORDS[m.group(1)]} years old ", value)
    value = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)-lane\b", lambda m: f" {EN_NUMBER_WORDS[m.group(1)]} lane ", value)
    value = re.sub(r"\bone\s+period\b", "a period", value)
    # Number words are numeric only in explicit quantitative contexts.  This
    # avoids treating idioms/pronouns such as ``one day`` or ``obtained one``
    # and substrings such as Korean ``두개골`` as measurements.
    en_count_nouns = r"(?:lanes?|times?|persons?|people|children|companions?|assistants?|cadd(?:y|ies)|members?|intervals?|periods?|doctors?|physicians?|tires?|acts?|weeks?|months?|years?|percent|degrees?|knots?|tons?)"
    for word, number in sorted(EN_NUMBER_WORDS.items(), key=lambda item: -len(item[0])):
        value = re.sub(rf"\b{re.escape(word)}\b(?=(?:\s+[a-z-]+){{0,5}}\s+{en_count_nouns}\b)", str(number), value)
    if re.fullmatch(r"\s*(?:" + "|".join(map(re.escape, EN_NUMBER_WORDS)) + r")\s*", value):
        value = str(EN_NUMBER_WORDS[value.strip()])
    ko_count_nouns = r"(?<![가-힣])(?:차례|번|명|개(?!골)|자녀|일행|경기보조원|기간|의사|주|개월|년|세|살|퍼센트|도|노트|톤)(?=(?:[은는이가을를의와과에로인였]|\s|$|%|[,.;:)]))"
    for word, number in sorted(KO_COUNTER_WORDS.items(), key=lambda item: -len(item[0])):
        value = re.sub(rf"(?<![가-힣]){word}(?=(?:\s+[가-힣]+){{0,3}}\s*{ko_count_nouns})", str(number), value)
    return value


def canonical_numbers(text: str) -> tuple[Counter[str], list[str]]:
    value = _number_ready(text); tokens: list[str] = []; consumed: list[tuple[int, int]] = []
    for match in re.finditer(r"FRACTION_(\d+)_(\d+)", value):
        fraction = Fraction(int(match.group(1)), int(match.group(2)))
        tokens.append(f"fraction:{fraction.numerator}/{fraction.denominator}"); consumed.append(match.span())
    def inside(position: int) -> bool: return any(start <= position < end for start, end in consumed)
    for match in re.finditer(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?", value):
        if not inside(match.start()): tokens.append(f"number:{float(match.group()):g}")
    unresolved = re.findall(r"\b(?:hundred|thousand|million)\b|(?:다섯|여섯|일곱|여덟|아홉|열)\s*(?:번|차례|명|개)", value)
    return Counter(tokens), unresolved


def canonical_measurements(text: str) -> tuple[Counter[tuple[str, str]], list[str]]:
    value = _number_ready(text)
    value = re.sub(r"시속\s*(?:약\s*)?(-?\d+(?:\.\d+)?)\s*km(?!\s*/\s*h)", r"\1 km/h", value)
    value = re.sub(r"\bage\s+(?:of\s+)?(-?\d+(?:\.\d+)?)\b", r"\1 years old", value)
    measurements: list[tuple[str, str]] = []; matched_spans: list[tuple[int, int]] = []
    number = r"(-?\d+(?:\.\d+)?)"
    for canonical, aliases in UNIT_ALIASES.items():
        unit_pattern = "(?:" + "|".join(aliases) + ")"
        pattern = re.compile(number + r"\s*" + unit_pattern, re.I)
        for match in pattern.finditer(value):
            measurements.append((f"{float(match.group(1)):g}", canonical)); matched_spans.append(match.span())
    unresolved = []
    for match in re.finditer(r"-?\d+(?:\.\d+)?\s*[A-Za-z가-힣°/%]+", value):
        if not any(start <= match.start() and match.end() <= end for start, end in matched_spans): unresolved.append(match.group())
    return Counter(measurements), unresolved


def language_residue(text: str, target_language: str) -> list[str]:
    cleaned = PLACEHOLDER_RE.sub(" ", text)
    cleaned = re.sub(r"\b(?:km/h|km|kg|m|°c)\b", " ", cleaned, flags=re.I)
    if target_language == "ko":
        # Unicode ``\b`` misses English residue followed by a Korean particle.
        return sorted(set(word for word in re.findall(r"(?<![A-Za-z])[A-Za-z]{3,}(?![A-Za-z])", cleaned) if not (word.isupper() and len(word) <= 5)))
    return sorted(set(re.findall(r"[가-힣]{2,}", cleaned)))


def _semantic_anchors(text: str, language: str) -> set[str]:
    lowered = text.casefold(); anchors: set[str] = set()
    patterns = {
        "death": ("died", "death", "killed", "사망", "숨졌", "죽었"),
        "minor_injury": ("minor injury", "slight injury", "경상"),
        "collision": ("collided", "collision", "struck", "impact", "crash", "충돌", "충격", "부딪", "추돌", "맞았"),
        "crushing": ("crushed", "caught between", "run over", "끼어", "끼였", "깔려", "깔렸", "압착"),
    }
    for name, terms in patterns.items():
        if any(term in lowered for term in terms): anchors.add(name)
    positive_knowledge = ("knew", "was aware", "알고 있었", "인지하고 있었")
    negative_knowledge = ("did not know", "didn't know", "was unaware", "알지 못", "몰랐")
    if any(term in lowered for term in negative_knowledge): anchors.add("knowledge_negative")
    elif any(term in lowered for term in positive_knowledge): anchors.add("knowledge_positive")
    return anchors


def translation_checks(master: dict[str, Any], translated: dict[str, Any], target_language: str) -> dict[str, Any]:
    warnings: list[str] = []; errors: list[str] = []
    master_units, translated_units = master.get("fact_units") or [], translated.get("translated_fact_units") or []
    master_ids = [str(item.get("fact_id")) for item in master_units]; translated_ids = [str(item.get("fact_id")) for item in translated_units]
    if master_ids != translated_ids: errors.append("fact_id_or_order_mismatch")
    master_text, translated_text = str(master.get("master_neutral_text") or ""), str(translated.get("translated_neutral_text") or "")
    master_placeholders, translated_placeholders = PLACEHOLDER_RE.findall(master_text), PLACEHOLDER_RE.findall(translated_text)
    placeholder_set_match = set(master_placeholders) == set(translated_placeholders)
    if not placeholder_set_match: errors.append("placeholder_identity_mismatch")
    if Counter(master_placeholders) != Counter(translated_placeholders): warnings.append("placeholder_occurrence_count_differs")
    master_by_id = {str(item.get("fact_id")): item for item in master_units}; translated_by_id = {str(item.get("fact_id")): item for item in translated_units}
    fact_placeholder_sets: dict[str, Any] = {}
    number_results: dict[str, Any] = {}; unit_results: dict[str, Any] = {}; semantic_results: dict[str, Any] = {}
    source_language = "ko" if target_language == "en" else "en"
    if master_ids == translated_ids:
        for fact_id in master_ids:
            source_value = str(master_by_id[fact_id].get("master_text") or ""); target_value = str(translated_by_id[fact_id].get("translated_text") or "")
            source_placeholder_set, target_placeholder_set = set(PLACEHOLDER_RE.findall(source_value)), set(PLACEHOLDER_RE.findall(target_value))
            fact_placeholder_sets[fact_id] = {"master": sorted(source_placeholder_set), "translation": sorted(target_placeholder_set), "match": source_placeholder_set == target_placeholder_set}
            if source_placeholder_set != target_placeholder_set: errors.append(f"{fact_id}:placeholder_identity_mismatch")
            source_numbers, source_unknown = canonical_numbers(source_value); target_numbers, target_unknown = canonical_numbers(target_value)
            number_value_match = set(source_numbers) == set(target_numbers)
            number_occurrence_match = source_numbers == target_numbers
            number_results[fact_id] = {"master": dict(source_numbers), "translation": dict(target_numbers), "match": number_value_match, "occurrence_match": number_occurrence_match, "unresolved": source_unknown + target_unknown}
            if number_value_match and not number_occurrence_match: warnings.append(f"{fact_id}:number_occurrence_count_differs")
            if not number_value_match:
                if source_unknown or target_unknown: warnings.append(f"{fact_id}:unresolved_number_normalization")
                else: errors.append(f"{fact_id}:numerical_value_changed")
            source_measurements, source_unit_unknown = canonical_measurements(source_value); target_measurements, target_unit_unknown = canonical_measurements(target_value)
            unit_value_match = set(source_measurements) == set(target_measurements)
            unit_occurrence_match = source_measurements == target_measurements
            unit_results[fact_id] = {"master": [list(item) + [count] for item, count in source_measurements.items()], "translation": [list(item) + [count] for item, count in target_measurements.items()], "match": unit_value_match, "occurrence_match": unit_occurrence_match, "unresolved": source_unit_unknown + target_unit_unknown}
            if unit_value_match and not unit_occurrence_match: warnings.append(f"{fact_id}:unit_occurrence_count_differs")
            if not unit_value_match:
                if source_unit_unknown or target_unit_unknown: warnings.append(f"{fact_id}:unit_surface_or_unresolved_difference")
                else: errors.append(f"{fact_id}:measurement_value_changed")
            source_anchors, target_anchors = _semantic_anchors(source_value, source_language), _semantic_anchors(target_value, target_language)
            semantic_results[fact_id] = {"master": sorted(source_anchors), "translation": sorted(target_anchors)}
            if source_anchors != target_anchors and ({"death", "minor_injury", "collision", "crushing"} & (source_anchors | target_anchors)): errors.append(f"{fact_id}:independent_fact_added_or_omitted")
            if ("knowledge_positive" in source_anchors and "knowledge_negative" in target_anchors) or ("knowledge_negative" in source_anchors and "knowledge_positive" in target_anchors): errors.append(f"{fact_id}:clear_polarity_reversal")
    legal, jurisdiction = find_terms(translated_text, LEGAL_TERMS[target_language]), find_terms(translated_text, JURISDICTION_TERMS[target_language])
    if legal: errors.append("legal_term_reintroduced")
    if jurisdiction: errors.append("jurisdiction_term_reintroduced")
    residue = language_residue(translated_text, target_language)
    if residue: warnings.append("target_language_residue")
    negation_markers = {"en": (" not ", " no ", "never", "without", "undisputed"), "ko": ("않", "없", "아니", "못", "다툼이 없")}
    lexical_negation_warning = False
    if master_ids == translated_ids:
        for fact_id in master_ids:
            source_value = f" {str(master_by_id[fact_id].get('master_text') or '').casefold()} "; target_value = f" {str(translated_by_id[fact_id].get('translated_text') or '').casefold()} "
            source_neg = any(marker in source_value for marker in negation_markers[source_language]); target_neg = any(marker in target_value for marker in negation_markers[target_language])
            undisputed_equivalent = ("다툼이 없" in source_value and "undisputed" in target_value) or ("undisputed" in source_value and "다툼이 없" in target_value)
            if source_neg != target_neg and not undisputed_equivalent: lexical_negation_warning = True
    if lexical_negation_warning: warnings.append("lexical_negation_mismatch_requires_verifier")
    source_len = max(1, len(_tokens(master_text))); ratio = len(_tokens(translated_text)) / source_len
    if not 0.35 <= ratio <= 3.0: warnings.append("translation_length_ratio_outlier")
    status = "fail" if errors else "warning" if warnings else "pass"
    return {"status": status, "errors": sorted(set(errors)), "warnings": sorted(set(warnings)), "placeholder_match": placeholder_set_match, "placeholder_occurrence_match": Counter(master_placeholders) == Counter(translated_placeholders), "fact_placeholder_sets": fact_placeholder_sets, "fact_id_match": master_ids == translated_ids, "number_unit_match": not any("numerical_value_changed" in error or "measurement_value_changed" in error for error in errors), "number_normalization": number_results, "unit_normalization": unit_results, "semantic_anchors": semantic_results, "legal_terms": legal, "jurisdiction_terms": jurisdiction, "language_residue": residue, "ko_chars": len(translated_text) if target_language == "ko" else len(master_text), "en_words": len(_tokens(translated_text)) if target_language == "en" else len(_tokens(master_text)), "translation_length_ratio": round(ratio, 4), "negation_warning": lexical_negation_warning, "temporal_order_preserved": master_ids == translated_ids}
