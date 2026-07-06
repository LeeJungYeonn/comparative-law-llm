from __future__ import annotations

import re
from typing import Iterable


KR_LEGAL_SIGNAL_TERMS = [
    "과실이 인정된다",
    "위법하다",
    "손해배상책임이 있다",
    "손해배상책임",
    "불법행위책임",
    "불법행위",
    "민법 제750조",
    "민법 제751조",
    "민법 제763조",
    "위자료",
    "책임이 인정된다",
]

US_LEGAL_SIGNAL_TERMS = [
    "negligence",
    "duty of care",
    "breach",
    "breach of duty",
    "proximate cause",
    "causation",
    "punitive damages",
    "comparative negligence",
    "strict liability",
    "liable",
    "liability",
]

JURISDICTION_TERMS = [
    "Korea",
    "Republic of Korea",
    "South Korea",
    "California",
    "New York",
    "United States",
    "U.S.",
    "federal",
    "대한민국",
    "한국",
    "서울",
    "부산",
    "대법원",
    "고등법원",
    "지방법원",
]

KR_FACT_START_HEADINGS = [
    "기초사실",
    "인정사실",
    "사실관계",
    "전제사실",
    "다툼 없는 사실",
]
KR_FACT_STOP_HEADINGS = [
    "원고의 주장",
    "원고 주장",
    "원고 주장의 요지",
    "피고의 주장",
    "피고 주장",
    "피고 주장의 요지",
    "당사자의 주장",
    "당사자들의 주장",
    "판단",
    "관련 법리",
    "손해배상책임",
    "피고의 금전 지급 책임",
    "피고의 항변",
    "항변과 그에 대한 판단",
    "청구원인에 관한 판단",
    "주위적 청구에 관한 판단",
    "예비적 청구에 관한 판단",
    "본안에 대한 판단",
    "결론",
]
KR_ORDER_CLAIM_TERMS = [
    "주문",
    "청구취지",
    "소송비용",
    "가집행",
]
KR_RECOGNITION_BASIS_TERMS = [
    "[인정 근거]",
    "인정 근거",
    "인정근거",
]
KR_CRIMINAL_SIGNAL_TERMS = [
    "피고인",
    "징역",
    "집행유예",
    "범죄사실",
    "범죄전력",
    "공소",
    "검사",
    "유죄",
    "형법",
]
US_CRIMINAL_SIGNAL_TERMS = [
    "People v.",
    "Crim.",
    "warden",
    "habeas",
    "prison",
    "sentence",
    "conviction",
]

_KR_HEADING_PREFIX = r"(?:\s*(?:제?\d+\s*[장절항]\s*)?(?:\d+|[가-힣]|[IVX]+)\s*[\.\)]\s*)?"
_KR_HEADING_OPEN = r"(?:【|\[|\()?\s*"
_KR_HEADING_CLOSE = r"\s*(?:】|\]|\))?"
KR_FACT_START_RE = re.compile(
    rf"(?m)^\s*{_KR_HEADING_PREFIX}{_KR_HEADING_OPEN}(?:{'|'.join(map(re.escape, KR_FACT_START_HEADINGS))}){_KR_HEADING_CLOSE}\s*$"
)
KR_FACT_END_RE = re.compile(
    rf"(?m)^\s*{_KR_HEADING_PREFIX}{_KR_HEADING_OPEN}(?:{'|'.join(map(re.escape, KR_FACT_STOP_HEADINGS))}){_KR_HEADING_CLOSE}\s*$"
)
KR_INLINE_FACT_START_RE = re.compile(
    r"(?:^|[\n\r ]+)(?:제?\d+\s*[장절항]\s*)?(?:\d+|[가-힣]|[IVX]+)?\s*[\.\)]?\s*"
    r"(?:【|\[|\()?\s*(?:기초\s*사실|인정\s*사실|사실\s*관계|전제\s*사실|다툼\s*없는\s*사실)"
    r"\s*(?:】|\]|\))?"
)
KR_INLINE_FACT_STOP_RE = re.compile(
    r"(?:^|[\n\r ]+)(?:(?:제?\d+\s*[장절항]\s*)?(?:\d+|[가-힣]|[IVX]+)\s*[\.\)]\s*|(?:【|\[|\())\s*"
    r"(?:원고의\s*주장|원고\s*주장(?:의\s*요지)?|피고의\s*주장|피고\s*주장(?:의\s*요지)?|"
    r"당사자의\s*주장|당사자들의\s*주장|판단|관련\s*법리|손해배상책임|"
    r"피고의\s*금전\s*지급\s*책임|피고의\s*항변|항변과\s*그에\s*대한\s*판단|"
    r"청구원인에\s*관한\s*판단|주위적\s*청구에\s*관한\s*판단|예비적\s*청구에\s*관한\s*판단|"
    r"본안에\s*대한\s*판단|결론|주문|청구취지|인정\s*근거|인정근거)"
    r"\s*(?:】|\]|\))?"
)
KR_REASON_HEADING_RE = re.compile(r"(?m)^\s*(?:이유|청구취지|주문)\s*$")
US_FACT_START_RE = re.compile(
    r"(?im)^\s*(?:I+\.|[0-9]+\.)?\s*(?:background|factual background|facts|statement of facts|findings of fact|relevant facts)\s*$"
)
US_FACT_END_RE = re.compile(
    r"(?im)^\s*(?:I+\.|[0-9]+\.)?\s*(?:discussion|analysis|legal standard|conclusions of law|conclusion|order)\s*$"
)
KR_COURT_NAME_RE = re.compile(
    r"(?:서울|부산|대구|인천|광주|대전|울산|수원|춘천|청주|전주|제주|창원|의정부|"
    r"서울중앙|서울동부|서울남부|서울북부|서울서부|부산동부|부산서부)?"
    r"(?:지방법원|고등법원|가정법원|행정법원)(?:\s*[가-힣]+지원)?"
)
KR_CASE_NUMBER_RE = re.compile(r"\b\d{4}\s*(?:가합|가단|가소|나|다|머|차|카합|카단)\s*\d+\b")
KR_EVIDENCE_RE = re.compile(
    r"(?:갑|을|병)\s*(?:제)?\s*\d+"
    r"(?:\s*(?:,|및|내지|부터|~|-)\s*\d+)*"
    r"\s*호증(?:의\s*\d+)?(?:\s*\([^)]*\))?"
)
KR_EVIDENCE_VARIANT_NOTE_RE = re.compile(r"\(?\s*가지번호\s*(?:있는\s*)?호증\s*포함\s*\)?")
KR_ORPHAN_EVIDENCE_RE = re.compile(r"\b\d+\s*호증(?:의\s*\d+)?(?:\s*\([^)]*\))?")
KR_RECOGNITION_BASIS_RE = re.compile(
    r"\[?\s*인정\s*근거\s*\]?\s*[:：]?\s*[^.。;\n]*(?:변론 전체의 취지)?"
)
KR_EVIDENCE_HEADER_RE = re.compile(
    r"^\s*\[?\s*증거\s*\]?\s*[^가-힣A-Za-z0-9]*(?:갑|을)\s*[^.。;\n]*?(?=\s*[가-힣]\s*[\.\)])"
)
KR_JUDGMENT_RESULT_SENTENCE_RE = re.compile(
    r"[^.。!?]*?(?:피고는\s+원고에게\s+[^.。!?]*?지급하라|청구를\s+기각한다|소송비용은\s+[^.。!?]*?부담한다|가집행할\s+수\s+있다)[.。!?\n]?"
)
KR_LEGAL_ANALYSIS_CUE_RE = re.compile(
    r"(?:^|[.。]\s*)(?:따라서|그러므로|그렇다면|살피건대|이에\s*대하여|이와\s*관련하여)"
)
KR_LEGAL_NEGATIVE_CONCLUSION_RE = re.compile(
    r"[^.。!?]*(?:의무가\s*있다고\s*볼\s*수\s*없다|이유\s*없다|받아들이지\s*아니한다)[.。!?\n]?"
)
KR_FACT_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(?:다음\s*각\s*사실은|아래\s*사실은)\s*당사자(?:들)?\s*사이에\s*다툼이\s*없거나[,，]?\s*"),
    re.compile(r"^\s*성립에\s*다툼이\s*없는\s*"),
    re.compile(r"(?:의\s*)?각\s*기재에\s*(?:변론\s*전체의\s*취지|변론의\s*전취지)?\s*를?\s*종합하면[,，]?\s*"),
    re.compile(r"변론\s*전체의\s*취지(?:를\s*종합하면)?[,，]?\s*"),
    re.compile(r"변론의\s*전취지(?:를\s*종합하면)?[,，]?\s*"),
    re.compile(r"성립에\s*다툼이\s*없는\s*"),
    re.compile(r"(?:의\s*)?각\s*기재(?:에)?\s*"),
    re.compile(r"(?:이를\s*)?인정할\s*수\s*있고[,，]?\s*"),
    re.compile(r"달리\s*반증이\s*없다[.。]?\s*"),
    re.compile(r"인정된다[.。]?\s*"),
    re.compile(r"인정할\s*수\s*있다[.。]?\s*"),
]
KR_PROCEDURAL_ONLY_RE = re.compile(
    r"(?:청구원인\s*사실\s*별지\s*[‘'\"“”]?\s*청구원인|"
    r"사건의\s*개요.*(?:판결\s*확정|항소기간\s*도과|전소|확정\s*판결)|"
    r"(?:판결\s*확정|항소기간\s*도과|소송\s*계속|전소\s*판결))",
    flags=re.DOTALL,
)


def normalize_whitespace(text: object) -> str:
    if text is None:
        return ""
    value = str(text).replace("\ufeff", "").replace("\xa0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def compact_inline(text: object) -> str:
    return re.sub(r"\s+", " ", normalize_whitespace(text)).strip()


def excerpt(text: object, limit: int = 1000) -> str:
    value = compact_inline(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


def split_sentences(text: str) -> list[str]:
    value = compact_inline(text)
    if not value:
        return []
    parts = re.split(r"(?<=[.!?。！？다])\s+(?=[A-Z0-9가-힣\"'(\[])", value)
    return [part.strip() for part in parts if part.strip()]


def contains_criminal_signal(text: str, origin: str) -> bool:
    terms = KR_CRIMINAL_SIGNAL_TERMS if origin == "KR" else US_CRIMINAL_SIGNAL_TERMS
    flags = 0 if origin == "KR" else re.IGNORECASE
    return any(re.search(re.escape(term), text, flags=flags) for term in terms)


def criminal_signal_terms(text: str, origin: str) -> list[str]:
    terms = KR_CRIMINAL_SIGNAL_TERMS if origin == "KR" else US_CRIMINAL_SIGNAL_TERMS
    flags = 0 if origin == "KR" else re.IGNORECASE
    found = [term for term in terms if re.search(re.escape(term), text, flags=flags)]
    return list(dict.fromkeys(found))


def starts_with_order_or_claim(text: str) -> bool:
    value = compact_inline(text)
    return bool(re.match(r"^(?:주문|청구취지)\b", value))


def contains_order_or_claim_section(text: str) -> bool:
    return any(re.search(re.escape(term), text) for term in KR_ORDER_CLAIM_TERMS)


def cleanup_kr_fact_text(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    value = normalize_whitespace(text)

    stop_match = KR_INLINE_FACT_STOP_RE.search(value)
    if stop_match:
        removed.append(compact_inline(value[stop_match.start() :]))
        value = value[: stop_match.start()]

    analysis_match = KR_LEGAL_ANALYSIS_CUE_RE.search(value)
    if analysis_match:
        removed.append(compact_inline(value[analysis_match.start() :]))
        value = value[: analysis_match.start()]

    for pattern in [
        KR_JUDGMENT_RESULT_SENTENCE_RE,
        KR_LEGAL_NEGATIVE_CONCLUSION_RE,
        KR_RECOGNITION_BASIS_RE,
        KR_EVIDENCE_HEADER_RE,
        KR_EVIDENCE_RE,
        KR_EVIDENCE_VARIANT_NOTE_RE,
        KR_ORPHAN_EVIDENCE_RE,
        KR_COURT_NAME_RE,
        KR_CASE_NUMBER_RE,
    ]:
        for match in pattern.findall(value):
            if isinstance(match, tuple):
                match = " ".join(match)
            removed.append(compact_inline(match))
        value = pattern.sub(" ", value)

    for term in KR_ORDER_CLAIM_TERMS + KR_RECOGNITION_BASIS_TERMS + ["변론 전체의 취지"]:
        if term in value:
            removed.append(term)
            value = value.replace(term, " ")

    for pattern in KR_FACT_BOILERPLATE_PATTERNS:
        for match in pattern.findall(value):
            removed.append(compact_inline(match))
        value = pattern.sub(" ", value)

    value = re.sub(r"\s+([,.;:!?。])", r"\1", value)
    value = re.sub(r"([,.;:!?。]){2,}", r"\1", value)
    value = re.sub(r"(?:^|[\n ])[,.;:：]\s*", " ", value)
    value = re.sub(r"\(\s*\)|\[\s*\]|【\s*】", " ", value)
    value = normalize_whitespace(value)
    value = re.sub(r"\s{2,}", " ", value).strip(" ,.;:：")
    return value, unique(removed)


def is_procedural_only_kr_fact(text: str) -> bool:
    value = compact_inline(text)
    if not value:
        return False
    if KR_PROCEDURAL_ONLY_RE.search(value):
        return True
    procedural_terms = ["판결 확정", "항소기간 도과", "전소", "확정판결", "소송 경과", "사건의 개요"]
    return sum(1 for term in procedural_terms if term in value) >= 2 and len(value) < 1200


def strip_case_citations(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    patterns = [
        r"\b\d+\s+U\.S\.\s+\d+\b",
        r"\b\d+\s+F\.(?:\s?Supp\.?\s?\d*d?|\d+d)\s+\d+\b",
        r"\b\d+\s+S\.\s?Ct\.\s+\d+\b",
        r"\b[A-Z][A-Za-z'&.-]+\s+v\.\s+[A-Z][A-Za-z'&.-]+(?:,?\s+\d+[^\n.;]*)?",
        r"\b(?:Civil|Case|Docket)\s+No\.?\s+[A-Za-z0-9:._/-]+",
        r"\bNo\.?\s+[A-Za-z0-9:._/-]+",
        r"\b\d{4}\s*[가-힣]{1,5}\s*\d+\b",
    ]
    value = text
    for pattern in patterns:
        for match in re.findall(pattern, value, flags=re.IGNORECASE):
            removed.append(compact_inline(match))
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    return normalize_whitespace(value), removed


def strip_statute_references(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    patterns = [
        r"\b\d+\s+U\.S\.C\.\s*(?:§|sec\.?|section)?\s*\d+[A-Za-z0-9.-]*",
        r"\b[A-Z][a-z]+\.?\s+(?:Civ|Civil|Penal|Bus|Gov|Lab|Ins|Code)\.?\s*(?:Code)?\s*(?:§|section|sec\.?)\s*\d+[A-Za-z0-9.-]*",
        r"(?:민법|상법|민사소송법|국가배상법)\s*제?\s*\d+\s*조(?:\s*제?\s*\d+\s*항)?",
    ]
    value = text
    for pattern in patterns:
        for match in re.findall(pattern, value, flags=re.IGNORECASE):
            removed.append(compact_inline(match))
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    return normalize_whitespace(value), removed


def remove_named_signals(text: str, title: str = "", court: str = "") -> tuple[str, list[str]]:
    removed: list[str] = []
    value = text
    for signal in [title, court]:
        signal = compact_inline(signal)
        if len(signal) >= 4 and signal in value:
            removed.append(signal)
            value = value.replace(signal, " ")
    for term in JURISDICTION_TERMS:
        if re.search(re.escape(term), value, flags=re.IGNORECASE):
            removed.append(term)
            value = re.sub(re.escape(term), " ", value, flags=re.IGNORECASE)
    return normalize_whitespace(value), removed


def neutralize_loaded_terms(text: str) -> tuple[str, list[str]]:
    replacements = {
        "과실이 인정된다": "부주의 여부가 문제 되었다",
        "위법하다": "문제가 있었다",
        "손해배상책임이 있다": "금전 지급 책임이 문제 되었다",
        "손해배상책임": "금전 지급 책임",
        "불법행위책임": "행위로 인한 책임",
        "불법행위": "문제가 된 행위",
        "위자료": "정신적 손실 관련 금액",
        "negligence": "careless conduct",
        "duty of care": "expected care",
        "breach of duty": "failure to act as expected",
        "breach": "failure to act as expected",
        "proximate cause": "connection between conduct and harm",
        "causation": "connection between conduct and harm",
        "punitive damages": "additional monetary award",
        "comparative negligence": "claimant's own conduct",
        "strict liability": "responsibility without focusing on intent",
        "liable": "responsible",
        "liability": "responsibility",
    }
    removed: list[str] = []
    value = text
    for source, target in replacements.items():
        if re.search(re.escape(source), value, flags=re.IGNORECASE):
            removed.append(source)
            value = re.sub(re.escape(source), target, value, flags=re.IGNORECASE)
    return normalize_whitespace(value), removed


def find_legal_signals(text: str) -> list[str]:
    terms = KR_LEGAL_SIGNAL_TERMS + US_LEGAL_SIGNAL_TERMS
    found = []
    for term in terms:
        if re.search(re.escape(term), text, flags=re.IGNORECASE):
            found.append(term)
    return list(dict.fromkeys(found))


def find_jurisdiction_signals(text: str) -> list[str]:
    found = []
    for term in JURISDICTION_TERMS:
        if re.search(re.escape(term), text, flags=re.IGNORECASE):
            found.append(term)
    return list(dict.fromkeys(found))


def extract_fact_section_with_metadata(text: str, origin: str) -> tuple[str, dict[str, object]]:
    value = normalize_whitespace(text)
    metadata: dict[str, object] = {
        "has_fact_heading": False,
        "fact_extraction_method": "failed",
        "excluded_reason": "",
    }
    if not value:
        metadata["excluded_reason"] = "empty_text"
        return "", metadata

    start_re = KR_FACT_START_RE if origin == "KR" else US_FACT_START_RE
    end_re = KR_FACT_END_RE if origin == "KR" else US_FACT_END_RE
    start_match = start_re.search(value)
    if origin == "KR" and not start_match:
        start_match = KR_INLINE_FACT_START_RE.search(value)
    if start_match:
        tail = value[start_match.end() :]
        end_match = KR_INLINE_FACT_STOP_RE.search(tail) if origin == "KR" else end_re.search(tail)
        metadata["has_fact_heading"] = True
        metadata["fact_extraction_method"] = "heading_based"
        return normalize_whitespace(tail[: end_match.start()] if end_match else tail), metadata

    if origin == "KR":
        reason_match = KR_REASON_HEADING_RE.search(value)
        if starts_with_order_or_claim(value):
            metadata["excluded_reason"] = "no_fact_section_detected"
            return "", metadata
        if reason_match:
            value = value[reason_match.end() :]
        value = KR_FACT_END_RE.split(value, maxsplit=1)[0]

    sentences = split_sentences(value)
    if not sentences:
        fallback = excerpt(value, 3000)
    else:
        sentence_limit = 8 if origin == "KR" else 18
        fallback = " ".join(sentences[: min(len(sentences), sentence_limit)])

    if origin == "KR" and starts_with_order_or_claim(fallback):
        metadata["excluded_reason"] = "no_fact_section_detected"
        return "", metadata
    metadata["fact_extraction_method"] = "fallback"
    metadata["excluded_reason"] = "no_fact_section_detected"
    return fallback, metadata


def extract_fact_section(text: str, origin: str) -> tuple[str, bool]:
    fact_text, metadata = extract_fact_section_with_metadata(text, origin)
    return fact_text, bool(metadata.get("has_fact_heading"))


def truncate_preserving_sentence(text: str, max_chars: int) -> str:
    value = compact_inline(text)
    if len(value) <= max_chars:
        return value
    sentences = split_sentences(value)
    kept: list[str] = []
    total = 0
    for sentence in sentences:
        if total + len(sentence) + 1 > max_chars:
            break
        kept.append(sentence)
        total += len(sentence) + 1
    if kept:
        return " ".join(kept).strip()
    return value[:max_chars].rstrip()


def unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
