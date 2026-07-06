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

KR_FACT_START_RE = re.compile(
    r"(?m)^\s*(?:\d+\.\s*)?(?:기초사실|인정사실|사실관계|전제사실|인정되는 사실|분쟁의 경위)\s*$"
)
KR_FACT_END_RE = re.compile(
    r"(?m)^\s*(?:\d+\.\s*)?(?:판단|손해배상책임|책임의 발생|책임의 제한|손해배상의 범위|결론|주문)\s*$"
)
US_FACT_START_RE = re.compile(
    r"(?im)^\s*(?:I+\.|[0-9]+\.)?\s*(?:background|factual background|facts|statement of facts|findings of fact|relevant facts)\s*$"
)
US_FACT_END_RE = re.compile(
    r"(?im)^\s*(?:I+\.|[0-9]+\.)?\s*(?:discussion|analysis|legal standard|conclusions of law|conclusion|order)\s*$"
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


def extract_fact_section(text: str, origin: str) -> tuple[str, bool]:
    value = normalize_whitespace(text)
    if not value:
        return "", False

    start_re = KR_FACT_START_RE if origin == "KR" else US_FACT_START_RE
    end_re = KR_FACT_END_RE if origin == "KR" else US_FACT_END_RE
    start_match = start_re.search(value)
    if start_match:
        tail = value[start_match.end() :]
        end_match = end_re.search(tail)
        return normalize_whitespace(tail[: end_match.start()] if end_match else tail), True

    sentences = split_sentences(value)
    if not sentences:
        return excerpt(value, 3000), False
    return " ".join(sentences[: min(len(sentences), 18)]), False


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
