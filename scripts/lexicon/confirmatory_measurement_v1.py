#!/usr/bin/env python3
"""Frozen v1 measurement definitions for real-reference confirmatory opinions."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence


PRIMARY_FEATURES = [
    "avg_chars_per_sentence",
    "statute_density",
    "precedent_density",
    "doctrine_density",
    "remedy_density",
    "procedure_density",
    "party_arg_density",
]
SENSITIVITY_FEATURES = PRIMARY_FEATURES + ["log1p_doc_length_sentences"]
CATEGORIES = ("doctrine", "remedy", "procedure", "party_arg")

SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?。！？]+[\"'’”\]\)]*(?=\s|$)|\n+)", re.UNICODE)
PROTECTED_ABBREVIATION_RE = re.compile(
    r"(?:\b(?:U\.S\.C|U\.S|S\.Ct|F\.Supp|No|Nos|v|vs|Inc|Co|Corp|Dr|Mr|Mrs|Ms|Jr|Sr|Ct|Supp)|(?:\b[A-Z]\.){2,})\.$",
    re.I,
)

KR_STATUTE_PATTERNS = [
    re.compile(r"[가-힣]{1,30}(?:법률|시행령|법령|규칙|법)\s*제?\s*\d+(?:의\d+)?조(?:의\d+)?"),
    re.compile(r"제\s*\d+(?:의\d+)?조(?:의\d+)?"),
]
US_STATUTE_PATTERNS = [
    re.compile(r"\b\d+\s+U\.?\s*S\.?\s*C\.?\s*§+\s*[\w.()\-]+", re.I),
    re.compile(r"\b(?:[A-Z][A-Za-z.&'\-]*\s+){0,5}(?:Code|Statutes?|Rev\.?\s*Stat\.?)\s*(?:§+|Section)\s*[\w.()\-]+", re.I),
    re.compile(r"§+\s*\d+[\w.()\-]*", re.I),
]
KR_PRECEDENT_PATTERNS = [
    re.compile(r"대법원\s*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*선고\s*\d{2,4}[가-힣]{1,4}\d+\s*(?:판결|결정)?"),
    re.compile(r"\b\d{2,4}[가-힣]{1,4}\d+\b"),
]
US_PRECEDENT_PATTERNS = [
    re.compile(r"\b\d+\s+(?:U\.S\.|S\.?\s*Ct\.|F\.?\s*(?:Supp\.?\s*)?\d*d?|A\.?\s*\d+d|S\.?\s*E\.?\s*\d+d|N\.?\s*W\.?\s*\d+d|So\.?\s*\d+d|P\.?\s*\d+d)\s+\d+\b", re.I),
    re.compile(r"\b\d+\s+(?:W\.\s*Va\.|Pa\.|Mich\.|La\.|Nev\.)\s+\d+\b", re.I),
    re.compile(r"\b\d{4}\s+WL\s+\d+\b", re.I),
]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Frozen sentence rule: terminal punctuation or newline, with exact offsets."""
    spans = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        token = match.group()
        if token.startswith("."):
            prefix = text[max(start, match.start() - 20):match.end()].rstrip()
            following = text[match.end():].lstrip()[:1]
            if PROTECTED_ABBREVIATION_RE.search(prefix) or (following.isdigit() and re.search(r"\d+\.$", prefix)):
                continue
        span = _trim_span(text, start, match.end())
        if span:
            spans.append(span)
        start = match.end()
    span = _trim_span(text, start, len(text))
    if span:
        spans.append(span)
    return spans or [(0, len(text))]


def _normalize_citation(value: str) -> str:
    return " ".join(value.casefold().split()).rstrip(".,;:")


def _nonoverlap_matches(text: str, patterns: Sequence[re.Pattern]) -> list[tuple[int, int, str]]:
    candidates = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), match.group()))
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[1], item[2]))
    accepted = []
    for candidate in candidates:
        if any(not (candidate[1] <= prior[0] or prior[1] <= candidate[0]) for prior in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted)


def statute_citations(text: str, country: str) -> list[tuple[int, int, str]]:
    return _nonoverlap_matches(text, KR_STATUTE_PATTERNS if country == "KR" else US_STATUTE_PATTERNS)


def precedent_citations(text: str, country: str) -> list[tuple[int, int, str]]:
    return _nonoverlap_matches(text, KR_PRECEDENT_PATTERNS if country == "KR" else US_PRECEDENT_PATTERNS)


def citation_counts(matches: Sequence[tuple[int, int, str]]) -> tuple[int, int]:
    return len(matches), len({_normalize_citation(match[2]) for match in matches})


def extract_features(
    text: str,
    country: str,
    matcher,
    surface_path: Path,
    concept_path: Path,
) -> tuple[dict, list[dict]]:
    assert country in {"KR", "US"}
    sentences = sentence_spans(text)
    denominator = len(sentences)
    assert denominator > 0
    statute_occurrences, unique_statutes = citation_counts(statute_citations(text, country))
    precedent_occurrences, unique_precedents = citation_counts(precedent_citations(text, country))
    language = "ko" if country == "KR" else "en"
    hits = matcher.match_text(text, surface_path, concept_path, language)
    for hit in hits:
        assert text[int(hit["start"]):int(hit["end"])] == hit["matched_text"]

    result = {
        "doc_length_sentences": denominator,
        "doc_length_chars": len(text),
        "avg_chars_per_sentence": len(text) / denominator,
        "statute_occurrence_count": statute_occurrences,
        "unique_statute_count": unique_statutes,
        "statute_density": unique_statutes / denominator * 100,
        "precedent_occurrence_count": precedent_occurrences,
        "unique_precedent_count": unique_precedents,
        "precedent_density": unique_precedents / denominator * 100,
    }
    for category in CATEGORIES:
        category_hits = [hit for hit in hits if hit["category"] == category]
        concepts = {hit["concept_id"] for hit in category_hits}
        families = {hit["concept_family"] for hit in category_hits}
        result[f"{category}_hit_count"] = len(category_hits)
        result[f"{category}_unique_concepts"] = len(concepts)
        result[f"{category}_unique_families"] = len(families)
        result[f"{category}_density"] = len(concepts) / denominator * 100
    result["log1p_doc_length_sentences"] = math.log1p(denominator)
    assert all(math.isfinite(float(result[feature])) for feature in SENSITIVITY_FEATURES)
    return result, hits
