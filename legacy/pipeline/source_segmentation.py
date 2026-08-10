from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from pipeline.stage2_schema import Stage2CaseInput
from pipeline.text_utils import extract_fact_section_with_metadata, normalize_whitespace


FACTUAL_KEYWORDS = {
    "KR": ("사고", "당시", "발생", "병원", "치료", "수술", "충돌", "운전", "경고", "검사", "사망", "손상", "부상", "진술", "주장"),
    "CA": ("accident", "occurred", "injury", "hospital", "surgery", "collision", "drove", "warned", "inspection", "died", "damage", "testified", "alleged", "background", "facts"),
}
PROCEDURAL_KEYWORDS = {
    "KR": ("원고는", "피고는", "청구", "주장"),
    "CA": ("plaintiff", "defendant", "complaint", "alleged", "testified"),
}


@dataclass(frozen=True)
class SourceSegment:
    source_sentence_id: str
    text: str
    start_char: int
    end_char: int


def segment_source(text: str) -> list[SourceSegment]:
    """Stable paragraph/sentence segmentation with offsets into the unmodified source."""
    segments: list[SourceSegment] = []
    boundary = re.compile(r"(?<=[.!?。！？])(?:[ \t]+|\r?\n+)|\r?\n{2,}")
    start = 0
    spans: list[tuple[int, int]] = []
    for match in boundary.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    for raw_start, raw_end in spans:
        piece = text[raw_start:raw_end]
        left = len(piece) - len(piece.lstrip())
        right = len(piece.rstrip())
        if right <= left:
            continue
        actual_start, actual_end = raw_start + left, raw_start + right
        segments.append(SourceSegment(f"SRC{len(segments) + 1:04d}", text[actual_start:actual_end], actual_start, actual_end))
    return segments


def _windows(indices: Iterable[int], total: int, radius: int = 2) -> set[int]:
    selected: set[int] = set()
    for index in indices:
        selected.update(range(max(0, index - radius), min(total, index + radius + 1)))
    return selected


def _estimated_tokens(text: str) -> int:
    # Korean and citation-heavy opinions tokenize more densely than ordinary English.
    return max(1, (len(text) + 2) // 3)


def _render_chunk(chunk_segments: list[SourceSegment], chunk_number: int) -> dict[str, object]:
    return {
        "chunk_id": f"CHUNK{chunk_number:03d}",
        "source_sentence_ids": [segment.source_sentence_id for segment in chunk_segments],
        "text": "\n".join(f"<{segment.source_sentence_id}>{segment.text}</{segment.source_sentence_id}>" for segment in chunk_segments),
        "start_source_sentence_id": chunk_segments[0].source_sentence_id,
        "end_source_sentence_id": chunk_segments[-1].source_sentence_id,
        "estimated_tokens": sum(_estimated_tokens(segment.text) for segment in chunk_segments),
    }


def select_candidate_chunks(case: Stage2CaseInput, segments: list[SourceSegment], *, max_input_tokens: int = 12000, overlap_sentences: int = 2, max_chunks: int = 8) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Cover every source segment in ordered chunks; keywords are metadata only."""
    del max_chunks  # Kept for CLI compatibility; it must never truncate source coverage.
    budget = max(256, max_input_tokens)
    full_estimate = sum(_estimated_tokens(segment.text) for segment in segments)
    chunks: list[dict[str, object]] = []
    if segments and full_estimate <= budget:
        chunks = [_render_chunk(segments, 1)]
        method = "full_source_single_ordered_chunk"
    else:
        method = "full_source_multiple_ordered_chunks"
        start = 0
        while start < len(segments):
            end = start
            estimate = 0
            while end < len(segments):
                next_estimate = _estimated_tokens(segments[end].text)
                if end > start and estimate + next_estimate > budget:
                    break
                estimate += next_estimate
                end += 1
            if end == start:  # A single oversized segment is still processed whole.
                end += 1
            chunks.append(_render_chunk(segments[start:end], len(chunks) + 1))
            if end >= len(segments):
                break
            start = max(start + 1, end - max(0, overlap_sentences))
    processed_ids = {source_id for chunk in chunks for source_id in chunk["source_sentence_ids"]}
    processed_segments = [segment for segment in segments if segment.source_sentence_id in processed_ids]
    processed_characters = sum(len(segment.text) for segment in processed_segments)
    source_characters = sum(len(segment.text) for segment in segments)
    missing = [segment.source_sentence_id for segment in segments if segment.source_sentence_id not in processed_ids]
    keyword_hits = [segment.source_sentence_id for segment in segments if any(term.casefold() in segment.text.casefold() for term in FACTUAL_KEYWORDS[case.case_origin])]
    _, section_meta = extract_fact_section_with_metadata(case.source_text, "KR" if case.case_origin == "KR" else "US")
    metadata = {
        "candidate_method": method,
        "fact_section_metadata": section_meta,
        "keyword_priority_source_sentence_ids": keyword_hits,
        "source_segment_count": len(segments),
        "processed_segment_count": len(processed_segments),
        "source_character_count": source_characters,
        "processed_character_count": processed_characters,
        "segment_coverage_ratio": round(len(processed_segments) / max(1, len(segments)), 6),
        "character_coverage_ratio": round(processed_characters / max(1, source_characters), 6),
        "extraction_call_count": len(chunks),
        "coverage_complete": not missing,
        "coverage_incomplete_reason": "" if not missing else "unprocessed_source_segments",
        "missing_source_sentence_ids": missing,
        "full_source_estimated_tokens": full_estimate,
    }
    return chunks, metadata


def segmentation_record(case: Stage2CaseInput, segments: list[SourceSegment], chunks: list[dict[str, object]], metadata: dict[str, object]) -> dict[str, object]:
    return {"case_id": case.case_id, "case_origin": case.case_origin, "source_text_sha256": case.source_text_sha256, "segments": [asdict(segment) for segment in segments], "candidate_chunks": chunks, "candidate_metadata": metadata}
