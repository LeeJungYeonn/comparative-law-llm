#!/usr/bin/env python3
"""Run the fixed v0.2.1-clean matcher on nested tuning 10 only.

This stage generates human-review evidence. It never changes the draft lexicon and
never accesses discovery, validation, or confirmatory opinion text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_draft_v0_2_1_clean"
CONCEPTS_PATH = DRAFT_DIR / "concepts_draft_v0_2_1_clean.csv"
SURFACES_PATH = DRAFT_DIR / "surface_forms_draft_v0_2_1_clean.csv"
MATCHER_PATH = DRAFT_DIR / "lexicon_matcher_draft_v0_2_1_clean.py"
TUNING_MANIFEST_PATH = DRAFT_DIR / "tuning_case_manifest.csv"
VALIDATION_LOCK_PATH = DRAFT_DIR / "validation_case_manifest_LOCKED.csv"
INPUT_REVISION_LOG_PATH = DRAFT_DIR / "tuning_revision_log.csv"
DRAFT_SUMS_PATH = DRAFT_DIR / "SHA256SUMS.txt"
NESTED_SPLIT_PATH = REPO_ROOT / "outputs_v2/lexicon_nested_split_v1/dev_nested_split_manifest.csv"
CORPUS_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
FREEZE_MANIFEST_PATH = REPO_ROOT / "outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json"
OUTPUT_DIR = REPO_ROOT / "outputs_v2/legal_lexicon_tuning_v0_2_1"
FN_SAMPLE_SALT = "tuning_fn_no_hit_span_sample_v0_2_1"
FN_SAMPLE_PER_CASE = 20

HIT_COLUMNS = [
    "hit_id", "case_id", "origin_country", "origin_state", "primary_domain",
    "category", "concept_id", "concept_family", "surface_id", "matched_text",
    "start", "end", "context_before", "context_after", "human_tp_fp",
    "human_category_correct", "human_concept_correct", "human_surface_too_broad",
    "human_notes",
]
FN_COLUMNS = [
    "case_id", "origin_country", "sample_id", "text_span", "start", "end",
    "existing_hit_count", "human_missed_concept", "human_expected_category",
    "human_expected_concept_family", "human_expected_surface", "human_notes",
]
SUMMARY_COLUMNS = [
    "case_id", "origin_country", "primary_domain", "total_hits", "doctrine_hits",
    "remedy_hits", "procedure_hits", "party_arg_hits", "unique_concept_count",
    "unique_concept_family_count", "zero_hit",
]
REVISION_COLUMNS = [
    "revision_id", "surface_or_concept_affected", "before", "after", "reason",
    "evidence_split", "case_id", "error_type", "reviewer_note",
]


@dataclass(frozen=True)
class TuningDocument:
    case_id: str
    origin_country: str
    origin_state: str
    primary_domain: str
    text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def verify_draft_inputs() -> dict[str, str]:
    expected = {}
    for line in DRAFT_SUMS_PATH.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    paths = [
        CONCEPTS_PATH, SURFACES_PATH, MATCHER_PATH, TUNING_MANIFEST_PATH,
        VALIDATION_LOCK_PATH, INPUT_REVISION_LOG_PATH,
    ]
    actual = {path.name: sha256_file(path) for path in paths}
    for path in paths:
        assert path.name in expected, f"Draft SHA256SUMS omits {path.name}"
        assert actual[path.name] == expected[path.name], f"Draft input hash mismatch: {path.name}"
    assert read_csv(INPUT_REVISION_LOG_PATH) == [], "Pre-tuning revision log must be empty"
    return actual


def verify_source_corpus() -> tuple[str, dict]:
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN"
    matches = [digest for name, digest in freeze["artifact_sha256"].items() if Path(name).name == CORPUS_PATH.name]
    assert len(matches) == 1
    actual = sha256_file(CORPUS_PATH)
    assert actual == matches[0], "Frozen corpus hash mismatch"
    return actual, freeze


def verify_tuning_manifests() -> tuple[dict[str, dict], set[str], str]:
    tuning_rows = read_csv(TUNING_MANIFEST_PATH)
    validation_rows = read_csv(VALIDATION_LOCK_PATH)
    nested_rows = read_csv(NESTED_SPLIT_PATH)
    tuning = {row["case_id"]: row for row in tuning_rows}
    validation = {row["case_id"] for row in validation_rows}
    nested_tuning = {row["case_id"]: row for row in nested_rows if row["nested_split"] == "tuning"}
    nested_validation = {row["case_id"] for row in nested_rows if row["nested_split"] == "validation"}
    assert set(tuning) == set(nested_tuning), "Draft tuning manifest differs from frozen nested split"
    assert validation == nested_validation, "Validation lock differs from frozen nested split"
    assert not (set(tuning) & validation)
    assert len(tuning) == 10 and len(validation) == 10
    assert Counter(row["origin_country"] for row in tuning.values()) == Counter({"KR": 5, "US": 5})
    for case_id, row in tuning.items():
        assert row["analysis_split"] == "development"
        assert row["nested_split"] == "tuning"
        assert nested_tuning[case_id]["nested_split"] == "tuning"
    return tuning, validation, sha256_file(NESTED_SPLIT_PATH)


def load_tuning_documents(tuning: dict[str, dict], validation_ids: set[str]) -> list[TuningDocument]:
    """Access main_opinion_text only inside the proven tuning-ID branch."""
    documents = []
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case_id = str(record["case_id"])
            if case_id not in tuning:
                # No opinion-text field access for discovery, validation, or confirmatory records.
                continue
            assert case_id not in validation_ids
            assert record["analysis_split"] == "development"
            assert tuning[case_id]["nested_split"] == "tuning"
            assert record["origin_country"] == tuning[case_id]["origin_country"]
            text = record.get("main_opinion_text")
            assert isinstance(text, str) and text.strip(), f"Empty tuning opinion: {case_id}"
            documents.append(TuningDocument(
                case_id=case_id,
                origin_country=record["origin_country"],
                origin_state=str(record.get("origin_state") or ""),
                primary_domain=str(record.get("primary_domain") or ""),
                text=text,
            ))
    documents.sort(key=lambda item: item.case_id)
    assert {item.case_id for item in documents} == set(tuning)
    assert Counter(item.origin_country for item in documents) == Counter({"KR": 5, "US": 5})
    return documents


def load_fixed_matcher(path: Path = MATCHER_PATH):
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("fixed_lexicon_matcher_v0_2_1_clean", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = prior


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?。！？]+(?:[\"'’”\]\)]*)?(?=\s|$)|\n+", re.UNICODE)


def review_spans(text: str) -> list[tuple[int, int]]:
    """Return deterministic paragraph/sentence spans with exact source offsets."""
    paragraphs = []
    cursor = 0
    for match in PARAGRAPH_BREAK_RE.finditer(text):
        span = _trimmed_span(text, cursor, match.start())
        if span:
            paragraphs.append(span)
        cursor = match.end()
    span = _trimmed_span(text, cursor, len(text))
    if span:
        paragraphs.append(span)

    result = []
    for start, end in paragraphs:
        if end - start <= 800:
            result.append((start, end))
            continue
        local_cursor = start
        for match in SENTENCE_END_RE.finditer(text, start, end):
            sentence = _trimmed_span(text, local_cursor, match.end())
            if sentence:
                result.append(sentence)
            local_cursor = match.end()
        sentence = _trimmed_span(text, local_cursor, end)
        if sentence:
            result.append(sentence)
    return result or [(0, len(text))]


def overlaps(start: int, end: int, hit: dict) -> bool:
    return not (end <= int(hit["start"]) or int(hit["end"]) <= start)


def build_fn_audit(document: TuningDocument, hits: Sequence[dict]) -> list[dict]:
    no_hit_spans = [span for span in review_spans(document.text) if not any(overlaps(*span, hit) for hit in hits)]
    chosen = sorted(no_hit_spans, key=lambda span: (
        stable_hash(f"{document.case_id}|{span[0]}|{span[1]}|{FN_SAMPLE_SALT}"), span[0], span[1]
    ))[:FN_SAMPLE_PER_CASE]
    chosen.sort()
    if not hits:
        assert chosen, f"Zero-hit tuning document lacks FN-review coverage: {document.case_id}"
    rows = []
    for index, (start, end) in enumerate(chosen, 1):
        rows.append({
            "case_id": document.case_id, "origin_country": document.origin_country,
            "sample_id": f"{document.case_id}::FN{index:03d}",
            "text_span": document.text[start:end], "start": start, "end": end,
            "existing_hit_count": 0, "human_missed_concept": "",
            "human_expected_category": "", "human_expected_concept_family": "",
            "human_expected_surface": "", "human_notes": "",
        })
    return rows


def run_matcher(documents: Sequence[TuningDocument], matcher) -> tuple[list[dict], list[dict], list[dict]]:
    hit_rows, summaries, fn_rows = [], [], []
    for document in documents:
        language = "ko" if document.origin_country == "KR" else "en"
        hits = matcher.match_text(document.text, SURFACES_PATH, CONCEPTS_PATH, language)
        normalized_hits = []
        for index, hit in enumerate(hits, 1):
            start, end = int(hit["start"]), int(hit["end"])
            assert document.text[start:end] == hit["matched_text"]
            row = {
                "hit_id": f"{document.case_id}::H{index:05d}",
                "case_id": document.case_id, "origin_country": document.origin_country,
                "origin_state": document.origin_state, "primary_domain": document.primary_domain,
                "category": hit["category"], "concept_id": hit["concept_id"],
                "concept_family": hit["concept_family"], "surface_id": hit["surface_id"],
                "matched_text": hit["matched_text"], "start": start, "end": end,
                "context_before": document.text[max(0, start - 160):start],
                "context_after": document.text[end:end + 160],
                "human_tp_fp": "", "human_category_correct": "",
                "human_concept_correct": "", "human_surface_too_broad": "", "human_notes": "",
            }
            hit_rows.append(row)
            normalized_hits.append(row)
        counts = Counter(hit["category"] for hit in normalized_hits)
        summaries.append({
            "case_id": document.case_id, "origin_country": document.origin_country,
            "primary_domain": document.primary_domain, "total_hits": len(normalized_hits),
            "doctrine_hits": counts["doctrine"], "remedy_hits": counts["remedy"],
            "procedure_hits": counts["procedure"], "party_arg_hits": counts["party_arg"],
            "unique_concept_count": len({hit["concept_id"] for hit in normalized_hits}),
            "unique_concept_family_count": len({hit["concept_family"] for hit in normalized_hits}),
            "zero_hit": str(not normalized_hits).lower(),
        })
        fn_rows.extend(build_fn_audit(document, normalized_hits))
    return hit_rows, summaries, fn_rows


def readme_text() -> str:
    return f"""# Legal lexicon tuning evidence v0.2.1 clean

The fixed, hash-verified `legal_lexicon_draft_v0_2_1_clean` matcher was applied exactly as supplied to
the nested tuning 10 only (KR 5, US 5). No lexicon, pattern, overlap rule, orientation flag, inclusion flag,
or concept mapping was changed. Discovery, validation, and confirmatory opinion text was not accessed.

`tuning_matcher_hits_for_review.csv` contains exact source spans and blank legal-review columns. It does
not assert that any hit is a true positive. `tuning_false_negative_audit.csv` likewise contains no automatic
false-negative judgment.

## Deterministic FN sampling

Opinion text is divided at blank-line paragraph boundaries. Paragraphs longer than 800 characters are
further divided at sentence terminators or line breaks. Spans overlapping any matcher hit are removed.
For each tuning case, up to {FN_SAMPLE_PER_CASE} remaining spans are selected by ascending
`sha256(case_id|start|end|{FN_SAMPLE_SALT})`, then written in source order. Every zero-hit document must
have at least one sampled span. Exact zero-based half-open offsets reproduce `text_span`.

## Stop condition

This run stops before any legal decision or revision. Human reviewers must complete the TP/FP and FN
columns. Any later revision must be recorded in `tuning_revision_log.csv` with evidence split `tuning`,
`discovery`, or `official_resource`; validation and confirmatory evidence are forbidden.
"""


def write_outputs(
    output_dir: Path, hits: Sequence[dict], summaries: Sequence[dict], fn_rows: Sequence[dict],
    input_hashes: dict[str, str], source_hash: str, source_freeze: dict, nested_hash: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tuning_matcher_hits_for_review.csv", hits, HIT_COLUMNS)
    write_csv(output_dir / "tuning_false_negative_audit.csv", fn_rows, FN_COLUMNS)
    write_csv(output_dir / "tuning_matcher_summary.csv", summaries, SUMMARY_COLUMNS)
    write_csv(output_dir / "tuning_revision_log.csv", [], REVISION_COLUMNS)
    (output_dir / "README.md").write_text(readme_text(), encoding="utf-8", newline="\n")

    zero_hits = sum(row["zero_hit"] == "true" for row in summaries)
    manifest = {
        "stage": "tuning_evidence_generation",
        "draft_version": "v0.2.1_clean",
        "draft_status": "fixed_pre_tuning",
        "tuning_cases_used": 10, "KR_tuning": 5, "US_tuning": 5,
        "tuning_case_ids": [row["case_id"] for row in summaries],
        "matcher_hits": len(hits), "zero_hit_documents": zero_hits,
        "fn_review_spans_generated": len(fn_rows),
        "discovery_text_used_in_this_run": False,
        "validation_text_used": False, "confirmatory_text_used": False,
        "lexicon_modified": False, "legal_review_columns_autofilled": False,
        "source_corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_corpus_sha256": source_hash,
        "source_corpus_version": source_freeze["corpus_version"],
        "nested_split_manifest_path": str(NESTED_SPLIT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "nested_split_manifest_sha256": nested_hash,
        "input_hashes": input_hashes,
        "fn_sampling": {
            "rule": "up to 20 non-hit paragraph/sentence spans per case ordered by SHA256(case_id|start|end|salt)",
            "salt": FN_SAMPLE_SALT, "maximum_per_case": FN_SAMPLE_PER_CASE,
        },
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "generation_timestamp_utc": json.loads((DRAFT_DIR / "run_manifest.json").read_text(encoding="utf-8"))["generated_at_utc"],
        "timestamp_policy": "deterministic fixed-draft generation timestamp",
        "randomness_used": False,
        "next_required_step": "human review of TP/FP and FN artifacts",
    }
    (output_dir / "tuning_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii", newline="\n"
    )
    return manifest


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    input_hashes_before = verify_draft_inputs()
    source_hash, source_freeze = verify_source_corpus()
    tuning, validation_ids, nested_hash = verify_tuning_manifests()
    documents = load_tuning_documents(tuning, validation_ids)
    matcher = load_fixed_matcher()
    hits, summaries, fn_rows = run_matcher(documents, matcher)
    assert {row["case_id"] for row in hits + fn_rows + summaries} <= set(tuning)
    assert not ({row["case_id"] for row in hits + fn_rows + summaries} & validation_ids)
    assert {row["case_id"] for row in fn_rows} == set(tuning), "Every tuning case needs no-hit-span audit coverage"
    manifest = write_outputs(output_dir, hits, summaries, fn_rows, input_hashes_before,
                             source_hash, source_freeze, nested_hash)
    assert verify_draft_inputs() == input_hashes_before, "Draft lexicon inputs changed during tuning run"
    assert sha256_file(CORPUS_PATH) == source_hash, "Source corpus changed during tuning run"
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.output_dir.resolve())
    print("draft lexicon verified: yes")
    print(f"tuning cases processed: {result['tuning_cases_used']}")
    print(f"KR tuning: {result['KR_tuning']}")
    print(f"US tuning: {result['US_tuning']}")
    print(f"matcher hits: {result['matcher_hits']}")
    print(f"zero-hit documents: {result['zero_hit_documents']}")
    print(f"FN-review spans generated: {result['fn_review_spans_generated']}")
    print("validation text accessed: 0")
    print("confirmatory text accessed: 0")
    print("lexicon modified: no")
    print(f"output directory: {args.output_dir.resolve()}")
    print("next required step: human review of TP/FP and FN artifacts")


if __name__ == "__main__":
    main()
