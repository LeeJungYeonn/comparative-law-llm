#!/usr/bin/env python3
"""One-shot held-out validation of the hash-frozen lexicon candidate v0.3."""

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
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PREVALIDATION_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_tuning_review_v0_3"
CONCEPTS_PATH = PREVALIDATION_DIR / "concepts_prevalidation_v0_3.csv"
SURFACES_PATH = PREVALIDATION_DIR / "surface_forms_prevalidation_v0_3.csv"
MATCHER_PATH = PREVALIDATION_DIR / "lexicon_matcher_prevalidation_v0_3.py"
PREVALIDATION_HASH_PATH = PREVALIDATION_DIR / "PREVALIDATION_SHA256.json"
VALIDATION_MANIFEST_PATH = PREVALIDATION_DIR / "validation_case_manifest_LOCKED.csv"
TUNING_MANIFEST_PATH = PREVALIDATION_DIR / "tuning_case_manifest.csv"
CORPUS_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
FREEZE_MANIFEST_PATH = REPO_ROOT / "outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json"
OUTPUT_DIR = REPO_ROOT / "outputs_v2/legal_lexicon_validation_v1"
FN_SAMPLE_SALT = "validation_fn_no_hit_span_sample_v1"
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
HUMAN_CSV_NAMES = (
    "validation_matcher_hits_for_review.csv",
    "validation_false_negative_audit.csv",
    "validation_matcher_summary.csv",
)


@dataclass(frozen=True)
class ValidationDocument:
    case_id: str
    origin_country: str
    origin_state: str
    primary_domain: str
    text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_human_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def verify_prevalidation_hashes() -> dict[str, str]:
    seal = json.loads(PREVALIDATION_HASH_PATH.read_text(encoding="utf-8"))
    assert seal.get("status") == "PRE_VALIDATION_HASHED_CANDIDATE"
    expected = seal.get("files", {})
    assert expected, "Empty pre-validation hash seal"
    actual = {}
    for name, digest in sorted(expected.items()):
        path = PREVALIDATION_DIR / name
        assert path.is_file(), f"Missing pre-validation file: {name}"
        actual[name] = sha256_file(path)
        assert actual[name] == digest, f"Pre-validation hash mismatch: {name}"
    required = {CONCEPTS_PATH.name, SURFACES_PATH.name, MATCHER_PATH.name}
    assert required <= set(actual)
    return actual


def verify_source_corpus() -> tuple[str, dict]:
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN"
    expected = [value for key, value in freeze["artifact_sha256"].items() if Path(key).name == CORPUS_PATH.name]
    assert len(expected) == 1
    actual = sha256_file(CORPUS_PATH)
    assert actual == expected[0], "Frozen source corpus hash mismatch"
    return actual, freeze


def verify_validation_roster() -> tuple[dict[str, dict], set[str], set[str], set[str]]:
    locked_rows = read_csv(VALIDATION_MANIFEST_PATH)
    tuning_rows = read_csv(TUNING_MANIFEST_PATH)
    validation = {row["case_id"]: row for row in locked_rows}
    tuning = {row["case_id"] for row in tuning_rows}
    development_ids = set()
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["analysis_split"] == "development":
                development_ids.add(str(record["case_id"]))
            # Metadata-only pass: main_opinion_text is never indexed here.
    validation_ids = set(validation)
    discovery = development_ids - validation_ids - tuning
    assert len(validation) == 10
    assert Counter(row["origin_country"] for row in validation.values()) == Counter({"KR": 5, "US": 5})
    assert len(development_ids) == 40 and len(tuning) == 10 and len(discovery) == 20
    assert development_ids == validation_ids | tuning | discovery
    assert not (validation_ids & discovery or validation_ids & tuning or discovery & tuning)
    for row in validation.values():
        assert row["analysis_split"] == "development"
        assert row["nested_split"] == "validation"
    assert all(row["analysis_split"] == "development" and row["nested_split"] == "tuning" for row in tuning_rows)
    return validation, discovery, tuning, validation_ids


def load_validation_documents(validation: dict[str, dict]) -> tuple[list[ValidationDocument], set[str]]:
    """Access main_opinion_text only after membership in the locked validation roster."""
    documents = []
    confirmatory_ids = set()
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case_id = str(record["case_id"])
            if case_id not in validation:
                if record["analysis_split"] == "confirmatory":
                    confirmatory_ids.add(case_id)
                # No opinion text access for discovery, tuning, or confirmatory records.
                continue
            assert record["analysis_split"] == "development"
            assert validation[case_id]["nested_split"] == "validation"
            assert record["origin_country"] == validation[case_id]["origin_country"]
            text = record.get("main_opinion_text")
            assert isinstance(text, str) and text.strip(), f"Empty validation opinion: {case_id}"
            documents.append(ValidationDocument(
                case_id=case_id,
                origin_country=record["origin_country"],
                origin_state=str(record.get("origin_state") or ""),
                primary_domain=str(record.get("primary_domain") or ""),
                text=text,
            ))
    documents.sort(key=lambda item: item.case_id)
    assert {item.case_id for item in documents} == set(validation)
    assert Counter(item.origin_country for item in documents) == Counter({"KR": 5, "US": 5})
    assert len(confirmatory_ids) == 160
    return documents, confirmatory_ids


def load_frozen_matcher():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("frozen_prevalidation_matcher_v0_3", MATCHER_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?。！？]+(?:[\"'’”\]\)]*)?(?=\s|$)|\n+", re.UNICODE)


def review_spans(text: str) -> list[tuple[int, int]]:
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
        local_start = start
        for match in SENTENCE_END_RE.finditer(text, start, end):
            sentence = _trimmed_span(text, local_start, match.end())
            if sentence:
                result.append(sentence)
            local_start = match.end()
        sentence = _trimmed_span(text, local_start, end)
        if sentence:
            result.append(sentence)
    return result or [(0, len(text))]


def overlaps(span: tuple[int, int], hit: dict) -> bool:
    return not (span[1] <= int(hit["start"]) or int(hit["end"]) <= span[0])


def build_fn_rows(document: ValidationDocument, hits: Sequence[dict]) -> list[dict]:
    eligible = [span for span in review_spans(document.text) if not any(overlaps(span, hit) for hit in hits)]
    selected = sorted(eligible, key=lambda span: (
        stable_hash(f"{document.case_id}|{span[0]}|{span[1]}|{FN_SAMPLE_SALT}"), span
    ))[:FN_SAMPLE_PER_CASE]
    selected.sort()
    if not hits:
        assert selected, f"Zero-hit validation document lacks FN audit coverage: {document.case_id}"
    return [{
        "case_id": document.case_id, "origin_country": document.origin_country,
        "sample_id": f"{document.case_id}::VFN{index:03d}",
        "text_span": document.text[start:end], "start": start, "end": end,
        "existing_hit_count": 0, "human_missed_concept": "",
        "human_expected_category": "", "human_expected_concept_family": "",
        "human_expected_surface": "", "human_notes": "",
    } for index, (start, end) in enumerate(selected, 1)]


def run_matcher(documents: Sequence[ValidationDocument], matcher) -> tuple[list[dict], list[dict], list[dict]]:
    hit_rows, summaries, fn_rows = [], [], []
    for document in documents:
        language = "ko" if document.origin_country == "KR" else "en"
        hits = matcher.match_text(document.text, SURFACES_PATH, CONCEPTS_PATH, language)
        case_hits = []
        for index, hit in enumerate(hits, 1):
            start, end = int(hit["start"]), int(hit["end"])
            assert document.text[start:end] == hit["matched_text"]
            row = {
                "hit_id": f"{document.case_id}::VH{index:05d}",
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
            case_hits.append(row)
        counts = Counter(row["category"] for row in case_hits)
        summaries.append({
            "case_id": document.case_id, "origin_country": document.origin_country,
            "primary_domain": document.primary_domain, "total_hits": len(case_hits),
            "doctrine_hits": counts["doctrine"], "remedy_hits": counts["remedy"],
            "procedure_hits": counts["procedure"], "party_arg_hits": counts["party_arg"],
            "unique_concept_count": len({row["concept_id"] for row in case_hits}),
            "unique_concept_family_count": len({row["concept_family"] for row in case_hits}),
            "zero_hit": str(not case_hits).lower(),
        })
        fn_rows.extend(build_fn_rows(document, case_hits))
    return hit_rows, summaries, fn_rows


def readme_text() -> str:
    return f"""# One-shot held-out legal-lexicon validation v1

The SHA-256-sealed pre-validation candidate v0.3 was applied exactly as supplied to the locked validation
10 (KR 5, US 5). Discovery, tuning, and confirmatory opinions were not processed. The lexicon and matcher
were hash-verified before and after execution and were not modified.

This directory contains evaluation review artifacts only. Human TP/FP and FN columns are blank. No term,
regex, mapping, flag, overlap rule, or threshold may be changed in response to these results.

## FN audit sampling

Text is divided at blank-line paragraph boundaries; spans longer than 800 characters are further divided
at sentence terminators or line breaks. Spans overlapping matcher hits are excluded. For every validation
document, up to {FN_SAMPLE_PER_CASE} no-hit spans are selected by ascending
`sha256(case_id|start|end|{FN_SAMPLE_SALT})`, then written in source order. Offsets are zero-based,
half-open, and reproduce the exact source text.

## Metrics boundary

There is no exhaustive gold-span annotation. Therefore recall and F1 are not reported. Reviewed-hit
precision remains unavailable until humans complete `human_tp_fp`. Current reportable quantities are total
matcher hits, zero-hit documents, and deterministic FN-audit document/span coverage.

Next: human evaluation of validation TP/FP and FN artifacts. This review is evaluation only and must not
modify the lexicon. Confirmatory application remains prohibited until the validation review is complete.
"""


def write_outputs(
    output_dir: Path, hits: Sequence[dict], summaries: Sequence[dict], fn_rows: Sequence[dict],
    prevalidation_hashes: dict[str, str], source_hash: str, source_freeze: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_human_csv(output_dir / HUMAN_CSV_NAMES[0], hits, HIT_COLUMNS)
    write_human_csv(output_dir / HUMAN_CSV_NAMES[1], fn_rows, FN_COLUMNS)
    write_human_csv(output_dir / HUMAN_CSV_NAMES[2], summaries, SUMMARY_COLUMNS)
    (output_dir / "README.md").write_text(readme_text(), encoding="utf-8", newline="\n")

    zero_hits = sum(row["zero_hit"] == "true" for row in summaries)
    fn_case_count = len({row["case_id"] for row in fn_rows})
    manifest = {
        "stage": "one_shot_held_out_validation",
        "prevalidation_version": "v0.3",
        "validation_cases_used": 10, "KR_validation": 5, "US_validation": 5,
        "validation_case_ids": [row["case_id"] for row in summaries],
        "discovery_cases_used": 0, "tuning_cases_used": 0, "confirmatory_cases_used": 0,
        "lexicon_modified": False,
        "validation_used_for_lexicon_modification": False,
        "matcher_hits": len(hits), "zero_hit_documents": zero_hits,
        "fn_review_spans": len(fn_rows), "fn_audit_documents_covered": fn_case_count,
        "fn_audit_document_coverage": fn_case_count / 10,
        "reviewed_hit_precision": None, "recall": None, "f1": None,
        "metrics_note": "precision pending human TP/FP review; recall/F1 unavailable without exhaustive gold spans",
        "input_hashes": {
            "concepts": prevalidation_hashes[CONCEPTS_PATH.name],
            "surface_forms": prevalidation_hashes[SURFACES_PATH.name],
            "matcher": prevalidation_hashes[MATCHER_PATH.name],
            "PREVALIDATION_SHA256.json": sha256_file(PREVALIDATION_HASH_PATH),
            "source_corpus": source_hash,
            "validation_manifest": sha256_file(VALIDATION_MANIFEST_PATH),
            "tuning_manifest_boundary_check": sha256_file(TUNING_MANIFEST_PATH),
        },
        "all_sealed_prevalidation_hashes": prevalidation_hashes,
        "source_corpus_version": source_freeze["corpus_version"],
        "fn_sampling": {"salt": FN_SAMPLE_SALT, "maximum_per_case": FN_SAMPLE_PER_CASE},
        "human_review_columns_autofilled": False,
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "generation_timestamp_utc": json.loads((PREVALIDATION_DIR / "run_manifest.json").read_text(encoding="utf-8"))["generated_at_utc"],
        "timestamp_policy": "deterministic pre-validation candidate timestamp",
        "randomness_used": False,
        "next_required_step": "human review of validation TP/FP and FN artifacts; evaluation only, no lexicon modification",
    }
    (output_dir / "validation_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii", newline="\n"
    )
    return manifest


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    sealed_before = verify_prevalidation_hashes()
    source_hash, source_freeze = verify_source_corpus()
    validation, discovery_ids, tuning_ids, validation_ids = verify_validation_roster()
    documents, confirmatory_ids = load_validation_documents(validation)
    processed = {document.case_id for document in documents}
    assert processed == validation_ids
    assert not (processed & discovery_ids or processed & tuning_ids or processed & confirmatory_ids)
    matcher = load_frozen_matcher()
    hits, summaries, fn_rows = run_matcher(documents, matcher)
    assert {row["case_id"] for row in hits + summaries + fn_rows} <= validation_ids
    assert {row["case_id"] for row in fn_rows} == validation_ids, "Every validation case needs FN-audit coverage"
    manifest = write_outputs(output_dir, hits, summaries, fn_rows, sealed_before, source_hash, source_freeze)
    assert verify_prevalidation_hashes() == sealed_before, "Pre-validation files changed during/after validation"
    assert sha256_file(CORPUS_PATH) == source_hash, "Source corpus changed during validation"
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.output_dir.resolve())
    print("pre-validation hashes verified: yes")
    print(f"validation cases processed: {result['validation_cases_used']}")
    print(f"KR validation: {result['KR_validation']}")
    print(f"US validation: {result['US_validation']}")
    print(f"matcher hits: {result['matcher_hits']}")
    print(f"zero-hit documents: {result['zero_hit_documents']}")
    print(f"FN-review spans: {result['fn_review_spans']}")
    print("lexicon modified: no")
    print("validation used for modification: no")
    print("confirmatory text accessed: 0")
    print(f"output directory: {args.output_dir.resolve()}")
    print("next required step: human review of validation TP/FP and FN artifacts, evaluation only - no lexicon modification")


if __name__ == "__main__":
    main()
