"""Build the frozen, parser-blind post-hoc human validation package.

This script reads only corpus text and the metadata columns needed for sampling.
It never imports or runs the frozen measurement parsers/matcher.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


SEED = 20260827
TARGET_PER_COUNTRY = 20
ALLOWED_COUNTRIES = ("KR", "US")
ELIGIBLE_CATEGORIES = ("doctrine", "remedy", "procedure", "party_arg")
ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
FEATURES_PATH = ROOT / "outputs_v2/real_reference_confirmatory/reference_features_160.csv"
LEXICON_DIR = ROOT / "legal_lexicon/legal_lexicon_v1_0"
CONCEPTS_PATH = LEXICON_DIR / "concepts_v1_0.csv"
DEFAULT_OUTPUT = ROOT / "outputs_v2/posthoc_measurement_validation_v1"

MANIFEST_FIELDS = [
    "case_id",
    "origin_country",
    "origin_state",
    "primary_domain",
    "doc_length_sentences",
    "doc_length_chars",
    "length_stratum",
    "sampling_seed",
    "selection_rule",
]

TEMPLATE_FIELDS = {
    "sentence_boundary_gold.csv": [
        "case_id",
        "boundary_id",
        "gold_sentence_text",
        "sentence_start",
        "sentence_end",
        "annotator_id",
        "annotator_notes",
    ],
    "citation_gold.csv": [
        "case_id",
        "annotation_id",
        "citation_type",
        "exact_text",
        "start",
        "end",
        "normalized_authority_id",
        "annotator_id",
        "annotator_notes",
    ],
    "lexicon_mention_gold.csv": [
        "case_id",
        "annotation_id",
        "exact_text",
        "start",
        "end",
        "category",
        "concept_id",
        "concept_family",
        "annotator_id",
        "annotator_notes",
    ],
    "out_of_inventory_concepts.csv": [
        "case_id",
        "annotation_id",
        "exact_text",
        "start",
        "end",
        "proposed_category",
        "concept_description",
        "importance",
        "annotator_id",
        "annotator_notes",
    ],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_utf8(path: Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def csv_bytes(rows: list[dict[str, object]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + stream.getvalue().encode("utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seeded_rank(label: str) -> int:
    material = f"{SEED}|{label}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest(), 16)


def proportional_quotas(labels: list[str], target: int, dimension: str) -> dict[str, int]:
    counts = Counter(labels)
    total = sum(counts.values())
    raw = {label: counts[label] * target / total for label in counts}
    quotas = {label: math.floor(value) for label, value in raw.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda label: (-(raw[label] - quotas[label]), seeded_rank(f"quota|{dimension}|{label}")),
    )
    for label in order[:remaining]:
        quotas[label] += 1
    assert sum(quotas.values()) == target
    return dict(sorted(quotas.items()))


def assign_length_strata(rows: list[dict[str, object]]) -> None:
    for country in ALLOWED_COUNTRIES:
        country_rows = [row for row in rows if row["origin_country"] == country]
        ordered = sorted(
            country_rows,
            key=lambda row: (
                int(row["doc_length_sentences"]),
                int(row["doc_length_chars"]),
                str(row["case_id"]),
            ),
        )
        for rank, row in enumerate(ordered):
            quartile = min(4, (rank * 4) // len(ordered) + 1)
            row["length_stratum"] = f"Q{quartile}"


def select_country(
    rows: list[dict[str, object]], country: str
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    country_rows = sorted(
        (row for row in rows if row["origin_country"] == country),
        key=lambda row: str(row["case_id"]),
    )
    dimensions = ["primary_domain", "length_stratum"]
    if country == "US":
        dimensions.append("origin_state")

    quotas: dict[str, dict[str, int]] = {}
    for dimension in dimensions:
        labels = [str(row[dimension] or "") for row in country_rows]
        quotas[dimension] = proportional_quotas(labels, TARGET_PER_COUNTRY, f"{country}|{dimension}")

    constraints: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for dimension in dimensions:
        for label, quota in quotas[dimension].items():
            constraints.append([1.0 if str(row[dimension] or "") == label else 0.0 for row in country_rows])
            lower.append(float(quota))
            upper.append(float(quota))

    constraints.append([1.0] * len(country_rows))
    lower.append(float(TARGET_PER_COUNTRY))
    upper.append(float(TARGET_PER_COUNTRY))

    # A unique fixed-seed SHA256 cost makes the feasible solution deterministic
    # without looking at any legal terminology or parser output.
    costs = np.array(
        [seeded_rank(f"select|{country}|{row['case_id']}") / (2**256) for row in country_rows],
        dtype=float,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(country_rows), dtype=int),
        bounds=Bounds(np.zeros(len(country_rows)), np.ones(len(country_rows))),
        constraints=LinearConstraint(np.asarray(constraints), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Exact stratified allocation was infeasible for {country}: {result.message}")
    selected = [row for row, value in zip(country_rows, result.x) if value > 0.5]
    if len(selected) != TARGET_PER_COUNTRY:
        raise AssertionError(f"Solver selected {len(selected)} {country} cases")
    return selected, quotas


def load_population() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    corpus = read_jsonl(CORPUS_PATH)
    case_ids = [str(row["case_id"]) for row in corpus]
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("Duplicate case_id in frozen corpus")
    corpus_by_id = {str(row["case_id"]): row for row in corpus}
    confirmatory = [row for row in corpus if row.get("analysis_split") == "confirmatory"]
    if Counter(row.get("origin_country") for row in confirmatory) != Counter({"KR": 80, "US": 80}):
        raise AssertionError("Expected confirmatory population KR=80, US=80")

    # Strict whitelist: no parser hits, counts, densities, or extracted values are loaded.
    feature_rows = read_csv(FEATURES_PATH)
    metadata_fields = {
        "case_id",
        "origin_country",
        "origin_state",
        "primary_domain",
        "doc_length_sentences",
        "doc_length_chars",
    }
    population: list[dict[str, object]] = []
    for raw in feature_rows:
        metadata = {field: raw[field] for field in metadata_fields}
        case_id = metadata["case_id"]
        source = corpus_by_id.get(case_id)
        if source is None or source.get("analysis_split") != "confirmatory":
            raise AssertionError(f"Feature row is not a confirmatory corpus case: {case_id}")
        for field in ("origin_country", "origin_state", "primary_domain"):
            source_value = source.get(field) or ""
            if metadata[field] != source_value:
                raise AssertionError(f"Metadata mismatch for {case_id}: {field}")
        metadata["doc_length_sentences"] = int(metadata["doc_length_sentences"])
        metadata["doc_length_chars"] = int(metadata["doc_length_chars"])
        text = source.get("main_opinion_text")
        if not isinstance(text, str) or not text:
            raise AssertionError(f"Missing main_opinion_text: {case_id}")
        if len(text) != metadata["doc_length_chars"]:
            raise AssertionError(f"doc_length_chars mismatch: {case_id}")
        population.append(metadata)

    if len(population) != 160 or len({str(row["case_id"]) for row in population}) != 160:
        raise AssertionError("Feature table must map one-to-one to 160 confirmatory cases")
    if {str(row["case_id"]) for row in population} != {str(row["case_id"]) for row in confirmatory}:
        raise AssertionError("Feature and corpus confirmatory case sets differ")
    assign_length_strata(population)
    return population, corpus_by_id


def build_selection() -> tuple[list[dict[str, object]], dict[str, dict[str, dict[str, int]]]]:
    population, _ = load_population()
    selected: list[dict[str, object]] = []
    quotas: dict[str, dict[str, dict[str, int]]] = {}
    for country in ALLOWED_COUNTRIES:
        country_selected, country_quotas = select_country(population, country)
        selected.extend(country_selected)
        quotas[country] = country_quotas
    selected.sort(key=lambda row: (str(row["origin_country"]), str(row["case_id"])))
    return selected, quotas


def validate_selection(
    selected: list[dict[str, object]],
    quotas: dict[str, dict[str, dict[str, int]]],
    corpus_by_id: dict[str, dict[str, object]],
) -> None:
    if len(selected) != 40 or len({str(row["case_id"]) for row in selected}) != 40:
        raise AssertionError("Sample must contain exactly 40 unique cases")
    if Counter(row["origin_country"] for row in selected) != Counter({"KR": 20, "US": 20}):
        raise AssertionError("Sample must contain KR=20 and US=20")
    for row in selected:
        source = corpus_by_id[str(row["case_id"])]
        if source.get("analysis_split") != "confirmatory":
            raise AssertionError("Development case entered validation sample")
    for country, dimensions in quotas.items():
        country_rows = [row for row in selected if row["origin_country"] == country]
        for dimension, expected in dimensions.items():
            observed = Counter(str(row[dimension] or "") for row in country_rows)
            if dict(sorted(observed.items())) != expected:
                raise AssertionError(f"Stratum quota mismatch: {country} {dimension}")


def manifest_rows(selected: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in selected:
        extra = "; exact proportional origin_state quota" if row["origin_country"] == "US" else ""
        rule = (
            "country quota=20; exact proportional primary_domain and country-specific "
            f"sentence-length-quartile quotas{extra}; minimum fixed-seed SHA256 cost"
        )
        rows.append(
            {
                "case_id": row["case_id"],
                "origin_country": row["origin_country"],
                "origin_state": row["origin_state"],
                "primary_domain": row["primary_domain"],
                "doc_length_sentences": row["doc_length_sentences"],
                "doc_length_chars": row["doc_length_chars"],
                "length_stratum": row["length_stratum"],
                "sampling_seed": SEED,
                "selection_rule": rule,
            }
        )
    return rows


ANNOTATION_GUIDE = """# Parser-Blind Human Annotation Guide

## Non-negotiable blindness rule

Complete and freeze human gold annotation before inspecting any sentence-parser, citation-parser, or lexicon-matcher output. Do not use parser hits, matcher hits, extracted feature values, densities, or validation error logs. The raw judgment text and `eligible_concept_inventory.csv` are the only substantive annotation aids in this package.

The character offsets in all files are zero-based Unicode code-point offsets with a half-open interval: `[start, end)`. Every completed span must satisfy `main_opinion_text[start:end] == exact_text`. Preserve the source text, whitespace, and line breaks. The offset helper uses exactly this convention and only searches raw text.

## Sentence boundaries

Create gold sentences directly from each original `main_opinion_text`; do not pre-fill or consult parser sentence boundaries. Record one row per sentence in document order. Use a case-local stable `boundary_id` such as `S0001`. Put the complete unchanged sentence span in `gold_sentence_text`, with its `sentence_start` and `sentence_end`. Together, the gold spans should account explicitly for all substantive text; document any intentionally excluded separators or whitespace in `annotator_notes`.

## Statutes and precedents

Annotate every citation occurrence in the full selected document. Allowed `citation_type` values are exactly `statute` and `precedent`. Repeated occurrences are separate rows, even when they cite the same authority. Assign the same `normalized_authority_id` to repeated references to the same underlying statute or precedent whenever the authority can be resolved reliably. Use a case-local stable `annotation_id`, for example `C0001`.

## Lexicon mentions

Annotate every textual occurrence that corresponds to an eligible concept in `eligible_concept_inventory.csv`, including repeated mentions. Copy `category`, `concept_id`, and `concept_family` from that inventory. Use a case-local stable `annotation_id`, for example `L0001`.

Keep two questions separate:

- **Matcher surface recall:** an eligible frozen concept is present in the text, but a surface-form matcher later fails to return that occurrence. This is a matcher false negative.
- **Lexicon concept coverage:** an important legal concept has no reasonable `concept_id` in the frozen inventory. Record it only in `out_of_inventory_concepts.csv`; it is not a matcher false negative.

Example: if `jury verdict` is an eligible concept but the later matcher misses `jury verdicts`, that occurrence is a matcher false negative. If `cross-claim` has no corresponding concept in the frozen inventory, it is an out-of-inventory coverage limitation and must not count as a matcher false negative.

## Out-of-inventory concepts

Use `out_of_inventory_concepts.csv` only for an important legal concept with no reasonable eligible `concept_id`. `proposed_category` should normally use one of `doctrine`, `remedy`, `procedure`, or `party_arg`. Describe why it matters and its importance. These entries assess content coverage only and must remain excluded from matcher-recall calculations.

## Independent annotation and freezing

Each annotator works in their own directory (`annotator_A` or `annotator_B`) and supplies a stable `annotator_id` on every row. Do not copy another annotator's work before both versions are frozen. The root `human_annotation` files are blank canonical templates. A second annotator is supported but not required at this stage. Adjudication and parser comparison occur only after gold files have been independently frozen and hashed.
"""


README = """# Post-hoc Measurement Validation v1

This is a frozen, parser-blind human annotation package for 40 confirmatory judgments (KR 20; US 20). It selects cases using metadata only and does not evaluate, tune, import, or execute the frozen measurement pipeline.

## Distribution boundary

Annotators should receive:

- `annotation_packet/`
- `eligible_concept_inventory.csv`
- `ANNOTATION_GUIDE.md`
- one independent template directory under `human_annotation/`
- `annotation_helper/resolve_offsets.py` if desired

Do not give annotators the confirmatory feature table, parser/matcher outputs, densities, hit files, error logs, or any later comparison output. `measurement_validation_sample_manifest.csv` and `sample_manifest.json` are study-administration records; they are not annotation aids.

## Contents

- `measurement_validation_sample_manifest.csv`: frozen sample and stratification metadata.
- `sample_manifest.json`: source hashes, allocation quotas, assertions, and the frozen CSV hash.
- `annotation_packet/annotation_documents.jsonl`: full unchanged `main_opinion_text` plus its SHA256.
- `annotation_packet/documents/*.txt`: byte-exact UTF-8 text exports of each opinion field.
- `eligible_concept_inventory.csv`: eligible frozen concepts only; no surface predictions or hits.
- `human_annotation/`: blank canonical and independent annotator templates.
- `annotation_helper/resolve_offsets.py`: raw-text exact-match offset resolver; ambiguous matches fail.
- `SHA256SUMS.txt`: SHA256 for every package file except itself.

## Sampling design

The sampling seed is `20260827`. Length strata are country-specific rank quartiles of `doc_length_sentences`, with `doc_length_chars` then `case_id` as deterministic tie-breakers. Each country has 20 sampled cases and exact largest-remainder proportional quotas for `primary_domain` and length quartile. The US sample also has exact largest-remainder proportional quotas for `origin_state`. Among feasible samples, fixed-seed SHA256 case costs select one deterministically.

## Next step

Manual parser-blind human gold annotation. Do not compute precision/recall or inspect parser output until completed annotation files are frozen.
"""


OFFSET_HELPER = r'''"""Resolve exact raw-text offsets without parser or matcher assistance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_DOCUMENTS = Path(__file__).resolve().parents[1] / "annotation_packet" / "annotation_documents.jsonl"


def load_case(path: Path, case_id: str) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["case_id"] == case_id:
                return row["main_opinion_text"]
    raise ValueError(f"Unknown case_id: {case_id}")


def resolve(text: str, exact: str, before: str | None, after: str | None) -> list[tuple[int, int]]:
    if not exact:
        raise ValueError("exact_text must not be empty")
    matches = []
    start = 0
    while True:
        index = text.find(exact, start)
        if index < 0:
            break
        end = index + len(exact)
        before_ok = before is None or text[max(0, index - len(before)):index] == before
        after_ok = after is None or text[end:end + len(after)] == after
        if before_ok and after_ok:
            matches.append((index, end))
        start = index + 1
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--exact-text", required=True)
    parser.add_argument("--context-before")
    parser.add_argument("--context-after")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    args = parser.parse_args()
    try:
        text = load_case(args.documents, args.case_id)
        matches = resolve(text, args.exact_text, args.context_before, args.context_after)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not matches:
        print("ERROR: no exact match; check text and immediate context", file=sys.stderr)
        return 3
    if len(matches) != 1:
        print(f"ERROR: ambiguous match ({len(matches)} occurrences); add immediate context", file=sys.stderr)
        for start, end in matches:
            preview = text[max(0, start - 30):min(len(text), end + 30)].replace("\n", "\\n")
            print(f"candidate [{start},{end}): {preview}", file=sys.stderr)
        return 4
    start, end = matches[0]
    if text[start:end] != args.exact_text:
        print("ERROR: internal offset validation failed", file=sys.stderr)
        return 5
    print(json.dumps({"case_id": args.case_id, "exact_text": args.exact_text, "start": start, "end": end}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def make_inventory() -> tuple[list[dict[str, object]], dict[str, int]]:
    source = read_csv(CONCEPTS_PATH)
    rows: list[dict[str, object]] = []
    for row in source:
        if row["category"] not in ELIGIBLE_CATEGORIES:
            continue
        # v1.0 has no description column. A review note is included only when present;
        # it is the sole available short explanatory text and is copied unchanged.
        rows.append(
            {
                "concept_id": row["concept_id"],
                "category": row["category"],
                "concept_family": row["concept_family"],
                "jurisdiction_scope": row["jurisdiction_scope"],
                "short_description_if_available": row.get("short_description", "") or row.get("review_note", ""),
            }
        )
    rows.sort(key=lambda row: (ELIGIBLE_CATEGORIES.index(str(row["category"])), str(row["concept_id"])))
    if len(rows) != len({str(row["concept_id"]) for row in rows}):
        raise AssertionError("Duplicate concept_id in eligible inventory")
    return rows, dict(sorted(Counter(str(row["category"]) for row in rows).items()))


def create_package(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {output}")

    input_hashes_before = {
        "frozen_corpus": sha256_file(CORPUS_PATH),
        "confirmatory_feature_table": sha256_file(FEATURES_PATH),
        "frozen_concepts": sha256_file(CONCEPTS_PATH),
    }
    population, corpus_by_id = load_population()
    selected, quotas = build_selection()
    validate_selection(selected, quotas, corpus_by_id)
    selected_again, quotas_again = build_selection()
    if [row["case_id"] for row in selected] != [row["case_id"] for row in selected_again] or quotas != quotas_again:
        raise AssertionError("Deterministic rerun did not reproduce sample")

    rows = manifest_rows(selected)
    manifest_data = csv_bytes(rows, MANIFEST_FIELDS)
    output.mkdir(parents=True)
    write_bytes(output / "measurement_validation_sample_manifest.csv", manifest_data)
    manifest_hash = sha256_bytes(manifest_data)

    docs: list[dict[str, object]] = []
    for row in selected:
        source = corpus_by_id[str(row["case_id"])]
        text = str(source["main_opinion_text"])
        text_bytes = text.encode("utf-8")
        docs.append(
            {
                "case_id": row["case_id"],
                "origin_country": row["origin_country"],
                "origin_state": row["origin_state"] or None,
                "primary_domain": row["primary_domain"],
                "main_opinion_text": text,
                "main_opinion_sha256": sha256_bytes(text_bytes),
            }
        )
        safe_case_id = str(row["case_id"])
        if not safe_case_id.replace("_", "").isalnum():
            raise AssertionError(f"Unsafe case_id for filename: {safe_case_id}")
        write_bytes(output / "annotation_packet/documents" / f"{safe_case_id}.txt", text_bytes)
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n" for row in docs)
    write_utf8(output / "annotation_packet/annotation_documents.jsonl", jsonl)

    inventory, inventory_counts = make_inventory()
    inventory_fields = [
        "concept_id",
        "category",
        "concept_family",
        "jurisdiction_scope",
        "short_description_if_available",
    ]
    write_bytes(output / "eligible_concept_inventory.csv", csv_bytes(inventory, inventory_fields))

    for name, fields in TEMPLATE_FIELDS.items():
        blank = csv_bytes([], fields)
        write_bytes(output / "human_annotation" / name, blank)
        write_bytes(output / "human_annotation/annotator_A" / name, blank)
        write_bytes(output / "human_annotation/annotator_B" / name, blank)

    write_utf8(output / "ANNOTATION_GUIDE.md", ANNOTATION_GUIDE)
    write_utf8(output / "README.md", README)
    write_utf8(output / "annotation_helper/resolve_offsets.py", OFFSET_HELPER)

    input_hashes_after = {
        "frozen_corpus": sha256_file(CORPUS_PATH),
        "confirmatory_feature_table": sha256_file(FEATURES_PATH),
        "frozen_concepts": sha256_file(CONCEPTS_PATH),
    }
    if input_hashes_before != input_hashes_after:
        raise AssertionError("An input changed while building the package")

    summary = {
        "package_version": "posthoc_measurement_validation_v1",
        "purpose": "post-hoc parser-blind human measurement validation",
        "sampling_seed": SEED,
        "confirmatory_population": {"KR": 80, "US": 80, "total": 160},
        "human_validation_sample": {"KR": 20, "US": 20, "total": 40},
        "source_paths": {
            "frozen_corpus": str(CORPUS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "confirmatory_feature_table": str(FEATURES_PATH.relative_to(ROOT)).replace("\\", "/"),
            "frozen_lexicon": str(LEXICON_DIR.relative_to(ROOT)).replace("\\", "/"),
            "path_note": "Request used outputs_v2/real_reference_confirmatory_v1 and outputs_v2/legal_lexicon_v1_0; repository's frozen assets are at the paths recorded here.",
        },
        "source_sha256_before_and_after": input_hashes_after,
        "metadata_columns_read_for_sampling": [
            "case_id",
            "origin_country",
            "origin_state",
            "primary_domain",
            "doc_length_sentences",
            "doc_length_chars",
        ],
        "length_stratum_definition": "Within-country rank quartile of doc_length_sentences; doc_length_chars and case_id break ties deterministically.",
        "selection_method": "Exact largest-remainder proportional margins solved as binary linear constraints; minimum fixed-seed SHA256 case cost.",
        "stratification_quotas": quotas,
        "measurement_validation_sample_manifest_sha256": manifest_hash,
        "eligible_concept_inventory": {"rows": len(inventory), "category_counts": inventory_counts},
        "assertions": {
            "exactly_40_cases": True,
            "KR_20": True,
            "US_20": True,
            "all_confirmatory": True,
            "development_cases": 0,
            "duplicate_cases": 0,
            "deterministic_rerun_identical": True,
            "parser_predictions_exposed_to_annotator": False,
            "gold_labels_created_automatically": False,
            "lexicon_modified": False,
            "measurement_pipeline_modified": False,
            "primary_analysis_modified": False,
        },
        "next_required_step": "manual parser-blind human gold annotation",
    }
    write_utf8(output / "sample_manifest.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_sums(output)
    validate_package(output, selected, corpus_by_id, manifest_hash)


def write_sums(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*"), key=lambda item: item.relative_to(output).as_posix()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n")
    write_utf8(output / "SHA256SUMS.txt", "".join(entries))


def validate_package(
    output: Path,
    selected: list[dict[str, object]],
    corpus_by_id: dict[str, dict[str, object]],
    expected_manifest_hash: str | None = None,
) -> None:
    manifest_path = output / "measurement_validation_sample_manifest.csv"
    if expected_manifest_hash and sha256_file(manifest_path) != expected_manifest_hash:
        raise AssertionError("Frozen sample manifest hash mismatch")
    manifest = read_csv(manifest_path)
    if len(manifest) != 40 or len({row["case_id"] for row in manifest}) != 40:
        raise AssertionError("Invalid frozen manifest row count or duplicates")
    if Counter(row["origin_country"] for row in manifest) != Counter({"KR": 20, "US": 20}):
        raise AssertionError("Invalid country sample counts")
    for row in manifest:
        if corpus_by_id[row["case_id"]].get("analysis_split") != "confirmatory":
            raise AssertionError("Non-confirmatory case in frozen manifest")

    documents = read_jsonl(output / "annotation_packet/annotation_documents.jsonl")
    if [row["case_id"] for row in documents] != [row["case_id"] for row in selected]:
        raise AssertionError("Annotation documents do not match manifest order")
    expected_doc_fields = {
        "case_id",
        "origin_country",
        "origin_state",
        "primary_domain",
        "main_opinion_text",
        "main_opinion_sha256",
    }
    for row in documents:
        if set(row) != expected_doc_fields:
            raise AssertionError("Unexpected annotation document field")
        text = row["main_opinion_text"]
        source_text = corpus_by_id[str(row["case_id"])]["main_opinion_text"]
        if text != source_text or sha256_bytes(str(text).encode("utf-8")) != row["main_opinion_sha256"]:
            raise AssertionError(f"Source text or hash mismatch: {row['case_id']}")
        text_file = output / "annotation_packet/documents" / f"{row['case_id']}.txt"
        if text_file.read_bytes() != str(source_text).encode("utf-8"):
            raise AssertionError(f"Per-case text changed: {row['case_id']}")

    for folder in [output / "human_annotation", output / "human_annotation/annotator_A", output / "human_annotation/annotator_B"]:
        for name, fields in TEMPLATE_FIELDS.items():
            rows = read_csv(folder / name)
            with (folder / name).open("r", encoding="utf-8-sig", newline="") as handle:
                actual_fields = csv.DictReader(handle).fieldnames
            if rows or actual_fields != fields:
                raise AssertionError(f"Annotation template is not blank/exact: {folder / name}")

    sums = {}
    for line in (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        sums[relative] = digest
    expected_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if set(sums) != expected_files:
        raise AssertionError("SHA256SUMS coverage mismatch")
    for relative, digest in sums.items():
        if sha256_file(output / relative) != digest:
            raise AssertionError(f"SHA256 mismatch: {relative}")


def verify_existing(output: Path) -> None:
    population, corpus_by_id = load_population()
    del population
    selected, quotas = build_selection()
    validate_selection(selected, quotas, corpus_by_id)
    expected = sha256_bytes(csv_bytes(manifest_rows(selected), MANIFEST_FIELDS))
    validate_package(output, selected, corpus_by_id, expected)
    sample_manifest = json.loads((output / "sample_manifest.json").read_text(encoding="utf-8"))
    if sample_manifest["measurement_validation_sample_manifest_sha256"] != expected:
        raise AssertionError("sample_manifest.json records the wrong manifest SHA256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    try:
        if args.verify_only:
            verify_existing(output)
            print(f"VERIFIED: {output}")
        else:
            create_package(output)
            print(f"CREATED_AND_VERIFIED: {output}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
