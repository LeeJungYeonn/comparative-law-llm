from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import jsonschema

from exp1.common import REPO_ROOT, sha256_file, sha256_text, stable_json

VERSION = "exp1-conclusion-reanalysis-v2"
PROMPT_VERSION = "conclusion_recode_v2"
SCHEMA_PATH = REPO_ROOT / "schemas/exp1_conclusion_recode_v2.schema.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts/exp1/conclusion_recode_v2_system.txt"
USER_PROMPT_PATH = REPO_ROOT / "prompts/exp1/conclusion_recode_v2_user.txt"
CONCLUSIONS = (
    "likely", "unlikely", "mixed_or_partial", "conditional", "uncertain", "not_assessed",
)
PARTY_PREFIXES = (
    "PERSON", "COMPANY", "MEDICAL_INSTITUTION", "PUBLIC_AGENCY",
    "EDUCATIONAL_INSTITUTION", "OTHER",
)
_PREFIX_PATTERN = "|".join(sorted(PARTY_PREFIXES, key=len, reverse=True))
CANONICAL_RE = re.compile(
    rf"(?<![A-Z0-9_])((?:{_PREFIX_PATTERN})_[A-Z0-9]+)(?![A-Z0-9_])",
    flags=re.IGNORECASE,
)
BRACKET_RE = re.compile(
    rf"\[((?:{_PREFIX_PATTERN})_[A-Z0-9]+)\]",
    flags=re.IGNORECASE,
)
ROLE_RE = re.compile(
    r"\b(estate|heirs?|survivors?|family|representative|guardian|유족|상속인?|대리인?|법정대리인)\b",
    flags=re.IGNORECASE,
)


def canonical_ids(value: str) -> list[str]:
    """Extract unique canonical IDs in first-appearance order."""
    return list(dict.fromkeys(match.upper() for match in CANONICAL_RE.findall(value or "")))


def canonicalize_single(value: str) -> str | None:
    ids = canonical_ids(value)
    return ids[0] if len(ids) == 1 else None


def party_type(canonical_party_id: str) -> str:
    return canonical_party_id.rsplit("_", 1)[0]


def source_party_surfaces(text: str) -> dict[str, list[str]]:
    surfaces: dict[str, list[str]] = defaultdict(list)
    for match in BRACKET_RE.finditer(text or ""):
        canonical = match.group(1).upper()
        surface = match.group(0)
        if surface not in surfaces[canonical]:
            surfaces[canonical].append(surface)
    return dict(surfaces)


def existing_evaluator_audit(
    evaluations: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], set[str]]:
    flags: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in evaluations:
        case_id = record["case_id"]
        seen: Counter[str] = Counter()
        for party in record["evaluation"]["parties"]:
            value = str(party["party_id"])
            ids = canonical_ids(value)
            if len(ids) > 1:
                for canonical in ids:
                    flags[(case_id, canonical)].add("existing_grouped_party_string")
            if ROLE_RE.search(value):
                for canonical in ids:
                    flags[(case_id, canonical)].add("existing_attached_legal_role")
            for canonical in ids:
                seen[canonical] += 1
        for canonical, count in seen.items():
            if count > 1:
                flags[(case_id, canonical)].add("existing_duplicate_party_mapping")
    return flags


def build_party_registry(
    accepted_rows: list[dict[str, Any]],
    existing_evaluations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    audit = existing_evaluator_audit(existing_evaluations or [])
    registry: list[dict[str, Any]] = []
    for source in sorted(accepted_rows, key=lambda row: row["case_id"]):
        ko_surfaces = source_party_surfaces(str(source["neutral_fact_ko"]))
        en_surfaces = source_party_surfaces(str(source["neutral_fact_en"]))
        ko_ids, en_ids = set(ko_surfaces), set(en_surfaces)
        mismatch = ko_ids != en_ids
        for canonical in sorted(ko_ids | en_ids):
            flags = set(audit.get((source["case_id"], canonical), set()))
            if mismatch:
                flags.add("source_party_set_mismatch")
            if party_type(canonical) == "OTHER":
                flags.add("ambiguous_other_placeholder_type")
            registry.append({
                "case_id": source["case_id"],
                "case_origin": source["case_origin"],
                "case_subtype": source["case_subtype"],
                "canonical_party_id": canonical,
                "party_type": party_type(canonical),
                "ko_present": canonical in ko_ids,
                "en_present": canonical in en_ids,
                "ko_source_surface_form": " | ".join(ko_surfaces.get(canonical, [])),
                "en_source_surface_form": " | ".join(en_surfaces.get(canonical, [])),
                "source_party_set_mismatch": mismatch,
                "unresolved_source_issue": mismatch,
                "audit_flags": ";".join(sorted(flags)),
            })
    return registry


def registry_by_case(registry: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in registry:
        result[row["case_id"]].append(row)
    return {case_id: sorted(rows, key=lambda row: row["canonical_party_id"]) for case_id, rows in result.items()}


def response_id(raw_record: dict[str, Any]) -> str:
    material = stable_json([
        raw_record["unique_key"], sha256_text(raw_record["raw_response"]), PROMPT_VERSION,
    ])
    return "cr2_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def recode_cache_key(
    raw_record: dict[str, Any], fact_text: str, canonical_parties: list[str], model: str,
) -> str:
    material = {
        "response_id": response_id(raw_record),
        "raw_sha256": sha256_text(raw_record["raw_response"]),
        "fact_sha256": sha256_text(fact_text),
        "canonical_parties": sorted(canonical_parties),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": sha256_file(SYSTEM_PROMPT_PATH),
        "user_prompt_sha256": sha256_file(USER_PROMPT_PATH),
        "schema_sha256": sha256_file(SCHEMA_PATH),
    }
    return sha256_text(stable_json(material))


def dynamic_schema(response: str, language: str, canonical_parties: list[str]) -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema["properties"]["response_id"] = {"type": "string", "enum": [response]}
    schema["properties"]["language"] = {"type": "string", "enum": [language]}
    parties_schema = schema["properties"]["parties"]
    parties_schema["minItems"] = len(canonical_parties)
    parties_schema["maxItems"] = len(canonical_parties)
    parties_schema["items"]["properties"]["canonical_party_id"] = {
        "type": "string", "enum": sorted(canonical_parties),
    }
    return schema


def render_recode_prompt(
    raw_record: dict[str, Any], fact_text: str, canonical_parties: list[str],
) -> tuple[str, str]:
    system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    template = USER_PROMPT_PATH.read_text(encoding="utf-8")
    user = (
        template
        .replace("{response_id}", response_id(raw_record))
        .replace("{language}", raw_record["condition"])
        .replace("{canonical_parties_json}", json.dumps(sorted(canonical_parties), ensure_ascii=False))
        .replace("{neutral_fact}", fact_text)
        .replace("{raw_response}", raw_record["raw_response"])
    )
    return system, user


def validate_recode_payload(
    payload: dict[str, Any], *, expected_response_id: str, expected_language: str,
    expected_parties: list[str],
) -> None:
    schema = dynamic_schema(expected_response_id, expected_language, expected_parties)
    jsonschema.Draft202012Validator(schema).validate(payload)
    parties = payload["parties"]
    returned = [party["canonical_party_id"] for party in parties]
    duplicates = sorted(party for party, count in Counter(returned).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate_canonical_party:{duplicates}")
    expected, actual = set(expected_parties), set(returned)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"canonical_party_set_mismatch:missing={missing}:extra={extra}")
    for party in parties:
        should_be_assessed = party["conclusion"] != "not_assessed"
        if party["assessed"] != should_be_assessed:
            raise ValueError(
                f"assessed_conclusion_inconsistent:{party['canonical_party_id']}"
            )
        if not party["aggregation_note"].strip():
            raise ValueError(f"empty_aggregation_note:{party['canonical_party_id']}")


def validate_flat_record(
    record: dict[str, Any], *, expected_case_id: str, expected_response_id: str,
    expected_language: str, expected_replicate: int,
) -> None:
    required = {
        "case_id", "response_id", "language", "replicate_id", "canonical_party_id",
        "conclusion", "assessed", "supporting_text", "aggregation_note",
        "evaluator_model", "evaluator_prompt_version",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"missing_fields:{missing}")
    if record["case_id"] != expected_case_id:
        raise ValueError("case_id_mismatch")
    if record["response_id"] != expected_response_id:
        raise ValueError("response_id_mismatch")
    if record["language"] != expected_language:
        raise ValueError("language_mismatch")
    if int(record["replicate_id"]) != int(expected_replicate):
        raise ValueError("replicate_mismatch")
    if record["conclusion"] not in CONCLUSIONS:
        raise ValueError("invalid_conclusion")


def aggregate_replicates(labels_by_replicate: dict[int, str]) -> dict[str, Any]:
    ordered = {int(key): value for key, value in sorted(labels_by_replicate.items())}
    if set(ordered) != {1, 2, 3}:
        return {
            "aggregation_status": "incomplete_replicates",
            "consensus_conclusion": None,
            "consensus_count": None,
            "replicate_labels": ordered,
        }
    counts = Counter(ordered.values())
    label, count = counts.most_common(1)[0]
    if count >= 2:
        return {
            "aggregation_status": "consensus",
            "consensus_conclusion": label,
            "consensus_count": count,
            "replicate_labels": ordered,
        }
    return {
        "aggregation_status": "replicate_disagreement",
        "consensus_conclusion": None,
        "consensus_count": 1,
        "replicate_labels": ordered,
    }


def legacy_conclusion_audit(evaluations: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in evaluations:
        grouped[record["case_id"]][record["condition"]].append(record)
    comparisons: list[dict[str, Any]] = []
    case_changes = case_flips = tie_comparisons = 0
    for case_id, conditions in sorted(grouped.items()):
        aggregated: dict[str, tuple[dict[str, list[str]], dict[str, str], dict[str, bool]]] = {}
        for language in ("ko", "en"):
            values: dict[str, list[str]] = defaultdict(list)
            for record in conditions[language]:
                for party in record["evaluation"]["parties"]:
                    values[party["party_id"]].append(party["conclusion"])
            modal, ties = {}, {}
            for party, labels in values.items():
                counts = Counter(labels)
                maximum = max(counts.values())
                modal[party] = counts.most_common(1)[0][0]
                ties[party] = sum(count == maximum for count in counts.values()) > 1
            aggregated[language] = (values, modal, ties)
        case_pairs = []
        for party in sorted(set(aggregated["ko"][1]) & set(aggregated["en"][1])):
            ko_label, en_label = aggregated["ko"][1][party], aggregated["en"][1][party]
            tied = aggregated["ko"][2][party] or aggregated["en"][2][party]
            tie_comparisons += tied
            case_pairs.append((ko_label, en_label))
            comparisons.append({
                "case_id": case_id,
                "exact_party_string": party,
                "ko_modal": ko_label,
                "en_modal": en_label,
                "ko_modal_tie": aggregated["ko"][2][party],
                "en_modal_tie": aggregated["en"][2][party],
                "changed": ko_label != en_label,
                "direct_flip": {ko_label, en_label} == {"likely", "unlikely"},
            })
        case_changes += any(a != b for a, b in case_pairs)
        case_flips += any({a, b} == {"likely", "unlikely"} for a, b in case_pairs)
    pairs = [(row["ko_modal"], row["en_modal"]) for row in comparisons]
    agree = sum(a == b for a, b in pairs) / len(pairs)
    a_counts, b_counts = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(
        (a_counts[label] / len(pairs)) * (b_counts[label] / len(pairs))
        for label in CONCLUSIONS
    )
    kappa = (agree - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "status": "reproduced",
        "exact_string_matched_comparisons": len(pairs),
        "agreement": agree,
        "cohen_kappa_unweighted": kappa,
        "any_conclusion_change_cases": case_changes,
        "case_denominator": len(grouped),
        "direct_likely_unlikely_flip_cases": case_flips,
        "modal_tie_matched_comparisons": tie_comparisons,
        "legacy_method": "exact evaluator party string + Counter.most_common first-label tie breaking",
    }, comparisons
