from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.llm_client import LLMClient
from pipeline.stage2_schema import Stage2CaseInput
from pipeline.stage2_v3_schema import (
    ATTRIBUTED_STATUSES,
    DATASET_VERSION,
    ENTITY_RELATION_SCHEMA,
    EPISTEMIC_STATUSES,
    EVIDENCE_SCHEMA,
    EXCLUDED_FINAL_STATUSES,
    GROUNDING_SCHEMA,
    NEUTRAL_SCHEMA,
    RELATION_TYPES,
    SCHEMA_VERSION,
    TRANSLATION_SCHEMA,
    TRANSLATION_VERIFIER_SCHEMA,
)


PLACEHOLDER_RE = re.compile(
    r"\[(?:PERSON|COMPANY|MEDICAL_INSTITUTION|PUBLIC_AGENCY|"
    r"EDUCATIONAL_INSTITUTION|PROPERTY|PRODUCT|LOCATION|OTHER)_[A-Z]+\]"
)
ANY_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
CASE_NUMBER_RE = re.compile(
    r"\b(?:\d{2,4}[- ]?(?:cv|ca|app)[- ]?\d+|\d{2,4}[가나다라마바사아자차카타파하]\d+)\b",
    re.I,
)
CURRENCY_RE = re.compile(r"(?:[$₩¥€£]|\b(?:USD|KRW|dollars?|won|원)\b)", re.I)

LEGAL_TERMS = {
    "en": (
        "negligence", "negligent", "negligently", "fault", "liability", "liable", "breach",
        "duty of care", "proximate cause", "comparative fault",
        "contributory negligence", "strict liability", "premises liability",
        "product liability", "malpractice", "unlawful", "unreasonable",
        "inadequate", "improper", "appropriate",
    ),
    "ko": (
        "과실", "책임비율", "책임", "주의의무", "의무 위반", "상당인과관계",
        "불법행위", "위법", "부당", "부적절", "적절", "손해배상", "위자료",
    ),
}
JURISDICTION_TERMS = {
    "en": (
        "california", "republic of korea", "korea", "court of appeal",
        "superior court", "supreme court", "los angeles", "san francisco",
    ),
    "ko": (
        "캘리포니아", "대한민국", "한국", "대법원", "고등법원", "지방법원",
        "항소법원", "로스앤젤레스", "샌프란시스코",
    ),
}
ATTRIBUTION_MARKERS = {
    "en": ("alleged", "claimed", "testified", "according to", "disputed", "assumed"),
    "ko": ("주장", "진술", "증언", "의견", "다투", "전제로", "가정"),
}
EVENT_TYPES = {"action", "omission", "event"}
HARM_TYPES = {"harm", "economic_harm"}


def load_prompt(root: Path, filename: str) -> str:
    return (root / "prompts" / filename).read_text(encoding="utf-8")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coverage_record(segment_record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(segment_record.get("candidate_metadata") or {})
    segments = segment_record.get("segments") or []
    missing = list(metadata.get("missing_source_sentence_ids") or [])
    missing_set = set(missing)
    ranges: list[dict[str, Any]] = []
    current: list[str] = []
    for segment in segments:
        source_id = str(segment["source_sentence_id"])
        if source_id in missing_set:
            current.append(source_id)
        elif current:
            ranges.append({"start": current[0], "end": current[-1]})
            current = []
    if current:
        ranges.append({"start": current[0], "end": current[-1]})
    complete = bool(metadata.get("coverage_complete")) and not missing
    return {
        "case_id": segment_record["case_id"],
        "case_origin": segment_record["case_origin"],
        "source_segment_count": int(metadata.get("source_segment_count") or 0),
        "processed_segment_count": int(metadata.get("processed_segment_count") or 0),
        "source_character_count": int(metadata.get("source_character_count") or 0),
        "processed_character_count": int(metadata.get("processed_character_count") or 0),
        "segment_coverage_ratio": float(metadata.get("segment_coverage_ratio") or 0),
        "character_coverage_ratio": float(metadata.get("character_coverage_ratio") or 0),
        "extraction_call_count": int(metadata.get("extraction_call_count") or 0),
        "unprocessed_segment_ranges": ranges,
        "coverage_status": "complete" if complete else "incomplete",
        "candidate_method": metadata.get("candidate_method"),
        "processed_source_ranges": [
            {
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start_source_sentence_id"],
                "end": chunk["end_source_sentence_id"],
            }
            for chunk in segment_record.get("candidate_chunks") or []
        ],
    }


def _aggregate_provenance(
    provenances: list[dict[str, Any]], client: LLMClient, prompt_version: str
) -> dict[str, Any]:
    usage = {
        key: sum(int((item.get("api_usage") or {}).get(key) or 0) for item in provenances)
        or None
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    return {
        "model": provenances[0].get("model") if provenances else client.model,
        "prompt_version": prompt_version,
        "request_hashes": [item.get("request_hash") for item in provenances],
        "raw_response_paths": [item.get("raw_response_path") for item in provenances],
        "api_usage": usage,
        "cache_hits": sum(bool(item.get("cache_hit")) for item in provenances),
        "new_api_calls": sum(not item.get("cache_hit") and not item.get("mock") for item in provenances),
        "mock": all(bool(item.get("mock")) for item in provenances) if provenances else False,
    }


def extract_evidence(
    case: Stage2CaseInput,
    segment_record: dict[str, Any],
    client: LLMClient,
    root: Path,
) -> dict[str, Any]:
    prompt_name = (
        "extract_evidence_ko_v3.txt" if case.case_origin == "KR"
        else "extract_evidence_en_v3.txt"
    )
    segment_map = {
        str(row["source_sentence_id"]): str(row["text"])
        for row in segment_record.get("segments") or []
    }
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    provenances: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(segment_record.get("candidate_chunks") or [], 1):
        chunk_ids = set(chunk.get("source_sentence_ids") or [])
        payload = {
            "case_id": case.case_id,
            "source_language": case.source_language,
            "chunk_number": chunk_index,
            "source_text_untrusted": (
                "<SOURCE_TEXT_UNTRUSTED>\n"
                + str(chunk.get("text") or "")
                + "\n</SOURCE_TEXT_UNTRUSTED>"
            ),
            "allowed_source_sentence_ids": list(chunk.get("source_sentence_ids") or []),
        }
        result = client.call(
            case_id=case.case_id,
            stage=f"v3_factual_evidence_{str(chunk['chunk_id']).lower()}",
            system_prompt=load_prompt(root, prompt_name),
            user_payload=payload,
            schema=EVIDENCE_SCHEMA,
            required_fields=tuple(EVIDENCE_SCHEMA["required"]),
            prompt_version=prompt_name.removesuffix(".txt"),
            schema_version=SCHEMA_VERSION,
        )
        provenances.append({**result.provenance, "chunk_id": chunk["chunk_id"]})
        if result.payload.get("case_id") != case.case_id:
            errors.append(f"{chunk['chunk_id']}:case_id_mismatch")
            continue
        for unit in result.payload.get("evidence_units") or []:
            source_ids = [str(value) for value in unit.get("source_sentence_ids") or []]
            unknown = [
                source_id for source_id in source_ids
                if source_id not in segment_map or source_id not in chunk_ids
            ]
            if not source_ids or unknown:
                errors.append(
                    f"{chunk['chunk_id']}:{unit.get('provisional_evidence_id')}:"
                    f"unknown_source_sentence_id:{','.join(unknown)}"
                )
                continue
            status = str(unit.get("epistemic_status") or "")
            if status not in EPISTEMIC_STATUSES:
                errors.append(f"{chunk['chunk_id']}:invalid_epistemic_status:{status}")
                continue
            candidates.append(
                {
                    **unit,
                    "source_sentence_ids": source_ids,
                    "exact_excerpts": [segment_map[source_id] for source_id in source_ids],
                    "source_chunk_ids": [chunk["chunk_id"]],
                }
            )
    deduplicated: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for unit in candidates:
        excerpt = " ".join(
            " ".join(unit["exact_excerpts"]).casefold().split()
        )
        actor_object = (
            tuple(sorted(str(value).casefold() for value in unit.get("actor_entity_mentions") or [])),
            tuple(sorted(str(value).casefold() for value in unit.get("object_entity_mentions") or [])),
        )
        key = (
            tuple(unit["source_sentence_ids"]),
            excerpt,
            tuple(sorted(unit.get("fact_types") or [])),
            unit.get("epistemic_status"),
            actor_object,
        )
        if key in seen:
            seen[key]["source_chunk_ids"] = sorted(
                set(seen[key]["source_chunk_ids"] + unit["source_chunk_ids"])
            )
            continue
        seen[key] = unit
        deduplicated.append(unit)
    for index, unit in enumerate(deduplicated, 1):
        unit["evidence_id"] = f"E{index:03d}"
        unit.pop("provisional_evidence_id", None)
    coverage = coverage_record(segment_record)
    coverage["extraction_call_count"] = len(segment_record.get("candidate_chunks") or [])
    status = "fail" if errors or coverage["coverage_status"] != "complete" else "pass"
    return {
        "dataset_version": DATASET_VERSION,
        "case_id": case.case_id,
        "case_origin": case.case_origin,
        "evidence_units": deduplicated,
        "extraction_status": status,
        "validation_errors": sorted(set(errors)),
        "source_coverage": coverage,
        "chunk_model_provenance": provenances,
        "model_provenance": _aggregate_provenance(
            provenances, client, prompt_name.removesuffix(".txt")
        ),
    }


def build_entity_relation_graph(
    case: Stage2CaseInput,
    evidence: dict[str, Any],
    client: LLMClient,
    root: Path,
    segment_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_name = (
        "extract_entity_relations_ko_v3.txt" if case.case_origin == "KR"
        else "extract_entity_relations_en_v3.txt"
    )
    evidence_units = [
        {
            key: unit.get(key)
            for key in (
                "evidence_id", "source_sentence_ids", "exact_excerpts", "fact_types",
                "epistemic_status", "actor_entity_mentions", "object_entity_mentions",
                "materiality",
            )
        }
        for unit in evidence.get("evidence_units") or []
        if unit.get("epistemic_status") not in EXCLUDED_FINAL_STATUSES
    ]
    evidence_hash = stable_hash(evidence_units)
    result = client.call(
        case_id=case.case_id,
        stage="v3_entity_relations",
        system_prompt=load_prompt(root, prompt_name),
        user_payload={
            "case_id": case.case_id,
            "source_language": case.source_language,
            "verified_evidence_untrusted": evidence_units,
            "evidence_hash": evidence_hash,
        },
        schema=ENTITY_RELATION_SCHEMA,
        required_fields=tuple(ENTITY_RELATION_SCHEMA["required"]),
        prompt_version=prompt_name.removesuffix(".txt"),
        schema_version=SCHEMA_VERSION,
        context_hashes={"evidence_hash": evidence_hash},
    )
    return normalize_entity_relation_graph(
        result.payload,
        case_id=case.case_id,
        case_origin=case.case_origin,
        evidence=evidence,
        provenance=result.provenance,
        segment_record=segment_record,
    )


def _alpha_suffix(index: int) -> str:
    value = ""
    current = index
    while current:
        current, remainder = divmod(current - 1, 26)
        value = chr(65 + remainder) + value
    return value


def _role_text(entity: dict[str, Any]) -> str:
    return " ".join(
        str(value).casefold()
        for value in (
            list(entity.get("roles") or [])
            + list(entity.get("source_mentions") or [])
        )
    )


def _normalize_relation_direction(
    relation: dict[str, Any], entity_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    updated = dict(relation)
    subject = entity_map.get(str(updated.get("subject_entity_id")), {})
    obj = entity_map.get(str(updated.get("object_entity_id")), {})
    subject_type = str(subject.get("entity_type") or "")
    object_type = str(obj.get("entity_type") or "")
    subject_roles = _role_text(subject)
    object_roles = _role_text(obj)
    relation_type = str(updated.get("relation_type") or "")
    organization_types = {
        "company", "medical_institution", "public_agency",
        "educational_institution",
    }
    asset_types = {"product", "property"}
    swap = False
    if relation_type == "owned_by":
        swap = subject_type in {"person", *organization_types} and object_type in asset_types
    elif relation_type == "employee_of":
        swap = subject_type in organization_types and object_type == "person"
    elif relation_type == "employer_of":
        swap = subject_type == "person" and object_type in organization_types
    elif relation_type in {"drove", "operated"}:
        swap = subject_type == "product" and object_type == "person"
    elif relation_type in {
        "manufactured", "distributed", "wholesaled", "retailed", "sold",
        "issued_warranty_for", "designed",
    }:
        swap = subject_type == "product" and object_type in organization_types
    elif relation_type in {
        "treated", "examined", "did_not_examine", "prescribed_to",
        "did_not_prescribe_to", "performed_surgery_on",
    }:
        patient_terms = ("patient", "환자", "피해자", "decedent", "망인")
        provider_terms = (
            "doctor", "physician", "surgeon", "medical", "의사", "의료",
            "병원", "간호",
        )
        swap = (
            any(term in subject_roles for term in patient_terms)
            and any(term in object_roles for term in provider_terms)
        )
    elif relation_type == "parent_of":
        parent_terms = ("parent", "father", "mother", "부모", "아버지", "어머니")
        child_terms = ("child", "son", "daughter", "자녀", "아들", "딸")
        swap = (
            any(term in subject_roles for term in child_terms)
            and any(term in object_roles for term in parent_terms)
        )
    elif relation_type == "child_of":
        parent_terms = ("parent", "father", "mother", "부모", "아버지", "어머니")
        child_terms = ("child", "son", "daughter", "자녀", "아들", "딸")
        swap = (
            any(term in subject_roles for term in parent_terms)
            and any(term in object_roles for term in child_terms)
        )
    elif relation_type == "died_in":
        swap = subject_type != "person" and object_type == "person"
    elif relation_type == "injured":
        victim_terms = ("injured", "victim", "patient", "피해자", "환자", "부상")
        swap = (
            subject_type == "person"
            and object_type != "person"
            and any(term in subject_roles for term in victim_terms)
        )
    elif relation_type == "located_at":
        swap = subject_type in {"location", "property"} and object_type not in {
            "location", "property"
        }
    if swap:
        updated["subject_entity_id"], updated["object_entity_id"] = (
            updated.get("object_entity_id"),
            updated.get("subject_entity_id"),
        )
        updated["direction_normalized"] = True
    else:
        updated["direction_normalized"] = False
    return updated


def normalize_entity_relation_graph(
    payload: dict[str, Any],
    *,
    case_id: str,
    case_origin: str,
    evidence: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    segment_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign deterministic anonymous IDs/placeholders and revalidate anchors."""
    errors: list[str] = []
    raw_entities = [dict(item) for item in payload.get("entities") or []]
    raw_relations = [dict(item) for item in payload.get("relations") or []]
    entity_id_map: dict[str, str] = {}
    entities: list[dict[str, Any]] = []
    prefix_by_type = {
        "person": "PERSON",
        "company": "COMPANY",
        "medical_institution": "MEDICAL_INSTITUTION",
        "public_agency": "PUBLIC_AGENCY",
        "educational_institution": "EDUCATIONAL_INSTITUTION",
        "property": "PROPERTY",
        "product": "PRODUCT",
        "location": "LOCATION",
        "other": "OTHER",
    }
    prefix_counts: Counter[str] = Counter()
    for index, raw in enumerate(raw_entities, 1):
        old_id = str(raw.get("entity_id") or f"__missing_{index}")
        new_id = f"ENT{index:03d}"
        entity_id_map[old_id] = new_id
        entity_type = str(raw.get("entity_type") or "other")
        prefix = prefix_by_type.get(entity_type, "OTHER")
        prefix_counts[prefix] += 1
        entities.append({
            **raw,
            "entity_id": new_id,
            "placeholder": f"[{prefix}_{_alpha_suffix(prefix_counts[prefix])}]",
        })
    entity_map = {str(item["entity_id"]): item for item in entities}
    relations: list[dict[str, Any]] = []
    dropped_self_relations: list[str] = []
    for index, raw in enumerate(raw_relations, 1):
        normalized = {
            **raw,
            "relation_id": f"R{index:03d}",
            "subject_entity_id": entity_id_map.get(
                str(raw.get("subject_entity_id") or ""),
                str(raw.get("subject_entity_id") or ""),
            ),
            "object_entity_id": entity_id_map.get(
                str(raw.get("object_entity_id") or ""),
                str(raw.get("object_entity_id") or ""),
            ),
        }
        normalized = _normalize_relation_direction(normalized, entity_map)
        if normalized["subject_entity_id"] == normalized["object_entity_id"]:
            dropped_self_relations.append(str(normalized["relation_id"]))
            continue
        relations.append(normalized)
    for index, relation in enumerate(relations, 1):
        relation["relation_id"] = f"R{index:03d}"
    entity_ids = [str(item.get("entity_id") or "") for item in entities]
    placeholders = [str(item.get("placeholder") or "") for item in entities]
    evidence_source_ids = {
        source_id
        for unit in evidence.get("evidence_units") or []
        for source_id in unit.get("source_sentence_ids") or []
    }
    source_ids = (
        {
            str(row.get("source_sentence_id"))
            for row in (segment_record or {}).get("segments") or []
        }
        or evidence_source_ids
    )
    if payload.get("case_id") != case_id:
        errors.append("case_id_mismatch")
    if not entity_ids or len(entity_ids) != len(set(entity_ids)) or "" in entity_ids:
        errors.append("entity_ids_missing_or_duplicate")
    if len(placeholders) != len(set(placeholders)):
        errors.append("entity_placeholders_duplicate")
    if any(not PLACEHOLDER_RE.fullmatch(value) for value in placeholders):
        errors.append("invalid_entity_placeholder")
    for entity in entities:
        if not set(entity.get("source_sentence_ids") or []) <= source_ids:
            errors.append(f"{entity.get('entity_id')}:unknown_source_sentence_id")
    relation_ids = [str(item.get("relation_id") or "") for item in relations]
    if len(relation_ids) != len(set(relation_ids)) or "" in relation_ids:
        errors.append("relation_ids_missing_or_duplicate")
    entity_id_set = set(entity_ids)
    for relation in relations:
        relation_id = relation.get("relation_id")
        if relation.get("subject_entity_id") not in entity_id_set:
            errors.append(f"{relation_id}:unknown_subject_entity")
        if relation.get("object_entity_id") not in entity_id_set:
            errors.append(f"{relation_id}:unknown_object_entity")
        if relation.get("relation_type") not in RELATION_TYPES:
            errors.append(f"{relation_id}:invalid_relation_type")
        if not set(relation.get("source_sentence_ids") or []) <= source_ids:
            errors.append(f"{relation_id}:unknown_source_sentence_id")
    completeness_warnings = list(payload.get("completeness_warnings") or [])
    completeness_warnings.extend(
        f"dropped_nonreflexive_self_relation:{relation_id}"
        for relation_id in dropped_self_relations
    )
    return {
        "dataset_version": DATASET_VERSION,
        "case_id": case_id,
        "case_origin": case_origin,
        "entities": entities,
        "relations": relations,
        "completeness_warnings": completeness_warnings,
        "graph_hash": stable_hash({"entities": entities, "relations": relations}),
        "graph_status": "fail" if errors else "warning" if completeness_warnings else "pass",
        "validation_errors": sorted(set(errors)),
        "model_provenance": provenance or {},
    }


def _find_terms(text: str, language: str, terms: dict[str, tuple[str, ...]]) -> list[str]:
    lowered = text.casefold()
    hits: list[str] = []
    for term in terms[language]:
        if language == "en":
            if re.search(rf"\b{re.escape(term.casefold())}\b", lowered):
                hits.append(term)
        elif term in text:
            if term == "책임" and all(
                text[match.end():match.end() + 1] == "자"
                for match in re.finditer(re.escape(term), text)
            ):
                continue
            hits.append(term)
    return hits


def source_checks(
    payload: dict[str, Any],
    evidence: dict[str, Any],
    graph: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    fact_units = list(payload.get("fact_units") or [])
    expected_fact_ids = [f"F{index:03d}" for index in range(1, len(fact_units) + 1)]
    actual_fact_ids = [str(unit.get("fact_id") or "") for unit in fact_units]
    if actual_fact_ids != expected_fact_ids:
        errors.append("fact_ids_missing_duplicate_or_out_of_order")
    evidence_ids = {
        str(unit.get("evidence_id"))
        for unit in evidence.get("evidence_units") or []
        if unit.get("epistemic_status") not in EXCLUDED_FINAL_STATUSES
    }
    entity_by_id = {
        str(entity["entity_id"]): entity for entity in graph.get("entities") or []
    }
    relation_by_id = {
        str(relation["relation_id"]): relation for relation in graph.get("relations") or []
    }
    relation_triples: dict[str, tuple[str, str, str]] = {}
    for relation_id, relation in relation_by_id.items():
        subject = entity_by_id.get(str(relation.get("subject_entity_id")), {})
        obj = entity_by_id.get(str(relation.get("object_entity_id")), {})
        relation_triples[relation_id] = (
            str(subject.get("placeholder") or ""),
            str(relation.get("relation_type") or ""),
            str(obj.get("placeholder") or ""),
        )
    realized_material: set[str] = set()
    for unit in fact_units:
        fact_id = str(unit.get("fact_id") or "")
        linked_evidence = set(unit.get("source_evidence_ids") or [])
        if not linked_evidence or not linked_evidence <= evidence_ids:
            errors.append(f"{fact_id}:invalid_source_evidence_ids")
        status = str(unit.get("epistemic_status") or "")
        if status not in EPISTEMIC_STATUSES:
            errors.append(f"{fact_id}:invalid_epistemic_status")
        if status in EXCLUDED_FINAL_STATUSES:
            errors.append(f"{fact_id}:excluded_epistemic_status")
        if status in ATTRIBUTED_STATUSES and not any(
            marker in str(unit.get("master_text") or "").casefold()
            for marker in ATTRIBUTION_MARKERS[language]
        ):
            warnings.append(f"{fact_id}:attribution_not_lexically_explicit")
        linked_relations = [str(value) for value in unit.get("relation_ids") or []]
        if any(relation_id not in relation_by_id for relation_id in linked_relations):
            errors.append(f"{fact_id}:unknown_relation_id")
        realized = {
            (
                str(item.get("subject_placeholder") or ""),
                str(item.get("relation_type") or ""),
                str(item.get("object_placeholder") or ""),
            )
            for item in unit.get("realized_relations") or []
        }
        expected = {
            relation_triples[relation_id]
            for relation_id in linked_relations if relation_id in relation_triples
        }
        if realized != expected:
            errors.append(f"{fact_id}:realized_relation_mismatch")
        for relation_id in linked_relations:
            if relation_by_id.get(relation_id, {}).get("material_relation"):
                realized_material.add(relation_id)
    material_ids = {
        relation_id for relation_id, relation in relation_by_id.items()
        if relation.get("material_relation")
    }
    missing_material = sorted(material_ids - realized_material)
    if missing_material:
        errors.append("material_relations_not_realized:" + ",".join(missing_material))
    text = str(payload.get("master_neutral_text") or "")
    composed = " ".join(
        str(unit.get("master_text") or "").strip()
        for unit in fact_units if str(unit.get("master_text") or "").strip()
    )
    if composed != text.strip():
        errors.append("master_text_not_exact_fact_unit_join")
    legal = _find_terms(text, language, LEGAL_TERMS)
    jurisdiction = _find_terms(text, language, JURISDICTION_TERMS)
    if legal:
        errors.append("legal_conclusion_leakage")
    if jurisdiction:
        errors.append("jurisdiction_leakage")
    if CURRENCY_RE.search(text):
        errors.append("currency_or_exact_award_leakage")
    if CASE_NUMBER_RE.search(text):
        errors.append("case_number_leakage")
    placeholders = set(PLACEHOLDER_RE.findall(text))
    invalid_placeholders = [
        value for value in ANY_PLACEHOLDER_RE.findall(text)
        if not PLACEHOLDER_RE.fullmatch(value)
    ]
    if invalid_placeholders:
        errors.append("invalid_placeholder")
    graph_placeholders = {
        str(entity.get("placeholder") or "") for entity in graph.get("entities") or []
    }
    if not placeholders <= graph_placeholders:
        errors.append("placeholder_not_in_entity_graph")
    event_positions = [
        index for index, unit in enumerate(fact_units)
        if set(unit.get("fact_types") or []) & EVENT_TYPES
    ]
    harm_positions = [
        index for index, unit in enumerate(fact_units)
        if set(unit.get("fact_types") or []) & HARM_TYPES
    ]
    event_present = bool(event_positions)
    harm_present = bool(harm_positions)
    sequence_present = bool(
        event_positions and harm_positions and min(event_positions) <= max(harm_positions)
    )
    if not fact_units:
        errors.append("no_fact_units")
    if not event_present:
        errors.append("missing_core_event")
    if not harm_present:
        errors.append("missing_harm")
    if not sequence_present:
        errors.append("missing_event_harm_sequence")
    if graph.get("graph_status") == "fail":
        errors.append("entity_relation_graph_invalid")
    coverage_complete = (
        (evidence.get("source_coverage") or {}).get("coverage_status") == "complete"
    )
    if not coverage_complete:
        errors.append("source_coverage_incomplete")
    return {
        "status": "fail" if errors else "warning" if warnings else "pass",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "coverage_status": "complete" if coverage_complete else "incomplete",
        "event_present": event_present,
        "harm_present": harm_present,
        "event_harm_sequence_present": sequence_present,
        "deterministic_factual_sufficiency": "pass" if (
            event_present and harm_present and sequence_present
        ) else "fail",
        "legal_terms": legal,
        "jurisdiction_terms": jurisdiction,
        "material_relation_ids": sorted(material_ids),
        "realized_material_relation_ids": sorted(realized_material),
        "missing_material_relation_ids": missing_material,
        "placeholder_set": sorted(placeholders),
        "entity_placeholder_set": sorted(graph_placeholders),
    }


def neutralize(
    case: Stage2CaseInput,
    evidence: dict[str, Any],
    graph: dict[str, Any],
    client: LLMClient,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_name = (
        "neutralize_ko_v6.txt" if case.case_origin == "KR" else "neutralize_en_v6.txt"
    )
    admissible_evidence = [
        {
            key: unit.get(key)
            for key in (
                "evidence_id", "source_sentence_ids", "exact_excerpts", "fact_types",
                "epistemic_status", "epistemic_status_confidence",
                "actor_entity_mentions", "object_entity_mentions", "materiality",
            )
        }
        for unit in evidence.get("evidence_units") or []
        if unit.get("epistemic_status") not in EXCLUDED_FINAL_STATUSES
    ]
    graph_payload = {
        "entities": graph.get("entities") or [],
        "relations": graph.get("relations") or [],
    }
    graph_hash = str(graph.get("graph_hash") or stable_hash(graph_payload))
    result = client.call(
        case_id=case.case_id,
        stage="v3_source_neutral",
        system_prompt=load_prompt(root, prompt_name),
        user_payload={
            "case_id": case.case_id,
            "master_language": case.source_language,
            "verified_evidence_untrusted": admissible_evidence,
            "entity_relation_graph": graph_payload,
            "entity_relation_graph_hash": graph_hash,
        },
        schema=NEUTRAL_SCHEMA,
        required_fields=tuple(NEUTRAL_SCHEMA["required"]),
        prompt_version=prompt_name.removesuffix(".txt"),
        schema_version=SCHEMA_VERSION,
        context_hashes={"entity_relation_graph_hash": graph_hash},
    )
    payload = dict(result.payload)
    payload["fact_units"] = [dict(unit) for unit in payload.get("fact_units") or []]
    payload["master_neutral_text"] = " ".join(
        str(unit.get("master_text") or "").strip()
        for unit in payload["fact_units"] if str(unit.get("master_text") or "").strip()
    )
    checks = source_checks(payload, evidence, graph, case.source_language)
    model_insufficient = bool(payload.get("insufficient_factual_detail"))
    status = (
        "fail" if checks["status"] == "fail"
        else "warning" if checks["status"] == "warning" or model_insufficient
        else "pass"
    )
    record = {
        "dataset_version": DATASET_VERSION,
        "case_id": case.case_id,
        "case_origin": case.case_origin,
        "case_subtype": case.case_subtype,
        "source_language": case.source_language,
        "master_language": case.source_language,
        "source_text_field": case.source_text_field,
        "source_text_sha256": case.source_text_sha256,
        "source_dataset": case.source_dataset,
        "source_record_id": case.source_record_id,
        "input_file_sha256": case.input_file_sha256,
        "master_neutral_text": payload["master_neutral_text"],
        "fact_units": payload["fact_units"],
        "removed_legal_signals": payload.get("removed_legal_signals") or [],
        "removed_jurisdiction_signals": payload.get("removed_jurisdiction_signals") or [],
        "anonymization_warnings": payload.get("anonymization_warnings") or [],
        "grounding_warnings": (
            list(payload.get("grounding_warnings") or []) + checks["warnings"]
        ),
        "model_insufficient_factual_detail": model_insufficient,
        "deterministic_factual_sufficiency": checks[
            "deterministic_factual_sufficiency"
        ],
        "case_is_usable_for_translation": status in {"pass", "warning"},
        "case_is_finally_usable": None,
        "neutralization_status": status,
        "source_coverage": evidence.get("source_coverage") or {},
        "entity_relation_graph_hash": graph_hash,
        "deterministic_checks": checks,
        "model_provenance": result.provenance,
    }
    check_record = {
        "case_id": case.case_id,
        "case_origin": case.case_origin,
        **checks,
    }
    return record, check_record


def verify_grounding(
    master: dict[str, Any],
    evidence: dict[str, Any],
    graph: dict[str, Any],
    client: LLMClient,
    root: Path,
) -> dict[str, Any]:
    prompt_name = (
        "verify_grounding_and_roles_ko_v4.txt"
        if master["case_origin"] == "KR"
        else "verify_grounding_and_roles_en_v4.txt"
    )
    graph_hash = str(graph.get("graph_hash") or "")
    request = {
        "case_id": master["case_id"],
        "verified_evidence_untrusted": [
            {
                "evidence_id": unit.get("evidence_id"),
                "source_sentence_ids": unit.get("source_sentence_ids"),
                "exact_excerpts": unit.get("exact_excerpts"),
                "epistemic_status": unit.get("epistemic_status"),
            }
            for unit in evidence.get("evidence_units") or []
            if unit.get("epistemic_status") not in EXCLUDED_FINAL_STATUSES
        ],
        "entity_relation_graph": {
            "entities": graph.get("entities") or [],
            "relations": graph.get("relations") or [],
        },
        "fact_units": master.get("fact_units") or [],
        "master_neutral_text": master.get("master_neutral_text") or "",
        "entity_relation_graph_hash": graph_hash,
    }
    result = client.call(
        case_id=master["case_id"],
        stage="v3_grounding_role_verifier",
        system_prompt=load_prompt(root, prompt_name),
        user_payload=request,
        schema=GROUNDING_SCHEMA,
        required_fields=tuple(GROUNDING_SCHEMA["required"]),
        prompt_version=prompt_name.removesuffix(".txt"),
        schema_version=SCHEMA_VERSION,
        context_hashes={"entity_relation_graph_hash": graph_hash},
    )
    return validate_grounding_payload(
        dict(result.payload),
        master,
        model_provenance=result.provenance,
    )


def validate_grounding_payload(
    payload: dict[str, Any],
    master: dict[str, Any],
    *,
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute verifier consistency without making an API request."""
    payload = dict(payload)
    missing_material_facts = list(payload.get("missing_material_facts") or [])
    legal_missing_pattern = re.compile(
        r"(?:court|judgment|appeal|procedur|statut|evidentiar|"
        r"negligen|liabilit|fault|duty|breach|damages?|"
        r"법원|판결|항소|절차|법률|증거|과실|책임|의무|위반|손해액|배상)",
        re.I,
    )
    excluded_missing_legal_findings = list(
        payload.get("excluded_missing_legal_findings") or []
    ) + [
        item for item in missing_material_facts
        if legal_missing_pattern.search(str(item))
    ]
    payload["missing_material_facts"] = [
        item for item in missing_material_facts
        if item not in excluded_missing_legal_findings
    ]
    hard_fields = (
        "unsupported_fact_ids", "overstated_fact_ids", "epistemic_status_errors",
        "entity_role_errors", "role_relation_errors", "subject_object_errors",
        "ownership_employment_errors", "medical_provider_role_errors",
        "product_chain_role_errors", "material_relation_errors",
        "legal_conclusion_leakage", "fault_allocation_leakage",
        "causation_conclusion_leakage", "damages_calculation_leakage",
        "jurisdiction_leakage",
    )
    reasons = [field for field in hard_fields if payload.get(field)]
    model_grounded = payload.get("model_grounded", payload.get("grounded")) is True
    if not model_grounded:
        reasons.append("grounded_false")
    warnings = []
    if payload.get("missing_material_facts"):
        warnings.append("missing_material_facts")
    if payload.get("evidentiary_evaluation_leakage"):
        warnings.append("evidentiary_evaluation_leakage")
    deterministic_fail = master.get("neutralization_status") == "fail"
    if deterministic_fail:
        reasons.append("deterministic_source_failure_not_acknowledged")
    model_status = str(
        payload.get("model_verifier_status")
        or payload.get("verifier_status")
        or "fail"
    )
    validated = (
        "fail" if reasons
        else "warning" if warnings
        else "fail" if model_status == "fail"
        else "pass"
    )
    return {
        **payload,
        "model_grounded": model_grounded,
        "excluded_missing_legal_findings": list(dict.fromkeys(
            excluded_missing_legal_findings
        )),
        "model_verifier_status": model_status,
        "validated_verifier_status": validated,
        "verifier_status": validated,
        "verifier_consistency_violation": model_status == "pass" and validated != "pass",
        "deterministic_verifier_reasons": reasons + warnings,
        "verification_type": "source_grounding_and_roles",
        "model_provenance": (
            model_provenance
            if model_provenance is not None
            else payload.get("model_provenance") or {}
        ),
    }


EN_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-three": 23, "twenty-five": 25,
}
KO_NUMBER_WORDS = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3,
    "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8,
    "아홉": 9, "열": 10,
}


def canonical_quantity_tokens(text: str) -> Counter[str]:
    value = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []

    def consume(pattern: str, label: str, transform=None) -> None:
        nonlocal value
        regex = re.compile(pattern, re.I)

        def replacement(match: re.Match[str]) -> str:
            groups = match.groups()
            normalized = transform(*groups) if transform else ":".join(groups)
            tokens.append(f"{label}:{normalized}")
            return " "

        value = regex.sub(replacement, value)

    consume(
        r"(\d+)\s*분의\s*(\d+)",
        "fraction",
        lambda denominator, numerator: f"{int(numerator)}/{int(denominator)}",
    )
    fractions = {
        r"\b(?:one|a)[ -]?third\b": "1/3",
        r"\b(?:one|a)[ -]?half\b": "1/2",
    }
    for pattern, normalized in fractions.items():
        if re.search(pattern, value):
            tokens.append(f"fraction:{normalized}")
            value = re.sub(pattern, " ", value)
    en_context = (
        r"(weeks?|months?|years?|percent|meters?|kilometers?|kilograms?|"
        r"lanes?|tires?|feet|foot|miles?|persons?|people)"
    )
    for word, number in sorted(EN_NUMBER_WORDS.items(), key=lambda item: -len(item[0])):
        value = re.sub(
            rf"\b{re.escape(word)}\b(?=[ -]+{en_context}\b)",
            str(number),
            value,
            flags=re.I,
        )
    ko_context = r"(?:주|개월|년|퍼센트|미터|킬로미터|킬로그램|차로|차선|피트|마일|명|개)"
    for word, number in sorted(KO_NUMBER_WORDS.items(), key=lambda item: -len(item[0])):
        value = re.sub(rf"{re.escape(word)}(?=\s*{ko_context})", str(number), value)
    consume(r"(\d+(?:\.\d+)?)\s*(?:%|percent|퍼센트)", "percent")
    consume(r"(\d+(?:\.\d+)?)\s*(?:years?\s+old|세)", "age")
    consume(r"시속\s*(\d+(?:\.\d+)?)\s*킬로미터", "km/h")
    unit_patterns = {
        "week": r"(\d+(?:\.\d+)?)\s*(?:weeks?|주)",
        "month": r"(\d+(?:\.\d+)?)\s*(?:months?|개월)",
        "year": r"(\d+(?:\.\d+)?)\s*(?:years?|년)",
        "km/h": r"(\d+(?:\.\d+)?)\s*(?:km\s*/\s*h|kilometers?\s+per\s+hour|킬로미터\s*(?:/\s*시|매시))",
        "m": r"(\d+(?:\.\d+)?)\s*(?:m(?![a-z])|meters?|metres?|미터)",
        "km": r"(\d+(?:\.\d+)?)\s*(?:km(?![a-z/])|kilometers?|kilometres?|킬로미터)",
        "kg": r"(\d+(?:\.\d+)?)\s*(?:kg(?![a-z])|kilograms?|킬로그램)",
        "foot": r"(\d+(?:\.\d+)?)\s*(?:feet|foot|피트)",
        "mile": r"(\d+(?:\.\d+)?)\s*(?:miles?|마일)",
        "lane": r"(\d+(?:\.\d+)?)\s*(?:[- ]?lanes?|차로|차선)",
    }
    for unit, pattern in unit_patterns.items():
        consume(pattern, unit)
    value = PLACEHOLDER_RE.sub(" ", value)
    for number in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", value):
        tokens.append(f"number:{float(number):g}")
    return Counter(tokens)


def _relation_signature(unit: dict[str, Any]) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    relation_ids = tuple(str(value) for value in unit.get("relation_ids") or [])
    triples = tuple(
        sorted(
            (
                str(item.get("subject_placeholder") or ""),
                str(item.get("relation_type") or ""),
                str(item.get("object_placeholder") or ""),
            )
            for item in unit.get("realized_relations") or []
        )
    )
    return relation_ids, triples


def _semantic_anchors(text: str) -> set[str]:
    lowered = text.casefold()
    groups = {
        "death": ("died", "death", "killed", "사망", "숨졌다", "죽었다"),
        "minor_injury": ("minor injury", "slight injury", "경상"),
        "collision": (
            "collided", "collision", "struck", "impact", "충돌", "충격",
            "부딪", "맞았", "맞았다",
        ),
        "crushing": (
            "crushed", "caught between", "run over", "끼어", "끼였",
            "깔려", "깔렸", "압착", "역과",
        ),
        "surgery": ("surgery", "operation", "수술"),
        "telephone": ("telephone", "phone", "전화"),
        "treatment": ("treated", "treatment", "치료"),
        "manufacturing": ("manufactured", "manufacturer", "제조"),
        "distribution": ("distributed", "distributor", "유통", "도매"),
    }
    return {
        name for name, terms in groups.items()
        if any(term in lowered for term in terms)
    }


def translation_checks(
    master: dict[str, Any],
    translated: dict[str, Any],
    target_language: str,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    master_units = list(master.get("fact_units") or [])
    translated_units = list(translated.get("translated_fact_units") or [])
    master_ids = [str(unit.get("fact_id") or "") for unit in master_units]
    translated_ids = [str(unit.get("fact_id") or "") for unit in translated_units]
    if master_ids != translated_ids:
        errors.append("fact_id_or_order_mismatch")
    master_by_id = {str(unit.get("fact_id")): unit for unit in master_units}
    translated_by_id = {str(unit.get("fact_id")): unit for unit in translated_units}
    number_details: dict[str, Any] = {}
    relation_details: dict[str, Any] = {}
    placeholder_details: dict[str, Any] = {}
    if master_ids == translated_ids:
        for fact_id in master_ids:
            source = master_by_id[fact_id]
            target = translated_by_id[fact_id]
            source_text = str(source.get("master_text") or "")
            target_text = str(target.get("translated_text") or "")
            source_placeholders = set(PLACEHOLDER_RE.findall(source_text))
            target_placeholders = set(PLACEHOLDER_RE.findall(target_text))
            placeholder_details[fact_id] = {
                "master": sorted(source_placeholders),
                "translation": sorted(target_placeholders),
                "match": source_placeholders == target_placeholders,
            }
            if source_placeholders != target_placeholders:
                errors.append(f"{fact_id}:placeholder_identity_mismatch")
            source_quantities = canonical_quantity_tokens(source_text)
            target_quantities = canonical_quantity_tokens(target_text)
            number_details[fact_id] = {
                "master": dict(source_quantities),
                "translation": dict(target_quantities),
                "match": source_quantities == target_quantities,
            }
            if source_quantities != target_quantities:
                errors.append(f"{fact_id}:numerical_or_unit_value_changed")
            source_anchors = _semantic_anchors(source_text)
            target_anchors = _semantic_anchors(target_text)
            if source_anchors != target_anchors:
                errors.append(f"{fact_id}:independent_fact_added_omitted_or_role_changed")
            source_relation = _relation_signature(source)
            target_relation = _relation_signature(target)
            relation_details[fact_id] = {
                "master": source_relation,
                "translation": target_relation,
                "match": source_relation == target_relation,
            }
            if source_relation != target_relation:
                errors.append(f"{fact_id}:material_relation_mismatch")
            source_norm = source_text.casefold()
            target_norm = target_text.casefold()
            source_undisputed = "다툼이 없" in source_norm or "undisputed" in source_norm
            target_undisputed = "다툼이 없" in target_norm or "undisputed" in target_norm
            source_negative = bool(
                re.search(r"\b(?:not|no|never|without|didn't|wasn't)\b", source_norm)
                or re.search(r"(않|없|못|아니)", source_norm)
            ) and not source_undisputed
            target_negative = bool(
                re.search(r"\b(?:not|no|never|without|didn't|wasn't)\b", target_norm)
                or re.search(r"(않|없|못|아니)", target_norm)
            ) and not target_undisputed
            if source_negative != target_negative:
                warnings.append(f"{fact_id}:negation_surface_mismatch_requires_verifier")
            source_knows = bool(re.search(r"\b(?:knew|aware)\b|알고\s*있", source_norm))
            source_not_knows = bool(
                re.search(r"\b(?:did not know|didn't know|unaware)\b|알지\s*못|몰랐", source_norm)
            )
            target_knows = bool(re.search(r"\b(?:knew|aware)\b|알고\s*있", target_norm))
            target_not_knows = bool(
                re.search(r"\b(?:did not know|didn't know|unaware)\b|알지\s*못|몰랐", target_norm)
            )
            if (source_knows and target_not_knows) or (source_not_knows and target_knows):
                errors.append(f"{fact_id}:clear_polarity_reversal")
    master_text = str(master.get("master_neutral_text") or "")
    translated_text = " ".join(
        str(unit.get("translated_text") or "").strip()
        for unit in translated_units if str(unit.get("translated_text") or "").strip()
    )
    if translated_text != str(translated.get("translated_neutral_text") or "").strip():
        errors.append("translated_text_not_exact_fact_unit_join")
    source_placeholder_occurrences = Counter(PLACEHOLDER_RE.findall(master_text))
    target_placeholder_occurrences = Counter(PLACEHOLDER_RE.findall(translated_text))
    if set(source_placeholder_occurrences) != set(target_placeholder_occurrences):
        errors.append("placeholder_identity_mismatch")
    elif source_placeholder_occurrences != target_placeholder_occurrences:
        warnings.append("placeholder_occurrence_count_differs")
    legal = _find_terms(translated_text, target_language, LEGAL_TERMS)
    jurisdiction = _find_terms(translated_text, target_language, JURISDICTION_TERMS)
    if legal:
        errors.append("legal_term_reintroduced")
    if jurisdiction:
        errors.append("jurisdiction_term_reintroduced")
    residue: list[str] = []
    cleaned = PLACEHOLDER_RE.sub(" ", translated_text)
    if target_language == "en":
        residue = sorted(set(re.findall(r"[가-힣]{2,}", cleaned)))
    else:
        allowed = {"km", "kg", "m", "cm", "mm"}
        residue = sorted(
            {
                word for word in re.findall(r"\b[A-Za-z]{3,}\b", cleaned)
                if word.casefold() not in allowed
            }
        )
    if residue:
        warnings.append("target_language_residue")
    return {
        "status": "fail" if errors else "warning" if warnings else "pass",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "fact_id_match": master_ids == translated_ids,
        "placeholder_identity_match": (
            set(source_placeholder_occurrences) == set(target_placeholder_occurrences)
        ),
        "placeholder_occurrence_match": (
            source_placeholder_occurrences == target_placeholder_occurrences
        ),
        "placeholder_comparison": placeholder_details,
        "number_unit_normalization": number_details,
        "relation_comparison": relation_details,
        "translation_relation_status": (
            "fail" if any("relation_mismatch" in error for error in errors) else "pass"
        ),
        "target_language_residue": residue,
        "temporal_order_preserved": master_ids == translated_ids,
        "negation_warning": any("negation" in warning for warning in warnings),
    }


def translate(
    master: dict[str, Any],
    client: LLMClient,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    origin = str(master["case_origin"])
    prompt_name = (
        "translate_ko_to_en_v4.txt" if origin == "KR" else "translate_en_to_ko_v4.txt"
    )
    direction = "ko_to_en" if origin == "KR" else "en_to_ko"
    target_language = "en" if origin == "KR" else "ko"
    graph_hash = str(master.get("entity_relation_graph_hash") or "")
    request = {
        "case_id": master["case_id"],
        "master_language": master["master_language"],
        "translation_direction": direction,
        "master_neutral_text": master.get("master_neutral_text") or "",
        "fact_units": [
            {
                key: unit.get(key)
                for key in (
                    "fact_id", "master_text", "epistemic_status", "relation_ids",
                    "realized_relations", "material_relation",
                )
            }
            for unit in master.get("fact_units") or []
        ],
        "placeholders": sorted(
            set(PLACEHOLDER_RE.findall(str(master.get("master_neutral_text") or "")))
        ),
        "entity_relation_graph_hash": graph_hash,
    }
    result = client.call(
        case_id=master["case_id"],
        stage="v3_translation",
        system_prompt=load_prompt(root, prompt_name),
        user_payload=request,
        schema=TRANSLATION_SCHEMA,
        required_fields=tuple(TRANSLATION_SCHEMA["required"]),
        prompt_version=prompt_name.removesuffix(".txt"),
        schema_version=SCHEMA_VERSION,
        context_hashes={"entity_relation_graph_hash": graph_hash},
    )
    payload = dict(result.payload)
    payload["translated_fact_units"] = [
        dict(unit) for unit in payload.get("translated_fact_units") or []
    ]
    payload["translated_neutral_text"] = " ".join(
        str(unit.get("translated_text") or "").strip()
        for unit in payload["translated_fact_units"]
        if str(unit.get("translated_text") or "").strip()
    )
    checks = translation_checks(master, payload, target_language)
    record = {
        "dataset_version": DATASET_VERSION,
        "case_id": master["case_id"],
        "case_origin": origin,
        "master_language": master["master_language"],
        "translation_direction": direction,
        "master_neutral_text": master.get("master_neutral_text") or "",
        "translated_neutral_text": payload["translated_neutral_text"],
        "translated_fact_units": payload["translated_fact_units"],
        "translation_status": checks["status"],
        "translation_relation_status": checks["translation_relation_status"],
        "translation_warnings": checks["warnings"] + checks["errors"],
        "deterministic_checks": checks,
        "model_provenance": result.provenance,
    }
    return record, {"case_id": master["case_id"], "case_origin": origin, **checks}


def verify_translation(
    master: dict[str, Any],
    translated: dict[str, Any],
    client: LLMClient,
    root: Path,
) -> dict[str, Any]:
    prompt_name = (
        "verify_translation_relations_ko_en_v4.txt"
        if master["case_origin"] == "KR"
        else "verify_translation_relations_en_ko_v4.txt"
    )
    graph_hash = str(master.get("entity_relation_graph_hash") or "")
    request = {
        "case_id": master["case_id"],
        "master_fact_units": [
            {
                "fact_id": unit.get("fact_id"),
                "text": unit.get("master_text"),
                "epistemic_status": unit.get("epistemic_status"),
                "relation_ids": unit.get("relation_ids"),
                "realized_relations": unit.get("realized_relations"),
            }
            for unit in master.get("fact_units") or []
        ],
        "translated_fact_units": translated.get("translated_fact_units") or [],
        "entity_relation_graph_hash": graph_hash,
    }
    result = client.call(
        case_id=master["case_id"],
        stage="v3_translation_verifier",
        system_prompt=load_prompt(root, prompt_name),
        user_payload=request,
        schema=TRANSLATION_VERIFIER_SCHEMA,
        required_fields=tuple(TRANSLATION_VERIFIER_SCHEMA["required"]),
        prompt_version=prompt_name.removesuffix(".txt"),
        schema_version=SCHEMA_VERSION,
        context_hashes={"entity_relation_graph_hash": graph_hash},
    )
    return validate_translation_verifier_payload(
        dict(result.payload),
        translated,
        model_provenance=result.provenance,
    )


def validate_translation_verifier_payload(
    payload: dict[str, Any],
    translated: dict[str, Any],
    *,
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute translation-verifier consistency without an API request."""
    payload = dict(payload)
    hard_fields = (
        "missing_fact_ids", "added_information", "omitted_information",
        "changed_negation", "changed_temporal_relation", "changed_epistemic_status",
        "changed_subject_object", "changed_entity_role", "changed_possession",
        "changed_employment_or_ownership", "changed_medical_provider_role",
        "changed_product_chain_role", "changed_directionality",
        "changed_material_relation", "legal_term_reintroduction",
        "jurisdiction_reintroduction", "jurisdiction_term_reintroduction",
        "placeholder_errors",
    )
    reasons = [field for field in hard_fields if payload.get(field)]
    if payload.get("meaning_preserved") is not True:
        reasons.append("meaning_preserved_false")
    if translated.get("translation_status") == "fail":
        reasons.append("deterministic_translation_failure_not_acknowledged")
    model_status = str(
        payload.get("model_verifier_status")
        or payload.get("translation_status")
        or "fail"
    )
    validated = (
        "fail" if reasons or model_status == "fail"
        else "warning" if model_status == "warning"
        else "pass"
    )
    return {
        **payload,
        "model_verifier_status": model_status,
        "validated_verifier_status": validated,
        "translation_status": validated,
        "verifier_consistency_violation": model_status == "pass" and validated != "pass",
        "deterministic_verifier_reasons": reasons,
        "verification_type": "translation_and_relations",
        "model_provenance": (
            model_provenance
            if model_provenance is not None
            else payload.get("model_provenance") or {}
        ),
    }
