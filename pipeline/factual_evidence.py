from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.llm_client import LLMClient
from pipeline.stage2_schema import EVIDENCE_SCHEMA, Stage2CaseInput


def load_prompt(root: Path, name: str) -> str:
    return (root / "prompts" / name).read_text(encoding="utf-8")


def extract_evidence(case: Stage2CaseInput, segment_record: dict[str, Any], client: LLMClient, root: Path) -> dict[str, Any]:
    prompt_name = "extract_evidence_ko_v1.txt" if case.case_origin == "KR" else "extract_evidence_en_v1.txt"
    chunks = segment_record["candidate_chunks"]
    segment_map = {row["source_sentence_id"]: row["text"] for row in segment_record.get("segments", [])}
    merged_units: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    provenances: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        result = client.call(
            case_id=case.case_id, stage=f"factual_evidence_{chunk_id.lower()}", system_prompt=load_prompt(root, prompt_name),
            user_payload={"case_id": case.case_id, "source_language": case.source_language, "ordered_chunk_untrusted": chunk},
            schema=EVIDENCE_SCHEMA, required_fields=("case_id", "evidence_units"), prompt_version=prompt_name.removesuffix(".txt"),
        )
        provenances.append({**result.provenance, "chunk_id": chunk_id})
        if result.payload.get("case_id") != case.case_id:
            validation_errors.append(f"{chunk_id}:case_id_mismatch")
            continue
        chunk_ids = set(chunk.get("source_sentence_ids") or [])
        for unit in result.payload.get("evidence_units", []):
            source_ids = [str(value) for value in unit.get("source_sentence_ids") or []]
            unknown = [source_id for source_id in source_ids if source_id not in segment_map or source_id not in chunk_ids]
            if not source_ids or unknown:
                validation_errors.append(f"{chunk_id}:{unit.get('evidence_id')}:unknown_source_sentence_id:{','.join(unknown)}")
                continue
            merged_units.append({
                "evidence_id": str(unit.get("evidence_id") or ""),
                "source_sentence_ids": source_ids,
                "exact_excerpts": [segment_map[source_id] for source_id in source_ids],
                "model_short_quote": unit.get("short_quote"),
                "proposed_fact_type": list(unit.get("proposed_fact_type") or []),
                "epistemic_status": unit.get("epistemic_status"),
                "epistemic_status_confidence": unit.get("epistemic_status_confidence"),
                "source_chunk_ids": [chunk_id],
            })
    deduplicated: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for unit in merged_units:
        normalized_excerpt = " ".join(" ".join(unit["exact_excerpts"]).casefold().split())
        key = (tuple(unit["source_sentence_ids"]), normalized_excerpt, tuple(sorted(unit["proposed_fact_type"])), unit["epistemic_status"])
        if key in seen:
            seen[key]["source_chunk_ids"] = sorted(set(seen[key]["source_chunk_ids"] + unit["source_chunk_ids"]))
            continue
        seen[key] = unit
        deduplicated.append(unit)
    for index, unit in enumerate(deduplicated, 1):
        unit["evidence_id"] = f"E{index:03d}"
    coverage = dict(segment_record.get("candidate_metadata") or {})
    coverage["extraction_call_count"] = len(chunks)
    status = "fail" if validation_errors or not coverage.get("coverage_complete", False) else "pass"
    total_usage = {key: sum(int((item.get("api_usage") or {}).get(key) or 0) for item in provenances) or None for key in ("input_tokens", "output_tokens", "total_tokens")}
    aggregate_provenance = {
        "model": provenances[0].get("model") if provenances else client.model,
        "prompt_version": prompt_name.removesuffix(".txt"),
        "request_hashes": [item.get("request_hash") for item in provenances],
        "raw_response_paths": [item.get("raw_response_path") for item in provenances],
        "api_usage": total_usage,
        "cache_hits": sum(bool(item.get("cache_hit")) for item in provenances),
        "new_api_calls": sum(not item.get("cache_hit") for item in provenances),
        "mock": all(bool(item.get("mock")) for item in provenances) if provenances else bool(client.mock_response_dir),
    }
    return {"case_id": case.case_id, "case_origin": case.case_origin, "evidence_units": deduplicated, "extraction_status": status, "validation_errors": validation_errors, "source_coverage": coverage, "chunk_model_provenance": provenances, "model_provenance": aggregate_provenance}


def recheck_evidence_record(record: dict[str, Any], segment_record: dict[str, Any]) -> dict[str, Any]:
    segment_map = {row["source_sentence_id"]: row["text"] for row in segment_record.get("segments", [])}
    errors: list[str] = []; units: list[dict[str, Any]] = []; seen: set[tuple[Any, ...]] = set()
    for unit in record.get("evidence_units", []):
        source_ids = [str(value) for value in unit.get("source_sentence_ids") or []]
        unknown = [source_id for source_id in source_ids if source_id not in segment_map]
        if not source_ids or unknown:
            errors.append(f"{unit.get('evidence_id')}:unknown_source_sentence_id:{','.join(unknown)}"); continue
        updated = dict(unit); updated["exact_excerpts"] = [segment_map[source_id] for source_id in source_ids]
        normalized = " ".join(" ".join(updated["exact_excerpts"]).casefold().split())
        key = (tuple(source_ids), normalized, tuple(sorted(updated.get("proposed_fact_type") or [])), updated.get("epistemic_status"))
        if key in seen: continue
        seen.add(key); units.append(updated)
    for index, unit in enumerate(units, 1): unit["evidence_id"] = f"E{index:03d}"
    coverage = dict(segment_record.get("candidate_metadata") or {})
    updated_record = dict(record); updated_record.update({"evidence_units": units, "validation_errors": errors, "source_coverage": coverage, "extraction_status": "fail" if errors or not coverage.get("coverage_complete", False) else "pass"})
    return updated_record
