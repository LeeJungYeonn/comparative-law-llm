from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.canonical_neutralization import neutralize, recheck_neutral_record
from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl, atomic_write_text
from pipeline.factual_evidence import extract_evidence, recheck_evidence_record
from pipeline.llm_client import LLMClient
from pipeline.source_segmentation import segment_source, segmentation_record, select_candidate_chunks
from pipeline.stage2_input import validate_inputs
from pipeline.stage2_runtime import (append_errors, append_quarantine, append_run_history, by_case,
    ensure_writable_outputs, merge_by_case, quarantine_record, record_counts, resolve_case_ids,
    run_parallel, select_by_origin_limit, write_usage)


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 full-source evidence extraction and source-language neutralization.")
    p.add_argument("--kr-input", type=Path, default=Path("outputs/raw/kr_v4/kr_cases_selected_35.jsonl")); p.add_argument("--ca-input", type=Path, default=Path("outputs/raw/ca_v4/ca_cases_selected_35.jsonl")); p.add_argument("--output-dir", type=Path, default=Path("outputs/neutral/stage2-neutral-35x35-v1"))
    p.add_argument("--model", default="gpt-5.6-luna"); p.add_argument("--base-url", default="https://gw.letsur.ai/v1")
    p.add_argument("--limit", type=int); p.add_argument("--case-id", action="append"); p.add_argument("--case-id-file", type=Path); p.add_argument("--batch-name", default="unnamed"); p.add_argument("--max-cases-per-origin", type=int)
    p.add_argument("--concurrency", type=int, default=2); p.add_argument("--max-retries", type=int, default=5); p.add_argument("--max-input-tokens", type=int, default=12000); p.add_argument("--chunk-overlap-sentences", type=int, default=2); p.add_argument("--candidate-max-chunks", type=int, default=8)
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--mock-response-dir", type=Path); p.add_argument("--temperature", type=float); p.add_argument("--seed", type=int)
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--retry-warnings", action="store_true"); p.add_argument("--recheck-deterministic", action="store_true"); p.add_argument("--recheck-existing", action="store_true", help=argparse.SUPPRESS); p.add_argument("--regenerate", action="store_true")
    p.add_argument("--stop-on-hard-failure", action="store_true"); p.add_argument("--max-hard-failure-rate", type=float, default=0.10); p.add_argument("--max-api-failure-rate", type=float, default=0.05)
    return p


def _retry_selected(record: dict[str, Any] | None, args: argparse.Namespace, status_field: str) -> bool:
    if record is None: return True
    if args.regenerate: return True
    status = str(record.get(status_field) or "fail")
    return (args.retry_failed and status == "fail") or (args.retry_warnings and status == "warning")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); args.recheck_deterministic = args.recheck_deterministic or args.recheck_existing
    output = args.output_dir
    targets = [output / name for name in ("input_validation_report.json", "input_manifest.json", "source_segments_kr.jsonl", "source_segments_ca.jsonl", "factual_evidence_kr.jsonl", "factual_evidence_ca.jsonl", "source_neutral_kr.jsonl", "source_neutral_ca.jsonl")]
    ensure_writable_outputs(targets, resume=args.resume, overwrite=args.overwrite)
    cases, report, input_manifest = validate_inputs(args.kr_input, args.ca_input, output)
    selected_ids = resolve_case_ids(args.case_id, args.case_id_file); selected = select_by_origin_limit(cases, selected_ids, args.max_cases_per_origin)
    if args.limit is not None: selected = selected[:max(0, args.limit)]
    if not selected: raise ValueError("No cases selected")
    (output / "raw_responses").mkdir(parents=True, exist_ok=True)
    for prompt in (ROOT / "prompts").glob("*.txt"):
        snapshot = output / "prompts" / prompt.name; content = prompt.read_text(encoding="utf-8")
        if snapshot.exists() and snapshot.read_text(encoding="utf-8") != content and not args.overwrite: raise FileExistsError(f"Prompt snapshot differs: {snapshot}")
        if not snapshot.exists() or args.overwrite: atomic_write_text(snapshot, content)
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, {**input_manifest, "phases": {}, "run_history": []})

    segment_records: list[dict[str, Any]] = []
    for case in selected:
        segments = segment_source(case.source_text); chunks, metadata = select_candidate_chunks(case, segments, max_input_tokens=args.max_input_tokens, overlap_sentences=args.chunk_overlap_sentences, max_chunks=args.candidate_max_chunks)
        segment_records.append(segmentation_record(case, segments, chunks, metadata))
    segment_map = {row["case_id"]: row for row in segment_records}
    for origin, suffix in (("KR", "kr"), ("CA", "ca")):
        path = output / f"source_segments_{suffix}.jsonl"; old = [] if args.overwrite else list(by_case(path).values())
        atomic_write_jsonl(path, merge_by_case(old, [row for row in segment_records if row["case_origin"] == origin]))

    mode = "dry_run" if args.dry_run else "mock" if args.mock_response_dir else "api"
    if args.dry_run:
        coverage_failures = [row for row in segment_records if not row["candidate_metadata"]["coverage_complete"]]
        phase = {"execution_status": "completed", "record_counts": {"pass": len(segment_records) - len(coverage_failures), "warning": 0, "fail": len(coverage_failures), "missing": 70 - len(segment_records)}}
        append_run_history(output, {"batch_name": args.batch_name, "selected_case_ids": [case.case_id for case in selected], "mode": mode, "new_api_calls": 0, "cache_hits": 0}, {"phase_0_input_validation": {"execution_status": "completed", "record_counts": {"pass": 70, "warning": 0, "fail": 0, "missing": 0}}, "phase_1_segmentation": phase})
        append_errors(output / "api_errors.jsonl", []); write_usage(output / "api_usage.csv", [])
        print(f"dry-run pass: validated 70 inputs; full-source segmented {len(selected)} selected cases; no API calls")
        return 2 if coverage_failures else 0

    client = LLMClient(output_dir=output, model=args.model, base_url=args.base_url, max_retries=args.max_retries, temperature=args.temperature, seed=args.seed, mock_response_dir=args.mock_response_dir, bypass_cache=bool(args.retry_failed or args.retry_warnings or args.regenerate))
    evidence_paths = {"KR": output / "factual_evidence_kr.jsonl", "CA": output / "factual_evidence_ca.jsonl"}; neutral_paths = {"KR": output / "source_neutral_kr.jsonl", "CA": output / "source_neutral_ca.jsonl"}
    existing_evidence = {origin: by_case(path) if not args.overwrite else {} for origin, path in evidence_paths.items()}; existing_neutral = {origin: by_case(path) if not args.overwrite else {} for origin, path in neutral_paths.items()}
    case_map = {case.case_id: case for case in selected}

    errors: list[dict[str, Any]] = []; evidence_new: list[dict[str, Any]] = []; neutral_new: list[dict[str, Any]] = []
    if args.recheck_deterministic:
        for case in selected:
            existing = existing_evidence[case.case_origin].get(case.case_id)
            if existing: evidence_new.append(recheck_evidence_record(existing, segment_map[case.case_id]))
    else:
        evidence_todo = [case for case in selected if _retry_selected(existing_evidence[case.case_origin].get(case.case_id), args, "extraction_status")]
        def evidence_worker(case): return extract_evidence(case, segment_map[case.case_id], client, ROOT)
        evidence_new, errors = run_parallel(evidence_todo, evidence_worker, args.concurrency)
    for origin in ("KR", "CA"):
        merged = merge_by_case(list(existing_evidence[origin].values()), [row for row in evidence_new if row["case_origin"] == origin]); atomic_write_jsonl(evidence_paths[origin], merged); existing_evidence[origin] = {row["case_id"]: row for row in merged}

    if args.recheck_deterministic:
        for case in selected:
            existing = existing_neutral[case.case_origin].get(case.case_id); evidence = existing_evidence[case.case_origin].get(case.case_id)
            if existing and evidence: neutral_new.append(recheck_neutral_record(existing, evidence, case.source_text))
    else:
        neutral_todo = [case for case in selected if existing_evidence[case.case_origin].get(case.case_id, {}).get("extraction_status") == "pass" and _retry_selected(existing_neutral[case.case_origin].get(case.case_id), args, "neutralization_status")]
        def neutral_worker(case): return neutralize(case, existing_evidence[case.case_origin][case.case_id], client, ROOT)
        neutral_new, neutral_errors = run_parallel(neutral_todo, neutral_worker, args.concurrency); errors.extend(neutral_errors)
    for origin in ("KR", "CA"):
        merged = merge_by_case(list(existing_neutral[origin].values()), [row for row in neutral_new if row["case_origin"] == origin]); atomic_write_jsonl(neutral_paths[origin], merged); existing_neutral[origin] = {row["case_id"]: row for row in merged}

    append_errors(output / "api_errors.jsonl", errors); write_usage(output / "api_usage.csv", evidence_new + neutral_new)
    current_evidence = [row for origin in ("KR", "CA") for row in existing_evidence[origin].values()]; current_neutral = [row for origin in ("KR", "CA") for row in existing_neutral[origin].values()]
    selected_evidence = [existing_evidence[case.case_origin].get(case.case_id) for case in selected]; selected_neutral = [existing_neutral[case.case_origin].get(case.case_id) for case in selected]
    hard_cases: list[tuple[Any, list[str]]] = []
    for case, evidence, neutral_record in zip(selected, selected_evidence, selected_neutral):
        reasons = []
        if evidence is None: reasons.append("missing_evidence_record")
        elif evidence.get("extraction_status") == "fail": reasons.extend(evidence.get("validation_errors") or ["evidence_extraction_fail"])
        elif not (evidence.get("source_coverage") or {}).get("coverage_complete", False): reasons.append("source_segment_coverage_incomplete")
        if neutral_record is None: reasons.append("missing_source_neutral_record")
        elif neutral_record.get("neutralization_status") == "fail": reasons.extend((neutral_record.get("deterministic_checks") or {}).get("errors") or ["source_neutral_fail"])
        if reasons: hard_cases.append((case, sorted(set(reasons))))
    quarantine = [quarantine_record(existing_neutral[case.case_origin].get(case.case_id) or existing_evidence[case.case_origin].get(case.case_id) or {"case_id": case.case_id, "case_origin": case.case_origin}, "source_generation", reasons, regeneration=bool(errors), deterministic_recheck=not bool(errors)) for case, reasons in hard_cases]
    append_quarantine(output / "quarantine.jsonl", quarantine)
    selected_count = len(selected); hard_rate = len(hard_cases) / selected_count; api_failure_rate = len(errors) / selected_count
    batch = {"batch_name": args.batch_name, "selected_case_ids": [case.case_id for case in selected], "new_api_calls": client.new_api_calls, "cache_hits": client.cache_hits, "completed_cases": sum(row is not None for row in selected_neutral), "pass_count": sum(row is not None and row.get("neutralization_status") == "pass" for row in selected_neutral), "warning_count": sum(row is not None and row.get("neutralization_status") == "warning" for row in selected_neutral), "fail_count": len(hard_cases), "quarantined_count": len(quarantine), "missing_count": sum(row is None for row in selected_neutral), "hard_failure_rate": hard_rate, "api_failure_rate": api_failure_rate}
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.batch_name); atomic_write_json(output / f"batch_{safe_name}_generation.json", batch)
    phases = {
        "phase_2_evidence": {"execution_status": "completed", "record_counts": record_counts(current_evidence, "extraction_status", 70)},
        "phase_3_neutralization": {"execution_status": "completed", "record_counts": record_counts(current_neutral, "neutralization_status", 70)},
        "phase_4_deterministic_source_checks": {"execution_status": "completed", "record_counts": record_counts(current_neutral, "neutralization_status", 70)},
    }
    append_run_history(output, {**batch, "mode": mode, "recheck_deterministic": args.recheck_deterministic, "retry_failed": args.retry_failed, "retry_warnings": args.retry_warnings, "regenerate": args.regenerate}, phases)
    blocked = bool(hard_cases) or hard_rate > args.max_hard_failure_rate or api_failure_rate > args.max_api_failure_rate
    print(f"generation complete: selected={selected_count} evidence_new={len(evidence_new)} neutral_new={len(neutral_new)} api_calls={client.new_api_calls} cache_hits={client.cache_hits} hard_fail={len(hard_cases)} blocked={blocked}")
    return 2 if blocked else 0


if __name__ == "__main__": raise SystemExit(main())
