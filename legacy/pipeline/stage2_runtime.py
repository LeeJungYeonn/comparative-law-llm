from __future__ import annotations

import csv
import json
import traceback
from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from pipeline.checkpoint import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_jsonl_recover


T = TypeVar("T")


def read_records(path: Path) -> list[dict[str, Any]]:
    rows, _ = read_jsonl_recover(path)
    return rows


def by_case(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_id")): row for row in read_records(path) if row.get("case_id")}


def ensure_writable_outputs(paths: Iterable[Path], *, resume: bool, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not (resume or overwrite):
        raise FileExistsError(f"Stage 2 outputs already exist; use --resume or --overwrite: {existing[0]}")


def run_parallel(items: list[T], worker: Callable[[T], dict[str, Any]], concurrency: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            case_id = getattr(item, "case_id", None) or (item.get("case_id") if isinstance(item, dict) else "")
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"case_id": case_id, "stage": getattr(worker, "__name__", "unknown"), "error_type": type(exc).__name__, "message": str(exc), "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip()})
    return results, errors


def merge_by_case(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {str(row.get("case_id")): row for row in existing if row.get("case_id")}
    for row in new:
        merged[str(row["case_id"])] = row
    return list(merged.values())


def append_errors(path: Path, errors: list[dict[str, Any]]) -> None:
    if not errors:
        if not path.exists(): atomic_write_jsonl(path, [])
        return
    existing = read_records(path)
    combined = existing + errors
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in combined:
        key = json.dumps({"case_id": row.get("case_id"), "stage": row.get("stage"), "error_type": row.get("error_type"), "message": row.get("message")}, ensure_ascii=False, sort_keys=True)
        deduplicated[key] = row
    atomic_write_jsonl(path, deduplicated.values())


def write_usage(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        provenances = record.get("chunk_model_provenance") or [record.get("model_provenance") or {}]
        for provenance in provenances:
            usage = provenance.get("api_usage") or {}
            rows.append({"case_id": record.get("case_id", ""), "stage": provenance.get("prompt_version", ""), "model": provenance.get("model", ""), "request_hash": provenance.get("request_hash", ""), "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "total_tokens": usage.get("total_tokens"), "cache_hit": provenance.get("cache_hit", False), "mock": provenance.get("mock", False)})
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle: existing = list(csv.DictReader(handle))
    keyed = {(str(row.get("case_id")), str(row.get("request_hash"))): row for row in existing}
    for row in rows: keyed[(str(row.get("case_id")), str(row.get("request_hash")))] = row
    fields = ["case_id", "stage", "model", "request_hash", "input_tokens", "output_tokens", "total_tokens", "cache_hit", "mock"]
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(keyed.values())
    atomic_write_text(path, stream.getvalue())


def json_hash(value: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def resolve_case_ids(case_ids: list[str] | None, case_id_file: Path | None) -> list[str]:
    values = list(case_ids or [])
    if case_id_file:
        values.extend(line.strip() for line in case_id_file.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#"))
    return list(dict.fromkeys(values))


def select_by_origin_limit(cases: list[T], selected_ids: list[str], max_per_origin: int | None) -> list[T]:
    selected = [case for case in cases if not selected_ids or getattr(case, "case_id", None) in set(selected_ids)]
    if max_per_origin is None: return selected
    counts: Counter[str] = Counter(); limited: list[T] = []
    for case in selected:
        origin = str(getattr(case, "case_origin", ""))
        if counts[origin] < max_per_origin: limited.append(case); counts[origin] += 1
    return limited


def record_counts(records: Iterable[dict[str, Any]], status_field: str, manifest_case_count: int) -> dict[str, int]:
    counts = Counter(str(record.get(status_field) or "missing") for record in records)
    recognized = {"pass": counts["pass"], "warning": counts["warning"], "fail": counts["fail"]}
    recognized["missing"] = max(0, manifest_case_count - sum(recognized.values()))
    return recognized


def append_run_history(output_dir: Path, entry: dict[str, Any], phase_updates: dict[str, Any] | None = None) -> dict[str, Any]:
    path = output_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"phases": {}, "run_history": []}
    manifest.setdefault("run_history", []).append({"timestamp": datetime.now(timezone.utc).isoformat(), **entry})
    if phase_updates: manifest.setdefault("phases", {}).update(phase_updates)
    atomic_write_json(path, manifest)
    return manifest


def append_quarantine(path: Path, records: Iterable[dict[str, Any]]) -> None:
    existing = read_records(path)
    keyed = {(str(row.get("case_id")), str(row.get("failed_stage"))): row for row in existing}
    for row in records: keyed[(str(row.get("case_id")), str(row.get("failed_stage")))] = row
    atomic_write_jsonl(path, keyed.values())


def quarantine_record(record: dict[str, Any], stage: str, reasons: list[str], *, regeneration: bool = False, deterministic_recheck: bool = False) -> dict[str, Any]:
    provenance = record.get("model_provenance") or {}
    raw_path = provenance.get("raw_response_path") or (provenance.get("raw_response_paths") or [""])[0]
    return {"case_id": record.get("case_id"), "case_origin": record.get("case_origin"), "failed_stage": stage, "failure_reasons": reasons, "raw_response_path": raw_path, "requires_api_regeneration": regeneration, "requires_deterministic_recheck": deterministic_recheck, "requires_human_qc": True}
