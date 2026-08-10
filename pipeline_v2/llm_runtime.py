from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, canonical_json, guard_outputs, sha256_text, write_json


def configured_model(cli_model: str | None) -> str:
    model = cli_model or os.getenv("FACT_EXTRACTION_MODEL")
    if not model:
        raise RuntimeError("No model configured. Pass --model or set FACT_EXTRACTION_MODEL.")
    return model


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. This pipeline never reads API credentials from files.")


def request_key(case_id: str, prompt_version: str, input_hash: str, model: str, stage: str) -> str:
    return sha256_text(canonical_json({"case_id": case_id, "prompt_version": prompt_version, "input_hash": input_hash, "model": model, "stage": stage}))


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "to_json"):
        return json.loads(response.to_json())
    return {"repr": repr(response)}


def call_structured(
    *, case_id: str, stage: str, prompt_version: str, model: str, system_prompt: str,
    user_payload: dict[str, Any], schema_name: str, schema: dict[str, Any], raw_root: Path,
    status_path: Path, max_retries: int, resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_hash = sha256_text(canonical_json(user_payload))
    key = request_key(case_id, prompt_version, input_hash, model, stage)
    request_dir = raw_root / stage / key
    result_path = request_dir / "result.json"
    if resume and result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        return cached["parsed"], cached["provenance"]
    require_api_key()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The official OpenAI Python SDK is required; install requirements-v2.txt") from exc
    request_dir.mkdir(parents=True, exist_ok=True)
    prior_attempts = []
    for path in request_dir.glob("attempt_*_*.json"):
        match = re.match(r"attempt_(\d+)_", path.name)
        if match:
            prior_attempts.append(int(match.group(1)))
    first_attempt = max(prior_attempts, default=0) + 1
    last_error: Exception | None = None
    for retry_index in range(max_retries):
        attempt = first_attempt + retry_index
        started = datetime.now(timezone.utc).isoformat()
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": canonical_json(user_payload)}],
                text={"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
                store=False,
            )
            raw = _response_dict(response)
            raw_path = request_dir / f"attempt_{attempt:03d}_raw.json"
            write_json(raw_path, raw)  # persist before parsing or deterministic validation
            output_text = getattr(response, "output_text", "")
            parsed = json.loads(output_text)
            provenance = {
                "model": model, "model_snapshot_or_returned_model_id": getattr(response, "model", model),
                "prompt_version": prompt_version, "request_id": getattr(response, "_request_id", None) or getattr(response, "id", None),
                "response_id": getattr(response, "id", None), "timestamp": started,
                "usage": raw.get("usage"), "input_hash": input_hash, "output_hash": sha256_text(canonical_json(parsed)),
                "request_key": key, "attempt": attempt, "raw_response_path": str(raw_path), "status": "success",
            }
            result = {"parsed": parsed, "provenance": provenance}
            write_json(result_path, result)
            append_jsonl(status_path, {"case_id": case_id, "stage": stage, **provenance})
            return parsed, provenance
        except Exception as exc:  # SDK exposes several transient subclasses across versions
            last_error = exc
            error_record = {"case_id": case_id, "stage": stage, "request_key": key, "attempt": attempt, "timestamp": started, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
            write_json(request_dir / f"attempt_{attempt:03d}_error.json", error_record)
            append_jsonl(status_path, error_record)
            if retry_index + 1 < max_retries:
                time.sleep(min(2 ** retry_index + random.Random(f"{key}:{attempt}").random(), 20))
    raise RuntimeError(f"{stage} failed for {case_id} after {max_retries} attempts: {last_error}")


def load_mock(mock_dir: Path, stage: str, case_id: str) -> dict[str, Any] | None:
    candidates = (mock_dir / stage / f"{case_id}.json", mock_dir / f"{case_id}.json")
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None
