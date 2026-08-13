from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, canonical_json, sha256_text, write_json

DEFAULT_LETSUR_BASE_URL = "https://gw.letsur.ai/v1"
DEFAULT_LETSUR_MODEL = "gpt-5.6-luna"
DEFAULT_API_KEY_ENV = "LETSUR_API_KEY"


class LLMRequestError(RuntimeError):
    pass


def load_environment(dotenv_path: Path | None = None) -> None:
    """Load .env without overriding explicitly exported values."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required; install requirements-v2.txt") from exc
    resolved = str(dotenv_path) if dotenv_path else find_dotenv(usecwd=True)
    if resolved:
        load_dotenv(resolved, override=False)


def configured_model(cli_model: str | None) -> str:
    return cli_model or os.getenv("FACT_EXTRACTION_MODEL") or os.getenv("LETSUR_MODEL") or DEFAULT_LETSUR_MODEL


def require_api_key(api_key_env: str = DEFAULT_API_KEY_ENV) -> str:
    key = os.getenv(api_key_env)
    if not key:
        raise RuntimeError(f"{api_key_env} is not set after loading .env")
    return key


def request_key(case_id: str, prompt_version: str, input_hash: str, model: str, stage: str,
                base_url: str = DEFAULT_LETSUR_BASE_URL) -> str:
    return sha256_text(canonical_json({
        "case_id": case_id, "prompt_version": prompt_version, "input_hash": input_hash,
        "model": model, "stage": stage, "base_url_identifier": base_url.rstrip("/"),
    }))


def _retry_after(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _safe_error(exc: BaseException, secret: str) -> str:
    message = str(exc)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", message, flags=re.I)[:2000]


def _http_chat(*, body: dict[str, Any], base_url: str, api_key: str) -> tuple[dict[str, Any], Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=canonical_json(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8", errors="replace")), response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error = LLMRequestError(f"HTTP {exc.code}: {detail[:1500]}")
        setattr(error, "status", exc.code)
        setattr(error, "headers", exc.headers)
        raise error from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMRequestError(str(exc)) from exc


def _content(envelope: dict[str, Any]) -> str:
    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMRequestError("Letsur response has no choices[0].message.content") from exc
    if not isinstance(content, str):
        raise LLMRequestError("Letsur response content is not text")
    return content


def _parse_json_content(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise LLMRequestError("Structured response must be a JSON object")
    return parsed


def call_structured(
    *, case_id: str, stage: str, prompt_version: str, model: str, system_prompt: str,
    user_payload: dict[str, Any], schema_name: str, schema: dict[str, Any], raw_root: Path,
    status_path: Path, max_retries: int, resume: bool,
    base_url: str = DEFAULT_LETSUR_BASE_URL, api_key_env: str = DEFAULT_API_KEY_ENV,
    dotenv_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    load_environment(dotenv_path)
    input_hash = sha256_text(canonical_json(user_payload))
    key = request_key(case_id, prompt_version, input_hash, model, stage, base_url)
    request_dir = raw_root / stage / key
    result_path = request_dir / "result.json"
    if resume and result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        return cached["parsed"], cached["provenance"]
    api_key = require_api_key(api_key_env)
    request_dir.mkdir(parents=True, exist_ok=True)
    prior_attempts = []
    for path in request_dir.glob("attempt_*_*.json"):
        if match := re.match(r"attempt_(\d+)_", path.name):
            prior_attempts.append(int(match.group(1)))
    attempt = max(prior_attempts, default=0)
    last_error: BaseException | None = None
    for mode in ("json_schema", "json_object", "json_only"):
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": canonical_json(user_payload)},
            ],
        }
        if mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        else:
            body["messages"][0]["content"] += "\nReturn one complete JSON object only."
        for retry_index in range(max_retries):
            attempt += 1
            started = datetime.now(timezone.utc).isoformat()
            try:
                envelope, _ = _http_chat(body=body, base_url=base_url, api_key=api_key)
                raw_path = request_dir / f"attempt_{attempt:03d}_raw.json"
                write_json(raw_path, envelope)  # save before parsing
                parsed = _parse_json_content(_content(envelope))
                missing = [field for field in schema.get("required", []) if field not in parsed]
                if missing:
                    raise LLMRequestError(f"Missing schema-critical fields: {missing}")
                provenance = {
                    "model": model, "model_snapshot_or_returned_model_id": envelope.get("model") or model,
                    "prompt_version": prompt_version, "request_id": envelope.get("id"), "response_id": envelope.get("id"),
                    "timestamp": started, "usage": envelope.get("usage") or {}, "input_hash": input_hash,
                    "output_hash": sha256_text(canonical_json(parsed)), "request_key": key, "attempt": attempt,
                    "raw_response_path": str(raw_path), "status": "success", "base_url_identifier": base_url.rstrip("/"),
                    "api_key_env": api_key_env, "structured_output_mode": mode,
                }
                write_json(result_path, {"parsed": parsed, "provenance": provenance})
                append_jsonl(status_path, {"case_id": case_id, "stage": stage, **provenance})
                return parsed, provenance
            except BaseException as exc:
                last_error = exc
                safe = _safe_error(exc, api_key)
                error_record = {
                    "case_id": case_id, "stage": stage, "request_key": key, "attempt": attempt,
                    "timestamp": started, "status": "error", "error_type": type(exc).__name__, "error": safe,
                    "structured_output_mode": mode, "base_url_identifier": base_url.rstrip("/"),
                }
                write_json(request_dir / f"attempt_{attempt:03d}_error.json", error_record)
                append_jsonl(status_path, error_record)
                status = getattr(exc, "status", None)
                if mode in {"json_schema", "json_object"} and status in {400, 404, 415, 422}:
                    break
                if retry_index + 1 >= max_retries or (status is not None and status not in {408, 409, 429} and status < 500):
                    if mode == "json_only":
                        raise RuntimeError(f"{stage} failed for {case_id}: {safe}") from exc
                    break
                delay = _retry_after(getattr(exc, "headers", None))
                if delay is None:
                    delay = min(2 ** retry_index + random.Random(f"{key}:{attempt}").random(), 20)
                time.sleep(delay)
    raise RuntimeError(f"{stage} failed for {case_id}: {_safe_error(last_error or RuntimeError('unknown error'), api_key)}")


def load_mock(mock_dir: Path, stage: str, case_id: str) -> dict[str, Any] | None:
    for path in (mock_dir / stage / f"{case_id}.json", mock_dir / f"{case_id}.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None
