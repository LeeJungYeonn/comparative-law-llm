from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from pipeline.checkpoint import atomic_write_json, atomic_write_text
from pipeline.response_parser import parse_json_response
from pipeline.stage2_schema import SCHEMA_VERSION


class LLMRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    payload: dict[str, Any]
    provenance: dict[str, Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


class LLMClient:
    """Small OpenAI-compatible client with caching and structured-output fallback."""

    def __init__(self, *, output_dir: Path, model: str, base_url: str, max_retries: int = 5,
                 temperature: float | None = None, seed: int | None = None,
                 mock_response_dir: Path | None = None, api_key_env: str = "LETSUR_API_KEY", bypass_cache: bool = False) -> None:
        self.output_dir = output_dir
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.temperature = temperature
        self.seed = seed
        self.mock_response_dir = mock_response_dir
        self.api_key_env = api_key_env
        self.bypass_cache = bypass_cache
        self.structured_mode: str | None = None
        self.new_api_calls = 0
        self.cache_hits = 0
        self.request_hashes_seen: set[str] = set()

    def _request_hash(self, *, case_id: str, stage: str, prompt_hash: str, user_payload: dict[str, Any], parameters: dict[str, Any], schema_version: str = SCHEMA_VERSION, context_hashes: dict[str, str] | None = None) -> str:
        material = {"case_id": case_id, "stage": stage, "input_content_hash": hashlib.sha256(_stable_json(user_payload).encode()).hexdigest(), "prompt_file_hash": prompt_hash, "model": self.model, "base_url_identifier": self.base_url, "effective_generation_parameters": parameters, "schema_version": schema_version, **(context_hashes or {})}
        return hashlib.sha256(_stable_json(material).encode()).hexdigest()

    def _mock(self, stage: str, case_id: str) -> dict[str, Any]:
        assert self.mock_response_dir is not None
        generic_stage = "factual_evidence" if stage.startswith("factual_evidence_chunk") else stage
        candidates = [self.mock_response_dir / stage / f"{case_id}.json", self.mock_response_dir / f"{stage}__{case_id}.json", self.mock_response_dir / generic_stage / f"{case_id}.json", self.mock_response_dir / f"{generic_stage}__{case_id}.json"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise LLMRequestError(f"Mock response not found for {stage}/{case_id}")
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict) and "choices" in value:
            content = value["choices"][0]["message"]["content"]
            return parse_json_response(content)
        if not isinstance(value, dict):
            raise LLMRequestError(f"Mock response must be an object: {path}")
        return value

    def _http(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMRequestError(f"Missing environment variable {self.api_key_env}")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=_stable_json(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw), dict(response.headers)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            error = LLMRequestError(f"HTTP {exc.code}: {message[:1000]}")
            setattr(error, "status", exc.code)
            setattr(error, "headers", exc.headers)
            raise error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMRequestError(str(exc)) from exc

    @staticmethod
    def _content(envelope: dict[str, Any]) -> str:
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("OpenAI-compatible response has no message content") from exc
        if not isinstance(content, str):
            raise LLMRequestError("Response content is not text")
        return content

    def call(self, *, case_id: str, stage: str, system_prompt: str, user_payload: dict[str, Any], schema: dict[str, Any], required_fields: tuple[str, ...], prompt_version: str, schema_version: str = SCHEMA_VERSION, context_hashes: dict[str, str] | None = None) -> LLMResult:
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        requested = {"temperature": self.temperature, "seed": self.seed, "structured_output": "json_schema"}
        mode_order = [self.structured_mode] if self.structured_mode else ["json_schema", "json_object", "json_only"]
        last_error: Exception | None = None
        for mode in mode_order:
            if mode is None:
                continue
            effective: dict[str, Any] = {"structured_output": mode}
            if self.temperature is not None:
                effective["temperature"] = self.temperature
            if self.seed is not None:
                effective["seed"] = self.seed
            request_hash = self._request_hash(case_id=case_id, stage=stage, prompt_hash=prompt_hash, user_payload=user_payload, parameters=effective, schema_version=schema_version, context_hashes=context_hashes)
            cache_path = self.output_dir / "request_cache" / stage / f"{request_hash}.json"
            raw_path = self.output_dir / "raw_responses" / stage / f"{case_id}__{request_hash[:12]}.json"
            if cache_path.is_file() and not self.bypass_cache:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_mock = bool((cached.get("provenance") or {}).get("mock"))
                if cached_mock != bool(self.mock_response_dir):
                    raise LLMRequestError("mock_real_provenance_contamination")
                self.cache_hits += 1
                return LLMResult(cached["payload"], {**cached["provenance"], "cache_hit": True})
            if request_hash in self.request_hashes_seen and not self.bypass_cache:
                raise LLMRequestError(f"duplicate_api_request:{request_hash}")
            self.request_hashes_seen.add(request_hash)
            if self.mock_response_dir:
                payload = self._mock(stage, case_id)
                envelope: dict[str, Any] = {"mock": True, "payload": payload}
                usage: dict[str, Any] = {}
            else:
                body: dict[str, Any] = {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": _stable_json(user_payload)}]}
                if self.temperature is not None: body["temperature"] = self.temperature
                if self.seed is not None: body["seed"] = self.seed
                if mode == "json_schema": body["response_format"] = {"type": "json_schema", "json_schema": {"name": stage, "strict": True, "schema": schema}}
                elif mode == "json_object": body["response_format"] = {"type": "json_object"}
                else: body["messages"][0]["content"] += "\nReturn one complete JSON object only."
                for attempt in range(self.max_retries + 1):
                    try:
                        envelope, _ = self._http(body)
                        break
                    except LLMRequestError as exc:
                        last_error = exc
                        status = getattr(exc, "status", None)
                        lowered = str(exc).casefold()
                        if status in {400, 422} and "seed" in body and "seed" in lowered:
                            body.pop("seed", None); effective.pop("seed", None)
                            continue
                        if status in {400, 422} and "temperature" in body and "temperature" in lowered:
                            body.pop("temperature", None); effective.pop("temperature", None)
                            continue
                        if mode in {"json_schema", "json_object"} and status in {400, 404, 415, 422}:
                            break
                        if attempt >= self.max_retries or (status is not None and status not in {408, 409, 429} and status < 500):
                            raise
                        delay = _retry_after(getattr(exc, "headers", None))
                        if delay is None: delay = min(30.0, (2 ** attempt) + random.random())
                        time.sleep(delay)
                else:
                    raise LLMRequestError("retry loop ended unexpectedly")
                if last_error is not None and 'envelope' not in locals():
                    continue
                usage = envelope.get("usage") or {}
                try:
                    payload = parse_json_response(self._content(envelope), required_fields)
                except Exception:
                    atomic_write_json(raw_path, envelope)
                    raise
            missing = [field for field in required_fields if field not in payload]
            if missing:
                raise LLMRequestError(f"Missing schema-critical fields: {missing}")
            self.structured_mode = mode
            if raw_path.exists():
                raw_path = raw_path.with_name(f"{raw_path.stem}__{int(time.time() * 1000)}{raw_path.suffix}")
            atomic_write_json(raw_path, envelope)
            provenance = {
                "model": self.model, "prompt_version": prompt_version, "request_hash": request_hash,
                "raw_response_path": str(raw_path), "api_usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens")},
                "requested_generation_parameters": requested, "effective_generation_parameters": effective,
                "structured_output_mode": mode, "cache_hit": False, "mock": bool(self.mock_response_dir),
            }
            self.new_api_calls += 0 if self.mock_response_dir else 1
            if cache_path.exists() and self.bypass_cache:
                history_path = cache_path.with_name(f"{cache_path.stem}__superseded_{int(time.time() * 1000)}{cache_path.suffix}")
                atomic_write_text(history_path, cache_path.read_text(encoding="utf-8"))
            atomic_write_json(cache_path, {"payload": payload, "provenance": provenance})
            return LLMResult(payload, provenance)
        raise LLMRequestError(f"No supported structured-output mode: {last_error}")
