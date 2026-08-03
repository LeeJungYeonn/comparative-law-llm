from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

from exp1 import EXPERIMENT_ID, GENERATION_PROMPT_VERSION
from exp1.design import EXP2_EXPERIMENT_ID, jurisdiction_metadata
from exp1.design import JURISDICTION_INSTRUCTION_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs/neutral/stage2-paired-qc-v1/accepted_pairs.jsonl"
PROMPT_DIR = REPO_ROOT / "prompts/exp1"
SCHEMA_PATH = REPO_ROOT / "schemas/exp1_evaluation_v1.schema.json"
ONTOLOGY_PATH = REPO_ROOT / "ontology/exp1_legal_concepts_v1.json"
FORBIDDEN_REQUEST_FIELDS = {
    "case_id", "case_origin", "case_subtype", "source_language", "master_language",
    "translation_direction", "generation_dataset_version", "qc_dataset_version",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_file(path: Path, *, override: bool = False) -> bool:
    """Load KEY=VALUE pairs without ever logging values."""
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not (key[0].isalpha() or key[0] == "_") or not all(
            character.isalnum() or character == "_" for character in key
        ):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    return True


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, dir=path.parent) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, dir=path.parent) as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_prompts() -> dict[str, str]:
    names = [
        "generation_ko_v1_system.txt", "generation_ko_v1_user.txt",
        "generation_en_v1_system.txt", "generation_en_v1_user.txt",
        "evaluator_v1_system.txt", "evaluator_v1_user.txt",
    ]
    return {name: (PROMPT_DIR / name).read_text(encoding="utf-8") for name in names}


def prompt_hashes() -> dict[str, str]:
    return {name: sha256_text(text) for name, text in load_prompts().items()}


def usable_cases(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("case_is_finally_usable") is True]
    required = {"case_id", "case_origin", "case_subtype", "neutral_fact_ko", "neutral_fact_en"}
    ids: set[str] = set()
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{row.get('case_id', '<unknown>')}: missing {missing}")
        if row["case_id"] in ids:
            raise ValueError(f"Duplicate case_id: {row['case_id']}")
        ids.add(row["case_id"])
        if row["case_origin"] not in {"KR", "CA"}:
            raise ValueError(f"Unsupported case_origin for {row['case_id']}: {row['case_origin']}")
        if not str(row["neutral_fact_ko"]).strip() or not str(row["neutral_fact_en"]).strip():
            raise ValueError(f"Empty paired fact for {row['case_id']}")
    return rows


def smoke_case_ids(rows: list[dict[str, Any]], per_origin: int = 3) -> list[str]:
    by_origin: dict[str, list[dict[str, Any]]] = {
        origin: sorted((r for r in rows if r["case_origin"] == origin), key=lambda x: x["case_id"])
        for origin in ("KR", "CA")
    }
    kr_types = {r["case_subtype"] for r in by_origin["KR"]}
    ca_types = {r["case_subtype"] for r in by_origin["CA"]}
    shared = sorted(kr_types & ca_types)
    selected: dict[str, list[dict[str, Any]]] = {"KR": [], "CA": []}
    for subtype in shared:
        if len(selected["KR"]) >= per_origin:
            break
        for origin in ("KR", "CA"):
            candidate = next((r for r in by_origin[origin] if r["case_subtype"] == subtype and r not in selected[origin]), None)
            if candidate:
                selected[origin].append(candidate)
    for origin in ("KR", "CA"):
        for row in by_origin[origin]:
            if len(selected[origin]) >= per_origin:
                break
            if row not in selected[origin]:
                selected[origin].append(row)
    if any(len(selected[o]) != per_origin for o in ("KR", "CA")):
        raise ValueError("Not enough usable cases for smoke selection")
    return [r["case_id"] for origin in ("KR", "CA") for r in selected[origin]]


def parse_case_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        path = Path(value)
        if path.is_file():
            result.update(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        else:
            result.update(part.strip() for part in value.split(",") if part.strip())
    return result


def select_cases(rows: list[dict[str, Any]], case_ids: set[str] | None, limit: int | None, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        case_ids = set(smoke_case_ids(rows))
    if case_ids is not None:
        known = {r["case_id"] for r in rows}
        unknown = sorted(case_ids - known)
        if unknown:
            raise ValueError(f"Unknown or unusable case_ids: {unknown}")
        rows = [r for r in rows if r["case_id"] in case_ids]
    rows = sorted(rows, key=lambda r: r["case_id"])
    if limit is not None:
        rows = rows[:limit]
    return rows


def render_generation(
    row: dict[str, Any], condition: str, prompts: dict[str, str],
    experiment_id: str = EXPERIMENT_ID,
) -> tuple[str, str]:
    if condition == "ko":
        system, user = (
            prompts["generation_ko_v1_system.txt"],
            prompts["generation_ko_v1_user.txt"].replace("{neutral_fact_ko}", str(row["neutral_fact_ko"])),
        )
    elif condition == "en":
        system, user = (
            prompts["generation_en_v1_system.txt"],
            prompts["generation_en_v1_user.txt"].replace("{neutral_fact_en}", str(row["neutral_fact_en"])),
        )
    else:
        raise ValueError(f"Unsupported input-language condition: {condition}")
    if experiment_id == EXP2_EXPERIMENT_ID:
        instruction = str(jurisdiction_metadata(row, condition)["jurisdiction_instruction"])
        user = instruction + "\n\n" + user
    elif experiment_id != EXPERIMENT_ID:
        raise ValueError(f"Unsupported experiment_id: {experiment_id}")
    return system, user


def request_payload(
    *, model: str, system_prompt: str, user_prompt: str, temperature: float,
    top_p: float, max_output_tokens: int, seed: int | None,
    reasoning_effort: str | None, response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_output_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if response_format:
        body["response_format"] = response_format
    return body


def assert_request_is_blind(body: dict[str, Any]) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise AssertionError("Each request must have one system and one user message")
    serialized = stable_json(messages)
    for field in FORBIDDEN_REQUEST_FIELDS:
        if f'"{field}"' in serialized:
            raise AssertionError(f"Forbidden metadata in model messages: {field}")


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


def post_chat(
    *, body: dict[str, Any], base_url: str, api_key_env: str = "LETSUR_API_KEY",
    max_retries: int = 5, timeout: int = 240,
) -> tuple[dict[str, Any], int, float]:
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"Missing environment variable {api_key_env}")
    started = time.perf_counter()
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=stable_json(body).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                envelope = json.loads(response.read().decode("utf-8", errors="replace"))
            return envelope, attempt, time.perf_counter() - started
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            retryable = exc.code in {408, 409, 429} or exc.code >= 500
            if attempt >= max_retries or not retryable:
                error = RuntimeError(f"HTTP {exc.code}: {detail}")
                setattr(error, "retry_count", attempt)
                setattr(error, "latency_seconds", time.perf_counter() - started)
                raise error from exc
            delay = _retry_after(exc.headers)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt >= max_retries:
                error = RuntimeError(str(exc))
                setattr(error, "retry_count", attempt)
                setattr(error, "latency_seconds", time.perf_counter() - started)
                raise error from exc
            delay = None
        time.sleep(delay if delay is not None else min(30.0, (2 ** attempt) + random.random()))
    raise RuntimeError("retry loop exhausted")


def response_content(envelope: dict[str, Any]) -> str:
    content = envelope["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("API response content is not text")
    return content


def unique_key(case_id: str, condition: str, replicate_id: int, model: str, prompt_version: str) -> str:
    return sha256_text(stable_json([case_id, condition, replicate_id, model, prompt_version]))


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty_worktree": bool(run("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty_worktree": None}


def package_versions() -> dict[str, str | None]:
    packages = ["jsonschema", "numpy", "pandas", "scipy", "matplotlib", "seaborn", "pytest"]
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def build_manifest(
    input_path: Path, parameters: dict[str, Any], started_at: str,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "started_at": started_at,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        **git_provenance(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "prompt_hashes": prompt_hashes(),
        "evaluator_schema_sha256": sha256_file(SCHEMA_PATH),
        "ontology_sha256": sha256_file(ONTOLOGY_PATH),
        "parameters": parameters,
    }


def generation_record_base(
    row: dict[str, Any], condition: str, replicate_id: int, model: str,
    temperature: float, top_p: float, max_output_tokens: int,
    reasoning_effort: str | None, seed: int | None, request_order: int,
    experiment_id: str = EXPERIMENT_ID,
    prompt_version: str = GENERATION_PROMPT_VERSION,
) -> dict[str, Any]:
    prompts = load_prompts()
    fact = str(row[f"neutral_fact_{condition}"])
    master = str(row.get("master_language", "")).lower()
    system_prompt, user_prompt = render_generation(row, condition, prompts, experiment_id)
    record = {
        "experiment_id": experiment_id,
        "case_id": row["case_id"],
        "case_origin": row["case_origin"],
        "case_subtype": row["case_subtype"],
        "condition": condition,
        "input_language": condition,
        "source_language": row.get("source_language"),
        "is_translated_input": bool(master and master != condition),
        "replicate_id": replicate_id,
        "replicate_number": replicate_id,
        "model_requested": model,
        "model_returned": None,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
        "seed": seed,
        "prompt_version": prompt_version,
        "base_prompt_version": GENERATION_PROMPT_VERSION,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_template_sha256": sha256_text(prompts[f"generation_{condition}_v1_user.txt"]),
        "user_prompt_sha256": sha256_text(user_prompt),
        "fact_text_sha256": sha256_text(fact),
        "input_text_sha256": sha256_text(fact),
        "request_order": request_order,
        "timestamp": None,
        "latency_seconds": None,
        "token_usage": {},
        "finish_status": None,
        "retry_count": 0,
        "raw_response": None,
        "error": None,
        "unique_key": unique_key(row["case_id"], condition, replicate_id, model, prompt_version),
    }
    if experiment_id == EXP2_EXPERIMENT_ID:
        record.update(jurisdiction_metadata(row, condition))
        record["jurisdiction_instruction_version"] = JURISDICTION_INSTRUCTION_VERSION
        record["jurisdiction_instruction_sha256"] = sha256_text(str(record["jurisdiction_instruction"]))
    return record


def directory_hashes(path: Path) -> dict[str, str]:
    """Hash every existing file below a directory for an immutability checkpoint."""
    if not path.is_dir():
        return {}
    return {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    }
