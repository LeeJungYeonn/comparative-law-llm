#!/usr/bin/env python3
"""Checkpointed Letsur generation for the frozen 1,920-request Experiment 1 run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.exp1.run_implicit_jurisdiction_full import (
    CASES_PATH, EXPECTED_REQUESTS, FACTS_PATH, PRIMARY_MODEL_ID, REPO_ROOT,
    REQUEST_ORDER_SEED, SYSTEM_PROMPTS, USER_TEMPLATES, canonical_json,
    read_jsonl, sha_file, sha_text, write_csv, write_json,
)


OUTPUT_DIR = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_full_v1"
MANIFEST_PATH = OUTPUT_DIR / "final_experiment_manifest.json"
REQUESTS_PATH = OUTPUT_DIR / "final_request_manifest.csv"
FINAL_QC_GATE = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_final_qc_v6/final_generation_gate.json"
ORIENTATION_MANIFEST = REPO_ROOT / "legal_lexicon/legal_lexicon_orientation_v1_1/run_manifest.json"
ANALYSIS_SCRIPT = REPO_ROOT / "scripts/exp1/analyze_implicit_jurisdiction_full.py"
EXECUTION_LOCK = OUTPUT_DIR / "EXECUTION_FROZEN.lock.json"
RAW_RESPONSES = OUTPUT_DIR / "raw_responses.jsonl"
ATTEMPT_LOG = OUTPUT_DIR / "request_attempt_log.jsonl"
GENERATION_QC = OUTPUT_DIR / "generation_qc.csv"
FAILED_REQUESTS = OUTPUT_DIR / "failed_requests.csv"
GENERATION_SUMMARY = OUTPUT_DIR / "generation_summary.json"

BASE_URL = "https://gw.letsur.ai/v1"
API_KEY_ENV = "LETSUR_API_KEY"
CONCURRENCY = 4
BATCH_SIZE = 20
MAX_RETRIES = 3
TIMEOUT_SECONDS = 180
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 8000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_request_manifest() -> list[dict]:
    with REQUESTS_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_REQUESTS or len({row["request_id"] for row in rows}) != EXPECTED_REQUESTS:
        raise RuntimeError("Frozen request manifest is not 1,920 unique requests")
    cells: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["input_language"], row["reasoning_effort"], row["replicate"])
        cells[key] = cells.get(key, 0) + 1
    if len(cells) != 12 or set(cells.values()) != {160}:
        raise RuntimeError({"unbalanced_factorial_cells": cells})
    return rows


def prepare_lock() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    final_qc = json.loads(FINAL_QC_GATE.read_text(encoding="utf-8"))
    orientation = json.loads(ORIENTATION_MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "PREFLIGHT_PASSED_READY_FOR_FULL_RUN" or manifest["blockers"]:
        raise RuntimeError("Experiment manifest is not blocker-free")
    if final_qc["status"] != "PREFLIGHT_PASSED_READY_FOR_FULL_RUN":
        raise RuntimeError("Final QC gate is not passed")
    if orientation["status"] != "FROZEN_PRE_GENERATION" or orientation["one_sided_orientation_concept_count"] != 0:
        raise RuntimeError("Orientation lexicon is not frozen and bilingual")
    read_request_manifest()
    if EXECUTION_LOCK.exists():
        raise RuntimeError(f"Refusing to overwrite execution lock: {EXECUTION_LOCK}")
    pre_generation_manifest = OUTPUT_DIR / "pre_generation_experiment_manifest.json"
    pre_generation_manifest.write_bytes(MANIFEST_PATH.read_bytes())
    protected = [
        pre_generation_manifest, REQUESTS_PATH, OUTPUT_DIR / "final_primary_case_manifest.csv",
        OUTPUT_DIR / "prompt_manifest.json", CASES_PATH, FACTS_PATH, FINAL_QC_GATE,
        ORIENTATION_MANIFEST, Path(__file__).resolve(), ANALYSIS_SCRIPT,
        REPO_ROOT / "scripts/lexicon/confirmatory_measurement_v1.py",
        REPO_ROOT / "outputs_v2/robustness_gate2/multivariate_full_6d_no_procedure.json",
    ]
    lock = {
        "status": "FROZEN_READY_TO_EXECUTE",
        "created_at_utc": now(),
        "expected_requests": EXPECTED_REQUESTS,
        "model": PRIMARY_MODEL_ID,
        "reasoning_efforts": ["low", "medium", "high"],
        "replicates": 2,
        "master_seed": REQUEST_ORDER_SEED,
        "generation_parameters": {
            "temperature": TEMPERATURE, "top_p": TOP_P, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "concurrency": CONCURRENCY, "batch_size": BATCH_SIZE, "max_retries": MAX_RETRIES,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
        "protected_artifact_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): sha_file(path) for path in protected
        },
        "api_requests_sent_at_lock": 0,
    }
    write_json(EXECUTION_LOCK, lock)
    return lock


def verify_lock() -> None:
    lock = json.loads(EXECUTION_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_READY_TO_EXECUTE" or lock["expected_requests"] != EXPECTED_REQUESTS:
        raise RuntimeError("Execution lock is invalid")
    mismatches = {}
    for relative, expected in lock["protected_artifact_sha256"].items():
        path = REPO_ROOT / relative
        actual = sha_file(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError({"execution_lock_hash_mismatches": mismatches})


def load_key() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required") from exc
    load_dotenv(REPO_ROOT / ".env", override=False)
    key = os.getenv(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is unavailable")
    return key


def request_payload(row: dict, fact: dict) -> tuple[dict, str]:
    language = row["input_language"].lower()
    neutral_fact = fact[f"neutral_fact_{language}"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[language]},
        {"role": "user", "content": USER_TEMPLATES[language].format(neutral_fact=neutral_fact)},
    ]
    if sha_text(canonical_json(messages)) != row["input_hash"]:
        raise RuntimeError(f"Frozen input hash mismatch: {row['request_id']}")
    body = {
        "model": PRIMARY_MODEL_ID, "messages": messages,
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": REQUEST_ORDER_SEED, "reasoning_effort": row["reasoning_effort"],
    }
    return body, neutral_fact


def http_chat(body: dict, api_key: str) -> tuple[dict, Any]:
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=canonical_json(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8", errors="replace")), response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error = RuntimeError(f"HTTP {exc.code}: {detail[:1200]}")
        setattr(error, "status", exc.code)
        setattr(error, "headers", exc.headers)
        raise error from exc


def retryable(exc: BaseException) -> bool:
    message = str(exc)
    if "usage_limit_exceeded" in message or "COST limit exceeded" in message:
        return False
    status = getattr(exc, "status", None)
    return status in {408, 409, 429} or (isinstance(status, int) and status >= 500) or isinstance(exc, (urllib.error.URLError, TimeoutError))


def call_one(row: dict, fact: dict, api_key: str) -> tuple[dict, list[dict]]:
    body, _ = request_payload(row, fact)
    body_hash = sha_text(canonical_json(body))
    attempts = []
    started = now()
    start_clock = time.perf_counter()
    last_error: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        attempt_started = now()
        try:
            envelope, headers = http_chat(body, api_key)
            choice = (envelope.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content")
            if not isinstance(text, str):
                raise RuntimeError("Response has no textual choices[0].message.content")
            attempts.append({
                "request_id": row["request_id"], "attempt": attempt, "started_at_utc": attempt_started,
                "completed_at_utc": now(), "status": "success", "http_status": 200,
                "request_body_sha256": body_hash, "error": "",
            })
            result = {
                **row, "request_started_at": started, "response_received_at": now(),
                "latency_seconds": round(time.perf_counter() - start_clock, 6), "api_attempts": attempt,
                "model_returned": envelope.get("model"), "response_id": envelope.get("id"),
                "finish_status": choice.get("finish_reason"), "token_usage": envelope.get("usage") or {},
                "response_text": text, "response_sha256": sha_text(text),
                "response_envelope_sha256": sha_text(canonical_json(envelope)), "response_envelope": envelope,
                "request_body_sha256": body_hash, "request_reasoning_effort_transmitted": True, "error": None,
            }
            return result, attempts
        except BaseException as exc:
            last_error = exc
            message = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", str(exc), flags=re.I)[:1600]
            attempts.append({
                "request_id": row["request_id"], "attempt": attempt, "started_at_utc": attempt_started,
                "completed_at_utc": now(), "status": "error", "http_status": getattr(exc, "status", ""),
                "request_body_sha256": body_hash, "error": message,
            })
            if attempt >= MAX_RETRIES or not retryable(exc):
                break
            time.sleep(min(2 ** (attempt - 1) + random.Random(f"{row['request_id']}:{attempt}").random(), 10))
    return {
        **row, "request_started_at": started, "response_received_at": now(),
        "latency_seconds": round(time.perf_counter() - start_clock, 6), "api_attempts": len(attempts),
        "model_returned": None, "response_id": None, "finish_status": None, "token_usage": {},
        "response_text": "", "response_sha256": sha_text(""), "response_envelope_sha256": "",
        "response_envelope": None, "request_body_sha256": body_hash,
        "request_reasoning_effort_transmitted": True, "error": str(last_error),
    }, attempts


def qc_row(response: dict) -> dict:
    text = response.get("response_text") or ""
    language = response["input_language"]
    hangul = len(re.findall(r"[가-힣]", text))
    language_pass = hangul >= 50 if language == "KO" else hangul < 20
    checks = {
        "nonempty_response_pass": bool(text.strip()),
        "correct_output_language_pass": language_pass,
        "finish_status_stop_pass": response.get("finish_status") == "stop",
        "response_hash_pass": response.get("response_sha256") == sha_text(text),
        "model_returned_present_pass": bool(response.get("model_returned")),
        "effort_logged_pass": bool(response.get("request_reasoning_effort_transmitted")),
        "api_error_absent_pass": response.get("error") is None,
    }
    return {
        "request_id": response["request_id"], "case_id": response["case_id"],
        "input_language": language, "reasoning_effort": response["reasoning_effort"],
        "replicate": response["replicate"], "model_returned": response.get("model_returned") or "",
        **checks, "response_character_count": len(text),
        "qc_failures": "|".join(name for name, passed in checks.items() if not passed),
        "qc_pass": all(checks.values()),
    }


def execute() -> None:
    verify_lock()
    requests = read_request_manifest()
    facts = {row["case_id"]: row for row in read_jsonl(FACTS_PATH)}
    api_key = load_key()
    existing = read_jsonl(RAW_RESPONSES) if RAW_RESPONSES.exists() else []
    existing_ids = {row["request_id"] for row in existing}
    if len(existing_ids) != len(existing) or not existing_ids <= {row["request_id"] for row in requests}:
        raise RuntimeError("Response checkpoint contains duplicate or unknown request IDs")
    pending = [row for row in requests if row["request_id"] not in existing_ids]
    print(canonical_json({"status": "GENERATION_START_OR_RESUME", "completed": len(existing), "pending": len(pending)}), flush=True)
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset:offset + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(call_one, row, facts[row["case_id"]], api_key): row for row in batch}
            results = [future.result() for future in as_completed(futures)]
        results.sort(key=lambda item: int(item[0]["randomized_order"]))
        for response, attempts in results:
            for attempt in attempts:
                append_jsonl(ATTEMPT_LOG, attempt)
            append_jsonl(RAW_RESPONSES, response)
        print(canonical_json({"completed": len(existing) + min(offset + len(batch), len(pending)), "total": EXPECTED_REQUESTS}), flush=True)

    responses = read_jsonl(RAW_RESPONSES)
    if len(responses) != EXPECTED_REQUESTS or len({row["request_id"] for row in responses}) != EXPECTED_REQUESTS:
        raise RuntimeError("Generation checkpoint is incomplete or duplicated")
    qc = [qc_row(row) for row in responses]
    write_csv(GENERATION_QC, qc, list(qc[0]))
    failed = [row for row in qc if not row["qc_pass"]]
    write_csv(FAILED_REQUESTS, failed, list(qc[0]))
    attempts = read_jsonl(ATTEMPT_LOG)
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    summary = {
        "status": "GENERATION_COMPLETE_QC_PASS" if not failed else "GENERATION_COMPLETE_WITH_QC_FAILURES",
        "completed_at_utc": now(), "planned_requests": EXPECTED_REQUESTS,
        "response_records": len(responses), "successful_api_responses": sum(row.get("error") is None for row in responses),
        "technical_retry_attempts": len(attempts) - len(responses), "generation_qc_pass": len(qc) - len(failed),
        "generation_qc_fail": len(failed),
        "token_usage": {field: sum(int((row.get("token_usage") or {}).get(field) or 0) for row in responses) for field in usage_fields},
        "model_returned_values": sorted({str(row.get("model_returned")) for row in responses if row.get("model_returned")}),
    }
    write_json(GENERATION_SUMMARY, summary)
    print(canonical_json(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-lock", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock == args.execute:
        raise RuntimeError("Choose exactly one of --prepare-lock or --execute")
    if args.prepare_lock:
        print(canonical_json(prepare_lock()))
    else:
        execute()


if __name__ == "__main__":
    main()
