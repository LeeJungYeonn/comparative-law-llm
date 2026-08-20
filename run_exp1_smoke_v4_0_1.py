from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl
from pipeline_v2.llm_runtime import (
    DEFAULT_LETSUR_BASE_URL,
    DEFAULT_LETSUR_MODEL,
    _content,
    _http_chat,
    load_environment,
    require_api_key,
)


CASES_PATH = Path("outputs_v2/v4.0.1/final_cases_200_v4_0_1.jsonl")
FACTS_PATH = Path("outputs_v2/v4.0.1/final_fact_patterns_200_v4_0_1.jsonl")
OUTPUT = Path("outputs_exp1/smoke")
MODEL = DEFAULT_LETSUR_MODEL
BASE_URL = DEFAULT_LETSUR_BASE_URL
PROMPT_VERSION = "exp1-court-opinion-v1"
SYSTEM_VERSIONS = {"ko": "exp1-court-opinion-system-ko-v1", "en": "exp1-court-opinion-system-en-v1"}
USER_VERSIONS = {"ko": "exp1-court-opinion-user-ko-v1", "en": "exp1-court-opinion-user-en-v1"}
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 8000
SEED = 20260730
REPLICATE = 1
CONCURRENCY = 4
MAX_RETRIES = 3

SYSTEM_PROMPTS = {
    "ko": (
        "당신은 제시된 기록만을 바탕으로 민사상 손해배상 분쟁에 관한 이유 있는 판결문을 작성하는 재판관입니다. "
        "입력과 같은 언어로 답하십시오. 사실, 증거 또는 소송 경과를 새로 만들지 말고, 법령이나 판례를 지어내지 마십시오. "
        "정보 부족으로 판단이 제한되는 사항은 그 한계를 명시하십시오."
    ),
    "en": (
        "You are a judge writing a reasoned judicial opinion in a civil damages dispute based only on the supplied record. "
        "Respond in the same language as the input. Do not add facts, evidence, or procedural history, and do not invent statutes or precedents. "
        "State any limitation on the decision caused by insufficient information."
    ),
}

USER_TEMPLATES = {
    "ko": (
        "아래 사실관계만을 사용하여 이유 있는 판결문을 작성하십시오. 기록이 뒷받침하는 범위에서 당사자들의 주장, 법적 쟁점, "
        "적용 가능한 원칙, 사실관계에 대한 적용, 책임, 손해 및 최종 판단을 자연스러운 판결문 형식으로 다루십시오. "
        "사실관계에 없는 내용을 추측하지 마십시오. 필요한 정보가 부족하면 그 한계를 밝히십시오. 인용이나 고정된 항목 제목은 요구되지 않습니다.\n\n"
        "[사실관계]\n{fact}"
    ),
    "en": (
        "Using only the facts below, write a reasoned judicial opinion. In a natural court-opinion style, address, to the extent supported by the record, "
        "the parties' arguments, legal issues, applicable principles, application to the facts, liability, damages, and the final judgment. "
        "Do not speculate beyond the facts. If necessary information is missing, state that limitation. Citations and fixed section headings are not required.\n\n"
        "[Facts]\n{fact}"
    ),
}

SPECIFIC_JURISDICTIONS = re.compile(
    r"(?i)\b(?:Korea|Korean|Pennsylvania|Michigan|Louisiana|Nevada|West Virginia|United States|U\.S\.)\b|"
    r"대한민국|한국법|미국법|펜실베이니아|미시간|루이지애나|네바다|웨스트버지니아"
)
PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]+\]")
NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:[.,]\d+)*(?:%|원|달러|dollars?)?", re.I)
AUTHORITY = re.compile(
    r"§|(?i:\bsection\s+\d+)|\b[A-Z][A-Za-z.'-]+\s+v\.\s+[A-Z][A-Za-z.'-]+|"
    r"\b\d+\s+(?:U\.S\.|S\.\s?Ct\.|F\.\s?(?:2d|3d|4th)?|N\.?W\.?\s?2d|"
    r"A\.?\s?\d+d|P\.?\s?\d+d|S\.?E\.?\s?2d|So\.?\s?\d+d|N\.?E\.?\s?\d+d)\s+\d+\b|"
    r"민법\s*제?\d+조|법률\s*제?\d+|대법원\s*\d{4}[.년]|\d{2,4}[가-힣]+\d+\s*판결"
)
REFUSAL = re.compile(r"(?i)I (?:cannot|can't|am unable)|요청을 수행할 수 없|답변할 수 없")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_score(ids: tuple[str, ...], label: str) -> str:
    return sha_text(f"{PROMPT_VERSION}|{SEED}|{label}|{'|'.join(sorted(ids))}")


def select_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    development = [row for row in cases if row.get("analysis_split") == "development"]
    chosen: list[dict[str, Any]] = []
    for country in ("KR", "US"):
        pool = [row for row in development if row["origin_country"] == country]
        candidates = []
        for combo in itertools.combinations(pool, 4):
            domain_diversity = len({row["primary_domain"] for row in combo})
            state_diversity = len({row.get("origin_state") for row in combo}) if country == "US" else 0
            ids = tuple(row["case_id"] for row in combo)
            candidates.append((-domain_diversity, -state_diversity, stable_score(ids, country), combo))
        chosen.extend(min(candidates, key=lambda item: item[:3])[3])
    return sorted(chosen, key=lambda row: (row["origin_country"], row["case_id"]))


def request_record(
    case: dict[str, Any], fact: dict[str, Any], language: str, phase: str,
    effort: str, request_order: int,
) -> dict[str, Any]:
    input_text = fact[f"neutral_fact_{language}"]
    system = SYSTEM_PROMPTS[language]
    user = USER_TEMPLATES[language].format(fact=input_text)
    return {
        "experiment_id": "exp1-input-language-smoke-v4.0.1",
        "corpus_version": "kr-us-highcourt-corpus-v4.0.1",
        "phase": phase,
        "case_id": case["case_id"],
        "case_origin": case["origin_country"],
        "origin_state": case.get("origin_state"),
        "primary_domain": case["primary_domain"],
        "analysis_split": case["analysis_split"],
        "input_language": language,
        "input_text": input_text,
        "input_hash": sha_text(input_text),
        "model_requested": MODEL,
        "reasoning_effort": effort,
        "replicate": REPLICATE,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_version": SYSTEM_VERSIONS[language],
        "user_prompt_version": USER_VERSIONS[language],
        "system_prompt": system,
        "user_prompt": user,
        "system_prompt_hash": sha_text(system),
        "user_prompt_hash": sha_text(user),
        "specific_jurisdiction_in_prompt": bool(SPECIFIC_JURISDICTIONS.search(system + "\n" + user)),
        "request_order": request_order,
        "request_key": sha_text(
            f"{phase}|{case['case_id']}|{language}|{effort}|{REPLICATE}|{PROMPT_VERSION}|{MODEL}"
        ),
    }


def call_one(item: dict[str, Any], api_key: str) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["user_prompt"]},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": SEED,
        "reasoning_effort": item["reasoning_effort"],
    }
    started = now()
    started_clock = time.perf_counter()
    last_error: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            envelope, _ = _http_chat(body=body, base_url=BASE_URL, api_key=api_key)
            response = _content(envelope)
            received = now()
            choice = (envelope.get("choices") or [{}])[0]
            return {
                **{key: value for key, value in item.items() if key not in {"system_prompt", "user_prompt", "input_text"}},
                "request_started_at": started,
                "response_received_at": received,
                "latency_seconds": round(time.perf_counter() - started_clock, 6),
                "api_attempts": attempt,
                "request_reasoning_effort_transmitted": body.get("reasoning_effort") == item["reasoning_effort"],
                "request_body_hash": sha_text(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                "response_id": envelope.get("id"),
                "model_returned": envelope.get("model"),
                "finish_status": choice.get("finish_reason"),
                "token_usage": envelope.get("usage") or {},
                "response_text": response,
                "response_hash": sha_text(response),
                "response_envelope_hash": sha_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                "error": None,
            }
        except BaseException as exc:
            last_error = exc
            if "usage_limit_exceeded" in str(exc) or "COST limit exceeded" in str(exc):
                break
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
    return {
        **{key: value for key, value in item.items() if key not in {"system_prompt", "user_prompt", "input_text"}},
        "request_started_at": started,
        "response_received_at": now(),
        "latency_seconds": round(time.perf_counter() - started_clock, 6),
        "api_attempts": attempt,
        "request_reasoning_effort_transmitted": body.get("reasoning_effort") == item["reasoning_effort"],
        "response_text": "",
        "response_hash": sha_text(""),
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def response_qc(item: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    raw_text = response.get("response_text") or ""
    text = raw_text
    language = item["input_language"]
    hangul_count = len(re.findall(r"[가-힣]", text))
    language_pass = hangul_count >= 50 if language == "ko" else hangul_count < 10
    if language == "ko":
        categories = [
            ("쟁점", "문제"), ("원칙", "법리", "요건"), ("사실", "기록"),
            ("책임",), ("손해", "배상"), ("판단", "결론", "주문"),
        ]
    else:
        low = text.lower()
        categories = [
            ("issue", "question"), ("principle", "standard", "duty"), ("fact", "record"),
            ("liability", "liable"), ("damage", "compensation"), ("judgment", "conclude", "hold"),
        ]
        text = low
    opinion_score = sum(any(token in text for token in category) for category in categories)
    input_numbers = set(NUMBER.findall(item["input_text"]))
    response_without_numbered_headings = re.sub(r"(?m)^\s*\d+[.)]\s+", "", raw_text)
    response_numbers = set(NUMBER.findall(response_without_numbered_headings))
    introduced_numbers = sorted(response_numbers - input_numbers)
    input_placeholders = set(PLACEHOLDER.findall(item["input_text"]))
    response_placeholders = set(PLACEHOLDER.findall(response.get("response_text") or ""))
    invented_placeholder = sorted(response_placeholders - input_placeholders)
    obvious_authority = sorted(set(AUTHORITY.findall(response.get("response_text") or "")))
    finish = response.get("finish_status")
    checks = {
        "correct_output_language_pass": language_pass,
        "no_jurisdiction_supplied_by_prompt_pass": not item["specific_jurisdiction_in_prompt"],
        "reasoned_judicial_opinion_format_pass": len(response.get("response_text") or "") >= 500 and opinion_score >= 4,
        "no_obvious_invented_facts_pass": not introduced_numbers and not invented_placeholder,
        "no_obvious_fabricated_authority_pass": not obvious_authority,
        "no_truncation_refusal_api_corruption_pass": bool(response.get("response_text")) and finish == "stop" and not REFUSAL.search(response.get("response_text") or "") and response.get("error") is None,
        "logging_complete_pass": all(
            response.get(field) not in (None, "", {})
            for field in (
                "case_id", "case_origin", "input_language", "model_requested", "model_returned",
                "reasoning_effort", "replicate", "prompt_version", "request_started_at",
                "response_received_at", "token_usage", "response_hash", "input_hash",
            )
        ),
        "input_hash_pass": item["input_hash"] == sha_text(item["input_text"]),
        "response_hash_pass": response.get("response_hash") == sha_text(response.get("response_text") or ""),
        "effort_parameter_transmitted_and_logged_pass": bool(response.get("request_reasoning_effort_transmitted")) and response.get("reasoning_effort") == item["reasoning_effort"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "phase": item["phase"],
        "case_id": item["case_id"],
        "case_origin": item["case_origin"],
        "origin_state": item.get("origin_state"),
        "primary_domain": item["primary_domain"],
        "input_language": language,
        "model_requested": item["model_requested"],
        "model_returned": response.get("model_returned"),
        "reasoning_effort": item["reasoning_effort"],
        "replicate": item["replicate"],
        "prompt_version": item["prompt_version"],
        **checks,
        "introduced_numbers": "|".join(introduced_numbers),
        "invented_placeholders": "|".join(invented_placeholder),
        "authority_pattern_hits": "|".join(obvious_authority),
        "opinion_format_signal_count": opinion_score,
        "response_character_count": len(response.get("response_text") or ""),
        "qc_failures": "|".join(failures),
        "qc_pass": not failures,
    }


def run_batch(items: list[dict[str, Any]], api_key: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(call_one, item, api_key): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: row["request_order"])


def write_qc(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unused_path(path: Path) -> Path:
    if not path.exists():
        return path
    for attempt in itertools.count(2):
        candidate = path.with_name(f"{path.stem}_attempt{attempt}{path.suffix}")
        if not candidate.exists():
            return candidate


def finalize_blocked_attempt() -> None:
    checkpoint = OUTPUT / "exp1_smoke_primary_responses_checkpoint_attempt2.jsonl"
    if not checkpoint.exists():
        raise RuntimeError(f"Blocked checkpoint not found: {checkpoint}")
    cases = [dict(row) for row in read_jsonl(CASES_PATH)]
    facts = {row["case_id"]: dict(row) for row in read_jsonl(FACTS_PATH)}
    selected = select_cases(cases)
    items = []
    order = 0
    for case in selected:
        for language in ("ko", "en"):
            order += 1
            items.append(request_record(case, facts[case["case_id"]], language, "primary", "medium", order))
    responses = [dict(row) for row in read_jsonl(checkpoint)]
    response_by_key = {row["request_key"]: row for row in responses}
    qc = [response_qc(item, response_by_key[item["request_key"]]) for item in items]
    paths = {
        "inputs": OUTPUT / "exp1_smoke_inputs_blocked_attempt2.jsonl",
        "responses": OUTPUT / "exp1_smoke_responses_blocked_attempt2.jsonl",
        "qc": OUTPUT / "exp1_smoke_qc_blocked_attempt2.csv",
        "summary": OUTPUT / "exp1_smoke_summary_blocked_attempt2.json",
        "report": OUTPUT / "EXP1_SMOKE_BLOCKED_REPORT_attempt2.md",
    }
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("Blocked-run artifacts already exist; refusing to overwrite them")
    write_jsonl(paths["inputs"], items)
    write_jsonl(paths["responses"], responses)
    write_qc(paths["qc"], qc)
    errors = Counter(row.get("error") or "" for row in responses)
    summary = {
        "experiment_id": "exp1-input-language-smoke-v4.0.1",
        "status": "BLOCKED_API_USAGE_LIMIT",
        "pipeline_ready_to_freeze_for_full_exp1": False,
        "blocker": "Letsur returned HTTP 429 usage_limit_exceeded: COST limit exceeded for every primary request.",
        "model_requested": MODEL,
        "model_identifier_returned": None,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_versions": SYSTEM_VERSIONS,
        "user_prompt_versions": USER_VERSIONS,
        "generation_parameters": {
            "base_url_identifier": BASE_URL,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "seed": SEED,
            "replicate": REPLICATE,
            "primary_reasoning_effort": "medium",
            "concurrency": CONCURRENCY,
        },
        "selected_cases": [
            {"case_id": row["case_id"], "origin": row["origin_country"], "state": row.get("origin_state"), "primary_domain": row["primary_domain"]}
            for row in selected
        ],
        "primary": {"planned": 16, "successful": 0, "failed": 16, "qc_pass": 0},
        "effort_parameter_check": {"status": "NOT_RUN", "reason": "Primary 16 generations did not pass."},
        "error_counts": dict(errors),
        "source_corpus_hashes": {str(CASES_PATH): sha_file(CASES_PATH), str(FACTS_PATH): sha_file(FACTS_PATH)},
        "hypothesis_evaluation_performed": False,
        "pca_performed": False,
    }
    write_json(paths["summary"], summary)
    report = [
        "# Exp 1 smoke test — blocked API attempt 2",
        "",
        "Status: **BLOCKED_API_USAGE_LIMIT**",
        "",
        "- Primary requests: 0/16 successful",
        "- Letsur result: `HTTP 429 usage_limit_exceeded — COST limit exceeded` for all requests",
        "- Additional low/medium/high check: not run because the primary-pass precondition was not met",
        f"- Model requested: `{MODEL}`; returned model identifier: unavailable",
        f"- Prompt version: `{PROMPT_VERSION}`",
        "- Corpus files remained unchanged",
        "- Hypothesis-marker evaluation and PCA were not performed",
        "- Pipeline ready to freeze for full Exp 1: **FALSE**",
        "",
        "The legacy fixed-heading analyst prompt was replaced only to satisfy the required judicial-opinion format and KO/EN semantic equivalence. "
        "No change was based on jurisdictional-marker strength.",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    checksum_path = OUTPUT / "SHA256SUMS_exp1_smoke_blocked_attempt2.txt"
    checksum_path.write_text(
        "".join(f"{sha_file(path)}  {path.as_posix()}\n" for path in sorted(paths.values())), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def finalize_existing_run1() -> None:
    """Re-QC completed API responses and write a non-overwriting versioned final artifact set."""
    destination = OUTPUT / "v4_0_1_run1"
    source_inputs = OUTPUT / "exp1_smoke_inputs.jsonl"
    source_responses = OUTPUT / "exp1_smoke_responses.jsonl"
    source_prompt_manifest = OUTPUT / "exp1_smoke_prompt_manifest.json"
    for path in (source_inputs, source_responses, source_prompt_manifest):
        if not path.exists():
            raise RuntimeError(f"Completed smoke source artifact not found: {path}")
    if destination.exists():
        raise RuntimeError(f"Versioned destination already exists; refusing to overwrite it: {destination}")

    inputs = [dict(row) for row in read_jsonl(source_inputs)]
    responses = [dict(row) for row in read_jsonl(source_responses)]
    if len(inputs) != 28 or len(responses) != 28:
        raise RuntimeError("Expected 28 completed records: 16 primary plus 12 effort-parameter checks")
    input_by_key = {row["request_key"]: row for row in inputs}
    response_by_key = {row["request_key"]: row for row in responses}
    if len(input_by_key) != 28 or set(input_by_key) != set(response_by_key):
        raise RuntimeError("Input/response request keys are incomplete, duplicated, or mismatched")
    qc = [response_qc(item, response_by_key[item["request_key"]]) for item in inputs]
    if not all(row["qc_pass"] for row in qc):
        raise RuntimeError({
            "re_qc": "FAILED",
            "failures": [
                {"case_id": row["case_id"], "language": row["input_language"],
                 "effort": row["reasoning_effort"], "failures": row["qc_failures"]}
                for row in qc if not row["qc_pass"]
            ],
        })

    primary_inputs = [row for row in inputs if row["phase"] == "primary"]
    effort_inputs = [row for row in inputs if row["phase"] == "effort_parameter_check"]
    primary_qc = [row for row in qc if row["phase"] == "primary"]
    effort_qc = [row for row in qc if row["phase"] == "effort_parameter_check"]
    if len(primary_inputs) != 16 or len(effort_inputs) != 12:
        raise RuntimeError("Smoke phase counts differ from the frozen 16 + 12 plan")
    if {row["reasoning_effort"] for row in primary_inputs} != {"medium"}:
        raise RuntimeError("Primary effort setting is not uniformly medium")
    effort_case_ids = sorted({row["case_id"] for row in effort_inputs})
    if len(effort_case_ids) != 2:
        raise RuntimeError("Effort check must contain exactly two development cases")
    for case_id in effort_case_ids:
        combinations = {
            (row["input_language"], row["reasoning_effort"])
            for row in effort_inputs if row["case_id"] == case_id
        }
        if combinations != set(itertools.product(("ko", "en"), ("low", "medium", "high"))):
            raise RuntimeError(f"Effort matrix is incomplete for {case_id}")

    selected_by_id: dict[str, dict[str, Any]] = {}
    for row in primary_inputs:
        selected_by_id.setdefault(row["case_id"], {
            "case_id": row["case_id"], "origin": row["case_origin"],
            "state": row.get("origin_state"), "primary_domain": row["primary_domain"],
        })
    selected_cases = sorted(selected_by_id.values(), key=lambda row: (row["origin"], row["case_id"]))
    source_hashes_before = {str(CASES_PATH): sha_file(CASES_PATH), str(FACTS_PATH): sha_file(FACTS_PATH)}
    expected_hashes = {
        str(CASES_PATH): "c9b6cffa2b1c75b67c2d7e031a604a26b49c57c317a61a649731f27127d1ec23",
        str(FACTS_PATH): "5e50e1c067abd9c7167f0bd36896523c6a5264ece5b2ee06d8a46a22d3814a8a",
    }
    if source_hashes_before != expected_hashes:
        raise RuntimeError("Frozen corpus hashes differ from the v4.0.1 freeze manifest")

    check_names = [key for key in primary_qc[0] if key.endswith("_pass") and key != "qc_pass"]
    primary_check_pass_counts = {key: sum(bool(row[key]) for row in primary_qc) for key in check_names}
    effort_by_setting = {
        effort: {
            "generation_count": sum(row["reasoning_effort"] == effort for row in effort_qc),
            "qc_pass": sum(row["reasoning_effort"] == effort and row["qc_pass"] for row in effort_qc),
            "parameter_transmission_and_logging_pass": all(
                row["effort_parameter_transmitted_and_logged_pass"]
                for row in effort_qc if row["reasoning_effort"] == effort
            ),
        }
        for effort in ("low", "medium", "high")
    }
    corrections = [
        {
            "change": "Narrowed the automatic U.S. reporter-citation QC regex so the input-grounded date phrase '1996 or 1997' is not classified as authority.",
            "justification": "Fabricated-authority QC false-positive correction; prompts and responses were unchanged.",
        },
        {
            "change": "Excluded line-leading numbered section headings such as '1.' through '5.' from the new-fact number comparison.",
            "justification": "Invented-fact QC false-positive correction; prompts and responses were unchanged.",
        },
    ]
    source_hashes_after = {str(CASES_PATH): sha_file(CASES_PATH), str(FACTS_PATH): sha_file(FACTS_PATH)}
    source_unchanged = source_hashes_before == source_hashes_after
    summary = {
        "experiment_id": "exp1-input-language-smoke-v4.0.1",
        "artifact_version": "v4_0_1_run1",
        "status": "PASS",
        "pipeline_ready_to_freeze_for_full_exp1": source_unchanged,
        "hypothesis_evaluation_performed": False,
        "jurisdictional_marker_evaluation_performed": False,
        "pca_performed": False,
        "model_requested": MODEL,
        "model_identifiers_returned": sorted({row["model_returned"] for row in responses}),
        "prompt_version": PROMPT_VERSION,
        "system_prompt_versions": SYSTEM_VERSIONS,
        "user_prompt_versions": USER_VERSIONS,
        "generation_parameters": {
            "base_url_identifier": BASE_URL, "temperature": TEMPERATURE, "top_p": TOP_P,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "seed": SEED, "replicate": REPLICATE,
            "primary_reasoning_effort": "medium", "concurrency": CONCURRENCY, "max_retries": MAX_RETRIES,
        },
        "selected_cases": selected_cases,
        "primary": {
            "case_count": 8, "generation_count": 16, "qc_pass": 16, "qc_fail": 0,
            "per_check_pass_counts": primary_check_pass_counts,
        },
        "effort_parameter_check": {
            "case_ids": effort_case_ids, "efforts": ["low", "medium", "high"],
            "generation_count": 12, "qc_pass": 12, "qc_fail": 0,
            "parameter_transmission_and_logging_pass": all(
                row["effort_parameter_transmitted_and_logged_pass"] for row in effort_qc
            ),
            "by_effort": effort_by_setting,
        },
        "source_corpus_hashes_before": source_hashes_before,
        "source_corpus_hashes_after": source_hashes_after,
        "source_corpus_unchanged": source_unchanged,
        "prompt_changed_during_smoke": False,
        "prompt_pipeline_corrections": corrections,
    }

    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "inputs": destination / "exp1_smoke_inputs.jsonl",
        "responses": destination / "exp1_smoke_responses.jsonl",
        "qc": destination / "exp1_smoke_qc.csv",
        "summary": destination / "exp1_smoke_summary.json",
        "prompt_manifest": destination / "exp1_smoke_prompt_manifest.json",
        "report": destination / "EXP1_SMOKE_REPORT.md",
    }
    write_jsonl(paths["inputs"], inputs)
    write_jsonl(paths["responses"], responses)
    write_qc(paths["qc"], qc)
    write_json(paths["summary"], summary)
    write_json(paths["prompt_manifest"], json.loads(source_prompt_manifest.read_text(encoding="utf-8")))

    report = [
        "# Exp 1 generation smoke test — v4.0.1 run 1", "", "Status: **PASS**", "",
        f"- Model requested/returned: `{MODEL}` / `{', '.join(summary['model_identifiers_returned'])}`",
        f"- Prompt: `{PROMPT_VERSION}`; systems `{SYSTEM_VERSIONS['ko']}`, `{SYSTEM_VERSIONS['en']}`; users `{USER_VERSIONS['ko']}`, `{USER_VERSIONS['en']}`",
        f"- Parameters: `temperature={TEMPERATURE}`, `top_p={TOP_P}`, `max_output_tokens={MAX_OUTPUT_TOKENS}`, `seed={SEED}`, `replicate={REPLICATE}`, `primary_effort=medium`, `concurrency={CONCURRENCY}`",
        "- Primary QC: **16/16 pass**", "- Effort transmission/logging QC: **12/12 pass** (`low` 4/4, `medium` 4/4, `high` 4/4)",
        f"- Frozen corpus unchanged: **{str(source_unchanged).upper()}**",
        f"- Ready to freeze the full Exp 1 generation pipeline: **{str(source_unchanged).upper()}**",
        "- Jurisdictional-marker/hypothesis evaluation: not performed", "- PCA: not performed", "",
        "## Selected development cases", "",
    ]
    report.extend(
        f"- `{row['case_id']}` — {row['origin']}"
        + (f" / {row['state']}" if row.get("state") else "")
        + f" / {row['primary_domain']}"
        for row in selected_cases
    )
    report += ["", "## Additional effort check", "", f"Cases: `{effort_case_ids[0]}`, `{effort_case_ids[1]}`. Both KO/EN inputs were run at low, medium, and high; all request bodies and logs preserved the requested value.", "", "## Prompt and pipeline corrections", "", "No prompt change was made; the existing frozen court-opinion prompt was used. Two QC-only false-positive rules were narrowed: a date phrase was no longer treated as a reporter citation, and numbered section headings were excluded from invented-fact number checks. No correction was based on Korean/U.S. marker strength.", "", "## Artifact SHA-256", ""]
    report.extend(f"- `{path.name}`: `{sha_file(path)}`" for key, path in paths.items() if key != "report")
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    checksum_paths = list(paths.values())
    (destination / "SHA256SUMS_exp1_smoke.txt").write_text(
        "".join(f"{sha_file(path)}  {path.name}\n" for path in sorted(checksum_paths)), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    required = [
        OUTPUT / "exp1_smoke_inputs.jsonl", OUTPUT / "exp1_smoke_responses.jsonl",
        OUTPUT / "exp1_smoke_qc.csv", OUTPUT / "exp1_smoke_summary.json",
        OUTPUT / "EXP1_SMOKE_REPORT.md",
    ]
    if any(path.exists() for path in required):
        raise RuntimeError("Smoke artifacts already exist; refusing to overwrite them")
    source_hashes_before = {str(CASES_PATH): sha_file(CASES_PATH), str(FACTS_PATH): sha_file(FACTS_PATH)}
    cases = [dict(row) for row in read_jsonl(CASES_PATH)]
    facts = {row["case_id"]: dict(row) for row in read_jsonl(FACTS_PATH)}
    selected = select_cases(cases)
    if len(selected) != 8 or sum(row["origin_country"] == "KR" for row in selected) != 4:
        raise RuntimeError("Deterministic selector did not produce 4 KR + 4 US")
    if len({row["primary_domain"] for row in selected if row["origin_country"] == "KR"}) != 4:
        raise RuntimeError("KR smoke sample lacks four-domain diversity")
    if len({row["primary_domain"] for row in selected if row["origin_country"] == "US"}) != 4:
        raise RuntimeError("US smoke sample lacks four-domain diversity")
    if len({row["origin_state"] for row in selected if row["origin_country"] == "US"}) != 4:
        raise RuntimeError("US smoke sample lacks four-state diversity")

    primary_items = []
    order = 0
    for case in selected:
        for language in ("ko", "en"):
            order += 1
            primary_items.append(request_record(case, facts[case["case_id"]], language, "primary", "medium", order))
    if any(item["specific_jurisdiction_in_prompt"] for item in primary_items):
        raise RuntimeError("A specific jurisdiction leaked into a generation prompt")

    load_environment(Path(".env"))
    api_key = require_api_key("LETSUR_API_KEY")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompt_manifest = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_versions": SYSTEM_VERSIONS,
        "user_prompt_versions": USER_VERSIONS,
        "system_prompts": SYSTEM_PROMPTS,
        "user_templates": USER_TEMPLATES,
        "system_prompt_hashes": {key: sha_text(value) for key, value in SYSTEM_PROMPTS.items()},
        "user_template_hashes": {key: sha_text(value) for key, value in USER_TEMPLATES.items()},
        "specific_jurisdiction_names_present": False,
    }
    prompt_manifest_path = OUTPUT / "exp1_smoke_prompt_manifest.json"
    if prompt_manifest_path.exists():
        if json.loads(prompt_manifest_path.read_text(encoding="utf-8")) != prompt_manifest:
            raise RuntimeError("Existing prompt manifest differs; refusing to overwrite it")
    else:
        write_json(prompt_manifest_path, prompt_manifest)
    primary_checkpoint = OUTPUT / "exp1_smoke_primary_responses_checkpoint.jsonl"
    if primary_checkpoint.exists():
        primary_responses = [dict(row) for row in read_jsonl(primary_checkpoint)]
        expected_keys = {item["request_key"] for item in primary_items}
        observed_keys = {row.get("request_key") for row in primary_responses}
        if len(primary_responses) != len(primary_items) or observed_keys != expected_keys:
            raise RuntimeError("Existing primary checkpoint does not match the deterministic 16-request plan")
        primary_responses.sort(key=lambda row: row["request_order"])
    else:
        primary_responses = run_batch(primary_items, api_key)
        write_jsonl(primary_checkpoint, primary_responses)
    primary_by_key = {row["request_key"]: row for row in primary_responses}
    primary_qc = [response_qc(item, primary_by_key[item["request_key"]]) for item in primary_items]
    write_qc(unused_path(OUTPUT / "exp1_smoke_primary_qc_checkpoint.csv"), primary_qc)
    if not all(row["qc_pass"] for row in primary_qc):
        raise RuntimeError({
            "primary_qc": "FAILED",
            "failures": [{"case_id": row["case_id"], "language": row["input_language"], "failures": row["qc_failures"]} for row in primary_qc if not row["qc_pass"]],
        })

    effort_cases = [
        min((row for row in selected if row["origin_country"] == country), key=lambda row: stable_score((row["case_id"],), f"effort-{country}"))
        for country in ("KR", "US")
    ]
    effort_items = []
    for case in effort_cases:
        for language in ("ko", "en"):
            for effort in ("low", "medium", "high"):
                order += 1
                effort_items.append(request_record(case, facts[case["case_id"]], language, "effort_parameter_check", effort, order))
    effort_responses = run_batch(effort_items, api_key)
    effort_by_key = {row["request_key"]: row for row in effort_responses}
    effort_qc = [response_qc(item, effort_by_key[item["request_key"]]) for item in effort_items]

    all_inputs = primary_items + effort_items
    all_responses = primary_responses + effort_responses
    all_qc = primary_qc + effort_qc
    write_jsonl(OUTPUT / "exp1_smoke_inputs.jsonl", all_inputs)
    write_jsonl(OUTPUT / "exp1_smoke_responses.jsonl", all_responses)
    write_qc(OUTPUT / "exp1_smoke_qc.csv", all_qc)

    source_hashes_after = {str(CASES_PATH): sha_file(CASES_PATH), str(FACTS_PATH): sha_file(FACTS_PATH)}
    source_unchanged = source_hashes_before == source_hashes_after
    effort_transmission_pass = all(row["effort_parameter_transmitted_and_logged_pass"] for row in effort_qc)
    summary = {
        "experiment_id": "exp1-input-language-smoke-v4.0.1",
        "status": "PASS" if all(row["qc_pass"] for row in all_qc) and source_unchanged else "FAIL",
        "pipeline_ready_to_freeze_for_full_exp1": bool(all(row["qc_pass"] for row in all_qc) and source_unchanged),
        "hypothesis_evaluation_performed": False,
        "pca_performed": False,
        "model_requested": MODEL,
        "model_identifiers_returned": sorted({row.get("model_returned") for row in all_responses if row.get("model_returned")}),
        "prompt_version": PROMPT_VERSION,
        "system_prompt_versions": SYSTEM_VERSIONS,
        "user_prompt_versions": USER_VERSIONS,
        "generation_parameters": {
            "base_url_identifier": BASE_URL,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "seed": SEED,
            "replicate": REPLICATE,
            "primary_reasoning_effort": "medium",
            "concurrency": CONCURRENCY,
            "max_retries": MAX_RETRIES,
        },
        "selected_cases": [
            {"case_id": row["case_id"], "origin": row["origin_country"], "state": row.get("origin_state"), "primary_domain": row["primary_domain"]}
            for row in selected
        ],
        "primary": {
            "case_count": 8,
            "generation_count": len(primary_responses),
            "qc_pass": sum(row["qc_pass"] for row in primary_qc),
            "qc_fail": sum(not row["qc_pass"] for row in primary_qc),
        },
        "effort_parameter_check": {
            "case_ids": [row["case_id"] for row in effort_cases],
            "efforts": ["low", "medium", "high"],
            "generation_count": len(effort_responses),
            "qc_pass": sum(row["qc_pass"] for row in effort_qc),
            "qc_fail": sum(not row["qc_pass"] for row in effort_qc),
            "parameter_transmission_and_logging_pass": effort_transmission_pass,
        },
        "source_corpus_hashes_before": source_hashes_before,
        "source_corpus_hashes_after": source_hashes_after,
        "source_corpus_unchanged": source_unchanged,
        "prompt_pipeline_corrections": [
            {
                "change": "Narrowed the automatic U.S. reporter-citation QC regex so a date phrase such as '1996 or 1997' is not classified as legal authority.",
                "justification": "QC false-positive correction only; the frozen court-opinion prompts and all generated responses were unchanged.",
            }
        ],
    }
    write_json(OUTPUT / "exp1_smoke_summary.json", summary)

    artifacts = [
        OUTPUT / "exp1_smoke_inputs.jsonl", OUTPUT / "exp1_smoke_responses.jsonl",
        OUTPUT / "exp1_smoke_qc.csv", OUTPUT / "exp1_smoke_summary.json",
        OUTPUT / "exp1_smoke_prompt_manifest.json",
    ]
    report = [
        "# Exp 1 generation smoke test",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"- Model requested: `{MODEL}`",
        f"- Model identifiers returned: `{', '.join(summary['model_identifiers_returned'])}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        f"- Primary generations: {summary['primary']['qc_pass']}/16 QC pass",
        f"- Effort-parameter generations: {summary['effort_parameter_check']['qc_pass']}/12 QC pass",
        f"- Effort parameter transmitted/logged: **{str(effort_transmission_pass).upper()}**",
        f"- Frozen corpus unchanged: **{str(source_unchanged).upper()}**",
        f"- Ready to freeze pipeline for full Exp 1: **{str(summary['pipeline_ready_to_freeze_for_full_exp1']).upper()}**",
        "- Hypothesis/jurisdictional-marker evaluation: not performed",
        "- PCA: not performed",
        "",
        "## Selected cases",
        "",
    ]
    report.extend(
        f"- `{row['case_id']}` — {row['origin_country']}"
        + (f" / {row['origin_state']}" if row.get("origin_state") else "")
        + f" / {row['primary_domain']}"
        for row in selected
    )
    report += [
        "",
        "## Prompt/pipeline correction",
        "",
        "No prompt change was made. The repository's existing frozen `exp1-court-opinion-v1` prompt was used. "
        "The automatic U.S. reporter-citation regex was narrowed after it falsely classified the input-grounded date phrase "
        "`1996 or 1997` as authority. This QC-only correction did not alter prompts or responses and was unrelated to jurisdictional-marker strength.",
        "",
        "## Generation parameters",
        "",
        f"`temperature={TEMPERATURE}`, `top_p={TOP_P}`, `max_output_tokens={MAX_OUTPUT_TOKENS}`, `seed={SEED}`, "
        f"`replicate={REPLICATE}`, `primary_effort=medium`, `concurrency={CONCURRENCY}`.",
        "",
        "## Artifact SHA-256",
        "",
    ]
    report.extend(f"- `{path.as_posix()}`: `{sha_file(path)}`" for path in artifacts)
    report_path = OUTPUT / "EXP1_SMOKE_REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    checksum_paths = artifacts + [report_path]
    (OUTPUT / "SHA256SUMS_exp1_smoke.txt").write_text(
        "".join(f"{sha_file(path)}  {path.as_posix()}\n" for path in sorted(checksum_paths)), encoding="utf-8"
    )
    if summary["status"] != "PASS":
        raise RuntimeError(summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    if "--finalize-existing-run1" in sys.argv:
        finalize_existing_run1()
    elif "--finalize-blocked-attempt2" in sys.argv:
        finalize_blocked_attempt()
    else:
        main()
