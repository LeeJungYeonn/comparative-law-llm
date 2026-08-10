from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from exp1.common import (
    REPO_ROOT, append_jsonl, assert_request_is_blind, load_env_file, now_iso,
    post_chat, read_jsonl, request_payload, response_content, sha256_file,
    sha256_text, stable_json, write_json, write_jsonl,
)
from exp1.conclusion_v2 import (
    PROMPT_VERSION, SCHEMA_PATH, SYSTEM_PROMPT_PATH, USER_PROMPT_PATH, VERSION,
    dynamic_schema, recode_cache_key, render_recode_prompt, response_id,
    validate_flat_record, validate_recode_payload,
)
from evaluate_exp1 import parse_json_content


def read_registry(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("ko_present", "en_present", "source_party_set_mismatch", "unresolved_source_issue"):
            row[key] = str(row[key]).casefold() == "true"
    return rows


def completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        record["recode_cache_key"] for record in read_jsonl(path)
        if record.get("recode_cache_key") and record.get("evaluation")
    }


def materialize_flat(response_path: Path, flat_path: Path) -> int:
    flattened: list[dict[str, Any]] = []
    if response_path.is_file():
        for record in read_jsonl(response_path):
            for party in record["evaluation"]["parties"]:
                flat = {
                    "version": VERSION,
                    "case_id": record["case_id"],
                    "case_origin": record["case_origin"],
                    "case_subtype": record["case_subtype"],
                    "response_id": record["response_id"],
                    "response_unique_key": record["response_unique_key"],
                    "language": record["language"],
                    "replicate_id": record["replicate_id"],
                    "canonical_party_id": party["canonical_party_id"],
                    "conclusion": party["conclusion"],
                    "assessed": party["assessed"],
                    "supporting_text": party["supporting_text"],
                    "aggregation_note": party["aggregation_note"],
                    "evaluator_model": record["evaluator_model_returned"],
                    "evaluator_prompt_version": PROMPT_VERSION,
                    "recode_cache_key": record["recode_cache_key"],
                }
                validate_flat_record(
                    flat,
                    expected_case_id=record["case_id"],
                    expected_response_id=record["response_id"],
                    expected_language=record["language"],
                    expected_replicate=record["replicate_id"],
                )
                flattened.append(flat)
    flattened.sort(key=lambda row: (
        row["case_id"], row["language"], int(row["replicate_id"]), row["canonical_party_id"],
    ))
    write_jsonl(flat_path, flattened)
    return len(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind canonical-party conclusion recoding v2.")
    parser.add_argument("--raw-input", type=Path, default=Path("outputs/exp1/raw_responses.jsonl"))
    parser.add_argument("--accepted-input", type=Path, default=Path("outputs/neutral/stage2-paired-qc-v1/accepted_pairs.jsonl"))
    parser.add_argument("--party-registry", type=Path, default=Path("outputs/exp1/party_registry_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://gw.letsur.ai/v1")
    parser.add_argument("--api-key-env", default="LETSUR_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--response-ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    load_env_file(args.env_file)
    raw_records = read_jsonl(args.raw_input)
    if args.response_ids:
        selected = set(args.response_ids)
        raw_records = [
            record for record in raw_records
            if response_id(record) in selected or record["unique_key"] in selected
        ]
    if args.limit is not None:
        raw_records = raw_records[:args.limit]
    accepted = {row["case_id"]: row for row in read_jsonl(args.accepted_input)}
    registry = read_registry(args.party_registry)
    parties_by_case: dict[str, list[str]] = defaultdict(list)
    for row in registry:
        parties_by_case[row["case_id"]].append(row["canonical_party_id"])
    parties_by_case = {
        case_id: sorted(set(parties)) for case_id, parties in parties_by_case.items()
    }

    response_path = args.output_dir / "conclusion_recode_responses_v2.jsonl"
    flat_path = args.output_dir / "party_conclusions_response_v2.jsonl"
    attempt_path = args.output_dir / "conclusion_evaluator_attempts_v2.jsonl"
    failed_path = args.output_dir / "conclusion_failed_evaluations_v2.jsonl"
    if response_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"{response_path} exists; pass --resume")
    done = completed_keys(response_path) if args.resume else set()
    pending: list[tuple[dict[str, Any], dict[str, Any], str, list[str], str]] = []
    skipped = 0
    for raw in raw_records:
        case_id, language = raw["case_id"], raw["condition"]
        if case_id not in accepted or case_id not in parties_by_case:
            raise ValueError(f"Missing accepted fact or party registry for {case_id}")
        fact = str(accepted[case_id][f"neutral_fact_{language}"])
        parties = parties_by_case[case_id]
        cache_key = recode_cache_key(raw, fact, parties, args.model)
        if cache_key in done:
            skipped += 1
            continue
        system, user = render_recode_prompt(raw, fact, parties)
        schema = dynamic_schema(response_id(raw), language, parties)
        body = request_payload(
            model=args.model,
            system_prompt=system,
            user_prompt=user,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "conclusion_recode_v2", "strict": True, "schema": schema},
            },
        )
        assert_request_is_blind(body)
        pending.append((raw, body, cache_key, parties, sha256_text(stable_json(schema))))

    def evaluate_one(
        prepared: tuple[dict[str, Any], dict[str, Any], str, list[str], str],
    ) -> tuple[bool, dict[str, Any], int, int, int, list[dict[str, Any]]]:
        raw, body, cache_key, parties, dynamic_schema_hash = prepared
        local_calls = local_transport_retries = 0
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for schema_attempt in range(args.schema_retries + 1):
            try:
                envelope, retry_count, latency = post_chat(
                    body=body,
                    base_url=args.base_url,
                    api_key_env=args.api_key_env,
                    max_retries=args.max_retries,
                )
            except RuntimeError as exc:
                last_error = exc
                if hasattr(exc, "retry_count"):
                    local_calls += 1
                    local_transport_retries += int(getattr(exc, "retry_count"))
                break
            local_calls += 1
            local_transport_retries += retry_count
            raw_evaluator_response = response_content(envelope)
            attempt = {
                "version": VERSION,
                "recode_cache_key": cache_key,
                "response_id": response_id(raw),
                "case_id": raw["case_id"],
                "language": raw["condition"],
                "replicate_id": raw["replicate_id"],
                "schema_attempt": schema_attempt,
                "timestamp": now_iso(),
                "model_returned": envelope.get("model"),
                "token_usage": envelope.get("usage") or {},
                "raw_evaluator_response": raw_evaluator_response,
                "validation_status": None,
                "validation_error": None,
            }
            try:
                evaluation = parse_json_content(raw_evaluator_response)
                validate_recode_payload(
                    evaluation,
                    expected_response_id=response_id(raw),
                    expected_language=raw["condition"],
                    expected_parties=parties,
                )
                attempt["validation_status"] = "valid"
                attempts.append(attempt)
                record = {
                    "version": VERSION,
                    "recode_cache_key": cache_key,
                    "response_id": response_id(raw),
                    "response_unique_key": raw["unique_key"],
                    "case_id": raw["case_id"],
                    "case_origin": raw["case_origin"],
                    "case_subtype": raw["case_subtype"],
                    "language": raw["condition"],
                    "replicate_id": raw["replicate_id"],
                    "raw_response_sha256": sha256_text(raw["raw_response"]),
                    "fact_text_sha256": raw["fact_text_sha256"],
                    "canonical_parties": parties,
                    "evaluator_model_requested": args.model,
                    "evaluator_model_returned": envelope.get("model"),
                    "evaluator_prompt_version": PROMPT_VERSION,
                    "system_prompt_sha256": sha256_file(SYSTEM_PROMPT_PATH),
                    "user_prompt_sha256": sha256_file(USER_PROMPT_PATH),
                    "base_schema_sha256": sha256_file(SCHEMA_PATH),
                    "dynamic_schema_sha256": dynamic_schema_hash,
                    "timestamp": now_iso(),
                    "latency_seconds": round(latency, 6),
                    "token_usage": envelope.get("usage") or {},
                    "schema_retry_count": schema_attempt,
                    "evaluation": evaluation,
                    "error": None,
                }
                return True, record, local_calls, local_transport_retries, schema_attempt, attempts
            except Exception as exc:
                last_error = exc
                attempt["validation_status"] = "invalid"
                attempt["validation_error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(attempt)
        failed = {
            "version": VERSION,
            "recode_cache_key": cache_key,
            "response_id": response_id(raw),
            "response_unique_key": raw["unique_key"],
            "case_id": raw["case_id"],
            "language": raw["condition"],
            "replicate_id": raw["replicate_id"],
            "timestamp": now_iso(),
            "error": f"{type(last_error).__name__}: {last_error}",
        }
        return False, failed, local_calls, local_transport_retries, args.schema_retries, attempts

    calls = failures = schema_retries = transport_retries = 0
    if not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(evaluate_one, item) for item in pending]
            for future in as_completed(futures):
                succeeded, record, local_calls, local_transport, schema_attempt, attempts = future.result()
                calls += local_calls
                transport_retries += local_transport
                for attempt in attempts:
                    append_jsonl(attempt_path, attempt)
                if succeeded:
                    schema_retries += schema_attempt
                    append_jsonl(response_path, record)
                else:
                    failures += 1
                    append_jsonl(failed_path, record)
        flat_rows = materialize_flat(response_path, flat_path)
    else:
        flat_rows = 0
    if not failed_path.exists() and not args.dry_run:
        write_jsonl(failed_path, [])
    cumulative_responses = read_jsonl(response_path) if response_path.is_file() else []
    cumulative_attempts = read_jsonl(attempt_path) if attempt_path.is_file() else []
    invalid_reasons = Counter(
        (attempt.get("validation_error") or "").split(":", 1)[0]
        for attempt in cumulative_attempts
        if attempt.get("validation_status") == "invalid"
    )
    summary = {
        "version": VERSION,
        "model": args.model,
        "parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
            "seed": args.seed,
            "concurrency": args.concurrency,
            "schema_retries": args.schema_retries,
        },
        "selected_responses": len(raw_records),
        "pending_responses": len(pending),
        "resume_skipped": skipped,
        "this_run_api_calls": calls + transport_retries,
        "this_run_transport_retries": transport_retries,
        "this_run_schema_retries": schema_retries,
        "this_run_final_failures": failures,
        "cumulative_successful_responses": len(cumulative_responses),
        "cumulative_api_response_attempts": len(cumulative_attempts),
        "cumulative_schema_invalid_attempts": sum(
            attempt.get("validation_status") == "invalid" for attempt in cumulative_attempts
        ),
        "cumulative_invalid_reason_types": dict(invalid_reasons),
        "flat_party_rows": flat_rows,
        "generation_api_calls": 0,
        "dry_run": args.dry_run,
    }
    write_json(args.output_dir / "conclusion_recode_run_summary_v2.json", summary)
    print(
        f"responses={len(raw_records)} pending={len(pending)} skipped={skipped} "
        f"api_calls={summary['this_run_api_calls']} failures={failures} "
        f"cumulative_success={len(cumulative_responses)} flat_rows={flat_rows}"
    )


if __name__ == "__main__":
    main()
