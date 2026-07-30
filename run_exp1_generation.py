from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from audit_exp1 import audit
from exp1 import EXPERIMENT_ID, GENERATION_PROMPT_VERSION
from exp1.common import (
    DEFAULT_INPUT, append_jsonl, assert_request_is_blind, build_manifest,
    generation_record_base, load_prompts, now_iso, parse_case_ids, post_chat,
    read_jsonl, render_generation, request_payload, response_content, select_cases,
    sha256_text, stable_json, unique_key, usable_cases, write_json, write_jsonl,
)
from exp1.common import REPO_ROOT, load_env_file


def build_plan(
    rows: list[dict[str, Any]], repetitions: int, model: str, seed: int,
) -> list[dict[str, Any]]:
    plan = []
    for row in rows:
        for replicate_id in range(1, repetitions + 1):
            replicate_seed = seed + replicate_id - 1
            for condition in ("ko", "en"):
                plan.append({
                    "row": row,
                    "condition": condition,
                    "replicate_id": replicate_id,
                    "seed": replicate_seed,
                    "unique_key": unique_key(
                        row["case_id"], condition, replicate_id, model, GENERATION_PROMPT_VERSION,
                    ),
                })
    random.Random(seed).shuffle(plan)
    for index, request in enumerate(plan, 1):
        request["request_order"] = index
    return plan


def completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        row["unique_key"] for row in read_jsonl(path)
        if row.get("unique_key") and row.get("error") is None and row.get("raw_response") is not None
    }


def public_plan(plan: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    return [{
        "case_id": item["row"]["case_id"],
        "condition": item["condition"],
        "replicate_id": item["replicate_id"],
        "seed": item["seed"],
        "request_order": item["request_order"],
        "model": model,
        "prompt_version": GENERATION_PROMPT_VERSION,
        "unique_key": item["unique_key"],
    } for item in plan]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 paired raw-response generation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--model", help="Exact model snapshot identifier; required for generation.")
    parser.add_argument("--base-url", default="https://gw.letsur.ai/v1")
    parser.add_argument("--api-key-env", default="LETSUR_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Select 3 KR + 3 CA cases; use repetitions=1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.smoke_test and args.repetitions != 1:
        raise SystemExit("--smoke-test requires --repetitions 1")
    preflight = audit(args.input, args.output_dir)
    if args.preflight_only:
        print(f"preflight={preflight['status']} api_calls=0")
        return
    if not args.model:
        raise SystemExit("--model with an exact snapshot identifier is required")
    load_env_file(args.env_file)

    rows = select_cases(
        usable_cases(args.input), parse_case_ids(args.case_ids), args.limit, args.smoke_test,
    )
    if not rows:
        raise SystemExit("No cases selected")
    plan = build_plan(rows, args.repetitions, args.model, args.seed)
    output_dir = args.output_dir
    raw_path = output_dir / "raw_responses.jsonl"
    failed_path = output_dir / "failed_requests.jsonl"
    if raw_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"{raw_path} exists; pass --resume to avoid duplicate calls")

    parameters = {
        "model": args.model,
        "base_url_identifier": args.base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "request_order_seed": args.seed,
        "repetitions": args.repetitions,
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "max_retries": args.max_retries,
        "concurrency": args.concurrency,
        "selected_case_count": len(rows),
        "planned_request_count": len(plan),
        "smoke_test": args.smoke_test,
        "dry_run": args.dry_run,
    }
    started_at = now_iso()
    manifest = build_manifest(args.input, parameters, started_at)
    write_json(output_dir / "config.json", parameters)
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "request_plan.json", public_plan(plan, args.model))

    prompts = load_prompts()
    already = completed_keys(raw_path) if args.resume else set()
    if args.resume and raw_path.is_file():
        existing = {
            record["unique_key"]: record for record in read_jsonl(raw_path)
            if record.get("unique_key") in already
        }
        for item in plan:
            record = existing.get(item["unique_key"])
            if not record:
                continue
            expected = {
                "model_requested": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_output_tokens": args.max_output_tokens,
                "reasoning_effort": args.reasoning_effort,
                "seed": item["seed"],
                "prompt_version": GENERATION_PROMPT_VERSION,
            }
            mismatched = {
                key: (record.get(key), value)
                for key, value in expected.items() if record.get(key) != value
            }
            if mismatched:
                raise SystemExit(
                    f"Resume parameter mismatch for {record.get('case_id')}/{record.get('condition')}: {mismatched}"
                )
    calls = failures = retries = skipped = 0
    pending: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for item in plan:
        if item["unique_key"] in already:
            skipped += 1
            continue
        row, condition = item["row"], item["condition"]
        system_prompt, user_prompt = render_generation(row, condition, prompts)
        body = request_payload(
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            seed=item["seed"],
            reasoning_effort=args.reasoning_effort,
        )
        assert_request_is_blind(body)
        if args.dry_run:
            continue
        base = generation_record_base(
            row, condition, item["replicate_id"], args.model, args.temperature,
            args.top_p, args.max_output_tokens, args.reasoning_effort,
            item["seed"], item["request_order"],
        )
        pending.append((item, body, base))

    def execute_request(
        prepared: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ) -> tuple[bool, dict[str, Any], int]:
        _, body, base = prepared
        try:
            envelope, retry_count, latency = post_chat(
                body=body, base_url=args.base_url, api_key_env=args.api_key_env,
                max_retries=args.max_retries,
            )
            choice = envelope.get("choices", [{}])[0]
            record = {
                **base,
                "model_returned": envelope.get("model"),
                "timestamp": now_iso(),
                "latency_seconds": round(latency, 6),
                "token_usage": envelope.get("usage") or {},
                "finish_status": choice.get("finish_reason"),
                "retry_count": retry_count,
                "raw_response": response_content(envelope),
                "error": None,
                "response_envelope_sha256": sha256_text(stable_json(envelope)),
            }
            return True, record, retry_count
        except Exception as exc:
            retry_count = int(getattr(exc, "retry_count", 0))
            record = {
                **base,
                "timestamp": now_iso(),
                "latency_seconds": round(float(getattr(exc, "latency_seconds", 0.0)), 6),
                "finish_status": "error",
                "retry_count": retry_count,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return False, record, retry_count

    if not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(execute_request, prepared) for prepared in pending]
            for future in as_completed(futures):
                succeeded, record, retry_count = future.result()
                calls += 1
                retries += retry_count
                if succeeded:
                    append_jsonl(raw_path, record)
                else:
                    failures += 1
                    append_jsonl(failed_path, record)

    cumulative_records = read_jsonl(raw_path) if raw_path.is_file() else []
    if not failed_path.exists():
        write_jsonl(failed_path, [])
    manifest.update({
        "completed_at": now_iso(),
        "execution": {
            "logical_requests_attempted": calls,
            "actual_api_calls": calls + retries,
            "successful_calls": calls - failures,
            "failed_calls": failures,
            "retry_count": retries,
            "resume_skipped": skipped,
            "cumulative_successful_records": (
                len(cumulative_records)
            ),
            "cumulative_recorded_api_calls": (
                sum(1 + int(record.get("retry_count", 0)) for record in cumulative_records)
            ),
            "first_response_timestamp": min(
                (record["timestamp"] for record in cumulative_records if record.get("timestamp")),
                default=None,
            ),
            "last_response_timestamp": max(
                (record["timestamp"] for record in cumulative_records if record.get("timestamp")),
                default=None,
            ),
        },
    })
    write_json(output_dir / "run_manifest.json", manifest)
    status = "dry_run" if args.dry_run else "completed"
    print(f"status={status} cases={len(rows)} planned={len(plan)} calls={calls} failures={failures} retries={retries} skipped={skipped}")


if __name__ == "__main__":
    main()
