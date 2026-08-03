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
    directory_hashes, generation_record_base, load_prompts, now_iso, parse_case_ids, post_chat,
    read_jsonl, render_generation, request_payload, response_content, select_cases,
    sha256_file, sha256_text, stable_json, unique_key, usable_cases, write_json, write_jsonl,
)
from exp1.common import REPO_ROOT, load_env_file
from exp1.design import (
    EXP1_GENERATION_PROMPT_VERSION, EXP2_EXPERIMENT_ID,
    JURISDICTION_INSTRUCTION_VERSION, experiment_values, jurisdiction_metadata,
)


def build_plan(
    rows: list[dict[str, Any]], repetitions: int, model: str, seed: int,
    experiment_id: str = EXPERIMENT_ID,
    prompt_version: str = GENERATION_PROMPT_VERSION,
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
                        row["case_id"], condition, replicate_id, model, prompt_version,
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


def public_plan(
    plan: list[dict[str, Any]], model: str, experiment_id: str = EXPERIMENT_ID,
    prompt_version: str = GENERATION_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    result = []
    for item in plan:
        row = {
        "experiment_id": experiment_id,
        "case_id": item["row"]["case_id"],
        "case_origin": item["row"]["case_origin"],
        "condition": item["condition"],
        "input_language": item["condition"],
        "replicate_id": item["replicate_id"],
        "replicate_number": item["replicate_id"],
        "seed": item["seed"],
        "request_order": item["request_order"],
        "model": model,
        "prompt_version": prompt_version,
        "unique_key": item["unique_key"],
        }
        if experiment_id == EXP2_EXPERIMENT_ID:
            row.update(jurisdiction_metadata(item["row"], item["condition"]))
        result.append(row)
    return result


def parse_args(default_experiment: str = "exp1") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared Exp 1/Exp 2 paired raw-response generator.")
    parser.add_argument("--experiment", choices=("exp1", "exp2"), default=default_experiment)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exp1-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--model", help="Exact model snapshot identifier; required for generation.")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="LETSUR_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Select 3 KR + 3 CA cases; use repetitions=1.")
    return parser.parse_args()


def _configure_args(args: argparse.Namespace) -> dict[str, Any] | None:
    legacy_defaults = {
        "base_url": "https://gw.letsur.ai/v1", "temperature": 1.0, "top_p": 1.0,
        "max_output_tokens": 8000, "reasoning_effort": None, "seed": 20260730,
        "repetitions": 3, "max_retries": 5, "concurrency": 1,
    }
    baseline: dict[str, Any] | None = None
    if args.experiment == "exp2":
        config_path = args.exp1_dir / "config.json"
        manifest_path = args.exp1_dir / "run_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            raise SystemExit("Exp 2 requires the completed Exp 1 config.json and run_manifest.json")
        baseline = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_sha256") != sha256_file(args.input):
            raise SystemExit("Exp 2 input differs from the immutable Exp 1 input")
        defaults = {
            "base_url": baseline["base_url_identifier"], "temperature": baseline["temperature"],
            "top_p": baseline["top_p"], "max_output_tokens": baseline["max_output_tokens"],
            "reasoning_effort": baseline.get("reasoning_effort"), "seed": baseline["seed"],
            "repetitions": baseline["repetitions"], "max_retries": baseline["max_retries"],
            "concurrency": baseline["concurrency"],
        }
        args.model = args.model or baseline["model"]
    else:
        defaults = legacy_defaults
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.output_dir = args.output_dir or Path(f"outputs/{args.experiment}")
    if args.experiment == "exp2":
        checked = {
            "model": args.model, "base_url_identifier": args.base_url,
            "temperature": args.temperature, "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens, "reasoning_effort": args.reasoning_effort,
            "seed": args.seed, "max_retries": args.max_retries, "concurrency": args.concurrency,
        }
        mismatches = {key: (value, baseline.get(key)) for key, value in checked.items() if value != baseline.get(key)}
        if not args.smoke_test and args.repetitions != baseline.get("repetitions"):
            mismatches["repetitions"] = (args.repetitions, baseline.get("repetitions"))
        if mismatches:
            raise SystemExit(f"Exp 2 settings must match Exp 1: {mismatches}")
        exp1_resolved = args.exp1_dir.resolve()
        output_resolved = args.output_dir.resolve()
        if output_resolved == exp1_resolved or exp1_resolved in output_resolved.parents:
            raise SystemExit("Exp 2 output directory must be separate from outputs/exp1")
    return baseline


def main(default_experiment: str = "exp1") -> None:
    args = parse_args(default_experiment)
    baseline = _configure_args(args)
    experiment_id, prompt_version = experiment_values(args.experiment)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.smoke_test and args.repetitions != 1:
        raise SystemExit("--smoke-test requires --repetitions 1")
    exp1_baseline_hashes = directory_hashes(args.exp1_dir) if args.experiment == "exp2" else None
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
    plan = build_plan(rows, args.repetitions, args.model, args.seed, experiment_id, prompt_version)
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
        "generation_prompt_version": prompt_version,
        "max_retries": args.max_retries,
        "concurrency": args.concurrency,
        "selected_case_count": len(rows),
        "planned_request_count": len(plan),
        "smoke_test": args.smoke_test,
        "dry_run": args.dry_run,
    }
    if args.experiment == "exp2":
        parameters.update({
            "base_generation_prompt_version": EXP1_GENERATION_PROMPT_VERSION,
            "jurisdiction_instruction_version": JURISDICTION_INSTRUCTION_VERSION,
            "exp1_config_sha256": sha256_file(args.exp1_dir / "config.json"),
            "raw_record_schema_sha256": sha256_file(REPO_ROOT / "schemas/exp2_raw_response_v1.schema.json"),
        })
    started_at = now_iso()
    manifest = build_manifest(args.input, parameters, started_at, experiment_id)
    if args.experiment == "exp2":
        manifest.update({
            "exp1_output_dir": str(args.exp1_dir.resolve()),
            "exp1_baseline_file_hashes": exp1_baseline_hashes,
            "exp1_reference_parameters": baseline,
            "prompt_delta_policy": "jurisdiction_instruction + two newlines + unchanged Exp 1 user prompt",
        })
    write_json(output_dir / "config.json", parameters)
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "request_plan.json", public_plan(plan, args.model, experiment_id, prompt_version))

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
                "prompt_version": prompt_version,
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
        system_prompt, user_prompt = render_generation(row, condition, prompts, experiment_id)
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
            experiment_id, prompt_version,
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
