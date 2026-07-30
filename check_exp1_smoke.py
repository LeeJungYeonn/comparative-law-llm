from __future__ import annotations

import argparse
import re
from pathlib import Path

from exp1.common import (
    DEFAULT_INPUT, assert_request_is_blind, load_prompts, read_jsonl, render_generation,
    request_payload, smoke_case_ids, usable_cases,
)

KO_SECTIONS = ["주요 쟁점", "적용 가능한 법리", "법리의 적용", "책임 인정 가능성", "손해배상"]
EN_SECTIONS = ["key issues", "applicable legal principles", "application", "potential liability", "recoverable damages"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--raw", type=Path, default=Path("outputs/exp1/raw_responses.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/exp1/run_manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/exp1_smoke_test.md"))
    parser.add_argument("--model", default="DRY_RUN_MODEL")
    args = parser.parse_args()
    rows = usable_cases(args.input)
    selected = set(smoke_case_ids(rows))
    selected_rows = {r["case_id"]: r for r in rows if r["case_id"] in selected}
    prompts = load_prompts()
    blind_ok = True
    for row in selected_rows.values():
        for condition in ("ko", "en"):
            system, user = render_generation(row, condition, prompts)
            body = request_payload(
                model=args.model, system_prompt=system, user_prompt=user, temperature=.2,
                top_p=1, max_output_tokens=3000, seed=20260730, reasoning_effort=None,
            )
            try:
                assert_request_is_blind(body)
            except AssertionError:
                blind_ok = False

    raw = []
    if args.raw.is_file():
        raw = [
            r for r in read_jsonl(args.raw)
            if r.get("case_id") in selected and r.get("replicate_id") == 1
        ]
    successes = [r for r in raw if r.get("raw_response") and not r.get("error")]
    expected_keys = {(case_id, condition, 1) for case_id in selected for condition in ("ko", "en")}
    actual_keys = {(r["case_id"], r["condition"], r["replicate_id"]) for r in successes}
    section_pass = truncation_pass = placeholder_pass = refusal_pass = metadata_pass = False
    details: list[str] = []
    if successes:
        section_pass = all(
            sum(term in r["raw_response"].casefold() for term in (KO_SECTIONS if r["condition"] == "ko" else EN_SECTIONS)) >= 4
            for r in successes
        )
        truncation_pass = all(r.get("finish_status") not in {"length", "max_tokens"} for r in successes)
        placeholder_checks = []
        for record in successes:
            fact = selected_rows[record["case_id"]][f"neutral_fact_{record['condition']}"]
            placeholders = set(re.findall(r"\[[^\]\n]{1,80}\]", fact))
            retained = sum(token in record["raw_response"] for token in placeholders)
            placeholder_checks.append(not placeholders or retained / len(placeholders) >= .5)
        placeholder_pass = all(placeholder_checks)
        refusal_pattern = re.compile(r"(cannot assist|can.?t assist|follow.?up question|추가 질문|답변할 수 없)", re.I)
        refusal_pass = all(not refusal_pattern.search(r["raw_response"]) for r in successes)
        required = {
            "experiment_id", "case_id", "case_origin", "case_subtype", "condition",
            "input_language", "replicate_id", "model_requested", "model_returned",
            "temperature", "top_p", "max_output_tokens", "seed", "prompt_version",
            "system_prompt_sha256", "user_prompt_template_sha256",
            "fact_text_sha256", "request_order", "timestamp", "latency_seconds",
            "token_usage", "finish_status", "retry_count", "raw_response", "error",
        }
        metadata_pass = all(required <= r.keys() for r in successes)
        details.append(f"Successful records: {len(successes)}/12")
        first = successes[0]
        details.append(
            "Generation parameters: "
            f"model_requested={first.get('model_requested')}, "
            f"model_returned={first.get('model_returned')}, "
            f"temperature={first.get('temperature')}, top_p={first.get('top_p')}, "
            f"max_output_tokens={first.get('max_output_tokens')}, "
            f"reasoning_effort={first.get('reasoning_effort')}, seed_base=20260730."
        )
        evaluation_path = args.raw.parent / "evaluations.jsonl"
        if evaluation_path.is_file():
            evaluations = read_jsonl(evaluation_path)
            details.append(
                f"Optional evaluator smoke: {len(evaluations)} schema-valid evaluations; "
                f"{sum(record.get('schema_retry_count', 0) for record in evaluations)} schema retries."
            )
    else:
        details.append("No real API response records were available; response-content checks were not run.")

    duplicate_free = len(actual_keys) == len(successes)
    resume_verified = False
    if args.manifest.is_file():
        import json
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        execution = manifest.get("execution", {})
        resume_verified = execution.get("resume_skipped", 0) >= 12 and execution.get("actual_api_calls", 1) == 0
    status = "PASS" if (
        actual_keys == expected_keys and section_pass and truncation_pass and placeholder_pass
        and refusal_pass and metadata_pass and duplicate_free and blind_ok and resume_verified
    ) else ("NOT EXECUTED" if not successes else "REVIEW REQUIRED")
    checks = [
        ("12 successful calls (3 KR + 3 CA × 2)", actual_keys == expected_keys),
        ("Five requested sections", section_pass),
        ("No truncation", truncation_pass),
        ("Placeholder retention", placeholder_pass),
        ("No refusal/follow-up request", refusal_pass),
        ("Raw response and metadata saved", metadata_pass),
        ("No duplicate unique request", duplicate_free),
        ("Blind request excludes case metadata", blind_ok),
        ("Second --resume run made zero calls", resume_verified),
    ]
    report = [
        "# Experiment 1 smoke test", "", f"Status: **{status}**", "",
        "Selection is deterministic and subtype-overlap-first: " + ", ".join(sorted(selected)), "",
        "| Check | Result |", "|---|---|",
    ]
    report.extend(f"| {name} | {'PASS' if passed else 'NOT VERIFIED'} |" for name, passed in checks)
    report.extend(["", *details, ""])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(f"smoke={status} successful={len(successes)} blind_payload={blind_ok}")


if __name__ == "__main__":
    main()
