from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jsonschema

from exp1 import EVALUATOR_PROMPT_VERSION
from exp1.common import (
    ONTOLOGY_PATH, REPO_ROOT, SCHEMA_PATH, append_jsonl, assert_request_is_blind,
    load_env_file, load_prompts,
    now_iso, post_chat, read_jsonl, request_payload, response_content, sha256_file,
    sha256_text, stable_json, write_csv, write_json, write_jsonl,
)

REASONING_LABELS = [
    "issue_identification", "governing_rule", "duty_or_protected_interest",
    "breach_or_wrongfulness", "fault_or_intent", "factual_causation",
    "legal_or_proximate_causation", "injury_or_damage", "plaintiff_fault_or_defense",
    "vicarious_or_organizational_liability", "multiple_tortfeasors", "damages_scope",
    "evidentiary_uncertainty", "procedural_reasoning", "policy_reasoning", "conclusion", "other",
]
DAMAGE_IDS = [
    "medical_expenses", "lost_earnings", "future_economic_loss", "property_damage",
    "pain_and_suffering_or_nonpecuniary", "emotional_distress", "wrongful_death_related",
    "punitive_damages", "mitigation", "comparative_reduction", "amount_or_proof_uncertainty",
]


def parse_json_content(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Evaluator output must be one JSON object")
    return value


def evaluation_key(raw: dict[str, Any], evaluator_model: str) -> str:
    return sha256_text(stable_json([
        raw["unique_key"], sha256_text(raw["raw_response"]), evaluator_model, EVALUATOR_PROMPT_VERSION,
        sha256_file(SCHEMA_PATH), sha256_file(ONTOLOGY_PATH),
    ]))


def completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {row["evaluation_key"] for row in read_jsonl(path) if row.get("evaluation_key") and row.get("evaluation")}


def write_derived(output_dir: Path) -> None:
    evaluation_path = output_dir / "evaluations.jsonl"
    if not evaluation_path.is_file():
        return
    evaluations = [r for r in read_jsonl(evaluation_path) if r.get("evaluation")]
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    concept_meta = {c["concept_id"]: c for c in ontology["concepts"]}
    reasoning_path = output_dir / "reasoning_units.jsonl"
    feature_rows: list[dict[str, Any]] = []
    reasoning_rows: list[dict[str, Any]] = []
    all_concept_ids = sorted(concept_meta)
    for record in evaluations:
        evaluation = record["evaluation"]
        units = evaluation["reasoning_units"]
        label_counts: dict[str, int] = {}
        for unit in units:
            for label in unit["labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1
            reasoning_rows.append({
                "experiment_id": record["experiment_id"],
                "response_unique_key": record["response_unique_key"],
                "case_id": record["case_id"],
                "case_origin": record["case_origin"],
                "condition": record["condition"],
                "replicate_id": record["replicate_id"],
                "condition_id": record.get("condition_id"),
                "target_jurisdiction": record.get("target_jurisdiction"),
                **unit,
            })
        detected = {c["concept_id"]: c for c in evaluation["concepts"] if c["present"]}
        row: dict[str, Any] = {
            "response_unique_key": record["response_unique_key"],
            "case_id": record["case_id"],
            "case_origin": record["case_origin"],
            "condition": record["condition"],
            "replicate_id": record["replicate_id"],
            "experiment_id": record["experiment_id"],
            "condition_id": record.get("condition_id"),
            "target_jurisdiction": record.get("target_jurisdiction"),
            "output_chars": len(record.get("raw_response", "")),
            "reasoning_unit_count": len(units),
            "average_labels_per_unit": (
                sum(len(u["labels"]) for u in units) / len(units) if units else 0.0
            ),
            "kr_marker_count": sum(
                1 for cid in detected if concept_meta.get(cid, {}).get("system") == "KR"
            ),
            "us_marker_count": sum(
                1 for cid in detected if concept_meta.get(cid, {}).get("system") == "US_COMMON_LAW"
            ),
            "strong_a_marker_count": sum(c["marker_strength"] == "A" for c in detected.values()),
            "explicit_jurisdiction": evaluation["jurisdiction_signals"]["explicit_jurisdiction"],
            "explicit_statute_reference": evaluation["jurisdiction_signals"]["explicit_statute_reference"],
            "explicit_precedent_reference": evaluation["jurisdiction_signals"]["explicit_precedent_reference"],
            "hallucinated_authority": evaluation["jurisdiction_signals"]["unsupported_or_hallucinated_authority"],
            "evaluator_confidence": evaluation["evaluator_confidence"],
        }
        raw_casefold = record.get("raw_response", "").casefold()
        for cid in all_concept_ids:
            row[f"concept__{cid}"] = int(cid in detected)
            labels = concept_meta[cid].get("labels", [])
            row[f"string__{cid}"] = sum(raw_casefold.count(label.casefold()) for label in labels)
        damage_present = {d["damage_id"] for d in evaluation["damages"] if d["present"]}
        for damage_id in DAMAGE_IDS:
            row[f"damage__{damage_id}"] = int(damage_id in damage_present)
        for label in REASONING_LABELS:
            count = label_counts.get(label, 0)
            row[f"reasoning_count__{label}"] = count
            row[f"reasoning_prop__{label}"] = count / len(units) if units else 0.0
        feature_rows.append(row)
    if feature_rows:
        fields = list(dict.fromkeys(key for row in feature_rows for key in row))
        write_csv(output_dir / "concept_features.csv", feature_rows, fields)
    write_jsonl(reasoning_path, reasoning_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blindly evaluate Exp 1 or Exp 2 responses one at a time.")
    parser.add_argument("--input", type=Path, default=Path("outputs/exp1/raw_responses.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--model", required=True, help="Exact evaluator model snapshot identifier.")
    parser.add_argument("--base-url", default="https://gw.letsur.ai/v1")
    parser.add_argument("--api-key-env", default="LETSUR_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    load_env_file(args.env_file)
    raw_rows = [r for r in read_jsonl(args.input) if r.get("raw_response") and not r.get("error")]
    if args.limit is not None:
        raw_rows = raw_rows[:args.limit]
    experiment_ids = {r.get("experiment_id") for r in raw_rows}
    if len(experiment_ids) != 1 or None in experiment_ids:
        raise SystemExit(f"Input must contain exactly one experiment_id, found {sorted(map(str, experiment_ids))}")
    experiment_id = str(next(iter(experiment_ids)))
    output_path = args.output_dir / "evaluations.jsonl"
    failed_path = args.output_dir / "failed_evaluations.jsonl"
    attempt_path = args.output_dir / "evaluator_raw_attempts.jsonl"
    if output_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"{output_path} exists; pass --resume")
    done = completed_keys(output_path) if args.resume else set()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    allowed_concept_ids = {concept["concept_id"] for concept in ontology["concepts"]}
    prompts = load_prompts()
    ontology_text = ONTOLOGY_PATH.read_text(encoding="utf-8")
    system = prompts["evaluator_v1_system.txt"] + "\n\nAllowed ontology:\n" + ontology_text
    calls = failures = schema_retry_total = transport_retries = skipped = 0
    pending: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for raw in raw_rows:
        key = evaluation_key(raw, args.model)
        if key in done:
            skipped += 1
            continue
        user = prompts["evaluator_v1_user.txt"].replace("{raw_response}", raw["raw_response"])
        body = request_payload(
            model=args.model, system_prompt=system, user_prompt=user,
            temperature=args.temperature, top_p=args.top_p, max_output_tokens=args.max_output_tokens,
            seed=args.seed, reasoning_effort=args.reasoning_effort,
            response_format={"type": "json_schema", "json_schema": {
                "name": "exp1_evaluation", "strict": True, "schema": schema,
            }},
        )
        assert_request_is_blind(body)
        if args.dry_run:
            continue
        pending.append((raw, key, body))

    def evaluate_one(
        prepared: tuple[dict[str, Any], str, dict[str, Any]],
    ) -> tuple[bool, dict[str, Any], int, int, int, list[dict[str, Any]]]:
        raw, key, body = prepared
        last_error: Exception | None = None
        local_calls = local_transport_retries = 0
        attempt_records: list[dict[str, Any]] = []
        for schema_attempt in range(args.schema_retries + 1):
            try:
                envelope, retry_count, latency = post_chat(
                    body=body, base_url=args.base_url, api_key_env=args.api_key_env,
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
            attempt_record = {
                "experiment_id": experiment_id,
                "evaluation_key": key,
                "response_unique_key": raw["unique_key"],
                "case_id": raw["case_id"],
                "condition": raw["condition"],
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
                validator.validate(evaluation)
                unknown_concepts = {
                    concept["concept_id"] for concept in evaluation["concepts"]
                    if concept["concept_id"] not in allowed_concept_ids
                }
                if unknown_concepts:
                    raise ValueError(f"Unknown ontology concept_id: {sorted(unknown_concepts)}")
                if any(len(unit["labels"]) != len(set(unit["labels"])) for unit in evaluation["reasoning_units"]):
                    raise ValueError("Duplicate reasoning label within a unit")
                attempt_record["validation_status"] = "valid"
                attempt_records.append(attempt_record)
                record = {
                    "experiment_id": experiment_id,
                    "evaluation_key": key,
                    "response_unique_key": raw["unique_key"],
                    "case_id": raw["case_id"],
                    "case_origin": raw["case_origin"],
                    "case_subtype": raw["case_subtype"],
                    "condition": raw["condition"],
                    "condition_id": raw.get("condition_id"),
                    "input_language": raw.get("input_language", raw["condition"]),
                    "target_jurisdiction": raw.get("target_jurisdiction"),
                    "jurisdiction_instruction": raw.get("jurisdiction_instruction"),
                    "replicate_id": raw["replicate_id"],
                    "evaluator_model_requested": args.model,
                    "evaluator_model_returned": envelope.get("model"),
                    "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
                    "evaluator_system_prompt_sha256": sha256_text(prompts["evaluator_v1_system.txt"]),
                    "evaluator_user_template_sha256": sha256_text(prompts["evaluator_v1_user.txt"]),
                    "evaluator_schema_sha256": sha256_file(SCHEMA_PATH),
                    "ontology_sha256": sha256_file(ONTOLOGY_PATH),
                    "raw_response_sha256": sha256_text(raw["raw_response"]),
                    "raw_response": raw["raw_response"],
                    "timestamp": now_iso(),
                    "latency_seconds": round(latency, 6),
                    "token_usage": envelope.get("usage") or {},
                    "schema_retry_count": schema_attempt,
                    "raw_evaluator_response": raw_evaluator_response,
                    "evaluation": evaluation,
                    "error": None,
                }
                return True, record, local_calls, local_transport_retries, schema_attempt, attempt_records
            except Exception as exc:
                last_error = exc
                attempt_record["validation_status"] = "invalid"
                attempt_record["validation_error"] = f"{type(exc).__name__}: {exc}"
                attempt_records.append(attempt_record)
        failed_record = {
            "experiment_id": experiment_id,
            "evaluation_key": key,
            "response_unique_key": raw["unique_key"],
            "case_id": raw["case_id"],
            "condition": raw["condition"],
            "replicate_id": raw["replicate_id"],
            "timestamp": now_iso(),
            "error": f"{type(last_error).__name__}: {last_error}",
        }
        return False, failed_record, local_calls, local_transport_retries, args.schema_retries, attempt_records

    if not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(evaluate_one, prepared) for prepared in pending]
            for future in as_completed(futures):
                succeeded, record, local_calls, local_transport, schema_attempt, attempt_records = future.result()
                calls += local_calls
                transport_retries += local_transport
                for attempt_record in attempt_records:
                    append_jsonl(attempt_path, attempt_record)
                if succeeded:
                    schema_retry_total += schema_attempt
                    append_jsonl(output_path, record)
                else:
                    failures += 1
                    append_jsonl(failed_path, record)
    if not args.dry_run:
        write_derived(args.output_dir)
    cumulative_attempts = read_jsonl(attempt_path) if attempt_path.is_file() else []
    invalid_reason_counts: Counter[str] = Counter()
    for attempt in cumulative_attempts:
        if attempt.get("validation_status") != "invalid":
            continue
        error = attempt.get("validation_error", "")
        if "Unknown ontology concept_id" in error:
            category = "unknown_ontology_concept_id"
        elif "Duplicate reasoning label" in error:
            category = "duplicate_reasoning_label"
        elif "JSONDecodeError" in error:
            category = "invalid_json"
        else:
            category = error.split(":", 1)[0] or "other"
        invalid_reason_counts[category] += 1
    summary = {
        "experiment_id": experiment_id,
        "model": args.model,
        "parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "schema_retries": args.schema_retries,
            "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
            "evaluator_schema_sha256": sha256_file(SCHEMA_PATH),
            "ontology_sha256": sha256_file(ONTOLOGY_PATH),
        },
        "planned": len(raw_rows),
        "logical_evaluation_attempts": calls,
        "actual_api_calls": calls + transport_retries,
        "failed_evaluations": failures,
        "schema_retries": schema_retry_total,
        "transport_retries": transport_retries,
        "resume_skipped": skipped,
        "dry_run": args.dry_run,
        "cumulative_successful_evaluations": (
            len(read_jsonl(output_path)) if output_path.is_file() else 0
        ),
        "cumulative_api_response_attempts": (
            len(cumulative_attempts)
        ),
        "cumulative_schema_invalid_attempts": (
            sum(
                attempt.get("validation_status") == "invalid"
                for attempt in cumulative_attempts
            )
        ),
        "cumulative_invalid_attempt_reasons": dict(invalid_reason_counts),
    }
    write_json(args.output_dir / "evaluation_run_summary.json", summary)
    write_json(args.output_dir / "evaluation_config.json", summary["parameters"])
    print(" ".join(f"{key}={value}" for key, value in summary.items() if key != "model"))


if __name__ == "__main__":
    main()
