"""Validate the Exp 2 design against the immutable Exp 1 artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonschema

from exp1.common import (
    DEFAULT_INPUT, REPO_ROOT, directory_hashes, generation_record_base, load_prompts, read_jsonl,
    render_generation, sha256_file, sha256_text, usable_cases, write_json,
)
from exp1.design import (
    EXP2_EXPERIMENT_ID, EXP2_GENERATION_PROMPT_VERSION, JURISDICTION_INSTRUCTIONS,
    TARGET_JURISDICTIONS, jurisdiction_metadata,
)


REQUIRED_RAW_FIELDS = {
    "experiment_id", "condition_id", "case_id", "case_origin", "input_language",
    "target_jurisdiction", "jurisdiction_instruction", "prompt_version",
    "base_prompt_version", "system_prompt_sha256", "user_prompt_template_sha256",
    "user_prompt_sha256", "fact_text_sha256", "input_text_sha256",
    "jurisdiction_instruction_version", "jurisdiction_instruction_sha256",
    "replicate_id", "replicate_number",
    "raw_response",
}
SETTING_MAP = {
    "model": "model", "base_url_identifier": "base_url_identifier",
    "temperature": "temperature", "top_p": "top_p",
    "max_output_tokens": "max_output_tokens", "reasoning_effort": "reasoning_effort",
    "seed": "seed", "request_order_seed": "request_order_seed",
    "max_retries": "max_retries", "concurrency": "concurrency",
}


def validate(
    input_path: Path, exp1_dir: Path, exp2_dir: Path,
) -> dict[str, Any]:
    config = json.loads((exp2_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((exp2_dir / "run_manifest.json").read_text(encoding="utf-8"))
    exp1_config = json.loads((exp1_dir / "config.json").read_text(encoding="utf-8"))
    exp1_manifest = json.loads((exp1_dir / "run_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((exp2_dir / "request_plan.json").read_text(encoding="utf-8"))
    rows = {row["case_id"]: row for row in usable_cases(input_path)}
    exp1_raw = read_jsonl(exp1_dir / "raw_responses.jsonl")
    exp2_raw = read_jsonl(exp2_dir / "raw_responses.jsonl") if (exp2_dir / "raw_responses.jsonl").is_file() else []

    conditions: dict[str, set[str]] = defaultdict(set)
    targets: dict[str, set[str]] = defaultdict(set)
    instructions_ok = True
    for item in plan:
        case_id, language = item["case_id"], item["input_language"]
        conditions[case_id].add(language)
        targets[case_id].add(item["target_jurisdiction"])
        origin = item["case_origin"]
        instructions_ok &= item["jurisdiction_instruction"] == JURISDICTION_INSTRUCTIONS[(origin, language)]
        instructions_ok &= item["target_jurisdiction"] == TARGET_JURISDICTIONS[origin]
        instructions_ok &= item["condition_id"] == jurisdiction_metadata(rows[case_id], language)["condition_id"]

    prompt_delta_ok = True
    prompts = load_prompts()
    for case_id in conditions:
        row = rows[case_id]
        for language in ("ko", "en"):
            exp1_system, exp1_user = render_generation(row, language, prompts)
            exp2_system, exp2_user = render_generation(row, language, prompts, EXP2_EXPERIMENT_ID)
            instruction = JURISDICTION_INSTRUCTIONS[(row["case_origin"], language)]
            prompt_delta_ok &= exp2_system == exp1_system
            prompt_delta_ok &= exp2_user == instruction + "\n\n" + exp1_user

    exp1_fact_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in exp1_raw:
        exp1_fact_hashes[(record["case_id"], record["input_language"])].add(record["fact_text_sha256"])
    facts_unchanged = all(
        exp1_fact_hashes[(case_id, language)] == {sha256_text(str(rows[case_id][f"neutral_fact_{language}"]))}
        for case_id in conditions for language in ("ko", "en")
    )

    settings_unchanged = all(config.get(exp2_key) == exp1_config.get(exp1_key) for exp2_key, exp1_key in SETTING_MAP.items())
    if not config.get("smoke_test"):
        settings_unchanged &= config.get("repetitions") == exp1_config.get("repetitions")
    settings_unchanged &= config.get("base_generation_prompt_version") == exp1_config.get("generation_prompt_version")

    exp1_case_ids = {record["case_id"] for record in exp1_raw}
    selected_case_ids = set(conditions)
    case_selection_ok = (
        selected_case_ids <= exp1_case_ids if config.get("smoke_test")
        else selected_case_ids == exp1_case_ids and len(selected_case_ids) == 70
    )
    pair_conditions_ok = bool(conditions) and all(value == {"ko", "en"} for value in conditions.values())
    paired_target_ok = all(len(value) == 1 for value in targets.values())

    prototype = generation_record_base(
        next(iter(rows.values())), "ko", 1, str(config["model"]), float(config["temperature"]),
        float(config["top_p"]), int(config["max_output_tokens"]), config.get("reasoning_effort"),
        int(config["seed"]), 1, EXP2_EXPERIMENT_ID, EXP2_GENERATION_PROMPT_VERSION,
    )
    storage_schema_ok = REQUIRED_RAW_FIELDS <= prototype.keys()
    raw_schema = json.loads((REPO_ROOT / "schemas/exp2_raw_response_v1.schema.json").read_text(encoding="utf-8"))
    raw_validator = jsonschema.Draft202012Validator(raw_schema)
    try:
        raw_validator.validate({**prototype, "raw_response": "LOCAL_SCHEMA_CHECK"})
    except jsonschema.ValidationError:
        storage_schema_ok = False
    if exp2_raw:
        storage_schema_ok &= all(REQUIRED_RAW_FIELDS <= record.keys() for record in exp2_raw)
        storage_schema_ok &= all(record.get("experiment_id") == EXP2_EXPERIMENT_ID for record in exp2_raw)
        storage_schema_ok &= all(not list(raw_validator.iter_errors(record)) for record in exp2_raw)

    baseline_hashes = manifest.get("exp1_baseline_file_hashes", {})
    exp1_files_unchanged = bool(baseline_hashes) and baseline_hashes == directory_hashes(exp1_dir)
    input_unchanged = (
        manifest.get("input_sha256") == exp1_manifest.get("input_sha256") == sha256_file(input_path)
    )
    checks = {
        "exact_jurisdiction_instructions": instructions_ok,
        "each_case_has_ko_and_en": pair_conditions_ok,
        "paired_languages_share_target_jurisdiction": paired_target_ok,
        "case_selection_matches_exp1": case_selection_ok,
        "fact_patterns_unchanged_from_exp1": facts_unchanged,
        "generation_and_runtime_settings_unchanged_from_exp1": settings_unchanged,
        "only_prompt_delta_is_jurisdiction_prefix": prompt_delta_ok,
        "input_file_unchanged_from_exp1": input_unchanged,
        "required_raw_record_schema_present": storage_schema_ok,
        "existing_exp1_files_unchanged": exp1_files_unchanged,
    }
    return {
        "experiment_id": EXP2_EXPERIMENT_ID,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "selected_case_count": len(selected_case_ids),
        "planned_request_count": len(plan),
        "raw_response_count": len(exp2_raw),
        "smoke_test": bool(config.get("smoke_test")),
        "smoke_repetition_exception": bool(config.get("smoke_test") and config.get("repetitions") != exp1_config.get("repetitions")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp 2 against Exp 1 without API calls.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--exp1-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--exp2-dir", type=Path, default=Path("outputs/exp2"))
    args = parser.parse_args()
    report = validate(args.input, args.exp1_dir, args.exp2_dir)
    write_json(args.exp2_dir / "validation_report.json", report)
    print(f"validation={report['status']} checks={sum(report['checks'].values())}/{len(report['checks'])}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
