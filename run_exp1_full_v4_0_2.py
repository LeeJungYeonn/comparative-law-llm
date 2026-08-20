from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from pipeline_v2.llm_runtime import load_environment, require_api_key
import run_exp1_smoke_v4_0_1 as smoke


CASES_PATH = Path("outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl")
FACTS_PATH = Path("outputs_v2/v4.0.2/final_fact_patterns_200_v4_0_2.jsonl")
CORPUS_MANIFEST_PATH = Path("outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json")
SMOKE_DIR = Path("outputs_exp1/smoke/v4_0_1_run1")
OUTPUT = Path("outputs_exp1/full_v4_0_2")
RUNNER_PATH = Path(__file__)
SMOKE_RUNNER_PATH = Path("run_exp1_smoke_v4_0_1.py")

EXPERIMENT_ID = "exp1-input-language-full-v4.0.2"
CORPUS_VERSION = "kr-us-highcourt-corpus-v4.0.2"
EFFORTS = ("low", "medium", "high")
LANGUAGES = ("ko", "en")
REPLICATE = 1
BATCH_SIZE = 20
EXPECTED_PROMPT_HASHES = {
    "system": {
        "ko": "5233e22b01fb7ece6df3423e6b47aefdb568f65dbcde2b4b56ab2e31b90d8ad8",
        "en": "abae01ca441ff601e381e3d89fe4e93d19c1fc35cb94d522113e8ed2552b0579",
    },
    "user_template": {
        "ko": "10072034993c28ca6e02e8fe76a13c452b6b05a18052662e072491edbbf615be",
        "en": "46bff1f6cdff606db576b4b28127bc853d91debdfbd44a826feaf0230a527bb7",
    },
}
EXPECTED_CORPUS_HASHES = {
    str(CASES_PATH): "e044ee01df3da9fa0d1c355063be82ccf07971700c2cd943619391948321068b",
    str(FACTS_PATH): "20c4bad65fca9f93050be261c5bf792e762ca1ee6208750285b440cc2bad49f2",
}
PRICING_SNAPSHOT = {
    "model": "gpt-5.6-luna",
    "source": "authenticated Letsur GET https://gw.letsur.ai/v1/models",
    "currency": "unit",
    "input_unit_per_token": 0.0000002,
    "output_unit_per_token": 0.0000012,
    "retrieved_on": "2026-08-20",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_frozen_configuration() -> None:
    actual_prompt_hashes = {
        "system": {key: smoke.sha_text(value) for key, value in smoke.SYSTEM_PROMPTS.items()},
        "user_template": {key: smoke.sha_text(value) for key, value in smoke.USER_TEMPLATES.items()},
    }
    if actual_prompt_hashes != EXPECTED_PROMPT_HASHES:
        raise RuntimeError("Court-opinion prompts differ from the smoke-tested hashes")
    settings = {
        "model": smoke.MODEL,
        "prompt_version": smoke.PROMPT_VERSION,
        "temperature": smoke.TEMPERATURE,
        "top_p": smoke.TOP_P,
        "max_output_tokens": smoke.MAX_OUTPUT_TOKENS,
        "seed": smoke.SEED,
        "replicate": smoke.REPLICATE,
        "concurrency": smoke.CONCURRENCY,
        "max_retries": smoke.MAX_RETRIES,
    }
    expected = {
        "model": "gpt-5.6-luna",
        "prompt_version": "exp1-court-opinion-v1",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 8000,
        "seed": 20260730,
        "replicate": 1,
        "concurrency": 4,
        "max_retries": 3,
    }
    if settings != expected:
        raise RuntimeError(f"Generation settings differ from the smoke-tested freeze: {settings}")
    corpus_hashes = {str(path): sha_file(path) for path in (CASES_PATH, FACTS_PATH)}
    if corpus_hashes != EXPECTED_CORPUS_HASHES:
        raise RuntimeError("v4.0.2 corpus hashes differ from the frozen full-run inputs")
    corpus_manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "FROZEN" or corpus_manifest.get("corpus_version") != CORPUS_VERSION:
        raise RuntimeError("v4.0.2 corpus manifest is not frozen")


def build_plan() -> list[dict[str, Any]]:
    cases = sorted((dict(row) for row in read_jsonl(CASES_PATH)), key=lambda row: row["case_id"])
    facts = {row["case_id"]: dict(row) for row in read_jsonl(FACTS_PATH)}
    if len(cases) != 200 or len(facts) != 200 or {row["case_id"] for row in cases} != set(facts):
        raise RuntimeError("Full Exp 1 requires a matching 200-case/200-fact corpus")
    plan: list[dict[str, Any]] = []
    order = 0
    for case in cases:
        if case.get("analysis_split") not in {"development", "confirmatory"}:
            raise RuntimeError(f"Unexpected analysis split: {case['case_id']}")
        fact = facts[case["case_id"]]
        for language, effort in itertools.product(LANGUAGES, EFFORTS):
            order += 1
            input_text = fact[f"neutral_fact_{language}"]
            system_prompt = smoke.SYSTEM_PROMPTS[language]
            user_prompt = smoke.USER_TEMPLATES[language].format(fact=input_text)
            item = {
                "experiment_id": EXPERIMENT_ID,
                "corpus_version": CORPUS_VERSION,
                "phase": "full_generation",
                "case_id": case["case_id"],
                "case_origin": case["origin_country"],
                "origin_state": case.get("origin_state"),
                "primary_domain": case["primary_domain"],
                "analysis_split": case["analysis_split"],
                "input_language": language,
                "input_text": input_text,
                "input_hash": smoke.sha_text(input_text),
                "model_requested": smoke.MODEL,
                "reasoning_effort": effort,
                "replicate": REPLICATE,
                "seed": smoke.SEED,
                "temperature": smoke.TEMPERATURE,
                "top_p": smoke.TOP_P,
                "max_output_tokens": smoke.MAX_OUTPUT_TOKENS,
                "prompt_version": smoke.PROMPT_VERSION,
                "system_prompt_version": smoke.SYSTEM_VERSIONS[language],
                "user_prompt_version": smoke.USER_VERSIONS[language],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "system_prompt_hash": smoke.sha_text(system_prompt),
                "user_prompt_hash": smoke.sha_text(user_prompt),
                "specific_jurisdiction_in_prompt": bool(
                    smoke.SPECIFIC_JURISDICTIONS.search(system_prompt + "\n" + user_prompt)
                ),
                "request_order": order,
            }
            item["request_key"] = smoke.sha_text(
                f"{EXPERIMENT_ID}|{CORPUS_VERSION}|{case['case_id']}|{language}|{effort}|"
                f"{REPLICATE}|{smoke.PROMPT_VERSION}|{smoke.MODEL}"
            )
            plan.append(item)
    if len(plan) != 1200 or len({row["request_key"] for row in plan}) != 1200:
        raise RuntimeError("Full Exp 1 plan is not the expected 1,200 unique requests")
    if any(row["specific_jurisdiction_in_prompt"] for row in plan):
        leaked = [row["request_key"] for row in plan if row["specific_jurisdiction_in_prompt"]]
        raise RuntimeError(f"Specific jurisdiction leaked into prompt(s): {leaked[:5]}")
    return plan


def linear_fit(pairs: list[tuple[int, int]]) -> tuple[float, float, float]:
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted, strict=True))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r_squared


def cost_usage_estimate(plan: list[dict[str, Any]]) -> dict[str, Any]:
    smoke_inputs = {
        row["request_key"]: dict(row)
        for row in read_jsonl(SMOKE_DIR / "exp1_smoke_inputs.jsonl")
    }
    smoke_responses = [dict(row) for row in read_jsonl(SMOKE_DIR / "exp1_smoke_responses.jsonl")]
    prompt_models: dict[str, dict[str, float]] = {}
    for language in LANGUAGES:
        pairs = []
        for response in smoke_responses:
            if response["phase"] == "primary" and response["input_language"] == language:
                source = smoke_inputs[response["request_key"]]
                pairs.append((len(source["input_text"]), int(response["token_usage"]["prompt_tokens"])))
        intercept, slope, r_squared = linear_fit(pairs)
        prompt_models[language] = {
            "sample_size": len(pairs), "intercept": intercept, "tokens_per_input_character": slope,
            "r_squared": r_squared,
        }

    completion_models: dict[str, dict[str, dict[str, float]]] = {}
    for effort in EFFORTS:
        completion_models[effort] = {}
        for language in LANGUAGES:
            samples = [
                int(row["token_usage"]["completion_tokens"])
                for row in smoke_responses
                if row["input_language"] == language
                and row["reasoning_effort"] == effort
                and ((effort == "medium" and row["phase"] == "primary") or
                     (effort != "medium" and row["phase"] == "effort_parameter_check"))
            ]
            completion_models[effort][language] = {
                "sample_size": len(samples), "mean": statistics.mean(samples),
                "minimum": min(samples), "maximum": max(samples),
            }

    aggregates: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"request_count": 0, "estimated_prompt_tokens": 0, "estimated_completion_tokens": 0,
                 "observed_range_low_completion_tokens": 0, "observed_range_high_completion_tokens": 0,
                 "hard_max_completion_tokens": 0}
    )
    for item in plan:
        prompt_model = prompt_models[item["input_language"]]
        completion_model = completion_models[item["reasoning_effort"]][item["input_language"]]
        prompt_tokens = max(1, round(
            prompt_model["intercept"] + prompt_model["tokens_per_input_character"] * len(item["input_text"])
        ))
        key = (item["analysis_split"], item["reasoning_effort"])
        row = aggregates[key]
        row["request_count"] += 1
        row["estimated_prompt_tokens"] += prompt_tokens
        row["estimated_completion_tokens"] += round(completion_model["mean"])
        row["observed_range_low_completion_tokens"] += completion_model["minimum"]
        row["observed_range_high_completion_tokens"] += completion_model["maximum"]
        row["hard_max_completion_tokens"] += smoke.MAX_OUTPUT_TOKENS

    def finish_row(split: str, effort: str, raw: dict[str, float]) -> dict[str, Any]:
        prompt = int(raw["estimated_prompt_tokens"])
        completion = int(raw["estimated_completion_tokens"])
        low_completion = int(raw["observed_range_low_completion_tokens"])
        high_completion = int(raw["observed_range_high_completion_tokens"])
        hard_completion = int(raw["hard_max_completion_tokens"])
        input_rate = PRICING_SNAPSHOT["input_unit_per_token"]
        output_rate = PRICING_SNAPSHOT["output_unit_per_token"]
        return {
            "analysis_split": split,
            "reasoning_effort": effort,
            "request_count": int(raw["request_count"]),
            "estimated_prompt_tokens": prompt,
            "estimated_completion_tokens": completion,
            "estimated_total_tokens": prompt + completion,
            "estimated_cost_units": round(prompt * input_rate + completion * output_rate, 6),
            "observed_smoke_range_total_tokens": {
                "low": prompt + low_completion, "high": prompt + high_completion,
            },
            "observed_smoke_range_cost_units": {
                "low": round(prompt * input_rate + low_completion * output_rate, 6),
                "high": round(prompt * input_rate + high_completion * output_rate, 6),
            },
            "hard_max_output_total_tokens": prompt + hard_completion,
            "hard_max_output_cost_units": round(prompt * input_rate + hard_completion * output_rate, 6),
        }

    matrix = [
        finish_row(split, effort, aggregates[(split, effort)])
        for split in ("development", "confirmatory") for effort in EFFORTS
    ]

    def combine(rows: list[dict[str, Any]], label_key: str, label: str) -> dict[str, Any]:
        return {
            label_key: label,
            "request_count": sum(row["request_count"] for row in rows),
            "estimated_prompt_tokens": sum(row["estimated_prompt_tokens"] for row in rows),
            "estimated_completion_tokens": sum(row["estimated_completion_tokens"] for row in rows),
            "estimated_total_tokens": sum(row["estimated_total_tokens"] for row in rows),
            "estimated_cost_units": round(sum(row["estimated_cost_units"] for row in rows), 6),
            "observed_smoke_range_total_tokens": {
                bound: sum(row["observed_smoke_range_total_tokens"][bound] for row in rows)
                for bound in ("low", "high")
            },
            "observed_smoke_range_cost_units": {
                bound: round(sum(row["observed_smoke_range_cost_units"][bound] for row in rows), 6)
                for bound in ("low", "high")
            },
            "hard_max_output_total_tokens": sum(row["hard_max_output_total_tokens"] for row in rows),
            "hard_max_output_cost_units": round(sum(row["hard_max_output_cost_units"] for row in rows), 6),
        }

    by_split = [combine([row for row in matrix if row["analysis_split"] == split], "analysis_split", split)
                for split in ("development", "confirmatory")]
    by_effort = [combine([row for row in matrix if row["reasoning_effort"] == effort], "reasoning_effort", effort)
                 for effort in EFFORTS]
    overall = combine(matrix, "scope", "full_exp1")
    return {
        "estimate_version": "exp1-smoke-calibrated-v1",
        "created_at_utc": utc_now(),
        "model": smoke.MODEL,
        "pricing_snapshot": PRICING_SNAPSHOT,
        "smoke_usage_source": str(SMOKE_DIR / "exp1_smoke_responses.jsonl"),
        "smoke_usage_source_sha256": sha_file(SMOKE_DIR / "exp1_smoke_responses.jsonl"),
        "method": {
            "prompt_tokens": "language-specific linear fit of prompt tokens on neutral-fact character count using the 16 primary smoke calls",
            "completion_tokens": "effort/language-specific smoke mean; medium uses the 16 primary calls, low/high use the 12 effort-check calls",
            "range": "projection using observed smoke completion-token minima/maxima within each effort-language cell",
            "hard_max": "8,000 completion tokens per request plus estimated prompt tokens",
            "cost": "estimated tokens multiplied by the authenticated Letsur /models unit rates",
        },
        "prompt_token_models": prompt_models,
        "completion_token_models": completion_models,
        "matrix_by_split_and_effort": matrix,
        "by_split": by_split,
        "by_effort": by_effort,
        "overall": overall,
        "caveats": [
            "This is a pre-run planning estimate, not an output analysis or scientific comparison.",
            "Low/high completion estimates have two smoke observations per language; realized usage may differ.",
            "Letsur pricing is expressed in gateway units; no USD conversion is assumed.",
            "Actual responses will preserve the complete usage object, including estimated_cost if Letsur returns it.",
        ],
    }


def count_manifest(plan: list[dict[str, Any]]) -> dict[str, Any]:
    def count(**criteria: str) -> int:
        return sum(all(row[key] == value for key, value in criteria.items()) for row in plan)
    return {
        "total_cases": 200,
        "total_requests": len(plan),
        "replicates": 1,
        "languages": list(LANGUAGES),
        "reasoning_efforts": list(EFFORTS),
        "case_counts": {"development": 40, "confirmatory": 160},
        "request_counts_by_split": {
            split: count(analysis_split=split) for split in ("development", "confirmatory")
        },
        "request_counts_by_effort": {effort: count(reasoning_effort=effort) for effort in EFFORTS},
        "request_matrix_by_split_and_effort": {
            split: {effort: count(analysis_split=split, reasoning_effort=effort) for effort in EFFORTS}
            for split in ("development", "confirmatory")
        },
        "request_counts_by_country_split": {
            country: {
                split: count(case_origin=country, analysis_split=split)
                for split in ("development", "confirmatory")
            }
            for country in ("KR", "US")
        },
    }


def prepare() -> None:
    assert_frozen_configuration()
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing full-run freeze directory: {OUTPUT}")
    plan = build_plan()
    counts = count_manifest(plan)
    if counts["request_matrix_by_split_and_effort"] != {
        "development": {"low": 80, "medium": 80, "high": 80},
        "confirmatory": {"low": 320, "medium": 320, "high": 320},
    }:
        raise RuntimeError("Development/confirmatory effort matrix is incorrect")
    estimate = cost_usage_estimate(plan)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    paths = {
        "config": OUTPUT / "exp1_full_config.json",
        "prompts": OUTPUT / "exp1_full_prompt_manifest.json",
        "inputs": OUTPUT / "exp1_full_inputs.jsonl",
        "manifest": OUTPUT / "exp1_full_run_manifest.json",
        "estimate": OUTPUT / "exp1_full_cost_usage_estimate.json",
        "lock": OUTPUT / "EXP1_FULL_FROZEN.lock.json",
        "report": OUTPUT / "EXP1_FULL_FREEZE_REPORT.md",
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "corpus_version": CORPUS_VERSION,
        "status": "FROZEN_NOT_LAUNCHED",
        "model": smoke.MODEL,
        "base_url_identifier": smoke.BASE_URL,
        "prompt_version": smoke.PROMPT_VERSION,
        "temperature": smoke.TEMPERATURE,
        "top_p": smoke.TOP_P,
        "max_output_tokens": smoke.MAX_OUTPUT_TOKENS,
        "reasoning_efforts": list(EFFORTS),
        "replicate": REPLICATE,
        "seed": smoke.SEED,
        "concurrency": smoke.CONCURRENCY,
        "batch_size": BATCH_SIZE,
        "max_retries": smoke.MAX_RETRIES,
        "api_key_env": "LETSUR_API_KEY",
        "logging_rules_source": str(SMOKE_RUNNER_PATH),
        "hashing_rules_source": str(SMOKE_RUNNER_PATH),
        "retry_rules_source": str(SMOKE_RUNNER_PATH),
        "qc_rules_source": str(SMOKE_RUNNER_PATH),
        "prompt_tuning_based_on_jurisdictional_markers": False,
        "marker_evaluation_enabled": False,
        "hypothesis_testing_enabled": False,
        "pca_enabled": False,
        "exp2_enabled": False,
    }
    prompt_manifest = {
        "prompt_version": smoke.PROMPT_VERSION,
        "system_prompt_versions": smoke.SYSTEM_VERSIONS,
        "user_prompt_versions": smoke.USER_VERSIONS,
        "system_prompts": smoke.SYSTEM_PROMPTS,
        "user_templates": smoke.USER_TEMPLATES,
        "system_prompt_hashes": EXPECTED_PROMPT_HASHES["system"],
        "user_template_hashes": EXPECTED_PROMPT_HASHES["user_template"],
        "same_as_smoke_test": True,
        "specific_jurisdiction_names_present": False,
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_NOT_LAUNCHED",
        "created_at_utc": utc_now(),
        "corpus": {
            "version": CORPUS_VERSION,
            "cases_path": str(CASES_PATH),
            "facts_path": str(FACTS_PATH),
            "hashes": EXPECTED_CORPUS_HASHES,
        },
        "configuration": config,
        "counts": counts,
        "input_plan_sha256": None,
        "full_generation_launched": False,
        "output_analysis_performed": False,
        "marker_evaluation_performed": False,
        "hypothesis_testing_performed": False,
        "pca_performed": False,
        "exp2_performed": False,
    }
    write_json(paths["config"], config)
    write_json(paths["prompts"], prompt_manifest)
    write_jsonl(paths["inputs"], plan)
    manifest["input_plan_sha256"] = sha_file(paths["inputs"])
    write_json(paths["manifest"], manifest)
    write_json(paths["estimate"], estimate)
    locked_files = [
        paths["config"], paths["prompts"], paths["inputs"], paths["manifest"], paths["estimate"],
        CASES_PATH, FACTS_PATH, CORPUS_MANIFEST_PATH, RUNNER_PATH, SMOKE_RUNNER_PATH,
    ]
    lock = {
        "status": "FROZEN_NOT_LAUNCHED",
        "created_at_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "frozen_artifact_sha256": {str(path): sha_file(path) for path in locked_files},
        "launch_command": "& .venv\\Scripts\\python.exe run_exp1_full_v4_0_2.py --execute-frozen-full-run",
        "launch_performed": False,
    }
    write_json(paths["lock"], lock)
    overall = estimate["overall"]
    report = [
        "# Frozen full Exp 1 generation pipeline", "", "Status: **FROZEN — NOT LAUNCHED**", "",
        f"- Corpus: `{CORPUS_VERSION}`", f"- Model: `{smoke.MODEL}`", f"- Prompt: `{smoke.PROMPT_VERSION}`",
        f"- Parameters: `temperature={smoke.TEMPERATURE}`, `top_p={smoke.TOP_P}`, `max_output_tokens={smoke.MAX_OUTPUT_TOKENS}`, `seed={smoke.SEED}`, `replicate={REPLICATE}`, `concurrency={smoke.CONCURRENCY}`, `max_retries={smoke.MAX_RETRIES}`",
        f"- Matrix: **{counts['total_requests']}** requests — development 240, confirmatory 960; low/medium/high 400 each",
        f"- Estimated usage: **{overall['estimated_total_tokens']:,} tokens** ({overall['estimated_prompt_tokens']:,} prompt + {overall['estimated_completion_tokens']:,} completion)",
        f"- Estimated cost: **{overall['estimated_cost_units']:.6f} Letsur units**",
        f"- Smoke-observed projection range: {overall['observed_smoke_range_total_tokens']['low']:,}–{overall['observed_smoke_range_total_tokens']['high']:,} tokens; {overall['observed_smoke_range_cost_units']['low']:.6f}–{overall['observed_smoke_range_cost_units']['high']:.6f} units",
        f"- Hard 8,000-output-token ceiling: {overall['hard_max_output_total_tokens']:,} tokens; {overall['hard_max_output_cost_units']:.6f} units", "",
        "The bilingual prompt hashes and generation/logging/hashing/retry/QC rules are locked to the passing smoke implementation. No prompt tuning was performed. Full generation, output analysis, marker evaluation, hypothesis testing, PCA, and Exp 2 were not run.", "",
        "## Request matrix", "",
        "| Split | Low | Medium | High | Total |", "|---|---:|---:|---:|---:|",
        "| Development | 80 | 80 | 80 | 240 |", "| Confirmatory | 320 | 320 | 320 | 960 |",
        "| Total | 400 | 400 | 400 | 1,200 |", "",
        "Detailed split/effort token and unit estimates are in `exp1_full_cost_usage_estimate.json`.",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    checksum_paths = list(paths.values())
    (OUTPUT / "SHA256SUMS_exp1_full_freeze.txt").write_text(
        "".join(f"{sha_file(path)}  {path.name}\n" for path in sorted(checksum_paths)), encoding="utf-8"
    )
    print(json.dumps({"status": "FROZEN_NOT_LAUNCHED", "counts": counts, "overall_estimate": overall}, ensure_ascii=False))


def verify_lock() -> list[dict[str, Any]]:
    assert_frozen_configuration()
    lock_path = OUTPUT / "EXP1_FULL_FROZEN.lock.json"
    if not lock_path.exists():
        raise RuntimeError("Frozen full-run lock is missing; run preparation first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mismatches = {
        path: {"expected": expected, "observed": sha_file(Path(path))}
        for path, expected in lock["frozen_artifact_sha256"].items()
        if not Path(path).exists() or sha_file(Path(path)) != expected
    }
    if mismatches:
        raise RuntimeError({"frozen_artifact_hash_mismatches": mismatches})
    plan = [dict(row) for row in read_jsonl(OUTPUT / "exp1_full_inputs.jsonl")]
    if len(plan) != 1200:
        raise RuntimeError("Frozen plan no longer contains 1,200 requests")
    return plan


def execute() -> None:
    plan = verify_lock()
    load_environment(Path(".env"))
    api_key = require_api_key("LETSUR_API_KEY")
    response_path = OUTPUT / "exp1_full_responses.jsonl"
    existing = [dict(row) for row in read_jsonl(response_path)] if response_path.exists() else []
    existing_keys = {row["request_key"] for row in existing}
    if len(existing_keys) != len(existing) or not existing_keys <= {row["request_key"] for row in plan}:
        raise RuntimeError("Existing response checkpoint has duplicate or unknown request keys")
    pending = [row for row in plan if row["request_key"] not in existing_keys]
    for offset in range(0, len(pending), BATCH_SIZE):
        for response in smoke.run_batch(pending[offset:offset + BATCH_SIZE], api_key):
            append_jsonl(response_path, response)
    responses = [dict(row) for row in read_jsonl(response_path)]
    response_by_key = {row["request_key"]: row for row in responses}
    if len(response_by_key) != 1200:
        raise RuntimeError("Full run ended without 1,200 unique responses")
    qc = [smoke.response_qc(item, response_by_key[item["request_key"]]) for item in plan]
    smoke.write_qc(OUTPUT / "exp1_full_generation_qc.csv", qc)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "GENERATION_COMPLETE_QC_PASS" if all(row["qc_pass"] for row in qc) else "GENERATION_COMPLETE_QC_FAIL",
        "completed_at_utc": utc_now(),
        "response_count": len(responses),
        "generation_qc_pass": sum(row["qc_pass"] for row in qc),
        "generation_qc_fail": sum(not row["qc_pass"] for row in qc),
        "actual_token_usage": {
            field: sum(int((row.get("token_usage") or {}).get(field) or 0) for row in responses)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "actual_estimated_cost_returned_by_gateway": sum(
            float((row.get("token_usage") or {}).get("estimated_cost") or 0) for row in responses
        ),
        "output_analysis_performed": False,
        "marker_evaluation_performed": False,
        "hypothesis_testing_performed": False,
        "pca_performed": False,
        "exp2_performed": False,
    }
    write_json(OUTPUT / "exp1_full_generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or explicitly execute the frozen full Exp 1 generation pipeline.")
    parser.add_argument("--execute-frozen-full-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.execute_frozen_full_run:
        execute()
    else:
        prepare()
