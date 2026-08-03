"""Shared experiment definitions for the Exp 1/Exp 2 generation pipeline."""

from __future__ import annotations

from typing import Any

from exp1 import EXPERIMENT_ID as EXP1_EXPERIMENT_ID
from exp1 import GENERATION_PROMPT_VERSION as EXP1_GENERATION_PROMPT_VERSION

EXP2_EXPERIMENT_ID = "exp2-explicit-jurisdiction-v1"
EXP2_GENERATION_PROMPT_VERSION = "generation_v1+jurisdiction_v1"
JURISDICTION_INSTRUCTION_VERSION = "jurisdiction_v1"

JURISDICTION_INSTRUCTIONS = {
    ("KR", "ko"): "다음 사실관계를 대한민국 법률에 따라 판단하라.",
    ("KR", "en"): "Analyze the following facts under the law of the Republic of Korea.",
    ("CA", "ko"): "다음 사실관계를 미국 캘리포니아주 법률에 따라 판단하라.",
    ("CA", "en"): "Analyze the following facts under California law.",
}
TARGET_JURISDICTIONS = {"KR": "KR", "CA": "US_CA"}


def experiment_values(experiment: str) -> tuple[str, str]:
    if experiment == "exp1":
        return EXP1_EXPERIMENT_ID, EXP1_GENERATION_PROMPT_VERSION
    if experiment == "exp2":
        return EXP2_EXPERIMENT_ID, EXP2_GENERATION_PROMPT_VERSION
    raise ValueError(f"Unsupported experiment: {experiment}")


def jurisdiction_metadata(row: dict[str, Any], input_language: str) -> dict[str, str | None]:
    origin = str(row["case_origin"])
    if origin not in TARGET_JURISDICTIONS or input_language not in {"ko", "en"}:
        raise ValueError(f"Unsupported condition: origin={origin}, language={input_language}")
    target = TARGET_JURISDICTIONS[origin]
    return {
        "condition_id": f"{origin}_{input_language}_{target}",
        "target_jurisdiction": target,
        "jurisdiction_instruction": JURISDICTION_INSTRUCTIONS[(origin, input_language)],
    }
