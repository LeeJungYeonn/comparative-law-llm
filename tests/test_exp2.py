from __future__ import annotations

from exp1.common import generation_record_base, load_prompts, render_generation
from exp1.design import (
    EXP2_EXPERIMENT_ID, EXP2_GENERATION_PROMPT_VERSION, JURISDICTION_INSTRUCTIONS,
    jurisdiction_metadata,
)
from analyze_exp2 import analyze


def case(origin: str) -> dict:
    return {
        "case_id": f"{origin}_1", "case_origin": origin, "case_subtype": "traffic",
        "neutral_fact_ko": "[원고]가 다쳤다.", "neutral_fact_en": "[Plaintiff] was injured.",
        "source_language": "ko" if origin == "KR" else "en", "master_language": "ko" if origin == "KR" else "en",
    }


def evaluation(jurisdiction: str, concept_id: str, conclusion: str = "likely", damage: str = "medical_expenses") -> dict:
    return {
        "parties": [{"party_id": "[D]", "conclusion": conclusion}],
        "jurisdiction_signals": {"explicit_jurisdiction": jurisdiction},
        "concepts": [{"concept_id": concept_id, "present": True}],
        "reasoning_units": [{"unit_id": 1, "text_span": "rule", "labels": ["governing_rule"]}],
        "damages": [{"damage_id": damage, "present": True}],
    }


def test_all_four_jurisdiction_instructions_are_exact() -> None:
    assert JURISDICTION_INSTRUCTIONS == {
        ("KR", "ko"): "다음 사실관계를 대한민국 법률에 따라 판단하라.",
        ("KR", "en"): "Analyze the following facts under the law of the Republic of Korea.",
        ("CA", "ko"): "다음 사실관계를 미국 캘리포니아주 법률에 따라 판단하라.",
        ("CA", "en"): "Analyze the following facts under California law.",
    }


def test_exp2_only_prefixes_exp1_user_prompt() -> None:
    prompts = load_prompts()
    for origin in ("KR", "CA"):
        row = case(origin)
        for language in ("ko", "en"):
            system1, user1 = render_generation(row, language, prompts)
            system2, user2 = render_generation(row, language, prompts, EXP2_EXPERIMENT_ID)
            assert system2 == system1
            assert user2 == JURISDICTION_INSTRUCTIONS[(origin, language)] + "\n\n" + user1


def test_exp2_record_contains_required_provenance() -> None:
    row = case("CA")
    record = generation_record_base(
        row, "en", 2, "snapshot", 1.0, 1.0, 8000, "low", 8, 3,
        EXP2_EXPERIMENT_ID, EXP2_GENERATION_PROMPT_VERSION,
    )
    assert record["condition_id"] == "CA_en_US_CA"
    assert record["target_jurisdiction"] == "US_CA"
    assert record["replicate_number"] == 2
    assert record["jurisdiction_instruction"] == JURISDICTION_INSTRUCTIONS[("CA", "en")]
    assert all(len(record[field]) == 64 for field in (
        "fact_text_sha256", "input_text_sha256", "system_prompt_sha256",
        "user_prompt_sha256", "jurisdiction_instruction_sha256",
    ))


def test_exp2_analysis_retains_required_metrics() -> None:
    exp1, exp2 = [], []
    for language in ("ko", "en"):
        base = {
            "case_id": "KR_1", "case_origin": "KR", "condition": language,
            "replicate_id": 1, "experiment_id": "exp1-input-language-v1",
            "evaluation": evaluation("NONE", "general_causation", damage="medical_expenses"),
        }
        treated = {
            **base, "experiment_id": EXP2_EXPERIMENT_ID, "target_jurisdiction": "KR",
            "evaluation": evaluation("KR", "kr_civil_act_750", damage="pain_and_suffering_or_nonpecuniary"),
        }
        exp1.append(base); exp2.append(treated)
    rows, summary = analyze(exp1, exp2)
    assert len(rows) == 2
    assert summary["instruction_jurisdiction_alignment"]["alignment_rate"] == 1.0
    assert summary["wrong_jurisdiction_terms"]["response_prevalence"] == 0.0
    assert summary["remedy_category_shift"]["response_shift_rate"] == 1.0
    assert "mean_js_distance" in summary["reasoning_unit_distribution_distance"]
