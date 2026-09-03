import json
from pathlib import Path

from scripts.exp1.run_implicit_jurisdiction_full import (
    SYSTEM_PROMPTS,
    USER_TEMPLATES,
    EXPECTED_PRIMARY_CASES,
    EXPECTED_REQUESTS,
    NOT_APPLICABLE_SINGLE_MODEL,
    PRIMARY_MODEL_ID,
    PRIMARY_REPLICATES,
    build_request_manifest,
    fact_preflight,
    prompt_manifest,
    read_jsonl,
    sha_text,
    CASES_PATH,
    FACTS_PATH,
)


def test_protocol_prompts_are_symmetric_and_jurisdiction_free():
    manifest, symmetry = prompt_manifest()
    assert symmetry["all_required_elements_symmetric"]
    assert not manifest["specific_jurisdiction_names_present"]
    assert not manifest["banned_instruction_hits"]
    assert "IRAC" not in "\n".join(USER_TEMPLATES.values())


def test_frozen_bilingual_inputs_pass_computable_integrity_checks():
    cases, facts = read_jsonl(CASES_PATH), read_jsonl(FACTS_PATH)
    summary, rows, _ = fact_preflight(cases, facts)
    assert summary["case_count"] == summary["fact_count"] == 200
    assert summary["nonempty_bilingual_pairs"] == 200
    assert summary["placeholder_consistency_pass"] == 200
    assert summary["input_hash_pass"] == 200
    assert summary["deterministic_legal_leakage_recheck_pass"] == 200
    assert all(row["translation_qc_accepted"] for row in rows)


def test_request_expansion_is_deterministic_and_replicate_hash_stable():
    cases, facts = read_jsonl(CASES_PATH), read_jsonl(FACTS_PATH)
    existing = {
        "configured_models": [PRIMARY_MODEL_ID], "reasoning_efforts": ["low", "medium", "high"],
        "replicate_count": 1, "provider": "legacy-provider-label",
    }
    prompt, _ = prompt_manifest()
    first = build_request_manifest(cases, facts, existing, prompt)
    second = build_request_manifest(cases, facts, existing, prompt)
    assert first == second
    assert len(first) == EXPECTED_REQUESTS == EXPECTED_PRIMARY_CASES * 2 * 1 * 3 * PRIMARY_REPLICATES
    assert len({row["case_id"] for row in first}) == EXPECTED_PRIMARY_CASES
    assert {row["analysis_split"] for row in first} == {"confirmatory"}
    assert {row["model_id"] for row in first} == {PRIMARY_MODEL_ID}
    assert {row["model_family"] for row in first} == {NOT_APPLICABLE_SINGLE_MODEL}
    assert {int(row["replicate"]) for row in first} == {1, 2}
    assert first[0]["randomized_order"] == 1
    assert first[0]["request_index"] != 1
    expected_prompt_hashes = {
        language.upper(): sha_text(SYSTEM_PROMPTS[language] + "\n" + USER_TEMPLATES[language])
        for language in ("ko", "en")
    }
    assert all(row["prompt_hash"] == expected_prompt_hashes[row["input_language"]] for row in first)
    grouped = {}
    for row in first:
        key = row["case_id"], row["input_language"], row["model_id"], row["reasoning_effort"]
        grouped.setdefault(key, set()).add(row["input_hash"])
    assert all(len(values) == 1 for values in grouped.values())
