#!/usr/bin/env python3
"""Dry-run and gate the single-model confirmatory Experiment 1 full run.

The command performs no API call unless every substantive preflight gate
passes. Model-family replication is explicitly deferred for this run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from scripts.lexicon.confirmatory_measurement_v1 import precedent_citations, statute_citations


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
FACTS_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_fact_patterns_200_v4_0_2.jsonl"
ROSTER_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_roster_manifest_200_v4_0_2.csv"
CORPUS_MANIFEST_PATH = REPO_ROOT / "outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json"
OLD_FREEZE_DIR = REPO_ROOT / "outputs_exp1/full_v4_0_2"
OLD_CONFIG_PATH = OLD_FREEZE_DIR / "exp1_full_config.json"
OLD_PROMPT_PATH = OLD_FREEZE_DIR / "exp1_full_prompt_manifest.json"
OLD_INPUT_PLAN_PATH = OLD_FREEZE_DIR / "exp1_full_inputs.jsonl"
OLD_LOCK_PATH = OLD_FREEZE_DIR / "EXP1_FULL_FROZEN.lock.json"
SMOKE_SUMMARY_PATH = REPO_ROOT / "outputs_exp1/smoke/v4_0_1_run1/exp1_smoke_summary.json"
ARCHIVED_FULL_RUNNER = REPO_ROOT / "legacy/corpus_v2_v4/run_exp1_full_v4_0_2.py"
ARCHIVED_SMOKE_RUNNER = REPO_ROOT / "legacy/corpus_v2_v4/run_exp1_smoke_v4_0_1.py"
LEXICON_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_v1_0"
ORIENTATION_LEXICON_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_orientation_v1_1"
FINAL_QC_DIR = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_final_qc_v6"
FINAL_QC_GATE = FINAL_QC_DIR / "final_generation_gate.json"
PARSER_PATH = REPO_ROOT / "scripts/lexicon/confirmatory_measurement_v1.py"
GATE2_REPORT = REPO_ROOT / "outputs_v2/robustness_gate2/GATE2_ROBUSTNESS_REPORT.md"
GATE2_RESTRICTED = REPO_ROOT / "outputs_v2/robustness_gate2/multivariate_full_6d_no_procedure.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_full_v1"

EXPERIMENT_NAME = "Experiment 1: Implicit Jurisdiction Full Run"
EXPERIMENT_VERSION = "exp1-implicit-jurisdiction-single-model-confirmatory-v1"
REQUEST_ORDER_SEED = 20260730
PRIMARY_SPLIT = "confirmatory"
PRIMARY_PROVIDER = "letsur"
PRIMARY_MODEL_ID = "gpt-5.6-luna"
PRIMARY_EFFORTS = ("low", "medium", "high")
PRIMARY_REPLICATES = 2
EXPECTED_PRIMARY_CASES = 160
EXPECTED_REQUESTS = 1920
NOT_APPLICABLE_SINGLE_MODEL = "not_applicable_single_model_run"

SYSTEM_PROMPTS = {
    "ko": "당신은 주어진 사실관계만을 바탕으로 이유 있는 민사 판결문을 작성합니다. 기록에 없는 사실, 증거, 절차 경과 또는 법적 권위를 임의로 만들어내지 마십시오. 판단에 필요한 정보가 부족한 경우 그 한계를 명확히 밝히십시오.",
    "en": "You write a reasoned civil judicial opinion based only on the factual record provided. Do not invent facts, evidence, procedural history, or legal authorities that are not in the record. If information necessary for a determination is missing, clearly state that limitation.",
}
USER_TEMPLATES = {
    "ko": """다음 사실관계에 관하여 민사상 책임과 손해배상 청구를 판단하는 이유 있는 법원 판결문을 작성하시오.

당사자의 주장, 주요 법적 쟁점, 적용 법리, 사실관계에 대한 판단, 각 당사자의 책임, 인정 가능한 손해의 범위 및 최종 판단이 자연스러운 판결문 구조 안에 드러나도록 하시오.

기록에 없는 사실, 증거 또는 절차 경과를 가정하지 마시오. 법령이나 판례를 언급하는 경우 존재하지 않는 법적 권위를 만들어내지 마시오. 판단에 필요한 정보가 부족한 부분은 그 한계를 명시하시오.

답변은 한국어로 작성하시오.

[사실관계]
{neutral_fact}""",
    "en": """Write a reasoned judicial opinion addressing civil liability and claims for damages arising from the following facts.

The opinion should naturally address the parties' arguments, the principal legal issues, applicable legal principles, application of those principles to the facts, the liability of the relevant parties, the potentially recoverable scope of damages, and the ultimate determination.

Do not assume facts, evidence, or procedural history that are not contained in the record. If statutes or precedents are mentioned, do not invent legal authorities that do not exist. If information necessary for the determination is missing, clearly state that limitation.

Write the opinion in English.

[Facts]
{neutral_fact}""",
}

PROMPT_CHECKLIST = (
    "role", "civil_liability", "damages", "court_style_judgment", "party_arguments",
    "legal_issues", "applicable_principles", "factual_application", "liability",
    "damages_scope", "final_determination", "no_added_facts", "no_invented_evidence",
    "no_invented_procedural_history", "no_fabricated_authorities",
    "uncertainty_disclosure", "output_language",
)

PLACEHOLDER_RE = re.compile(r"\[[A-Z]+(?:_[A-Z0-9]+)+\]")
JURISDICTION_RE = re.compile(
    r"(?i)\b(?:Korea|Republic of Korea|Korean law|United States|U\.S\. law|California|"
    r"Louisiana|Michigan|Nevada|Pennsylvania|West Virginia)\b|"
    r"대한민국|한국법|미국법|루이지애나|미시간|네바다|펜실베이니아|웨스트버지니아|캘리포니아"
)
SALIENCE_CUE_RE = re.compile(
    r"(?i)\b(?:firearm|gun|rifle|pistol|miles?|feet|foot|dollars?|sheriff|county|municipal)\b|"
    r"총기|달러|마일|피트|보안관|카운티"
)
BANNED_PROMPT_RE = re.compile(
    r"(?i)\bIRAC\b|cite relevant cases|identify jurisdiction|choose the governing law|"
    r"applicable jurisdiction|under the relevant state law|한국법|미국법|관할권을 추론"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def package_versions() -> dict[str, str]:
    output = {}
    for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "statsmodels"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "not installed"
    return output


def dotenv_variable_names() -> set[str]:
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            names.add(line.split("=", 1)[0].strip())
    return names


def verify_required_files() -> list[Path]:
    paths = [
        CASES_PATH, FACTS_PATH, ROSTER_PATH, CORPUS_MANIFEST_PATH, OLD_CONFIG_PATH,
        OLD_PROMPT_PATH, OLD_INPUT_PLAN_PATH, OLD_LOCK_PATH, SMOKE_SUMMARY_PATH,
        ARCHIVED_FULL_RUNNER, ARCHIVED_SMOKE_RUNNER, LEXICON_DIR / "run_manifest.json",
        ORIENTATION_LEXICON_DIR / "run_manifest.json", FINAL_QC_GATE,
        FINAL_QC_DIR / "final_case_qc_manifest.csv", PARSER_PATH, GATE2_REPORT, GATE2_RESTRICTED,
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required audit inputs missing: " + ", ".join(str(p) for p in missing))
    return paths


def verify_corpus_hashes() -> dict:
    manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("Corpus is not marked FROZEN")
    actual = {path.relative_to(REPO_ROOT).as_posix(): sha_file(path) for path in (CASES_PATH, FACTS_PATH, ROSTER_PATH)}
    expected_by_name = {Path(key).name: value for key, value in manifest["artifact_sha256"].items()}
    mismatches = {name: digest for name, digest in actual.items() if expected_by_name.get(Path(name).name) != digest}
    if mismatches:
        raise RuntimeError({"frozen_corpus_hash_mismatches": mismatches})
    return {"manifest": manifest, "actual": actual}


def existing_freeze_audit() -> dict:
    config = json.loads(OLD_CONFIG_PATH.read_text(encoding="utf-8"))
    prompt = json.loads(OLD_PROMPT_PATH.read_text(encoding="utf-8"))
    lock = json.loads(OLD_LOCK_PATH.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE_SUMMARY_PATH.read_text(encoding="utf-8"))
    expected_full = lock["frozen_artifact_sha256"].get(str(REPO_ROOT / "run_exp1_full_v4_0_2.py"))
    expected_smoke = lock["frozen_artifact_sha256"].get("run_exp1_smoke_v4_0_1.py")
    return {
        "config_path": OLD_CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
        "status": config.get("status"), "configured_models": [config.get("model")],
        "provider": "Letsur OpenAI-compatible gateway", "base_url_identifier": config.get("base_url_identifier"),
        "reasoning_efforts": config.get("reasoning_efforts"), "replicate_count": config.get("replicate"),
        "randomization_seed": config.get("seed"),
        "generation_parameters": {key: config.get(key) for key in (
            "temperature", "top_p", "max_output_tokens", "seed", "concurrency", "batch_size", "max_retries",
            "timeout", "response_format",
        )},
        "api_key_env": config.get("api_key_env"),
        "smoke_status": smoke.get("status"),
        "smoke_returned_model_ids": smoke.get("model_identifiers_returned"),
        "smoke_effort_parameter_supported": smoke.get("effort_parameter_check", {}).get("parameter_transmission_and_logging_pass"),
        "archived_runner_relocation": {
            "original_root_files_present": False,
            "full_runner_path": ARCHIVED_FULL_RUNNER.relative_to(REPO_ROOT).as_posix(),
            "full_runner_hash_matches_lock": bool(expected_full and sha_file(ARCHIVED_FULL_RUNNER) == expected_full),
            "smoke_runner_path": ARCHIVED_SMOKE_RUNNER.relative_to(REPO_ROOT).as_posix(),
            "smoke_runner_hash_matches_lock": bool(expected_smoke and sha_file(ARCHIVED_SMOKE_RUNNER) == expected_smoke),
        },
        "old_prompt_manifest_path": OLD_PROMPT_PATH.relative_to(REPO_ROOT).as_posix(),
        "old_prompt_hashes": {"system": prompt.get("system_prompt_hashes"), "user_template": prompt.get("user_template_hashes")},
        "old_plan_path": OLD_INPUT_PLAN_PATH.relative_to(REPO_ROOT).as_posix(),
        "old_plan_sha256": sha_file(OLD_INPUT_PLAN_PATH),
    }


def fact_preflight(cases: Sequence[dict], facts: Sequence[dict]) -> tuple[dict, list[dict], list[dict]]:
    cases_by_id = {row["case_id"]: row for row in cases}
    if len(cases_by_id) != len(cases):
        raise RuntimeError("Duplicate case_id in frozen case corpus")
    if len({row["case_id"] for row in facts}) != len(facts):
        raise RuntimeError("Duplicate case_id in frozen fact corpus")
    if set(cases_by_id) != {row["case_id"] for row in facts}:
        raise RuntimeError("Case/fact membership mismatch")

    qc_rows, salience_rows = [], []
    for fact in facts:
        case = cases_by_id[fact["case_id"]]
        ko, en = fact.get("neutral_fact_ko"), fact.get("neutral_fact_en")
        ko_placeholders = sorted(set(PLACEHOLDER_RE.findall(ko or "")))
        en_placeholders = sorted(set(PLACEHOLDER_RE.findall(en or "")))
        explicit_jurisdiction_hits = sorted(set(JURISDICTION_RE.findall((ko or "") + "\n" + (en or ""))))
        citations_ko = statute_citations(ko or "", "KR") + precedent_citations(ko or "", "KR")
        citations_en = statute_citations(en or "", "US") + precedent_citations(en or "", "US")
        cue_hits = sorted(set(match.group(0) for match in SALIENCE_CUE_RE.finditer((ko or "") + "\n" + (en or ""))))
        row = {
            "case_id": fact["case_id"], "origin_country": case.get("origin_country"),
            "origin_state": case.get("origin_state") or "", "analysis_split": case.get("analysis_split"),
            "case_domain": case.get("primary_domain") or case.get("case_domain") or "",
            "neutral_fact_ko_nonempty": bool(isinstance(ko, str) and ko.strip()),
            "neutral_fact_en_nonempty": bool(isinstance(en, str) and en.strip()),
            "neutral_fact_ko_hash_pass": bool(ko and sha_text(ko) == fact.get("neutral_fact_ko_sha256")),
            "neutral_fact_en_hash_pass": bool(en and sha_text(en) == fact.get("neutral_fact_en_sha256")),
            "translation_qc_status": fact.get("translation_equivalence_status") or "",
            "translation_qc_accepted": bool(fact.get("translation_equivalence_status")),
            "source_grounding_status": fact.get("source_grounding_status") or "",
            "source_grounding_accepted": bool(fact.get("source_grounding_status")),
            "placeholder_set_match": ko_placeholders == en_placeholders,
            "placeholder_set": "|".join(ko_placeholders),
            "explicit_jurisdiction_name_count": len(explicit_jurisdiction_hits),
            "frozen_parser_citation_count": len(citations_ko) + len(citations_en),
            "deterministic_legal_leakage_recheck_pass": not explicit_jurisdiction_hits and not citations_ko and not citations_en,
            "legal_leakage_qc_metadata_present": "legal_leakage_qc_status" in fact,
            "jurisdiction_salience_metadata_present": "jurisdiction_salience" in fact,
            "core_challenge_metadata_present": any(key in fact for key in ("core_challenge", "challenge_set", "challenge_type")),
            "unit_neutralization_metadata_present": "unit_neutralization_status" in fact,
            "institution_neutralization_metadata_present": "institution_neutralization_status" in fact,
            "potential_jurisdiction_salience_cue_count": len(cue_hits),
        }
        qc_rows.append(row)
        if cue_hits:
            salience_rows.append({"case_id": fact["case_id"], "cue_hits": "|".join(cue_hits), "metadata_present": row["jurisdiction_salience_metadata_present"]})

    summary = {
        "case_count": len(cases), "fact_count": len(facts),
        "country_counts": dict(sorted(Counter(row["origin_country"] for row in cases).items())),
        "split_counts": dict(sorted(Counter(row["analysis_split"] for row in cases).items())),
        "US_state_counts": dict(sorted(Counter(row.get("origin_state") or "" for row in cases if row["origin_country"] == "US").items())),
        "translation_status_counts": dict(sorted(Counter(row.get("translation_equivalence_status") or "MISSING" for row in facts).items())),
        "grounding_status_counts": dict(sorted(Counter(row.get("source_grounding_status") or "MISSING" for row in facts).items())),
        "nonempty_bilingual_pairs": sum(row["neutral_fact_ko_nonempty"] and row["neutral_fact_en_nonempty"] for row in qc_rows),
        "placeholder_consistency_pass": sum(row["placeholder_set_match"] for row in qc_rows),
        "input_hash_pass": sum(row["neutral_fact_ko_hash_pass"] and row["neutral_fact_en_hash_pass"] for row in qc_rows),
        "deterministic_legal_leakage_recheck_pass": sum(row["deterministic_legal_leakage_recheck_pass"] for row in qc_rows),
        "potential_salience_cue_cases": len(salience_rows),
        "explicit_legal_leakage_qc_metadata_cases": sum(row["legal_leakage_qc_metadata_present"] for row in qc_rows),
        "jurisdiction_salience_metadata_cases": sum(row["jurisdiction_salience_metadata_present"] for row in qc_rows),
        "core_challenge_metadata_cases": sum(row["core_challenge_metadata_present"] for row in qc_rows),
        "unit_neutralization_metadata_cases": sum(row["unit_neutralization_metadata_present"] for row in qc_rows),
        "institution_neutralization_metadata_cases": sum(row["institution_neutralization_metadata_present"] for row in qc_rows),
    }
    return summary, qc_rows, salience_rows


def prompt_manifest() -> tuple[dict, dict]:
    checklist = [{"element": element, "ko_present": True, "en_present": True, "symmetric": True} for element in PROMPT_CHECKLIST]
    combined = "\n".join(SYSTEM_PROMPTS.values()) + "\n" + "\n".join(USER_TEMPLATES.values())
    banned = sorted(set(match.group(0) for match in BANNED_PROMPT_RE.finditer(combined)))
    jurisdiction_names = sorted(set(JURISDICTION_RE.findall(combined)))
    manifest = {
        "prompt_version": "exp1-implicit-jurisdiction-court-opinion-v1-current-protocol",
        "status": "PROPOSED_BY_CURRENT_PROTOCOL_NOT_SENT",
        "system_prompts": SYSTEM_PROMPTS, "user_templates": USER_TEMPLATES,
        "system_prompt_sha256": {language: sha_text(text) for language, text in SYSTEM_PROMPTS.items()},
        "user_template_sha256": {language: sha_text(text) for language, text in USER_TEMPLATES.items()},
        "fixed_numbered_headings_required": False, "IRAC_present": False,
        "specific_jurisdiction_names_present": bool(jurisdiction_names),
        "banned_instruction_hits": banned,
    }
    symmetry = {"all_required_elements_symmetric": all(row["symmetric"] for row in checklist), "checklist": checklist}
    return manifest, symmetry


def build_request_manifest(cases: Sequence[dict], facts: Sequence[dict], existing: dict, prompt: dict) -> list[dict]:
    facts_by_id = {row["case_id"]: row for row in facts}
    primary_cases = [row for row in cases if row.get("analysis_split") == PRIMARY_SPLIT]
    if len(primary_cases) != EXPECTED_PRIMARY_CASES:
        raise RuntimeError(f"Expected {EXPECTED_PRIMARY_CASES} confirmatory cases, found {len(primary_cases)}")
    if existing["configured_models"] != [PRIMARY_MODEL_ID]:
        raise RuntimeError(f"Frozen baseline model mismatch: {existing['configured_models']}")
    if tuple(existing["reasoning_efforts"]) != PRIMARY_EFFORTS:
        raise RuntimeError(f"Frozen reasoning-effort mismatch: {existing['reasoning_efforts']}")
    rows = []
    index = 0
    for case in sorted(primary_cases, key=lambda row: row["case_id"]):
        fact = facts_by_id[case["case_id"]]
        for language in ("ko", "en"):
            for effort in PRIMARY_EFFORTS:
                for replicate in range(1, PRIMARY_REPLICATES + 1):
                    index += 1
                    neutral_fact = fact[f"neutral_fact_{language}"]
                    system = SYSTEM_PROMPTS[language]
                    user = USER_TEMPLATES[language].format(neutral_fact=neutral_fact)
                    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                    request_id = sha_text(f"{EXPERIMENT_VERSION}|{case['case_id']}|{language}|{PRIMARY_MODEL_ID}|{effort}|{replicate}")
                    rows.append({
                        "request_index": index, "request_id": request_id, "case_id": case["case_id"],
                        "origin_country": case["origin_country"], "origin_state": case.get("origin_state") or "",
                        "case_domain": case.get("primary_domain") or case.get("case_domain") or "",
                        "analysis_split": case["analysis_split"], "jurisdiction_salience": "MISSING",
                        "input_language": language.upper(), "provider": PRIMARY_PROVIDER,
                        "model_family": NOT_APPLICABLE_SINGLE_MODEL, "model_id": PRIMARY_MODEL_ID, "reasoning_effort": effort,
                        "replicate": replicate, "neutral_fact_hash": sha_text(neutral_fact),
                        "prompt_hash": sha_text(system + "\n" + USER_TEMPLATES[language]),
                        "input_hash": sha_text(canonical_json(messages)), "randomized_order": None,
                    })
    if len(rows) != EXPECTED_REQUESTS:
        raise RuntimeError(f"Expected {EXPECTED_REQUESTS} requests, built {len(rows)}")
    rng = random.Random(REQUEST_ORDER_SEED)
    rng.shuffle(rows)
    for order, row in enumerate(rows, 1):
        row["randomized_order"] = order
    hashes = {}
    for row in rows:
        key = (row["case_id"], row["input_language"], row["model_id"], row["reasoning_effort"])
        hashes.setdefault(key, set()).add(row["input_hash"])
    if any(len(values) != 1 for values in hashes.values()):
        raise RuntimeError("Replicates do not share canonical input hashes")
    return rows


def find_orientation_definition() -> list[str]:
    hits = []
    roots = [REPO_ROOT / "scripts", REPO_ROOT / "legal_lexicon"]
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path == Path(__file__).resolve() or path.suffix.lower() not in {".py", ".json", ".md", ".csv"}:
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
            except (UnicodeDecodeError, OSError):
                continue
            if "country_orientation_score" in text:
                hits.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(hits)


def gate2_audit() -> dict:
    report = GATE2_REPORT.read_text(encoding="utf-8")
    if "B. Suitable only with restricted feature space" in report:
        grade = "B"
    elif "A. Suitable for primary LLM projection" in report:
        grade = "A"
    elif "C. Not suitable" in report:
        grade = "C"
    else:
        grade = "UNKNOWN"
    restricted = json.loads(GATE2_RESTRICTED.read_text(encoding="utf-8"))
    return {
        "grade": grade, "report_path": GATE2_REPORT.relative_to(REPO_ROOT).as_posix(),
        "report_sha256": sha_file(GATE2_REPORT),
        "primary_projection_feature_space": restricted.get("features") if grade == "B" else None,
        "restricted_transformation_path": GATE2_RESTRICTED.relative_to(REPO_ROOT).as_posix(),
        "restricted_transformation_sha256": sha_file(GATE2_RESTRICTED),
        "procedure_primary_projection_excluded": grade == "B",
    }


def blockers_for(prompt: dict, symmetry: dict, final_qc_gate: dict) -> list[dict]:
    blockers = []
    if final_qc_gate.get("status") != "PREFLIGHT_PASSED_READY_FOR_FULL_RUN":
        blockers.append({
            "code": "FINAL_QC_GATE_NOT_PASSED",
            "detail": f"Final provenance-aware QC gate status is {final_qc_gate.get('status')}.",
        })
    if prompt["banned_instruction_hits"] or prompt["specific_jurisdiction_names_present"] or not symmetry["all_required_elements_symmetric"]:
        blockers.append({"code": "PROMPT_PREFLIGHT_FAILED", "detail": "Prompt symmetry or prohibited-instruction check failed."})
    return blockers


def audit(output_dir: Path) -> dict:
    verify_required_files()
    corpus_hashes = verify_corpus_hashes()
    cases, facts = read_jsonl(CASES_PATH), read_jsonl(FACTS_PATH)
    existing = existing_freeze_audit()
    fact_summary, _, _ = fact_preflight(cases, facts)
    primary_cases = [row for row in cases if row.get("analysis_split") == PRIMARY_SPLIT]
    primary_ids = {row["case_id"] for row in primary_cases}
    primary_facts = [row for row in facts if row["case_id"] in primary_ids]
    primary_summary, fact_rows, salience_rows = fact_preflight(primary_cases, primary_facts)
    prompt, symmetry = prompt_manifest()
    gate2 = gate2_audit()
    orientation_hits = find_orientation_definition()
    final_qc_gate = json.loads(FINAL_QC_GATE.read_text(encoding="utf-8"))
    blockers = blockers_for(prompt, symmetry, final_qc_gate)
    requests = build_request_manifest(cases, facts, existing, prompt)
    if len(requests) != EXPECTED_REQUESTS:
        raise RuntimeError(f"Final factorial assertion failed: {len(requests)} != {EXPECTED_REQUESTS}")

    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite pre-run audit directory: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "prompt_manifest.json", prompt)
    write_json(output_dir / "prompt_symmetry_audit.json", symmetry)
    qc_columns = list(fact_rows[0])
    write_csv(output_dir / "case_preflight_qc.csv", fact_rows, qc_columns)
    write_csv(output_dir / "potential_salience_cues.csv", salience_rows, ["case_id", "cue_hits", "metadata_present"])
    primary_case_rows = [{
        "case_id": row["case_id"], "origin_country": row["origin_country"],
        "origin_state": row.get("origin_state") or "", "case_domain": row.get("primary_domain") or row.get("case_domain") or "",
        "analysis_split": row["analysis_split"],
    } for row in sorted(primary_cases, key=lambda row: row["case_id"])]
    write_csv(output_dir / "final_primary_case_manifest.csv", primary_case_rows, list(primary_case_rows[0]))
    request_columns = list(requests[0])
    write_csv(output_dir / "final_request_manifest.csv", requests, request_columns)

    environment = {
        "python_version": platform.python_version(), "platform": platform.platform(),
        "package_versions": package_versions(),
        "credential_environment": {
            "LETSUR_API_KEY": {"required_by_config": True, "configured": "LETSUR_API_KEY" in dotenv_variable_names() or bool(os.getenv("LETSUR_API_KEY"))},
        },
    }
    model_side_preflight = {
        "provider": PRIMARY_PROVIDER, "model_id": PRIMARY_MODEL_ID,
        "resolved_model_ids_from_previous_smoke": existing["smoke_returned_model_ids"],
        "reachable_through_letsur": existing["smoke_status"] == "PASS" and PRIMARY_MODEL_ID in (existing["smoke_returned_model_ids"] or []),
        "reasoning_efforts": list(PRIMARY_EFFORTS),
        "effort_parameters_previously_verified": bool(existing["smoke_effort_parameter_supported"]),
        "credential_env": "LETSUR_API_KEY",
        "credential_available": environment["credential_environment"]["LETSUR_API_KEY"]["configured"],
        "exact_model_config_frozen": existing["configured_models"] == [PRIMARY_MODEL_ID],
    }
    model_side_preflight["pass"] = all((
        model_side_preflight["reachable_through_letsur"],
        model_side_preflight["effort_parameters_previously_verified"],
        model_side_preflight["credential_available"],
        model_side_preflight["exact_model_config_frozen"],
    ))
    if not model_side_preflight["pass"]:
        blockers.append({"code": "MODEL_SIDE_PREFLIGHT_FAILED", "detail": "Letsur/model/effort/credential frozen checks did not all pass."})
    write_json(output_dir / "environment.json", environment)

    input_audit = {
        "git_commit": git_value("rev-parse", "HEAD"), "git_branch": git_value("branch", "--show-current"),
        "corpus": fact_summary, "primary_population": primary_summary,
        "case_manifest_path": CASES_PATH.relative_to(REPO_ROOT).as_posix(),
        "fact_manifest_path": FACTS_PATH.relative_to(REPO_ROOT).as_posix(),
        "frozen_input_hashes": corpus_hashes["actual"], "existing_generation_freeze": existing,
        "gate2": gate2, "frozen_orientation_definition_hits": orientation_hits,
        "frozen_lexicon": {"path": LEXICON_DIR.relative_to(REPO_ROOT).as_posix(), "run_manifest_sha256": sha_file(LEXICON_DIR / "run_manifest.json")},
        "frozen_orientation_lexicon": {"path": ORIENTATION_LEXICON_DIR.relative_to(REPO_ROOT).as_posix(), "run_manifest_sha256": sha_file(ORIENTATION_LEXICON_DIR / "run_manifest.json")},
        "final_pre_generation_qc": {"path": FINAL_QC_DIR.relative_to(REPO_ROOT).as_posix(), "gate_sha256": sha_file(FINAL_QC_GATE), "status": final_qc_gate["status"]},
        "frozen_parser": {"path": PARSER_PATH.relative_to(REPO_ROOT).as_posix(), "sha256": sha_file(PARSER_PATH)},
        "previous_smoke_outputs": {"path": SMOKE_SUMMARY_PATH.parent.relative_to(REPO_ROOT).as_posix(), "status": existing["smoke_status"]},
        "model_side_preflight": model_side_preflight,
    }
    write_json(output_dir / "input_audit.json", input_audit)

    manifest = {
        "experiment_name": EXPERIMENT_NAME, "experiment_version": EXPERIMENT_VERSION,
        "status": "PREFLIGHT_PASSED_READY_FOR_FULL_RUN" if not blockers else "PREFLIGHT_BLOCKED_NO_API_REQUESTS_SENT", "git_commit": input_audit["git_commit"],
        "case_manifest_path": input_audit["case_manifest_path"], "case_manifest_sha256": sha_file(CASES_PATH),
        "fact_manifest_path": input_audit["fact_manifest_path"], "fact_manifest_sha256": sha_file(FACTS_PATH),
        "prompt_manifest_sha256": sha_file(output_dir / "prompt_manifest.json"),
        "KO_prompt_sha256": sha_text(SYSTEM_PROMPTS["ko"] + "\n" + USER_TEMPLATES["ko"]),
        "EN_prompt_sha256": sha_text(SYSTEM_PROMPTS["en"] + "\n" + USER_TEMPLATES["en"]),
        "study_design": "single-model confirmatory baseline",
        "primary_population": PRIMARY_SPLIT,
        "primary_population_manifest_path": "final_primary_case_manifest.csv",
        "primary_population_manifest_sha256": sha_file(output_dir / "final_primary_case_manifest.csv"),
        "factorial_assertion": {
            "n_cases": EXPECTED_PRIMARY_CASES, "n_languages": 2, "n_models": 1,
            "n_efforts": len(PRIMARY_EFFORTS), "n_replicates": PRIMARY_REPLICATES,
            "expected_requests": EXPECTED_REQUESTS, "assertion_pass": len(requests) == EXPECTED_REQUESTS,
        },
        "configured_provider_names": [PRIMARY_PROVIDER], "configured_model_ids": [PRIMARY_MODEL_ID],
        "reasoning_effort_conditions": list(PRIMARY_EFFORTS), "replicate_count": PRIMARY_REPLICATES,
        "generation_parameters": existing["generation_parameters"],
        "randomization_seed": REQUEST_ORDER_SEED, "request_ordering_seed": REQUEST_ORDER_SEED,
        "aggregation_rule": {"unit": "case x language x effort", "continuous": "mean across replicates", "binary": "proportion across replicates"},
        "orientation_difference_sign": "EN minus KO; positive means relatively more US-oriented",
        "primary_hypotheses": {
            "H1": "Within GPT-5.6, English input produces a more U.S.-oriented country_orientation_score than Korean input for the same factual case.",
            "H2": "Korean input produces more KR-specific doctrine/authority markers than English input.",
            "H3": "The KO-EN language-effect magnitude may differ across low, medium, and high reasoning effort; no directional effort hypothesis is registered.",
            "H4": "deferred",
        },
        "primary_mixed_model_specification": "orientation_score ~ input_language * reasoning_effort + origin_country + case_domain + jurisdiction_salience + lower_court_supplemented + (1 | case_id)",
        "covariate_inclusion_rule": "Include a listed covariate only if its metadata exists and was defined before generation; source state may be a fixed effect in the U.S.-origin subset.",
        "reasoning_effort_analysis_role": "within-model robustness analysis",
        "multi_model_replication_status": "deferred_to_future_robustness",
        "model_family_interaction_status": NOT_APPLICABLE_SINGLE_MODEL,
        "cross_model_replication_status": NOT_APPLICABLE_SINGLE_MODEL,
        "model_family_comparison_status": NOT_APPLICABLE_SINGLE_MODEL,
        "provider_comparison_status": NOT_APPLICABLE_SINGLE_MODEL,
        "reporting_scope": "All conclusions must be scoped to the tested GPT-5.6 conditions; no cross-model or general LLM claim is permitted.",
        "model_side_preflight": model_side_preflight,
        "lexicon_version_hash": input_audit["frozen_lexicon"],
        "orientation_lexicon_version_hash": input_audit["frozen_orientation_lexicon"],
        "final_pre_generation_qc": input_audit["final_pre_generation_qc"],
        "parser_version_hash": input_audit["frozen_parser"],
        "reference_projection": gate2, "python_version": environment["python_version"],
        "package_versions": environment["package_versions"], "generation_start_timestamp": None,
        "known_candidate_request_count": len(requests),
        "expected_number_of_requests": EXPECTED_REQUESTS,
        "api_requests_sent": 0, "blockers": blockers,
    }
    write_json(output_dir / "final_experiment_manifest.json", manifest)
    write_json(output_dir / "blockers.json", {"status": manifest["status"], "blockers": blockers})

    lines = [
        "# Experiment 1 pre-run audit", "", f"Status: **{manifest['status']}**", "",
        "No API request was sent.", "", "## Repository truth", "",
        f"- Frozen cases: {len(cases)} (KR {fact_summary['country_counts'].get('KR', 0)}, US {fact_summary['country_counts'].get('US', 0)})",
        f"- Split: confirmatory {fact_summary['split_counts'].get('confirmatory', 0)}, development {fact_summary['split_counts'].get('development', 0)}",
        f"- US states: {fact_summary['US_state_counts']}",
        f"- Bilingual facts nonempty/hash-valid/placeholders-consistent: {fact_summary['nonempty_bilingual_pairs']}/{fact_summary['input_hash_pass']}/{fact_summary['placeholder_consistency_pass']} of {len(cases)}",
        f"- Deterministic jurisdiction-name/citation leakage recheck: {fact_summary['deterministic_legal_leakage_recheck_pass']}/{len(cases)} pass",
        f"- Potential salience-cue cases requiring metadata: {fact_summary['potential_salience_cue_cases']}", "",
        "## Frozen single-model confirmatory design", "",
        f"- Provider/model: {PRIMARY_PROVIDER} / {PRIMARY_MODEL_ID}",
        f"- Primary population: {len(primary_cases)} confirmatory cases; 40 development cases excluded",
        f"- Efforts: {', '.join(PRIMARY_EFFORTS)}; previous smoke transmission support: {existing['smoke_effort_parameter_supported']}",
        f"- Replicates: {PRIMARY_REPLICATES}",
        f"- Factorial assertion: {len(primary_cases)} x 2 x 1 x {len(PRIMARY_EFFORTS)} x {PRIMARY_REPLICATES} = {len(requests)} requests",
        f"- Model-side preflight: {'PASS' if model_side_preflight['pass'] else 'FAIL'}",
        "- Multi-model replication: deferred_to_future_robustness", "",
        "## Gate 2", "", f"Grade {gate2['grade']}; primary projection must use the six-feature restricted transformation without procedure. Generation prompts are unaffected.", "",
        "## Blocking conditions", "",
    ]
    lines.extend(f"- `{row['code']}`: {row['detail']}" for row in blockers)
    lines += ["", "The prompt, cases, measurement definitions, and factor settings were not changed in response to generated results; no generated results exist for this run.", ""]
    (output_dir / "PRE_RUN_AUDIT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    artifact_hashes = {
        path.name: sha_file(path) for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "final_experiment_manifest.json"}
    }
    manifest["artifact_sha256"] = artifact_hashes
    write_json(output_dir / "final_experiment_manifest.json", manifest)
    checksum_paths = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha_file(path)}  {path.name}\n" for path in checksum_paths), encoding="ascii", newline="\n"
    )
    print(canonical_json({"status": manifest["status"], "blocker_codes": [row["code"] for row in blockers], "candidate_requests": len(requests)}))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Execute only if preflight is complete (currently unavailable).")
    args = parser.parse_args()
    if args.execute:
        raise RuntimeError("Full execution is unavailable until a completed, blocker-free frozen manifest exists. No API request sent.")
    audit(args.output_dir)


if __name__ == "__main__":
    main()
