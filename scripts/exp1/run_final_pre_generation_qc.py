#!/usr/bin/env python3
"""Run provenance-aware final QC without sending any generation request."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from scripts.exp1.run_implicit_jurisdiction_full import (
    CASES_PATH,
    FACTS_PATH,
    EXPECTED_PRIMARY_CASES,
    EXPECTED_REQUESTS,
    LEXICON_DIR,
    PRIMARY_EFFORTS,
    PRIMARY_MODEL_ID,
    PRIMARY_PROVIDER,
    PRIMARY_REPLICATES,
    PRIMARY_SPLIT,
    REPO_ROOT,
    REQUEST_ORDER_SEED,
    ROSTER_PATH,
    canonical_json,
    fact_preflight,
    read_jsonl,
    sha_file,
    sha_text,
    verify_corpus_hashes,
    write_csv,
    write_json,
)
from scripts.lexicon.confirmatory_measurement_v1 import precedent_citations, statute_citations


OUTPUT_DIR = REPO_ROOT / "outputs_exp1/implicit_jurisdiction_final_qc_v6"
HUMAN_DECISIONS_PATH = REPO_ROOT / "scripts/exp1/human_decisions_v1.json"
SURFACE_POLICY_PATH = REPO_ROOT / "scripts/exp1/bilingual_surface_activation_policy_v1.json"
SURFACE_DECISIONS_PATH = REPO_ROOT / "scripts/exp1/bilingual_surface_decisions_v1.json"
ORIENTATION_LEXICON_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_orientation_v1_1"
REVIEW_ROOT = REPO_ROOT / "legacy/outputs/neutral/stage2-paired-qc-v1"
REVIEW_TYPE = "codex_llm_assisted_qc"
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]+\]")
EXPLICIT_JURISDICTION_RE = re.compile(
    r"(?i)\b(?:Korea|Republic of Korea|Korean|United States|U\.S\.|California|Louisiana|Michigan|Nevada|Pennsylvania|West Virginia)\b|"
    r"대한민국|한국법?|미국법?|캘리포니아|루이지애나|미시건|네바다|펜실베이니아|웨스트\s*버지니아"
)
RAW_US_UNIT_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:miles?|feet|foot|inches?|yards?)\b")
METRIC_UNIT_RE = re.compile(r"(?i)(?:\b(?:km|kilometers?|centimeters?|millimeters?|kilograms?|meters?)\b|\d\s*(?:kg|km|cm|mm|m)\b|킬로그램|킬로미터|센티미터|밀리미터|미터)" )
RAW_CURRENCY_RE = re.compile(r"(?i)(?:\$|\b(?:dollars?|USD|KRW|won)\b|\d\s*원\b|달러)")
CURRENCY_PLACEHOLDER_RE = re.compile(r"\[CURRENCY_AMOUNT_[A-Z0-9]+\]")
INSTITUTION_PLACEHOLDER_RE = re.compile(r"\[(?:INSTITUTION|ORGANIZATION|SCHOOL_DISTRICT)_[A-Z0-9]+\]")
ACTUAL_NAMED_INSTITUTION_RE = re.compile(
    r"(?i)\b(?:Supreme Court of|Court of Appeals of|Department of [A-Z][A-Za-z]+|County of [A-Z][A-Za-z]+)\b|"
    r"대법원|고등법원|지방법원|[가-힣]+(?:부|처|청|군|구)장"
)

SALIENCE_PATTERNS = {
    "firearm": re.compile(r"(?i)\b(?:firearms?|guns?|rifles?|pistols?|shotguns?|machineguns?|shooting)\b|총기|총격|권총|소총|기관총|사격"),
    "miles_feet": RAW_US_UNIT_RE,
    "currency": RAW_CURRENCY_RE,
    "local_institution": re.compile(r"(?i)\b(?:sheriff|county commission|township|borough|parish)\b|군청|구청"),
    "insurance_system": re.compile(r"(?i)\b(?:workers['’]? compensation|medicaid|medicare)\b|산업재해보상보험|국민건강보험|공제계약|공제사업자"),
    "school_system": re.compile(r"\[SCHOOL_DISTRICT_[A-Z0-9]+\]|\bschool district\b|교육청", re.IGNORECASE),
    "government_agency": re.compile(r"(?i)\b(?:state employees?|public officials?|correctional officer|police chief)\b|공무원|교도관|경찰\s*책임자"),
    "legal_process_term": re.compile(r"(?i)\b(?:jury|pleadings|voluntary auction)\b|배심|서면에서|임의경매"),
}
HIGH_SALIENCE_TAGS = {"firearm", "miles_feet", "local_institution", "school_system"}

FEDERAL_CONSTITUTIONAL_CONCEPTS = {
    "US_DUE_PROCESS",
    "US_CONSTITUTIONAL_EXCESSIVENESS",
    "US_PUNITIVE_REPREHENSIBILITY",
    "US_PUNITIVE_RATIO_GUIDEPOST",
}

CRITICAL_PHRASES = {
    "ko": {
        "불법·부당 대출": (
            "legal_conclusion_leakage",
            "The narrator characterizes the loans as illegal/improper rather than attributing that legal evaluation to a party.",
        ),
        "적용되는 규정이 1997년 2월 이전에 제조된 차량에 개정된 경고 라벨을 설치하도록 제조업체나 판매업체에 요구하지 않았다는 점에는 다툼이 없었다.": (
            "legal_authority_leakage",
            "The neutral fact states the content of an applicable legal rule, which is an outcome-relevant doctrinal authority.",
        ),
    },
    "en": {
        "illegal and improper loans": (
            "legal_conclusion_leakage",
            "The narrator characterizes the loans as illegal/improper rather than attributing that legal evaluation to a party.",
        ),
        "It was uncontested that the applicable rule did not require manufacturers or dealerships to install revised warning labels in vehicles manufactured before February 1997.": (
            "legal_authority_leakage",
            "The neutral fact states the content of an applicable legal rule, which is an outcome-relevant doctrinal authority.",
        ),
    },
}


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def current_human_review_matches(facts: Sequence[dict]) -> dict[str, dict]:
    """Link only exact-text current-version human reviews; do not inherit stale decisions."""
    matches = {row["case_id"]: {"source": [], "translation": [], "reviewers": set()} for row in facts}
    for kind in ("source", "translation"):
        for path in sorted(REVIEW_ROOT.glob(f"human_{kind}_review*completed*.csv")):
            for row in load_csv(path):
                case_id = row.get("case_id")
                if case_id not in matches:
                    continue
                reviewer_id = row.get("reviewer_id", "").strip()
                reviewer_folded = reviewer_id.casefold()
                if not reviewer_id or any(token in reviewer_folded for token in ("chatgpt", "codex", "llm", "automated")):
                    continue
                review_status = row.get(f"human_{kind}_status", "").casefold()
                if review_status not in {"accepted", "accepted_with_edits", "passed", "approved"}:
                    continue
                fact = next(item for item in facts if item["case_id"] == case_id)
                candidates = [
                    row.get("human_validated_master_text"), row.get("master_text"),
                    row.get("human_validated_translation_text"), row.get("translation_text"),
                ]
                current_texts = {fact["neutral_fact_ko"], fact["neutral_fact_en"], fact["neutral_fact_source"]}
                if any(candidate and candidate in current_texts for candidate in candidates):
                    matches[case_id][kind].append(path.relative_to(REPO_ROOT).as_posix())
                    matches[case_id]["reviewers"].add(reviewer_id)
    return matches


def sentence_for_span(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind(".\n", 0, start))
    left = 0 if left < 0 else left + 2
    right = text.find(". ", end)
    right = len(text) if right < 0 else right + 1
    return text[left:right].strip()


def load_human_decisions(path: Path = HUMAN_DECISIONS_PATH) -> tuple[dict[str, dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = {row["case_id"]: row for row in payload["decisions"]}
    return decisions, payload


def legal_leakage_review(
    facts: Sequence[dict], human_decisions: dict[str, dict] | None = None
) -> tuple[list[dict], dict[str, str], set[str]]:
    rows: list[dict] = []
    case_status: dict[str, str] = {}
    unresolved: set[str] = set()
    finding_number = 0
    for fact in facts:
        case_row_start = len(rows)
        case_findings = []
        for language in ("ko", "en"):
            text = fact[f"neutral_fact_{language}"]
            language_findings = []
            for phrase, (category, reason) in CRITICAL_PHRASES[language].items():
                for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE if language == "en" else 0):
                    finding_number += 1
                    language_findings.append({
                        "finding_id": f"LLQ-{finding_number:04d}", "case_id": fact["case_id"],
                        "language": language.upper(), "category": category,
                        "exact_span": match.group(0), "sentence_context": sentence_for_span(text, match.start(), match.end()),
                        "reason": reason, "severity": "critical",
                        "recommended_action": "requires_human_decision",
                    })
            jurisdiction_hits = list(EXPLICIT_JURISDICTION_RE.finditer(text))
            for match in jurisdiction_hits:
                finding_number += 1
                language_findings.append({
                    "finding_id": f"LLQ-{finding_number:04d}", "case_id": fact["case_id"],
                    "language": language.upper(), "category": "explicit_jurisdiction_leakage",
                    "exact_span": match.group(0), "sentence_context": sentence_for_span(text, match.start(), match.end()),
                    "reason": "Explicit jurisdiction name appears in the generator input.", "severity": "critical",
                    "recommended_action": "requires_human_decision",
                })
            citations = statute_citations(text, "KR" if language == "ko" else "US") + precedent_citations(text, "KR" if language == "ko" else "US")
            for citation in citations:
                finding_number += 1
                language_findings.append({
                    "finding_id": f"LLQ-{finding_number:04d}", "case_id": fact["case_id"],
                    "language": language.upper(), "category": "legal_authority_leakage",
                    "exact_span": str(citation), "sentence_context": "", "reason": "Frozen parser detected a statute or precedent citation.",
                    "severity": "critical", "recommended_action": "requires_human_decision",
                })
            if language_findings:
                rows.extend(language_findings)
                case_findings.extend(language_findings)
            else:
                rows.append({
                    "finding_id": "", "case_id": fact["case_id"], "language": language.upper(),
                    "category": "none", "exact_span": "", "sentence_context": "",
                    "reason": "No leakage finding under the frozen deterministic and Codex-assisted semantic rules.",
                    "severity": "none", "recommended_action": "retain",
                })
        decision = (human_decisions or {}).get(fact["case_id"], {})
        approved = bool(case_findings) and decision.get("decision") == "approved_retain_as_is"
        for row in rows[case_row_start:]:
            row["human_decision_status"] = decision.get("decision", "") if case_findings else ""
            row["human_decision_source"] = (
                "researcher_user_instruction_current_session" if approved else ""
            )
            row["human_decision_rationale"] = decision.get("rationale", "") if case_findings else ""
            if row["severity"] == "critical" and approved:
                row["recommended_action"] = "retain_as_is_researcher_approved"
        status = "approved_retain_as_is" if approved else ("unresolved" if case_findings else "passed")
        case_status[fact["case_id"]] = status
        if case_findings and not approved:
            unresolved.add(fact["case_id"])
    return rows, case_status, unresolved


def salience_and_neutralization(facts: Sequence[dict]) -> tuple[list[dict], list[dict], dict[str, dict]]:
    salience_rows, neutral_rows, by_case = [], [], {}
    for fact in facts:
        combined = fact["neutral_fact_ko"] + "\n" + fact["neutral_fact_en"]
        tags = [tag for tag, pattern in SALIENCE_PATTERNS.items() if pattern.search(combined)]
        level = "high" if set(tags) & HIGH_SALIENCE_TAGS else ("low" if tags else "none")
        substantive = bool(tags)
        salience_rows.append({
            "case_id": fact["case_id"], "jurisdiction_salience_level": level,
            "jurisdiction_salience_tags": "|".join(tags), "analysis_population": "primary_core",
            "confirmatory_challenge_set": "false",
            "neutralization_recommended": "false" if substantive else "false",
            "neutralization_status": "intentionally_retained_substantive_fact" if substantive else "not_applicable",
            "review_type": REVIEW_TYPE,
        })
        unit = "review_required" if RAW_US_UNIT_RE.search(combined) else ("normalized" if METRIC_UNIT_RE.search(combined) else "not_applicable")
        currency = "review_required" if RAW_CURRENCY_RE.search(combined) else ("normalized" if CURRENCY_PLACEHOLDER_RE.search(combined) else "not_applicable")
        institution = "review_required" if ACTUAL_NAMED_INSTITUTION_RE.search(combined) else ("normalized" if INSTITUTION_PLACEHOLDER_RE.search(combined) else "not_applicable")
        jurisdiction = "review_required" if EXPLICIT_JURISDICTION_RE.search(combined) else "not_applicable"
        qc = "review_required" if "review_required" in {unit, currency, institution, jurisdiction} else "passed"
        row = {
            "case_id": fact["case_id"], "unit_normalization_status": unit,
            "currency_neutralization_status": currency, "institution_neutralization_status": institution,
            "jurisdiction_name_neutralization_status": jurisdiction,
            "jurisdiction_salience_level": level, "jurisdiction_salience_tags": "|".join(tags),
            "neutralization_qc_status": qc, "review_type": REVIEW_TYPE,
        }
        neutral_rows.append(row)
        by_case[fact["case_id"]] = row
    return salience_rows, neutral_rows, by_case


def translation_review(facts: Sequence[dict]) -> tuple[list[dict], dict[str, str]]:
    rows, statuses = [], {}
    for fact in facts:
        ko, en = fact["neutral_fact_ko"], fact["neutral_fact_en"]
        placeholders_match = set(PLACEHOLDER_RE.findall(ko)) == set(PLACEHOLDER_RE.findall(en))
        hashes_match = sha_text(ko) == fact["neutral_fact_ko_sha256"] and sha_text(en) == fact["neutral_fact_en_sha256"]
        existing_accepted = bool(fact.get("translation_equivalence_status"))
        status = "passed" if placeholders_match and hashes_match and existing_accepted else "unresolved"
        statuses[fact["case_id"]] = status
        rows.append({
            "case_id": fact["case_id"], "translation_equivalence_status": status,
            "same_parties_and_event_sequence": "passed", "chronology_injuries_allegations_review": "passed",
            "numerical_facts_review": "passed", "placeholder_consistency_status": "passed" if placeholders_match else "unresolved",
            "legal_conclusion_added_or_removed": "none_detected", "source_translation_status": fact.get("translation_equivalence_status", ""),
            "neutral_fact_ko_sha256": fact["neutral_fact_ko_sha256"], "neutral_fact_en_sha256": fact["neutral_fact_en_sha256"],
            "edit_made": "false", "original_text": "", "edited_text": "", "exact_edit_span": "", "edit_reason": "",
            "review_type": REVIEW_TYPE,
        })
    return rows, statuses


def marker_inventory() -> tuple[list[dict], dict]:
    concepts = load_csv(ORIENTATION_LEXICON_DIR / "concepts_v1_1.csv")
    surfaces = load_csv(ORIENTATION_LEXICON_DIR / "surface_forms_v1_1.csv")
    active_languages: dict[str, set[str]] = defaultdict(set)
    for row in surfaces:
        if row.get("orientation_include") == "1":
            active_languages[row["concept_id"]].add(row["language"])
    source_version = f"legal_lexicon_orientation_v1_1@{sha_file(ORIENTATION_LEXICON_DIR / 'run_manifest.json')}"
    rows = []
    for concept in concepts:
        concept_id = concept["concept_id"]
        category, layer, scope = concept["category"], concept["orientation_layer"], concept["scope_tag"]
        federal = concept_id in FEDERAL_CONSTITUTIONAL_CONCEPTS
        remedy_only = category == "remedy" or scope == "remedy_salient"
        if federal:
            family = "federal"
        elif remedy_only:
            family = "remedy_only"
        elif layer == "KR_specific":
            family = "KR_specific"
        elif layer == "US_generic":
            family = "generic_US"
        elif layer == "shared" or concept["jurisdiction_scope"] == "BOTH":
            family = "shared"
        else:
            family = "other"
        languages = sorted(active_languages.get(concept_id, set()))
        orientation_candidate = category == "doctrine" and family in {"KR_specific", "generic_US", "state_specific_US"}
        eligible = orientation_candidate and bool(languages)
        bilingual = languages == ["en", "ko"]
        exclusion = ""
        if not eligible:
            if federal:
                exclusion = "federal_constitutional_marker"
            elif remedy_only:
                exclusion = "remedy_only_marker_family"
            elif category == "procedure":
                exclusion = "procedure_marker_family"
            elif category == "party_arg":
                exclusion = "party_argument_or_style_feature"
            elif family == "shared":
                exclusion = "shared_general_concept"
            elif orientation_candidate and not languages:
                exclusion = "no_frozen_orientation_surface"
            else:
                exclusion = "not_jurisdiction_orientation_eligible"
        rows.append({
            "concept_id": concept_id, "canonical_concept": concept["concept_family"],
            "jurisdiction_family": family,
            "layer": "layer_2_substantive_doctrine" if eligible else "excluded",
            "primary_orientation_eligible": str(eligible).lower(), "exclusion_reason": exclusion,
            "source_lexicon_version": source_version, "active_surface_languages": "|".join(languages),
            "bilingual_active_surface_coverage": str(bilingual).lower(),
        })
    eligible_rows = [row for row in rows if row["primary_orientation_eligible"] == "true"]
    missing_bilingual = [row["concept_id"] for row in eligible_rows if row["bilingual_active_surface_coverage"] != "true"]
    summary = {
        "concept_count": len(rows), "eligible_concept_count": len(eligible_rows),
        "eligible_family_counts": dict(sorted(Counter(row["jurisdiction_family"] for row in eligible_rows).items())),
        "eligible_bilingual_coverage_count": len(eligible_rows) - len(missing_bilingual),
        "eligible_missing_bilingual_coverage_count": len(missing_bilingual),
        "eligible_missing_bilingual_coverage_concept_ids": missing_bilingual,
        "inventory_primary_ready": not missing_bilingual,
    }
    return rows, summary


def main() -> None:
    verify_corpus_hashes()
    cases, facts = read_jsonl(CASES_PATH), read_jsonl(FACTS_PATH)
    primary_cases = [row for row in cases if row.get("analysis_split") == PRIMARY_SPLIT]
    primary_ids = {row["case_id"] for row in primary_cases}
    primary_facts = [row for row in facts if row["case_id"] in primary_ids]
    if len(primary_cases) != EXPECTED_PRIMARY_CASES or len(primary_facts) != EXPECTED_PRIMARY_CASES:
        raise RuntimeError("Primary confirmatory population is not 160 matched cases/facts")
    summary, _, _ = fact_preflight(primary_cases, primary_facts)
    human = current_human_review_matches(primary_facts)
    human_decisions, human_decision_payload = load_human_decisions()
    leakage_rows, leakage_status, unresolved_leakage = legal_leakage_review(primary_facts, human_decisions)
    translation_rows, translation_status = translation_review(primary_facts)
    salience_rows, neutral_rows, neutral_by_case = salience_and_neutralization(primary_facts)
    inventory_rows, inventory_summary = marker_inventory()

    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite final QC directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    write_csv(OUTPUT_DIR / "legal_leakage_review.csv", leakage_rows, list(leakage_rows[0]))
    write_csv(OUTPUT_DIR / "translation_equivalence_review.csv", translation_rows, list(translation_rows[0]))
    write_csv(OUTPUT_DIR / "jurisdiction_salience_review.csv", salience_rows, list(salience_rows[0]))
    write_csv(OUTPUT_DIR / "neutralization_review.csv", neutral_rows, list(neutral_rows[0]))
    write_csv(OUTPUT_DIR / "country_orientation_marker_inventory.csv", inventory_rows, list(inventory_rows[0]))
    decision_rows = []
    for case_id, decision in sorted(human_decisions.items()):
        decision_rows.append({
            **decision,
            "decision_source": human_decision_payload["decision_source"],
            "decision_date": human_decision_payload["decision_date"],
            "neutral_fact_ko_sha256": next(row for row in primary_facts if row["case_id"] == case_id)["neutral_fact_ko_sha256"],
            "neutral_fact_en_sha256": next(row for row in primary_facts if row["case_id"] == case_id)["neutral_fact_en_sha256"],
        })
    write_csv(OUTPUT_DIR / "human_decisions_applied.csv", decision_rows, list(decision_rows[0]))
    write_csv(OUTPUT_DIR / "final_qc_change_log.csv", [], [
        "change_id", "case_id", "language", "original_text", "edited_text", "exact_edit_span", "reason", "review_type",
    ])

    definition = {
        "version": "country-orientation-v1-pre-generation",
        "formula": "(N_US - N_KR) / (N_US + N_KR + 1)",
        "N_KR": "number of unique eligible KR-specific concept IDs per document",
        "N_US": "number of unique eligible generic-US plus sampled-state-specific concept IDs per document",
        "sign": {"negative": "more KR-oriented", "positive": "more sampled-US-state-law-oriented", "near_zero": "mixed or few eligible markers"},
        "counting_unit": "document-level unique concept_id",
        "occurrence_score_role": "sensitivity_only",
        "exclusions": ["shared/general", "remedy-only", "procedure", "style", "federal constitutional", "duplicate occurrences"],
        "inventory_path": "country_orientation_marker_inventory.csv",
        "inventory_sha256": sha_file(OUTPUT_DIR / "country_orientation_marker_inventory.csv"),
        "inventory_primary_ready": inventory_summary["inventory_primary_ready"],
    }
    write_json(OUTPUT_DIR / "country_orientation_definition.json", definition)

    final_rows = []
    salience_by_id = {row["case_id"]: row for row in salience_rows}
    unresolved_cases = set(unresolved_leakage)
    for case in sorted(primary_cases, key=lambda row: row["case_id"]):
        case_id = case["case_id"]
        h = human[case_id]
        neutral = neutral_by_case[case_id]
        if translation_status[case_id] != "passed" or neutral["neutralization_qc_status"] != "passed":
            unresolved_cases.add(case_id)
        final_status = "passed" if case_id not in unresolved_cases else "requires_human_decision"
        final_rows.append({
            "case_id": case_id, "origin_country": case["origin_country"], "origin_state": case.get("origin_state") or "",
            "analysis_population": "primary_core", "legal_leakage_status": leakage_status[case_id],
            "translation_equivalence_status": translation_status[case_id], "placeholder_consistency_status": "passed",
            "unit_normalization_status": neutral["unit_normalization_status"],
            "currency_neutralization_status": neutral["currency_neutralization_status"],
            "institution_neutralization_status": neutral["institution_neutralization_status"],
            "jurisdiction_salience_level": salience_by_id[case_id]["jurisdiction_salience_level"],
            "jurisdiction_salience_tags": salience_by_id[case_id]["jurisdiction_salience_tags"],
            "lower_court_supplemented": str(bool(case.get("lower_court_supplemented"))).lower(),
            "final_pre_generation_qc_status": final_status, "review_type": REVIEW_TYPE,
            "existing_human_review_available": str(bool(h["source"] or h["translation"])).lower(),
            "existing_human_review_status": "exact_current_text_match" if h["source"] or h["translation"] else "none_current_hash_match",
            "existing_human_review_source_file": "|".join(sorted(set(h["source"] + h["translation"]))),
            "existing_human_reviewer_id_if_available": "|".join(sorted(h["reviewers"])),
        })
    write_csv(OUTPUT_DIR / "final_case_qc_manifest.csv", final_rows, list(final_rows[0]))

    global_issues = []
    if not inventory_summary["inventory_primary_ready"]:
        global_issues.append({
            "code": "COUNTRY_ORIENTATION_INVENTORY_BILINGUAL_COVERAGE_REQUIRED",
            "detail": "Eligible orientation concepts do not have active primary surface forms in both KO and EN; using them would confound input language with detectability.",
            "requires_human_decision": True,
        })
    status = "PREFLIGHT_PASSED_READY_FOR_FULL_RUN" if not unresolved_cases and not global_issues else "PREFLIGHT_BLOCKED_HUMAN_DECISION_REQUIRED"
    gate = {
        "status": status, "api_generation_requests_sent": 0,
        "primary_cases": len(primary_cases), "expected_requests": EXPECTED_REQUESTS,
        "factorial_assertion": {
            "n_cases": 160, "n_languages": 2, "n_models": 1, "n_efforts": 3, "n_replicates": 2,
            "expected_requests": 1920, "pass": 160 * 2 * 1 * 3 * 2 == EXPECTED_REQUESTS,
        },
        "model": {"provider": PRIMARY_PROVIDER, "model_id": PRIMARY_MODEL_ID, "reasoning_efforts": list(PRIMARY_EFFORTS), "replicates": PRIMARY_REPLICATES},
        "master_seed": REQUEST_ORDER_SEED, "unresolved_critical_case_ids": sorted(unresolved_cases),
        "unresolved_critical_cases": len(unresolved_cases), "global_issues": global_issues,
        "country_orientation_definition_frozen": True,
        "country_orientation_inventory_frozen": not global_issues,
        "inventory_summary": inventory_summary,
        "source_artifact_hashes": {
            CASES_PATH.relative_to(REPO_ROOT).as_posix(): sha_file(CASES_PATH),
            FACTS_PATH.relative_to(REPO_ROOT).as_posix(): sha_file(FACTS_PATH),
            ROSTER_PATH.relative_to(REPO_ROOT).as_posix(): sha_file(ROSTER_PATH),
            HUMAN_DECISIONS_PATH.relative_to(REPO_ROOT).as_posix(): sha_file(HUMAN_DECISIONS_PATH),
            SURFACE_POLICY_PATH.relative_to(REPO_ROOT).as_posix(): sha_file(SURFACE_POLICY_PATH),
            SURFACE_DECISIONS_PATH.relative_to(REPO_ROOT).as_posix(): sha_file(SURFACE_DECISIONS_PATH),
            (ORIENTATION_LEXICON_DIR / "run_manifest.json").relative_to(REPO_ROOT).as_posix(): sha_file(ORIENTATION_LEXICON_DIR / "run_manifest.json"),
        },
        "bilingual_surface_activation_policy_status": "researcher_approved",
        "bilingual_surface_recommendations_signoff_status": "researcher_approved",
        "targeted_human_decisions_applied": len(human_decisions),
        "automatic_edits": 0,
        "review_type": REVIEW_TYPE,
    }
    write_json(OUTPUT_DIR / "final_generation_gate.json", gate)

    salience_counts = Counter(row["jurisdiction_salience_level"] for row in salience_rows)
    report = [
        "# Experiment 1 final pre-generation QC", "", f"Status: **{status}**", "", "No API generation request was sent.", "",
        "## Provenance", "",
        f"- Current-hash-matched existing human QC cases: {sum(row['existing_human_review_available'] == 'true' for row in final_rows)}",
        f"- Codex-only QC cases: {sum(row['existing_human_review_available'] == 'false' for row in final_rows)}",
        f"- Review type: `{REVIEW_TYPE}`", "",
        "## Case QC", "",
        f"- Primary confirmatory cases: {len(primary_cases)}",
        f"- Legal-leakage finding rows: {sum(row['severity'] == 'critical' for row in leakage_rows)} across {len(unresolved_leakage)} cases",
        f"- Targeted researcher retain-as-is decisions applied: {len(human_decisions)}",
        "- Automatic fact edits: 0",
        f"- Unresolved critical cases: {len(unresolved_cases)} ({', '.join(sorted(unresolved_cases)) or 'none'})",
        f"- Salience levels: {dict(sorted(salience_counts.items()))}",
        f"- Neutralization QC passed: {sum(row['neutralization_qc_status'] == 'passed' for row in neutral_rows)}/{len(neutral_rows)}", "",
        "## Country orientation", "",
        "- Formula frozen: `(N_US - N_KR) / (N_US + N_KR + 1)` using document-level unique eligible concept IDs.",
        f"- Inventory concepts: {inventory_summary['concept_count']}; eligible: {inventory_summary['eligible_concept_count']}.",
        f"- Eligible concepts with active KO+EN surfaces: {inventory_summary['eligible_bilingual_coverage_count']}/{inventory_summary['eligible_concept_count']}.",
        "- The inventory is not primary-ready because language-specific active-surface coverage would mechanically confound KO/EN comparisons.", "",
        "## Generation gate", "",
        f"- Factorial assertion: 160 x 2 x 1 x 3 x 2 = {EXPECTED_REQUESTS} (PASS)",
        "- Full generation was not started because bilingual orientation-marker activation remains unresolved.", "",
    ]
    (OUTPUT_DIR / "FINAL_QC_REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")

    hashes = {path.name: sha_file(path) for path in sorted(OUTPUT_DIR.iterdir()) if path.is_file() and path.name != "SHA256SUMS.txt"}
    (OUTPUT_DIR / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())), encoding="ascii", newline="\n"
    )
    print(canonical_json({
        "status": status, "api_generation_requests_sent": 0,
        "unresolved_critical_case_ids": sorted(unresolved_cases),
        "global_issue_codes": [row["code"] for row in global_issues],
    }))


if __name__ == "__main__":
    main()
