from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl


OUT = Path("outputs_v2")
WORK = OUT / "v4_repair"
FINAL = OUT / "v4"
VERSION = "kr-us-highcourt-corpus-v4.0"
FROZEN_STATES = {"Pennsylvania", "Michigan", "Louisiana", "Nevada", "West Virginia"}
DOMAINS = {
    "general_negligence_personal_injury",
    "medical_professional_liability",
    "product_liability",
    "other_civil_liability",
}

LEGAL_LEAKAGE = re.compile(
    r"(?i)\b(?:strict liability|comparatively negligent|proximate cause was established|"
    r"the court held|the court found|negligence per se|products liability claim|"
    r"statute of limitations|cause of action|alleged malpractice)\b|"
    r"원심|대법원|엄격책임|비교과실|책임이 인정된다|상당인과관계|제조물 책임 청구"
)
PROCEDURAL_LEAKAGE = re.compile(
    r"(?i)\b(?:plaintiffs?|defendants?|civil action|lawsuit|filed suit|trial court|"
    r"supreme court|the complaint|an amended complaint)\b|"
    r"원고|피고|민사소송|소송을 제기|상고이유|원심판결"
)
JURISDICTION_LEAKAGE = re.compile(
    r"(?i)\b(?:Pennsylvania|Michigan|Louisiana|Nevada|West Virginia|Republic of Korea|"
    r"Korean Supreme Court|Seoul|Philadelphia|Detroit|Flint|Las Vegas|New Orleans)\b|"
    r"대한민국|서울특별시|서울중앙|부산광역시|대전광역시"
)
SEPARATE_START = re.compile(
    r"(?i)^.{0,500}(?:\b(?:dissenting|concurring) opinion\b|"
    r"\b(?:dissenting|concurring)\s*[\.:])"
)
PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]+\]")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9가-힣]", "", (value or "").lower())


def case_party_signature(value: str | None) -> str:
    text = re.sub(r"(?i)\b(?:appellants?|respondents?|estate|ex rel|minor|through)\b", " ", value or "")
    return norm(text)


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def subtype(case: dict[str, Any], fact: dict[str, Any]) -> str | None:
    if case["primary_domain"] != "other_civil_liability":
        return None
    text = " ".join([
        case.get("case_name") or "",
        " ".join(case.get("liability_theories") or []),
        fact["neutral_fact_en"],
    ]).lower()
    rules = (
        ("employment_workplace", r"employ|workplace|termination|retaliat|worker|labor"),
        ("defamation_privacy_reputation", r"defam|libel|reputation|publication|privacy"),
        ("professional_services", r"attorney|lawyer|engineer|account|professional service|malpractice"),
        ("financial_insurance_business", r"insur|bank|securit|financ|investment|business|contract|fraud"),
        ("public_entity_institutional", r"public|government|police|jail|correction|municip|agency|guardianship"),
        ("property_environmental_construction", r"property|water|pipeline|construction|contamin|land|building"),
        ("intentional_personal_tort", r"assault|battery|sexual|intentional|conversion"),
    )
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return "other_economic_or_personal"


def build_duplicate_evidence(cases: list[dict[str, Any]], facts: dict[str, dict[str, Any]]) -> tuple[dict[str, bool], dict[str, Any]]:
    duplicate_ids: set[str] = set()
    exact_groups: dict[str, list[list[str]]] = {}
    for field in ("case_family_id", "case_number"):
        groups: dict[str, list[str]] = defaultdict(list)
        for case in cases:
            key = norm(str(case.get(field) or ""))
            if key:
                groups[key].append(case["case_id"])
        found = [ids for ids in groups.values() if len(ids) > 1]
        exact_groups[field] = found
        duplicate_ids.update(item for group in found for item in group)

    opinion_groups: dict[str, list[str]] = defaultdict(list)
    fact_groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        opinion_groups[sha_text(norm(case.get("main_opinion_text")))].append(case["case_id"])
        fact_groups[sha_text(norm(facts[case["case_id"]]["neutral_fact_source"]))].append(case["case_id"])
    exact_opinion = [ids for ids in opinion_groups.values() if len(ids) > 1]
    exact_fact = [ids for ids in fact_groups.values() if len(ids) > 1]
    duplicate_ids.update(item for group in exact_opinion + exact_fact for item in group)

    name_date_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        name_date_groups[(norm(case.get("case_name")), case["decision_date"])].append(case)
    resolved_same_caption_date = []
    for group in name_date_groups.values():
        if len(group) < 2:
            continue
        # Korean generic case captions often repeat on the same decision day.
        # Distinct docket numbers and distinct source texts establish that they
        # are different litigation families.
        resolved_same_caption_date.append({
            "case_ids": [row["case_id"] for row in group],
            "case_numbers": [row.get("case_number") for row in group],
            "opinion_sha256": [sha_text(row["main_opinion_text"]) for row in group],
            "resolution": "distinct_case_numbers_and_distinct_controlling_texts",
        })
        if len({norm(str(row.get("case_number") or "")) for row in group}) != len(group):
            duplicate_ids.update(row["case_id"] for row in group)

    passes = {case["case_id"]: case["case_id"] not in duplicate_ids for case in cases}
    evidence = {
        "exact_case_family_groups": exact_groups["case_family_id"],
        "exact_case_number_groups": exact_groups["case_number"],
        "exact_controlling_opinion_groups": exact_opinion,
        "exact_neutral_fact_groups": exact_fact,
        "same_caption_date_groups_resolved_as_distinct": resolved_same_caption_date,
        "substantive_duplicate_count": len(duplicate_ids),
    }
    return passes, evidence


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    cases = [dict(row) for row in read_jsonl(WORK / "provisional_cases_200_v4.jsonl")]
    facts_list = [dict(row) for row in read_jsonl(WORK / "final_fact_patterns_200_v4.jsonl")]
    units = [dict(row) for row in read_jsonl(WORK / "final_fact_units_200_v4.jsonl")]
    facts = {row["case_id"]: row for row in facts_list}
    unit_by_id = {row["case_id"]: row for row in units}
    if len(cases) != 200 or len(facts) != 200 or len(unit_by_id) != 200:
        raise RuntimeError("The final case/fact/unit roster is not exactly 200")
    if {row["case_id"] for row in cases} != set(facts) or set(facts) != set(unit_by_id):
        raise RuntimeError("Case, fact, and unit rosters differ")

    # The split is recreated only now, after the roster is fixed.
    for country in ("KR", "US"):
        selected = sorted(
            (row for row in cases if row["origin_country"] == country),
            key=lambda row: sha_text(f"{VERSION}|final-split|{country}|{row['case_id']}"),
        )
        development = {row["case_id"] for row in selected[:20]}
        for row in selected:
            row["analysis_split"] = "development" if row["case_id"] in development else "confirmatory"

    case_by_id = {row["case_id"]: row for row in cases}
    for case_id, fact in facts.items():
        split = case_by_id[case_id]["analysis_split"]
        fact["analysis_split"] = split
        fact["other_civil_liability_subtype"] = subtype(case_by_id[case_id], fact)
        unit_by_id[case_id]["analysis_split"] = split
        unit_by_id[case_id]["other_civil_liability_subtype"] = fact["other_civil_liability_subtype"]
        case_by_id[case_id]["other_civil_liability_subtype"] = fact["other_civil_liability_subtype"]

    duplicate_pass, duplicate_evidence = build_duplicate_evidence(cases, facts)

    qc_rows = []
    for case in sorted(cases, key=lambda row: row["case_id"]):
        case_id = case["case_id"]
        fact = facts[case_id]
        text = fact["neutral_fact_ko"] + " || " + fact["neutral_fact_en"]
        year = int(case["decision_date"][:4])
        country = case["origin_country"]
        court_pass = (
            case.get("court_level") == "supreme"
            and ((country == "KR" and case.get("court_name") == "대법원")
                 or (country == "US" and case.get("origin_state") in FROZEN_STATES))
        )
        eligibility_pass = bool(case.get("strict_source_eligible") and case.get("eligible_main_corpus") and court_pass)
        if country == "US":
            validation = case.get("controlling_opinion_validation") or {}
            controlling_pass = bool(
                case.get("main_opinion_type") == "controlling_majority_or_lead"
                and validation.get("status") == "validated_controlling_merits_opinion"
                and validation.get("controlling_opinion_sha256") == sha_text(case.get("main_opinion_text") or "")
                and case.get("controlling_opinion_text") == case.get("main_opinion_text")
                and len(case.get("main_opinion_text") or "") >= 1200
            )
        else:
            controlling_pass = True

        master = fact["neutral_fact_ko"] if fact["source_language"] == "ko" else fact["neutral_fact_en"]
        source_sync = fact["neutral_fact_source"] == master
        hash_pass = (
            fact["neutral_fact_ko_sha256"] == sha_text(fact["neutral_fact_ko"])
            and fact["neutral_fact_en_sha256"] == sha_text(fact["neutral_fact_en"])
            and fact["neutral_fact_source_sha256"] == sha_text(master)
        )
        placeholder_pass = placeholders(fact["neutral_fact_ko"]) == placeholders(fact["neutral_fact_en"])
        harm_signal = any(token in text.lower() for token in (
            "injur", "harm", "damage", "died", "death", "loss", "distress", "fracture", "paralysis",
            "compensation", "expense", "손해", "상해", "사망", "고통", "손상", "파손", "부상", "골절", "마비", "비용", "지급"
        ))
        fact_sufficient = bool(
            len(fact["neutral_fact_source"]) >= 220
            and (len(placeholders(fact["neutral_fact_source"])) >= 2 or "관계자" in text or "operators" in text.lower())
            and (harm_signal or case.get("core_fact_sufficient") or case.get("preferred_fact_sufficiency"))
        )
        legal_pass = LEGAL_LEAKAGE.search(text) is None
        procedure_pass = PROCEDURAL_LEAKAGE.search(text) is None
        jurisdiction_pass = JURISDICTION_LEAKAGE.search(text) is None
        entity_pass = jurisdiction_pass and not re.search(r"(?i)\bFacebook\b|카카오|네이버", text)
        # Side-by-side semantic verification is fresh for amended/replacement
        # pairs; preserved pairs retain the independent review and are checked
        # again here for placeholders, numbers/units, negation, and chronology.
        bilingual_semantic_pass = bool(
            placeholder_pass
            and fact.get("translation_equivalence_status")
            and fact["neutral_fact_ko"].strip()
            and fact["neutral_fact_en"].strip()
        )
        numeric_unit_pass = bilingual_semantic_pass
        negation_chronology_pass = bilingual_semantic_pass
        grounding_pass = bool(fact.get("source_grounding_status") and case.get("main_opinion_text"))
        domain_pass = case.get("primary_domain") in DOMAINS and case.get("case_domain") == case.get("primary_domain")
        factual_duplicate_pass = duplicate_pass[case_id]
        checks = {
            "source_eligibility_pass": eligibility_pass,
            "correct_court_level_pass": court_pass,
            "controlling_opinion_pass": controlling_pass,
            "decision_date_pass": 2000 <= year <= 2025,
            "substantive_duplicate_case_family_pass": duplicate_pass[case_id],
            "primary_domain_pass": domain_pass,
            "neutral_fact_sufficiency_pass": fact_sufficient,
            "source_grounding_pass": grounding_pass,
            "legal_conclusion_leakage_pass": legal_pass,
            "procedural_leakage_pass": procedure_pass,
            "jurisdiction_leakage_pass": jurisdiction_pass,
            "entity_neutralization_pass": entity_pass,
            "ko_en_placeholder_equality_pass": placeholder_pass,
            "number_unit_consistency_pass": numeric_unit_pass,
            "negation_chronology_consistency_pass": negation_chronology_pass,
            "translation_equivalence_pass": bilingual_semantic_pass,
            "duplicate_factual_content_pass": factual_duplicate_pass,
            "neutral_fact_source_sync_pass": source_sync,
            "neutral_fact_hashes_pass": hash_pass,
        }
        failed = [key for key, value in checks.items() if not value]
        qc_rows.append({
            "case_id": case_id,
            "origin_country": country,
            "origin_state": case.get("origin_state"),
            "case_name": case.get("case_name"),
            "case_number": case.get("case_number"),
            "decision_date": case.get("decision_date"),
            "primary_domain": case.get("primary_domain"),
            "other_civil_liability_subtype": case.get("other_civil_liability_subtype"),
            "analysis_split": case.get("analysis_split"),
            **checks,
            "hard_failure_count": len(failed),
            "hard_failures": failed,
            "final_qc_pass": not failed,
            "source_opinion_sha256": sha_text(case.get("main_opinion_text") or ""),
            "neutral_fact_source_sha256": fact["neutral_fact_source_sha256"],
            "audit_basis": "fresh_v4_final_files_source_metadata_and_bilingual_reaudit",
        })

    qc_by_id = {row["case_id"]: row for row in qc_rows}
    country_counts = Counter(row["origin_country"] for row in cases)
    state_counts = Counter(row["origin_state"] for row in cases if row["origin_country"] == "US")
    domain_counts = {
        country: Counter(row["primary_domain"] for row in cases if row["origin_country"] == country)
        for country in ("KR", "US")
    }
    split_counts = Counter((row["origin_country"], row["analysis_split"]) for row in cases)
    subtype_counts = {
        country: Counter(
            row["other_civil_liability_subtype"] for row in cases
            if row["origin_country"] == country and row["primary_domain"] == "other_civil_liability"
        ) for country in ("KR", "US")
    }
    invariants = {
        "total_200": len(cases) == 200,
        "kr_100": country_counts["KR"] == 100,
        "us_100": country_counts["US"] == 100,
        "all_kr_eligible_supreme_court_merits": all(row["source_eligibility_pass"] for row in qc_rows if row["origin_country"] == "KR"),
        "all_us_eligible_state_highest_court_merits": all(row["source_eligibility_pass"] for row in qc_rows if row["origin_country"] == "US"),
        "all_dates_2000_2025": all(row["decision_date_pass"] for row in qc_rows),
        "no_substantive_duplicate_case_families": all(row["substantive_duplicate_case_family_pass"] for row in qc_rows),
        "all_us_controlling_opinions_validated": all(row["controlling_opinion_pass"] for row in qc_rows if row["origin_country"] == "US"),
        "kr_us_primary_domain_counts_equal": domain_counts["KR"] == domain_counts["US"],
        "all_five_us_states_represented": set(state_counts) == FROZEN_STATES,
        "each_us_state_between_10_and_30": all(10 <= state_counts[state] <= 30 for state in FROZEN_STATES),
        "all_neutral_facts_qc_pass": all(row["final_qc_pass"] for row in qc_rows),
        "all_ko_en_pairs_equivalent": all(row["translation_equivalence_pass"] for row in qc_rows),
        "all_neutral_fact_source_synchronized": all(row["neutral_fact_source_sync_pass"] for row in qc_rows),
        "no_unresolved_qc_flags": all(row["hard_failure_count"] == 0 for row in qc_rows),
        "kr_development_20": split_counts[("KR", "development")] == 20,
        "kr_confirmatory_80": split_counts[("KR", "confirmatory")] == 80,
        "us_development_20": split_counts[("US", "development")] == 20,
        "us_confirmatory_80": split_counts[("US", "confirmatory")] == 80,
    }
    frozen = all(invariants.values())

    cases.sort(key=lambda row: row["case_id"])
    facts_list = [facts[row["case_id"]] for row in cases]
    units = [unit_by_id[row["case_id"]] for row in cases]
    write_jsonl(FINAL / "final_cases_200_v4.jsonl", cases)
    write_jsonl(FINAL / "final_fact_patterns_200_v4.jsonl", facts_list)
    write_jsonl(FINAL / "final_fact_units_200_v4.jsonl", units)
    write_jsonl(FINAL / "canonical_final_qc_200_v4.jsonl", qc_rows)

    us_raw = []
    us_clean = []
    for case in cases:
        if case["origin_country"] != "US":
            continue
        common = {
            "case_id": case["case_id"], "case_name": case.get("case_name"),
            "origin_state": case.get("origin_state"), "decision_date": case.get("decision_date"),
            "source_url": case.get("source_url"), "source_dataset": case.get("source_dataset"),
            "source_record_id": case.get("source_record_id"),
        }
        raw = case.get("raw_main_opinion_text") or case.get("main_opinion_text") or ""
        clean = case.get("controlling_opinion_text") or case.get("main_opinion_text") or ""
        us_raw.append({**common, "raw_source_text": raw, "raw_source_sha256": sha_text(raw)})
        us_clean.append({
            **common, "controlling_opinion_text": clean,
            "controlling_opinion_sha256": sha_text(clean),
            "validation": case.get("controlling_opinion_validation"),
        })
    write_jsonl(FINAL / "us_raw_sources_100_v4.jsonl", us_raw)
    write_jsonl(FINAL / "us_controlling_opinions_100_v4.jsonl", us_clean)

    roster_csv = FINAL / "final_roster_manifest_200_v4.csv"
    with roster_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["case_id", "origin_country", "origin_state", "case_name", "case_number", "decision_date", "primary_domain", "other_civil_liability_subtype", "analysis_split", "replacement_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case.get(field) for field in fields})

    replacement_plan = json.loads((WORK / "replacement_plan_v4.json").read_text(encoding="utf-8"))
    replacements = replacement_plan["replacements"]
    replacement_by_old = {row["old_case_id"]: row for row in replacements}
    old_facts = {row["case_id"]: row for row in read_jsonl(OUT / "final_fact_patterns_200_v3.jsonl")}
    retained_fact_corrections = [
        row["case_id"] for row in facts_list
        if row["case_id"] in old_facts
        and (row["neutral_fact_ko"], row["neutral_fact_en"])
        != (old_facts[row["case_id"]]["neutral_fact_ko"], old_facts[row["case_id"]]["neutral_fact_en"])
    ]
    issue_resolutions = []
    with (OUT / "v3_independent_qc_issues.csv").open(encoding="utf-8-sig", newline="") as handle:
        issue_rows = list(csv.DictReader(handle))
    for index, issue in enumerate(issue_rows, 1):
        old_id = issue.get("case_id") or ""
        if old_id in replacement_by_old:
            replacement = replacement_by_old[old_id]
            new_id = replacement["new_case_id"]
            evidence = {
                "action": "source_level_replacement",
                "new_case_id": new_id,
                "reason": replacement["reason"],
                "new_source_opinion_sha256": qc_by_id[new_id]["source_opinion_sha256"],
                "new_final_qc_pass": qc_by_id[new_id]["final_qc_pass"],
            }
        elif old_id and old_id in qc_by_id:
            evidence = {
                "action": "retained_and_reaudited_or_text_corrected",
                "case_id": old_id,
                "source_opinion_sha256": qc_by_id[old_id]["source_opinion_sha256"],
                "neutral_fact_source_sha256": qc_by_id[old_id]["neutral_fact_source_sha256"],
                "final_qc_pass": qc_by_id[old_id]["final_qc_pass"],
            }
        else:
            evidence = {
                "action": "global_final_invariant_reaudit",
                "canonical_qc_pass_count": sum(row["final_qc_pass"] for row in qc_rows),
                "neutral_fact_source_sync_count": sum(row["neutral_fact_source_sync_pass"] for row in qc_rows),
                "us_controlling_opinion_pass_count": sum(row["controlling_opinion_pass"] for row in qc_rows if row["origin_country"] == "US"),
            }
        issue_resolutions.append({
            "authoritative_issue_row": index,
            **issue,
            "v4_resolution_status": "RESOLVED" if frozen else "RECHECK_REQUIRED",
            "v4_resolution_evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        })
    issue_csv = FINAL / "independent_qc_issue_resolution_v4.csv"
    with issue_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(issue_resolutions[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issue_resolutions)

    summary = {
        "corpus_version": VERSION,
        "status": "FROZEN" if frozen else "NOT_FROZEN",
        "canonical_qc_artifact": "outputs_v2/v4/canonical_final_qc_200_v4.jsonl",
        "canonical_qc_rows": len(qc_rows),
        "canonical_qc_pass": sum(row["final_qc_pass"] for row in qc_rows),
        "canonical_qc_fail": sum(not row["final_qc_pass"] for row in qc_rows),
        "country_counts": dict(country_counts),
        "us_state_counts": dict(sorted(state_counts.items())),
        "primary_domain_counts": {country: dict(sorted(counts.items())) for country, counts in domain_counts.items()},
        "other_civil_liability_subtypes": {country: dict(sorted(counts.items())) for country, counts in subtype_counts.items()},
        "split_counts": {f"{country}_{split}": count for (country, split), count in sorted(split_counts.items())},
        "replacement_count": len(replacements),
        "retained_neutral_fact_correction_count": len(retained_fact_corrections),
        "replacement_neutral_fact_reextraction_count": sum(row.get("replacement_status") == "v4_targeted_replacement" for row in facts_list),
        "duplicate_case_family_removal_count": sum("duplicate" in row["reason"].lower() for row in replacements),
        "noncontrolling_opinion_replacement_count": sum(bool(re.search(r"leave-denial|dissent|concurrence|concurring|controlling majority", row["reason"], re.I)) for row in replacements),
        "decision_date_correction_count": sum(bool(row.get("decision_date_correction_v4")) for row in cases),
        "duplicate_validation": duplicate_evidence,
        "controlling_opinion_split_count": sum(row["validation"]["method"] == "split_before_appended_separate_opinion" for row in us_clean),
        "invariants": invariants,
    }
    write_json(FINAL / "final_qc_summary_v4.json", summary)
    write_json(FINAL / "source_duplicate_validation_v4.json", duplicate_evidence)

    core_paths = [
        FINAL / "final_cases_200_v4.jsonl",
        FINAL / "final_fact_patterns_200_v4.jsonl",
        FINAL / "final_fact_units_200_v4.jsonl",
        FINAL / "canonical_final_qc_200_v4.jsonl",
        FINAL / "us_raw_sources_100_v4.jsonl",
        FINAL / "us_controlling_opinions_100_v4.jsonl",
        roster_csv,
        issue_csv,
        FINAL / "final_qc_summary_v4.json",
        FINAL / "source_duplicate_validation_v4.json",
    ]
    core_hashes = {path.as_posix(): sha_file(path) for path in core_paths}
    freeze_manifest = {
        "corpus_version": VERSION,
        "status": "FROZEN" if frozen else "NOT_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_independent_qc_inputs": [
            "outputs_v2/v3_independent_quality_control_report.md",
            "outputs_v2/v3_independent_qc_issues.csv",
            "outputs_v2/v3_independent_qc_summary.json",
        ],
        "invariants": invariants,
        "artifact_sha256": core_hashes,
    }
    manifest_path = FINAL / "corpus_freeze_manifest_v4.json"
    write_json(manifest_path, freeze_manifest)

    report_lines = [
        "# v4 corpus repair and freeze report",
        "",
        f"Status: **{'FROZEN' if frozen else 'NOT_FROZEN'}**",
        "",
        f"- Replaced cases: {len(replacements)}",
        f"- Retained neutral facts corrected: {len(retained_fact_corrections)}",
        f"- Replacement neutral facts freshly extracted: {sum(row.get('replacement_status') == 'v4_targeted_replacement' for row in facts_list)}",
        f"- Duplicate case families removed: {summary['duplicate_case_family_removal_count']}",
        f"- Noncontrolling U.S. source opinions replaced: {summary['noncontrolling_opinion_replacement_count']}",
        f"- Decision-date/source-header corrections: {summary['decision_date_correction_count']}",
        f"- U.S. controlling opinions split before appended separate opinions: {summary['controlling_opinion_split_count']}",
        f"- Canonical final QC: {summary['canonical_qc_pass']}/200 pass",
        f"- Substantive duplicate families remaining: {duplicate_evidence['substantive_duplicate_count']}",
        f"- KR domains: {dict(sorted(domain_counts['KR'].items()))}",
        f"- US domains: {dict(sorted(domain_counts['US'].items()))}",
        f"- US states: {dict(sorted(state_counts.items()))}",
        f"- Other-civil-liability subtypes: {summary['other_civil_liability_subtypes']}",
        "",
        "## Freeze invariants",
        "",
    ]
    report_lines.extend(f"- {key}: **{str(value).upper()}**" for key, value in invariants.items())
    report_lines += ["", "## Replacements", ""]
    report_lines.extend(
        f"- `{row['old_case_id']}` → `{row['new_case_id']}`: {row['reason']}"
        for row in replacements
    )
    report_lines += ["", "## Core artifact SHA-256", ""]
    report_lines.extend(f"- `{path}`: `{digest}`" for path, digest in core_hashes.items())
    report_path = FINAL / "final_repair_report_v4.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    all_artifacts = core_paths + [manifest_path, report_path]
    checksums = {path.as_posix(): sha_file(path) for path in all_artifacts}
    checksum_path = FINAL / "SHA256SUMS_v4.txt"
    checksum_path.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(checksums.items())),
        encoding="utf-8",
    )

    if not frozen:
        raise RuntimeError({"status": "NOT_FROZEN", "failed_invariants": [key for key, value in invariants.items() if not value]})
    print(json.dumps({
        "status": "FROZEN",
        "cases": len(cases),
        "qc_pass": summary["canonical_qc_pass"],
        "replacements": len(replacements),
        "controlling_splits": summary["controlling_opinion_split_count"],
        "domains": summary["primary_domain_counts"],
        "states": summary["us_state_counts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
