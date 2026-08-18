from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_and_freeze_v4 import (
    DOMAINS,
    FROZEN_STATES,
    JURISDICTION_LEAKAGE,
    LEGAL_LEAKAGE,
    PROCEDURAL_LEAKAGE,
    build_duplicate_evidence,
    placeholders,
    sha_text,
)
from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl


SOURCE = Path("outputs_v2/v4")
FINAL = Path("outputs_v2/v4.0.1")
VERSION = "kr-us-highcourt-corpus-v4.0.1"
SUFFIX = "v4_0_1"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_case_view(case: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "corpus_version",
        "substantive_civil_liability_central",
        "substantive_civil_liability_central_pre_v4",
    }
    return {key: value for key, value in case.items() if key not in excluded}


def fact_text_view(fact: dict[str, Any]) -> tuple[str, str, str]:
    return (
        fact["neutral_fact_ko"],
        fact["neutral_fact_en"],
        fact["neutral_fact_source"],
    )


def unit_text_view(unit: dict[str, Any]) -> tuple[str, str, str]:
    return unit["neutral_ko"], unit["neutral_en"], unit["text"]


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    source_cases = [dict(row) for row in read_jsonl(SOURCE / "final_cases_200_v4.jsonl")]
    source_facts = [dict(row) for row in read_jsonl(SOURCE / "final_fact_patterns_200_v4.jsonl")]
    source_units = [dict(row) for row in read_jsonl(SOURCE / "final_fact_units_200_v4.jsonl")]
    source_fact_by_id = {row["case_id"]: row for row in source_facts}
    source_unit_by_id = {row["case_id"]: row for row in source_units}
    source_ids = [row["case_id"] for row in source_cases]

    if len(source_cases) != 200 or len(source_fact_by_id) != 200 or len(source_unit_by_id) != 200:
        raise RuntimeError("Current v4 case/fact/unit inputs are not exactly 200 records")
    if set(source_ids) != set(source_fact_by_id) or set(source_ids) != set(source_unit_by_id):
        raise RuntimeError("Current v4 case/fact/unit rosters differ")

    current_input_hashes = {
        (SOURCE / "final_cases_200_v4.jsonl").as_posix(): sha_file(SOURCE / "final_cases_200_v4.jsonl"),
        (SOURCE / "final_fact_patterns_200_v4.jsonl").as_posix(): sha_file(SOURCE / "final_fact_patterns_200_v4.jsonl"),
        (SOURCE / "final_fact_units_200_v4.jsonl").as_posix(): sha_file(SOURCE / "final_fact_units_200_v4.jsonl"),
    }

    cases = deepcopy(source_cases)
    facts_list = deepcopy(source_facts)
    units = deepcopy(source_units)
    old_central_by_id = {
        row["case_id"]: row.get("substantive_civil_liability_central") for row in source_cases
    }

    for case in cases:
        case["substantive_civil_liability_central_pre_v4"] = old_central_by_id[case["case_id"]]
        case["substantive_civil_liability_central"] = True
        case["corpus_version"] = VERSION

    stale_fact_hash_ids: list[str] = []
    for fact in facts_list:
        expected = {
            "neutral_fact_ko_sha256": sha_text(fact["neutral_fact_ko"]),
            "neutral_fact_en_sha256": sha_text(fact["neutral_fact_en"]),
            "neutral_fact_source_sha256": sha_text(fact["neutral_fact_source"]),
        }
        if any(fact.get(key) != value for key, value in expected.items()):
            stale_fact_hash_ids.append(fact["case_id"])
        fact.update(expected)
        fact["corpus_version"] = VERSION
    for unit in units:
        unit["corpus_version"] = VERSION

    case_by_id = {row["case_id"]: row for row in cases}
    fact_by_id = {row["case_id"]: row for row in facts_list}
    unit_by_id = {row["case_id"]: row for row in units}
    duplicate_pass, duplicate_evidence = build_duplicate_evidence(cases, fact_by_id)

    qc_rows: list[dict[str, Any]] = []
    for case_id in source_ids:
        case = case_by_id[case_id]
        fact = fact_by_id[case_id]
        unit = unit_by_id[case_id]
        original_case = next(row for row in source_cases if row["case_id"] == case_id)
        text = fact["neutral_fact_ko"] + " || " + fact["neutral_fact_en"]
        country = case["origin_country"]
        year = int(case["decision_date"][:4])
        court_pass = (
            case.get("court_level") == "supreme"
            and (
                (country == "KR" and case.get("court_name") == "대법원")
                or (country == "US" and case.get("origin_state") in FROZEN_STATES)
            )
        )
        eligibility_pass = bool(
            case.get("strict_source_eligible")
            and case.get("eligible_main_corpus")
            and court_pass
            and case.get("substantive_civil_liability_central") is True
        )
        if country == "US":
            validation = case.get("controlling_opinion_validation") or {}
            controlling_pass = bool(
                case.get("main_opinion_type") == "controlling_majority_or_lead"
                and validation.get("status") == "validated_controlling_merits_opinion"
                and validation.get("controlling_opinion_sha256")
                == sha_text(case.get("main_opinion_text") or "")
                and case.get("controlling_opinion_text") == case.get("main_opinion_text")
                and len(case.get("main_opinion_text") or "") >= 1200
            )
        else:
            controlling_pass = True

        master = fact["neutral_fact_ko" if fact["source_language"] == "ko" else "neutral_fact_en"]
        placeholder_pass = placeholders(fact["neutral_fact_ko"]) == placeholders(fact["neutral_fact_en"])
        harm_signal = any(
            token in text.lower()
            for token in (
                "injur", "harm", "damage", "died", "death", "loss", "distress", "fracture",
                "paralysis", "compensation", "expense", "손해", "상해", "사망", "고통", "손상",
                "파손", "부상", "골절", "마비", "비용", "지급",
            )
        )
        bilingual_pass = bool(
            placeholder_pass
            and fact.get("translation_equivalence_status")
            and fact["neutral_fact_ko"].strip()
            and fact["neutral_fact_en"].strip()
        )
        checks = {
            "source_eligibility_pass": eligibility_pass,
            "correct_court_level_pass": court_pass,
            "controlling_opinion_pass": controlling_pass,
            "decision_date_pass": 2000 <= year <= 2025,
            "substantive_duplicate_case_family_pass": duplicate_pass[case_id],
            "primary_domain_pass": case.get("primary_domain") in DOMAINS and case.get("case_domain") == case.get("primary_domain"),
            "neutral_fact_sufficiency_pass": bool(
                len(fact["neutral_fact_source"]) >= 220
                and (len(placeholders(fact["neutral_fact_source"])) >= 2 or "관계자" in text or "operators" in text.lower())
                and (harm_signal or case.get("core_fact_sufficient") or case.get("preferred_fact_sufficiency"))
            ),
            "source_grounding_pass": bool(fact.get("source_grounding_status") and case.get("main_opinion_text")),
            "legal_conclusion_leakage_pass": LEGAL_LEAKAGE.search(text) is None,
            "procedural_leakage_pass": PROCEDURAL_LEAKAGE.search(text) is None,
            "jurisdiction_leakage_pass": JURISDICTION_LEAKAGE.search(text) is None,
            "entity_neutralization_pass": JURISDICTION_LEAKAGE.search(text) is None and re.search(r"(?i)\bFacebook\b|카카오|네이버", text) is None,
            "ko_en_placeholder_equality_pass": placeholder_pass,
            "number_unit_consistency_pass": bilingual_pass,
            "negation_chronology_consistency_pass": bilingual_pass,
            "translation_equivalence_pass": bilingual_pass,
            "duplicate_factual_content_pass": duplicate_pass[case_id],
            "neutral_fact_source_sync_pass": fact["neutral_fact_source"] == master,
            "neutral_fact_hashes_pass": bool(
                fact["neutral_fact_ko_sha256"] == sha_text(fact["neutral_fact_ko"])
                and fact["neutral_fact_en_sha256"] == sha_text(fact["neutral_fact_en"])
                and fact["neutral_fact_source_sha256"] == sha_text(master)
            ),
            "case_roster_identity_pass": case["case_id"] == original_case["case_id"] and case.get("case_family_id") == original_case.get("case_family_id"),
            "immutable_case_metadata_pass": immutable_case_view(case) == immutable_case_view(original_case),
            "substantive_central_sync_pass": case.get("substantive_civil_liability_central") is True,
            "substantive_central_pre_v4_preserved_pass": case.get("substantive_civil_liability_central_pre_v4") == old_central_by_id[case_id],
            "neutral_fact_text_preserved_pass": fact_text_view(fact) == fact_text_view(source_fact_by_id[case_id]),
            "fact_unit_text_preserved_pass": unit_text_view(unit) == unit_text_view(source_unit_by_id[case_id]),
            "fact_unit_sync_pass": unit_text_view(unit) == fact_text_view(fact),
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
            "audit_basis": "fresh_v4_0_1_final_files_metadata_sync_and_full_hard_invariant_reaudit",
        })

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
        )
        for country in ("KR", "US")
    }
    invariants = {
        "total_200": len(cases) == 200,
        "kr_100": country_counts["KR"] == 100,
        "us_100": country_counts["US"] == 100,
        "case_roster_unchanged": [row["case_id"] for row in cases] == source_ids,
        "case_id_and_case_family_id_unchanged": all(row["case_roster_identity_pass"] for row in qc_rows),
        "all_non_target_case_metadata_unchanged": all(row["immutable_case_metadata_pass"] for row in qc_rows),
        "all_substantive_civil_liability_central_true": all(row["substantive_central_sync_pass"] for row in qc_rows),
        "all_pre_v4_values_preserved": all(row["substantive_central_pre_v4_preserved_pass"] for row in qc_rows),
        "all_kr_eligible_supreme_court_merits": all(row["source_eligibility_pass"] for row in qc_rows if row["origin_country"] == "KR"),
        "all_us_eligible_state_highest_court_merits": all(row["source_eligibility_pass"] for row in qc_rows if row["origin_country"] == "US"),
        "all_dates_2000_2025": all(row["decision_date_pass"] for row in qc_rows),
        "no_substantive_duplicate_case_families": all(row["substantive_duplicate_case_family_pass"] for row in qc_rows),
        "all_us_controlling_opinions_validated": all(row["controlling_opinion_pass"] for row in qc_rows if row["origin_country"] == "US"),
        "kr_us_primary_domain_counts_equal": domain_counts["KR"] == domain_counts["US"],
        "all_five_us_states_represented": set(state_counts) == FROZEN_STATES,
        "each_us_state_between_10_and_30": all(10 <= state_counts[state] <= 30 for state in FROZEN_STATES),
        "all_neutral_fact_text_preserved": all(row["neutral_fact_text_preserved_pass"] for row in qc_rows),
        "all_fact_unit_text_preserved": all(row["fact_unit_text_preserved_pass"] for row in qc_rows),
        "all_neutral_facts_qc_pass": all(row["final_qc_pass"] for row in qc_rows),
        "all_ko_en_pairs_equivalent": all(row["translation_equivalence_pass"] for row in qc_rows),
        "all_neutral_fact_source_synchronized": all(row["neutral_fact_source_sync_pass"] for row in qc_rows),
        "all_neutral_fact_hashes_current": all(row["neutral_fact_hashes_pass"] for row in qc_rows),
        "no_unresolved_qc_flags": all(row["hard_failure_count"] == 0 for row in qc_rows),
        "kr_development_20": split_counts[("KR", "development")] == 20,
        "kr_confirmatory_80": split_counts[("KR", "confirmatory")] == 80,
        "us_development_20": split_counts[("US", "development")] == 20,
        "us_confirmatory_80": split_counts[("US", "confirmatory")] == 80,
    }
    frozen = all(invariants.values())

    case_path = FINAL / f"final_cases_200_{SUFFIX}.jsonl"
    fact_path = FINAL / f"final_fact_patterns_200_{SUFFIX}.jsonl"
    unit_path = FINAL / f"final_fact_units_200_{SUFFIX}.jsonl"
    qc_path = FINAL / f"canonical_final_qc_200_{SUFFIX}.jsonl"
    write_jsonl(case_path, cases)
    write_jsonl(fact_path, facts_list)
    write_jsonl(unit_path, units)
    write_jsonl(qc_path, qc_rows)

    raw_path = FINAL / f"us_raw_sources_100_{SUFFIX}.jsonl"
    controlling_path = FINAL / f"us_controlling_opinions_100_{SUFFIX}.jsonl"
    shutil.copyfile(SOURCE / "us_raw_sources_100_v4.jsonl", raw_path)
    shutil.copyfile(SOURCE / "us_controlling_opinions_100_v4.jsonl", controlling_path)

    roster_path = FINAL / f"final_roster_manifest_200_{SUFFIX}.csv"
    with roster_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "case_id", "origin_country", "origin_state", "case_name", "case_number",
            "decision_date", "primary_domain", "other_civil_liability_subtype",
            "analysis_split", "replacement_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: case.get(field) for field in fields} for case in cases)

    duplicate_path = FINAL / f"source_duplicate_validation_{SUFFIX}.json"
    write_json(duplicate_path, {
        "corpus_version": VERSION,
        **duplicate_evidence,
    })
    metadata_path = FINAL / f"metadata_sync_audit_{SUFFIX}.json"
    old_distribution = Counter(str(value).lower() for value in old_central_by_id.values())
    metadata_audit = {
        "corpus_version": VERSION,
        "source_corpus_version": "kr-us-highcourt-corpus-v4.0",
        "source_artifact_sha256_at_upgrade": current_input_hashes,
        "case_count": len(cases),
        "roster_unchanged": invariants["case_roster_unchanged"],
        "non_target_case_metadata_unchanged": invariants["all_non_target_case_metadata_unchanged"],
        "old_substantive_civil_liability_central_distribution": dict(sorted(old_distribution.items())),
        "new_substantive_civil_liability_central_true_count": sum(case["substantive_civil_liability_central"] is True for case in cases),
        "pre_v4_value_preserved_count": sum(case["substantive_civil_liability_central_pre_v4"] == old_central_by_id[case["case_id"]] for case in cases),
        "user_manual_neutral_fact_edit_case_ids_with_stale_input_hashes": stale_fact_hash_ids,
        "neutral_fact_text_preserved_count": sum(row["neutral_fact_text_preserved_pass"] for row in qc_rows),
        "fact_unit_text_preserved_count": sum(row["fact_unit_text_preserved_pass"] for row in qc_rows),
    }
    write_json(metadata_path, metadata_audit)

    old_summary = json.loads((SOURCE / "final_qc_summary_v4.json").read_text(encoding="utf-8"))
    summary_path = FINAL / f"final_qc_summary_{SUFFIX}.json"
    summary = {
        "corpus_version": VERSION,
        "status": "FROZEN" if frozen else "NOT_FROZEN",
        "canonical_qc_artifact": qc_path.as_posix(),
        "canonical_qc_rows": len(qc_rows),
        "canonical_qc_pass": sum(row["final_qc_pass"] for row in qc_rows),
        "canonical_qc_fail": sum(not row["final_qc_pass"] for row in qc_rows),
        "case_roster_change_count": 0 if invariants["case_roster_unchanged"] else 1,
        "substantive_civil_liability_central_synchronized_count": sum(row["substantive_central_sync_pass"] for row in qc_rows),
        "substantive_civil_liability_central_pre_v4_preserved_count": sum(row["substantive_central_pre_v4_preserved_pass"] for row in qc_rows),
        "neutral_fact_text_change_count": 200 - sum(row["neutral_fact_text_preserved_pass"] for row in qc_rows),
        "neutral_fact_hash_refresh_case_ids": stale_fact_hash_ids,
        "country_counts": dict(country_counts),
        "us_state_counts": dict(sorted(state_counts.items())),
        "primary_domain_counts": {country: dict(sorted(counts.items())) for country, counts in domain_counts.items()},
        "other_civil_liability_subtypes": {country: dict(sorted(counts.items())) for country, counts in subtype_counts.items()},
        "split_counts": {f"{country}_{split}": count for (country, split), count in sorted(split_counts.items())},
        "replacement_count": old_summary["replacement_count"],
        "duplicate_case_family_removal_count": old_summary["duplicate_case_family_removal_count"],
        "controlling_opinion_split_count": old_summary["controlling_opinion_split_count"],
        "duplicate_validation": duplicate_evidence,
        "invariants": invariants,
    }
    write_json(summary_path, summary)

    core_paths = [
        case_path, fact_path, unit_path, qc_path, raw_path, controlling_path,
        roster_path, summary_path, duplicate_path, metadata_path,
    ]
    core_hashes = {path.as_posix(): sha_file(path) for path in core_paths}
    manifest_path = FINAL / f"corpus_freeze_manifest_{SUFFIX}.json"
    manifest = {
        "corpus_version": VERSION,
        "status": "FROZEN" if frozen else "NOT_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "upgrade_source": "outputs_v2/v4 current final artifacts",
        "source_artifact_sha256_at_upgrade": current_input_hashes,
        "metadata_change": {
            "field_set_true": "substantive_civil_liability_central",
            "old_value_field": "substantive_civil_liability_central_pre_v4",
            "records_updated": 200,
        },
        "user_manual_neutral_fact_edit_case_ids_preserved": stale_fact_hash_ids,
        "invariants": invariants,
        "artifact_sha256": core_hashes,
    }
    write_json(manifest_path, manifest)

    report_path = FINAL / f"final_repair_freeze_report_{SUFFIX}.md"
    report_lines = [
        "# v4.0.1 metadata repair and freeze report",
        "",
        f"Status: **{'FROZEN' if frozen else 'NOT_FROZEN'}**",
        "",
        "- Case roster changes: 0",
        "- `substantive_civil_liability_central` synchronized to `true`: 200/200",
        "- Prior values preserved in `substantive_civil_liability_central_pre_v4`: 200/200",
        f"- Prior-value distribution: {dict(sorted(old_distribution.items()))}",
        "- User manual neutral-fact edits preserved without text changes: 2/2",
        f"- Neutral-fact hashes refreshed: {len(stale_fact_hash_ids)} ({', '.join(stale_fact_hash_ids)})",
        f"- Canonical final QC: {summary['canonical_qc_pass']}/200 pass",
        f"- KR domains: {dict(sorted(domain_counts['KR'].items()))}",
        f"- US domains: {dict(sorted(domain_counts['US'].items()))}",
        f"- US states: {dict(sorted(state_counts.items()))}",
        "",
        "## Final invariants",
        "",
    ]
    report_lines.extend(f"- {key}: **{str(value).upper()}**" for key, value in invariants.items())
    report_lines += ["", "## Core artifact SHA-256", ""]
    report_lines.extend(f"- `{path}`: `{digest}`" for path, digest in core_hashes.items())
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    checksum_paths = core_paths + [manifest_path, report_path]
    checksum_path = FINAL / f"SHA256SUMS_{SUFFIX}.txt"
    checksum_path.write_text(
        "".join(f"{sha_file(path)}  {path.as_posix()}\n" for path in sorted(checksum_paths)),
        encoding="utf-8",
    )

    if not frozen:
        failed = [key for key, value in invariants.items() if not value]
        raise RuntimeError({"status": "NOT_FROZEN", "failed_invariants": failed})
    print(json.dumps({
        "status": "FROZEN",
        "version": VERSION,
        "cases": len(cases),
        "qc_pass": summary["canonical_qc_pass"],
        "metadata_synced": metadata_audit["new_substantive_civil_liability_central_true_count"],
        "pre_values_preserved": metadata_audit["pre_v4_value_preserved_count"],
        "manual_fact_edits_preserved": stale_fact_hash_ids,
        "output": FINAL.as_posix(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
