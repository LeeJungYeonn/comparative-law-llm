from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_json, write_jsonl


OUT = Path("outputs_v2")
WORK = OUT / "v4_repair"
VERSION = "kr-us-highcourt-corpus-v4.0"

# Every removal is tied either to an independent-QC HARD row, a direct
# controlling-opinion defect found while rechecking all U.S. texts, or the
# minimum two KR swaps needed after recomputing the actual replacement domains.
REPLACEMENTS: dict[str, tuple[str, str]] = {
    "US_a07db3e0a97840c5b1": ("US_59362b11f24be41b92", "extraordinary-writ-only opinion"),
    "US_c387cc2f242da3c0dc": ("US_a15bd1ea6aa4ea1ed9", "administrative prison-policy appeal"),
    "US_b2c7f8c81dc1b0c361": ("US_5374560fdfdf536ed1", "public civil-enforcement retroactivity case"),
    "US_0852f7949bcaa2bb53": ("US_4a0f056bcec6698d7d", "jury-selection procedure was the high-court issue"),
    "US_1cfe591dc59be857da": ("US_3aa1784247c016e937", "leave-denial order; substance appeared only in separate opinions"),
    "US_8636c5a0c5ede55860": ("US_979762a92670a0552a", "leave-denial order; substance appeared only in dissent; the initially considered wet-floor candidate was also rejected because its controlling opinion addressed only lost-evidence procedure"),
    "US_9b0dcb923fdb190dd3": ("US_6fc55880fb74718547", "leave-denial order; substance appeared only in concurrence"),
    "US_8026310f8d6a3819bd": ("US_f57bc43fc6478fd8b2", "controlling decision was issued in 1999"),
    "US_5905b849c15a6fb3d6": ("US_384364b54638f78623", "duplicate Grove litigation/case family"),
    "US_9cac14d9a55b155cd4": ("US_b675276d7ca44f64e2", "duplicate Craig litigation/case family"),
    "US_fa07baffd06e422309": ("US_472f038dea6238a51e", "stored reference text is a concurring-and-dissenting opinion, not the majority"),
    "US_67cc0f02a85dfe7f69": ("US_a4ba9463b694840f34", "stored reference text is a concurrence/opinion in support of affirmance, not a controlling majority"),
    "US_e0f4e43491c639af90": ("US_0ffb071db74bea0121", "stored reference text is a dissent, not the controlling majority"),
    "US_231c5e7fb867d9b8c2": ("US_2999172a4de14d7523", "source-scope recheck: contractual indemnity and defense-cost allocation, not direct civil-liability merits"),
    "US_04cf5031ab373b2ef6": ("US_107f5beb1553bc775c", "source-scope recheck: multi-claim mortgage pleading appeal, not a clean substantive civil-liability merits case"),
    "US_d15af88bf67507bd93": ("US_b267b8668935378e25", "source-scope recheck: implied-indemnity defense costs after the underlying action"),
    "US_7f556627654c5d3f9a": ("US_97456da392d82990a3", "source-scope recheck: post-judgment compensation-fund payment procedure"),
    "US_74c861440b330e42dc": ("US_0bdc6cb112770284df", "source-scope recheck: the controlling opinion concerns discovery conduct and trial sanctions in a construction-contract dispute, not civil-liability merits"),
    "KR_4a417b3e4eb96c19c1": ("KR_c0c025922aca493f10", "HARD judicial-evaluation leakage; minimum product-to-general domain swap"),
    "KR_f3fe352a4c02d16f80": ("KR_d47698374deaa59285", "HARD fact insufficiency: controlling source omitted the accident conduct and causal sequence"),
    "KR_5beca7bd705b05bb94": ("KR_25ab15d21ac1967afd", "HARD fact insufficiency: controlling source omitted the patient outcome and surrounding treatment sequence"),
    "KR_adf561060b125d6b87": ("KR_e61f8571b55fd48e4c", "HARD legal-rule leakage and controlling source omitted the accident conduct needed for a sufficient neutral fact"),
    "KR_882ce243b36c889fda": ("KR_1346b7ea49f678b93c", "source-scope recheck: controlling opinion centered on a special-hiring clause rather than the civil-liability merits"),
    "KR_1e296dbb0bcebaab8f": ("KR_6a3e8f577c00b9c20c", "procedural-role wording; minimum other-to-medical domain swap"),
    "KR_c9b0b61cc030200a6a": ("KR_a1301398c0525889ea", "semantic redundancy; minimum additional other-to-general swap after rejecting a fee-only U.S. candidate"),
    "KR_c7ec53e12d00dabce6": ("KR_fd8f69f92c35adf0ac", "minimum other-to-general swap required after direct source review reclassified the Nevada replacement"),
    "KR_293763f5a50c79c279": ("KR_6078abc75c440541ab", "minimum other-to-general domain swap required by the U.S. source-scope replacements"),
    "KR_644078b169f572b0fe": ("KR_8f9a8d4ff8f0f379fb", "minimum other-to-general domain swap required by the U.S. source-scope replacements"),
    "KR_d35502ba94fbfc6c92": ("KR_de99291f16b6fe1b2a", "minimum other-to-general domain swap required by the U.S. source-scope replacements"),
    "KR_c2567a4fb9e71fc1f7": ("KR_46eecae742ccab2781", "minimum general-to-medical swap required after rejecting the procedure-only wet-floor U.S. replacement"),
}

MANUAL_ELIGIBILITY = {
    "US_5374560fdfdf536ed1": "retaliatory employment action and a damages verdict are the substantive merits",
    "US_4a0f056bcec6698d7d": "negligent sponsorship/endorsement and claimed financial loss are the substantive merits",
    "US_0bdc6cb112770284df": "direct recovery by an accident victim against the vehicle owner's insurer is the substantive damages merits",
    "US_0ffb071db74bea0121": "the controlling opinion reviews causation and damages from the vehicle collision on the merits",
    "US_107f5beb1553bc775c": "the controlling opinion reviews the injured worker's premises and equipment accident on the merits",
    "US_2999172a4de14d7523": "the controlling opinion reviews responsibility for the falling-truss injury on the merits",
    "US_3aa1784247c016e937": "the controlling opinion allocates responsibility and damages for the highway collision on the merits",
}

MANUAL_DOMAINS = {
    "US_0bdc6cb112770284df": "other_civil_liability",
}

DATE_CORRECTIONS = {
    "US_45c1de8fa5ac8eefab": ("2003-07-03", "opinion header: Decided July 3, 2003; July 7 is the concurrence date"),
    "US_9c0d0f392a9c3505c1": ("2000-07-18", "opinion header: Decided July 18, 2000; September 19 is the amendment date"),
}

# Heading forms used by the five state courts when a combined source object
# appends a separate opinion to the controlling opinion.  The patterns are
# anchored to paragraph/line starts to avoid cutting citations or narrative
# references to other cases.
SEPARATE_HEADING_PATTERNS = (
    re.compile(r"(?im)^\s*(?:CONCURRING(?:\s+AND\s+DISSENTING|\s+IN\s+PART\s+AND\s+DISSENTING\s+IN\s+PART)?|DISSENTING)\s+OPINION\b"),
    re.compile(r"(?im)^\s*(?:CHIEF\s+)?JUSTICE\s+[A-Z][A-Z .'-]{1,40},\s*(?:Concurring|Dissenting)\b"),
    re.compile(r"(?im)^\s*[A-Z][A-Z .'-]{1,40},\s*J\.,\s*(?:concurring|dissenting)\b"),
    re.compile(r"(?im)^\s*(?:Chief\s+)?Justice\s+[A-Z][A-Za-z .'-]{1,40},\s*(?:Concurring|Dissenting)\."),
)

# These source objects have no reliable line breaks.  Each marker below was
# checked in the original opinion and is the first heading after the
# controlling majority/lead disposition.  This avoids citation-based false
# splits while excluding the appended separate-opinion discussion.
VERIFIED_SEPARATE_OPINION_MARKERS = {
    "US_03aa3f07d9d0e5332a": "BECKER, J., concurring in part and dissenting in part.",
    "US_061c987e5f6b8cf2e8": "Chief Justice CAPPY, Concurring.",
    "US_0cdf24e76568435f05": "30 STATE OF MICHIGAN SUPREME COURT",
    "US_0eaf2fe14baa5be05e": "NIGRO, Justice, dissenting.",
    "US_1042602a184dd1c50b": "JOHNSON, J. dissents, assigning reasons:",
    "US_143504a555ba2b1572": "KIMBALL, Justice, dissenting.",
    "US_157a99d9bc34ad07c0": "MARILYN J. KELLY, J. (concurring in part and dissenting in part).",
    "US_1aba36d98e0653f066": "MICHAEL F. CAVANAGH, J. (concurring in part and dissenting in part).",
    "US_2dc42d5d55548348de": "49 STATE OF MICHIGAN SUPREME COURT",
    "US_32ce5cc46af8c47b58": "JOHNSON, Justice dissenting.",
    "US_38d924f1ad3609200a": "Justice NEWMAN, concurring and dissenting.",
    "US_3aa1784247c016e937": "VICTORY, J., dissenting.",
    "US_3f0101285fcc76e2b1": "Justice NEWMAN, dissenting.",
    "US_45c1de8fa5ac8eefab": "MAYNARD, Justice, dissenting:",
    "US_46a80b1c5e06a72d45": "Justice SAYLOR, concurring.",
    "US_47210f2421779c756f": "Justice EAKIN, dissenting.",
    "US_5d2b0b2b4752431c53": "Justice NEWMAN, concurring and dissenting.",
    "US_60f81ce937468b4218": "MARILYN J. KELLY, J. (dissenting).",
    "US_63ffb974f8488a5664": "CONCURRING OPINION Justice SAYLOR.",
    "US_667a9b663657012cdc": "21 No. 21-0096",
    "US_6d3421be0f42fb2f6b": "14 STATE OF MICHIGAN SUPREME COURT",
    "US_6e94c6d394acfdcdc7": "Justice EAKIN, concurring.",
    "US_77921eca4293bb5c27": "Justice CASTILLE, concurring.",
    "US_78f2e4c0127c877a51": "HATHAWAY, J. I concur",
    "US_7cae584411e6f22f4b": "43 S T A T E O F M I C H I G A N SUPREME COURT",
    "US_8f953fe2f1a7911efc": "GIBBONS, J., concurring in part and dissenting in part.",
    "US_97456da392d82990a3": "LEMMON, J., Subscribes to the Opinion and Assigns Additional Reasons.",
    "US_9acb209b5020b6dec4": "VICTORY, J., dissenting.",
    "US_9c0d0f392a9c3505c1": "MARILYN J. KELLY, J. (dissenting).",
    "US_9dc742b4409b555931": "PICKERING, J., concurring in part and dissenting in part:",
    "US_a0047ad180b5e21d95": "JOHNSON, Justice, dissents and assigns reasons. The majority concludes",
    "US_b24ef2018816af9946": "14 06/30/15 SUPREME COURT OF LOUISIANA",
    "US_b267b8668935378e25": "WRITING SEPARATELY:",
    "US_b473e7585c6e5638dd": "Viviano, J. (concurring in result only ).",
    "US_b732d6870be7eac6fb": "35 STATE OF MICHIGAN SUPREME COURT",
    "US_cc8f60960ef40a7434": "NIGRO, Justice, dissenting.",
    "US_e05b7f2567e5800f56": "VICTORY, J., dissenting.",
    "US_f5ad9cb96e57eb50d6": "YOUNG, J. (concurring).",
    "US_f57bc43fc6478fd8b2": "JOHNSON, J., concurring in part and dissenting in part.",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_candidates() -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for name in (
        "us_replacement_candidates_review_v3.jsonl",
        "us_replacement_candidates_review_remaining_v3.jsonl",
        "us_replacement_candidates_review_extra_v3.jsonl",
        "kr_replacement_candidates_review_v3.jsonl",
    ):
        for row in read_jsonl(OUT / name):
            candidates[row["case_id"]] = row
    reviews: dict[str, dict[str, Any]] = {}
    for name in (
        "us_candidate_domain_reclassification_v3.jsonl",
        "us_candidate_domain_reclassification_remaining_v3.jsonl",
        "us_candidate_domain_reclassification_extra_v3.jsonl",
        "kr_candidate_domain_reclassification_v3.jsonl",
    ):
        for row in read_jsonl(OUT / name):
            reviews[row["case_id"]] = row
    for case_id, review in reviews.items():
        if case_id not in candidates:
            continue
        candidates[case_id].update({
            "eligible_main_corpus": review.get("eligible_main_corpus"),
            "primary_domain": review.get("primary_domain"),
            "case_domain": review.get("primary_domain"),
            "domain_review_status": review.get("domain_review_status"),
            "domain_evidence": review.get("domain_evidence_spans") or [],
            "eligibility_evidence": review.get("eligibility_evidence_spans") or [],
            "liability_theories": review.get("liability_theories") or candidates[case_id].get("liability_theories") or [],
        })
    # The full collected candidate pool remains authoritative for a directly
    # reviewed replacement that was not included in the smaller review batch.
    for row in read_jsonl(OUT / "us_state_highcourt_candidates.jsonl"):
        if row["case_id"] in MANUAL_ELIGIBILITY and row["case_id"] not in candidates:
            candidates[row["case_id"]] = row
    return candidates


def split_controlling_opinion(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = case.get("main_opinion_text") or ""
    starts: list[tuple[int, str]] = []
    verified_marker = VERIFIED_SEPARATE_OPINION_MARKERS.get(case["case_id"])
    if verified_marker:
        marker_offset = raw.find(verified_marker, 1800)
        if marker_offset < 0:
            raise RuntimeError(f"Verified separate-opinion marker missing: {case['case_id']}")
        starts.append((marker_offset, verified_marker))
    for pattern in SEPARATE_HEADING_PATTERNS:
        for match in pattern.finditer(raw):
            # A genuine appended section needs a substantial controlling text
            # before it.  Two source records whose first section is separate
            # were removed above instead of being truncated.
            if match.start() >= 1800:
                starts.append((match.start(), match.group(0).strip()))
    if starts:
        boundary, heading = min(starts)
        clean = raw[:boundary].rstrip()
        method = "split_before_appended_separate_opinion"
    else:
        boundary, heading, clean = None, None, raw.strip()
        method = "source_opinion_object_already_controlling"
    if len(clean) < 1200:
        raise RuntimeError(f"Controlling opinion too short after split: {case['case_id']}")
    detail = {
        "status": "validated_controlling_merits_opinion",
        "method": method,
        "raw_main_opinion_chars": len(raw),
        "controlling_opinion_chars": len(clean),
        "split_offset": boundary,
        "first_excluded_heading": heading,
        "raw_main_opinion_sha256": sha256_text(raw),
        "controlling_opinion_sha256": sha256_text(clean),
    }
    return clean, detail


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    original = list(read_jsonl(OUT / "final_cases_200_v3.jsonl"))
    candidates = load_candidates()
    by_id = {row["case_id"]: row for row in original}
    missing_old = sorted(set(REPLACEMENTS) - set(by_id))
    missing_new = sorted({new for new, _ in REPLACEMENTS.values()} - set(candidates))
    if missing_old or missing_new:
        raise RuntimeError({"missing_old": missing_old, "missing_new": missing_new})

    removed = set(REPLACEMENTS)
    cases = [dict(row) for row in original if row["case_id"] not in removed]
    replacement_rows = []
    for old_id, (new_id, reason) in REPLACEMENTS.items():
        row = dict(candidates[new_id])
        if not row.get("eligible_main_corpus") and new_id not in MANUAL_ELIGIBILITY:
            raise RuntimeError(f"Replacement lacks source-level eligibility: {new_id}")
        if new_id in MANUAL_ELIGIBILITY:
            row["eligible_main_corpus"] = True
            row["strict_source_eligible"] = True
            row["domain_review_status"] = "direct_source_adjudication_v4"
            row.setdefault("eligibility_evidence", []).append(MANUAL_ELIGIBILITY[new_id])
        if new_id in MANUAL_DOMAINS:
            row["primary_domain"] = MANUAL_DOMAINS[new_id]
            row["case_domain"] = MANUAL_DOMAINS[new_id]
            row["domain_review_status"] = "direct_source_adjudication_v4"
        row["replacement_status"] = "v4_targeted_replacement"
        row["replaces_case_id"] = old_id
        row["replacement_reason_v4"] = reason
        row["corpus_version"] = VERSION
        cases.append(row)
        replacement_rows.append({
            "old_case_id": old_id,
            "new_case_id": new_id,
            "origin_country": row["origin_country"],
            "old_domain": by_id[old_id]["primary_domain"],
            "new_domain": row["primary_domain"],
            "old_state": by_id[old_id].get("origin_state"),
            "new_state": row.get("origin_state"),
            "reason": reason,
        })

    if len(cases) != 200 or len({row["case_id"] for row in cases}) != 200:
        raise RuntimeError("Provisional v4 roster is not 200 unique cases")

    controlling_rows = []
    for case in cases:
        case["corpus_version"] = VERSION
        if case["case_id"] in DATE_CORRECTIONS:
            old_date = case.get("decision_date")
            case["decision_date"], reason = DATE_CORRECTIONS[case["case_id"]]
            case["secondary_opinion_or_amendment_date"] = old_date
            case["decision_date_correction_v4"] = reason
        if case["origin_country"] != "US":
            continue
        raw = case.get("main_opinion_text") or ""
        clean, validation = split_controlling_opinion(case)
        case["raw_main_opinion_text"] = raw
        case["controlling_opinion_text"] = clean
        case["main_opinion_text"] = clean
        case["main_opinion_type_raw"] = case.get("main_opinion_type")
        case["main_opinion_type"] = "controlling_majority_or_lead"
        case["controlling_opinion_validation"] = validation
        case["raw_text_chars"] = len(clean)
        case["raw_text_sha256"] = sha256_text(clean)
        controlling_rows.append({
            "case_id": case["case_id"],
            "case_name": case.get("case_name"),
            "origin_state": case.get("origin_state"),
            "decision_date": case.get("decision_date"),
            **validation,
        })

    cases.sort(key=lambda row: row["case_id"])
    write_jsonl(WORK / "provisional_cases_200_v4.jsonl", cases)
    write_jsonl(WORK / "us_controlling_opinion_validation_v4.jsonl", controlling_rows)

    with (OUT / "v3_independent_qc_issues.csv").open(encoding="utf-8-sig", newline="") as handle:
        issues = list(csv.DictReader(handle))
    issue_targets = {
        row["case_id"] for row in issues
        if row["case_id"] and row["layer"] == "neutral_fact" and row["case_id"] not in removed
    }
    new_ids = {new for new, _ in REPLACEMENTS.values()}
    extraction_ids = issue_targets | new_ids
    targeted = [row for row in cases if row["case_id"] in extraction_ids]
    write_jsonl(WORK / "targeted_fact_repair_cases_v4.jsonl", targeted)

    country = Counter(row["origin_country"] for row in cases)
    state = Counter(row.get("origin_state") for row in cases if row["origin_country"] == "US")
    domains = {
        name: Counter(row["primary_domain"] for row in cases if row["origin_country"] == name)
        for name in ("KR", "US")
    }
    plan = {
        "status": "not_frozen_targeted_repair_in_progress",
        "corpus_version": VERSION,
        "authoritative_qc_inputs": [
            "outputs_v2/v3_independent_quality_control_report.md",
            "outputs_v2/v3_independent_qc_issues.csv",
            "outputs_v2/v3_independent_qc_summary.json",
        ],
        "replacement_count": len(replacement_rows),
        "replacements": replacement_rows,
        "targeted_fact_repair_count": len(targeted),
        "targeted_fact_repair_case_ids": sorted(extraction_ids),
        "provisional_country_counts": dict(country),
        "provisional_state_counts": dict(state),
        "provisional_domain_counts": {key: dict(value) for key, value in domains.items()},
    }
    write_json(WORK / "replacement_plan_v4.json", plan)
    print(json.dumps({
        "cases": len(cases),
        "replacements": len(replacement_rows),
        "targeted_fact_repairs": len(targeted),
        "state_counts": dict(state),
        "domain_counts": {key: dict(value) for key, value in domains.items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
