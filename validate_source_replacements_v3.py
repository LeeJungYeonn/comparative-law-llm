from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_jsonl


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reopen and validate externally flagged source replacements.")
    p.add_argument("--cases", type=Path, default=Path("outputs_v2/final_cases_200.jsonl"))
    p.add_argument("--audit", type=Path, default=Path("outputs_v2/neutral_fact_qc_audit_200.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs_v2/source_replacement_validation_v3.jsonl"))
    p.add_argument("--overwrite", action="store_true")
    return p


def reason_code(note: str) -> str:
    value = note.casefold()
    for needle, code in (
        ("attorney disciplinary", "attorney_disciplinary"),
        ("workers' compensation", "workers_compensation"),
        ("insurance coverage", "insurance_coverage_only"),
        ("insurance-contract", "insurance_coverage_only"),
        ("insurer", "insurance_coverage_only"),
        ("notice/statute-of-limitations", "limitations_or_notice_only"),
        ("prescription/direct-action", "limitations_or_notice_only"),
        ("claim/issue-preclusion", "claim_or_issue_preclusion"),
        ("personal jurisdiction", "personal_jurisdiction_only"),
        ("controlling-opinion", "unusable_controlling_opinion"),
        ("extraordinary-writ", "extraordinary_writ_only"),
    ):
        if needle in value:
            return code
    return "other_non_merits_proceeding"


PATTERNS = {
    "attorney_disciplinary": r"ATTORNEY DISCIPLINARY PROCEEDINGS|disciplinary (?:matter|proceeding)",
    "workers_compensation": r"workers['’]? compensation",
    "insurance_coverage_only": r"insurance coverage|pollution exclusion|duty to (?:defend|indemnify)|declaratory (?:judgment|action)|insurance policy",
    "limitations_or_notice_only": r"statute of limitations|notice of intent|prescription|prescriptive period|direct action",
    "claim_or_issue_preclusion": r"claim preclusion|issue preclusion|res judicata",
    "personal_jurisdiction_only": r"personal jurisdiction|writ of prohibition",
    "unusable_controlling_opinion": r"dissenting|writ of prohibition|extraordinary writ",
    "extraordinary_writ_only": r"writ of mandamus|writ of prohibition|settlement agreement|allocation of fees",
    "other_non_merits_proceeding": r"appeal|proceeding|jurisdiction|procedure",
}


def evidence(source: str, code: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(PATTERNS[code], source, re.I):
        start = max(0, match.start() - 140)
        end = min(len(source), match.end() + 220)
        span = source[start:end].strip()
        if span and span not in found:
            found.append(span)
        if len(found) == 2:
            break
    return found


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cases = {row["case_id"]: row for row in read_jsonl(args.cases)}
    audit = [row for row in read_jsonl(args.audit) if row.get("case_level_status") == "replacement_required"]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in audit:
        case = cases.get(item["case_id"])
        if not case:
            failures.append(f"{item['case_id']}:missing_source_record")
            continue
        code = reason_code(item.get("case_level_note") or "")
        source = case.get("main_opinion_text") or case.get("full_opinion_text") or ""
        spans = evidence(source, code)
        if not spans:
            failures.append(f"{item['case_id']}:no_source_evidence:{code}")
        rows.append({
            "case_id": item["case_id"], "prior_review_status": "replacement_required",
            "source_recheck_status": "confirmed_replace" if spans else "unresolved",
            "reason_code": code, "evidence_spans": spans,
            "notes": item.get("case_level_note"),
        })
    write_jsonl(args.output, rows)
    print({"replacement_required": len(audit), "confirmed_replace": sum(r["source_recheck_status"] == "confirmed_replace" for r in rows), "failures": failures})
    return 0 if len(rows) == len(audit) and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
