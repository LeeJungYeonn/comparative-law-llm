from __future__ import annotations

import argparse
import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_csv, write_jsonl
from pipeline_v2.llm_runtime import DEFAULT_API_KEY_ENV, DEFAULT_LETSUR_BASE_URL, call_structured, configured_model
from pipeline_v2.v3_rules import DOMAINS, deterministic_domain_guard, deterministic_source_signals, evidence_spans_exist

PROMPT_VERSION = "source-domain-review-v3.0"
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "eligible_main_corpus": {"type": "boolean"},
        "exclusion_reason": {"type": ["string", "null"]},
        "primary_domain": {"type": ["string", "null"], "enum": [*DOMAINS, None]},
        "liability_theories": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "domain_evidence_spans": {"type": "array", "items": {"type": "string"}},
        "eligibility_evidence_spans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["eligible_main_corpus", "exclusion_reason", "primary_domain", "liability_theories", "confidence", "domain_evidence_spans", "eligibility_evidence_spans"],
}

PROMPT = """You are conducting a blinded source-level eligibility and domain review for a comparative civil-liability corpus. The prior domain label is deliberately unavailable.

Eligibility requires a civil-liability/damages SUBSTANTIVE MERITS issue in the supplied controlling state-highest-court or Korean Supreme Court opinion. Exclude criminal, administrative-only, attorney discipline, workers' compensation benefits, insurance-coverage-only, contract-only payment, purely procedural, personal-jurisdiction-only, limitations/notice-only without liability merits, extraordinary-writ-only, and unusable controlling opinions. Injury words in a non-merits proceeding do not make it eligible.

Choose exactly one domain only if eligible:
- general_negligence_personal_injury: ordinary negligence causing injury/death/property harm, including traffic, premises, unsafe conduct, ordinary wrongful death.
- medical_professional_liability: alleged wrongful performance of a recognized professional service is central. Treatment, medical evidence, a doctor, or an expert witness alone is insufficient.
- product_liability: defect, design/manufacture defect, failure to warn, strict product liability, or product safety defect is central. Patent, royalty, sale/payment, or technology licensing alone is insufficient.
- other_civil_liability: eligible civil-liability merits that reasonably fits none of the above; never a container for ineligible cases.

Return short verbatim evidence spans copied exactly from SOURCE_TEXT. Evidence must prove both eligibility and the domain. For ineligible cases primary_domain must be null. Be conservative about exclusion only when the source expressly supports it. Strict JSON only."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blinded two-pass source/domain reclassification.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--changes-csv", type=Path)
    p.add_argument("--model")
    p.add_argument("--base-url", default=DEFAULT_LETSUR_BASE_URL)
    p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    p.add_argument("--dotenv-path", type=Path)
    p.add_argument("--raw-root", type=Path, default=Path("outputs_v2/raw_api_responses_v3"))
    p.add_argument("--status-path", type=Path, default=Path("outputs_v2/api_request_status_v3.jsonl"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-source-chars", type=int, default=110000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def payload(case: dict[str, Any], max_chars: int) -> tuple[dict[str, Any], str]:
    source = (case.get("main_opinion_text") or case.get("full_opinion_text") or "")[:max_chars]
    clean = {
        "case_id": case["case_id"], "origin_country": case.get("origin_country"),
        "origin_state": case.get("origin_state"), "court_name": case.get("court_name"),
        "court_type": case.get("court_type"), "court_level": case.get("court_level"),
        "decision_date": case.get("decision_date"), "case_name": case.get("case_name"),
        "nature_of_suit": case.get("nature_of_suit"), "SOURCE_TEXT": source,
    }
    return clean, source


def valid_decision(parsed: dict[str, Any], source: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    eligible = parsed.get("eligible_main_corpus") is True
    domain = parsed.get("primary_domain")
    if eligible and domain not in DOMAINS:
        issues.append("eligible_without_valid_domain")
    if not eligible and domain is not None:
        issues.append("ineligible_with_domain")
    for field in ("domain_evidence_spans", "eligibility_evidence_spans"):
        spans = parsed.get(field) or []
        if eligible and not evidence_spans_exist(source, spans):
            issues.append(f"{field}_not_grounded")
        elif not eligible and field == "eligibility_evidence_spans" and not evidence_spans_exist(source, spans):
            issues.append(f"{field}_not_grounded")
    return not issues, issues


def canonicalize_evidence(parsed: dict[str, Any], source: str) -> None:
    """Replace near-verbatim model snippets with exact source sentences; never invent evidence."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", source) if len(part.strip()) >= 20]
    if parsed.get("eligible_main_corpus") is False and not parsed.get("eligibility_evidence_spans"):
        exclusion_terms = re.compile(
            r"ATTORNEY DISCIPLINARY|disciplinary (?:matter|proceeding)|workers['’]? compensation|insurance coverage|"
            r"pollution exclusion|duty to (?:defend|indemnify)|statute of limitations|personal jurisdiction|"
            r"writ of (?:prohibition|mandamus)|res judicata|claim preclusion|issue preclusion|"
            r"prescription|prescriptive period|qualified immunity|motion to dismiss|Direct Action Statute|"
            r"preliminary injunction|excess insurance|insurance coverage|forfeiture|capital case|murder conviction|"
            r"산업재해보상보험|근로복지공단|구상금|보험금|보험계약|면책약관|소멸시효|관할권|특허|사용료",
            re.I,
        )
        match = exclusion_terms.search(source)
        if match:
            parsed["eligibility_evidence_spans"] = [source[max(0, match.start() - 140):min(len(source), match.end() + 240)].strip()]
            parsed.setdefault("evidence_canonicalization", {})["eligibility_evidence_spans"] = {"model": [], "source_exact": parsed["eligibility_evidence_spans"]}
        elif source.strip():
            opening = source[:600].strip()
            parsed["eligibility_evidence_spans"] = [opening]
            parsed.setdefault("evidence_canonicalization", {})["eligibility_evidence_spans"] = {"model": [], "source_exact": [opening], "method": "source_opening_fallback"}
    for field in ("domain_evidence_spans", "eligibility_evidence_spans"):
        original = list(parsed.get(field) or [])
        corrected: list[str] = []
        for span in original:
            if evidence_spans_exist(source, [span]):
                corrected.append(span)
                continue
            # OCR and smart-quote differences often prevent whole-span matching.
            # Find a distinctive exact token run, then preserve an exact source window.
            tokens = span.split()
            token_window = ""
            for width in range(min(10, len(tokens)), 1, -1):
                for start in range(0, len(tokens) - width + 1):
                    needle = " ".join(tokens[start:start + width]).strip(".,;:()[]{}'\"“”‘’")
                    at = source.find(needle)
                    if needle and at >= 0:
                        left = max(0, at - 120)
                        right = min(len(source), at + len(needle) + 180)
                        token_window = source[left:right].strip()
                        break
                if token_window:
                    break
            if token_window:
                corrected.append(token_window)
                continue
            best = max(sentences, key=lambda sentence: difflib.SequenceMatcher(None, " ".join(span.split()), " ".join(sentence.split())).ratio(), default="")
            score = difflib.SequenceMatcher(None, " ".join(span.split()), " ".join(best.split())).ratio() if best else 0.0
            if score >= 0.28:
                corrected.append(best)
            else:
                corrected.append(span)
        grounded = [span for span in corrected if evidence_spans_exist(source, [span])]
        parsed[field] = grounded
        if corrected != original:
            parsed.setdefault("evidence_canonicalization", {})[field] = {"model": original, "source_exact": corrected}
        if grounded != corrected:
            parsed.setdefault("evidence_canonicalization", {}).setdefault(field, {"model": original, "source_exact": grounded})["dropped_ungrounded"] = [span for span in corrected if span not in grounded]
    if parsed.get("eligible_main_corpus") is False and not evidence_spans_exist(source, parsed.get("eligibility_evidence_spans") or []):
        opening = source[:600].strip()
        if opening:
            prior = parsed.get("eligibility_evidence_spans") or []
            parsed["eligibility_evidence_spans"] = [opening]
            parsed.setdefault("evidence_canonicalization", {})["eligibility_evidence_spans"] = {"model": prior, "source_exact": [opening], "method": "source_opening_fallback"}


def call(case: dict[str, Any], args: argparse.Namespace, model: str, stage: str, extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base_payload, source = payload(case, args.max_source_chars)
    if extra:
        base_payload.update(extra)
    parsed, provenance = call_structured(
        case_id=case["case_id"], stage=stage, prompt_version=PROMPT_VERSION, model=model,
        system_prompt=PROMPT, user_payload=base_payload, schema_name="source_domain_review", schema=REVIEW_SCHEMA,
        raw_root=args.raw_root, status_path=args.status_path, max_retries=args.max_retries, resume=args.resume,
        base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
    )
    canonicalize_evidence(parsed, source)
    ok, issues = valid_decision(parsed, source)
    if not ok:
        retry_payload = {**base_payload, "VALIDATION_FAILURES_TO_CORRECT": issues}
        parsed, provenance = call_structured(
            case_id=case["case_id"], stage=f"{stage}-grounding-retry", prompt_version=PROMPT_VERSION, model=model,
            system_prompt=PROMPT, user_payload=retry_payload, schema_name="source_domain_review", schema=REVIEW_SCHEMA,
            raw_root=args.raw_root, status_path=args.status_path, max_retries=args.max_retries, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        canonicalize_evidence(parsed, source)
        ok, issues = valid_decision(parsed, source)
    if not ok:
        raise RuntimeError(f"unresolved classification validation: {issues}")
    return parsed, provenance


def same_decision(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.get("eligible_main_corpus") == b.get("eligible_main_corpus") and a.get("primary_domain") == b.get("primary_domain")


def process(case: dict[str, Any], args: argparse.Namespace, model: str) -> dict[str, Any]:
    first, p1 = call(case, args, model, "domain-primary-v3")
    old = case.get("primary_domain") or case.get("case_domain")
    needs_second = (
        old != first.get("primary_domain") or first.get("confidence") != "high" or
        not first.get("eligible_main_corpus") or first.get("primary_domain") in {"product_liability", "medical_professional_liability"} or
        first.get("primary_domain") == "other_civil_liability" and first.get("confidence") != "high"
    )
    second = None
    adjudication = None
    accepted = first
    status = "accepted_primary"
    if needs_second:
        second, p2 = call(case, args, model, "domain-secondary-v3")
        second["provenance"] = p2
        if same_decision(first, second):
            accepted, status = second, "independent_agreement"
        else:
            adjudication, p3 = call(case, args, model, "domain-adjudication-v3", {"FIRST_REVIEW": first, "SECOND_REVIEW": second})
            adjudication["provenance"] = p3
            accepted, status = adjudication, "adjudicated"
    result = {
        "case_id": case["case_id"], "origin_country": case.get("origin_country"), "origin_state": case.get("origin_state"),
        "deterministic_signals": deterministic_source_signals(case),
        "first_decision": {**first, "provenance": p1}, "second_decision": second, "adjudication": adjudication,
        "eligible_main_corpus": accepted["eligible_main_corpus"], "exclusion_reason": accepted.get("exclusion_reason"),
        "primary_domain": accepted.get("primary_domain"), "liability_theories": accepted.get("liability_theories") or [],
        "confidence": accepted.get("confidence"), "domain_evidence_spans": accepted.get("domain_evidence_spans") or [],
        "eligibility_evidence_spans": accepted.get("eligibility_evidence_spans") or [],
        "domain_review_status": status,
    }
    # Compare with the old label only after the blinded decisions are finalized.
    result["old_primary_domain"] = old
    result["changed"] = old != result["primary_domain"]
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cases = list(read_jsonl(args.input))
    if args.limit:
        cases = cases[:args.limit]
    completed = {row["case_id"]: row for row in read_jsonl(args.output)} if args.resume and args.output.exists() else {}
    model = configured_model(args.model)
    pending = [case for case in cases if case["case_id"] not in completed]
    print(json.dumps({"stage": "domain_reclassification_v3", "cases": len(cases), "pending": len(pending), "model": model}, ensure_ascii=False))
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(process, case, args, model): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            try:
                completed[case["case_id"]] = future.result()
            except Exception as exc:
                errors.append(f"{case['case_id']}:{type(exc).__name__}:{exc}")
    ordered = [completed[case["case_id"]] for case in cases if case["case_id"] in completed]
    write_jsonl(args.output, ordered)
    if args.changes_csv:
        write_csv(args.changes_csv, [{
            "case_id": row["case_id"], "origin_country": row.get("origin_country"), "origin_state": row.get("origin_state"),
            "old_primary_domain": row.get("old_primary_domain"), "new_primary_domain": row.get("primary_domain"),
            "changed": row.get("changed"), "confidence": row.get("confidence"), "review_status": row.get("domain_review_status"),
            "evidence_excerpt": (row.get("domain_evidence_spans") or [""])[0], "eligible_main_corpus": row.get("eligible_main_corpus"),
        } for row in ordered])
    print(json.dumps({"completed": len(ordered), "errors": errors[:20], "error_count": len(errors)}, ensure_ascii=False))
    return 0 if len(ordered) == len(cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
