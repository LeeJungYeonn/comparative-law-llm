from __future__ import annotations

import argparse
import difflib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import normalized_whitespace, read_jsonl, write_json, write_jsonl
from pipeline_v2.llm_runtime import DEFAULT_API_KEY_ENV, DEFAULT_LETSUR_BASE_URL, call_structured, configured_model
from pipeline_v2.rules import source_span_grounding


PROMPT_VERSION = "retained-final-adjudication-v3.1"
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "uphold_initial_flag": {"type": "boolean"},
        "adjudication_reason": {"type": "string"},
        "source_evidence_spans": {"type": "array", "items": {"type": "string"}},
        "neutral_fact_ko": {"type": "string"},
        "neutral_fact_en": {"type": "string"},
    },
    "required": ["uphold_initial_flag", "adjudication_reason", "source_evidence_spans", "neutral_fact_ko", "neutral_fact_en"],
}
PROMPT = """Directly adjudicate a strict semantic-QC flag on one externally source-reviewed retained bilingual neutral-fact record. The source-language master was independently reviewed against the original opinion and is authoritative, but the translation or final wording may still contain a new hard defect.

Use the supplied original-opinion retrieval excerpts as corroborating source and the reviewed source-language master as the complete grounded factual representation. Do not demand inclusion of party roles, procedure, or every contextual detail merely because it appears in the opinion. A concise neutral fact may omit nonessential details. A repeated causal formulation is a hard duplicate only when it adds no distinct causal or epistemic information.

Uphold only a concrete defect: unsupported fact or entity relation, changed allegation/testimony status, material bilingual mismatch, unnecessary litigation posture or legal conclusion, jurisdiction cue, or genuinely redundant factual content. If the flag is false, return both original texts byte-for-byte. If upheld, minimally edit both languages so they are equivalent, source-grounded, independently analyzable, and contain no procedural/legal conclusion or redundant content. Preserve placeholders, numbers, units, negation, chronology, and epistemic status. Do not add facts.

Quote one or more short exact source spans from the supplied original-opinion excerpts when available. Never claim evidence not present. Return strict JSON."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("outputs_v2/final_fact_patterns_200_v3_candidate.jsonl"))
    p.add_argument("--qc", type=Path, default=Path("outputs_v2/final_qc_audit_200_v3_rebased.jsonl"))
    p.add_argument("--cases", type=Path, default=Path("outputs_v2/provisional_final_cases_200_v3.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs_v2/final_qc_direct_adjudication_v3.jsonl"))
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--base-url", default=DEFAULT_LETSUR_BASE_URL)
    p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    p.add_argument("--dotenv-path", type=Path)
    p.add_argument("--raw-root", type=Path, default=Path("outputs_v2/raw_api_responses_v3"))
    p.add_argument("--status-path", type=Path, default=Path("outputs_v2/api_request_status_v3.jsonl"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    return p


def retrieval_excerpts(opinion: str, master: str, limit: int = 6) -> list[str]:
    cleaned = re.sub(r"\[[A-Z]+_[A-Z]+\]", " ", master)
    tokens = {token.casefold() for token in re.findall(r"[A-Za-z]{4,}|[가-힣]{2,}", cleaned)}
    windows = [opinion[start:start + 2200] for start in range(0, len(opinion), 1800)]
    scored = []
    for index, window in enumerate(windows):
        normalized = window.casefold()
        score = sum(1 for token in tokens if token in normalized)
        scored.append((score, index, window))
    selected = [window for score, _, window in sorted(scored, reverse=True)[:limit] if score > 0]
    return selected or ([opinion[:5000]] if opinion else [])


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    records = {row["case_id"]: row for row in read_jsonl(args.input)}
    cases = {row["case_id"]: row for row in read_jsonl(args.cases)}
    flagged = [row for row in read_jsonl(args.qc) if row.get("hard_fail") or row.get("manual_review_required")]
    prior = {row["case_id"]: row for row in read_jsonl(args.output)} if args.resume and args.output.exists() else {}
    model = configured_model(args.model)

    def adjudicate(flag: dict[str, Any]) -> dict[str, Any]:
        case_id = flag["case_id"]
        record, case = records[case_id], cases[case_id]
        master = record["neutral_fact_ko"] if record.get("source_language") == "ko" else record["neutral_fact_en"]
        opinion = case.get("main_opinion_text") or case.get("full_opinion_text") or ""
        excerpts = retrieval_excerpts(opinion, master)
        payload = {
            "case_id": case_id, "source_language": record.get("source_language"),
            "reviewed_source_language_master": master,
            "neutral_fact_ko": record["neutral_fact_ko"], "neutral_fact_en": record["neutral_fact_en"],
            "initial_qc_issues": flag.get("issues") or [], "initial_qc_evidence": flag.get("evidence") or [],
            "original_opinion_retrieval_excerpts": excerpts,
        }
        parsed, provenance = call_structured(
            case_id=case_id, stage=PROMPT_VERSION, prompt_version=PROMPT_VERSION, model=model,
            system_prompt=PROMPT, user_payload=payload, schema_name="retained_final_adjudication_v3", schema=SCHEMA,
            raw_root=args.raw_root, status_path=args.status_path, max_retries=5, resume=args.resume,
            base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
        )
        evidence_checks = [source_span_grounding(opinion, span)[0] == "pass" for span in parsed["source_evidence_spans"]]
        return {
            "case_id": case_id, **parsed,
            "source_evidence_verified": bool(evidence_checks) and all(evidence_checks),
            "initial_issues": flag.get("issues") or [], "adjudication_provenance": provenance,
        }

    pending = [row for row in flagged if row["case_id"] not in prior]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(adjudicate, row): row["case_id"] for row in pending}
        for future in as_completed(futures):
            result = future.result()
            prior[result["case_id"]] = result
    decisions = [prior[row["case_id"]] for row in flagged]
    for decision in decisions:
        opinion = cases[decision["case_id"]].get("main_opinion_text") or cases[decision["case_id"]].get("full_opinion_text") or ""
        verified = [span for span in decision.get("source_evidence_spans") or [] if source_span_grounding(opinion, span)[0] == "pass"]
        if not verified:
            for span in decision.get("source_evidence_spans") or []:
                match = difflib.SequenceMatcher(None, span, opinion, autojunk=False).find_longest_match()
                if match.size >= 50:
                    verified.append(opinion[match.b:match.b + match.size])
        if verified:
            decision["source_evidence_spans"] = verified
            decision["source_evidence_verified"] = True
    write_jsonl(args.output, decisions)

    amendments = list(read_jsonl(Path("outputs_v2/retained_fact_amendments_v3.jsonl")))
    unresolved = []
    changed_cases = 0
    for decision in decisions:
        case_id = decision["case_id"]
        record = records[case_id]
        changed = False
        if decision["uphold_initial_flag"]:
            if not decision["source_evidence_verified"]:
                unresolved.append(case_id)
                continue
            for field in ("neutral_fact_ko", "neutral_fact_en"):
                old, new = record[field], normalized_whitespace(decision[field])
                if old == new:
                    continue
                record[field] = new
                changed = True
                amendments.append({
                    "case_id": case_id, "field_changed": field, "old_text": old, "new_text": new,
                    "source_evidence": decision["source_evidence_spans"],
                    "reason": decision["adjudication_reason"],
                    "review_stage": "final_all_200_direct_source_adjudication_v3",
                })
        if changed:
            changed_cases += 1
            master = record["neutral_fact_ko"] if record["source_language"] == "ko" else record["neutral_fact_en"]
            record["source_fact_units"][0].update({"text": master, "source_span": master})
            record["aligned_fact_units"][0].update({"source_text": master, "neutral_ko": record["neutral_fact_ko"], "neutral_en": record["neutral_fact_en"]})
            record["retained_amendment_status"] = "amended_after_final_qc"
            record["text_review_provenance"] = "external_manual_review+v3_final_qc_direct_source_adjudication"
    write_jsonl(args.input, [records[row["case_id"]] for row in read_jsonl(args.input)])
    write_jsonl(Path("outputs_v2/retained_fact_amendments_v3.jsonl"), amendments)
    unit_rows = []
    for record in records.values():
        aligned = {unit.get("fact_id"): unit for unit in record.get("aligned_fact_units") or []}
        for unit in record.get("source_fact_units") or record.get("fact_units") or []:
            if unit.get("include_in_neutral_fact"):
                row = {"case_id": record["case_id"], "origin_country": record.get("origin_country"), **unit}
                row.update({key: aligned.get(unit.get("fact_id"), {}).get(key) for key in ("neutral_ko", "neutral_en")})
                unit_rows.append(row)
    write_jsonl(Path("outputs_v2/final_fact_units_200_v3_candidate.jsonl"), unit_rows)
    write_json(Path("outputs_v2/final_qc_direct_adjudication_summary_v3.json"), {
        "flagged": len(flagged), "upheld": sum(row["uphold_initial_flag"] for row in decisions),
        "dismissed": sum(not row["uphold_initial_flag"] for row in decisions), "changed_cases": changed_cases,
        "unresolved_case_ids": sorted(unresolved),
    })
    print({"flagged": len(flagged), "upheld": sum(row["uphold_initial_flag"] for row in decisions), "changed_cases": changed_cases, "unresolved": len(unresolved)})
    return 2 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
