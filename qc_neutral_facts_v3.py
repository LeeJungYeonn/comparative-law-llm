from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, write_csv, write_jsonl
from pipeline_v2.llm_runtime import DEFAULT_API_KEY_ENV, DEFAULT_LETSUR_BASE_URL, call_structured, configured_model
from pipeline_v2.v3_rules import bilingual_deterministic_qc

PROMPT_VERSION = "neutral-fact-qc-v3.1"
QC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "hard_fail": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "manual_review_required": {"type": "boolean"},
    },
    "required": ["hard_fail", "issues", "evidence", "manual_review_required"],
}

PROMPT = """Perform strict final QC of one bilingual neutral-fact record using the supplied source-grounded units. Judge concrete defects actually present in neutral_fact_ko or neutral_fact_en; do not fail merely because a source-support excerpt contains legal language or because hidden pipeline metadata once contained a warning. A payment, refusal, communication, injury, or other real-world event is factual, not a procedural outcome merely because litigation later discussed it.

Independently ask: (1) is any final neutral statement unsupported by the corresponding source unit, (2) is any court/jury/legal conclusion retained in the final text, (3) is any litigation posture or disposition retained, (4) is jurisdiction revealed unnecessarily, (5) is any substantive fact missing in one language, (6) did allegation/testimony/dispute status change, (7) do placeholders denote the same entities, (8) are numbers, units, negation, and temporal order semantically equivalent, (9) is factual content duplicated, and (10) is the record independently analyzable. Unit conversion and number words are acceptable when quantitatively equivalent. Set hard_fail only for a concrete final-text defect and quote it in evidence. Set manual_review_required only for a specific unresolved ambiguity, not general caution. Return strict JSON; never silently correct."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict deterministic and semantic v3 neutral-fact QC.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--semantic-output", type=Path)
    p.add_argument("--llm-model")
    p.add_argument("--base-url", default=DEFAULT_LETSUR_BASE_URL)
    p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    p.add_argument("--dotenv-path", type=Path)
    p.add_argument("--raw-root", type=Path, default=Path("outputs_v2/raw_api_responses_v3"))
    p.add_argument("--status-path", type=Path, default=Path("outputs_v2/api_request_status_v3.jsonl"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def deterministic(record: dict[str, Any]) -> dict[str, Any]:
    checks = bilingual_deterministic_qc(record)
    units = record.get("source_fact_units") or record.get("fact_units") or []
    aligned = record.get("aligned_fact_units") or []
    included = [unit for unit in units if unit.get("include_in_neutral_fact")]
    expected_ids = [unit.get("fact_id") for unit in included]
    actual_ids = [unit.get("fact_id") for unit in aligned]
    external_review = record.get("external_manual_source_review_status") == "pass"
    grounding = "pass" if external_review or (included and all(unit.get("source_grounding_status") == "pass" for unit in included)) else "fail"
    coverage = "pass" if expected_ids and expected_ids == actual_ids else "fail"
    mandatory_types = {unit.get("fact_type") for unit in included}
    sufficiency = "pass" if external_review or {"parties", "conduct", "harm", "causation"} <= mandatory_types else "fail"
    translation = "pass" if record.get("translation_equivalence_status") == "pass" and coverage == "pass" else "fail"
    statuses = {
        "source_grounding_status": grounding, "mandatory_factual_sufficiency_status": sufficiency,
        "fact_unit_coverage_status": coverage, "translation_equivalence_status": translation, **checks,
    }
    hard_fields = (
        "source_grounding_status", "mandatory_factual_sufficiency_status", "fact_unit_coverage_status",
        "translation_equivalence_status", "language_sanity_status", "placeholder_equivalence_status",
        "duplicate_sentence_status", "legal_leakage_status", "procedural_leakage_status",
        "jurisdiction_leakage_status",
    )
    issues = [field for field in hard_fields if statuses.get(field) != "pass"]
    review_issues = ([] if statuses.get("numerical_unit_status") == "pass" else ["numerical_unit_status"])
    return {"case_id": record["case_id"], **statuses, "deterministic_issues": issues, "deterministic_review_issues": review_issues, "deterministic_pass": not issues}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.csv.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Refusing to overwrite {args.csv}")
    records = list(read_jsonl(args.input))
    rows = {record["case_id"]: deterministic(record) for record in records}
    semantic: dict[str, dict[str, Any]] = {}
    if args.semantic_output and args.resume and args.semantic_output.exists():
        semantic = {row["case_id"]: row for row in read_jsonl(args.semantic_output)}
    if args.llm_model:
        model = configured_model(args.llm_model)
        def review(record: dict[str, Any]) -> dict[str, Any]:
            payload = {
                "case_id": record["case_id"],
                "source_grounded_fact_units": [{
                    key: unit.get(key) for key in ("fact_id", "text", "epistemic_status", "fact_type", "source_span", "source_grounding_status")
                } for unit in (record.get("source_fact_units") or record.get("fact_units") or []) if unit.get("include_in_neutral_fact")],
                "aligned_fact_units": [{
                    key: unit.get(key) for key in ("fact_id", "source_text", "neutral_ko", "neutral_en")
                } for unit in (record.get("aligned_fact_units") or [])],
                "neutral_fact_ko": record.get("neutral_fact_ko"), "neutral_fact_en": record.get("neutral_fact_en"),
            }
            parsed, provenance = call_structured(
                case_id=record["case_id"], stage="final-qc-v3.1", prompt_version=PROMPT_VERSION, model=model,
                system_prompt=PROMPT, user_payload=payload, schema_name="final_neutral_fact_qc_v3", schema=QC_SCHEMA,
                raw_root=args.raw_root, status_path=args.status_path, max_retries=args.max_retries, resume=args.resume,
                base_url=args.base_url, api_key_env=args.api_key_env, dotenv_path=args.dotenv_path,
            )
            return {"case_id": record["case_id"], **parsed, "qc_provenance": provenance}
        pending = [record for record in records if record["case_id"] not in semantic]
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {executor.submit(review, record): record for record in pending}
            for future in as_completed(futures):
                record = futures[future]
                semantic[record["case_id"]] = future.result()
    output_rows = []
    for record in records:
        row = rows[record["case_id"]]
        sem = semantic.get(record["case_id"])
        row["semantic_qc_status"] = "not_run" if sem is None else ("pass" if not sem.get("hard_fail") and not sem.get("manual_review_required") else "review")
        row["semantic_issues"] = [] if sem is None else sem.get("issues") or []
        row["manual_review_required"] = False if sem is None else bool(sem.get("manual_review_required"))
        numeric_resolved = row["numerical_unit_status"] == "pass" or row["semantic_qc_status"] == "pass"
        row["final_pass"] = row["deterministic_pass"] and numeric_resolved and row["semantic_qc_status"] in {"not_run", "pass"}
        output_rows.append(row)
    write_csv(args.csv, output_rows)
    if args.semantic_output:
        write_jsonl(args.semantic_output, [semantic[record["case_id"]] for record in records if record["case_id"] in semantic])
    print(json.dumps({"cases": len(records), "deterministic_pass": sum(row["deterministic_pass"] for row in output_rows), "semantic_pass": sum(row["semantic_qc_status"] == "pass" for row in output_rows), "final_pass": sum(row["final_pass"] for row in output_rows)}, ensure_ascii=False))
    return 0 if all(row["final_pass"] for row in output_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
