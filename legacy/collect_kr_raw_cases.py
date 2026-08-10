from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset

from pipeline.stage1_raw import compact, require_outputs, short_hash, unique, write_summary
from pipeline.text_utils import normalize_whitespace
from qc_rules import (
    CASE_NAME_COLUMNS,
    CASE_TYPE_COLUMNS,
    TEXT_COLUMNS,
    apply_duplicate_and_related_qc,
    assess_factual_sufficiency,
    broad_candidate,
    classify_claim_posture,
    classify_court_level,
    classify_liability_basis,
    classify_subtype,
    extract_case_numbers,
    extract_court,
    extract_dates,
    first_existing,
    harm_flags,
    record_hash_fields,
)


LOGGER = logging.getLogger(__name__)
COLLECTION_VERSION = "stage1-kr-direct-tort-appellate-v4"
SOURCE_DATASET = "lbox/lbox_open::precedent_corpus"
DEFAULT_OUTPUT_DIR = Path("outputs/raw/kr_v4")
DEFAULT_MANIFEST_OUTPUT = Path("outputs/manifests/kr_v4_case_manifest.csv")
DEFAULT_SAMPLING_CONFIG = Path("configs/tort_n50_sampling.yaml")

QC_FIELDS = [
    "case_id", "court_level", "court_level_confidence", "claim_posture", "liability_basis",
    "case_subtype", "facts_independently_reconstructable", "fact_source_quality",
    "factual_sufficiency_score", "direct_tort_evidence", "exclusion_reasons",
    "human_qc_status", "human_qc_notes",
]

MANIFEST_FIELDS = [
    "case_id", "collection_version", "source_dataset", "source_record_id", "current_case_number",
    "current_case_number_verified", "court_name", "court_level", "court_level_confidence",
    "decision_date", "decision_year", "decision_date_verified", "incident_date", "incident_year",
    "claim_posture", "liability_basis", "case_subtype", "factual_sufficiency_score",
    "strict_eligible", "shortlisted", "pre_qc_selected", "selected", "related_case_group_id",
    "duplicate_or_related_reason", "raw_text_sha256", "raw_length_chars", "raw_path",
]


def stable_case_id(source_dataset: str, source_record_id: str, raw_text: str) -> str:
    stable_source = compact(source_record_id)
    digest = short_hash(source_dataset, stable_source, raw_text[:1000], length=16)
    return f"KR_{digest}"


def _default_arg(args: argparse.Namespace, name: str, default: object) -> object:
    return getattr(args, name, default)


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, object] | None:
    """Evaluate one broad candidate; target_count is intentionally not consulted."""
    text_col = str(_default_arg(args, "text_col", ""))
    raw_text = first_existing(row, [text_col] if text_col else TEXT_COLUMNS)
    if not raw_text:
        return None
    raw_text = normalize_whitespace(raw_text)
    source_dataset = f"{_default_arg(args, 'dataset', 'lbox/lbox_open')}::{_default_arg(args, 'config', 'precedent_corpus')}"
    source_record_id = compact(row.get("id", ""))
    case_name = first_existing(row, CASE_NAME_COLUMNS)
    case_type = first_existing(row, CASE_TYPE_COLUMNS) or case_name
    gate_text = f"{case_name}\n{case_type}\n{raw_text[:16000]}"
    broad, broad_evidence = broad_candidate(gate_text)

    case_numbers = extract_case_numbers(row, raw_text)
    dates = extract_dates(row, raw_text)
    court_name, court_verified = extract_court(row, raw_text)
    court = classify_court_level(
        raw_text,
        court_name,
        case_numbers["current_case_number"],
        bool(case_numbers["current_case_number_verified"]),
    )
    posture, posture_confidence, direct_evidence = classify_claim_posture(f"{case_name}\n{case_type}\n{raw_text[:16000]}")
    liability_basis, liability_evidence = classify_liability_basis(posture, gate_text)
    subtype = classify_subtype(gate_text)
    factual = assess_factual_sufficiency(raw_text)
    hashes = record_hash_fields(raw_text)

    exclusion_reasons: list[str] = []
    if not broad:
        exclusion_reasons.append("not_broad_civil_tort_candidate")
        if "피고인" in raw_text or "공소사실" in raw_text or "범죄사실" in raw_text:
            exclusion_reasons.append("criminal_case")
    required_level = str(_default_arg(args, "court_level", "appellate"))
    if court["court_level"] != required_level:
        exclusion_reasons.append("current_document_not_verified_appellate")
        if court["court_level"] == "supreme":
            exclusion_reasons.append("supreme_court_excluded")
    if court["court_level_confidence"] not in {"high", "medium"}:
        exclusion_reasons.append("court_level_confidence_too_low")
    if court["court_level"] == "appellate" and court["court_level_confidence"] == "medium" and int(court["appellate_evidence_count"]) < 2:
        exclusion_reasons.append("medium_appellate_has_fewer_than_two_current_signals")
    if posture != "direct_tort_claim":
        exclusion_reasons.append(f"non_direct_claim:{posture}")
    if liability_basis != "non_contractual_tort":
        exclusion_reasons.append(f"not_non_contractual_tort:{liability_basis}")
    if not factual["factual_background_sufficient"]:
        exclusion_reasons.append("factual_background_insufficient")
    if not factual["facts_independently_reconstructable"]:
        exclusion_reasons.append("facts_not_independently_reconstructable")
    if factual["fact_source_quality"] not in {"self_contained", "partially_incorporated"}:
        exclusion_reasons.append(f"fact_source_quality:{factual['fact_source_quality']}")
    if bool(_default_arg(args, "require_verified_decision_year", False)) and not dates["decision_date_verified"]:
        exclusion_reasons.append("verified_decision_year_required")
    min_chars = int(_default_arg(args, "min_text_chars", 1200))
    max_chars = int(_default_arg(args, "max_text_chars", 0))
    if len(raw_text) < min_chars:
        exclusion_reasons.append("too_short_or_no_full_opinion_text")
    if max_chars and len(raw_text) > max_chars:
        exclusion_reasons.append("too_long")

    strict_eligible = not exclusion_reasons
    record: dict[str, object] = {
        "case_id": stable_case_id(source_dataset, source_record_id, raw_text),
        "collection_version": COLLECTION_VERSION,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "raw_text": raw_text,
        "case_name": case_name,
        "case_type": case_type,
        **case_numbers,
        "case_number": case_numbers["current_case_number"],
        "case_number_or_citation": case_numbers["current_case_number"],
        "court_name": court_name,
        "court_name_verified": court_verified,
        **court,
        **dates,
        "liability_basis": liability_basis,
        "liability_basis_evidence": liability_evidence,
        "claim_posture": posture,
        "direct_tort_evidence": direct_evidence,
        "claim_posture_confidence": posture_confidence,
        "case_subtype": subtype,
        **harm_flags(gate_text),
        **factual,
        "broad_candidate": broad,
        "broad_candidate_evidence": broad_evidence,
        "strict_eligible": strict_eligible,
        "exclusion_reasons": unique(exclusion_reasons),
        "related_case_group_id": None,
        "related_case_ids": [],
        "underlying_incident_fingerprint": None,
        "duplicate_or_related_reason": None,
        **hashes,
        "raw_length_chars": len(raw_text),
        "sampling_rank": None,
        "sampling_reasons": [],
        "human_qc_status": "",
        "human_qc_notes": "",
        "shortlisted": False,
        "pre_qc_selected": False,
        "selected": False,
        # Backward-compatible aliases used by downstream code/tests.
        "court_level_evidence": court["current_case_appellate_evidence"],
        "tort_evidence": liability_evidence,
        "factual_sufficiency_reasons": factual["factual_sufficiency_evidence"],
        "include_signals": broad_evidence,
        "exclude_signals": unique(exclusion_reasons),
        "collection_status": "pass" if strict_eligible else "fail",
    }
    return record


def keyword_gate(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    text_col = str(_default_arg(args, "text_col", ""))
    raw_text = first_existing(row, [text_col] if text_col else TEXT_COLUMNS)
    case_name = first_existing(row, CASE_NAME_COLUMNS)
    case_type = first_existing(row, CASE_TYPE_COLUMNS)
    return broad_candidate(f"{case_name}\n{case_type}\n{raw_text[:16000]}")


def finalize_strict_eligibility(records: list[dict[str, object]]) -> None:
    for row in records:
        if row.get("duplicate_or_related_reason"):
            reasons = list(row.get("exclusion_reasons") or [])
            reasons.append(f"duplicate_or_related:{row['duplicate_or_related_reason']}")
            row["exclusion_reasons"] = unique(reasons)
            row["exclude_signals"] = row["exclusion_reasons"]
            row["strict_eligible"] = False
            row["collection_status"] = "fail"


def mark_duplicate_candidates(records: list[dict[str, object]]) -> dict[str, int]:
    counts = apply_duplicate_and_related_qc(records)
    finalize_strict_eligibility(records)
    if "exact_duplicate" in counts:
        counts["duplicate_exact_hash"] = counts["exact_duplicate"]
    if "normalized_text_duplicate" in counts:
        counts["duplicate_normalized_text_hash"] = counts["normalized_text_duplicate"]
    return counts


def split_pools(records: list[dict[str, object]], args: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    broad = [row for row in records if row.get("broad_candidate") is True]
    appellate = [
        row for row in broad
        if row.get("court_level") == getattr(args, "court_level", "appellate")
        and row.get("court_level_confidence") in {"high", "medium"}
        and not (row.get("court_level_confidence") == "medium" and int(row.get("appellate_evidence_count") or 0) < 2)
    ]
    direct = [row for row in appellate if row.get("claim_posture") == "direct_tort_claim"]
    factual_direct = [
        row for row in direct
        if row.get("factual_background_sufficient") is True
        and row.get("facts_independently_reconstructable") is True
        and row.get("fact_source_quality") in {"self_contained", "partially_incorporated"}
    ]
    strict = [row for row in records if row.get("strict_eligible") is True]
    excluded_non_direct = [row for row in appellate if row.get("claim_posture") != "direct_tort_claim"]
    return {
        "broad_candidates": broad,
        "appellate_candidates": appellate,
        "direct_tort_candidates": direct,
        "factually_sufficient_direct_tort": factual_direct,
        "strict_eligible": strict,
        "excluded_non_direct_claims": excluded_non_direct,
        # Compatibility aliases.
        "keyword_hits": broad,
        "civil_candidates": broad,
        "tort_candidates": direct,
    }


def load_sampling_config(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)  # JSON is valid YAML; avoids a runtime YAML dependency.
    quotas = config.get("quotas")
    if not isinstance(quotas, dict) or sum(int(value) for value in quotas.values()) != int(config.get("target_count", 0)):
        raise ValueError("sampling config quotas must sum to target_count")
    return config


def quota_group(subtype: object, config: dict[str, object]) -> str:
    value = str(subtype or "unclear")
    groups = config.get("quota_groups") or {}
    for group, members in groups.items():
        if value in members:
            return str(group)
    return value


def deterministic_rank(row: dict[str, object], seed: int) -> tuple[int, int, int, str]:
    confidence = {"high": 0, "medium": 1, "low": 2}.get(str(row.get("court_level_confidence")), 3)
    score = -int(row.get("factual_sufficiency_score") or 0)
    jitter = int(hashlib.sha256(f"{seed}:{row.get('case_id')}".encode()).hexdigest()[:12], 16)
    return confidence, score, jitter, str(row.get("case_id"))


def _eligible_for_sampling(records: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    rows = [row for row in records if row.get("strict_eligible") is True and not row.get("duplicate_or_related_reason")]
    if bool(_default_arg(args, "require_verified_decision_year", False)):
        rows = [row for row in rows if row.get("decision_date_verified") is True and isinstance(row.get("decision_year"), int)]
    if bool(_default_arg(args, "require_human_accept", False)):
        rows = [row for row in rows if str(row.get("human_qc_status", "")).strip().lower() == "accept"]
    return rows


def select_final_sample(
    records: list[dict[str, object]], args: argparse.Namespace, config: dict[str, object] | None = None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = config or load_sampling_config(str(_default_arg(args, "sampling_config", DEFAULT_SAMPLING_CONFIG)))
    pool = _eligible_for_sampling(records, args)
    target = int(_default_arg(args, "target_count", 50))
    quotas = {str(key): int(value) for key, value in dict(config["quotas"]).items()}
    max_share = float(config.get("max_single_subtype_share", 0.25))
    max_per_subtype = max(1, int(target * max_share))
    by_group: dict[str, list[dict[str, object]]] = {group: [] for group in quotas}
    for row in pool:
        by_group.setdefault(quota_group(row.get("case_subtype"), config), []).append(row)
    for rows in by_group.values():
        rows.sort(key=lambda item: deterministic_rank(item, int(_default_arg(args, "seed", 42))))

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    actual_subtypes: Counter[str] = Counter()
    shortages: dict[str, dict[str, int]] = {}
    for group, quota in quotas.items():
        if len(selected) >= target:
            take = 0
        else:
            candidates = by_group.get(group, [])
            wanted = min(quota, target - len(selected))
            chosen = []
            for row in candidates:
                subtype = str(row.get("case_subtype") or "unclear")
                subtype_cap = 10 if subtype == "traffic_accident" else max_per_subtype
                if actual_subtypes[subtype] >= subtype_cap:
                    continue
                chosen.append(row)
                actual_subtypes[subtype] += 1
                if len(chosen) >= wanted:
                    break
            selected.extend(chosen)
            selected_ids.update(str(row["case_id"]) for row in chosen)
            take = len(chosen)
        shortages[group] = {"quota": quota, "available": len(by_group.get(group, [])), "selected": take, "shortage": max(0, quota - take)}

    if bool(_default_arg(args, "relax_subtype_quota", False)) and len(selected) < target:
        leftovers = [row for row in pool if str(row.get("case_id")) not in selected_ids]
        leftovers.sort(key=lambda item: deterministic_rank(item, int(_default_arg(args, "seed", 42))))
        for row in leftovers:
            subtype = str(row.get("case_subtype") or "unclear")
            cap = 10 if subtype == "traffic_accident" else max_per_subtype
            if actual_subtypes[subtype] >= cap:
                continue
            selected.append(row)
            selected_ids.add(str(row["case_id"]))
            actual_subtypes[subtype] += 1
            if len(selected) >= target:
                break

    selected.sort(key=lambda row: str(row.get("case_id")))
    for rank, row in enumerate(selected, start=1):
        row["sampling_rank"] = rank
        row["sampling_reasons"] = [f"quota_group:{quota_group(row.get('case_subtype'), config)}", f"seed:{_default_arg(args, 'seed', 42)}"]
    meta = {
        "sampling_method": "strict_pool_configured_subtype_quota_no_year",
        "seed": int(_default_arg(args, "seed", 42)),
        "subtype_quotas": quotas,
        "quota_shortage_report": shortages,
        "shortage": max(0, target - len(selected)),
        "relax_subtype_quota": bool(_default_arg(args, "relax_subtype_quota", False)),
        "do_not_use_year_for_sampling": bool(_default_arg(args, "do_not_use_year_for_sampling", True)),
        "require_human_accept": bool(_default_arg(args, "require_human_accept", False)),
    }
    return selected, meta


def build_shortlist(records: list[dict[str, object]], args: argparse.Namespace, config: dict[str, object]) -> list[dict[str, object]]:
    pool = _eligible_for_sampling(records, argparse.Namespace(**{**vars(args), "require_human_accept": False}))
    target = min(int(config.get("shortlist_count", 100)), len(pool))
    doubled = {key: int(value) * 2 for key, value in dict(config["quotas"]).items()}
    shortlist_config = {**config, "quotas": doubled, "target_count": sum(doubled.values())}
    shortlist_args = argparse.Namespace(**{**vars(args), "target_count": target, "relax_subtype_quota": True, "require_human_accept": False})
    selected, _ = select_final_sample(pool, shortlist_args, shortlist_config)
    for rank, row in enumerate(selected, start=1):
        row["shortlisted"] = True
        row["shortlist_rank"] = rank
    return selected


def apply_manual_qc(records: list[dict[str, object]], path: str) -> int:
    if not path:
        return 0
    statuses: dict[str, tuple[str, str]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            statuses[str(row.get("case_id", ""))] = (str(row.get("human_qc_status", "")), str(row.get("human_qc_notes", "")))
    applied = 0
    for row in records:
        if str(row.get("case_id")) in statuses:
            row["human_qc_status"], row["human_qc_notes"] = statuses[str(row["case_id"])]
            applied += 1
    return applied


def count_by(rows: Iterable[dict[str, object]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) if row.get(field) not in {None, ""} else "unknown") for row in rows).items()))


def summarize(
    *, scanned: int, pools: dict[str, list[dict[str, object]]], shortlist: list[dict[str, object]],
    preselected: list[dict[str, object]], selected: list[dict[str, object]], duplicate_counts: dict[str, int],
    args: argparse.Namespace, sampling_meta: dict[str, object], manual_qc_applied: int,
) -> dict[str, object]:
    broad = pools["broad_candidates"]
    direct = pools["direct_tort_candidates"]
    strict = pools["strict_eligible"]
    posture_counts = Counter(str(row.get("claim_posture") or "unclear") for row in pools["appellate_candidates"])
    verified_years = sum(1 for row in broad if row.get("decision_date_verified") and isinstance(row.get("decision_year"), int))
    return {
        "collection_version": COLLECTION_VERSION,
        "total_scanned": scanned,
        "broad_candidate_count": len(broad),
        "appellate_candidate_count": len(pools["appellate_candidates"]),
        "direct_tort_candidate_count": len(direct),
        "factually_sufficient_direct_tort_count": len(pools["factually_sufficient_direct_tort"]),
        "strict_eligible_pool_count": len(strict),
        "preselected_count": len(preselected),
        "shortlist_count": len(shortlist),
        "selected_count": len(selected),
        "pre_qc_selected_count": len(preselected),
        "target_count": int(args.target_count),
        "verified_decision_year_count": verified_years,
        "decision_year_unknown_count": len(broad) - verified_years,
        "incident_year_available_count": sum(1 for row in broad if isinstance(row.get("incident_year"), int)),
        "claim_posture_counts": dict(sorted(posture_counts.items())),
        "insurer_subrogation_excluded_count": posture_counts.get("insurer_subrogation", 0),
        "joint_tortfeasor_contribution_excluded_count": posture_counts.get("joint_tortfeasor_contribution", 0),
        "contract_payment_wage_enforcement_excluded_count": sum(posture_counts.get(label, 0) for label in ("contract_or_payment", "wage_or_compensation", "judgment_enforcement")),
        "strict_eligible_by_subtype": count_by(strict, "case_subtype"),
        "shortlist_by_subtype": count_by(shortlist, "case_subtype"),
        "final_selected_subtype_distribution": count_by(selected, "case_subtype"),
        "appellate_confidence_counts": count_by(pools["appellate_candidates"], "court_level_confidence"),
        "duplicate_related_removed_counts": duplicate_counts,
        "manual_qc_rows_applied": manual_qc_applied,
        "scan_limit": int(args.scan_limit),
        "strict_direct_tort_only": bool(args.strict_direct_tort_only),
        "require_verified_decision_year": bool(args.require_verified_decision_year),
        **sampling_meta,
    }


def iter_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.local_arrow_dir:
        paths = sorted(Path(args.local_arrow_dir).glob("*.arrow"))
        if not paths:
            raise FileNotFoundError(f"No .arrow files found in {args.local_arrow_dir}")
        for path in paths:
            yield from Dataset.from_file(str(path))
        return
    yield from load_dataset(args.dataset, args.config, split=args.split, streaming=True)


def collect(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    scanned = 0
    for scanned, row in enumerate(iter_rows(args), start=1):
        if args.scan_limit and scanned > args.scan_limit:
            scanned -= 1
            break
        keep, _ = keyword_gate(row, args)
        if keep:
            record = evaluate_row(row, args)
            if record:
                records.append(record)
        if args.preview_only and len(records) >= args.preview_count:
            break
        if args.progress_every and scanned % args.progress_every == 0:
            LOGGER.info("scanned=%s broad_candidates=%s", scanned, len(records))

    duplicate_counts = mark_duplicate_candidates(records)
    pools = split_pools(records, args)
    manual_qc_applied = apply_manual_qc(records, args.manual_qc_file)
    config = load_sampling_config(args.sampling_config)
    shortlist = build_shortlist(pools["strict_eligible"], args, config)
    pre_args = argparse.Namespace(**{**vars(args), "require_human_accept": False})
    preselected, pre_meta = select_final_sample(pools["strict_eligible"], pre_args, config)
    selected, sampling_meta = select_final_sample(pools["strict_eligible"], args, config)
    pre_ids = {str(row["case_id"]) for row in preselected}
    selected_ids = {str(row["case_id"]) for row in selected}
    for row in records:
        row["pre_qc_selected"] = str(row["case_id"]) in pre_ids
        row["selected"] = str(row["case_id"]) in selected_ids
    sampling_meta["pre_qc_sampling_shortage"] = pre_meta["shortage"]
    summary = summarize(
        scanned=scanned, pools=pools, shortlist=shortlist, preselected=preselected, selected=selected,
        duplicate_counts=duplicate_counts, args=args, sampling_meta=sampling_meta,
        manual_qc_applied=manual_qc_applied,
    )
    return pools, shortlist, preselected, selected, summary


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = Path(args.output_dir)
    return {
        "broad_candidates": output / "kr_broad_candidates_all.jsonl",
        "appellate_candidates": output / "kr_appellate_candidates_all.jsonl",
        "direct_tort_candidates": output / "kr_direct_tort_candidates_all.jsonl",
        "strict_eligible": output / "kr_strict_eligible_all.jsonl",
        "excluded_non_direct_claims": output / "kr_excluded_non_direct_claims.jsonl",
        "shortlist": output / "kr_direct_tort_shortlist_100.jsonl",
        "shortlist_qc": output / "kr_direct_tort_shortlist_100_qc.csv",
        "pre_qc": output / "kr_cases_selected_50_pre_qc.jsonl",
        "final": output / "kr_cases_selected_50_final.jsonl",
        "summary": output / "kr_cases_summary.json",
        "manifest": Path(args.manifest_output),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_qc_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in QC_FIELDS})


def write_manifest(path: Path, rows: list[dict[str, object]], paths: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in MANIFEST_FIELDS}
            out["raw_path"] = str(paths["strict_eligible"] if row.get("strict_eligible") else paths["broad_candidates"])
            writer.writerow({field: csv_value(out.get(field, "")) for field in MANIFEST_FIELDS})


def print_summary_lines(summary: dict[str, object]) -> None:
    for key in (
        "total_scanned", "broad_candidate_count", "appellate_candidate_count", "direct_tort_candidate_count",
        "factually_sufficient_direct_tort_count", "strict_eligible_pool_count", "preselected_count",
        "selected_count", "target_count",
    ):
        print(f"{key}={summary.get(key, 0)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Korean direct-tort appellate collector v4 (deterministic QC only).")
    parser.add_argument("--dataset", default="lbox/lbox_open")
    parser.add_argument("--config", default="precedent_corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-col", default="")
    parser.add_argument("--export-all-candidates", action="store_true")
    parser.add_argument("--build-shortlist", action="store_true")
    parser.add_argument("--select-final-sample", action="store_true")
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--court-level", choices=["appellate", "trial", "supreme", "unknown"], default="appellate")
    parser.add_argument("--strict-direct-tort-only", action="store_true", default=True)
    parser.add_argument("--strict-tort-only", action="store_true", dest="strict_direct_tort_only", help=argparse.SUPPRESS)
    parser.add_argument("--require-verified-decision-year", action="store_true")
    parser.add_argument("--do-not-use-year-for-sampling", action="store_true", default=True)
    parser.add_argument("--manual-qc-file", default="")
    parser.add_argument("--require-human-accept", action="store_true")
    parser.add_argument("--sampling-config", default=str(DEFAULT_SAMPLING_CONFIG))
    parser.add_argument("--relax-subtype-quota", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scan-limit", type=int, default=0, help="0 scans the complete dataset")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--min-text-chars", type=int, default=1200)
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST_OUTPUT))
    parser.add_argument("--local-arrow-dir", default="")
    args = parser.parse_args(argv)
    if not any((args.export_all_candidates, args.build_shortlist, args.select_final_sample)):
        args.export_all_candidates = args.build_shortlist = args.select_final_sample = True
    if args.require_human_accept and not args.manual_qc_file:
        parser.error("--require-human-accept requires --manual-qc-file")
    if args.target_count != int(load_sampling_config(args.sampling_config)["target_count"]):
        LOGGER.warning("target_count differs from config target_count; quotas remain explicit and are not auto-scaled")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    pools, shortlist, preselected, selected, summary = collect(args)
    print_summary_lines(summary)
    if args.preview_only or args.dry_run:
        return

    paths = output_paths(args)
    targets = [paths["summary"], paths["manifest"]]
    if args.export_all_candidates:
        targets.extend(paths[key] for key in ("broad_candidates", "appellate_candidates", "direct_tort_candidates", "strict_eligible", "excluded_non_direct_claims"))
    if args.build_shortlist:
        targets.extend([paths["shortlist"], paths["shortlist_qc"]])
    if args.select_final_sample:
        targets.extend([paths["pre_qc"], paths["final"]])
    require_outputs(targets, args.overwrite)

    if args.export_all_candidates:
        for key in ("broad_candidates", "appellate_candidates", "direct_tort_candidates", "strict_eligible", "excluded_non_direct_claims"):
            write_jsonl(paths[key], pools[key])
    if args.build_shortlist:
        write_jsonl(paths["shortlist"], shortlist)
        write_qc_csv(paths["shortlist_qc"], shortlist)
    if args.select_final_sample:
        write_jsonl(paths["pre_qc"], preselected)
        write_jsonl(paths["final"], selected)
    write_summary(paths["summary"], summary)
    write_manifest(paths["manifest"], pools["broad_candidates"], paths)
    print(f"summary={paths['summary']}")
    print(f"manifest={paths['manifest']}")


if __name__ == "__main__":
    main()
