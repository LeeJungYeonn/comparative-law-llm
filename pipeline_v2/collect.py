from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .io_utils import normalized_whitespace, sha256_text, stable_id
from .rules import (
    assess_fact_sufficiency, civil_liability_candidate, classify_domain, classify_kr_court,
    duplicate_family_id, extract_kr_decision_date, is_us_state_highcourt, parse_date,
    select_main_opinion, state_from_jurisdiction,
)


def find_lbox_arrows(local_arrow_dir: Path | None = None) -> tuple[list[Path], str | None]:
    if local_arrow_dir:
        arrows = sorted(local_arrow_dir.glob("*.arrow"))
        return arrows, local_arrow_dir.name
    base = Path.home() / ".cache" / "huggingface" / "datasets" / "lbox___lbox_open" / "precedent_corpus"
    arrows = sorted(base.glob("*/*/*.arrow")) if base.exists() else []
    revision = arrows[0].parent.name if arrows else None
    return arrows, revision


def iter_lbox_rows(local_arrow_dir: Path | None = None, revision: str | None = None) -> tuple[Iterator[dict[str, Any]], str | None]:
    arrows, cached_revision = find_lbox_arrows(local_arrow_dir)
    if arrows:
        from datasets import Dataset

        def local_rows() -> Iterator[dict[str, Any]]:
            for arrow in arrows:
                dataset = Dataset.from_file(str(arrow))
                yield from dataset
        return local_rows(), cached_revision
    from datasets import load_dataset
    dataset = load_dataset("lbox/lbox_open", "precedent_corpus", split="train", revision=revision)
    return iter(dataset), revision


def evaluate_kr_row(row: dict[str, Any], *, start_date: str, end_date: str, min_chars: int = 1200) -> dict[str, Any] | None:
    raw = normalized_whitespace(row.get("precedent") or row.get("raw_text") or row.get("text"))
    if not raw:
        return None
    civil, include_evidence, incidental_exclusions = civil_liability_candidate(raw)
    if not include_evidence:
        return None
    source_id = normalized_whitespace(row.get("id"))
    court = classify_kr_court(raw, row.get("court_name") or row.get("court"), row.get("case_number"))
    decision = extract_kr_decision_date(raw, row.get("decision_date"))
    domain = classify_domain(raw)
    sufficiency = assess_fact_sufficiency(raw)
    exclusions: list[str] = []
    if court["court_level"] != "supreme" or court["court_level_confidence"] != "high":
        exclusions.append("not_high_confidence_supreme")
    if not decision["decision_date"]:
        exclusions.append("decision_date_unknown")
    elif not (start_date <= decision["decision_date"] <= end_date):
        exclusions.append("decision_date_out_of_range")
    if not civil:
        exclusions.append("not_civil_liability_candidate")
    exclusions.extend(f"incidental:{item}" for item in incidental_exclusions if item in {"criminal_case", "administrative_only"})
    if len(raw) < min_chars:
        exclusions.append("opinion_text_too_short")
    if not sufficiency["factual_background_sufficient"]:
        exclusions.append("fact_insufficient_before_supplementation")
    raw_hash = sha256_text(raw)
    record: dict[str, Any] = {
        "case_id": stable_id("KR", "lbox/lbox_open", source_id, raw_hash),
        "source_dataset": "lbox/lbox_open", "source_config": "precedent_corpus", "source_record_id": source_id,
        "origin_country": "KR", "origin_state": None, "court_name": "대법원" if court["court_level"] == "supreme" else None,
        **court, **decision, **domain, **sufficiency,
        "civil_liability_evidence": include_evidence,
        "full_opinion_text": raw, "main_opinion_text": raw, "main_opinion_type": "court_opinion",
        "opinion_selection_reason": "LBox precedent_corpus provides one opinion text field",
        "has_concurrence": False, "has_dissent": False,
        "raw_text_sha256": raw_hash, "raw_text_chars": len(raw),
        "strict_source_eligible": not exclusions,
        "exclusion_reasons": list(dict.fromkeys(exclusions)),
        "lower_court_supplemented": False, "lower_court_case_ids": [],
        "lower_court_supplementation_status": "not_attempted",
        "lower_court_link_confidence": "none", "lower_court_link_evidence": [],
    }
    record["case_family_id"] = duplicate_family_id(record)
    record["highest_court_case_id"] = record["case_id"]
    return record


def _citation(row: dict[str, Any]) -> str | None:
    citations = row.get("citations")
    if isinstance(citations, list):
        values = []
        for value in citations:
            values.append(normalized_whitespace(value.get("cite") or value.get("citation")) if isinstance(value, dict) else normalized_whitespace(value))
        return "; ".join(filter(None, values)) or None
    return normalized_whitespace(citations) or None


def evaluate_us_row(row: dict[str, Any], *, start_date: str, end_date: str, min_chars: int = 1200) -> dict[str, Any]:
    highcourt, court_evidence = is_us_state_highcourt(row)
    state = state_from_jurisdiction(row.get("court_jurisdiction"))
    decision_date, date_confidence, date_evidence = parse_date(row.get("date_filed"))
    opinion = select_main_opinion(row, minimum_chars=min_chars)
    main_text = opinion["main_opinion_text"]
    metadata_text = "\n".join(normalized_whitespace(row.get(key)) for key in ("case_name", "case_name_full", "nature_of_suit", "posture", "summary", "syllabus", "headnotes") if row.get(key))
    classification_text = f"{metadata_text}\n{main_text}"
    civil, include_evidence, incidental_exclusions = civil_liability_candidate(classification_text)
    domain = classify_domain(classification_text)
    sufficiency = assess_fact_sufficiency(main_text)
    exclusions: list[str] = []
    if not highcourt:
        exclusions.append("court_type_not_S_or_federal")
    if not state:
        exclusions.append("state_jurisdiction_unresolved")
    if not decision_date:
        exclusions.append("decision_date_unknown")
    elif not (start_date <= decision_date <= end_date):
        exclusions.append("decision_date_out_of_range")
    if not civil:
        exclusions.append("not_civil_liability_candidate")
    if any(value in {"criminal_case", "administrative_only"} for value in incidental_exclusions):
        exclusions.extend(incidental_exclusions)
    if not opinion["main_opinion_usable"]:
        exclusions.append("no_usable_controlling_opinion")
    if not sufficiency["factual_background_sufficient"]:
        exclusions.append("fact_insufficient_before_supplementation")
    source_id = normalized_whitespace(row.get("id"))
    full_text = "\n\n".join(normalized_whitespace(op.get("opinion_text") or op.get("text") or op.get("ocr")) for op in (row.get("opinions") or []) if isinstance(op, dict))
    raw_hash = sha256_text(full_text)
    record: dict[str, Any] = {
        "case_id": stable_id("US", "harvard-lil/cold-cases", source_id),
        "source_dataset": "harvard-lil/cold-cases", "source_config": "default", "source_record_id": source_id,
        "origin_country": "US", "origin_state": state,
        "court_name": normalized_whitespace(row.get("court_full_name") or row.get("court_short_name")),
        "court_short_name": normalized_whitespace(row.get("court_short_name")),
        "court_jurisdiction": normalized_whitespace(row.get("court_jurisdiction")),
        "court_type": normalized_whitespace(row.get("court_type")).upper(), "court_level": "supreme" if highcourt else "other",
        "court_level_confidence": "high" if highcourt else "low", "court_level_evidence": court_evidence,
        "decision_date": decision_date, "decision_date_confidence": date_confidence, "decision_date_evidence": date_evidence,
        "case_name": normalized_whitespace(row.get("case_name") or row.get("case_name_full")),
        "case_number": normalized_whitespace(row.get("docket_number")) or None, "citation": _citation(row),
        "precedential_status": normalized_whitespace(row.get("precedential_status")),
        "nature_of_suit": normalized_whitespace(row.get("nature_of_suit")),
        **opinion, **domain, **sufficiency,
        "civil_liability_evidence": include_evidence,
        "full_opinion_text": full_text, "raw_text_sha256": raw_hash, "raw_text_chars": len(full_text),
        "strict_source_eligible": not exclusions, "exclusion_reasons": list(dict.fromkeys(exclusions)),
        "lower_court_supplemented": False, "lower_court_case_ids": [],
        "lower_court_supplementation_status": "not_attempted",
        "lower_court_link_confidence": "none", "lower_court_link_evidence": [],
        "history": row.get("history"), "cross_reference": row.get("cross_reference"),
    }
    record["case_family_id"] = duplicate_family_id(record)
    record["highest_court_case_id"] = record["case_id"]
    return record


def collection_funnel(records: list[dict[str, Any]], scanned: int) -> dict[str, int]:
    return {
        "source_scanned": scanned,
        "civil_liability_candidate": len(records),
        "date_court_eligible": sum(not any(reason in row["exclusion_reasons"] for reason in ("not_high_confidence_supreme", "court_type_not_S_or_federal", "decision_date_unknown", "decision_date_out_of_range")) for row in records),
        "adequate_opinion_text": sum("opinion_text_too_short" not in row["exclusion_reasons"] and "no_usable_controlling_opinion" not in row["exclusion_reasons"] for row in records),
        "fact_sufficient": sum(row.get("factual_background_sufficient") is True for row in records),
        "strict_source_eligible": sum(row.get("strict_source_eligible") is True for row in records),
    }
