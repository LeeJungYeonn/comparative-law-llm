from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset

from pipeline.io_utils import ensure_parent, require_overwrite, stable_case_id
from pipeline.text_utils import compact_inline, excerpt, normalize_whitespace


LOGGER = logging.getLogger(__name__)

SOURCE_DATASET = "harvard-lil/cold-cases"
SOURCE_SPLIT = "train"
MIN_TEXT_LENGTH = 3_000
PREVIEW_CHARS = 2_000

HIGH_PRIORITY_KEYWORDS = [
    "personal injury",
    "premises liability",
    "automobile accident",
    "car accident",
    "wrongful death",
    "property damage",
    "injury",
    "accident",
    "damages",
    "tort",
    "negligence",
    "slip and fall",
    "defective condition",
    "product liability",
    "misrepresentation",
    "fraud damages",
]

FEDERAL_EXCLUDE_PATTERNS = [
    r"United States District Court",
    r"U\.S\. District Court",
    r"\bN\.D\. Cal\.",
    r"\bC\.D\. Cal\.",
    r"\bE\.D\. Cal\.",
    r"\bS\.D\. Cal\.",
    r"\b9th Cir\.",
    r"Ninth Circuit",
    r"\bF\. ?Supp\.?",
    r"\bF\.2d\b",
    r"\bF\.3d\b",
    r"\bU\.S\.C\.",
]
OUT_OF_STATE_COURT_PATTERNS = [
    r"\bAriz\.",
    r"Arizona Court of Appeals",
    r"New York",
    r"Texas",
    r"Supreme Court of (?!California\b)[A-Z][A-Za-z ]+",
    r"Court of Appeals of (?!California\b)[A-Z][A-Za-z ]+",
    r"Appellate Court of (?!California\b)[A-Z][A-Za-z ]+",
]
CRIMINAL_EXCLUDE_PATTERNS = [
    r"\bPeople v\.",
    r"\bCrim\.",
    r"\bcriminal\b",
    r"\bhabeas\b",
    r"\bwarden\b",
    r"\bprison\b",
    r"\bconviction\b",
    r"\bsentence\b",
    r"defendant was convicted",
    r"\bprosecution\b",
]
ADMIN_EXCLUDE_PATTERNS = [
    r"Public Utilities Commission",
    r"administrative review",
    r"agency decision",
    r"writ of mandate",
    r"mandamus",
    r"Workers'? Compensation Appeals Board",
    r"Industrial Accident Commission",
    r"Occupational Safety & Health Appeals Board",
    r"Department of Transportation",
    r"eminent domain",
    r"\bcondemnation\b",
    r"^In re\b",
    r"Juvenile Court Law",
    r"Department of Children and Family Services",
    r"\bdependency\b",
    r"child custody",
    r"parental rights",
]
IP_EXCLUDE_PATTERNS = [
    r"\bcopyright\b",
    r"\bpatent\b",
    r"\btrademark\b",
    r"Lanham Act",
]
PROCEDURE_ONLY_PATTERNS = [
    r"res judicata",
    r"statute of limitations",
    r"default judgment",
    r"summary judgment",
    r"\bdemurrer\b",
    r"\bremand\b",
    r"\bv\. Superior Court\b",
    r"litigation expenses",
    r"final offers?",
    r"final demands?",
    r"attorney fees?",
    r"attorneys(?:'|&#x2019;)? fees?",
    r"attorney fee award",
    r"\bsection 1717\b",
    r"Civil Code section 1717",
    r"\bcosts only\b",
]
ATTORNEY_FEE_ONLY_PATTERNS = [
    r"attorney fees?",
    r"attorneys(?:'|&#x2019;)? fees?",
    r"attorney fee award",
    r"prevailing party fees?",
    r"\bsection 1717\b",
    r"Civil Code section 1717",
]
CONTRACT_ONLY_PATTERNS = [
    r"breach of contract",
    r"contract interpretation",
    r"specific performance",
    r"escrow instructions",
    r"promissory note",
    r"action on a promissory note",
    r"monthly installments of rent",
    r"lease of a storeroom",
    r"endorse the note",
    r"\bguaranty\b",
    r"\bdebt\b",
    r"\bcollection\b",
]
INSURANCE_ONLY_PATTERNS = [
    r"\binsurance\b",
    r"Inter-Insurance",
    r"Casualty Insurance",
    r"Prudential Insurance",
    r"\binsured\b",
    r"insurance coverage",
    r"duty to defend",
    r"coverage dispute",
    r"insurer",
]
NON_TARGET_CIVIL_EXCLUDE_PATTERNS = [
    r"\bFEHA\b",
    r"wrongful termination",
    r"sexual harassment",
    r"employment discrimination",
    r"retaliation",
    r"retaliatory discharge",
    r"wrongful discharge",
]
FAMILY_LAW_STRONG_PATTERNS = [
    r"\bdivorce\b",
    r"interlocutory decree of divorce",
    r"separate maintenance",
    r"\bmarital\b",
    r"\bspousal\b",
    r"cross-complaint for divorce",
    r"decree of separate maintenance",
]
FAMILY_LAW_SUPPORT_PATTERNS = [
    r"husband and wife",
    r"\bcommunity property\b",
    r"\balimony\b",
]
MANUAL_EXCLUDE_CASES = {
    "US_california_3285859": "family_divorce_case",
    "US_california_3293895": "family_separate_maintenance_case",
    "US_california_5799794": "family_divorce_decree_case",
    "US_california_3286933": "contract_only_promissory_note_case",
    "US_california_3283498": "contract_only_rent_or_lease_case",
    "US_california_3279743": "contract_only_promissory_note_building_contract_case",
    "US_california_2136310": "contract_equipment_sale_judgment_roll_case",
    "US_california_5798321": "prior_litigation_fraud_procedural_case",
}
CATEGORY_OVERRIDES = {
    "US_california_1117178": "property_damage",
    "US_california_5796609": "personal_injury",
    "US_california_5802325": "property_damage",
    "US_california_2136310": "contract_damages",
}
FACT_BACKGROUND_PATTERNS = [
    r"FACTUAL BACKGROUND",
    r"FACTS",
    r"BACKGROUND",
    r"factual and procedural background",
    r"was injured",
    r"were injured",
    r"was killed",
    r"died",
    r"accident",
    r"collision",
    r"fell",
    r"slipped",
    r"struck",
    r"damaged",
    r"property damage",
    r"misrepresented",
    r"fraud",
    r"entered into a written contract",
    r"purchase agreement",
    r"sale agreement",
]

CATEGORY_PATTERNS = {
    "wrongful_death": [r"wrongful death", r"death action", r"was killed", r"were killed", r"fatal injur"],
    "auto_accident": [r"automobile accident", r"car accident", r"vehicle accident", r"collision", r"truck", r"motorist"],
    "personal_injury": [
        r"personal injur",
        r"bodily injur",
        r"was injured",
        r"were injured",
        r"\binjur(?:y|ies|ed)\b",
        r"slip and fall",
        r"fell",
        r"fall",
        r"accident",
        r"struck",
        r"burned",
    ],
    "property_damage": [
        r"property damage",
        r"damaged property",
        r"damage to .*property",
        r"fire damage",
        r"flood",
        r"trespass",
        r"nuisance",
    ],
    "product_liability": [
        r"product liability",
        r"defective product",
        r"defective condition",
        r"defective .*product",
        r"manufacturer",
    ],
    "professional_negligence": [r"professional negligence", r"medical malpractice", r"legal malpractice", r"malpractice"],
    "fraud_damages": [r"fraud", r"misrepresentation", r"deceit", r"fraud damages"],
    "contract_damages": [r"contract damages", r"breach of contract", r"purchase agreement", r"escrow"],
}

TARGET_CATEGORY_QUOTAS = {
    "personal_injury": 15,
    "auto_accident": 10,
    "wrongful_death": 10,
    "property_damage": 10,
    "product_liability": 5,
    "professional_negligence": 5,
    "fraud_damages": 10,
    "contract_damages": 10,
}


def regex_any(patterns: Iterable[str], text: str, *, flags: int = re.IGNORECASE) -> bool:
    return any(re.search(pattern, text, flags=flags) for pattern in patterns)


def regex_hits(patterns: Iterable[str], text: str, *, flags: int = re.IGNORECASE) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=flags)]


def row_text(row: dict[str, Any]) -> str:
    opinions = row.get("opinions") or []
    if isinstance(opinions, list):
        chunks = []
        for opinion in opinions:
            if isinstance(opinion, dict):
                text = opinion.get("opinion_text") or opinion.get("text") or ""
                if text:
                    chunks.append(normalize_whitespace(text))
        if chunks:
            return "\n\n".join(chunks).strip()
    return normalize_whitespace(row.get("opinion_text") or row.get("text") or row.get("raw_text") or "")


def first_download_url(row: dict[str, Any]) -> str:
    opinions = row.get("opinions") or []
    if isinstance(opinions, list):
        for opinion in opinions:
            if isinstance(opinion, dict) and opinion.get("download_url"):
                return compact_inline(opinion.get("download_url"))
    return ""


def citations_text(row: dict[str, Any]) -> str:
    citations = row.get("citations") or []
    if not isinstance(citations, list):
        return compact_inline(citations)
    values = []
    for citation in citations:
        if isinstance(citation, dict):
            values.append(compact_inline(citation.get("cite") or citation.get("citation") or citation))
        else:
            values.append(compact_inline(citation))
    return " ".join(value for value in values if value)


def metadata_text(row: dict[str, Any]) -> str:
    columns = [
        "case_name",
        "case_name_full",
        "case_name_short",
        "court_full_name",
        "court_jurisdiction",
        "court_short_name",
        "court_type",
        "nature_of_suit",
        "headmatter",
        "headnotes",
        "summary",
        "syllabus",
        "posture",
        "disposition",
    ]
    return "\n".join(compact_inline(row.get(column, "")) for column in columns) + "\n" + citations_text(row)


def is_california_state_court(row: dict[str, Any]) -> bool:
    court = compact_inline(row.get("court_full_name") or row.get("court_short_name") or "")
    jurisdiction = compact_inline(row.get("court_jurisdiction", ""))
    court_type = compact_inline(row.get("court_type", "")).upper()
    if regex_any(FEDERAL_EXCLUDE_PATTERNS, f"{court}\n{jurisdiction}\n{citations_text(row)}"):
        return False
    california_court = bool(
        re.search(r"California (?:Court of Appeal|Supreme Court)", court, flags=re.IGNORECASE)
        or re.search(r"Court of Appeals? of California", court, flags=re.IGNORECASE)
        or re.search(r"Supreme Court of California", court, flags=re.IGNORECASE)
    )
    california_jurisdiction = jurisdiction.lower() in {"california", "cal.", "ca"}
    state_appellate = court_type in {"SA", "ST", ""}
    return state_appellate and (california_court or california_jurisdiction and "california" in court.lower())


def court_level(row: dict[str, Any]) -> str:
    court = compact_inline(row.get("court_full_name") or row.get("court_short_name") or "")
    if re.search(r"Supreme Court", court, flags=re.IGNORECASE):
        return "California Supreme Court"
    if re.search(r"Court of Appeal|Court of Appeals", court, flags=re.IGNORECASE):
        return "California Court of Appeal"
    return "unknown"


def has_potential_fact_background(text: str, metadata: str) -> bool:
    if len(text) < MIN_TEXT_LENGTH:
        return False
    haystack = f"{metadata[:PREVIEW_CHARS]}\n{text[:6000]}"
    return regex_any(FACT_BACKGROUND_PATTERNS, haystack)


def has_concrete_liability_context(text: str, metadata: str, category: str) -> bool:
    haystack = f"{metadata}\n{text[:10000]}"
    if category in {
        "personal_injury",
        "auto_accident",
        "wrongful_death",
        "property_damage",
        "product_liability",
        "professional_negligence",
        "fraud_damages",
    }:
        return True
    if category == "contract_damages":
        return regex_any(
            [
                r"misrepresentation",
                r"fraud",
                r"property damage",
                r"construction defect",
                r"failed to disclose",
                r"physical damage",
                r"injur",
                r"accident",
            ],
            haystack,
        )
    if category == "unclear":
        return regex_any(
            [
                r"personal injur",
                r"bodily injur",
                r"was injured",
                r"were injured",
                r"\binjur(?:y|ies|ed)\b",
                r"accident",
                r"collision",
                r"slip and fall",
                r"fell",
                r"property damage",
                r"damage to .*property",
                r"defective condition",
                r"misrepresentation",
                r"fraud",
            ],
            haystack,
        )
    return False


def case_id_from_row(row: dict[str, Any], text: str = "") -> str:
    source_id = compact_inline(row.get("id", ""))
    title = compact_inline(row.get("case_name") or row.get("case_name_full") or row.get("case_name_short") or "")
    date = compact_inline(row.get("date_filed", ""))
    return stable_case_id(
        case_origin="US",
        jurisdiction="California",
        source_dataset=SOURCE_DATASET,
        source_id=source_id,
        title=title,
        date=date,
        raw_text=text,
    )


def apply_category_override(case_id: str, category: str) -> str:
    return CATEGORY_OVERRIDES.get(case_id, category)


def has_family_law_signal(title_and_preview: str) -> bool:
    if regex_any(FAMILY_LAW_STRONG_PATTERNS, title_and_preview):
        return True
    return regex_any(FAMILY_LAW_SUPPORT_PATTERNS, title_and_preview) and regex_any(
        [r"separate maintenance", r"\bdivorce\b", r"interlocutory decree"],
        title_and_preview,
    )


def has_contract_only_signal(title_and_preview: str, has_concrete_context: bool) -> bool:
    strong_contract_signal = regex_any(
        [
            r"promissory note",
            r"action on a promissory note",
            r"monthly installments of rent",
            r"lease of a storeroom",
            r"endorse the note",
            r"\bguaranty\b",
            r"\bdebt\b",
            r"\bcollection\b",
            r"\brent\b",
        ],
        title_and_preview,
    )
    if strong_contract_signal and not has_concrete_context:
        return True
    return regex_any([r"breach of contract"], title_and_preview) and not has_concrete_context


def category_guess(text: str, metadata: str) -> str:
    haystack = f"{metadata}\n{text[:8000]}"
    for category, patterns in CATEGORY_PATTERNS.items():
        if regex_any(patterns, haystack):
            return category
    if regex_any([r"damages", r"injury", r"accident", r"negligence", r"tort"], haystack):
        return "unclear"
    return "unclear"


def matches_collection_keywords(text: str, metadata: str) -> bool:
    return regex_any(HIGH_PRIORITY_KEYWORDS, f"{metadata}\n{text[:8000]}")


def evaluate_row(row: dict[str, Any], text: str) -> tuple[dict[str, object], str, str]:
    metadata = metadata_text(row)
    preview = f"{metadata}\n{text[:6000]}"
    title = compact_inline(row.get("case_name") or row.get("case_name_full") or row.get("case_name_short") or "")
    title_and_preview = f"{title}\n{text[:3000]}"
    court_preview = "\n".join(
        [
            compact_inline(row.get("court_full_name", "")),
            compact_inline(row.get("court_jurisdiction", "")),
            compact_inline(row.get("court_short_name", "")),
            citations_text(row),
            compact_inline(row.get("case_name") or row.get("case_name_full") or ""),
            compact_inline(row.get("headmatter", ""))[:PREVIEW_CHARS],
        ]
    )

    is_ca = is_california_state_court(row)
    is_federal = regex_any(FEDERAL_EXCLUDE_PATTERNS, court_preview)
    is_out_of_state = not is_ca or regex_any(OUT_OF_STATE_COURT_PATTERNS, court_preview)
    contains_criminal = regex_any(CRIMINAL_EXCLUDE_PATTERNS, preview)
    contains_admin = regex_any(ADMIN_EXCLUDE_PATTERNS, preview)
    contains_ip = regex_any(IP_EXCLUDE_PATTERNS, preview)
    contains_insurance = regex_any(INSURANCE_ONLY_PATTERNS, preview)
    contains_non_target_civil = regex_any(NON_TARGET_CIVIL_EXCLUDE_PATTERNS, preview)
    contains_family_law = has_family_law_signal(title_and_preview)
    contains_attorney_fee = regex_any(ATTORNEY_FEE_ONLY_PATTERNS, preview)
    contains_contract_only = regex_any(CONTRACT_ONLY_PATTERNS, preview)
    contains_procedure = regex_any(PROCEDURE_ONLY_PATTERNS, preview)
    has_full_text = len(text) >= MIN_TEXT_LENGTH
    has_fact_background = has_potential_fact_background(text, metadata)
    has_keywords = matches_collection_keywords(text, metadata)
    case_id = case_id_from_row(row, text)
    category = apply_category_override(case_id, category_guess(text, metadata))
    has_concrete_context = has_concrete_liability_context(text, metadata, category)
    contains_contract_only = contains_contract_only or has_contract_only_signal(title_and_preview, has_concrete_context)

    excluded_reason = ""
    notes = []
    status = "pass"
    if case_id in MANUAL_EXCLUDE_CASES:
        excluded_reason = MANUAL_EXCLUDE_CASES[case_id]
    elif not has_full_text:
        excluded_reason = "no_full_text_or_too_short"
    elif is_federal:
        excluded_reason = "federal_case"
    elif is_out_of_state:
        excluded_reason = "out_of_state_or_not_california_state"
    elif contains_criminal:
        excluded_reason = "criminal_habeas_signal"
    elif contains_admin:
        excluded_reason = "administrative_public_law_signal"
    elif contains_ip:
        excluded_reason = "ip_federal_statutory_signal"
    elif contains_insurance:
        excluded_reason = "insurance_coverage_only_signal"
    elif contains_non_target_civil:
        excluded_reason = "non_target_employment_or_public_policy_signal"
    elif contains_family_law:
        excluded_reason = "family_divorce_signal"
    elif contains_attorney_fee:
        excluded_reason = "attorney_fee_only_signal"
    elif contains_procedure and not has_fact_background:
        excluded_reason = "procedure_only_signal"
    elif contains_contract_only and not has_concrete_context:
        excluded_reason = "contract_only_without_tort_damages_facts"
    elif not has_keywords:
        excluded_reason = "no_liability_damages_keyword"
    elif not has_concrete_context:
        excluded_reason = "no_specific_liability_category"
    elif not has_fact_background:
        excluded_reason = "no_potential_fact_background"

    if excluded_reason:
        status = "fail"
    elif contains_procedure:
        status = "warning"
        notes.append("procedure term present but factual background detected")

    qc = {
        "is_california_state_case": is_ca,
        "is_federal_case": is_federal,
        "is_out_of_state_case": is_out_of_state,
        "contains_criminal_signal": contains_criminal,
        "contains_admin_signal": contains_admin,
        "contains_procedure_only_signal": contains_procedure,
        "has_full_text": has_full_text,
        "has_potential_fact_background": has_fact_background,
        "status": status,
        "excluded_reason": excluded_reason or None,
    }
    if contains_ip:
        notes.append("ip signal present")
    if contains_insurance:
        notes.append("insurance coverage signal present")
    if contains_family_law:
        notes.append("family/divorce signal present")
    return qc, "; ".join(notes), category


def make_record(
    row: dict[str, Any],
    text: str,
    qc: dict[str, object],
    notes: str,
    category: str,
    source_dataset: str,
) -> dict[str, object]:
    source_id = compact_inline(row.get("id", ""))
    title = compact_inline(row.get("case_name") or row.get("case_name_full") or row.get("case_name_short") or "")
    date = compact_inline(row.get("date_filed", ""))
    court = compact_inline(row.get("court_full_name") or row.get("court_short_name") or "")
    case_id = stable_case_id(
        case_origin="US",
        jurisdiction="California",
        source_dataset=source_dataset,
        source_id=source_id,
        title=title,
        date=date,
        raw_text=text,
    )
    return {
        "case_id": case_id,
        "case_origin": "US",
        "jurisdiction": "California",
        "court": court,
        "court_level": court_level(row),
        "source_title": title,
        "decision_date": date,
        "source_dataset": source_dataset,
        "source_url": first_download_url(row),
        "case_category_guess": category,
        "raw_text": text,
        "raw_text_excerpt": excerpt(text, 1000),
        "text_length_chars": len(text),
        "collection_qc": qc,
        "notes": notes,
    }


def qc_row(record: dict[str, object]) -> dict[str, object]:
    qc = record["collection_qc"]
    return {
        "case_id": record["case_id"],
        "source_title": record["source_title"],
        "court": record["court"],
        "decision_date": record["decision_date"],
        "text_length_chars": record["text_length_chars"],
        "case_category_guess": record["case_category_guess"],
        "is_california_state_case": qc["is_california_state_case"],
        "is_federal_case": qc["is_federal_case"],
        "is_out_of_state_case": qc["is_out_of_state_case"],
        "contains_criminal_signal": qc["contains_criminal_signal"],
        "contains_admin_signal": qc["contains_admin_signal"],
        "contains_procedure_only_signal": qc["contains_procedure_only_signal"],
        "has_full_text": qc["has_full_text"],
        "has_potential_fact_background": qc["has_potential_fact_background"],
        "status": qc["status"],
        "excluded_reason": qc["excluded_reason"] or "",
        "notes": record.get("notes", ""),
    }


def pass_quota_allows(record: dict[str, object], category_counts: Counter[str], target: int) -> bool:
    category = str(record["case_category_guess"])
    if category == "unclear":
        return category_counts[category] < max(3, target // 10)
    quota = TARGET_CATEGORY_QUOTAS.get(category, max(3, target // 10))
    # Allow overflow once underrepresented buckets are hard to find, but keep one category from dominating early.
    return category_counts[category] < quota or sum(category_counts.values()) >= target * 0.8


def collect(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    all_records: list[dict[str, object]] = []
    pass_records: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()

    for scanned, row in enumerate(dataset, start=1):
        if args.scan_limit and scanned > args.scan_limit:
            LOGGER.warning("Stopping at --scan-limit=%s", args.scan_limit)
            break
        text = row_text(row)
        qc, notes, category = evaluate_row(row, text)

        # Count only plausible California candidates toward max-candidates. This lets the stream skip the many
        # non-California rows without exhausting the candidate budget immediately.
        plausible_candidate = bool(qc["is_california_state_case"] or "california" in metadata_text(row).lower())
        if not plausible_candidate:
            counters["non_candidate_stream_rows"] += 1
            continue

        record = make_record(row, text, qc, notes, category, args.dataset)
        all_records.append(record)
        counters["candidates"] += 1
        counters[f"status_{qc['status']}"] += 1
        if qc["excluded_reason"]:
            counters[f"excluded_{qc['excluded_reason']}"] += 1

        if qc["status"] == "pass" and pass_quota_allows(record, category_counts, args.target_pass_count):
            pass_records.append(record)
            category_counts[str(record["case_category_guess"])] += 1

        if args.preview_only and len(all_records) >= args.preview_count:
            break
        if len(pass_records) >= args.target_pass_count:
            # Keep scanning a little after target if the candidate budget allows, so QC summary remains useful.
            if counters["candidates"] >= min(args.max_candidates, args.target_pass_count + args.extra_candidates_after_target):
                break
        if counters["candidates"] >= args.max_candidates:
            break
        if args.progress_every and scanned % args.progress_every == 0:
            LOGGER.info(
                "Scanned stream=%s candidates=%s pass_selected=%s",
                scanned,
                counters["candidates"],
                len(pass_records),
            )

    summary = summarize(all_records, pass_records)
    summary["stream_rows_skipped_as_non_candidates"] = int(counters["non_candidate_stream_rows"])
    return all_records, pass_records, summary


def summarize(all_records: list[dict[str, object]], pass_records: list[dict[str, object]]) -> dict[str, object]:
    qc_rows = [qc_row(record) for record in all_records]
    df = pd.DataFrame(qc_rows)
    pass_df = pd.DataFrame([qc_row(record) for record in pass_records])
    if df.empty:
        return {}
    excluded_reasons = Counter(row["excluded_reason"] for row in qc_rows if row["excluded_reason"])
    return {
        "total_candidates": int(len(all_records)),
        "california_state_case_candidates": int(df["is_california_state_case"].astype(bool).sum()),
        "federal_case_excluded": int(df["excluded_reason"].eq("federal_case").sum()),
        "out_of_state_case_excluded": int(df["excluded_reason"].eq("out_of_state_or_not_california_state").sum()),
        "criminal_habeas_excluded": int(df["excluded_reason"].eq("criminal_habeas_signal").sum()),
        "administrative_public_law_excluded": int(df["excluded_reason"].eq("administrative_public_law_signal").sum()),
        "procedure_only_excluded": int(df["excluded_reason"].eq("procedure_only_signal").sum()),
        "no_full_text_excluded": int(df["excluded_reason"].eq("no_full_text_or_too_short").sum()),
        "final_pass_count": int(pass_df["status"].eq("pass").sum()) if not pass_df.empty else 0,
        "warning_count": int(df["status"].eq("warning").sum()),
        "fail_count": int(df["status"].eq("fail").sum()),
        "pass_category_counts": {
            key: int(value) for key, value in Counter(record["case_category_guess"] for record in pass_records).items()
        },
        "average_raw_text_length": float(round(pass_df["text_length_chars"].astype(int).mean(), 2)) if not pass_df.empty else 0.0,
        "excluded_reason_top_20": dict(excluded_reasons.most_common(20)),
    }


def sanity_check(records: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    disallowed = FEDERAL_EXCLUDE_PATTERNS + [
        r"\bPeople v\.",
        r"\bhabeas\b",
        r"\bwarden\b",
        r"\bconviction\b",
        r"\bsentence\b",
    ]
    for record in records:
        qc = record["collection_qc"]
        text = f"{record['court']}\n{record['source_title']}\n{record['raw_text'][:PREVIEW_CHARS]}"
        if qc["status"] not in {"pass", "warning"}:
            errors.append(f"{record['case_id']}: non-pass selected")
        if regex_any(disallowed, text):
            errors.append(f"{record['case_id']}: disallowed federal/criminal signal")
        if not qc["is_california_state_case"]:
            errors.append(f"{record['case_id']}: missing California state court metadata")
        if int(record["text_length_chars"]) < MIN_TEXT_LENGTH:
            errors.append(f"{record['case_id']}: raw text too short")
        if not qc["has_potential_fact_background"]:
            errors.append(f"{record['case_id']}: no potential factual background")
    return errors


def write_outputs(
    all_records: list[dict[str, object]],
    pass_records: list[dict[str, object]],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Path]:
    raw_path = output_dir / "us_california_cases_raw.jsonl"
    qc_path = output_dir / "us_california_cases_qc.csv"
    summary_path = output_dir / "us_california_cases_summary.json"
    for path in [raw_path, qc_path, summary_path]:
        require_overwrite(path, overwrite)
        ensure_parent(path)

    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in pass_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    qc_df = pd.DataFrame(qc_row(record) for record in all_records)
    qc_df.to_csv(qc_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary = summarize(all_records, pass_records)
    summary["sanity_check_errors"] = sanity_check(pass_records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"raw": raw_path, "qc": qc_path, "summary": summary_path}


def print_summary(summary: dict[str, object]) -> None:
    print("California raw case collection summary")
    labels = [
        "total_candidates",
        "california_state_case_candidates",
        "federal_case_excluded",
        "out_of_state_case_excluded",
        "criminal_habeas_excluded",
        "administrative_public_law_excluded",
        "procedure_only_excluded",
        "no_full_text_excluded",
        "final_pass_count",
        "warning_count",
        "fail_count",
        "pass_category_counts",
        "average_raw_text_length",
        "excluded_reason_top_20",
        "sanity_check_errors",
        "stream_rows_skipped_as_non_candidates",
    ]
    for label in labels:
        if label in summary:
            print(f"{label}: {summary[label]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect California state civil liability/damages raw opinions.")
    parser.add_argument("--dataset", default=SOURCE_DATASET)
    parser.add_argument("--split", default=SOURCE_SPLIT)
    parser.add_argument("--target-pass-count", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--extra-candidates-after-target", type=int, default=40)
    parser.add_argument("--scan-limit", type=int, default=500_000)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    for noisy_logger in ["httpx", "httpcore", "huggingface_hub", "datasets"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    all_records, pass_records, summary = collect(args)
    summary["sanity_check_errors"] = sanity_check(pass_records)
    if args.preview_only:
        print_summary(summary)
        for record in pass_records[: args.preview_count]:
            preview = json.dumps({k: record[k] for k in record if k != "raw_text"}, ensure_ascii=False, indent=2)
            print(preview.encode("utf-8", errors="replace").decode("utf-8"))
        return

    paths = write_outputs(all_records, pass_records, Path(args.output_dir), args.overwrite)
    print_summary(summary)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
