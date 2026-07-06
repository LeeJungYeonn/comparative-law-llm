from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from pipeline.io_utils import require_overwrite, stable_case_id
from pipeline.text_utils import compact_inline


LOGGER = logging.getLogger(__name__)

US_DATASET = "harvard-lil/cold-cases"
US_SPLIT = "train"
US_COURT_TYPES = {"FD", "ST", "FA", "SA"}
US_NOS_KEYWORDS = [
    "tort",
    "personal injury",
    "negligence",
    "property damage",
    "product liability",
    "medical malpractice",
    "wrongful death",
    "civil rights",
]
US_TEXT_KEYWORDS = [
    "damages",
    "negligence",
    "tort",
    "duty of care",
    "proximate cause",
    "personal injury",
    "wrongful death",
    "property damage",
    "civil liability",
    "compensatory damages",
]

STATE_ALIASES = {
    "California": ["california", "cal.", "ca"],
    "New York": ["new york", "n.y.", "ny"],
}

STATE_METADATA_COLUMNS = (
    "state",
    "state_name",
    "jurisdiction",
    "jurisdiction_name",
    "court_state",
)

STATE_TEXT_COLUMNS = (
    "court_full_name",
    "court",
    "court_name",
    "case_name",
    "name",
    "jurisdiction",
)


def extract_opinion_text(row: dict[str, Any]) -> str:
    opinions = row.get("opinions") or []
    if isinstance(opinions, list):
        text = "\n\n".join(
            compact_inline(opinion.get("opinion_text") or opinion.get("text") or "")
            for opinion in opinions
            if isinstance(opinion, dict)
        ).strip()
        if text:
            return text
    return compact_inline(row.get("opinion_text") or row.get("text") or row.get("raw_text") or "")


def matches_keyword_criteria(row: dict[str, Any], text: str) -> tuple[bool, str]:
    court_type = compact_inline(row.get("court_type", "")).upper()
    if court_type and court_type not in US_COURT_TYPES:
        return False, f"excluded_court_type_{court_type}"

    nature_of_suit = compact_inline(row.get("nature_of_suit", "")).lower()
    if any(keyword in nature_of_suit for keyword in US_NOS_KEYWORDS):
        return True, "matched_nature_of_suit"

    lowered = text.lower()
    if any(keyword in lowered for keyword in US_TEXT_KEYWORDS):
        return True, "matched_text_keyword"
    return False, "no_tort_damages_keyword"


def _state_match(value: object, target_state: str) -> bool:
    text = compact_inline(value).lower()
    if not text:
        return False
    aliases = STATE_ALIASES[target_state]
    for alias in aliases:
        if len(alias) <= 2 and alias.isalpha():
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return True
        elif alias in text:
            return True
    return False


def classify_state(row: dict[str, Any], target_state: str | None) -> tuple[bool, str, str]:
    if not target_state:
        return True, "unavailable", ""
    if target_state not in STATE_ALIASES:
        raise ValueError(f"Unsupported --state {target_state!r}. Use California or New York.")

    exact_hits = []
    for column in STATE_METADATA_COLUMNS:
        if column in row and _state_match(row.get(column), target_state):
            exact_hits.append(column)

    other_state_hits = []
    for state in STATE_ALIASES:
        if state == target_state:
            continue
        for column in STATE_METADATA_COLUMNS + STATE_TEXT_COLUMNS:
            if column in row and _state_match(row.get(column), state):
                other_state_hits.append(f"{state}:{column}")

    inferred_hits = []
    for column in STATE_TEXT_COLUMNS:
        if column in row and _state_match(row.get(column), target_state):
            inferred_hits.append(column)

    if exact_hits and not other_state_hits:
        return True, "exact", "; ".join(exact_hits)
    if (exact_hits or inferred_hits) and other_state_hits:
        return False, "ambiguous", "; ".join(exact_hits + inferred_hits + other_state_hits)
    if inferred_hits:
        return True, "inferred", "; ".join(inferred_hits)
    return False, "unavailable", ""


def collect_us_cases(args: argparse.Namespace) -> pd.DataFrame:
    LOGGER.info("Loading %s split=%s streaming=True", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.seed and args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    records: list[dict[str, object]] = []
    scanned = 0
    skipped = {"state": 0, "keyword": 0, "length": 0}

    for row in dataset:
        scanned += 1
        if args.scan_limit and scanned > args.scan_limit:
            LOGGER.warning("Stopped at --scan-limit=%s with %s collected rows", args.scan_limit, len(records))
            break

        text = extract_opinion_text(row)
        text_length = len(text)
        if text_length < args.min_text_length or (args.max_text_length and text_length > args.max_text_length):
            skipped["length"] += 1
            continue

        state_keep, state_status, state_notes = classify_state(row, args.state)
        if not state_keep and not (args.include_ambiguous and state_status == "ambiguous"):
            skipped["state"] += 1
            continue

        keyword_keep, keyword_reason = matches_keyword_criteria(row, text)
        if not keyword_keep:
            skipped["keyword"] += 1
            continue

        source_id = compact_inline(row.get("id", "")) or f"scanned_{scanned}"
        case_name = compact_inline(row.get("case_name") or row.get("name") or "")
        court_name = compact_inline(row.get("court_full_name") or row.get("court") or "")
        court_type = compact_inline(row.get("court_type", "")).upper()
        court_level = "trial" if court_type in {"FD", "ST"} else "appellate" if court_type else ""
        jurisdiction = args.state or "Unknown"
        case_id = stable_case_id(
            case_origin="US",
            jurisdiction=jurisdiction,
            source_dataset=args.dataset,
            source_id=source_id,
            title=case_name,
            date=compact_inline(row.get("date_filed") or row.get("date") or ""),
            raw_text=text,
        )
        records.append(
            {
                "case_id": case_id,
                "source_id": source_id,
                "jurisdiction": "US",
                "state": jurisdiction,
                "case_name": case_name,
                "court_name": court_name,
                "court_level": court_level,
                "decision_date": compact_inline(row.get("date_filed") or row.get("date") or ""),
                "nature_of_suit": compact_inline(row.get("nature_of_suit", "")),
                "source_dataset": args.dataset,
                "state_filter_status": state_status,
                "state_filter_notes": state_notes,
                "us_filter_reason": keyword_reason,
                "collection_notes": "; ".join(
                    note for note in [keyword_reason, f"state_{state_status}", state_notes] if note
                ),
                "quality_flags": "ambiguous_us_state" if state_status == "ambiguous" else "",
                "raw_text": text,
            }
        )

        if len(records) >= args.limit:
            break

    LOGGER.info(
        "Scanned=%s collected=%s skipped_state=%s skipped_keyword=%s skipped_length=%s",
        scanned,
        len(records),
        skipped["state"],
        skipped["keyword"],
        skipped["length"],
    )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect US tort/damages cases from harvard-lil/cold-cases.")
    parser.add_argument("--dataset", default=US_DATASET)
    parser.add_argument("--split", default=US_SPLIT)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--scan-limit", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--state", choices=sorted(STATE_ALIASES), default="")
    parser.add_argument("--include-ambiguous", action="store_true")
    parser.add_argument("--output", default="outputs/us_cases.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-text-length", type=int, default=1_000)
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    output_path = Path(args.output)
    require_overwrite(output_path, args.overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collected = collect_us_cases(args)
    collected.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Collected US cases: {len(collected):,}")
    if not collected.empty:
        print("state_filter_status")
        print(collected["state_filter_status"].value_counts(dropna=False).to_string())
        print("filter_reason")
        print(collected["us_filter_reason"].value_counts(dropna=False).to_string())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
