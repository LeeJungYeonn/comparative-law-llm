from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset


DEFAULT_KEYWORDS = [
    "손해배상",
    "손해배상(기)",
    "손해배상(의)",
    "불법행위",
    "위자료",
    "지연손해금",
    "과실상계",
    "민법 제750조",
    "민법 제393조",
    "민법 제763조",
]

TEXT_COLUMN_CANDIDATES = ("precedent", "raw_text", "text", "facts", "reason", "ruling")
CASE_NUMBER_COLUMN_CANDIDATES = (
    "case_number_or_citation",
    "case_number",
    "case_no",
    "caseno",
    "사건번호",
)
COURT_COLUMN_CANDIDATES = ("court_name", "court", "court_full_name", "법원명")
DATE_COLUMN_CANDIDATES = ("decision_date", "date", "선고일자", "판결선고일")
CASE_NAME_COLUMN_CANDIDATES = ("case_name", "casename", "사건명")

INCLUDE_CASE_CODES = ("가합", "가단", "가소", "나")
EXCLUDE_CASE_CODES = ("다",)
INCLUDE_COURT_MARKERS = ("지방법원", "지원", "고등법원")
EXCLUDE_COURT_MARKERS = ("대법원",)

SUPREME_TEXT_MARKERS = (
    "상고이유",
    "상고를 기각",
    "상고를 모두 기각",
    "상고를 받아들",
    "상고심",
    "원심판결을 파기",
    "원심법원에 환송",
    "대법관의 일치된 의견",
)
APPELLATE_TEXT_MARKERS = (
    "항소",
    "항소취지",
    "제1심 판결",
    "제1심판결",
    "원심판결",
    "당심",
)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def extract_case_number(text: str) -> str:
    match = re.search(r"\b\d{4}\s*[가-힣]{1,5}\s*\d+\b", text)
    return normalize_text(match.group(0)) if match else ""


def extract_target_case_number(text: str) -> str:
    target_codes = "|".join(INCLUDE_CASE_CODES + EXCLUDE_CASE_CODES)
    match = re.search(rf"\b\d{{4}}\s*(?:{target_codes})\s*\d+\b", text)
    return normalize_text(match.group(0)) if match else ""


def extract_case_code(case_number: str) -> str:
    match = re.search(r"\d{4}\s*([가-힣]{1,5})\s*\d+", normalize_text(case_number))
    return match.group(1) if match else ""


def infer_level(court_name: str, case_number: str, text: str) -> str:
    case_code = extract_case_code(case_number)
    if case_code in EXCLUDE_CASE_CODES or any(marker in court_name for marker in EXCLUDE_COURT_MARKERS):
        return "supreme"
    if case_code == "나" or "고등법원" in court_name:
        return "appellate"
    if case_code in {"가합", "가단", "가소"} or "지방법원" in court_name or "지원" in court_name:
        return "trial"
    if any(marker in text for marker in APPELLATE_TEXT_MARKERS):
        return "appellate"
    return "trial_or_appellate_unknown"


def classify_kr_row(court_name: str, case_number: str, text: str) -> tuple[bool, str, str]:
    case_code = extract_case_code(case_number)

    if any(marker in court_name for marker in EXCLUDE_COURT_MARKERS):
        return False, "excluded_court_supreme", "supreme"
    if case_code in EXCLUDE_CASE_CODES:
        return False, f"excluded_case_code_{case_code}", "supreme"
    if any(marker in court_name for marker in INCLUDE_COURT_MARKERS):
        return True, "included_court_trial_appellate", infer_level(court_name, case_number, text)
    if case_code in INCLUDE_CASE_CODES:
        return True, f"included_case_code_{case_code}", infer_level(court_name, case_number, text)

    if any(marker in text for marker in SUPREME_TEXT_MARKERS):
        return False, "excluded_text_supreme_marker", "supreme"
    if any(marker in text for marker in APPELLATE_TEXT_MARKERS):
        return True, "included_text_appellate_marker", "appellate"

    # precedent_corpus has only id + precedent, so many first-instance cases lack metadata.
    return True, "included_no_supreme_signal", "trial_or_appellate_unknown"


def keyword_mask(series: pd.Series, keywords: list[str]) -> pd.Series:
    pattern = "|".join(re.escape(keyword) for keyword in keywords if keyword)
    if not pattern:
        return pd.Series([True] * len(series), index=series.index)
    return series.str.contains(pattern, na=False, regex=True)


def collect_kr_cases(args: argparse.Namespace) -> pd.DataFrame:
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    df = dataset.to_pandas()

    text_col = args.text_col or first_existing_column(df.columns, TEXT_COLUMN_CANDIDATES)
    if not text_col:
        raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

    court_col = first_existing_column(df.columns, COURT_COLUMN_CANDIDATES)
    case_number_col = first_existing_column(df.columns, CASE_NUMBER_COLUMN_CANDIDATES)
    date_col = first_existing_column(df.columns, DATE_COLUMN_CANDIDATES)
    case_name_col = first_existing_column(df.columns, CASE_NAME_COLUMN_CANDIDATES)

    df[text_col] = df[text_col].fillna("").astype(str)
    filtered = df[keyword_mask(df[text_col], args.keywords)].copy()

    records: list[dict[str, object]] = []
    scanned = 0
    for _, row in filtered.iterrows():
        scanned += 1
        raw_text = normalize_text(row.get(text_col, ""))
        if len(raw_text) < args.min_text_length:
            continue
        if args.max_text_length and len(raw_text) > args.max_text_length:
            continue
        court_name = normalize_text(row.get(court_col, "")) if court_col else ""
        case_number = normalize_text(row.get(case_number_col, "")) if case_number_col else ""
        case_number = case_number or extract_target_case_number(raw_text)

        keep, reason, court_level = classify_kr_row(court_name, case_number, raw_text)
        if not keep:
            continue

        records.append(
            {
                "case_id": normalize_text(row.get("id", "")),
                "jurisdiction": "KR",
                "case_name": normalize_text(row.get(case_name_col, "")) if case_name_col else "",
                "case_number_or_citation": case_number,
                "court_name": court_name,
                "court_level": court_level,
                "decision_date": normalize_text(row.get(date_col, "")) if date_col else "",
                "kr_filter_reason": reason,
                "raw_text": row.get(text_col, ""),
            }
        )

        if args.scan_limit and scanned >= args.scan_limit:
            break

    collected = pd.DataFrame(records)
    if collected.empty:
        return collected

    sample_size = args.limit if args.limit is not None else args.sample_size
    if sample_size and len(collected) > sample_size:
        collected = collected.sample(n=sample_size, random_state=args.seed)

    return collected.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Korean trial/appellate-oriented cases from lbox_open."
    )
    parser.add_argument("--dataset", default="lbox/lbox_open")
    parser.add_argument("--config", default="precedent_corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-col", default="")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Alias for --sample-size.")
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/kr_cases.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-text-length", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collected = collect_kr_cases(args)
    collected.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Collected KR cases: {len(collected):,}")
    if not collected.empty:
        print("court_level")
        print(collected["court_level"].value_counts(dropna=False).to_string())
        print("filter_reason")
        print(collected["kr_filter_reason"].value_counts(dropna=False).to_string())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
