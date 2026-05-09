from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


KR_SOURCE = "lbox/lbox_open::precedent_corpus"
US_SOURCE = "harvard-lil/cold-cases"

KR_KEYWORDS = [
    "손해배상",
    "손해배상(기)",
    "손해배상(의)",
    "불법행위",
    "위자료",
    "지연손해금",
    "과실상계",
    "상당인과관계",
    "민법 제750조",
    "민법 제393조",
    "민법 제763조",
]

US_KEYWORDS = [
    "damages",
    "civil damages",
    "negligence",
    "tort",
    "duty of care",
    "proximate cause",
    "comparative negligence",
    "compensatory damages",
    "punitive damages",
    "emotional distress",
    "personal injury",
    "breach of duty",
]

METADATA_COLUMNS = [
    "case_id",
    "jurisdiction",
    "country",
    "court_name",
    "court_level",
    "decision_date",
    "case_name",
    "case_number_or_citation",
    "source",
    "case_type_keyword",
]

TEXT_COLUMNS = [
    "raw_text",
    "clean_text",
    "reasoning_text",
]


@dataclass(frozen=True)
class PreprocessConfig:
    min_reasoning_tokens: int = 500
    drop_short: bool = False


def normalize_whitespace(text: object) -> str:
    """Normalize OCR/API spacing while keeping paragraph boundaries."""
    if pd.isna(text):
        return ""

    value = html.unescape(str(text))
    value = value.replace("\ufeff", "").replace("\xa0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def strip_kr_appendix_and_footnotes(text: str) -> str:
    value = text
    value = re.sub(r"\n?\[별지\s*생략\].*$", "", value, flags=re.DOTALL)
    value = re.sub(r"\n?별\s*지\s*$.*", "", value, flags=re.DOTALL | re.MULTILINE)
    value = re.sub(r"\n주\d+\)\s+.*$", "", value, flags=re.DOTALL)
    return normalize_whitespace(value)


def strip_us_notes_and_page_markers(text: str) -> str:
    value = text
    value = re.split(r"\n(?:NOTES|FOOTNOTES)\n", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\*\d{1,4}", " ", value)
    value = re.sub(r"\[(?:\d+|[ivxlcdm]+)\]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\n\s*Page\s+\d+\s*\n", "\n", value, flags=re.IGNORECASE)
    return normalize_whitespace(value)


def extract_kr_reasoning_text(clean_text: str) -> tuple[str, bool]:
    patterns = [
        r"(?m)^\s*이\s*유\s*$",
        r"(?m)^\s*이유\s*$",
        r"(?m)^\s*\d+\.\s*이\s*유\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean_text)
        if match:
            return normalize_whitespace(clean_text[match.end() :]), True

    fallback = re.search(r"(?m)^\s*(?:[가-하]\.|[IVX]+\.)?\s*판\s*단\s*$", clean_text)
    if fallback:
        return normalize_whitespace(clean_text[fallback.start() :]), True

    return clean_text, False


def extract_us_opinion_text(clean_text: str) -> tuple[str, bool]:
    patterns = [
        r"(?m)^\s*(?:MEMORANDUM OPINION|MEMORANDUM AND ORDER|OPINION AND ORDER|DECISION AND ORDER|ORDER AND REASONS|FINDINGS OF FACT AND CONCLUSIONS OF LAW|OPINION|ORDER)\s*$",
        r"(?m)^\s*[A-Z][A-Z .,'-]{2,},\s+(?:Chief\s+)?(?:Circuit|District|Magistrate|Bankruptcy)\s+Judge\.?\s*$",
        r"(?m)^\s*(?:I\.|1\.)\s+(?:BACKGROUND|INTRODUCTION|FACTUAL BACKGROUND|PROCEDURAL BACKGROUND)\s*$",
    ]

    candidates: list[re.Match[str]] = []
    for pattern in patterns:
        candidates.extend(re.finditer(pattern, clean_text, flags=re.IGNORECASE))

    if not candidates:
        return clean_text, False

    # Prefer a body marker after the citation/caption block, but do not skip too far.
    candidates.sort(key=lambda item: item.start())
    for match in candidates:
        if match.start() > 100:
            return normalize_whitespace(clean_text[match.start() :]), True
    return normalize_whitespace(clean_text[candidates[0].start() :]), True


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:,\d{3})*(?:\.\d+)?|§+|제\d+조",
        text,
    )


def split_sentences(text: str) -> list[str]:
    value = normalize_whitespace(text)
    if not value:
        return []

    protected = protect_common_abbreviations(value)
    parts = re.split(r"(?<=[.!?。！？])\s+(?=[A-Z가-힣\"'“‘(\[])", protected)
    sentences = [restore_common_abbreviations(part).strip() for part in parts]
    return [sentence for sentence in sentences if sentence]


def protect_common_abbreviations(text: str) -> str:
    replacements = {
        "U.S.": "U<S>",
        "U.S.C.": "U<S<C>",
        "F. Supp.": "F< Supp>",
        "F.2d": "F<2d",
        "F.3d": "F<3d",
        "S. Ct.": "S< Ct>",
        "L. Ed.": "L< Ed>",
        "No.": "No<",
        "Inc.": "Inc<",
        "Corp.": "Corp<",
        "Co.": "Co<",
        "Ltd.": "Ltd<",
        "v.": "v<",
    }
    value = text
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def restore_common_abbreviations(text: str) -> str:
    replacements = {
        "U<S<C>": "U.S.C.",
        "U<S>": "U.S.",
        "F< Supp>": "F. Supp.",
        "F<2d": "F.2d",
        "F<3d": "F.3d",
        "S< Ct>": "S. Ct.",
        "L< Ed>": "L. Ed.",
        "No<": "No.",
        "Inc<": "Inc.",
        "Corp<": "Corp.",
        "Co<": "Co.",
        "Ltd<": "Ltd.",
        "v<": "v.",
    }
    value = text
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def detect_keywords(text: str, keywords: Iterable[str], case_sensitive: bool = False) -> str:
    haystack = text if case_sensitive else text.lower()
    found = []
    for keyword in keywords:
        needle = keyword if case_sensitive else keyword.lower()
        if needle in haystack:
            found.append(keyword)
    return "; ".join(dict.fromkeys(found))


def infer_country(jurisdiction: str) -> str:
    return "Korea" if jurisdiction == "KR" else "United States"


def infer_kr_court_level(court_name: str, text: str) -> str:
    if "대법원" in court_name:
        return "supreme"
    if any(marker in text for marker in ["항소", "원심판결", "제1심 판결", "당심"]):
        return "appellate"
    if any(marker in text for marker in ["지방법원", "지원", "1심"]):
        return "trial"
    return "unknown"


def infer_us_court_level(court_name: str, current_value: str) -> str:
    current = normalize_whitespace(current_value).lower()
    if current:
        return current

    court = court_name.lower()
    if "supreme court" in court:
        return "supreme"
    if "court of appeals" in court or "cir." in court:
        return "appellate"
    if "district court" in court or "bankruptcy court" in court:
        return "trial"
    return "unknown"


def normalize_date(value: object) -> str:
    if pd.isna(value):
        return ""
    text = normalize_whitespace(value)
    if not text:
        return ""

    iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    kr_match = re.search(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?", text)
    if kr_match:
        year, month, day = kr_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return text


def extract_kr_decision_date(text: str) -> str:
    anchored_patterns = [
        r"판결\s*선고일인\s*(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?)",
        r"선고일인\s*(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?)",
        r"선고\s*(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?)",
    ]
    for pattern in anchored_patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_date(match.group(1))
    return ""


def extract_kr_court_name(text: str) -> str:
    courts = [
        "대법원",
        "서울고등법원",
        "부산고등법원",
        "대구고등법원",
        "광주고등법원",
        "대전고등법원",
        "수원고등법원",
        "서울중앙지방법원",
        "서울동부지방법원",
        "서울서부지방법원",
        "서울남부지방법원",
        "서울북부지방법원",
        "부산지방법원",
        "대구지방법원",
        "인천지방법원",
        "광주지방법원",
        "대전지방법원",
        "수원지방법원",
    ]
    for court in courts:
        if court in text:
            return court
    return ""


def extract_case_number_or_citation(jurisdiction: str, text: str) -> str:
    if jurisdiction == "KR":
        patterns = [
            r"\d{4}[가-힣]{1,5}\d+",
            r"\d{4}\s*[가-힣]{1,5}\s*\d+",
        ]
    else:
        patterns = [
            r"\d+\s+F\.\s?Supp\.?\s?\d*d?\s+\d+\s*\(\d{4}\)",
            r"\d+\s+F\.\d+d\s+\d+",
            r"\d+\s+U\.S\.\s+\d+",
            r"(?:Civil|Case|Docket)\s+No\.?\s+[A-Za-z0-9:._/-]+",
            r"No\.?\s+[A-Za-z0-9:._/-]+",
        ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_whitespace(match.group(0))
    return ""


def build_preprocess_notes(
    has_reasoning_marker: bool,
    clean_text: str,
    reasoning_text: str,
    min_reasoning_tokens: int,
) -> str:
    notes = []
    notes.append("reasoning_marker_found" if has_reasoning_marker else "reasoning_marker_missing_used_clean_text")
    if "[별지 생략]" in clean_text or re.search(r"(?m)^별\s*지\s*$", clean_text):
        notes.append("possible_appendix_remaining")
    if len(tokenize(reasoning_text)) < min_reasoning_tokens:
        notes.append(f"below_min_reasoning_tokens_{min_reasoning_tokens}")
    return "; ".join(notes)


def preprocess_kr(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        raw_text = normalize_whitespace(row.get("raw_text", ""))
        clean_text = strip_kr_appendix_and_footnotes(raw_text)
        reasoning_text, has_reasoning_marker = extract_kr_reasoning_text(clean_text)

        court_name = normalize_whitespace(row.get("court_name", "")) or extract_kr_court_name(raw_text)
        decision_date = normalize_date(row.get("decision_date", "")) or extract_kr_decision_date(raw_text)
        court_level = normalize_whitespace(row.get("court_level", "")) or infer_kr_court_level(
            court_name, raw_text
        )

        records.append(
            build_record(
                row=row,
                jurisdiction="KR",
                country="Korea",
                source=KR_SOURCE,
                court_name=court_name,
                court_level=court_level,
                decision_date=decision_date,
                case_type_keyword=detect_keywords(raw_text, KR_KEYWORDS, case_sensitive=True),
                case_number_or_citation=extract_case_number_or_citation("KR", raw_text),
                raw_text=raw_text,
                clean_text=clean_text,
                reasoning_text=reasoning_text,
                has_reasoning_marker=has_reasoning_marker,
                config=config,
            )
        )
    return pd.DataFrame(records)


def preprocess_us(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        raw_text = normalize_whitespace(row.get("raw_text", ""))
        clean_text = strip_us_notes_and_page_markers(raw_text)
        reasoning_text, has_reasoning_marker = extract_us_opinion_text(clean_text)

        court_name = normalize_whitespace(row.get("court_name", ""))
        court_level = infer_us_court_level(court_name, row.get("court_level", ""))

        keyword_text = " ".join(
            [
                normalize_whitespace(row.get("nature_of_suit", "")),
                raw_text,
            ]
        )

        records.append(
            build_record(
                row=row,
                jurisdiction="US",
                country="United States",
                source=US_SOURCE,
                court_name=court_name,
                court_level=court_level,
                decision_date=normalize_date(row.get("decision_date", "")),
                case_type_keyword=detect_keywords(keyword_text, US_KEYWORDS),
                case_number_or_citation=extract_case_number_or_citation("US", raw_text),
                raw_text=raw_text,
                clean_text=clean_text,
                reasoning_text=reasoning_text,
                has_reasoning_marker=has_reasoning_marker,
                config=config,
            )
        )
    return pd.DataFrame(records)


def build_record(
    *,
    row: pd.Series,
    jurisdiction: str,
    country: str,
    source: str,
    court_name: str,
    court_level: str,
    decision_date: str,
    case_type_keyword: str,
    case_number_or_citation: str,
    raw_text: str,
    clean_text: str,
    reasoning_text: str,
    has_reasoning_marker: bool,
    config: PreprocessConfig,
) -> dict[str, object]:
    clean_tokens = tokenize(clean_text)
    reasoning_tokens = tokenize(reasoning_text)
    clean_sentences = split_sentences(clean_text)
    reasoning_sentences = split_sentences(reasoning_text)
    avg_sentence_length = (
        round(len(reasoning_tokens) / len(reasoning_sentences), 2) if reasoning_sentences else 0
    )

    return {
        "case_id": normalize_whitespace(row.get("case_id", "")),
        "jurisdiction": jurisdiction,
        "country": country,
        "court_name": court_name,
        "court_level": court_level or "unknown",
        "decision_date": decision_date,
        "case_name": normalize_whitespace(row.get("case_name", "")),
        "case_number_or_citation": case_number_or_citation,
        "source": source,
        "case_type_keyword": case_type_keyword,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "reasoning_text": reasoning_text,
        "raw_char_count": len(raw_text),
        "clean_char_count": len(clean_text),
        "reasoning_char_count": len(reasoning_text),
        "clean_token_count": len(clean_tokens),
        "clean_sentence_count": len(clean_sentences),
        "token_count": len(reasoning_tokens),
        "sentence_count": len(reasoning_sentences),
        "avg_sentence_length": avg_sentence_length,
        "has_reasoning_or_opinion_marker": has_reasoning_marker,
        "meets_min_reasoning_tokens": len(reasoning_tokens) >= config.min_reasoning_tokens,
        "preprocess_notes": build_preprocess_notes(
            has_reasoning_marker, clean_text, reasoning_text, config.min_reasoning_tokens
        ),
    }


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def summarize(df: pd.DataFrame) -> dict[str, object]:
    grouped = df.groupby("jurisdiction", dropna=False)
    return {
        "total_rows": int(len(df)),
        "rows_by_jurisdiction": {key: int(value) for key, value in grouped.size().to_dict().items()},
        "reasoning_token_count": {
            key: {
                "min": int(group["token_count"].min()),
                "median": float(group["token_count"].median()),
                "max": int(group["token_count"].max()),
            }
            for key, group in grouped
        },
        "below_min_reasoning_tokens": {
            key: int((~group["meets_min_reasoning_tokens"]).sum()) for key, group in grouped
        },
        "missing_reasoning_marker": {
            key: int((~group["has_reasoning_or_opinion_marker"]).sum()) for key, group in grouped
        },
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    config = PreprocessConfig(
        min_reasoning_tokens=args.min_reasoning_tokens,
        drop_short=args.drop_short,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kr_df = load_csv(Path(args.kr_csv))
    us_df = load_csv(Path(args.us_csv))

    combined = pd.concat(
        [
            preprocess_kr(kr_df, config),
            preprocess_us(us_df, config),
        ],
        ignore_index=True,
    )

    if config.drop_short:
        combined = combined[combined["meets_min_reasoning_tokens"]].copy()

    processed_path = output_dir / args.processed_name
    metadata_path = output_dir / args.metadata_name
    summary_path = output_dir / args.summary_name

    combined.to_csv(processed_path, index=False, encoding="utf-8-sig")
    combined[METADATA_COLUMNS].to_csv(metadata_path, index=False, encoding="utf-8-sig")
    summary = summarize(combined)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "processed": processed_path,
        "metadata": metadata_path,
        "summary": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Korean and US case CSVs for comparative law corpus analysis."
    )
    parser.add_argument("--kr-csv", default="outputs/kr_cases.csv")
    parser.add_argument("--us-csv", default="outputs/us_cases.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--processed-name", default="preprocessed_cases.csv")
    parser.add_argument("--metadata-name", default="case_metadata.csv")
    parser.add_argument("--summary-name", default="preprocessing_summary.json")
    parser.add_argument(
        "--min-reasoning-tokens",
        type=int,
        default=500,
        help="Minimum reasoning/opinion body token threshold used for quality flags.",
    )
    parser.add_argument(
        "--drop-short",
        action="store_true",
        help="Drop rows below --min-reasoning-tokens instead of only flagging them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    paths = run(parse_args())
    print("Preprocessing complete")
    for label, path in paths.items():
        print(f"{label}: {path}")
