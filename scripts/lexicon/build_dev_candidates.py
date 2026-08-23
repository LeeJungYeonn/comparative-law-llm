#!/usr/bin/env python3
"""Build transparent legal-lexicon candidates from the frozen development split.

Confirmatory opinion text is never selected, tokenized, or passed to analysis.
The module is standard-library only and contains no random operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs_v2/lexicon_dev_candidates_v0_1"
SEED_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_seed_v0_1_1"
NGRAM_RANGE = (1, 2, 3, 4, 5)
PRIOR_STRENGTH = 100.0
DOMAIN_CONCENTRATION_THRESHOLD = 0.80
STATE_CONCENTRATION_THRESHOLD = 0.80
CONTEXT_LIMIT = 5
STOPWORD_VERSION = "lexicon-dev-stopwords-v0.1"

EN_STOPWORDS = frozenset(
    "a an and are as at be been being but by did do does for from had has have he her "
    "hers him his i if in into is it its may nor not of on or our she so than that the "
    "their them there these they this those to under upon was we were which who whom "
    "will with would you your v vs".split()
)
KO_STOPWORDS = frozenset(
    "그 및 이 저 것 수 등 또는 그리고 그러나 또한 대한 대하여 관한 관하여 의 에 에서 로 으로 와 과 은 는 이 가 을 를 도 만 한 하여 있는 있다 없는 없다 한다 하였다 경우 때 바".split()
)

METADATA_FIELDS = (
    "case_id", "analysis_split", "origin_country", "origin_state", "primary_domain",
    "case_domain", "decision_date", "court_name", "main_opinion_type",
)

BASE_COLUMNS = [
    "candidate_id", "ngram", "normalized_ngram", "language", "n", "corpus_direction",
    "kr_count", "us_count", "kr_df", "us_df", "kr_df_prop", "us_df_prop",
    "log_odds_occurrence", "z_log_odds_occurrence",
    "log_odds_document_presence", "z_log_odds_document_presence",
    "meets_df2_threshold", "kr_domain_count", "us_domain_count", "kr_domains",
    "us_domains", "max_domain_df", "max_domain_df_share", "domain_concentrated",
    "us_state_count", "us_states", "max_state_df", "max_state_df_share",
    "state_concentrated", "max_state_vs_rest_z", "max_state_vs_rest_state",
    "ranking_score", "noise_flags", "example_case_ids", "example_contexts_short",
    "candidate_source", "already_in_seed", "seed_surface_ids", "seed_concept_ids",
]
HUMAN_COLUMNS = [
    "human_category", "human_concept_family", "human_concept_id",
    "human_jurisdiction_scope", "human_keep", "human_notes",
]
ALL_COLUMNS = BASE_COLUMNS + HUMAN_COLUMNS


@dataclass(frozen=True)
class Document:
    case_id: str
    analysis_split: str
    origin_country: str
    origin_state: str
    primary_domain: str
    case_domain: str
    decision_date: str
    court_name: str
    main_opinion_type: str
    text: str


@dataclass(frozen=True)
class Token:
    normalized: str
    start: int
    end: int


@dataclass
class CandidateStats:
    n: int
    display: str
    counts: Counter = field(default_factory=Counter)
    documents: dict[str, set[str]] = field(default_factory=lambda: {"KR": set(), "US": set()})
    domains: dict[str, Counter] = field(default_factory=lambda: {"KR": Counter(), "US": Counter()})
    states: Counter = field(default_factory=Counter)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash(manifest: dict, corpus_path: Path) -> str:
    target = corpus_path.name
    matches = [value for key, value in manifest.get("artifact_sha256", {}).items() if Path(key).name == target]
    if len(matches) != 1:
        raise AssertionError(f"Freeze manifest must contain exactly one SHA-256 for {target}")
    return matches[0].lower()


def load_and_verify_corpus(corpus_path: Path, manifest_path: Path) -> tuple[list[Document], list[dict], dict, str]:
    """Verify the frozen roster before exposing development text to analysis."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("status") == "FROZEN", "Corpus status is not FROZEN"
    counts = manifest.get("counts", {})
    assert counts.get("total") == 200, "Freeze manifest total must be 200"
    assert counts.get("by_split") == {"confirmatory": 160, "development": 40}, "Unexpected manifest split counts"
    expected_country = {
        "KR_development": 20, "US_development": 20,
        "KR_confirmatory": 80, "US_confirmatory": 80,
    }
    assert counts.get("by_country_split") == expected_country, "Unexpected manifest country/split counts"

    actual_hash = sha256_file(corpus_path)
    assert actual_hash == _manifest_hash(manifest, corpus_path), "Frozen final_cases SHA-256 mismatch"

    development: list[Document] = []
    inventory: list[dict] = []
    observed = Counter()
    confirmatory_ids: set[str] = set()
    all_case_ids: set[str] = set()
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            split = record.get("analysis_split")
            country = record.get("origin_country")
            case_id = str(record.get("case_id", ""))
            assert split in {"development", "confirmatory"}, f"Invalid split on line {line_number}"
            assert country in {"KR", "US"}, f"Invalid country on line {line_number}"
            assert case_id, f"Missing case_id on line {line_number}"
            assert case_id not in all_case_ids, f"Duplicate case_id: {case_id}"
            all_case_ids.add(case_id)
            observed[(country, split)] += 1

            # This branch is the hard text-access boundary. Confirmatory records contribute
            # only roster metadata; main_opinion_text is not indexed or copied here.
            if split == "confirmatory":
                confirmatory_ids.add(case_id)
                continue
            assert split == "development"
            text = record.get("main_opinion_text")
            assert isinstance(text, str) and text.strip(), f"Empty development main_opinion_text: {case_id}"
            metadata = {key: record.get(key) for key in METADATA_FIELDS}
            development.append(Document(
                case_id=case_id,
                analysis_split=split,
                origin_country=country,
                origin_state=str(record.get("origin_state") or ""),
                primary_domain=str(record.get("primary_domain") or ""),
                case_domain=str(record.get("case_domain") or ""),
                decision_date=str(record.get("decision_date") or ""),
                court_name=str(record.get("court_name") or ""),
                main_opinion_type=str(record.get("main_opinion_type") or ""),
                text=text,
            ))
            inventory.append(metadata)

    assert sum(observed.values()) == 200, f"Actual corpus total is {sum(observed.values())}, expected 200"
    assert observed == Counter({
        ("KR", "development"): 20, ("US", "development"): 20,
        ("KR", "confirmatory"): 80, ("US", "confirmatory"): 80,
    }), f"Actual country/split counts differ: {observed}"
    assert len(development) == 40
    assert Counter(doc.origin_country for doc in development) == Counter({"KR": 20, "US": 20})
    assert not ({doc.case_id for doc in development} & confirmatory_ids)
    development.sort(key=lambda doc: doc.case_id)
    inventory.sort(key=lambda row: str(row["case_id"]))
    return development, inventory, manifest, actual_hash


def normalize_space(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalize_token(text: str, language: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("’", "'").replace("‘", "'")
    for char in "‐‑‒–—―":
        value = value.replace(char, "-")
    return value.lower() if language == "en" else value


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _strip_surrounding_punctuation(text: str, absolute_start: int) -> tuple[str, int, int]:
    left, right = 0, len(text)
    while left < right and unicodedata.category(text[left])[0] in {"P", "S", "Z", "C"}:
        left += 1
    while right > left and unicodedata.category(text[right - 1])[0] in {"P", "S", "Z", "C"}:
        right -= 1
    return text[left:right], absolute_start + left, absolute_start + right


EN_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-‐‑‒–—―'’‘][^\W\d_]+)*", re.UNICODE)
NONSPACE_RE = re.compile(r"\S+", re.UNICODE)


def tokenize(text: str, country: str) -> list[Token]:
    """Return normalized analysis tokens with exact offsets into unchanged source text."""
    language = "ko" if country == "KR" else "en"
    tokens: list[Token] = []
    if language == "en":
        for match in EN_TOKEN_RE.finditer(text):
            normalized = normalize_token(match.group(), language)
            if normalized and any(_is_letter(c) for c in normalized):
                tokens.append(Token(normalized, match.start(), match.end()))
    else:
        for match in NONSPACE_RE.finditer(text):
            surface, start, end = _strip_surrounding_punctuation(match.group(), match.start())
            normalized = normalize_token(surface, language)
            if normalized and len(normalized) <= 200 and any(_is_letter(c) for c in normalized):
                tokens.append(Token(normalized, start, end))
    return tokens


def iter_ngrams(tokens: Sequence[Token]) -> Iterator[tuple[str, int, int, int]]:
    for index in range(len(tokens)):
        for n in NGRAM_RANGE:
            stop = index + n
            if stop > len(tokens):
                break
            selected = tokens[index:stop]
            yield " ".join(token.normalized for token in selected), n, selected[0].start, selected[-1].end


def infer_language(ngram: str) -> str:
    for char in ngram:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
            return "ko"
        if 0x3400 <= code <= 0x9FFF:
            return "ko"
    return "en"


def collect_statistics(documents: Sequence[Document]) -> tuple[dict[tuple[int, str], CandidateStats], dict[str, int]]:
    assert all(doc.analysis_split == "development" for doc in documents), "Non-development document reached lexical analysis"
    assert len(documents) == 40 and Counter(doc.origin_country for doc in documents) == Counter({"KR": 20, "US": 20})
    candidates: dict[tuple[int, str], CandidateStats] = {}
    token_counts: dict[str, int] = {}
    for doc in documents:
        tokens = tokenize(doc.text, doc.origin_country)
        token_counts[doc.case_id] = len(tokens)
        local: Counter = Counter()
        first_span: dict[tuple[int, str], tuple[int, int]] = {}
        for normalized, n, start, end in iter_ngrams(tokens):
            key = (n, normalized)
            local[key] += 1
            first_span.setdefault(key, (start, end))
        for key, count in local.items():
            n, normalized = key
            if key not in candidates:
                start, end = first_span[key]
                candidates[key] = CandidateStats(n=n, display=doc.text[start:end])
            stats = candidates[key]
            stats.counts[doc.origin_country] += count
            stats.documents[doc.origin_country].add(doc.case_id)
            stats.domains[doc.origin_country][doc.primary_domain] += 1
            if doc.origin_country == "US":
                stats.states[doc.origin_state] += 1
    return candidates, token_counts


def weighted_log_odds(
    count_a: int, count_b: int, total_a: int, total_b: int,
    pooled_feature_count: int, pooled_total: int, prior_strength: float = PRIOR_STRENGTH,
) -> tuple[float, float]:
    """Monroe-style weighted log odds with a pooled informative Dirichlet prior."""
    assert pooled_total > 0 and pooled_feature_count > 0
    alpha = prior_strength * pooled_feature_count / pooled_total
    # A degenerate one-feature synthetic vocabulary otherwise assigns no mass to
    # the complement. Real runs have large vocabularies, but keep the primitive total.
    epsilon = 1e-12
    alpha = min(max(alpha, epsilon), prior_strength - epsilon)
    other_alpha = prior_strength - alpha
    a_other = total_a - count_a
    b_other = total_b - count_b
    left = (count_a + alpha) / (a_other + other_alpha)
    right = (count_b + alpha) / (b_other + other_alpha)
    delta = math.log(left) - math.log(right)
    variance = 1.0 / (count_a + alpha) + 1.0 / (count_b + alpha)
    z_score = delta / math.sqrt(variance)
    assert math.isfinite(delta) and math.isfinite(z_score)
    return delta, z_score


def state_vs_rest(stats: CandidateStats, us_documents: Sequence[Document]) -> tuple[float | None, str]:
    state_sizes = Counter(doc.origin_state for doc in us_documents)
    best: tuple[float, str] | None = None
    total_df = len(stats.documents["US"])
    for state in sorted(state_sizes):
        in_total = state_sizes[state]
        out_total = len(us_documents) - in_total
        if in_total < 2 or out_total < 2:
            continue
        present_in = stats.states[state]
        present_out = total_df - present_in
        # Haldane-Anscombe 0.5 correction on the binary document 2x2 table.
        cells = (present_in + 0.5, in_total - present_in + 0.5,
                 present_out + 0.5, out_total - present_out + 0.5)
        log_odds = math.log(cells[0] / cells[1]) - math.log(cells[2] / cells[3])
        z_score = log_odds / math.sqrt(sum(1.0 / cell for cell in cells))
        if best is None or z_score > best[0] or (z_score == best[0] and state < best[1]):
            best = (z_score, state)
    return (best if best is not None else (None, ""))


def noise_flags(normalized: str, n: int, language: str) -> list[str]:
    words = normalized.split()
    stopwords = KO_STOPWORDS if language == "ko" else EN_STOPWORDS
    flags: list[str] = []
    if n == 1 and normalized in stopwords:
        flags.append("stopword_unigram")
    elif words and all(word in stopwords for word in words):
        flags.append("all_stopwords")
    compact = normalized.replace(" ", "")
    if re.fullmatch(r"(?:https?|www|com|org|gov|html?|pdf)+", compact, re.I):
        flags.append("url_fragment")
    if re.fullmatch(r"(?:no|nos|number|case|docket|page|pp|vol|volume)", normalized, re.I):
        flags.append("metadata_or_citation")
    if re.fullmatch(r"(?:court|opinion|justice|judge|chapter|section|appendix)", normalized, re.I):
        flags.append("header_boilerplate")
    if any(len(word) > 100 for word in words):
        flags.append("oversized_ocr_token")
    if not any(_is_letter(char) for char in normalized):
        flags.append("no_letters")
    return sorted(set(flags))


def load_seed_crosswalk(seed_dir: Path = SEED_DIR) -> tuple[dict[tuple[str, str], tuple[list[str], list[str]]], list[str]]:
    path = seed_dir / "surface_forms.csv"
    if not path.exists():
        return {}, []
    gathered: dict[tuple[str, str], tuple[set[str], set[str]]] = defaultdict(lambda: (set(), set()))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            language = row.get("language", "")
            surface = normalize_space(row.get("surface_form", ""))
            if language == "en":
                surface = surface.lower()
            if not surface:
                continue
            surface_ids, concept_ids = gathered[(language, surface)]
            surface_ids.add(row.get("surface_id", ""))
            concept_ids.add(row.get("concept_id", ""))
    crosswalk = {
        key: (sorted(filter(None, values[0])), sorted(filter(None, values[1])))
        for key, values in gathered.items()
    }
    used = [str(path.relative_to(REPO_ROOT))]
    concepts = seed_dir / "concepts.csv"
    if concepts.exists():
        used.append(str(concepts.relative_to(REPO_ROOT)))
    return crosswalk, used


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _candidate_id(language: str, n: int, normalized: str) -> str:
    raw = f"{language}\0{n}\0{normalized}".encode("utf-8")
    return "DEVNG_" + hashlib.sha256(raw).hexdigest()[:16]


def build_rows(candidates: dict[tuple[int, str], CandidateStats], documents: Sequence[Document], seed_crosswalk: dict) -> list[dict]:
    totals_occ = {country: Counter() for country in ("KR", "US")}
    totals_df = {country: Counter() for country in ("KR", "US")}
    pooled_occ, pooled_df = Counter(), Counter()
    for (n, _), stats in candidates.items():
        for country in ("KR", "US"):
            count = stats.counts[country]
            df = len(stats.documents[country])
            totals_occ[country][n] += count
            totals_df[country][n] += df
            pooled_occ[n] += count
            pooled_df[n] += df

    us_documents = [doc for doc in documents if doc.origin_country == "US"]
    rows: list[dict] = []
    for (n, normalized), stats in candidates.items():
        language = infer_language(normalized)
        kr_count, us_count = stats.counts["KR"], stats.counts["US"]
        kr_df, us_df = len(stats.documents["KR"]), len(stats.documents["US"])
        occurrence, z_occurrence = weighted_log_odds(
            kr_count, us_count, totals_occ["KR"][n], totals_occ["US"][n],
            kr_count + us_count, pooled_occ[n],
        )
        presence, z_presence = weighted_log_odds(
            kr_df, us_df, totals_df["KR"][n], totals_df["US"][n],
            kr_df + us_df, pooled_df[n],
        )
        direction = "KR" if z_presence > 1e-12 else "US" if z_presence < -1e-12 else "balanced"
        all_domains = stats.domains["KR"] + stats.domains["US"]
        max_domain_df = max(all_domains.values(), default=0)
        total_df = kr_df + us_df
        max_domain_share = max_domain_df / total_df if total_df else 0.0
        max_state_df = max(stats.states.values(), default=0)
        max_state_share = max_state_df / us_df if us_df else 0.0
        state_z, state_name = state_vs_rest(stats, us_documents) if us_df else (None, "")
        flags = noise_flags(normalized, n, language)
        ranking_score = (
            10.0 * abs(z_presence) + 2.0 * math.log2(1 + total_df)
            + (1.0 if total_df >= 2 else 0.0) + 0.15 * min(n, 3)
        )
        seed_surface_ids, seed_concept_ids = seed_crosswalk.get((language, normalized), ([], []))
        row = {
            "candidate_id": _candidate_id(language, n, normalized),
            "ngram": stats.display,
            "normalized_ngram": normalized,
            "language": language,
            "n": n,
            "corpus_direction": direction,
            "kr_count": kr_count, "us_count": us_count,
            "kr_df": kr_df, "us_df": us_df,
            "kr_df_prop": kr_df / 20, "us_df_prop": us_df / 20,
            "log_odds_occurrence": occurrence, "z_log_odds_occurrence": z_occurrence,
            "log_odds_document_presence": presence, "z_log_odds_document_presence": z_presence,
            "meets_df2_threshold": kr_df >= 2 or us_df >= 2,
            "kr_domain_count": len(stats.domains["KR"]),
            "us_domain_count": len(stats.domains["US"]),
            "kr_domains": _json_list(stats.domains["KR"].keys()),
            "us_domains": _json_list(stats.domains["US"].keys()),
            "max_domain_df": max_domain_df,
            "max_domain_df_share": max_domain_share,
            "domain_concentrated": total_df >= 2 and max_domain_share >= DOMAIN_CONCENTRATION_THRESHOLD,
            "us_state_count": len(stats.states), "us_states": _json_list(stats.states.keys()),
            "max_state_df": max_state_df, "max_state_df_share": max_state_share,
            "state_concentrated": us_df >= 2 and max_state_share >= STATE_CONCENTRATION_THRESHOLD,
            "max_state_vs_rest_z": state_z,
            "max_state_vs_rest_state": state_name,
            "ranking_score": ranking_score,
            "noise_flags": "|".join(flags),
            "example_case_ids": "", "example_contexts_short": "",
            "candidate_source": "development_corpus_ngram",
            "already_in_seed": bool(seed_surface_ids),
            "seed_surface_ids": "|".join(seed_surface_ids),
            "seed_concept_ids": "|".join(seed_concept_ids),
            **{column: "" for column in HUMAN_COLUMNS},
        }
        rows.append(row)
    return rows


def ranked_rows(rows: Sequence[dict]) -> list[dict]:
    selected = [row for row in rows if row["meets_df2_threshold"] and not row["noise_flags"]]
    return sorted(selected, key=lambda row: (
        -row["ranking_score"], -(row["kr_df"] + row["us_df"]), -row["n"],
        row["language"], row["normalized_ngram"], row["candidate_id"],
    ))


def collect_contexts(documents: Sequence[Document], wanted_rows: Sequence[dict]) -> list[dict]:
    assert all(doc.analysis_split == "development" for doc in documents), "Confirmatory document reached context extraction"
    wanted = {(int(row["n"]), row["normalized_ngram"]): row for row in wanted_rows}
    by_key: dict[tuple[int, str], list[dict]] = defaultdict(list)
    seen_documents: dict[tuple[int, str], set[str]] = defaultdict(set)
    for doc in documents:
        tokens = tokenize(doc.text, doc.origin_country)
        for normalized, n, start, end in iter_ngrams(tokens):
            key = (n, normalized)
            if key not in wanted or doc.case_id in seen_documents[key] or len(by_key[key]) >= CONTEXT_LIMIT:
                continue
            matched = doc.text[start:end]
            assert matched == doc.text[start:end]
            context = {
                "ngram": wanted[key]["ngram"],
                "normalized_ngram": normalized,
                "language": "ko" if doc.origin_country == "KR" else "en",
                "case_id": doc.case_id,
                "origin_country": doc.origin_country,
                "origin_state": doc.origin_state,
                "primary_domain": doc.primary_domain,
                "start": start, "end": end, "matched_text": matched,
                "context_before": doc.text[max(0, start - 160):start],
                "context_after": doc.text[end:end + 160],
            }
            by_key[key].append(context)
            seen_documents[key].add(doc.case_id)
    contexts: list[dict] = []
    for row in wanted_rows:
        key = (int(row["n"]), row["normalized_ngram"])
        examples = by_key.get(key, [])
        assert examples, f"No context found for ranked candidate {row['candidate_id']}"
        row["example_case_ids"] = "|".join(item["case_id"] for item in examples)
        row["example_contexts_short"] = " || ".join(
            normalize_space(item["context_before"][-60:] + item["matched_text"] + item["context_after"][:60])
            for item in examples[:3]
        )
        contexts.extend(examples)
    return contexts


def _csv_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else format(value, ".12g")
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def inventory_rows(inventory: Sequence[dict], documents: Sequence[Document], token_counts: dict[str, int]) -> list[dict]:
    text_hashes = {doc.case_id: hashlib.sha256(doc.text.encode("utf-8")).hexdigest() for doc in documents}
    result = []
    for row in inventory:
        case_id = str(row["case_id"])
        item = dict(row)
        item["main_opinion_chars"] = len(next(doc.text for doc in documents if doc.case_id == case_id))
        item["main_opinion_sha256"] = text_hashes[case_id]
        item["analysis_token_count"] = token_counts[case_id]
        result.append(item)
    return result


def readme_text() -> str:
    return """# Development-corpus legal lexicon candidates v0.1

This directory contains deterministic **candidate evidence**, not a validated or frozen lexicon. Only
`analysis_split == development` opinion text was tokenized. Confirmatory opinion text was excluded at a
hard branch before the `main_opinion_text` field was accessed. Neutral facts and fact-pattern/unit files
were not read.

## Method

- Source integrity and the frozen 200/40/160, KR 20/80, and US 20/80 roster are asserted before analysis.
- Text is normalized to Unicode NFKC in a separate analysis representation; exact source text and offsets
  are retained for contexts. English is lowercased and tokenized as alphabetic words with internal
  hyphens/apostrophes. Korean uses deterministic whitespace/eojeol tokens with surrounding punctuation
  removed. There is no stemming, lemmatization, learned model, LLM, translation, or randomness.
- Contiguous 1--5 grams are counted. Raw output retains every extracted candidate with DF >= 1. Ranked
  review output uses DF >= 2 and removes only candidates carrying conservative noise flags. Function words
  inside multiword phrases remain unless the entire phrase is stopwords.
- `meets_df2_threshold` means country-specific document frequency is >= 2 in KR or US.
- Domain/state concentration means a maximum document-frequency share >= 0.80 and total relevant DF >= 2.

## Log-odds

For each n separately, pooled informative-prior mass is `alpha_w = 100 * pooled_w / pooled_total` and
`alpha_other = 100 - alpha_w`. At a degenerate numerical boundary either mass is clamped to `1e-12`.
For countries A and B:

`delta = log((c_A + alpha_w)/(N_A-c_A + alpha_other)) - log((c_B + alpha_w)/(N_B-c_B + alpha_other))`

`z = delta / sqrt(1/(c_A+alpha_w) + 1/(c_B+alpha_w))`

The occurrence version uses raw n-gram occurrences. The primary document-presence version replaces each
document's term count by one; its country totals are sums of document-presence events for that n. The
state-vs-rest diagnostic instead uses binary document 2x2 tables with a 0.5 Haldane-Anscombe correction,
only where both state and rest contain at least two development documents.

**High KR-vs-US n-gram log-odds does not establish that a phrase is jurisdiction-specific. It is used only
to prioritize candidate expressions for subsequent legal and bilingual human review.** Language,
jurisdiction, opinion length, domain composition, and the small development sample are confounded. No
candidate is automatically called KR-specific, US-generic, state-specific, shared in a legal sense, or
assigned to doctrine/remedy/procedure/party_arg. `corpus_direction` is statistical direction only.

Human-review columns are blank. Seed v0.1.1 is used solely for exact NFKC/whitespace-normalized surface
cross-reference and is never modified. `SHA256SUMS.txt` covers every generated artifact except itself.
"""


def write_outputs(
    output_dir: Path, documents: Sequence[Document], inventory: Sequence[dict], token_counts: dict[str, int],
    all_rows: list[dict], ranked: list[dict], contexts: Sequence[dict], manifest: dict,
    source_hash: str, source_manifest_path: Path, corpus_path: Path, seed_files: Sequence[str],
    generation_timestamp: str,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_sorted = sorted(all_rows, key=lambda row: (row["language"], row["n"], row["normalized_ngram"], row["candidate_id"]))
    ranked_ids = {row["candidate_id"] for row in ranked}
    # Context fields belong only to ranked/human-review views, leaving the archival table compact.
    archival = [{**row, "example_case_ids": "", "example_contexts_short": ""} for row in all_sorted]
    write_csv(output_dir / "dev_corpus_inventory.csv", inventory_rows(inventory, documents, token_counts),
              list(METADATA_FIELDS) + ["main_opinion_chars", "main_opinion_sha256", "analysis_token_count"])
    write_csv(output_dir / "ngram_candidates_all.csv", archival, ALL_COLUMNS)
    write_csv(output_dir / "ngram_candidates_ranked.csv", ranked, ALL_COLUMNS)
    write_csv(output_dir / "ngram_candidates_kr.csv", [r for r in ranked if r["corpus_direction"] == "KR"], ALL_COLUMNS)
    write_csv(output_dir / "ngram_candidates_us.csv", [r for r in ranked if r["corpus_direction"] == "US"], ALL_COLUMNS)
    write_csv(output_dir / "ngram_candidates_shared.csv", [r for r in ranked if r["kr_df"] and r["us_df"]], ALL_COLUMNS)
    write_csv(output_dir / "ngram_candidates_state_concentrated.csv", [r for r in ranked if r["state_concentrated"]], ALL_COLUMNS)
    write_csv(output_dir / "ngram_candidates_domain_concentrated.csv", [r for r in ranked if r["domain_concentrated"]], ALL_COLUMNS)
    write_csv(output_dir / "ngram_candidates_for_human_review.csv", ranked, ALL_COLUMNS)
    with (output_dir / "candidate_contexts.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for context in contexts:
            handle.write(json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "README.md").write_text(readme_text(), encoding="utf-8", newline="\n")

    confirmatory_count = manifest["counts"]["by_split"]["confirmatory"]
    run_manifest = {
        "source_corpus_version": manifest["corpus_version"],
        "source_final_cases_path": str(corpus_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_final_cases_sha256": source_hash,
        "source_freeze_manifest_path": str(source_manifest_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_corpus_verified": True,
        "development_case_ids": [doc.case_id for doc in documents],
        "development_count": len(documents),
        "kr_development_count": sum(doc.origin_country == "KR" for doc in documents),
        "us_development_count": sum(doc.origin_country == "US" for doc in documents),
        "excluded_confirmatory_count": confirmatory_count,
        "confirmatory_documents_used_for_lexical_analysis": 0,
        "text_field": "main_opinion_text",
        "forbidden_inputs_not_read": ["neutral_fact_ko", "neutral_fact_en", "neutral_fact_source",
                                      "final_fact_patterns_200_v4_0_2.jsonl", "final_fact_units_200_v4_0_2.jsonl"],
        "tokenization_rules": {
            "normalization": "Unicode NFKC plus repeated-whitespace normalization for normalized surfaces",
            "english": "lowercase alphabetic tokens; meaningful internal hyphens/apostrophes retained; no stemming/lemmatization/model",
            "korean": "deterministic whitespace/eojeol tokens; surrounding Unicode punctuation/symbols stripped; no particle stripping/model",
            "context_offsets": "zero-based half-open offsets into unchanged source main_opinion_text",
        },
        "ngram_range": list(NGRAM_RANGE),
        "raw_df_threshold": 1, "ranked_df_threshold": 2,
        "stopword_lists_version": STOPWORD_VERSION,
        "stopword_lists": {"en": sorted(EN_STOPWORDS), "ko": sorted(KO_STOPWORDS)},
        "log_odds_formula": "weighted informative-Dirichlet log odds; see README.md",
        "dirichlet_prior_strength": PRIOR_STRENGTH,
        "state_vs_rest_smoothing": 0.5,
        "domain_concentration_threshold": DOMAIN_CONCENTRATION_THRESHOLD,
        "state_concentration_threshold": STATE_CONCENTRATION_THRESHOLD,
        "ranking_formula": "10*abs(z_document_presence)+2*log2(1+combined_df)+I(df>=2)+0.15*min(n,3)",
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed_cross_reference_files": list(seed_files),
        "generation_timestamp_utc": generation_timestamp,
        "random_seed": None,
        "randomness_used": False,
        "counts": {
            "total_raw_ngrams": len(all_rows), "ranked_candidates": len(ranked),
            "kr_direction_candidates": sum(r["corpus_direction"] == "KR" for r in ranked),
            "us_direction_candidates": sum(r["corpus_direction"] == "US" for r in ranked),
            "shared_candidates": sum(bool(r["kr_df"] and r["us_df"]) for r in ranked),
            "state_concentrated_candidates": sum(bool(r["state_concentrated"]) for r in ranked),
            "domain_concentrated_candidates": sum(bool(r["domain_concentrated"]) for r in ranked),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    generated = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    sums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in generated)
    (output_dir / "SHA256SUMS.txt").write_text(sums, encoding="ascii", newline="\n")
    assert ranked_ids == {row["candidate_id"] for row in ranked}
    return run_manifest["counts"]


def run(corpus_path: Path, manifest_path: Path, output_dir: Path, generation_timestamp: str | None = None) -> dict[str, int]:
    documents, inventory, manifest, source_hash = load_and_verify_corpus(corpus_path, manifest_path)
    candidates, token_counts = collect_statistics(documents)
    seed_crosswalk, seed_files = load_seed_crosswalk()
    all_rows = build_rows(candidates, documents, seed_crosswalk)
    ranked = ranked_rows(all_rows)
    contexts = collect_contexts(documents, ranked)
    # Contexts can only originate in the already-asserted development Document collection.
    development_ids = {doc.case_id for doc in documents}
    assert all(item["case_id"] in development_ids for item in contexts)
    timestamp = generation_timestamp or datetime.now(timezone.utc).isoformat()
    return write_outputs(output_dir, documents, inventory, token_counts, all_rows, ranked, contexts,
                         manifest, source_hash, manifest_path, corpus_path, seed_files, timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-timestamp", help="Fixed ISO timestamp for fully byte-identical manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = run(args.corpus.resolve(), args.freeze_manifest.resolve(), args.output_dir.resolve(), args.generation_timestamp)
    print("source corpus verified: yes")
    print("development documents: 40")
    print("KR development: 20")
    print("US development: 20")
    print("confirmatory documents used for lexical analysis: 0")
    print()
    print(f"total raw n-grams: {counts['total_raw_ngrams']}")
    print(f"ranked candidates: {counts['ranked_candidates']}")
    print(f"KR-direction candidates: {counts['kr_direction_candidates']}")
    print(f"US-direction candidates: {counts['us_direction_candidates']}")
    print(f"shared candidates: {counts['shared_candidates']}")
    print(f"state-concentrated candidates: {counts['state_concentrated_candidates']}")
    print(f"domain-concentrated candidates: {counts['domain_concentrated_candidates']}")
    print()
    print(f"output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
