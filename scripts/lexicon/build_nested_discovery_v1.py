#!/usr/bin/env python3
"""Create a held-out nested development split and discovery-only lexicon evidence.

The split is assigned from metadata only. Opinion text is subsequently accessed only
for the 20 discovery cases. Tuning, validation, and confirmatory text remain untouched.
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
from pathlib import Path
from typing import Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
FREEZE_MANIFEST_PATH = REPO_ROOT / "outputs_v2/v4.0.2/corpus_freeze_manifest_v4_0_2.json"
SEED_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_seed_v0_1_1"
OUTPUT_DIR = REPO_ROOT / "outputs_v2/lexicon_nested_split_v1"
SPLIT_SALT = "lexicon_nested_split_v1"
SPLITS = ("discovery", "tuning", "validation")
COUNTRY_TARGETS = {"discovery": 10, "tuning": 5, "validation": 5}
NGRAMS = (1, 2, 3, 4, 5)
PRIOR_STRENGTH = 100.0
CONCENTRATION_THRESHOLD = 0.80
CONTEXT_LIMIT = 5
STOPWORD_VERSION = "nested-discovery-stopwords-v1"

EN_STOPWORDS = frozenset(
    "a an and are as at be been being but by did do does for from had has have he her "
    "hers him his i if in into is it its may nor not of on or our she so than that the "
    "their them there these they this those to under upon was we were which who whom "
    "will with would you your v vs".split()
)
KO_STOPWORDS = frozenset(
    "그 및 이 저 것 수 등 또는 그리고 그러나 또한 대한 대하여 관한 관하여 의 에 에서 로 으로 와 과 은 는 이 가 을 를 도 만 한 하여 있는 있다 없는 없다 한다 하였다 경우 때 바".split()
)

SPLIT_COLUMNS = [
    "case_id", "analysis_split", "nested_split", "origin_country", "origin_state",
    "primary_domain", "case_domain", "decision_date", "court_name", "main_opinion_type",
    "tie_break_sha256",
]

ALL_CANDIDATE_COLUMNS = [
    "candidate_id", "surface_form", "normalized_surface_form", "language", "n",
    "corpus_direction", "kr_count", "us_count", "kr_df", "us_df", "kr_df_prop",
    "us_df_prop", "discovery_df", "log_odds_occurrence", "z_log_odds_occurrence",
    "log_odds_document_presence", "z_log_odds_document_presence", "discovery_log_odds",
    "meets_review_rule", "candidate_stability", "candidate_stability_runs",
    "kr_domain_count", "us_domain_count", "kr_domains", "us_domains",
    "max_domain_df", "max_domain_df_share", "domain_concentration",
    "us_state_count", "us_states", "max_state_df", "max_state_df_share",
    "state_concentration", "max_state_vs_rest_z", "max_state_vs_rest_state",
    "ranking_score", "noise_flags", "already_in_official_seed",
    "official_seed_surface_ids", "official_seed_concept_id", "source_provenance",
    "example_case_ids", "example_contexts",
]

REVIEW_COLUMNS = [
    "candidate_id", "surface_form", "language", "discovery_df", "discovery_log_odds",
    "candidate_stability", "candidate_stability_runs", "state_concentration",
    "domain_concentration", "already_in_official_seed", "official_seed_concept_id",
    "example_contexts", "human_keep", "human_category", "human_concept_family",
    "human_concept_id", "human_jurisdiction_scope", "human_notes",
]


@dataclass(frozen=True)
class CaseMetadata:
    case_id: str
    analysis_split: str
    origin_country: str
    origin_state: str
    primary_domain: str
    case_domain: str
    decision_date: str
    court_name: str
    main_opinion_type: str


@dataclass(frozen=True)
class DiscoveryDocument:
    metadata: CaseMetadata
    text: str


@dataclass(frozen=True)
class Token:
    value: str
    start: int
    end: int


@dataclass
class CandidateStats:
    n: int
    surface_form: str
    counts: Counter = field(default_factory=Counter)
    documents: dict[str, set[str]] = field(default_factory=lambda: {"KR": set(), "US": set()})
    domains: dict[str, Counter] = field(default_factory=lambda: {"KR": Counter(), "US": Counter()})
    states: Counter = field(default_factory=Counter)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_space(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _manifest_corpus_hash(manifest: dict, corpus_path: Path) -> str:
    matches = [value for key, value in manifest["artifact_sha256"].items() if Path(key).name == corpus_path.name]
    assert len(matches) == 1, f"Expected one manifest hash for {corpus_path.name}"
    return matches[0].lower()


def read_verified_metadata(corpus_path: Path, manifest_path: Path) -> tuple[list[CaseMetadata], set[str], dict, str]:
    """Verify the corpus and read roster metadata without indexing any opinion text."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("status") == "FROZEN"
    assert manifest.get("counts", {}).get("total") == 200
    assert manifest["counts"]["by_split"] == {"confirmatory": 160, "development": 40}
    assert manifest["counts"]["by_country_split"] == {
        "KR_development": 20, "KR_confirmatory": 80,
        "US_development": 20, "US_confirmatory": 80,
    }
    source_hash = sha256_file(corpus_path)
    assert source_hash == _manifest_corpus_hash(manifest, corpus_path), "Source corpus SHA256 mismatch"

    development: list[CaseMetadata] = []
    confirmatory_ids: set[str] = set()
    all_ids: set[str] = set()
    counts = Counter()
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            case_id = str(record.get("case_id", ""))
            country = record.get("origin_country")
            split = record.get("analysis_split")
            assert case_id and case_id not in all_ids, f"Invalid/duplicate case_id at line {line_number}"
            assert country in {"KR", "US"} and split in {"development", "confirmatory"}
            all_ids.add(case_id)
            counts[(country, split)] += 1
            if split == "confirmatory":
                confirmatory_ids.add(case_id)
                continue
            # Metadata only: main_opinion_text is deliberately not accessed here.
            development.append(CaseMetadata(
                case_id=case_id, analysis_split=split, origin_country=country,
                origin_state=str(record.get("origin_state") or ""),
                primary_domain=str(record.get("primary_domain") or ""),
                case_domain=str(record.get("case_domain") or ""),
                decision_date=str(record.get("decision_date") or ""),
                court_name=str(record.get("court_name") or ""),
                main_opinion_type=str(record.get("main_opinion_type") or ""),
            ))
    assert counts == Counter({
        ("KR", "development"): 20, ("US", "development"): 20,
        ("KR", "confirmatory"): 80, ("US", "confirmatory"): 80,
    })
    assert len(development) == 40 and len(confirmatory_ids) == 160
    development.sort(key=lambda item: item.case_id)
    return development, confirmatory_ids, manifest, source_hash


def _allocation_objective(
    assigned: dict[str, list[CaseMetadata]], country_cases: Sequence[CaseMetadata], country: str,
) -> float:
    total = len(country_cases)
    domain_totals = Counter(item.primary_domain for item in country_cases)
    state_totals = Counter(item.origin_state for item in country_cases) if country == "US" else Counter()
    score = 0.0
    for split in SPLITS:
        ratio = COUNTRY_TARGETS[split] / total
        domain_counts = Counter(item.primary_domain for item in assigned[split])
        for domain, count in domain_totals.items():
            target = count * ratio
            score += (domain_counts[domain] - target) ** 2 / (target + 0.5)
        if country == "US":
            state_counts = Counter(item.origin_state for item in assigned[split])
            for state, count in state_totals.items():
                target = count * ratio
                score += 0.75 * (state_counts[state] - target) ** 2 / (target + 0.5)
    return score


def assign_nested_split(development: Sequence[CaseMetadata]) -> dict[str, str]:
    """Metadata-only deterministic stratified greedy allocation."""
    assert len(development) == 40
    result: dict[str, str] = {}
    for country in ("KR", "US"):
        cases = [item for item in development if item.origin_country == country]
        assert len(cases) == 20
        domain_sizes = Counter(item.primary_domain for item in cases)
        state_sizes = Counter(item.origin_state for item in cases)
        ordered = sorted(cases, key=lambda item: (
            domain_sizes[item.primary_domain],
            state_sizes[item.origin_state] if country == "US" else 0,
            stable_hash(f"{item.case_id}|{SPLIT_SALT}"),
        ))
        assigned: dict[str, list[CaseMetadata]] = {split: [] for split in SPLITS}
        for item in ordered:
            choices = []
            for split in SPLITS:
                if len(assigned[split]) >= COUNTRY_TARGETS[split]:
                    continue
                assigned[split].append(item)
                objective = _allocation_objective(assigned, cases, country)
                assigned[split].pop()
                tie = stable_hash(f"{item.case_id}|{split}|{SPLIT_SALT}")
                choices.append((objective, tie, split))
            assert choices
            chosen = min(choices)[2]
            assigned[chosen].append(item)
            result[item.case_id] = chosen
        assert {split: len(assigned[split]) for split in SPLITS} == COUNTRY_TARGETS
    validate_nested_split(development, result)
    return result


def validate_nested_split(development: Sequence[CaseMetadata], split_map: dict[str, str]) -> None:
    development_ids = {item.case_id for item in development}
    assert set(split_map) == development_ids and len(split_map) == 40
    assert all(value in SPLITS for value in split_map.values())
    assert Counter(split_map.values()) == Counter({"discovery": 20, "tuning": 10, "validation": 10})
    countries = {item.case_id: item.origin_country for item in development}
    assert Counter((split_map[case_id], countries[case_id]) for case_id in split_map) == Counter({
        ("discovery", "KR"): 10, ("discovery", "US"): 10,
        ("tuning", "KR"): 5, ("tuning", "US"): 5,
        ("validation", "KR"): 5, ("validation", "US"): 5,
    })
    subsets = [{case_id for case_id, value in split_map.items() if value == split} for split in SPLITS]
    assert not (subsets[0] & subsets[1] or subsets[0] & subsets[2] or subsets[1] & subsets[2])


def split_rows(development: Sequence[CaseMetadata], split_map: dict[str, str]) -> list[dict]:
    rows = []
    for item in development:
        rows.append({
            **item.__dict__, "nested_split": split_map[item.case_id],
            "tie_break_sha256": stable_hash(f"{item.case_id}|{SPLIT_SALT}"),
        })
    order = {value: index for index, value in enumerate(SPLITS)}
    return sorted(rows, key=lambda row: (order[row["nested_split"]], row["origin_country"], row["case_id"]))


def distribution_rows(development: Sequence[CaseMetadata], split_map: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    dimensions = {
        "origin_country": lambda item: item.origin_country,
        "primary_domain": lambda item: item.primary_domain,
        "origin_state": lambda item: item.origin_state or "(none)",
    }
    for dimension, getter in dimensions.items():
        for split in SPLITS:
            selected = [item for item in development if split_map[item.case_id] == split]
            counts = Counter(getter(item) for item in selected)
            for value in sorted(counts):
                rows.append({
                    "dimension": dimension, "value": value, "nested_split": split,
                    "count": counts[value], "proportion_within_split": counts[value] / len(selected),
                })
    return rows


def load_discovery_documents(corpus_path: Path, split_map: dict[str, str]) -> list[DiscoveryDocument]:
    """Access main_opinion_text only after a case is proven to be discovery."""
    discovery_ids = {case_id for case_id, split in split_map.items() if split == "discovery"}
    assert len(discovery_ids) == 20
    documents: list[DiscoveryDocument] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case_id = str(record["case_id"])
            if case_id not in discovery_ids:
                continue
            assert record["analysis_split"] == "development"
            assert split_map[case_id] == "discovery"
            text = record.get("main_opinion_text")
            assert isinstance(text, str) and text.strip(), f"Empty discovery opinion: {case_id}"
            metadata = CaseMetadata(
                case_id=case_id, analysis_split="development", origin_country=record["origin_country"],
                origin_state=str(record.get("origin_state") or ""),
                primary_domain=str(record.get("primary_domain") or ""),
                case_domain=str(record.get("case_domain") or ""),
                decision_date=str(record.get("decision_date") or ""),
                court_name=str(record.get("court_name") or ""),
                main_opinion_type=str(record.get("main_opinion_type") or ""),
            )
            documents.append(DiscoveryDocument(metadata, text))
    documents.sort(key=lambda item: item.metadata.case_id)
    assert len(documents) == 20
    assert Counter(item.metadata.origin_country for item in documents) == Counter({"KR": 10, "US": 10})
    assert {item.metadata.case_id for item in documents} == discovery_ids
    return documents


EN_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-‐‑‒–—―'’‘][^\W\d_]+)*", re.UNICODE)
NONSPACE_RE = re.compile(r"\S+", re.UNICODE)


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _normalize_token(value: str, language: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("’", "'").replace("‘", "'")
    for char in "‐‑‒–—―":
        value = value.replace(char, "-")
    return value.lower() if language == "en" else value


def tokenize(text: str, country: str) -> list[Token]:
    result: list[Token] = []
    if country == "US":
        for match in EN_TOKEN_RE.finditer(text):
            value = _normalize_token(match.group(), "en")
            if value and any(_is_letter(char) for char in value):
                result.append(Token(value, match.start(), match.end()))
        return result
    for match in NONSPACE_RE.finditer(text):
        raw = match.group()
        left, right = 0, len(raw)
        while left < right and unicodedata.category(raw[left])[0] in {"P", "S", "Z", "C"}:
            left += 1
        while right > left and unicodedata.category(raw[right - 1])[0] in {"P", "S", "Z", "C"}:
            right -= 1
        value = _normalize_token(raw[left:right], "ko")
        if value and len(value) <= 200 and any(_is_letter(char) for char in value):
            result.append(Token(value, match.start() + left, match.start() + right))
    return result


def iter_ngrams(tokens: Sequence[Token]) -> Iterator[tuple[int, str, int, int]]:
    for start_index in range(len(tokens)):
        for n in NGRAMS:
            end_index = start_index + n
            if end_index > len(tokens):
                break
            selected = tokens[start_index:end_index]
            yield n, " ".join(token.value for token in selected), selected[0].start, selected[-1].end


def infer_language(value: str) -> str:
    for char in value:
        code = ord(char)
        if 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F or 0xAC00 <= code <= 0xD7AF:
            return "ko"
        if 0x3400 <= code <= 0x9FFF:
            return "ko"
    return "en"


def collect_candidate_stats(documents: Sequence[DiscoveryDocument]) -> dict[tuple[int, str], CandidateStats]:
    assert len(documents) == 20
    assert Counter(item.metadata.origin_country for item in documents) == Counter({"KR": 10, "US": 10})
    candidates: dict[tuple[int, str], CandidateStats] = {}
    for document in documents:
        metadata = document.metadata
        local = Counter()
        first_spans: dict[tuple[int, str], tuple[int, int]] = {}
        tokens = tokenize(document.text, metadata.origin_country)
        for n, normalized, start, end in iter_ngrams(tokens):
            key = (n, normalized)
            local[key] += 1
            first_spans.setdefault(key, (start, end))
        for key, count in local.items():
            if key not in candidates:
                start, end = first_spans[key]
                candidates[key] = CandidateStats(key[0], document.text[start:end])
            stats = candidates[key]
            country = metadata.origin_country
            stats.counts[country] += count
            stats.documents[country].add(metadata.case_id)
            stats.domains[country][metadata.primary_domain] += 1
            if country == "US":
                stats.states[metadata.origin_state] += 1
    return candidates


def weighted_log_odds(
    count_a: int, count_b: int, total_a: int, total_b: int,
    pooled_feature: int, pooled_total: int,
) -> tuple[float, float]:
    alpha = PRIOR_STRENGTH * pooled_feature / pooled_total
    alpha = min(max(alpha, 1e-12), PRIOR_STRENGTH - 1e-12)
    other_alpha = PRIOR_STRENGTH - alpha
    delta = math.log((count_a + alpha) / (total_a - count_a + other_alpha)) - math.log(
        (count_b + alpha) / (total_b - count_b + other_alpha)
    )
    variance = 1 / (count_a + alpha) + 1 / (count_b + alpha)
    z_score = delta / math.sqrt(variance)
    assert math.isfinite(delta) and math.isfinite(z_score)
    return delta, z_score


def noise_flags(value: str, n: int, language: str) -> list[str]:
    words = value.split()
    stopwords = KO_STOPWORDS if language == "ko" else EN_STOPWORDS
    flags = []
    if n == 1 and value in stopwords:
        flags.append("stopword_unigram")
    elif words and all(word in stopwords for word in words):
        flags.append("all_stopwords")
    compact = value.replace(" ", "")
    if re.fullmatch(r"(?:https?|www|com|org|gov|html?|pdf)+", compact, re.I):
        flags.append("url_fragment")
    if re.fullmatch(r"(?:no|nos|number|case|docket|page|pp|vol|volume)", value, re.I):
        flags.append("metadata_or_citation")
    if re.fullmatch(r"(?:court|opinion|justice|judge|chapter|section|appendix)", value, re.I):
        flags.append("header_boilerplate")
    if any(len(word) > 100 for word in words):
        flags.append("oversized_ocr_token")
    return sorted(set(flags))


def _state_vs_rest(stats: CandidateStats, documents: Sequence[DiscoveryDocument]) -> tuple[float | None, str]:
    us_documents = [item for item in documents if item.metadata.origin_country == "US"]
    state_sizes = Counter(item.metadata.origin_state for item in us_documents)
    total_df = len(stats.documents["US"])
    best: tuple[float, str] | None = None
    for state in sorted(state_sizes):
        in_total, out_total = state_sizes[state], len(us_documents) - state_sizes[state]
        if in_total < 2 or out_total < 2:
            continue
        present_in = stats.states[state]
        present_out = total_df - present_in
        cells = (present_in + 0.5, in_total - present_in + 0.5,
                 present_out + 0.5, out_total - present_out + 0.5)
        delta = math.log(cells[0] / cells[1]) - math.log(cells[2] / cells[3])
        z_score = delta / math.sqrt(sum(1 / value for value in cells))
        if best is None or (z_score, state) > best:
            best = (z_score, state)
    return best if best else (None, "")


def load_official_seed(seed_dir: Path = SEED_DIR) -> tuple[list[dict], dict[tuple[str, str], list[dict]], dict[str, str]]:
    combined_path = seed_dir / "seed_lexicon_combined.csv"
    surfaces_path = seed_dir / "surface_forms.csv"
    concepts_path = seed_dir / "concepts.csv"
    with combined_path.open(encoding="utf-8-sig", newline="") as handle:
        combined = list(csv.DictReader(handle))
    with surfaces_path.open(encoding="utf-8-sig", newline="") as handle:
        surfaces = list(csv.DictReader(handle))
    with concepts_path.open(encoding="utf-8-sig", newline="") as handle:
        concepts = {row["concept_id"]: row.get("category", "") for row in csv.DictReader(handle)}
    crosswalk: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in surfaces:
        language = row["language"]
        normalized = normalize_space(row["surface_form"])
        if language == "en":
            normalized = normalized.lower()
        crosswalk[(language, normalized)].append(row)
    return combined, crosswalk, concepts


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _candidate_id(language: str, n: int, value: str) -> str:
    return "DISC_" + stable_hash(f"{language}\0{n}\0{value}")[:16]


def build_candidate_rows(
    candidates: dict[tuple[int, str], CandidateStats], documents: Sequence[DiscoveryDocument],
    seed_crosswalk: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    occurrence_totals = {country: Counter() for country in ("KR", "US")}
    presence_totals = {country: Counter() for country in ("KR", "US")}
    pooled_occurrence, pooled_presence = Counter(), Counter()
    for (n, _), stats in candidates.items():
        for country in ("KR", "US"):
            occurrence_totals[country][n] += stats.counts[country]
            presence_totals[country][n] += len(stats.documents[country])
            pooled_occurrence[n] += stats.counts[country]
            pooled_presence[n] += len(stats.documents[country])

    rows = []
    for (n, value), stats in candidates.items():
        language = infer_language(value)
        kr_count, us_count = stats.counts["KR"], stats.counts["US"]
        kr_df, us_df = len(stats.documents["KR"]), len(stats.documents["US"])
        occurrence, z_occurrence = weighted_log_odds(
            kr_count, us_count, occurrence_totals["KR"][n], occurrence_totals["US"][n],
            kr_count + us_count, pooled_occurrence[n],
        )
        presence, z_presence = weighted_log_odds(
            kr_df, us_df, presence_totals["KR"][n], presence_totals["US"][n],
            kr_df + us_df, pooled_presence[n],
        )
        direction = "KR" if z_presence > 1e-12 else "US" if z_presence < -1e-12 else "balanced"
        flags = noise_flags(value, n, language)
        meets_rule = (kr_df >= 2 or us_df >= 2) and not flags
        satisfying_runs = 0
        for document in documents:
            case_id = document.metadata.case_id
            loo_kr = kr_df - int(case_id in stats.documents["KR"])
            loo_us = us_df - int(case_id in stats.documents["US"])
            satisfying_runs += int((loo_kr >= 2 or loo_us >= 2) and not flags)
        stability = satisfying_runs / len(documents)
        domains = stats.domains["KR"] + stats.domains["US"]
        total_df = kr_df + us_df
        max_domain_df = max(domains.values(), default=0)
        max_domain_share = max_domain_df / total_df if total_df else 0.0
        max_state_df = max(stats.states.values(), default=0)
        max_state_share = max_state_df / us_df if us_df else 0.0
        state_z, state_name = _state_vs_rest(stats, documents) if us_df else (None, "")
        seed_matches = seed_crosswalk.get((language, value), [])
        ranking_score = 10 * abs(z_presence) + 2 * math.log2(1 + total_df) + int(meets_rule) + 0.15 * min(n, 3)
        rows.append({
            "candidate_id": _candidate_id(language, n, value),
            "surface_form": stats.surface_form, "normalized_surface_form": value,
            "language": language, "n": n, "corpus_direction": direction,
            "kr_count": kr_count, "us_count": us_count, "kr_df": kr_df, "us_df": us_df,
            "kr_df_prop": kr_df / 10, "us_df_prop": us_df / 10, "discovery_df": total_df,
            "log_odds_occurrence": occurrence, "z_log_odds_occurrence": z_occurrence,
            "log_odds_document_presence": presence, "z_log_odds_document_presence": z_presence,
            "discovery_log_odds": z_presence, "meets_review_rule": meets_rule,
            "candidate_stability": stability, "candidate_stability_runs": len(documents),
            "kr_domain_count": len(stats.domains["KR"]), "us_domain_count": len(stats.domains["US"]),
            "kr_domains": _json_list(stats.domains["KR"]), "us_domains": _json_list(stats.domains["US"]),
            "max_domain_df": max_domain_df, "max_domain_df_share": max_domain_share,
            "domain_concentration": total_df >= 2 and max_domain_share >= CONCENTRATION_THRESHOLD,
            "us_state_count": len(stats.states), "us_states": _json_list(stats.states),
            "max_state_df": max_state_df, "max_state_df_share": max_state_share,
            "state_concentration": us_df >= 2 and max_state_share >= CONCENTRATION_THRESHOLD,
            "max_state_vs_rest_z": state_z, "max_state_vs_rest_state": state_name,
            "ranking_score": ranking_score, "noise_flags": "|".join(flags),
            "already_in_official_seed": bool(seed_matches),
            "official_seed_surface_ids": "|".join(sorted({row["surface_id"] for row in seed_matches})),
            "official_seed_concept_id": "|".join(sorted({row["concept_id"] for row in seed_matches})),
            "source_provenance": "both" if seed_matches else "discovery_corpus",
            "example_case_ids": "", "example_contexts": "",
        })
    return rows


def rank_candidates(rows: Sequence[dict]) -> list[dict]:
    return sorted((row for row in rows if row["meets_review_rule"]), key=lambda row: (
        -row["ranking_score"], -row["discovery_df"], -row["n"],
        row["language"], row["normalized_surface_form"], row["candidate_id"],
    ))


def collect_contexts(documents: Sequence[DiscoveryDocument], ranked: Sequence[dict]) -> list[dict]:
    wanted = {(row["n"], row["normalized_surface_form"]): row for row in ranked}
    contexts: dict[tuple[int, str], list[dict]] = defaultdict(list)
    seen: dict[tuple[int, str], set[str]] = defaultdict(set)
    for document in documents:
        metadata = document.metadata
        tokens = tokenize(document.text, metadata.origin_country)
        for n, value, start, end in iter_ngrams(tokens):
            key = (n, value)
            if key not in wanted or metadata.case_id in seen[key] or len(contexts[key]) >= CONTEXT_LIMIT:
                continue
            matched = document.text[start:end]
            contexts[key].append({
                "candidate_id": wanted[key]["candidate_id"], "surface_form": wanted[key]["surface_form"],
                "normalized_surface_form": value, "language": "ko" if metadata.origin_country == "KR" else "en",
                "case_id": metadata.case_id, "nested_split": "discovery",
                "origin_country": metadata.origin_country, "origin_state": metadata.origin_state,
                "primary_domain": metadata.primary_domain, "start": start, "end": end,
                "matched_text": matched,
                "context_before": document.text[max(0, start - 160):start],
                "context_after": document.text[end:end + 160],
            })
            seen[key].add(metadata.case_id)
    flattened = []
    for row in ranked:
        key = (row["n"], row["normalized_surface_form"])
        examples = contexts[key]
        assert examples
        row["example_case_ids"] = "|".join(item["case_id"] for item in examples)
        row["example_contexts"] = json.dumps([
            normalize_space(item["context_before"][-60:] + item["matched_text"] + item["context_after"][:60])
            for item in examples[:3]
        ], ensure_ascii=False, separators=(",", ":"))
        flattened.extend(examples)
    return flattened


def merged_seed_evidence(official_seed: Sequence[dict], ranked: Sequence[dict]) -> tuple[list[dict], list[str]]:
    columns = list(official_seed[0]) + [
        "normalized_surface_form", "source_provenance", "discovery_candidate_id",
        "discovery_df", "discovery_log_odds", "candidate_stability", "discovery_example_case_ids",
    ]
    discovery_by_surface = defaultdict(list)
    for row in ranked:
        discovery_by_surface[(row["language"], row["normalized_surface_form"])].append(row)
    merged = []
    covered_candidates = set()
    for seed in official_seed:
        language = seed["language"]
        normalized = normalize_space(seed["surface_form"])
        if language == "en":
            normalized = normalized.lower()
        matches = discovery_by_surface.get((language, normalized), [])
        if not matches:
            merged.append({**seed, "normalized_surface_form": normalized,
                           "source_provenance": "official_resource_seed", "discovery_candidate_id": "",
                           "discovery_df": "", "discovery_log_odds": "", "candidate_stability": "",
                           "discovery_example_case_ids": ""})
        for match in matches:
            covered_candidates.add(match["candidate_id"])
            merged.append({**seed, "normalized_surface_form": normalized, "source_provenance": "both",
                           "discovery_candidate_id": match["candidate_id"], "discovery_df": match["discovery_df"],
                           "discovery_log_odds": match["discovery_log_odds"],
                           "candidate_stability": match["candidate_stability"],
                           "discovery_example_case_ids": match["example_case_ids"]})
    for match in ranked:
        if match["candidate_id"] in covered_candidates:
            continue
        blank = {column: "" for column in official_seed[0]}
        blank.update({
            "language": match["language"], "surface_form": match["surface_form"],
            "normalized_surface_form": match["normalized_surface_form"],
            "source_provenance": "discovery_corpus", "discovery_candidate_id": match["candidate_id"],
            "discovery_df": match["discovery_df"], "discovery_log_odds": match["discovery_log_odds"],
            "candidate_stability": match["candidate_stability"],
            "discovery_example_case_ids": match["example_case_ids"],
        })
        merged.append(blank)
    return merged, columns


def review_rows(ranked: Sequence[dict]) -> list[dict]:
    return [{**row, **{
        "human_keep": "", "human_category": "", "human_concept_family": "",
        "human_concept_id": "", "human_jurisdiction_scope": "", "human_notes": "",
    }} for row in ranked]


def validate_revision_evidence(rows: Sequence[dict], split_map: dict[str, str], confirmatory_ids: set[str]) -> None:
    """Guard for the later human tuning stage; validation/confirmatory evidence is forbidden."""
    for row in rows:
        evidence_split = row.get("evidence_split", "")
        case_id = row.get("case_id", "")
        assert evidence_split in {"discovery", "tuning", "official_resource"}
        assert case_id not in confirmatory_ids
        if case_id:
            assert case_id in split_map and split_map[case_id] == evidence_split


def assert_artifact_case_ids(case_ids: Iterable[str], expected_split: str, split_map: dict[str, str]) -> None:
    """Reusable hard gate for future tuning and one-shot validation artifacts."""
    assert expected_split in SPLITS
    for case_id in case_ids:
        assert case_id in split_map, f"Non-development case in {expected_split} artifact: {case_id}"
        assert split_map[case_id] == expected_split, f"Wrong nested split in {expected_split} artifact: {case_id}"


def _csv_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else format(value, ".12g")
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def readme_text() -> str:
    return """# Nested legal-lexicon development pipeline v1

Status: **DISCOVERY EVIDENCE GENERATED — HUMAN REVIEW REQUIRED**.

The frozen v4.0.2 development set is deterministically nested into discovery 20, tuning 10, and
validation 10. Split assignment uses metadata only (`origin_country`, `primary_domain`, and US
`origin_state`) with SHA-256 tie-breaking. Opinion text is accessed only for discovery cases in this run.
Tuning, validation, and all 160 confirmatory opinions remain lexically untouched.

Candidate evidence consists of NFKC-normalized 1–5 grams, raw occurrence counts, binary document
frequency, weighted informative-prior log-odds, domain/state concentration, exact source contexts, and
analytical leave-one-document-out stability. The review rule is country-specific DF >= 2 plus conservative
noise filtering. DF=1 candidates remain in the archival file.

`candidate_stability` is the fraction of the 20 possible leave-one-discovery-document-out datasets in
which that same review rule remains satisfied. It is diagnostic only. No candidate is accepted or rejected
as legally valid based on DF, log-odds, concentration, or stability.

High KR-vs-US log-odds does **not** establish jurisdiction specificity. Language, domain, state composition,
and opinion length are confounded; the statistic only prioritizes bilingual legal review.

The official-resource seed v0.1.1 is merged by exact language + NFKC/whitespace-normalized surface form.
Provenance remains `official_resource_seed`, `discovery_corpus`, or `both`. Discovery candidates are not
automatically added to a lexicon.

## Stop condition

No draft reviewed lexicon is available yet. Therefore this run stops before reading tuning text. A human
must complete `discovery_candidates_for_human_review.csv` using only categories doctrine, remedy,
procedure, and party_arg. Only then may a draft lexicon be matched on tuning 10. Validation must be run
once on a pre-validation frozen candidate and can never trigger modification.

Historical all-40-development candidate artifacts are not inputs; see `DEPRECATED_ARTIFACTS.md`.
"""


def deprecation_text() -> str:
    return """# Deprecated overlapping-development methodology

The following historical artifacts are marked:

`DEPRECATED_METHOD_OVERLAPPING_DEV_VALIDATION`

- `legal_lexicon_dev_reviewed_v0_9`
- `lexicon_review_shortlist_v0_1`
- `outputs_v2/lexicon_dev_candidates_v0_1`

They remain unchanged for provenance but are forbidden as lexicon-development evidence or pipeline input.
Only the independent official-resource seed `legal_lexicon_seed_v0_1_1` is reused.
"""


def write_outputs(
    output_dir: Path, development: Sequence[CaseMetadata], split_map: dict[str, str],
    distribution: Sequence[dict], documents: Sequence[DiscoveryDocument], all_rows: list[dict],
    ranked: list[dict], contexts: Sequence[dict], official_seed: Sequence[dict],
    merged: Sequence[dict], merged_columns: Sequence[str], source_manifest: dict,
    source_hash: str, seed_hashes: dict[str, str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery_dir = output_dir / "discovery_candidates"
    discovery_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "dev_nested_split_manifest.csv"
    write_csv(manifest_path, split_rows(development, split_map), SPLIT_COLUMNS)
    split_hash = sha256_file(manifest_path)
    write_csv(output_dir / "nested_split_distribution.csv", distribution,
              ["dimension", "value", "nested_split", "count", "proportion_within_split"])
    split_report = {
        "split_version": SPLIT_SALT, "assignment_uses_opinion_text": False,
        "stratification": ["origin_country", "primary_domain", "origin_state_for_US"],
        "algorithm": "deterministic greedy minimum squared target-distribution deviation",
        "tie_break": 'sha256(f"{case_id}|{nested_split}|lexicon_nested_split_v1")',
        "counts": {split: Counter(split_map.values())[split] for split in SPLITS},
        "country_counts": {
            split: dict(sorted(Counter(item.origin_country for item in development if split_map[item.case_id] == split).items()))
            for split in SPLITS
        },
        "dev_nested_split_manifest_sha256": split_hash,
    }
    (output_dir / "nested_split_report.json").write_text(
        json.dumps(split_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    archival = sorted(all_rows, key=lambda row: (row["language"], row["n"], row["normalized_surface_form"], row["candidate_id"]))
    write_csv(discovery_dir / "discovery_candidates_all.csv", archival, ALL_CANDIDATE_COLUMNS)
    write_csv(discovery_dir / "discovery_candidates_ranked.csv", ranked, ALL_CANDIDATE_COLUMNS)
    reviews = review_rows(ranked)
    write_csv(discovery_dir / "discovery_candidates_for_human_review.csv", reviews, REVIEW_COLUMNS)
    write_csv(discovery_dir / "official_seed_discovery_merged_evidence.csv", merged, merged_columns)
    with (discovery_dir / "discovery_candidate_contexts.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for context in contexts:
            handle.write(json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    inventory = [{
        **document.metadata.__dict__, "nested_split": "discovery",
        "main_opinion_chars": len(document.text),
        "main_opinion_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
    } for document in documents]
    write_csv(discovery_dir / "discovery_corpus_inventory.csv", inventory,
              [field for field in CaseMetadata.__dataclass_fields__] + ["nested_split", "main_opinion_chars", "main_opinion_sha256"])

    overlap_count = sum(row["already_in_official_seed"] for row in ranked)
    candidate_manifest = {
        "stage": "discovery_candidate_generation", "status": "AWAITING_HUMAN_REVIEW",
        "nested_split_manifest_sha256": split_hash,
        "discovery_case_ids": [document.metadata.case_id for document in documents],
        "discovery_count": 20, "kr_discovery_count": 10, "us_discovery_count": 10,
        "tuning_text_accessed": False, "validation_text_accessed": False,
        "confirmatory_text_accessed": False,
        "ngram_range": list(NGRAMS), "raw_df_threshold": 1,
        "review_rule": "kr_df >= 2 or us_df >= 2, and no conservative noise flag",
        "stopword_version": STOPWORD_VERSION,
        "stopwords": {"en": sorted(EN_STOPWORDS), "ko": sorted(KO_STOPWORDS)},
        "log_odds": {
            "primary": "binary document presence weighted informative-Dirichlet log odds",
            "secondary": "occurrence weighted informative-Dirichlet log odds",
            "prior_strength": PRIOR_STRENGTH,
            "formula": "delta=log((a+alpha)/(N_a-a+alpha_other))-log((b+alpha)/(N_b-b+alpha_other)); z=delta/sqrt(1/(a+alpha)+1/(b+alpha))",
        },
        "candidate_stability": "analytical leave-one-discovery-document-out review-rule fraction; 20 runs",
        "raw_candidate_count": len(all_rows), "ranked_candidate_count": len(ranked),
        "official_seed_overlap_count": overlap_count,
        "official_seed_row_count": len(official_seed),
        "human_review_completed": False, "draft_lexicon_built": False,
    }
    (discovery_dir / "discovery_candidate_manifest.json").write_text(
        json.dumps(candidate_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "README.md").write_text(readme_text(), encoding="utf-8", newline="\n")
    (output_dir / "DEPRECATED_ARTIFACTS.md").write_text(deprecation_text(), encoding="utf-8", newline="\n")
    write_csv(output_dir / "tuning_revision_log_template.csv", [], [
        "revision_id", "surface_or_concept_affected", "before", "after", "reason",
        "evidence_split", "case_id", "reviewer_note",
    ])

    run_manifest = {
        "pipeline_version": "legal_lexicon_nested_v1",
        "stage_completed": "discovery_candidate_generation",
        "source_corpus_version": source_manifest["corpus_version"],
        "source_corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_corpus_sha256": source_hash, "source_corpus_verified": True,
        "source_corpus_status": "FROZEN", "source_corpus_unchanged": sha256_file(CORPUS_PATH) == source_hash,
        "nested_split_manifest_sha256": split_hash,
        "official_seed_version": "v0.1.1", "official_seed_sha256": seed_hashes,
        "deprecated_artifacts_used_as_input": False,
        "neutral_fact_inputs_used": False,
        "confirmatory_text_used_for_lexicon_development": False,
        "validation_text_used_for_lexicon_modification": False,
        "tuning_text_used_in_this_run": False, "validation_text_used_in_this_run": False,
        "confirmatory_text_used_in_this_run": False,
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "generation_timestamp_utc": source_manifest["frozen_at_utc"],
        "timestamp_policy": "deterministic source-corpus freeze timestamp",
        "randomness_used": False, "random_seed": None,
        "counts": candidate_manifest,
        "next_required_step": "human review of discovery_candidates_for_human_review.csv before any tuning-text access",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in files
    )
    (output_dir / "SHA256SUMS.txt").write_text(checksum_text, encoding="ascii", newline="\n")
    return candidate_manifest


def run(corpus_path: Path = CORPUS_PATH, manifest_path: Path = FREEZE_MANIFEST_PATH,
        output_dir: Path = OUTPUT_DIR) -> dict:
    development, confirmatory_ids, source_manifest, source_hash = read_verified_metadata(corpus_path, manifest_path)
    split_map = assign_nested_split(development)
    assert not (set(split_map) & confirmatory_ids)
    documents = load_discovery_documents(corpus_path, split_map)
    assert not ({document.metadata.case_id for document in documents} & confirmatory_ids)
    candidates = collect_candidate_stats(documents)
    official_seed, seed_crosswalk, _ = load_official_seed()
    rows = build_candidate_rows(candidates, documents, seed_crosswalk)
    ranked = rank_candidates(rows)
    contexts = collect_contexts(documents, ranked)
    discovery_ids = {document.metadata.case_id for document in documents}
    assert all(context["case_id"] in discovery_ids and context["nested_split"] == "discovery" for context in contexts)
    merged, merged_columns = merged_seed_evidence(official_seed, ranked)
    seed_hashes = {
        path.name: sha256_file(path) for path in sorted(SEED_DIR.glob("*.csv"))
        if path.name in {"concepts.csv", "surface_forms.csv", "seed_lexicon_combined.csv"}
    }
    return write_outputs(
        output_dir, development, split_map, distribution_rows(development, split_map), documents,
        rows, ranked, contexts, official_seed, merged, merged_columns, source_manifest, source_hash, seed_hashes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--freeze-manifest", type=Path, default=FREEZE_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.corpus.resolve(), args.freeze_manifest.resolve(), args.output_dir.resolve())
    print("source corpus verified: yes")
    print("nested split created: yes")
    print("discovery: KR 10 / US 10")
    print("tuning: KR 5 / US 5")
    print("validation: KR 5 / US 5")
    print("confirmatory untouched: 160")
    print()
    print(f"discovery raw candidates: {result['raw_candidate_count']}")
    print(f"discovery ranked candidates: {result['ranked_candidate_count']}")
    print(f"official-seed overlaps: {result['official_seed_overlap_count']}")
    print()
    print(f"output directory: {args.output_dir.resolve()}")
    print("next required human-review step: complete discovery_candidates_for_human_review.csv")


if __name__ == "__main__":
    main()
