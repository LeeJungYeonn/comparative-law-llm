"""Compare Korean and California case corpora with Cliff's delta and PCA.

This script adapts the feature definitions introduced in commit 35e086c to the
v4 JSONL schemas.  Positive Cliff's delta means that Korean values tend to be
larger; negative values mean that California values tend to be larger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comparative-law-llm-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_714

KR_STATUTE_RE = re.compile(
    r"(?:민법|상법|형법|민사소송법|민사집행법|국가배상법|도로교통법|근로기준법"
    r"|제조물 ?책임법|자동차손해배상 ?보장법|의료법|건축법|환경정책기본법"
    r"|같은 법|이 법|위 법|이 사건 법률)"
    r"\s*제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?"
)
KR_PRECEDENT_RE = re.compile(
    r"(?:대법원|헌법재판소|서울고등법원|부산고등법원|대구고등법원)"
    r"(?:\s*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*선고\s*)?"
    r"\s*\d{4}\s*(?:다|가|나|라|마|카|허|두|므|브|초|재|행|구|모|오|전합)\s*\d+"
    r"(?:\s*판결|\s*결정|\s*선고)?"
)
CA_STATUTE_RE = re.compile(
    r"(?:\d+\s+U\.S\.C\.\s*§+\s*\d+"
    r"|[A-Z][a-z]+\.?\s*(?:Civ|Pen|Bus|Lab|Gov|Ins|Prob|Rev|Ann|Gen|Code)\.?\s*"
    r"(?:Code)?\s*§+\s*\d+"
    r"|(?:Section|section|§)\s*\d+\s+of\s+[A-Z])"
)
CA_PRECEDENT_RE = re.compile(
    r"[A-Z][A-Za-z'\-]+\s+v\.\s+[A-Z][A-Za-z'\-]+"
    r"|\d+\s+(?:U\.S\.|F\.(?:\d+d|Supp\.(?:\s*\d+d)?)|S\.\s*Ct\.|L\.\s*Ed\.)"
    r"\s+\d+"
)
KR_CONCLUSION_RE = re.compile(
    r"따라서|이유\s*(?:있다|없다)|청구를\s*(?:인용|기각)|주문과\s*같이\s*판결|원고의\s*청구는"
)
CA_CONCLUSION_RE = re.compile(
    r"[Ww]e\s+(?:therefore\s+)?hold|[Aa]ccordingly[,.]?"
    r"|[Ff]or\s+(?:the\s+)?(?:foregoing\s+)?reasons"
    r"|[Jj]udgment\s+(?:is\s+)?(?:entered|granted|denied)"
    r"|[Pp]laintiff(?:'s)? (?:claims?|motion) (?:is |are )?(?:granted|denied|dismissed)"
)
KR_PARTY_ARG_RE = re.compile(
    r"원고(?:는|의|가|측)\s*(?:주장|청구|진술)"
    r"|피고(?:는|의|가|측)\s*(?:주장|항변|답변)"
    r"|(?:원고|피고)(?:는|가)\s+이\s*사건"
)
CA_PARTY_ARG_RE = re.compile(
    r"[Pp]laintiff\s+(?:argues?|contends?|alleges?|claims?|asserts?)"
    r"|[Dd]efendant\s+(?:argues?|contends?|asserts?|moves?|responds?)"
    r"|[Pp]etitioner\s+(?:argues?|contends?)"
    r"|[Aa]ppellant\s+(?:argues?|contends?)"
)

KR_DOCTRINE_TERMS = [
    "상당인과관계", "과실상계", "신의성실의 원칙", "신의칙", "불법행위",
    "고의 또는 과실", "위법성", "손해", "인과관계", "책임제한", "위자료",
    "재산상 손해", "정신적 손해", "지연손해금", "원상회복", "손해배상책임",
    "채무불이행", "손해배상의 범위",
]
KR_REMEDY_TERMS = [
    "손해배상", "위자료", "지연손해금", "원상회복", "재산적 손해", "일실수입",
    "일실이익", "치료비", "개호비", "향후치료비", "장례비",
]
KR_PROCEDURE_TERMS = [
    "원심", "항소", "상고", "파기환송", "청구기각", "일부인용", "전부인용",
    "소 제기", "항소심", "1심", "이 사건 소",
]
KR_JURISDICTION_TERMS = [
    "대한민국", "대법원", "고등법원", "지방법원", "서울중앙지방법원",
    "서울고등법원", "헌법재판소",
]
CA_DOCTRINE_TERMS = [
    "duty of care", "breach of duty", "negligence", "proximate cause", "causation",
    "foreseeability", "comparative negligence", "contributory negligence",
    "strict liability", "standard of care", "reasonable person", "but-for causation",
    "substantial factor",
]
CA_REMEDY_TERMS = [
    "compensatory damages", "punitive damages", "nominal damages", "injunction", "damages",
    "pain and suffering", "emotional distress", "lost wages", "medical expenses",
    "wrongful death",
]
CA_PROCEDURE_TERMS = [
    "summary judgment", "motion to dismiss", "appeal", "remand", "affirmed", "reversed",
    "de novo", "motion for judgment", "directed verdict", "judgment as a matter of law",
]
CA_JURISDICTION_TERMS = [
    "United States", "federal court", "district court", "circuit court", "Supreme Court",
    "Court of Appeals", "state court",
]

ALL_FEATURES = [
    "doc_length_tokens", "doc_length_sentences", "avg_sentence_length",
    "statute_ref_count", "precedent_citation_count", "citation_count",
    "doctrine_term_count", "remedy_term_count", "procedure_term_count",
    "jurisdiction_mention_count", "statute_per_1k", "precedent_per_1k",
    "citation_density", "doctrine_per_1k", "remedy_per_1k", "procedure_per_1k",
    "jurisdiction_per_1k", "conclusion_position", "party_arg_density",
]
PCA_FEATURES = [
    "statute_per_1k", "precedent_per_1k", "doctrine_per_1k", "remedy_per_1k",
    "procedure_per_1k", "jurisdiction_per_1k", "avg_sentence_length",
    "party_arg_density",
]
FEATURE_MEANINGS = {
    "doc_length_tokens": "판결문 토큰 수",
    "doc_length_sentences": "판결문 문장 수",
    "avg_sentence_length": "문장당 평균 토큰 수",
    "statute_ref_count": "법 조문 인용 횟수",
    "precedent_citation_count": "판례 인용 횟수",
    "citation_count": "조문·판례 인용 합계",
    "doctrine_term_count": "법리 용어 횟수",
    "remedy_term_count": "구제수단 용어 횟수",
    "procedure_term_count": "소송절차 용어 횟수",
    "jurisdiction_mention_count": "관할 관련 언급 횟수",
    "statute_per_1k": "1,000토큰당 조문 인용",
    "precedent_per_1k": "1,000토큰당 판례 인용",
    "citation_density": "1,000토큰당 전체 인용",
    "doctrine_per_1k": "1,000토큰당 법리 용어",
    "remedy_per_1k": "1,000토큰당 구제수단 용어",
    "procedure_per_1k": "1,000토큰당 소송절차 용어",
    "jurisdiction_per_1k": "1,000토큰당 관할 관련 언급",
    "conclusion_position": "최초 결론 표현의 상대 위치",
    "party_arg_density": "1,000토큰당 당사자 주장 표현",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    """Use the same cross-language regex tokenizer as preprocess_cases.py."""
    return re.findall(
        r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:,\d{3})*(?:\.\d+)?|§+|제\d+조",
        text,
    )


def split_sentences(text: str) -> list[str]:
    value = re.sub(r"\s+", " ", text).strip()
    if not value:
        return []
    replacements = {
        "U.S.": "U<S>", "U.S.C.": "U<S<C>", "F. Supp.": "F< Supp>",
        "F.2d": "F<2d", "F.3d": "F<3d", "S. Ct.": "S< Ct>",
        "L. Ed.": "L< Ed>", "No.": "No<", "Inc.": "Inc<", "Corp.": "Corp<",
        "Co.": "Co<", "Ltd.": "Ltd<", "v.": "v<",
    }
    protected = value
    for source, target in replacements.items():
        protected = protected.replace(source, target)
    parts = re.split(r"(?<=[.!?。！？])\s+(?=[A-Z가-힣\"'“‘(\[])", protected)
    restored = []
    for part in parts:
        for source, target in replacements.items():
            part = part.replace(target, source)
        if part.strip():
            restored.append(part.strip())
    return restored


def count_terms(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term.lower()) for term in terms)


def density(count: int, token_count: int) -> float:
    return round(count / token_count * 1000, 4) if token_count else 0.0


def conclusion_position(text: str, pattern: re.Pattern) -> float:
    match = pattern.search(text)
    return round(match.start() / max(len(text), 1), 4) if match else -1.0


def extract_features(record: dict, jurisdiction: str) -> dict:
    is_kr = jurisdiction == "KR"
    text_key = "raw_text" if is_kr else "main_opinion_text"
    text = str(record.get(text_key) or "")
    tokens = tokenize(text)
    sentences = split_sentences(text)
    token_count = len(tokens)
    sentence_count = max(len(sentences), 1)

    statute_pattern = KR_STATUTE_RE if is_kr else CA_STATUTE_RE
    precedent_pattern = KR_PRECEDENT_RE if is_kr else CA_PRECEDENT_RE
    conclusion_pattern = KR_CONCLUSION_RE if is_kr else CA_CONCLUSION_RE
    party_pattern = KR_PARTY_ARG_RE if is_kr else CA_PARTY_ARG_RE
    statute_count = len(statute_pattern.findall(text))
    precedent_count = len(precedent_pattern.findall(text))
    doctrine_count = count_terms(text, KR_DOCTRINE_TERMS if is_kr else CA_DOCTRINE_TERMS)
    remedy_count = count_terms(text, KR_REMEDY_TERMS if is_kr else CA_REMEDY_TERMS)
    procedure_count = count_terms(text, KR_PROCEDURE_TERMS if is_kr else CA_PROCEDURE_TERMS)
    jurisdiction_count = count_terms(
        text, KR_JURISDICTION_TERMS if is_kr else CA_JURISDICTION_TERMS
    )
    party_count = len(party_pattern.findall(text))

    return {
        "case_id": record.get("case_id", ""),
        "jurisdiction": jurisdiction,
        "text_field": text_key,
        "doc_length_tokens": token_count,
        "doc_length_sentences": sentence_count,
        "avg_sentence_length": round(token_count / sentence_count, 4),
        "statute_ref_count": statute_count,
        "precedent_citation_count": precedent_count,
        "citation_count": statute_count + precedent_count,
        "doctrine_term_count": doctrine_count,
        "remedy_term_count": remedy_count,
        "procedure_term_count": procedure_count,
        "jurisdiction_mention_count": jurisdiction_count,
        "statute_per_1k": density(statute_count, token_count),
        "precedent_per_1k": density(precedent_count, token_count),
        "citation_density": density(statute_count + precedent_count, token_count),
        "doctrine_per_1k": density(doctrine_count, token_count),
        "remedy_per_1k": density(remedy_count, token_count),
        "procedure_per_1k": density(procedure_count, token_count),
        "jurisdiction_per_1k": density(jurisdiction_count, token_count),
        "conclusion_position": conclusion_position(text, conclusion_pattern),
        "party_arg_density": density(party_count, token_count),
    }


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    dominance = np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])
    return float(dominance / (len(a) * len(b)))


def bootstrap_delta_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, resamples: int
) -> tuple[float, float]:
    """Percentile bootstrap CI using bounded-memory chunks."""
    deltas = np.empty(resamples, dtype=float)
    chunk_size = 500
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        size = stop - start
        a_samples = rng.choice(a, size=(size, len(a)), replace=True)
        b_samples = rng.choice(b, size=(size, len(b)), replace=True)
        greater = (a_samples[:, :, None] > b_samples[:, None, :]).sum(axis=(1, 2))
        less = (a_samples[:, :, None] < b_samples[:, None, :]).sum(axis=(1, 2))
        deltas[start:stop] = (greater - less) / (len(a) * len(b))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def effect_size_label(delta: float) -> str:
    absolute = abs(delta)
    if absolute < 0.147:
        return "negligible"
    if absolute < 0.330:
        return "small"
    if absolute < 0.474:
        return "medium"
    return "large"


def feature_statistics(features: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n_tests = len(ALL_FEATURES)
    for feature in ALL_FEATURES:
        kr = features.loc[features["jurisdiction"] == "KR", feature].replace(-1, np.nan).dropna().to_numpy()
        ca = features.loc[features["jurisdiction"] == "CA", feature].replace(-1, np.nan).dropna().to_numpy()
        delta = cliffs_delta(kr, ca)
        ci_low, ci_high = bootstrap_delta_ci(kr, ca, rng, resamples)
        u_stat, p_raw = stats.mannwhitneyu(kr, ca, alternative="two-sided")
        rows.append({
            "feature": feature,
            "meaning": FEATURE_MEANINGS[feature],
            "n_KR": len(kr),
            "n_CA": len(ca),
            "KR_mean": kr.mean(),
            "CA_mean": ca.mean(),
            "KR_median": np.median(kr),
            "CA_median": np.median(ca),
            "cliffs_delta_KR_vs_CA": delta,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "effect_size": effect_size_label(delta),
            "mann_whitney_u": u_stat,
            "p_raw": p_raw,
            "p_bonferroni": min(p_raw * n_tests, 1.0),
        })
    result = pd.DataFrame(rows)
    return result.sort_values("cliffs_delta_KR_vs_CA", key=lambda s: s.abs(), ascending=False)


def run_pca(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    scaled = StandardScaler().fit_transform(features[PCA_FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    loadings = pca.components_.T.copy()

    # PCA signs are arbitrary. Orient PC1 so the Korean centroid is positive.
    kr_mask = features["jurisdiction"].eq("KR").to_numpy()
    if coordinates[kr_mask, 0].mean() < coordinates[~kr_mask, 0].mean():
        coordinates[:, 0] *= -1
        loadings[:, 0] *= -1

    scores = features[["case_id", "jurisdiction"]].copy()
    scores["PC1"] = coordinates[:, 0]
    scores["PC2"] = coordinates[:, 1]
    loading_df = pd.DataFrame(loadings, index=PCA_FEATURES, columns=["PC1", "PC2"])
    loading_df.index.name = "feature"
    return scores, loading_df.reset_index(), pca.explained_variance_ratio_


def configure_fonts() -> None:
    candidates = ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans KR"]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    matplotlib.rc("font", family=selected)
    matplotlib.rc("axes", unicode_minus=False)


def plot_deltas(statistics: pd.DataFrame, output: Path) -> None:
    ordered = statistics.sort_values("cliffs_delta_KR_vs_CA")
    y = np.arange(len(ordered))
    values = ordered["cliffs_delta_KR_vs_CA"].to_numpy()
    low = ordered["ci_95_low"].to_numpy()
    high = ordered["ci_95_high"].to_numpy()
    colors = np.where(values >= 0, "#4472C4", "#ED7D31")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.errorbar(values, y, xerr=np.vstack([values - low, high - values]), fmt="none", ecolor="black", capsize=2)
    ax.set_yticks(y, ordered["feature"])
    ax.axvline(0, color="black", linewidth=0.8)
    for threshold in (-0.474, -0.33, -0.147, 0.147, 0.33, 0.474):
        ax.axvline(threshold, color="grey", linewidth=0.5, linestyle="--", alpha=0.45)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's delta (positive = KR larger; negative = CA larger)")
    ax.set_title("Korea vs California: Cliff's delta with 95% bootstrap CI")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca(scores: pd.DataFrame, variance: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    styles = {"KR": ("#4472C4", "o", "Korea"), "CA": ("#ED7D31", "^", "California")}
    for jurisdiction, (color, marker, label) in styles.items():
        group = scores[scores["jurisdiction"] == jurisdiction]
        ax.scatter(group["PC1"], group["PC2"], color=color, marker=marker, label=label,
                   alpha=0.75, s=60, edgecolors="white", linewidths=0.5)
        ax.scatter(group["PC1"].mean(), group["PC2"].mean(), color=color, marker="X",
                   s=180, edgecolors="black", linewidths=0.8)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance; positive oriented toward KR)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    ax.set_title("PCA of Legal Feature Vectors: Korea vs California")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loadings(loadings: pd.DataFrame, output: Path) -> None:
    ordered = loadings.iloc[loadings["PC1"].abs().sort_values().index]
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, component in zip(axes, ["PC1", "PC2"]):
        values = ordered[component]
        ax.barh(y, values, color=np.where(values >= 0, "#4472C4", "#ED7D31"), alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, ordered["feature"])
        ax.set_xlabel(f"{component} loading")
        ax.grid(axis="x", linestyle=":", alpha=0.3)
    axes[0].set_title("PC1 loadings (positive oriented toward KR)")
    axes[1].set_title("PC2 loadings (sign arbitrary)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(statistics: pd.DataFrame) -> str:
    lines = [
        "| feature | KR mean | CA mean | Cliff's δ | 95% CI | effect | Bonf. p |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in statistics.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.KR_mean:.4f} | {row.CA_mean:.4f} | "
            f"{row.cliffs_delta_KR_vs_CA:.4f} | [{row.ci_95_low:.4f}, {row.ci_95_high:.4f}] | "
            f"{row.effect_size} | {row.p_bonferroni:.4g} |"
        )
    return "\n".join(lines)


def write_report(
    output: Path,
    statistics: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    variance: np.ndarray,
    metadata: dict,
) -> None:
    centroids = scores.groupby("jurisdiction")[["PC1", "PC2"]].mean()
    centroid_distance = float(np.linalg.norm(centroids.loc["KR"] - centroids.loc["CA"]))
    kr_pc1_positive = int(((scores["jurisdiction"] == "KR") & (scores["PC1"] > 0)).sum())
    ca_pc1_negative = int(((scores["jurisdiction"] == "CA") & (scores["PC1"] < 0)).sum())
    top_pc1 = loadings.iloc[loadings["PC1"].abs().sort_values(ascending=False).index[:4]]
    top_loading_text = ", ".join(f"{row.feature} ({row.PC1:+.3f})" for row in top_pc1.itertuples())
    notable = statistics[statistics["effect_size"].isin(["large", "medium"])]
    positive = notable[notable["cliffs_delta_KR_vs_CA"] > 0].head(3)["feature"].tolist()
    negative = notable[notable["cliffs_delta_KR_vs_CA"] < 0].head(3)["feature"].tolist()
    report = f"""# 한국–California 판례 feature 비교 (각 35건)

## 핵심 결과

- Cliff's δ가 양수이면 한국(KR), 음수이면 California(CA) 값이 큰 경향을 뜻한다.
- 큰/중간 효과 중 KR 방향 상위 feature: {', '.join(positive) if positive else '없음'}.
- 큰/중간 효과 중 CA 방향 상위 feature: {', '.join(negative) if negative else '없음'}.
- PCA 설명분산은 PC1 {variance[0] * 100:.2f}%, PC2 {variance[1] * 100:.2f}% (합계 {variance[:2].sum() * 100:.2f}%)이다.
- 2차원 PCA 공간의 집단 중심 간 거리는 {centroid_distance:.3f}이다. 이는 시각적 분리의 기술통계이며 분류 정확도나 인과효과가 아니다.
- PC1=0을 기술적 기준으로 보면 KR {kr_pc1_positive}/35건이 양수, CA {ca_pc1_negative}/35건이 음수였다. 이는 같은 표본에서 관찰된 값이며 표본 밖 예측성능이 아니다.
- PC1 절댓값 기준 주요 loading: {top_loading_text}.

## Cliff's delta와 검정 결과

{markdown_table(statistics)}

## 방법

- 입력: `ca_cases_selected_35.jsonl` 35건, `kr_cases_selected_35.jsonl` 35건.
- 본문 필드: CA=`main_opinion_text`, KR=`raw_text`; 빈 본문 0건, case_id 중복 0건.
- feature 정의는 2026-05-10 `feature_analysis.ipynb`의 regex·사전·1,000토큰당 밀도를 재사용했다.
- 토큰·문장 분리는 `preprocess_cases.py`와 동일한 정규식 규칙을 사용했다.
- Cliff's δ 95% CI는 percentile bootstrap {metadata['bootstrap_resamples']:,}회, seed={metadata['bootstrap_seed']}이다.
- Mann–Whitney U 양측검정과 {len(ALL_FEATURES)}개 feature Bonferroni 보정 p값을 함께 제시했다.
- PCA는 8개 밀도/구조 feature를 z-score 표준화한 뒤 계산했다. PC1 부호는 KR 중심이 양수가 되도록 정렬했으며 PCA 부호 자체는 임의적이다.
- 이 결과는 언어별 정규식·용어 사전의 탐지 민감도와 표본 구성에 영향을 받으므로, 법체계의 본질적 차이에 대한 직접적 인과증거로 해석하지 않는다.

## 재현 정보

- CA SHA-256: `{metadata['ca_sha256']}`
- KR SHA-256: `{metadata['kr_sha256']}`
- 생성 파일: `feature_vectors.csv`, `cliffs_delta_by_feature.csv`, `pca_scores.csv`, `pca_loadings.csv`, `analysis_metadata.json`, `plots/*.png`.
"""
    output.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-jsonl", type=Path, default=Path("outputs/raw/ca_v4/ca_cases_selected_35.jsonl"))
    parser.add_argument("--kr-jsonl", type=Path, default=Path("outputs/raw/kr_v4/kr_cases_selected_35.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/kr_ca_35"))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ca_records = read_jsonl(args.ca_jsonl)
    kr_records = read_jsonl(args.kr_jsonl)
    if len(ca_records) != 35 or len(kr_records) != 35:
        raise ValueError(f"Expected 35 records per group; found KR={len(kr_records)}, CA={len(ca_records)}")
    for records, field, label in [(kr_records, "raw_text", "KR"), (ca_records, "main_opinion_text", "CA")]:
        if any(not str(record.get(field) or "").strip() for record in records):
            raise ValueError(f"{label} contains an empty {field}")
        ids = [record.get("case_id") for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label} contains duplicate case_id values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    configure_fonts()

    feature_rows = [extract_features(record, "KR") for record in kr_records]
    feature_rows.extend(extract_features(record, "CA") for record in ca_records)
    features = pd.DataFrame(feature_rows)
    statistics = feature_statistics(features, args.bootstrap_resamples, args.bootstrap_seed)
    scores, loadings, variance = run_pca(features)

    features.to_csv(args.output_dir / "feature_vectors.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(args.output_dir / "cliffs_delta_by_feature.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(args.output_dir / "pca_scores.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(args.output_dir / "pca_loadings.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "ca_input": str(args.ca_jsonl),
        "kr_input": str(args.kr_jsonl),
        "ca_sha256": file_sha256(args.ca_jsonl),
        "kr_sha256": file_sha256(args.kr_jsonl),
        "n_CA": len(ca_records),
        "n_KR": len(kr_records),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "pca_features": PCA_FEATURES,
        "pca_explained_variance_ratio": variance.tolist(),
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_deltas(statistics, plot_dir / "01_cliffs_delta.png")
    plot_pca(scores, variance, plot_dir / "02_pca.png")
    plot_loadings(loadings, plot_dir / "03_pca_loadings.png")
    write_report(args.output_dir / "analysis_summary.md", statistics, scores, loadings, variance, metadata)

    print(f"Completed KR={len(kr_records)}, CA={len(ca_records)}")
    print(f"Output: {args.output_dir}")
    print(f"PCA variance: PC1={variance[0]:.4f}, PC2={variance[1]:.4f}")
    print(statistics[["feature", "cliffs_delta_KR_vs_CA", "ci_95_low", "ci_95_high", "effect_size"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
