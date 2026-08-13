from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from .io_utils import normalized_whitespace, sha256_text, stable_id
from .rules import assess_fact_sufficiency, civil_liability_candidate, classify_domain, duplicate_family_id

SOURCE_REPOSITORY = "https://github.com/legalize-kr/precedent-kr"
DEFAULT_SOURCE_DIR = Path(".cache_v2/precedent-kr")


def repository_revision(source_dir: Path) -> str:
    git_dir = source_dir / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[5:]
        loose = git_dir / Path(ref)
        if loose.exists():
            return loose.read_text(encoding="ascii").strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line and not line.startswith(("#", "^")) and line.endswith(f" {ref}"):
                    return line.split(" ", 1)[0]
    return head


def parse_markdown_record(path: Path) -> tuple[dict[str, Any], str, str]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements-v2.txt") from exc
    markdown = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", markdown, re.S)
    if not match:
        return {}, markdown, ""
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    body = markdown[match.end():]
    section = re.search(r"(?ms)^##\s+판례내용\s*$\n?(.*?)(?=^##\s+|\Z)", body)
    opinion = section.group(1).strip() if section else ""
    return metadata, markdown, opinion


def iter_precedent_files(source_dir: Path, *, limit: int = 0) -> Iterator[Path]:
    root = source_dir / "민사" / "대법원"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Structured Korean source is missing at {root}. Sparse-clone {SOURCE_REPOSITORY} "
            "and check out the 민사/대법원 path."
        )
    for index, path in enumerate(sorted(root.glob("*.md"))):
        if limit and index >= limit:
            break
        yield path


def evaluate_legalize_file(
    path: Path, *, source_dir: Path, revision: str, start_date: str, end_date: str,
    min_chars: int = 1200,
) -> dict[str, Any] | None:
    metadata, markdown, opinion = parse_markdown_record(path)
    serial = normalized_whitespace(metadata.get("판례일련번호"))
    case_number = normalized_whitespace(metadata.get("사건번호"))
    case_name = normalized_whitespace(metadata.get("사건명"))
    court_name = normalized_whitespace(metadata.get("법원명"))
    court_level = normalized_whitespace(metadata.get("법원등급"))
    case_type = normalized_whitespace(metadata.get("사건종류"))
    decision_date = normalized_whitespace(metadata.get("선고일자"))
    source_url = normalized_whitespace(metadata.get("출처"))
    classification_text = f"{case_name}\n{markdown}"
    civil, include_evidence, incidental_exclusions = civil_liability_candidate(classification_text)
    # Broad retrieval includes damages/tort-like titles even when doctrine terms are sparse.
    broad = civil or bool(re.search(
        r"손해배상|위자료|구상금|부당이득|재해|산재|사망|상해|의료|제조물|제품|안전사고|채무부존재확인", case_name
    ))
    if not broad:
        return None
    title_central = bool(re.search(r"손해배상|위자료|구상금|불법행위|사망보상|장애보상", case_name))
    domain = classify_domain(classification_text)
    sufficiency = assess_fact_sufficiency(opinion)
    exclusions: list[str] = []
    if case_type != "민사":
        exclusions.append("case_type_not_civil")
    if court_name != "대법원" or court_level != "대법원":
        exclusions.append("not_structured_supreme_court")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", decision_date):
        exclusions.append("decision_date_unknown")
    elif not start_date <= decision_date <= end_date:
        exclusions.append("decision_date_out_of_range")
    if not opinion:
        exclusions.append("opinion_text_missing")
    elif len(opinion) < min_chars:
        exclusions.append("opinion_text_too_short")
    if not civil or not title_central:
        exclusions.append("substantive_civil_liability_not_confirmed")
    exclusions.extend(item for item in incidental_exclusions if item in {"criminal_case", "administrative_only"})
    if "insurance_only" in incidental_exclusions and re.search(r"보험|공제", case_name) and not title_central:
        exclusions.append("insurance_only")
    if "contract_only" in incidental_exclusions and not title_central:
        exclusions.append("contract_only")
    if not sufficiency["core_fact_sufficient"]:
        exclusions.append("core_fact_insufficient_before_supplementation")
    raw_hash = sha256_text(markdown)
    case_id = stable_id("KR", "legalize-kr/precedent-kr", serial or path.name, revision)
    record: dict[str, Any] = {
        "case_id": case_id, "source_dataset": "legalize-kr/precedent-kr", "source_config": "git-markdown",
        "source_revision": revision, "source_record_id": serial or path.stem,
        "source_repository": SOURCE_REPOSITORY, "source_markdown_path": path.relative_to(source_dir).as_posix(),
        "source_url": source_url, "origin_country": "KR", "origin_state": None,
        "court_name": court_name, "court_level": "supreme" if court_level == "대법원" else "other",
        "court_level_confidence": "high" if court_name == court_level == "대법원" else "low",
        "court_level_evidence": [f"structured:법원명={court_name}", f"structured:법원등급={court_level}", "path:민사/대법원"],
        "decision_date": decision_date or None,
        "decision_date_confidence": "high" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", decision_date) else "low",
        "decision_date_evidence": ["structured:선고일자"] if decision_date else [],
        "case_number": case_number or None, "case_name": case_name or None,
        "case_type": case_type, "precedent_serial_number": serial or None,
        "판례일련번호": serial or None, "사건번호": case_number or None, "사건명": case_name or None,
        "법원명": court_name or None, "법원등급": court_level or None, "사건종류": case_type or None,
        "선고일자": decision_date or None, "출처": source_url or None, "판례내용": opinion,
        **domain, **sufficiency, "civil_liability_evidence": include_evidence,
        "substantive_civil_liability_central": title_central and civil,
        "full_opinion_text": markdown, "main_opinion_text": opinion, "main_opinion_type": "court_opinion",
        "opinion_selection_reason": "structured Markdown 판례내용 section from a single Supreme Court record",
        "has_concurrence": False, "has_dissent": False, "raw_text_sha256": raw_hash,
        "raw_text_chars": len(markdown), "strict_source_eligible": not exclusions,
        "exclusion_reasons": list(dict.fromkeys(exclusions)), "lower_court_supplemented": False,
        "lower_court_case_ids": [], "lower_court_supplementation_status": "not_attempted",
        "lower_court_link_confidence": "none", "lower_court_link_evidence": [],
    }
    record["case_family_id"] = duplicate_family_id(record)
    record["highest_court_case_id"] = case_id
    return record
