from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.io_utils import read_jsonl, sha256_text, write_json, write_jsonl


SOURCE = Path("outputs_v2/v4.0.1")
DESTINATION = Path("outputs_v2/v4.0.2")
SOURCE_VERSION = "kr-us-highcourt-corpus-v4.0.1"
VERSION = "kr-us-highcourt-corpus-v4.0.2"
SUFFIX = "v4_0_2"
TARGETS = {"US_9dc742b4409b555931", "KR_6e91996913ed443be6"}
SOURCE_FILES = {
    "cases": SOURCE / "final_cases_200_v4_0_1.jsonl",
    "facts": SOURCE / "final_fact_patterns_200_v4_0_1.jsonl",
    "units": SOURCE / "final_fact_units_200_v4_0_1.jsonl",
    "roster": SOURCE / "final_roster_manifest_200_v4_0_1.csv",
    "prior_qc": SOURCE / "canonical_final_qc_200_v4_0_1.jsonl",
}
EXPECTED_SOURCE_HASHES = {
    "cases": "c9b6cffa2b1c75b67c2d7e031a604a26b49c57c317a61a649731f27127d1ec23",
    "facts": "5e50e1c067abd9c7167f0bd36896523c6a5264ece5b2ee06d8a46a22d3814a8a",
    "units": "65bc8417b0c8b174cce0a1a1aee01edca2be0cfceff9296898e63f038521c902",
    "roster": "b2d7569eba508ca6e04bd3da2c16f570254127095e6abab3074a19bc3d6b3d9e",
    "prior_qc": "e22bf7366553291b3fbb39b17395b5c1eb9313bb1cf4eca6b6d01f86e9a6b270",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_exact(text: str, old: str, new: str, expected_count: int) -> str:
    observed = text.count(old)
    if observed != expected_count:
        raise RuntimeError(f"Expected {expected_count} occurrences of {old!r}, found {observed}")
    return text.replace(old, new)


def without_version(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "corpus_version"}


def changed_keys(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {key for key in set(before) | set(after) if before.get(key) != after.get(key)}


def main() -> None:
    if DESTINATION.exists():
        raise RuntimeError(f"Refusing to overwrite existing freeze directory: {DESTINATION}")
    observed_source_hashes = {name: sha_file(path) for name, path in SOURCE_FILES.items()}
    if observed_source_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("v4.0.1 source hashes do not match the frozen inputs")

    source_cases = [dict(row) for row in read_jsonl(SOURCE_FILES["cases"])]
    source_facts = [dict(row) for row in read_jsonl(SOURCE_FILES["facts"])]
    source_units = [dict(row) for row in read_jsonl(SOURCE_FILES["units"])]
    cases, facts, units = deepcopy(source_cases), deepcopy(source_facts), deepcopy(source_units)
    if not (len(cases) == len(facts) == len(units) == 200):
        raise RuntimeError("Frozen inputs must each contain exactly 200 rows")
    case_ids = [row["case_id"] for row in cases]
    if set(case_ids) != {row["case_id"] for row in facts} or set(case_ids) != {row["case_id"] for row in units}:
        raise RuntimeError("Case/fact/unit rosters differ")

    for collection in (cases, facts, units):
        for row in collection:
            if row.get("corpus_version") != SOURCE_VERSION:
                raise RuntimeError(f"Unexpected source corpus version for {row['case_id']}")
            row["corpus_version"] = VERSION

    fact_by_id = {row["case_id"]: row for row in facts}
    unit_by_id = {row["case_id"]: row for row in units}
    source_fact_by_id = {row["case_id"]: row for row in source_facts}
    source_unit_by_id = {row["case_id"]: row for row in source_units}

    us_fact = fact_by_id["US_9dc742b4409b555931"]
    us_fact["neutral_fact_en"] = replace_exact(
        us_fact["neutral_fact_en"], "The Fierles alleged", "[PERSON_A] and [PERSON_B] alleged", 3
    )
    us_fact["neutral_fact_ko"] = replace_exact(
        us_fact["neutral_fact_ko"], "피얼스 부부는", "[PERSON_A]와 [PERSON_B]는", 3
    )
    us_fact["neutral_fact_source"] = us_fact["neutral_fact_en"]

    kr_fact = fact_by_id["KR_6e91996913ed443be6"]
    kr_fact["neutral_fact_en"] = replace_exact(
        kr_fact["neutral_fact_en"], "a Japanese company", "[COMPANY_F]", 1
    )
    kr_fact["neutral_fact_ko"] = replace_exact(
        kr_fact["neutral_fact_ko"], "일본 업체", "[COMPANY_F]", 1
    )
    kr_fact["neutral_fact_source"] = kr_fact["neutral_fact_ko"]

    for case_id in TARGETS:
        fact = fact_by_id[case_id]
        fact["neutral_fact_ko_sha256"] = sha256_text(fact["neutral_fact_ko"])
        fact["neutral_fact_en_sha256"] = sha256_text(fact["neutral_fact_en"])
        fact["neutral_fact_source_sha256"] = sha256_text(fact["neutral_fact_source"])
        unit = unit_by_id[case_id]
        unit["neutral_ko"] = fact["neutral_fact_ko"]
        unit["neutral_en"] = fact["neutral_fact_en"]
        unit["text"] = fact["neutral_fact_source"]

    allowed_fact_changes = {
        "corpus_version", "neutral_fact_ko", "neutral_fact_en", "neutral_fact_source",
        "neutral_fact_ko_sha256", "neutral_fact_en_sha256", "neutral_fact_source_sha256",
    }
    allowed_unit_changes = {"corpus_version", "neutral_ko", "neutral_en", "text"}
    for before, after in zip(source_cases, cases, strict=True):
        if changed_keys(before, after) != {"corpus_version"}:
            raise RuntimeError(f"Unexpected case metadata change: {before['case_id']}")
    for before, after in zip(source_facts, facts, strict=True):
        changes = changed_keys(before, after)
        expected = allowed_fact_changes if before["case_id"] in TARGETS else {"corpus_version"}
        if not changes <= expected or "corpus_version" not in changes:
            raise RuntimeError(f"Unexpected fact change for {before['case_id']}: {sorted(changes)}")
        if before["case_id"] not in TARGETS and without_version(before) != without_version(after):
            raise RuntimeError(f"Non-target fact changed: {before['case_id']}")
    for before, after in zip(source_units, units, strict=True):
        changes = changed_keys(before, after)
        expected = allowed_unit_changes if before["case_id"] in TARGETS else {"corpus_version"}
        if not changes <= expected or "corpus_version" not in changes:
            raise RuntimeError(f"Unexpected unit change for {before['case_id']}: {sorted(changes)}")
        if before["case_id"] not in TARGETS and without_version(before) != without_version(after):
            raise RuntimeError(f"Non-target fact unit changed: {before['case_id']}")

    for fact in facts:
        master = fact["neutral_fact_ko" if fact["source_language"] == "ko" else "neutral_fact_en"]
        if fact["neutral_fact_source"] != master:
            raise RuntimeError(f"Source-language neutral fact is not synchronized: {fact['case_id']}")
        for field in ("neutral_fact_ko", "neutral_fact_en", "neutral_fact_source"):
            if fact[f"{field}_sha256"] != sha256_text(fact[field]):
                raise RuntimeError(f"Stale {field} hash: {fact['case_id']}")
        unit = unit_by_id[fact["case_id"]]
        if (unit["neutral_ko"], unit["neutral_en"], unit["text"]) != (
            fact["neutral_fact_ko"], fact["neutral_fact_en"], fact["neutral_fact_source"]
        ):
            raise RuntimeError(f"Fact unit is not synchronized: {fact['case_id']}")

    residuals = {
        "US_9dc742b4409b555931": any(
            token in fact_by_id["US_9dc742b4409b555931"][field]
            for field in ("neutral_fact_en", "neutral_fact_ko", "neutral_fact_source")
            for token in ("Fierles", "피얼스")
        ),
        "KR_6e91996913ed443be6": any(
            token in fact_by_id["KR_6e91996913ed443be6"][field]
            for field in ("neutral_fact_en", "neutral_fact_ko", "neutral_fact_source")
            for token in ("Japanese company", "일본 업체")
        ),
    }
    if any(residuals.values()):
        raise RuntimeError(f"Residual target strings remain: {residuals}")

    DESTINATION.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "cases": DESTINATION / f"final_cases_200_{SUFFIX}.jsonl",
        "facts": DESTINATION / f"final_fact_patterns_200_{SUFFIX}.jsonl",
        "units": DESTINATION / f"final_fact_units_200_{SUFFIX}.jsonl",
        "roster": DESTINATION / f"final_roster_manifest_200_{SUFFIX}.csv",
        "audit": DESTINATION / f"targeted_neutral_fact_corrections_{SUFFIX}.json",
        "report": DESTINATION / f"EXP1_CORPUS_FREEZE_REPORT_{SUFFIX}.md",
        "manifest": DESTINATION / f"corpus_freeze_manifest_{SUFFIX}.json",
    }
    write_jsonl(output_paths["cases"], cases)
    write_jsonl(output_paths["facts"], facts)
    write_jsonl(output_paths["units"], units)
    shutil.copyfile(SOURCE_FILES["roster"], output_paths["roster"])

    correction_rows = []
    for case_id in sorted(TARGETS):
        before, after = source_fact_by_id[case_id], fact_by_id[case_id]
        correction_rows.append({
            "case_id": case_id,
            "analysis_split": after["analysis_split"],
            "origin_country": after["origin_country"],
            "primary_domain": after["primary_domain"],
            "changed_fact_fields": sorted(changed_keys(before, after) - {"corpus_version"}),
            "changed_unit_fields": sorted(changed_keys(source_unit_by_id[case_id], unit_by_id[case_id]) - {"corpus_version"}),
            "before_hashes": {field: before[f"{field}_sha256"] for field in ("neutral_fact_ko", "neutral_fact_en", "neutral_fact_source")},
            "after_hashes": {field: after[f"{field}_sha256"] for field in ("neutral_fact_ko", "neutral_fact_en", "neutral_fact_source")},
        })
    audit = {
        "corpus_version": VERSION,
        "source_corpus_version": SOURCE_VERSION,
        "status": "PASS",
        "scope": "two user-directed neutral-fact entity-neutralization corrections only",
        "collection_rerun": False,
        "corpus_qc_rerun": False,
        "prior_canonical_qc_artifact": str(SOURCE_FILES["prior_qc"]),
        "prior_canonical_qc_sha256": observed_source_hashes["prior_qc"],
        "target_case_ids": sorted(TARGETS),
        "corrections": correction_rows,
        "non_target_fact_change_count": 0,
        "non_target_unit_change_count": 0,
        "case_metadata_change_count_excluding_version": 0,
        "domain_change_count": 0,
        "split_change_count": 0,
        "residual_target_strings": residuals,
        "all_fact_hashes_current": True,
        "all_fact_units_synchronized": True,
    }
    write_json(output_paths["audit"], audit)

    counts = {
        "total": len(cases),
        "by_split": dict(sorted(Counter(row["analysis_split"] for row in cases).items())),
        "by_country_split": {
            f"{country}_{split}": sum(
                row["origin_country"] == country and row["analysis_split"] == split for row in cases
            )
            for country in ("KR", "US") for split in ("development", "confirmatory")
        },
    }
    artifact_hashes = {
        str(path): sha_file(path)
        for key, path in output_paths.items() if key not in {"manifest", "report"}
    }
    manifest = {
        "corpus_version": VERSION,
        "status": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus_version": SOURCE_VERSION,
        "source_artifact_sha256_verified": {str(SOURCE_FILES[key]): value for key, value in observed_source_hashes.items()},
        "change_scope": "only the two specified bilingual neutral-fact entity substitutions and affected hashes/units",
        "target_case_ids": sorted(TARGETS),
        "counts": counts,
        "prior_canonical_qc_carried_forward_without_rerun": str(SOURCE_FILES["prior_qc"]),
        "invariants": {
            "total_200": len(cases) == 200,
            "case_roster_and_order_unchanged": [row["case_id"] for row in source_cases] == case_ids,
            "case_metadata_unchanged_excluding_version": True,
            "all_non_target_facts_unchanged_excluding_version": True,
            "all_non_target_units_unchanged_excluding_version": True,
            "only_two_target_facts_changed": True,
            "domains_unchanged": all(a["primary_domain"] == b["primary_domain"] for a, b in zip(source_cases, cases, strict=True)),
            "splits_unchanged": all(a["analysis_split"] == b["analysis_split"] for a, b in zip(source_cases, cases, strict=True)),
            "all_neutral_fact_hashes_current": True,
            "all_fact_units_synchronized": True,
            "all_residual_target_strings_removed": not any(residuals.values()),
            "kr_development_20": counts["by_country_split"]["KR_development"] == 20,
            "kr_confirmatory_80": counts["by_country_split"]["KR_confirmatory"] == 80,
            "us_development_20": counts["by_country_split"]["US_development"] == 20,
            "us_confirmatory_80": counts["by_country_split"]["US_confirmatory"] == 80,
        },
        "artifact_sha256": artifact_hashes,
        "collection_rerun": False,
        "corpus_qc_rerun": False,
    }
    write_json(output_paths["manifest"], manifest)
    report = [
        "# Exp 1 corpus freeze v4.0.2", "", "Status: **FROZEN**", "",
        "Only two user-directed neutral-fact corrections were applied:", "",
        "- `US_9dc742b4409b555931`: `The Fierles` / `피얼스 부부` was replaced by the existing `[PERSON_A]` and `[PERSON_B]` placeholders.",
        "- `KR_6e91996913ed443be6`: `a Japanese company` / `일본 업체` was replaced by `[COMPANY_F]`.", "",
        "Affected KO/EN/source hashes were recomputed and corresponding fact units were synchronized. Case metadata, domains, splits, all non-target facts, prompts, and source opinions were unchanged. Collection and corpus QC were not rerun; the frozen v4.0.1 canonical QC is referenced by hash and the two edits received deterministic scope/hash/synchronization checks.", "",
        "## Verification", "",
    ]
    report.extend(f"- {key}: **{str(value).upper()}**" for key, value in manifest["invariants"].items())
    output_paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    checksum_paths = list(output_paths.values())
    (DESTINATION / f"SHA256SUMS_{SUFFIX}.txt").write_text(
        "".join(f"{sha_file(path)}  {path.as_posix()}\n" for path in sorted(checksum_paths)), encoding="utf-8"
    )

    source_hashes_after = {name: sha_file(path) for name, path in SOURCE_FILES.items()}
    if source_hashes_after != observed_source_hashes:
        raise RuntimeError("Source v4.0.1 artifacts changed during the freeze")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
