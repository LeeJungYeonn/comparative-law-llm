from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from exp1.common import DEFAULT_INPUT, REPO_ROOT, read_jsonl, sha256_file, write_csv, write_json
from exp1.conclusion_v2 import (
    PROMPT_VERSION, SCHEMA_PATH, SYSTEM_PROMPT_PATH, USER_PROMPT_PATH, VERSION,
    build_party_registry, legacy_conclusion_audit,
)


def protected_paths(input_path: Path, output_dir: Path, reports_dir: Path) -> list[Path]:
    paths = [
        input_path,
        output_dir / "raw_responses.jsonl",
        output_dir / "evaluations.jsonl",
        output_dir / "pair_metrics.csv",
        output_dir / "summary.json",
        reports_dir / "exp1_results.md",
    ]
    paths.extend(sorted((output_dir / "graphs").glob("*")))
    return [path for path in paths if path.is_file()]


def audit(input_path: Path, output_dir: Path, reports_dir: Path) -> dict:
    accepted = [row for row in read_jsonl(input_path) if row.get("case_is_finally_usable") is True]
    evaluations = read_jsonl(output_dir / "evaluations.jsonl")
    legacy, legacy_rows = legacy_conclusion_audit(evaluations)
    expected = {
        "exact_string_matched_comparisons": 432,
        "agreement": 0.5787037037037037,
        "cohen_kappa_unweighted": 0.4244385230300724,
        "any_conclusion_change_cases": 63,
        "direct_likely_unlikely_flip_cases": 2,
        "modal_tie_matched_comparisons": 107,
    }
    differences = {
        key: {"expected": value, "observed": legacy.get(key)}
        for key, value in expected.items()
        if abs(float(legacy.get(key, float("nan"))) - float(value)) > 1e-12
    }
    legacy["status"] = "reproduced" if not differences else "not_reproduced"
    legacy["differences"] = differences

    registry = build_party_registry(accepted, evaluations)
    write_csv(output_dir / "party_registry_v2.csv", registry, list(registry[0]))
    write_csv(
        output_dir / "legacy_conclusion_comparisons_v2.csv",
        legacy_rows,
        list(legacy_rows[0]),
    )
    flags = Counter(
        flag
        for row in registry
        for flag in str(row["audit_flags"]).split(";")
        if flag
    )
    hashes = {
        str(path.resolve()): sha256_file(path)
        for path in protected_paths(input_path, output_dir, reports_dir)
    }
    manifest = {
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "legacy_audit": legacy,
        "party_registry": {
            "rows": len(registry),
            "cases": len({row["case_id"] for row in registry}),
            "source_party_set_mismatch_cases": len({
                row["case_id"] for row in registry if row["source_party_set_mismatch"]
            }),
            "audit_flag_counts": dict(flags),
            "party_type_counts": dict(Counter(row["party_type"] for row in registry)),
        },
        "protected_file_hashes_before": hashes,
        "prompt_hashes": {
            SYSTEM_PROMPT_PATH.name: sha256_file(SYSTEM_PROMPT_PATH),
            USER_PROMPT_PATH.name: sha256_file(USER_PROMPT_PATH),
        },
        "schema_sha256": sha256_file(SCHEMA_PATH),
    }
    write_json(output_dir / "conclusion_reanalysis_manifest_v2.json", manifest)
    write_json(output_dir / "legacy_conclusion_audit_v2.json", legacy)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = audit(args.input, args.output_dir, args.reports_dir)
    print(
        f"legacy={result['legacy_audit']['status']} "
        f"registry_rows={result['party_registry']['rows']} "
        f"source_mismatch_cases={result['party_registry']['source_party_set_mismatch_cases']}"
    )


if __name__ == "__main__":
    main()
