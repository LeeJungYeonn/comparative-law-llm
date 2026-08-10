from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from exp1.common import (
    DEFAULT_INPUT, prompt_hashes, sha256_text, smoke_case_ids, usable_cases, write_csv, write_json,
)


def audit(input_path: Path, output_dir: Path) -> dict:
    rows = usable_cases(input_path)
    smoke = set(smoke_case_ids(rows))
    audit_rows = []
    for row in rows:
        ko, en = str(row["neutral_fact_ko"]), str(row["neutral_fact_en"])
        audit_rows.append({
            "case_id": row["case_id"],
            "case_origin": row["case_origin"],
            "case_subtype": row["case_subtype"],
            "source_language": row.get("source_language", ""),
            "master_language": row.get("master_language", ""),
            "translation_direction": row.get("translation_direction", ""),
            "ko_chars": len(ko),
            "en_chars": len(en),
            "ko_sha256": sha256_text(ko),
            "en_sha256": sha256_text(en),
            "ko_placeholder_count": ko.count("["),
            "en_placeholder_count": en.count("["),
            "smoke_selected": row["case_id"] in smoke,
            "audit_status": "pass",
        })
    fields = list(audit_rows[0]) if audit_rows else []
    write_csv(output_dir / "preflight_audit.csv", audit_rows, fields)
    summary = {
        "input_rows": len(rows),
        "origin_counts": dict(Counter(r["case_origin"] for r in rows)),
        "subtype_counts": dict(Counter(r["case_subtype"] for r in rows)),
        "smoke_case_ids": sorted(smoke),
        "prompt_hashes": prompt_hashes(),
        "status": "pass" if len(rows) == 70 and Counter(r["case_origin"] for r in rows) == {"KR": 35, "CA": 35} else "warning",
    }
    write_json(output_dir / "preflight_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp1"))
    args = parser.parse_args()
    result = audit(args.input, args.output_dir)
    print(f"preflight={result['status']} rows={result['input_rows']} origins={result['origin_counts']}")


if __name__ == "__main__":
    main()
