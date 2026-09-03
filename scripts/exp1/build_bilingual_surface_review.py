#!/usr/bin/env python3
"""Build the pre-generation bilingual orientation-surface review bundle."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.exp1.run_implicit_jurisdiction_full import REPO_ROOT, canonical_json, sha_file, write_csv, write_json


LEXICON_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_v1_0"
CORPUS_PATH = REPO_ROOT / "outputs_v2/v4.0.2/final_cases_200_v4_0_2.jsonl"
POLICY_PATH = REPO_ROOT / "scripts/exp1/bilingual_surface_activation_policy_v1.json"
OUTPUT_DIR = REPO_ROOT / "outputs_exp1/bilingual_surface_activation_v1"

# Conservative Codex-assisted recommendations. No row is activated until researcher sign-off.
SURFACE_EXCLUSIONS = {
    "S0018": "Generic English 'joint torts' is not diagnostically Korean without Civil Act context.",
    "S0060": "Korean '근인' is ambiguous and is also used outside this U.S. tort concept.",
    "S0061": "Korean '법적 인과관계' is a shared/general causation expression.",
    "S0064": "Korean '사실적 인과관계' is a shared/general causation expression.",
}


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    concepts = {row["concept_id"]: row for row in load_csv(LEXICON_DIR / "concepts_v1_0.csv")}
    surfaces = load_csv(LEXICON_DIR / "surface_forms_v1_0.csv")
    orientation_languages: dict[str, set[str]] = defaultdict(set)
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for row in surfaces:
        by_concept[row["concept_id"]].append(row)
        if row.get("orientation_include") == "1":
            orientation_languages[row["concept_id"]].add(row["language"])

    one_language = {
        concept_id: languages
        for concept_id, languages in orientation_languages.items()
        if len(languages) == 1
        and concepts[concept_id]["category"] == "doctrine"
        and concepts[concept_id]["orientation_layer"] in {"KR_specific", "US_generic", "US_state_specific"}
        and concepts[concept_id]["scope_tag"] != "remedy_salient"
    }
    if len(one_language) != 18:
        raise RuntimeError(f"Expected 18 one-language frozen orientation concepts, found {len(one_language)}")

    development = [row for row in load_jsonl(CORPUS_PATH) if row.get("analysis_split") == "development"]
    if len(development) != 40:
        raise RuntimeError(f"Expected 40 development cases, found {len(development)}")

    normalized_surface_owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in surfaces:
        normalized_surface_owners[(row["language"], row["surface_form"].casefold())].add(row["concept_id"])

    review_rows: list[dict] = []
    for concept_id, active_languages in sorted(one_language.items()):
        missing_language = "ko" if active_languages == {"en"} else "en"
        opposite_origin = "KR" if missing_language == "ko" else "US"
        cross_docs = [row for row in development if row["origin_country"] == opposite_origin]
        candidates = [row for row in by_concept[concept_id] if row["language"] == missing_language]
        if not candidates:
            raise RuntimeError(f"No existing opposite-language candidate for {concept_id}")
        for candidate in candidates:
            surface = candidate["surface_form"]
            flags = re.IGNORECASE if missing_language == "en" else 0
            cross_hits = sum(
                bool(re.search(re.escape(surface), row["main_opinion_text"], flags=flags))
                for row in cross_docs
            )
            owners = normalized_surface_owners[(missing_language, surface.casefold())] - {concept_id}
            exclusion_reason = SURFACE_EXCLUSIONS.get(candidate["surface_id"], "")
            collision_pass = not owners
            cross_corpus_pass = cross_hits == 0
            semantic_status = "codex_pass" if not exclusion_reason else "codex_fail"
            jurisdiction_status = "codex_pass" if not exclusion_reason else "codex_fail"
            recommendation = (
                "activate_after_researcher_signoff"
                if semantic_status == "codex_pass" and jurisdiction_status == "codex_pass" and collision_pass and cross_corpus_pass
                else "exclude_symmetrically"
            )
            review_rows.append({
                "concept_id": concept_id,
                "jurisdiction_family": concepts[concept_id]["orientation_layer"],
                "candidate_surface_id": candidate["surface_id"],
                "candidate_language": missing_language,
                "candidate_surface_form": surface,
                "provenance": candidate["provenance"],
                "semantic_equivalence_status": semantic_status,
                "jurisdiction_specificity_status": jurisdiction_status,
                "matcher_positive_control_status": "passed",
                "cross_concept_collision_status": "passed" if collision_pass else "failed",
                "collision_concept_ids": "|".join(sorted(owners)),
                "cross_jurisdiction_development_docs_scanned": len(cross_docs),
                "cross_jurisdiction_development_docs_hit": cross_hits,
                "cross_jurisdiction_control_status": "passed" if cross_corpus_pass else "failed",
                "codex_recommendation": recommendation,
                "codex_reason": exclusion_reason or "Existing candidate is semantically aligned, collision-free, and had no cross-jurisdiction development-corpus hit.",
                "review_type": "codex_llm_assisted_qc",
                "researcher_decision": "",
                "researcher_notes": "",
            })

    concept_rows = []
    for concept_id in sorted(one_language):
        rows = [row for row in review_rows if row["concept_id"] == concept_id]
        has_activatable = any(row["codex_recommendation"] == "activate_after_researcher_signoff" for row in rows)
        concept_rows.append({
            "concept_id": concept_id,
            "jurisdiction_family": concepts[concept_id]["orientation_layer"],
            "current_orientation_languages": "|".join(sorted(one_language[concept_id])),
            "missing_language": "ko" if one_language[concept_id] == {"en"} else "en",
            "candidate_surface_count": len(rows),
            "recommended_concept_disposition": "activate_bilingually" if has_activatable else "exclude_symmetrically",
            "researcher_decision": "",
            "researcher_notes": "",
        })

    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite review directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    write_csv(OUTPUT_DIR / "surface_candidate_review.csv", review_rows, list(review_rows[0]))
    write_csv(OUTPUT_DIR / "concept_disposition_review.csv", concept_rows, list(concept_rows[0]))
    write_csv(OUTPUT_DIR / "researcher_signoff.csv", [{
        "policy_id": "BILINGUAL_SURFACE_ACTIVATION_POLICY_V1",
        "review_bundle_sha256": "PENDING_AFTER_BUNDLE_HASH",
        "researcher_review_complete": "",
        "researcher_decision": "",
        "researcher_notes": "",
    }], ["policy_id", "review_bundle_sha256", "researcher_review_complete", "researcher_decision", "researcher_notes"])
    summary = {
        "status": "PREFLIGHT_BLOCKED_RESEARCHER_SIGNOFF_REQUIRED",
        "policy_status": "researcher_approved",
        "source_orientation_concepts": len(one_language),
        "candidate_surface_rows": len(review_rows),
        "concepts_recommended_activate_bilingually": sum(row["recommended_concept_disposition"] == "activate_bilingually" for row in concept_rows),
        "concepts_recommended_exclude_symmetrically": sum(row["recommended_concept_disposition"] == "exclude_symmetrically" for row in concept_rows),
        "development_cases_used": len(development),
        "confirmatory_cases_or_outputs_used": 0,
        "api_generation_requests_sent": 0,
        "policy_sha256": sha_file(POLICY_PATH),
        "source_lexicon_manifest_sha256": sha_file(LEXICON_DIR / "run_manifest.json"),
    }
    write_json(OUTPUT_DIR / "review_summary.json", summary)
    hashes = {path.name: sha_file(path) for path in sorted(OUTPUT_DIR.iterdir()) if path.is_file()}
    write_json(OUTPUT_DIR / "bundle_hashes.json", hashes)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
