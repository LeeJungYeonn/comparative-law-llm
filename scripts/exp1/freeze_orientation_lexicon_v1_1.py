#!/usr/bin/env python3
"""Apply signed bilingual-surface decisions and freeze orientation lexicon v1.1."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from scripts.exp1.run_implicit_jurisdiction_full import REPO_ROOT, canonical_json, sha_file, write_csv, write_json


SOURCE_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_v1_0"
REVIEW_DIR = REPO_ROOT / "outputs_exp1/bilingual_surface_activation_v1"
POLICY_PATH = REPO_ROOT / "scripts/exp1/bilingual_surface_activation_policy_v1.json"
DECISIONS_PATH = REPO_ROOT / "scripts/exp1/bilingual_surface_decisions_v1.json"
OUTPUT_DIR = REPO_ROOT / "legal_lexicon/legal_lexicon_orientation_v1_1"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    decision = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    for name, expected in decision["review_artifacts"].items():
        actual = sha_file(REVIEW_DIR / name)
        if actual != expected:
            raise RuntimeError(f"Signed review hash mismatch for {name}: {actual} != {expected}")
    if decision["decision"] != "approve_all_codex_recommendations_in_frozen_review_bundle":
        raise RuntimeError("Researcher sign-off does not approve the frozen recommendations")

    concepts = load_csv(SOURCE_DIR / "concepts_v1_0.csv")
    surfaces = load_csv(SOURCE_DIR / "surface_forms_v1_0.csv")
    surface_review = load_csv(REVIEW_DIR / "surface_candidate_review.csv")
    concept_review = load_csv(REVIEW_DIR / "concept_disposition_review.csv")
    surface_recommendation = {row["candidate_surface_id"]: row for row in surface_review}
    concept_disposition = {row["concept_id"]: row["recommended_concept_disposition"] for row in concept_review}

    activated_surface_ids: list[str] = []
    symmetrically_excluded_concepts = sorted(
        concept_id for concept_id, disposition in concept_disposition.items()
        if disposition == "exclude_symmetrically"
    )
    for row in surfaces:
        concept_id = row["concept_id"]
        if concept_id in symmetrically_excluded_concepts:
            row["orientation_include"] = "0"
            row["note"] = (row.get("note", "") + " Exp1 orientation v1.1: symmetrically excluded by approved bilingual policy.").strip()
        review = surface_recommendation.get(row["surface_id"])
        if review and review["codex_recommendation"] == "activate_after_researcher_signoff":
            row["primary_include"] = "1"
            row["orientation_include"] = "1"
            row["status"] = "researcher_approved_bilingual_activation"
            row["version"] = "orientation_v1.1"
            row["note"] = (row.get("note", "") + " Activated under researcher-approved BILINGUAL_SURFACE_ACTIVATION_POLICY_V1.").strip()
            activated_surface_ids.append(row["surface_id"])

    orientation_languages: dict[str, set[str]] = defaultdict(set)
    for row in surfaces:
        if row["orientation_include"] == "1":
            orientation_languages[row["concept_id"]].add(row["language"])
    one_sided = {concept_id: sorted(languages) for concept_id, languages in orientation_languages.items() if len(languages) != 2}
    if one_sided:
        raise RuntimeError({"one_sided_orientation_concepts": one_sided})
    if len(orientation_languages) != 16:
        raise RuntimeError(f"Expected 16 bilingual orientation concepts, found {len(orientation_languages)}")
    family = {row["concept_id"]: row["orientation_layer"] for row in concepts}
    family_counts = {
        name: sum(family[concept_id] == name for concept_id in orientation_languages)
        for name in ("KR_specific", "US_generic", "US_state_specific")
    }
    if family_counts != {"KR_specific": 3, "US_generic": 13, "US_state_specific": 0}:
        raise RuntimeError({"unexpected_orientation_family_counts": family_counts})

    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite frozen orientation lexicon: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    write_csv(OUTPUT_DIR / "concepts_v1_1.csv", concepts, list(concepts[0]))
    write_csv(OUTPUT_DIR / "surface_forms_v1_1.csv", surfaces, list(surfaces[0]))
    active = [row for row in surfaces if row["primary_include"] == "1"]
    write_csv(OUTPUT_DIR / "lexicon_active_primary_v1_1.csv", active, list(active[0]))
    shutil.copy2(SOURCE_DIR / "lexicon_matcher_v1_0.py", OUTPUT_DIR / "lexicon_matcher_v1_1.py")
    write_csv(OUTPUT_DIR / "researcher_signoff_applied.csv", [{
        "decision_id": decision["decision_id"],
        "decision_date": decision["decision_date"],
        "decision_source": decision["decision_source"],
        "decision": decision["decision"],
        "policy_sha256": sha_file(POLICY_PATH),
        "decisions_sha256": sha_file(DECISIONS_PATH),
        "concept_review_sha256": sha_file(REVIEW_DIR / "concept_disposition_review.csv"),
        "surface_review_sha256": sha_file(REVIEW_DIR / "surface_candidate_review.csv"),
    }], [
        "decision_id", "decision_date", "decision_source", "decision", "policy_sha256",
        "decisions_sha256", "concept_review_sha256", "surface_review_sha256",
    ])
    manifest = {
        "version": "legal-lexicon-orientation-v1.1",
        "status": "FROZEN_PRE_GENERATION",
        "purpose": "Experiment 1 bilingual country-orientation measurement",
        "source_lexicon": "legal_lexicon_v1_0",
        "source_lexicon_manifest_sha256": sha_file(SOURCE_DIR / "run_manifest.json"),
        "policy_sha256": sha_file(POLICY_PATH),
        "researcher_decisions_sha256": sha_file(DECISIONS_PATH),
        "bilingual_orientation_concept_count": len(orientation_languages),
        "orientation_family_counts": family_counts,
        "activated_opposite_language_surface_ids": sorted(activated_surface_ids),
        "symmetrically_excluded_concept_ids": symmetrically_excluded_concepts,
        "one_sided_orientation_concept_count": 0,
        "confirmatory_cases_or_outputs_used_for_activation": 0,
        "api_generation_requests_sent": 0,
    }
    artifact_names = [
        "concepts_v1_1.csv", "surface_forms_v1_1.csv", "lexicon_active_primary_v1_1.csv",
        "lexicon_matcher_v1_1.py", "researcher_signoff_applied.csv",
    ]
    manifest["artifact_sha256"] = {name: sha_file(OUTPUT_DIR / name) for name in artifact_names}
    write_json(OUTPUT_DIR / "run_manifest.json", manifest)
    (OUTPUT_DIR / "README.md").write_text(
        "# Legal lexicon orientation v1.1\n\n"
        "Frozen pre-generation derivative of legal_lexicon_v1_0. It changes only the bilingual "
        "surface activation used for Experiment 1 country-orientation measurement. The source v1.0 remains immutable.\n",
        encoding="utf-8", newline="\n",
    )
    checksum_paths = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (OUTPUT_DIR / "SHA256SUMS.txt").write_text(
        "".join(f"{sha_file(path)}  {path.name}\n" for path in checksum_paths), encoding="ascii", newline="\n"
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
