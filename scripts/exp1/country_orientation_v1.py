"""Frozen document-level country-orientation score for Experiment 1."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


KR_FAMILY = "KR_specific"
US_FAMILIES = {"generic_US", "state_specific_US"}


def score_country_orientation(
    hits: Iterable[Mapping[str, object]],
    inventory: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Count unique eligible concept IDs and apply (N_US-N_KR)/(N_US+N_KR+1)."""
    eligible_ids: set[str] = set()
    occurrence_counts: Counter[str] = Counter()
    for hit in hits:
        concept_id = str(hit["concept_id"])
        row = inventory.get(concept_id)
        if not row or str(row.get("primary_orientation_eligible", "")).lower() not in {"1", "true"}:
            continue
        eligible_ids.add(concept_id)
        occurrence_counts[concept_id] += 1

    kr_ids = sorted(concept_id for concept_id in eligible_ids if inventory[concept_id]["jurisdiction_family"] == KR_FAMILY)
    us_ids = sorted(concept_id for concept_id in eligible_ids if inventory[concept_id]["jurisdiction_family"] in US_FAMILIES)
    n_kr, n_us = len(kr_ids), len(us_ids)
    return {
        "N_KR": n_kr,
        "N_US": n_us,
        "country_orientation_score": (n_us - n_kr) / (n_us + n_kr + 1),
        "KR_concept_ids": kr_ids,
        "US_concept_ids": us_ids,
        "eligible_occurrence_counts_sensitivity": dict(sorted(occurrence_counts.items())),
    }

