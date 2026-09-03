from scripts.exp1.country_orientation_v1 import score_country_orientation
from scripts.exp1.run_final_pre_generation_qc import (
    FACTS_PATH,
    legal_leakage_review,
    load_human_decisions,
    marker_inventory,
    read_jsonl,
    salience_and_neutralization,
)


def test_country_orientation_counts_unique_concepts_and_excludes_ineligible():
    inventory = {
        "kr": {"primary_orientation_eligible": "true", "jurisdiction_family": "KR_specific"},
        "us": {"primary_orientation_eligible": "true", "jurisdiction_family": "generic_US"},
        "shared": {"primary_orientation_eligible": "false", "jurisdiction_family": "shared"},
    }
    result = score_country_orientation(
        [{"concept_id": "kr"}, {"concept_id": "kr"}, {"concept_id": "us"}, {"concept_id": "shared"}], inventory
    )
    assert result["N_KR"] == 1
    assert result["N_US"] == 1
    assert result["country_orientation_score"] == 0
    assert result["eligible_occurrence_counts_sensitivity"] == {"kr": 2, "us": 1}


def test_frozen_inventory_uses_only_existing_concepts_and_surfaces_coverage_problem():
    rows, summary = marker_inventory()
    assert len(rows) == summary["concept_count"] == 177
    assert summary["eligible_concept_count"] == 16
    assert summary["eligible_missing_bilingual_coverage_count"] == 0
    assert summary["eligible_bilingual_coverage_count"] == 16
    assert summary["inventory_primary_ready"]
    assert all(row["jurisdiction_family"] in {
        "KR_specific", "generic_US", "state_specific_US", "federal", "shared", "remedy_only", "other"
    } for row in rows)


def test_unit_salience_requires_numeric_measurement_not_anatomy_or_place():
    facts = [{"case_id": "anatomy", "neutral_fact_ko": "", "neutral_fact_en": "His foot was injured in the yard."}]
    salience, neutralization, _ = salience_and_neutralization(facts)
    assert salience[0]["jurisdiction_salience_level"] == "none"
    assert neutralization[0]["unit_normalization_status"] == "not_applicable"

    facts = [{"case_id": "measurement", "neutral_fact_ko": "", "neutral_fact_en": "The distance was 10 feet."}]
    salience, neutralization, _ = salience_and_neutralization(facts)
    assert "miles_feet" in salience[0]["jurisdiction_salience_tags"]
    assert neutralization[0]["unit_normalization_status"] == "review_required"


def test_confirmatory_legal_leakage_review_requires_two_human_decisions():
    facts = [row for row in read_jsonl(FACTS_PATH) if row["analysis_split"] == "confirmatory"]
    rows, statuses, unresolved = legal_leakage_review(facts)
    assert unresolved == {"KR_0dcfdd46d74a8951e1", "US_f9417e4d262285b1c6"}
    assert sum(row["severity"] == "critical" for row in rows) == 6
    assert sum(status == "passed" for status in statuses.values()) == 158


def test_researcher_decisions_resolve_only_the_two_flagged_cases():
    facts = [row for row in read_jsonl(FACTS_PATH) if row["analysis_split"] == "confirmatory"]
    decisions, payload = load_human_decisions()
    rows, statuses, unresolved = legal_leakage_review(facts, decisions)
    assert payload["decision_source"] == "researcher_user_instruction_current_session"
    assert unresolved == set()
    assert statuses["KR_0dcfdd46d74a8951e1"] == "approved_retain_as_is"
    assert statuses["US_f9417e4d262285b1c6"] == "approved_retain_as_is"
    assert sum(row["recommended_action"] == "retain_as_is_researcher_approved" for row in rows) == 6
