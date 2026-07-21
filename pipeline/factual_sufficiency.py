from __future__ import annotations

import re
from typing import Any


FACT_PATTERNS: dict[str, list[str]] = {
    "party_relationship": [r"plaintiff.{0,120}defendant", r"employee.{0,100}employer", r"patient.{0,100}(?:doctor|physician|hospital)", r"owner.{0,100}(?:tenant|visitor|customer)"],
    "specific_conduct_or_omission": [r"defendant.{0,180}(?:failed|drove|struck|published|designed|manufactured|treated|performed|allowed)", r"failed to (?:warn|maintain|protect|supervise|diagnose|treat)", r"breach(?:ed)? (?:a |the )?duty"],
    "injury_producing_event": [r"accident|collision|incident|attack|fall occurred|(?:procedure|surgery).{0,100}(?:failed|caused|resulted)", r"failed to (?:diagnose|treat)", r"was (?:struck|injured|killed|assaulted)"],
    "chronology": [r"\b(?:18|19|20)\d{2}\b", r"\b(?:before|after|later|then|subsequently|earlier|the next|during)\b"],
    "harm": [r"personal injur|bodily injur|physical injur|property damage|emotional distress|medical expenses|lost wages|died|death"],
    "causation_relevant": [r"caus(?:e|ed|ation)|proximate cause|substantial factor|result(?:ed|ing) in|because of"],
    "defense_or_dispute": [r"defendant (?:denied|argued|contended|claimed)|comparative (?:fault|negligence)|assumption of risk|disputed|affirmative defense|plaintiff contends"],
}


def _evidence(text: str, patterns: list[str]) -> list[str]:
    found = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            found.append(re.sub(r"\s+", " ", match.group(0))[:220])
    return list(dict.fromkeys(found))


def classify_procedural_posture(text: str) -> tuple[str, list[str]]:
    checks = [
        ("anti_slapp", [r"anti-SLAPP", r"special motion to strike"]),
        ("demurrer_or_motion_to_dismiss", [r"demurrer", r"motion to dismiss", r"failure to state a cause of action"]),
        ("summary_judgment", [r"summary judgment", r"summary adjudication"]),
        ("judgment_notwithstanding_verdict", [r"judgment notwithstanding the verdict", r"\bJNOV\b"]),
        ("new_trial_motion", [r"motion for (?:a )?new trial"]),
        ("post_trial_appeal", [r"jury (?:returned|found|verdict)", r"following (?:a )?(?:jury|bench) trial", r"trial court found"]),
        ("interlocutory_or_procedural", [r"interlocutory", r"appealability", r"attorney fees? only"]),
    ]
    for label, patterns in checks:
        evidence = _evidence(text[:30000], patterns)
        if evidence:
            return label, evidence
    return "unknown", []


def classify_fact_status(text: str, procedural_posture: str) -> tuple[str, list[str]]:
    checks = [
        ("assumed_true_for_pleading", [r"accept(?:ed)? as true", r"assume(?:d)?.{0,40}(?:pleaded|alleged) facts?", r"for purposes of (?:the )?demurrer", r"facts alleged in the complaint"]),
        ("jury_found_fact", [r"jury found", r"jury returned a verdict", r"special verdict"]),
        ("trial_court_found_fact", [r"trial court found", r"court found that", r"statement of decision"]),
        ("undisputed_fact", [r"undisputed facts?", r"it is undisputed"]),
        ("record_evidence", [r"the record shows", r"evidence at trial", r"deposition testimony"]),
        ("disputed_fact", [r"disputed fact", r"parties dispute", r"conflicting evidence"]),
        ("plaintiff_allegation", [r"plaintiff alleges?", r"complaint alleges?"])]
    for label, patterns in checks:
        evidence = _evidence(text[:30000], patterns)
        if evidence:
            return label, evidence
    if procedural_posture == "demurrer_or_motion_to_dismiss":
        return "assumed_true_for_pleading", ["pleading-stage allegations"]
    return "unclear", []


def assess_factual_sufficiency(text: str, *, full_main_opinion_available: bool) -> dict[str, Any]:
    value = text[:50000]
    evidence: list[str] = []
    categories: list[str] = []
    for category, patterns in FACT_PATTERNS.items():
        hits = _evidence(value, patterns)
        if hits:
            categories.append(category)
            evidence.extend(f"{category}:{hit}" for hit in hits[:2])

    fact_heading = bool(re.search(r"(?im)^\s*(?:background|factual background|facts|statement of facts|relevant facts)\s*$", value))
    procedure_terms = len(re.findall(r"appeal|motion|order|judgment|standard of review|trial court", value, re.I))
    fact_terms = sum(len(re.findall(pattern, value, re.I)) for patterns in FACT_PATTERNS.values() for pattern in patterns)
    score = min(100, len(categories) * 12 + (8 if fact_heading else 0) + (8 if len(value) >= 5000 else 0))
    reasons: list[str] = []
    if not full_main_opinion_available:
        reasons.append("full_main_opinion_unavailable")
    required = {"party_relationship", "specific_conduct_or_omission", "injury_producing_event", "chronology", "harm", "causation_relevant"}
    missing = sorted(required - set(categories))
    if missing:
        reasons.extend(f"missing_{item}" for item in missing)
    if "defense_or_dispute" not in categories:
        reasons.append("missing_defense_or_disputed_facts")
    if procedure_terms > max(30, fact_terms * 4) and not fact_heading:
        reasons.append("procedural_or_legal_discussion_dominates")
    sufficient = full_main_opinion_available and not missing and "defense_or_dispute" in categories and score >= 80
    quality = "self_contained" if sufficient and fact_heading else ("partially_incorporated" if sufficient else "insufficient")
    return {
        "factual_background_sufficient": sufficient,
        "facts_independently_reconstructable": sufficient,
        "fact_source_quality": quality,
        "factual_sufficiency_score": score,
        "factual_sufficiency_evidence": evidence[:24],
        "factual_insufficiency_reasons": list(dict.fromkeys(reasons)),
    }
