from __future__ import annotations

import re
from typing import Any


TORT_SUBTYPES = (
    "traffic_accident", "medical_professional", "employer_vicarious_liability",
    "premises_facility_safety", "privacy_reputation", "intentional_tort",
    "product_safety", "property_damage", "general_personal_injury", "other_tort", "unclear",
)


PATTERNS: dict[str, list[str]] = {
    "traffic_accident": [r"motor vehicle", r"automobile", r"traffic accident", r"car accident", r"vehicle collision", r"pedestrian.{0,80}(?:struck|vehicle)"],
    "medical_professional": [r"medical malpractice", r"professional negligence", r"legal malpractice", r"malpractice", r"physician|surgeon|hospital|attorney.{0,50}neglig"],
    "employer_vicarious_liability": [r"respondeat superior", r"scope of employment", r"vicarious liability", r"negligent (?:hiring|supervision|retention)", r"employer.{0,80}(?:employee|neglig)"],
    "premises_facility_safety": [r"premises liability", r"dangerous condition", r"inadequate security", r"failure to maintain", r"slip and fall", r"property owner.{0,80}(?:duty|condition)"],
    "privacy_reputation": [r"defamation", r"\blibel\b", r"\bslander\b", r"invasion of privacy", r"false light", r"public disclosure of private"],
    "intentional_tort": [r"\bbattery\b", r"\bassault\b", r"intentional infliction of emotional distress", r"false imprisonment", r"intentional tort"],
    "product_safety": [r"product(?:s)? liability", r"design defect", r"manufacturing defect", r"failure to warn", r"defective product", r"strict liability"],
    "property_damage": [r"property damage", r"damage to (?:the )?(?:property|building|land)", r"trespass", r"private nuisance", r"conversion"],
    "general_personal_injury": [r"personal injur", r"bodily injur", r"physical injur", r"negligent infliction of emotional distress"],
    "other_tort": [r"\bnegligence\b", r"\btort\b", r"wrongful death", r"fraudulent concealment"],
}


def _count(value: str, patterns: list[str]) -> tuple[int, list[str]]:
    evidence = []
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            evidence.append(match.group(0)[:120])
    return len(evidence), evidence


def classify_tort_subtype(text: str) -> tuple[str, list[str]]:
    value = text[:30000]
    scored: list[tuple[int, int, str, list[str]]] = []
    # Earlier entries win ties; a specific theory needs more weight than generic injury language.
    for priority, (subtype, patterns) in enumerate(PATTERNS.items()):
        count, evidence = _count(value, patterns)
        specificity = 0 if subtype in {"general_personal_injury", "other_tort"} else 2
        scored.append((count * 3 + (specificity if count else 0), -priority, subtype, evidence))
    score, _, subtype, evidence = max(scored)
    if score <= 0:
        return "unclear", []
    return subtype, evidence


def classify_harm_flags(text: str) -> dict[str, bool]:
    value = text[:30000]
    return {
        "death_involved": bool(re.search(r"wrongful death|\bdied\b|\bdeath\b|fatal", value, re.I)),
        "physical_injury_involved": bool(re.search(r"personal injur|bodily injur|physical injur|medical treatment|hospitali[sz]ed", value, re.I)),
        "property_damage_involved": bool(re.search(r"property damage|damage to (?:the )?(?:property|vehicle|building|land)|economic loss", value, re.I)),
        "emotional_harm_involved": bool(re.search(r"emotional distress|mental anguish|reputational harm|humiliation", value, re.I)),
    }
