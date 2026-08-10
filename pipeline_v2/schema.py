from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Domain = Literal[
    "general_negligence_personal_injury",
    "medical_professional_liability",
    "product_liability",
    "employer_supervisory_vicarious_liability",
    "other_civil_liability",
]
EpistemicStatus = Literal["established_record_fact", "party_allegation", "testimony", "disputed_fact"]


@dataclass(slots=True)
class FactUnit:
    fact_id: str
    text: str
    source_level: Literal["highest_court", "lower_court"]
    source_case_id: str
    source_span: str
    source_start: int | None
    source_end: int | None
    fact_type: Literal["parties", "conduct", "context", "timeline", "harm", "causation", "defense_context", "other"]
    epistemic_status: EpistemicStatus
    include_in_neutral_fact: bool
    exclusion_reason: str | None = None
    source_grounding_status: Literal["pending", "pass", "fail"] = "pending"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class FactExtraction:
    case_id: str
    source_language: Literal["ko", "en"]
    fact_units: list[FactUnit] = field(default_factory=list)
    entity_mapping: dict[str, str] = field(default_factory=dict)
    neutral_fact_source: str = ""
    unit_normalization_status: str = "not_needed"
    institution_neutralization_status: str = "not_needed"


FACT_CATEGORIES = (
    "fact_has_parties", "fact_has_conduct", "fact_has_context", "fact_has_timeline",
    "fact_has_harm", "fact_has_causation", "fact_has_defense_context",
)

DOMAIN_TARGET = {
    "general_negligence_personal_injury": 40,
    "medical_professional_liability": 20,
    "product_liability": 20,
    "employer_supervisory_vicarious_liability": 20,
}

