from __future__ import annotations

import re
from typing import Any


POSTURE_LABELS = {
    "direct_tort_claim",
    "direct_action_against_liability_insurer",
    "declaratory_nonliability_action",
    "insurer_subrogation",
    "joint_tortfeasor_contribution",
    "insurance_coverage",
    "contract_or_payment",
    "wage_or_compensation",
    "judgment_enforcement",
    "property_or_title_dispute",
    "family_or_domestic_dispute",
    "administrative_or_state_liability",
    "civil_rights_only",
    "procedural_only",
    "unclear",
}


def _hits(text: str, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            found.append(match.group(0)[:160])
    return list(dict.fromkeys(found))


TORT_THEORY = [
    r"\bneglig(?:ence|ent)\b", r"\btort(?:ious)?\b", r"wrongful death",
    r"premises liability", r"product(?:s)? liability", r"medical malpractice",
    r"professional negligence", r"respondeat superior", r"vicarious liability",
    r"\bdefamation\b", r"\blibel\b", r"\bslander\b", r"invasion of privacy",
    r"intentional infliction of emotional distress", r"\bbattery\b", r"\bassault\b",
    r"\btrespass\b", r"\bnuisance\b", r"negligent (?:hiring|supervision|retention)",
]
CONDUCT = [
    r"breach(?:ed)? (?:a |the )?duty", r"duty of care", r"failed to (?:warn|maintain|protect|supervise|diagnose|treat)",
    r"drove|driving|driver", r"struck|collided|collision", r"manufactur(?:ed|er)|design defect|failure to warn",
    r"published|stated|represented", r"attacked|assaulted|battered|shot|stabbed",
]
HARM = [
    r"personal injur", r"bodily injur", r"physical injur", r"property damage", r"emotional distress",
    r"suffered (?:injur|harm|damage)", r"died|death", r"medical expenses", r"lost wages", r"damages",
]
CAUSATION = [r"proximate cause", r"caus(?:e|ed|ation)", r"result(?:ed|ing) in", r"because of", r"substantial factor"]
PLAINTIFF_DIRECT = [
    r"plaintiff[s]? (?:sued|filed|brought|alleged|claimed|seeks?|sought|recovered)",
    r"plaintiff[s]? suffered (?:injur|harm|damage)", r"(?:negligence|tort|malpractice) claim",
    r"complaint (?:alleged|asserted|for)", r"surviv(?:or|ing spouse|ing child)", r"heirs? (?:of|sued)",
]


def classify_claim_posture(text: str, *, case_name: str = "", metadata: str = "") -> dict[str, Any]:
    value = f"{case_name}\n{metadata}\n{text[:30000]}"

    direct_insurer = _hits(value, [
        r"direct action against (?:the )?(?:liability )?insurer", r"Insurance Code section 11580",
        r"judgment creditor.{0,120}(?:insurer|insurance company)", r"injured (?:plaintiff|party).{0,120}sued (?:the )?insurer",
    ])
    declaratory_nonliability = _hits(value, [
        r"declaratory (?:judgment|relief).{0,180}(?:no|not) (?:liable|liability|duty)",
        r"declaration that .{0,120}(?:owed no|has no) (?:liability|duty)",
    ])
    subrogation = _hits(value, [
        r"\bsubrogat(?:e|ed|ion)\b", r"\bsubrogee\b", r"insurer.{0,160}(?:paid|payment).{0,160}(?:recover|reimbursement)",
        r"insurance company.{0,160}(?:as subrogee|reimbursement)",
    ])
    contribution = _hits(value, [
        r"equitable indemnity", r"comparative indemnity", r"contribution (?:from|among|between)",
        r"cross-complaint for indemnity", r"joint tortfeasor.{0,120}(?:contribution|indemnity)",
    ])
    coverage = _hits(value, [
        r"duty to defend", r"duty to indemnify", r"insurance coverage", r"coverage dispute",
        r"policy exclusion", r"insurance polic(?:y|ies).{0,120}(?:coverage|exclusion|insured)",
    ])
    wage = _hits(value, [
        r"unpaid wages", r"minimum wage", r"overtime pay", r"wage and hour", r"Labor Code.{0,80}(?:wage|overtime)",
        r"salary|severance pay|employee compensation",
    ])
    enforcement = _hits(value, [
        r"enforce(?:ment)? of (?:the )?judgment", r"judgment debtor", r"judgment creditor",
        r"writ of execution", r"postjudgment", r"collection of (?:the )?judgment", r"charging order",
    ])
    contract = _hits(value, [
        r"breach of contract", r"contractual obligation", r"purchase agreement", r"lease agreement",
        r"loan agreement", r"promissory note", r"unpaid (?:fee|rent|invoice)", r"amount due under",
        r"specific performance", r"contract damages",
    ])
    title = _hits(value, [
        r"quiet title", r"foreclosure", r"deed of trust", r"title to (?:the )?(?:property|land)",
        r"partition action", r"unlawful detainer",
    ])
    family = _hits(value, [r"dissolution of marriage", r"child custody", r"child support", r"probate", r"decedent['’]s estate", r"trust beneficiary"])
    administrative = _hits(value, [r"workers['’] compensation", r"administrative mandamus", r"agency decision", r"Public Utilities Commission"])
    civil_rights = _hits(value, [r"42 U\.?S\.?C\.? ?§? ?1983", r"section 1983", r"civil rights action", r"constitutional violation"])
    procedural = _hits(value, [
        r"appealability", r"attorney fees? (?:award|order)", r"statute of limitations", r"lack of jurisdiction",
        r"motion to disqualify", r"discovery sanction", r"special motion to strike",
    ])
    tort = _hits(value, TORT_THEORY)
    conduct = _hits(value, CONDUCT)
    harm = _hits(value, HARM)
    causation = _hits(value, CAUSATION)
    plaintiff = _hits(value, PLAINTIFF_DIRECT)
    direct_evidence = list(dict.fromkeys(tort + conduct + harm + causation + plaintiff))
    direct_groups = sum(bool(group) for group in (tort, conduct, harm, plaintiff))

    # Exclusionary current-claim postures take precedence over incidental tort vocabulary.
    ordered = [
        ("direct_action_against_liability_insurer", direct_insurer),
        ("declaratory_nonliability_action", declaratory_nonliability),
        ("insurer_subrogation", subrogation),
        ("joint_tortfeasor_contribution", contribution),
        ("insurance_coverage", coverage),
        ("wage_or_compensation", wage),
        ("judgment_enforcement", enforcement),
        ("property_or_title_dispute", title),
        ("family_or_domestic_dispute", family),
        ("administrative_or_state_liability", administrative),
    ]
    for label, evidence in ordered:
        if evidence:
            # Incidental insurance background does not displace a clearly pleaded
            # direct victim-versus-tortfeasor damages claim.
            if label == "insurance_coverage" and direct_groups >= 4 and len(evidence) < 2:
                continue
            return {"claim_posture": label, "confidence": "high" if len(evidence) >= 2 else "medium", "evidence": evidence}

    # A direct claim needs tort theory plus independently visible conduct/harm and a plaintiff-side claim signal.
    if direct_groups >= 4 and (causation or len(tort) >= 2):
        confidence = "high" if causation and len(direct_evidence) >= 6 else "medium"
        return {"claim_posture": "direct_tort_claim", "confidence": confidence, "evidence": direct_evidence[:20]}

    if contract and not (conduct and harm and plaintiff):
        return {"claim_posture": "contract_or_payment", "confidence": "high" if len(contract) >= 2 else "medium", "evidence": contract}
    if civil_rights and not tort:
        return {"claim_posture": "civil_rights_only", "confidence": "medium", "evidence": civil_rights}
    if procedural and not (conduct and harm):
        return {"claim_posture": "procedural_only", "confidence": "medium", "evidence": procedural}
    return {"claim_posture": "unclear", "confidence": "low", "evidence": direct_evidence[:20]}


def classify_liability_basis(text: str, claim_posture: str) -> tuple[str, list[str]]:
    value = text[:30000]
    contract = _hits(value, [r"breach of contract", r"contractual duty", r"arising under (?:the )?(?:agreement|contract)"])
    tort = _hits(value, TORT_THEORY)
    if claim_posture == "direct_tort_claim":
        if contract and len(contract) >= 2:
            return "mixed_tort_contract", contract + tort[:5]
        return "non_contractual_tort", tort[:10]
    if claim_posture in {"insurance_coverage", "direct_action_against_liability_insurer", "insurer_subrogation"}:
        return "insurance_only", []
    if claim_posture in {"contract_or_payment", "wage_or_compensation"}:
        return "contract_only", contract
    if claim_posture in {"judgment_enforcement", "procedural_only", "declaratory_nonliability_action"}:
        return "procedural_only", []
    if claim_posture == "administrative_or_state_liability":
        return "administrative_or_public_law", []
    if claim_posture == "civil_rights_only":
        return "civil_rights_only", []
    if claim_posture == "family_or_domestic_dispute":
        return "family_or_probate", []
    return "unclear", tort[:10]
