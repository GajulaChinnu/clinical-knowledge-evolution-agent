"""Deterministic verdict and answer-brief assembly for a clinician Treatment Check.

The verdict comes from rules over verified comparison results, never from free LLM text.
If no verified citation remains, the verdict is insufficient_grounded_evidence.
"""

from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from app.schemas.clinician_query import ClinicianQuery
from app.schemas.treatment_check import (
    VERDICT_LABELS,
    Citation,
    ClinicianAnswer,
    ComparisonOutcome,
    Finding,
    GovernanceStatus,
    SourceChecked,
    StepRecord,
)
from app.services.taxonomy import Taxonomy

GUIDANCE_SOURCE_TYPES = ("guideline", "safety_notice")


def _src(c: Citation) -> str:
    title = c.source_title or c.source_identity
    when = f", {c.published_date}" if c.published_date else ""
    return f"{title} v{c.version}{when}"


def determine_verdict(query: ClinicianQuery, outcome: ComparisonOutcome) -> Tuple[str, str]:
    """(verdict, basis). Only verified citations from guidelines or safety notices can ground a verdict."""
    verified = [f for f in outcome.findings if f.citation.verified]
    grounded = [
        f for f in verified
        if f.citation.source_type in GUIDANCE_SOURCE_TYPES and set(f.treatments) & set(query.treatment_ids)
        or (f.kind == "contraindication" and f.citation.source_type in GUIDANCE_SOURCE_TYPES)
    ]
    if not grounded:
        return "insufficient_grounded_evidence", (
            "No verified excerpt from a monitored guideline or safety notice covers this treatment."
        )

    applies = [f for f in grounded if f.kind == "contraindication" and f.relation == "applies"]
    if applies:
        f = applies[0]
        return "conflicts_with_latest_guidance", (
            f"A current contraindication applies ({f.applicability_basis}): \"{f.citation.excerpt}\" [{_src(f.citation)}]."
        )

    governing = next((f for f in grounded if f.kind == "governing_recommendation"), None)
    withdrawn = next((f for f in grounded if f.kind == "withdrawal" and f.relation == "withdrawn_matches_plan"), None)
    superseded = next((v for v in outcome.version_changes if v.plan_matches_previous and v.latest and v.latest.verified), None)
    unsupported = [c for c in outcome.plan_components if c.status == "unsupported"]

    if unsupported:
        # Some part of the plan is contradicted by every current recommendation that states it.
        if all(c.superseded_support is not None for c in unsupported):
            parts = "; ".join(
                f"{c.attribute} {c.plan_value.replace('_', ' ')} was stated in version {c.superseded_support.version} "
                f"(\"{c.superseded_support.excerpt}\"), latest says \"{c.contradicted_by[0].excerpt}\" "
                f"[{_src(c.contradicted_by[0])}]"
                for c in unsupported
            )
            return "guidance_updated_follow_new_version", f"The plan follows a superseded version: {parts}."
        parts = "; ".join(
            f"{c.attribute} {c.plan_value.replace('_', ' ')} vs \"{c.contradicted_by[0].excerpt}\" [{_src(c.contradicted_by[0])}]"
            for c in unsupported
        )
        return "conflicts_with_latest_guidance", (
            f"The plan differs from current guidance and does not match any earlier version: {parts}."
        )
    if governing is not None and governing.relation == "matches_plan":
        return "consistent_with_latest_guidance", (
            f"The plan matches the current recommendation \"{governing.citation.excerpt}\" [{_src(governing.citation)}]."
        )
    if withdrawn is not None:
        return "guidance_updated_follow_new_version", (
            f"The plan follows a recommendation that was withdrawn in {_src(withdrawn.citation)}: "
            f"\"{withdrawn.citation.excerpt}\"."
        )
    if governing is not None and governing.relation == "differs_from_plan":
        if superseded is not None:
            return "guidance_updated_follow_new_version", (
                f"The plan matches the superseded version {superseded.previous.version} "
                f"(\"{superseded.previous.excerpt}\"); the latest version {superseded.latest.version} says "
                f"\"{superseded.latest.excerpt}\"."
            )
        diffs = "; ".join(f"{d.kind}: plan {d.before} vs guidance {d.after}" for d in governing.differences)
        return "conflicts_with_latest_guidance", (
            f"The plan differs from current guidance ({diffs}) and does not match any earlier version: "
            f"\"{governing.citation.excerpt}\" [{_src(governing.citation)}]."
        )
    current = [f for f in grounded if f.kind in ("supporting_recommendation", "governing_recommendation")
               and f.relation in ("informational", "matches_plan")]
    if current:
        return "consistent_with_latest_guidance", (
            f"Current guidance recommends this treatment (no comparable dose in the plan): "
            f"\"{current[0].citation.excerpt}\" [{_src(current[0].citation)}]."
        )
    return "insufficient_grounded_evidence", (
        "Monitored sources mention this treatment but contain no current recommendation for the plan."
    )


_ACTIONS = {
    "consistent_with_latest_guidance": [
        "Proceed in line with the cited current recommendation.",
        "Confirm the patient-specific checks listed under 'Check before prescribing'.",
    ],
    "guidance_updated_follow_new_version": [
        "Follow the latest version of the guidance shown under 'Latest guidance'.",
        "Review the plan against the 'What changed' comparison before prescribing.",
        "If the hospital protocol is out of date, note the specialist review status below.",
    ],
    "conflicts_with_latest_guidance": [
        "Do not proceed as planned without reviewing the cited contraindication or difference.",
        "Consider the alternative stated in current guidance, or seek specialist advice.",
        "Document the clinical reasoning if a decision is made to proceed.",
    ],
    "insufficient_grounded_evidence": [
        "No grounded guidance was found in the monitored sources; consult a specialist or the local formulary.",
        "Ask the governance team to add a relevant source to the watchlist if this treatment is in use.",
    ],
}


def build_answer(
    query: ClinicianQuery,
    outcome: Optional[ComparisonOutcome],
    findings: Sequence[Finding],
    sources_checked: Sequence[SourceChecked],
    steps: Sequence[StepRecord],
    taxonomy: Taxonomy,
    governance: Sequence[GovernanceStatus] = (),
) -> ClinicianAnswer:
    now = datetime.now(timezone.utc).isoformat()
    base = dict(
        department=query.department, department_name=query.department_name, treatment=query.plan.text,
        condition=query.condition, sources_checked=list(sources_checked), steps=list(steps),
        governance=list(governance), generated_at=now,
    )
    if outcome is None:
        # Degraded mode: comparison unavailable. Show grounded passages only; no verdict.
        cites = [f.citation for f in findings if f.citation.verified]
        return ClinicianAnswer(
            **base, verdict=None, verdict_label="Retrieved passages only (no verdict)",
            verdict_basis="The comparison step was unavailable, so no verdict was produced.",
            findings=list(findings), citations=cites, degraded_mode=True,
            recommended_actions=["Review the retrieved passages directly; no automated verdict was produced."],
        )

    outcome = outcome.model_copy(update={"findings": list(findings)})
    verdict, basis = determine_verdict(query, outcome)
    verified = [f for f in findings if f.citation.verified]
    governing = next((f for f in verified if f.kind == "governing_recommendation"), None)
    latest = []
    if governing is not None:
        latest.append(governing.citation)
    latest += [f.citation for f in verified if f.kind == "supporting_recommendation" and f.relation == "matches_plan"
               and f.citation.source_type in GUIDANCE_SOURCE_TYPES and f.citation not in latest]
    what_changed = [v for v in outcome.version_changes if (v.latest is None or v.latest.verified)]
    safety = [f for f in verified if f.citation.source_type == "safety_notice"]
    to_check = [f for f in verified if f.kind == "contraindication" and f.relation == "check_applicability"]

    summary: List[str] = [f"{VERDICT_LABELS[verdict]}: {basis}"]
    plan_changes = [v for v in what_changed if v.plan_matches_previous]
    if plan_changes:
        v = plan_changes[0]
        prev = f"v{v.previous.version} ({v.previous.published_date})" if v.previous else "an earlier version"
        new = f"v{v.latest.version} ({v.latest.published_date})" if v.latest else "the latest version"
        summary.append(f"Changed between {prev} and {new}: {v.change_category.replace('_', ' ')}.")
    applied = [f for f in verified if f.kind == "contraindication" and f.relation == "applies"]
    if applied or to_check:
        summary.append(f"Contraindications: {len(applied)} apply, {len(to_check)} to confirm before prescribing.")
    out_of_date = [p for p in outcome.protocol_positions if p.status == "out_of_date"]
    if out_of_date:
        summary.append("Hospital protocol " + ", ".join(f"{p.protocol_id} {p.protocol_version}" for p in out_of_date)
                       + " is out of date with the latest guidance.")
    elif outcome.protocol_positions:
        p = outcome.protocol_positions[0]
        summary.append(f"Hospital protocol {p.protocol_id} {p.protocol_version}: {p.status.replace('_', ' ')}.")

    cited: List[Citation] = []
    for c in [*latest, *[f.citation for f in verified], *[f.previous_citation for f in verified if f.previous_citation]]:
        if c is not None and c.verified and all(c.statement_id != x.statement_id for x in cited):
            cited.append(c)
    if verdict != "insufficient_grounded_evidence" and not cited:
        verdict, basis = "insufficient_grounded_evidence", "No verified citation remained."

    departments = sorted({d for f in verified for d in f.departments} | {query.department})
    pathways = sorted({p for f in verified for p in f.pathways})
    return ClinicianAnswer(
        **base,
        verdict=verdict,
        verdict_label=VERDICT_LABELS[verdict],
        verdict_basis=basis,
        summary=summary[:5],
        latest_guidance=latest,
        what_changed=what_changed,
        safety_notices=safety,
        check_before_prescribing=to_check,
        protocol_positions=list(outcome.protocol_positions),
        findings=list(findings),
        affected_departments=[taxonomy.department_name(d) for d in departments],
        affected_pathways=[taxonomy.pathways[p].name for p in pathways if p in taxonomy.pathways],
        recommended_actions=list(_ACTIONS[verdict]),
        citations=cited,
    )
