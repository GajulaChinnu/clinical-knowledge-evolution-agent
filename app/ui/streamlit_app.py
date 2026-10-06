"""Clinical Knowledge Evolution Agent (CKEA) - Streamlit Governance & Demonstration UI.

Phase 13: Final End-to-End Application Integration & Human Governance Interface.

CORE SAFETY PRINCIPLE:
"System prepares evidence and review material. Authorized clinicians make the final decision."
The application NEVER makes automated clinical decisions, approves, rejects, defers, or closes
briefs without explicit human action.
"""

from datetime import datetime, timezone, timedelta
import html
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path for direct `streamlit run` execution
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st
from sqlalchemy import desc, select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.governance_agent import GovernanceAgent
from app.models.database import get_engine, get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    Notification,
    ReviewAssignment,
)
from app.schemas.briefs import BriefStatus, StructuredBriefPayload, validate_brief_completeness
from app.schemas.changes import ChangeStatus
from app.schemas.documents import DocumentStatus
from app.schemas.gaps import DifferenceType, GapStatus
from app.schemas.governance import (
    ReviewAssignmentStatus,
    ReviewDecision,
    UnauthorizedReviewerError,
)
from app.schemas.impact import ImpactStatus, ImpactTier
from app.schemas.orchestration import HumanGate
from app.services.brief_renderer import BriefRenderer
from app.services.config_service import AppConfig, load_config
from app.services.reviewer_authorization import ReviewerAuthorizationService

logger = logging.getLogger("ckea.ui.streamlit_app")

# Authorized reviewers available for selection
AUTHORIZED_REVIEWERS = [
    "dr_smith",
    "dr_jones",
    "cmo_director",
    "pharmacy_lead",
    "clinical_reviewer_1",
    "clinical_reviewer_2",
    "rapid_governance_chair",
    "dr_wilson",
]


# ==============================================================================
# DATA ACCESS & LOGIC LAYER (Stateless, Isolated, Deterministic)
# ==============================================================================

def get_db_session_factory(config: Optional[AppConfig] = None) -> sessionmaker[Session]:
    """Retrieve session factory using application configuration."""
    cfg = config or load_config()
    engine = get_engine(db_url=cfg.database_url)
    return get_session_factory(engine=engine)


def get_dashboard_metrics(session_factory: sessionmaker[Session]) -> Dict[str, Any]:
    """Compute aggregate counts for the governance dashboard.
    
    Informational only. Does not fabricate clinical KPIs.
    """
    with session_factory() as session:
        total_docs = session.query(IngestedDocument).count()
        processed_docs = session.query(IngestedDocument).filter(
            IngestedDocument.status.in_([DocumentStatus.COMPLETE.value, DocumentStatus.PARSED.value])
        ).count()
        g1_held = session.query(ChangeRecord).filter(
            ChangeRecord.status == ChangeStatus.HELD_FOR_G1.value
        ).count()
        g2_held = session.query(GapRecord).filter(
            GapRecord.status == GapStatus.REVIEW_REQUIRED.value
        ).count()
        g3_held = session.query(GapRecord).filter(
            GapRecord.status == GapStatus.NO_MATCH.value
        ).count()
        awaiting_gov = session.query(ChangeBrief).filter(
            ChangeBrief.status.in_([BriefStatus.DRAFT.value, BriefStatus.ASSIGNED.value, BriefStatus.IN_REVIEW.value])
        ).count()
        deferred_briefs = session.query(ChangeBrief).filter(
            ChangeBrief.status == BriefStatus.DEFERRED.value
        ).count()
        decided_briefs = session.query(ChangeBrief).filter(
            ChangeBrief.status == BriefStatus.DECIDED.value
        ).count()
        closed_briefs = session.query(ChangeBrief).filter(
            ChangeBrief.status == BriefStatus.CLOSED.value
        ).count()

        # Overdue SLA reviews
        now = datetime.now(timezone.utc)
        overdue_reviews = session.query(ReviewAssignment).filter(
            ReviewAssignment.status.in_([ReviewAssignmentStatus.ASSIGNED.value, ReviewAssignmentStatus.IN_REVIEW.value]),
            ReviewAssignment.due_date < now,
        ).count()

        escalated_notifs = session.query(Notification).filter(
            Notification.notification_type == "sla_escalation"
        ).count()

        return {
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "held_g1": g1_held,
            "held_g2": g2_held,
            "held_g3": g3_held,
            "awaiting_governance": awaiting_gov,
            "deferred_briefs": deferred_briefs,
            "decided_briefs": decided_briefs,
            "closed_briefs": closed_briefs,
            "overdue_reviews": overdue_reviews,
            "escalated_notifications": escalated_notifs,
        }


def get_source_documents(session_factory: sessionmaker[Session]) -> List[Dict[str, Any]]:
    """Retrieve all ingested synthetic clinical documents."""
    with session_factory() as session:
        docs = session.query(IngestedDocument).order_by(desc(IngestedDocument.ingest_timestamp)).all()
        return [
            {
                "id": d.id,
                "source_identifier": d.source_identifier,
                "source_path": d.source_path,
                "source_version": d.source_version,
                "sha256_hash": d.sha256_hash,
                "status": d.status,
                "ingest_timestamp": d.ingest_timestamp,
                "error_message": d.error_message,
                "change_records_count": len(d.change_records) if d.change_records else 0,
            }
            for d in docs
        ]


def get_changes_with_gaps(session_factory: sessionmaker[Session]) -> List[Dict[str, Any]]:
    """Retrieve extracted changes joined with comparison gap findings."""
    with session_factory() as session:
        changes = session.query(ChangeRecord).order_by(desc(ChangeRecord.created_at)).all()
        results = []
        for c in changes:
            gap = c.gap_record
            results.append({
                "change_id": c.id,
                "document_id": c.ingested_document_id,
                "verbatim_text": c.verbatim_text,
                "recommendation_type": c.recommendation_type,
                "target_population": c.target_population,
                "intervention": c.intervention,
                "evidence_grade": c.evidence_grade or "Unspecified",
                "confidence": c.confidence,
                "page": c.page,
                "section": c.section,
                "source_excerpt": c.source_excerpt,
                "change_status": c.status,
                # Gap details
                "has_gap": gap is not None,
                "gap_id": gap.id if gap else None,
                "comparison_result": gap.comparison_result if gap else "Pending",
                "matched_protocol_id": gap.matched_protocol_id if gap else None,
                "matched_protocol_version": gap.matched_protocol_version if gap else None,
                "difference_type": gap.difference_type if gap else None,
                "similarity": gap.similarity if gap else None,
                "comparison_confidence": gap.comparison_confidence if gap else None,
                "gap_status": gap.status if gap else "Not compared",
            })
        return results


def get_impact_records(session_factory: sessionmaker[Session]) -> List[Dict[str, Any]]:
    """Retrieve all persisted deterministic impact assessments."""
    with session_factory() as session:
        impacts = session.query(ImpactRecord).order_by(desc(ImpactRecord.created_at)).all()
        return [
            {
                "id": imp.id,
                "gap_record_id": imp.gap_record_id,
                "clinical_urgency": imp.clinical_urgency,
                "evidence_strength": imp.evidence_strength,
                "pathway_breadth": imp.pathway_breadth,
                "total_score": imp.total_score,
                "tier": imp.tier or "Unassigned",
                "sla_deadline": imp.sla_deadline,
                "urgency_basis": imp.urgency_basis,
                "evidence_basis": imp.evidence_basis,
                "breadth_basis": imp.breadth_basis,
                "rule_ids": imp.rule_ids or [],
                "scoring_yaml_version": imp.scoring_yaml_version,
                "status": imp.status,
            }
            for imp in impacts
        ]


def get_brief_summaries(session_factory: sessionmaker[Session]) -> List[Dict[str, Any]]:
    """Retrieve summary information for all generated ChangeBriefs."""
    with session_factory() as session:
        briefs = session.query(ChangeBrief).order_by(desc(ChangeBrief.created_at)).all()
        items = []
        for b in briefs:
            assignment = b.review_assignments[-1] if b.review_assignments else None
            imp = b.impact_record
            gap = imp.gap_record if imp else None
            change = gap.change_record if gap else None

            # Calculate overdue state
            now = datetime.now(timezone.utc)
            is_overdue = False
            if assignment and assignment.due_date:
                due = assignment.due_date if assignment.due_date.tzinfo else assignment.due_date.replace(tzinfo=timezone.utc)
                if due < now and assignment.status in (ReviewAssignmentStatus.ASSIGNED.value, ReviewAssignmentStatus.IN_REVIEW.value):
                    is_overdue = True

            items.append({
                "brief_id": b.id,
                "status": b.status,
                "rendered_file_path": b.rendered_file_path,
                "rendered_file_hash": b.rendered_file_hash,
                "tier": imp.tier if imp else "Unassigned",
                "sla_deadline": imp.sla_deadline if imp else None,
                "recommendation_summary": change.verbatim_text[:120] + "..." if change and change.verbatim_text else "N/A",
                "matched_protocol": gap.matched_protocol_id if gap else "None",
                "assigned_reviewer": assignment.reviewer_id if assignment else "Unassigned",
                "reviewer_role": assignment.reviewer_role if assignment else "None",
                "assignment_status": assignment.status if assignment else "None",
                "due_date": assignment.due_date if assignment else None,
                "decision": assignment.decision if assignment else None,
                "is_overdue": is_overdue,
            })
        return items


def get_brief_full_payload(session_factory: sessionmaker[Session], brief_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full structured payload and rendered companion files for a brief."""
    with session_factory() as session:
        brief = session.get(ChangeBrief, brief_id)
        if not brief:
            return None

        payload = brief.structured_payload or {}
        html_content = None
        md_content = None

        if brief.rendered_file_path:
            html_p = Path(brief.rendered_file_path)
            md_p = html_p.with_suffix(".md")
            if html_p.exists():
                html_content = html_p.read_text(encoding="utf-8")
            if md_p.exists():
                md_content = md_p.read_text(encoding="utf-8")

        # Fallback render if files not yet written
        if not html_content and payload:
            renderer = BriefRenderer()
            try:
                structured = StructuredBriefPayload.model_validate(payload)
                rendered = renderer.render_brief(structured, write_files=False, validate=False)
                html_content = rendered.html_content
                md_content = rendered.markdown_content
            except Exception as e:
                logger.warning("Error rendering brief on the fly: %s", e)

        return {
            "brief_id": brief.id,
            "status": brief.status,
            "rendered_file_path": brief.rendered_file_path,
            "rendered_file_hash": brief.rendered_file_hash,
            "structured_payload": payload,
            "html_content": html_content,
            "md_content": md_content,
        }


def get_audit_trail(session_factory: sessionmaker[Session], entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve immutable audit history from the AuditLog entity."""
    with session_factory() as session:
        query = session.query(AuditLog)
        if entity_id:
            query = query.filter(AuditLog.entity_id == entity_id)
        logs = query.order_by(desc(AuditLog.timestamp)).limit(200).all()
        return [
            {
                "id": log.id,
                "timestamp": log.timestamp,
                "actor": log.actor,
                "action": log.action if hasattr(log, "action") else log.new_status,
                "entity_id": log.entity_id,
                "entity_type": log.entity_type,
                "previous_status": log.previous_status,
                "new_status": log.new_status,
                "reason": log.reason,
                "metadata": log.audit_metadata,
            }
            for log in logs
        ]


def execute_governance_action(
    governance_agent: GovernanceAgent,
    brief_id: str,
    action: str,
    reviewer_id: str,
    rationale: str = "",
    follow_up_date: Optional[datetime] = None,
    defer_follow_up_date: Optional[datetime] = None,
    actor: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute an explicit human governance action through GovernanceAgent.

    Enforces that:
    1. Authorized reviewer identity is required.
    2. Non-empty rationale is required for decisions and closure.
    3. Future due date is required for deferral.
    4. Prohibits automated decisions.
    """
    actor_name = actor or reviewer_id
    action_clean = action.strip().lower()
    effective_follow_up = follow_up_date or defer_follow_up_date

    if not reviewer_id or not reviewer_id.strip():
        raise ValueError("Authorized reviewer identity is strictly required.")

    if not governance_agent.auth_service.is_authorized(reviewer_id):
        raise UnauthorizedReviewerError(f"Reviewer '{reviewer_id}' is not an authorized clinical reviewer.")

    if action_clean == "assign":
        assignment = governance_agent.assign_reviewer(
            change_brief_id=brief_id,
            reviewer_id=reviewer_id,
            reviewer_role=governance_agent.auth_service.get_reviewer_role(reviewer_id),
            actor=actor_name,
        )
        return {"action": "assign", "status": "success", "entity_id": assignment.id, "brief_status": BriefStatus.ASSIGNED.value}

    elif action_clean == "start_review":
        brief = governance_agent.start_review(change_brief_id=brief_id, reviewer_id=reviewer_id)
        return {"action": "start_review", "status": "success", "brief_id": brief.id, "brief_status": brief.status}

    elif action_clean in ("approve", "reject"):
        if not rationale or not rationale.strip():
            raise ValueError(f"A non-empty clinical rationale is strictly required for {action_clean.upper()}.")
        decision = ReviewDecision.APPROVE if action_clean == "approve" else ReviewDecision.REJECT
        brief = governance_agent.decide(
            change_brief_id=brief_id,
            reviewer_id=reviewer_id,
            decision=decision,
            rationale=rationale.strip(),
        )
        return {"action": action_clean, "status": "success", "decision": action_clean, "brief_status": brief.status}

    elif action_clean == "defer":
        if not rationale or not rationale.strip():
            raise ValueError("A non-empty clinical rationale is strictly required for DEFER.")
        if not effective_follow_up:
            raise ValueError("A future follow-up due date is strictly required for DEFER.")
        now = datetime.now(timezone.utc)
        if effective_follow_up.tzinfo is None:
            effective_follow_up = effective_follow_up.replace(tzinfo=timezone.utc)
        if effective_follow_up <= now:
            raise ValueError("Deferral follow-up date must be strictly in the future.")
        brief = governance_agent.defer(
            change_brief_id=brief_id,
            reviewer_id=reviewer_id,
            rationale=rationale.strip(),
            defer_follow_up_date=effective_follow_up,
        )
        return {"action": "defer", "status": "success", "brief_status": brief.status}

    elif action_clean == "close":
        if not rationale or not rationale.strip():
            raise ValueError("A non-empty administrative rationale is required for brief closure.")
        brief = governance_agent.close_after_decision(
            change_brief_id=brief_id,
            actor=actor_name,
            closure_note=rationale.strip(),
        )
        return {"action": "close", "status": "success", "brief_id": brief.id, "brief_status": brief.status}

    else:
        raise ValueError(f"Unsupported governance action: '{action}'.")


def load_evaluation_reports(reports_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load machine-readable Phase 12 evaluation reports from disk."""
    r_dir = reports_dir or Path("data/evaluation/reports")
    reports = {}
    expected_files = [
        "evaluation_summary.json",
        "extraction_report.json",
        "comparison_report.json",
        "impact_report.json",
        "briefing_report.json",
        "governance_safety_report.json",
        "end_to_end_report.json",
        "threshold_calibration_report.json",
    ]
    for fn in expected_files:
        p = r_dir / fn
        if p.exists():
            try:
                reports[fn] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                reports[fn] = {"error": f"Failed to load: {e}"}
        else:
            reports[fn] = None
    return reports


# ==============================================================================
# STREAMLIT UI PRESENTATION LAYER
# ==============================================================================

def render_header() -> None:
    """Render the standard CKEA clinical decision-support banner."""
    st.title("🏥 Clinical Knowledge Evolution Agent (CKEA)")
    st.caption("Demonstration & Governance Interface — Prototype Clinical Decision Support")
    st.info(
        "🛡️ **Clinical Governance Boundary:** System prepares evidence, highlights gaps, and assesses impact. "
        "**Authorized clinical reviewers make all governance decisions.** "
        "No recommendation or brief is ever automatically approved, rejected, deferred, or closed."
    )


def render_gate_legend() -> None:
    """Render the human gate indicators in the sidebar."""
    st.sidebar.markdown("### 🚪 Human Safety Gates")
    st.sidebar.markdown(
        """
        - 🔵 **G1** — Extraction Quality Review
        - 🟡 **G2** — Ambiguous Comparison Review
        - 🟠 **G3** — Unmatched Clinical Finding
        - 🔴 **G4** — Authorized Governance Sign-off
        - ⏰ **G5** — SLA Escalation (Overdue Review)
        """
    )
    st.sidebar.markdown("---")


def render_dashboard(session_factory: sessionmaker[Session]) -> None:
    """Render Dashboard overview metrics."""
    st.header("📊 Governance & Pipeline Dashboard")
    metrics = get_dashboard_metrics(session_factory)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents Monitored", metrics["total_documents"])
        st.metric("Documents Processed", metrics["processed_documents"])
    with col2:
        st.metric("Held at G1 (Extraction)", metrics["held_g1"])
        st.metric("Held at G2 (Comparison)", metrics["held_g2"])
    with col3:
        st.metric("Held at G3 (No-Match)", metrics["held_g3"])
        st.metric("Briefs Awaiting Review (G4)", metrics["awaiting_governance"])
    with col4:
        st.metric("Overdue Reviews (SLA)", metrics["overdue_reviews"], delta_color="inverse")
        st.metric("Closed Briefs", metrics["closed_briefs"])

    st.markdown("---")
    st.subheader("📌 System Status Summary")
    st.write(
        f"The system currently tracks **{metrics['total_documents']}** clinical source document(s). "
        f"**{metrics['held_g1'] + metrics['held_g2'] + metrics['held_g3']}** change item(s) are currently held at safety gates G1–G3. "
        f"**{metrics['awaiting_governance']}** evidence brief(s) are in active clinical governance stages."
    )


def render_sources_view(session_factory: sessionmaker[Session]) -> None:
    """Render Document and Monitoring inspection view."""
    st.header("📄 Ingested Clinical Sources")
    st.caption("Read-only inspection of discovered and parsed clinical documents.")

    docs = get_source_documents(session_factory)
    if not docs:
        st.warning("No clinical source documents found in the database.")
        return

    st.dataframe(
        [
            {
                "Identifier": d["source_identifier"],
                "Version": d["source_version"],
                "Status": d["status"],
                "Changes Extracted": d["change_records_count"],
                "SHA-256": d["sha256_hash"][:16] + "...",
                "Timestamp": str(d["ingest_timestamp"]),
                "Path": d["source_path"],
            }
            for d in docs
        ],
        use_container_width=True,
    )


def render_changes_view(session_factory: sessionmaker[Session]) -> None:
    """Render Extracted Changes & Protocol Comparison view."""
    st.header("🔍 Extracted Recommendations & Protocol Comparison")
    st.caption("Inspect discrete extracted clinical changes and matched hospital protocols.")

    changes = get_changes_with_gaps(session_factory)
    if not changes:
        st.warning("No extracted recommendations found.")
        return

    for c in changes:
        with st.expander(f"Recommendation: {c['verbatim_text'][:80]}... (Status: {c['change_status']})"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📝 Extracted Recommendation")
                st.write(f"**Verbatim Text:** {c['verbatim_text']}")
                st.write(f"**Type:** `{c['recommendation_type']}` | **Evidence:** `{c['evidence_grade']}`")
                st.write(f"**Population:** {c['target_population']}")
                st.write(f"**Intervention:** {c['intervention']}")
                st.write(f"**Confidence:** `{c['confidence']:.2f}` (Threshold: 0.70)")
                st.write(f"**Location:** Page {c['page']}, Section: *{c['section']}*")
                if c["source_excerpt"]:
                    st.info(f"**Source Excerpt:**\n> {c['source_excerpt']}")

            with c2:
                st.markdown("#### 🏥 Protocol Comparison Finding")
                if not c["has_gap"]:
                    st.write("Comparison not yet performed.")
                elif c["comparison_result"] == "no_match":
                    st.warning("⚠️ **No corresponding protocol section was found in the hospital formulary.**")
                    st.write(f"Comparison Result: `{c['comparison_result']}`")
                    st.write(f"Routing: **Human Gate G3 (Committee Review Required)**")
                else:
                    st.write(f"**Matched Protocol ID:** `{c['matched_protocol_id']}` (v{c['matched_protocol_version']})")
                    st.write(f"**Difference Type:** `{c['difference_type']}`")
                    sim_txt = f"{c['similarity']:.2f}" if c['similarity'] else "N/A"
                    st.write(f"**Similarity Score:** `{sim_txt}` | **Confidence:** `{c['comparison_confidence']}`")
                    st.write(f"**Result Classification:** `{c['comparison_result']}`")


def render_impact_view(session_factory: sessionmaker[Session]) -> None:
    """Render Deterministic Multidimensional Impact Assessments."""
    st.header("⚡ Deterministic Impact Assessments")
    st.caption("Deterministic scoring evaluated against scoring.yaml. Zero LLM calculations.")

    impacts = get_impact_records(session_factory)
    if not impacts:
        st.warning("No impact assessment records found.")
        return

    for imp in impacts:
        tier_color = {
            "Critical": "🔴",
            "High": "🟠",
            "Standard": "🟡",
            "Low": "🟢",
        }.get(imp["tier"], "⚪")

        with st.expander(f"{tier_color} Impact Tier: {imp['tier']} (Score: {imp['total_score']}) - ID: {imp['id'][:8]}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Clinical Urgency", imp["clinical_urgency"])
                st.caption(f"Basis: {imp['urgency_basis']}")
            with c2:
                st.metric("Evidence Strength", imp["evidence_strength"])
                st.caption(f"Basis: {imp['evidence_basis']}")
            with c3:
                st.metric("Pathway Breadth", imp["pathway_breadth"])
                st.caption(f"Basis: {imp['breadth_basis']}")

            st.write(f"**Total Multidimensional Score:** `{imp['total_score']}`")
            st.write(f"**Preserved Scoring Rules:** `{', '.join(imp['rule_ids'])}` (YAML Version: `{imp['scoring_yaml_version']}`)")
            st.write(f"**Assigned SLA Deadline:** `{imp['sla_deadline']}`")


def render_briefs_view(session_factory: sessionmaker[Session]) -> None:
    """Render 7-Section Change Briefs with HTML/Markdown/JSON views."""
    st.header("📋 Clinical Change Briefs")
    st.caption("Audit-grade evidence briefs structured for clinical governance committees.")

    briefs = get_brief_summaries(session_factory)
    if not briefs:
        st.warning("No ChangeBriefs generated yet.")
        return

    selected_brief_id = st.selectbox(
        "Select Change Brief:",
        options=[b["brief_id"] for b in briefs],
        format_func=lambda b_id: f"{b_id[:8]}... | Status: {next(b['status'] for b in briefs if b['brief_id'] == b_id)} | Tier: {next(b['tier'] for b in briefs if b['brief_id'] == b_id)}",
    )

    if not selected_brief_id:
        return

    full_brief = get_brief_full_payload(session_factory, selected_brief_id)
    if not full_brief:
        st.error("Failed to load brief payload.")
        return

    payload = full_brief["structured_payload"]

    tab1, tab2, tab3 = st.tabs(["📑 Seven Required Sections", "🌐 Rendered HTML", "📄 Raw Markdown / JSON"])

    with tab1:
        st.subheader("1. What Changed")
        rec = payload.get("recommendation", {})
        st.write(f"**Verbatim Recommendation:** {rec.get('verbatim_text')}")
        st.write(f"**Population:** {rec.get('target_population')}")
        st.write(f"**Intervention:** {rec.get('intervention')}")
        st.write(f"**Evidence Grade:** {rec.get('evidence_grade')}")

        st.subheader("2. Current Protocol")
        prot = payload.get("protocol", {})
        st.write(f"**Protocol ID:** `{prot.get('protocol_id')}` (Version: {prot.get('protocol_version')})")
        st.write(f"**Section Heading:** {prot.get('section_heading')}")
        st.write(f"**Current Text:** {prot.get('section_text')}")

        st.subheader("3. Specific Difference")
        diff = payload.get("comparison", {})
        st.write(f"**Difference Type:** `{diff.get('difference_type')}`")
        st.write(f"**Detailed Clinical Difference:** {diff.get('specific_difference')}")

        st.subheader("4. Impact Assessment")
        imp = payload.get("impact", {})
        st.write(f"**Routing Tier:** `{imp.get('tier')}` | **Total Score:** `{imp.get('total_score')}`")
        st.write(f"**SLA Deadline:** `{imp.get('sla_deadline')}`")
        st.write(f"**Urgency Basis:** {imp.get('urgency_basis')}")

        st.subheader("5. Affected Workflows")
        wf = payload.get("workflow", {})
        st.write(f"**Impacted Care Pathways:** {', '.join(wf.get('affected_departments', [])) or 'General Medical'}")
        st.write(f"**EHR Changes Required:** {wf.get('workflow_description')}")

        st.subheader("6. Proposed Review Actions")
        actions = payload.get("proposed_actions", [])
        for a in actions:
            st.markdown(f"- **{a.get('action_type', 'Action')}:** {a.get('description')}")

        st.subheader("7. Source Excerpt")
        src = payload.get("source_excerpt", {})
        st.info(f"**Exact Source Excerpt (Page {src.get('page')}, Section {src.get('section')}):**\n> {src.get('excerpt')}")

    with tab2:
        if full_brief.get("html_content"):
            st.components.v1.html(full_brief["html_content"], height=600, scrolling=True)
        else:
            st.info("HTML companion file not available.")

    with tab3:
        st.markdown("#### Companion Markdown")
        st.code(full_brief.get("md_content") or "Markdown not available", language="markdown")
        st.markdown("#### Companion JSON Structured Payload")
        st.json(payload)


def render_governance_view(session_factory: sessionmaker[Session], governance_agent: GovernanceAgent) -> None:
    """Render Human Governance Decision View (Human Gate G4)."""
    st.header("⚖️ Clinical Governance & Human Decisions (G4)")
    st.caption("Authorized clinicians review evidence briefs and record explicit governance decisions.")

    briefs = get_brief_summaries(session_factory)
    if not briefs:
        st.warning("No ChangeBriefs available for governance.")
        return

    # Filter selector
    brief_options = [b["brief_id"] for b in briefs]
    selected_brief_id = st.selectbox(
        "Select Brief for Governance Action:",
        options=brief_options,
        format_func=lambda b_id: f"{b_id[:8]}... | Status: {next(b['status'] for b in briefs if b['brief_id'] == b_id)} | Reviewer: {next(b['assigned_reviewer'] for b in briefs if b['brief_id'] == b_id)} | Overdue: {'🚨 YES' if next(b['is_overdue'] for b in briefs if b['brief_id'] == b_id) else 'NO'}",
    )

    current_brief = next(b for b in briefs if b["brief_id"] == selected_brief_id)

    # Status callout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Brief Status", current_brief["status"])
    with col2:
        st.metric("Assigned Reviewer", current_brief["assigned_reviewer"])
    with col3:
        st.metric("Review Decision", current_brief["decision"] or "Pending")
    with col4:
        st.metric("Overdue Status", "🚨 ESCALATED" if current_brief["is_overdue"] else "Normal")

    st.markdown("---")
    st.subheader("✍️ Record Explicit Human Governance Decision")
    st.warning("⚠️ **Reminder:** This action records an immutable human decision in the clinical audit log.")

    with st.form("governance_action_form"):
        action = st.selectbox(
            "Governance Action:",
            options=["start_review", "approve", "reject", "defer", "close", "assign"],
            format_func=lambda a: {
                "start_review": "▶️ Start Review (Transition to In Review)",
                "approve": "✅ APPROVE (Adopt Protocol Update)",
                "reject": "❌ REJECT (Do Not Adopt Recommendation)",
                "defer": "⏸️ DEFER (Request Additional Evidence/Review)",
                "close": "🔒 Close Brief (After Decision Recorded)",
                "assign": "👤 Assign / Reassign Reviewer",
            }.get(a, a),
        )

        reviewer_id = st.selectbox(
            "Authorized Clinical Reviewer Identity:",
            options=AUTHORIZED_REVIEWERS,
            index=0,
        )

        rationale = st.text_area(
            "Clinical Rationale / Written Justification (Mandatory):",
            placeholder="Enter clinical rationale, evidence justification, or committee discussion notes...",
        )

        follow_up_date_input = st.date_input(
            "Deferral Follow-up Due Date (Required ONLY if Deferring):",
            value=datetime.now(timezone.utc).date() + timedelta(days=14),
        )

        submitted = st.form_submit_button("Record Human Governance Decision")

        if submitted:
            follow_up_dt = None
            if action == "defer":
                follow_up_dt = datetime.combine(
                    follow_up_date_input, datetime.min.time(), tzinfo=timezone.utc
                )

            try:
                result = execute_governance_action(
                    governance_agent=governance_agent,
                    brief_id=selected_brief_id,
                    action=action,
                    reviewer_id=reviewer_id,
                    rationale=rationale,
                    follow_up_date=follow_up_dt,
                )
                st.success(f"Governance action '{action}' recorded successfully! Result: {result}")
                st.rerun()
            except Exception as e:
                st.error(f"Governance action rejected: {e}")


def render_audit_view(session_factory: sessionmaker[Session]) -> None:
    """Render Immutable Governance Audit Trail."""
    st.header("📜 Immutable Audit Log")
    st.caption("Cryptographically auditable and append-only record of all system events and human decisions.")

    logs = get_audit_trail(session_factory)
    if not logs:
        st.warning("No audit events recorded.")
        return

    st.dataframe(
        [
            {
                "Timestamp": str(log["timestamp"]),
                "Actor": log["actor"],
                "Action": log["action"],
                "Entity Type": log["entity_type"],
                "Entity ID": log["entity_id"][:8] + "...",
                "Old Status": log["previous_status"] or "None",
                "New Status": log["new_status"],
                "Reason / Rationale": log["reason"] or "N/A",
            }
            for log in logs
        ],
        use_container_width=True,
    )


def render_evaluation_reports_view() -> None:
    """Render Phase 12 Evaluation Framework Reports."""
    st.header("📈 Evaluation Framework Reports (Phase 12)")
    st.caption("Read-only inspection of reproducible evaluation results across the 20-document synthetic corpus.")

    reports = load_evaluation_reports()
    report_names = [k for k in reports.keys() if reports[k] is not None]

    if not report_names:
        st.warning("No evaluation reports found in data/evaluation/reports/.")
        return

    selected_report = st.selectbox("Select Evaluation Report:", options=report_names)

    if selected_report:
        rep_data = reports[selected_report]
        st.json(rep_data)


# ==============================================================================
# MAIN APPLICATION ROUTER
# ==============================================================================

def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="CKEA - Clinical Knowledge Evolution Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_header()
    render_gate_legend()

    config = load_config()
    session_factory = get_db_session_factory(config)
    governance_agent = GovernanceAgent(session_factory=session_factory, config=config)

    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "Workflow Area:",
        options=[
            "📊 Dashboard",
            "📄 Source Documents",
            "🔍 Clinical Changes & Comparison",
            "⚡ Impact Assessment",
            "📋 Change Briefs",
            "⚖️ Governance & Human Review",
            "📜 Immutable Audit Log",
            "📈 Evaluation Reports",
        ],
    )

    if menu == "📊 Dashboard":
        render_dashboard(session_factory)
    elif menu == "📄 Source Documents":
        render_sources_view(session_factory)
    elif menu == "🔍 Clinical Changes & Comparison":
        render_changes_view(session_factory)
    elif menu == "⚡ Impact Assessment":
        render_impact_view(session_factory)
    elif menu == "📋 Change Briefs":
        render_briefs_view(session_factory)
    elif menu == "⚖️ Governance & Human Review":
        render_governance_view(session_factory, governance_agent)
    elif menu == "📜 Immutable Audit Log":
        render_audit_view(session_factory)
    elif menu == "📈 Evaluation Reports":
        render_evaluation_reports_view()


if __name__ == "__main__":
    main()
