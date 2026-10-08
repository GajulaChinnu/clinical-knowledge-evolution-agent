"""Data access, identity and ingestion entry points for the CKEA workspace (no rendering)."""

import logging
from contextlib import contextmanager
from datetime import (
    datetime,
    timezone,
)
import json
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import streamlit as st
from sqlalchemy import desc
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)
from app.agents.governance_agent import GovernanceAgent
from app.agents.monitoring_agent import MonitoringAgent
from app.models.database import (
    get_engine,
    get_session_factory,
    init_db,
)
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
from app.orchestration.pipeline import ClinicalKnowledgePipeline
from app.schemas.briefs import (
    BriefStatus,
    StructuredBriefPayload,
)
from app.schemas.changes import ChangeStatus
from app.schemas.documents import DocumentStatus
from app.schemas.gaps import GapStatus
from app.schemas.governance import (
    ReviewAssignmentStatus,
    ReviewDecision,
    UnauthorizedReviewerError,
)
from app.services.brief_renderer import BriefRenderer
from app.services.config_service import (
    AppConfig,
    load_config,
)
from app.services.protocol_lookup import (
    find_protocol_section,
    section_id_from_candidates,
)
from app.services.reviewer_authorization import (
    Reviewer,
    ReviewerAuthorizationService,
)
from app.services.source_ingestion_service import SourceIngestionService
from app.services.url_ingestion_service import URLIngestionService

logger = logging.getLogger("ckea.ui.queries")


def resolve_acting_reviewer(
    auth_service: ReviewerAuthorizationService,
    config: AppConfig,
    user: Any = None,
) -> Tuple[Optional[Reviewer], str, Optional[str]]:
    """Determine who is acting in the governance workspace.

    Returns (reviewer, mode, message) where mode is:
    - "sso": identity comes from the SSO sign-in; reviewer is None if the email is not registered.
    - "signin_required": SSO is enabled but nobody is signed in; decisions are blocked.
    - "demo": SSO disabled; the identity is self-selected and the UI must say so.
    """
    if not config.sso_enabled:
        return None, "demo", (
            "Demo mode: SSO is not enabled, so the reviewer identity below is self-selected and "
            "unauthenticated. Enable SSO_ENABLED with an OIDC provider before clinical use."
        )
    user = user if user is not None else st.user
    try:
        logged_in = bool(user.is_logged_in)
        email = user.get("email") if hasattr(user, "get") else getattr(user, "email", None)
    except (AttributeError, KeyError):
        logged_in, email = False, None
    if not logged_in:
        return None, "signin_required", "Sign in to record governance decisions."
    reviewer = auth_service.reviewer_for_email(email)
    if reviewer is None:
        return None, "sso", f"Signed in as {email}, who is not in the reviewer registry. Decisions are blocked."
    return reviewer, "sso", None


@st.cache_resource(show_spinner=False)
def _initialized_engine(db_url: str):
    """Create the engine once per process and bring the schema to the latest migration."""
    engine = get_engine(db_url=db_url)
    init_db(engine=engine)
    return engine


def get_db_session_factory(config: Optional[AppConfig] = None) -> sessionmaker[Session]:
    """Retrieve session factory using application configuration (schema migrated on first use)."""
    cfg = config or load_config()
    return get_session_factory(engine=_initialized_engine(cfg.database_url))


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


def _retrieval_provider_label(doc_metadata: Optional[Dict[str, Any]]) -> str:
    meta = doc_metadata or {}
    if meta.get("retrieval_provider"):
        return meta["retrieval_provider"]
    return "Direct HTTP" if meta.get("source_url") else "Local Upload"


def get_source_documents(session_factory: sessionmaker[Session]) -> List[Dict[str, Any]]:
    """Retrieve all ingested synthetic clinical documents."""
    with session_factory() as session:
        docs = session.query(IngestedDocument).order_by(desc(IngestedDocument.ingest_timestamp)).all()
        
        version_by_id = {d.id: (d.source_version or d.document_version) for d in docs}
        first_seen_map = {}
        for d in sorted(docs, key=lambda x: x.ingest_timestamp):
            if d.source_identifier not in first_seen_map:
                first_seen_map[d.source_identifier] = d.ingest_timestamp

        return [
            {
                "id": d.id,
                "source_identifier": d.source_identifier,
                "source_path": d.source_path,
                "source_version": d.source_version,
                "previous_source_version_id": d.previous_source_version_id,
                "previous_source_version": version_by_id.get(d.previous_source_version_id),
                "sha256_hash": d.sha256_hash,
                "previous_sha256_hash": d.previous_sha256_hash,
                "change_status": d.change_status,
                "first_seen": first_seen_map[d.source_identifier],
                "last_retrieved": d.ingest_timestamp,
                "status": d.status,
                "ingest_timestamp": d.ingest_timestamp,
                "error_message": d.error_message,
                "change_records_count": len(d.change_records) if d.change_records else 0,
                "retrieval_provider": _retrieval_provider_label(d.doc_metadata),
                "routing_decision": (d.doc_metadata or {}).get("routing_decision"),
                "title": (d.doc_metadata or {}).get("title"),
                "display_name": (
                    (d.doc_metadata or {}).get("title")
                    or (d.doc_metadata or {}).get("source_url")
                    or (d.doc_metadata or {}).get("original_filename")
                    or d.source_identifier
                ),
                "input_type": (d.doc_metadata or {}).get("input_type"),
                "source_diff": (d.doc_metadata or {}).get("source_diff"),
                "resolved_source_url": (d.doc_metadata or {}).get("resolved_source_url"),
                "provider_warnings": (d.doc_metadata or {}).get("provider_warnings") or [],
            }
            for d in docs
        ]


def resolve_protocol_section_details(
    protocol_id: Optional[str],
    protocol_version: Optional[str] = None,
    candidate_section_ids: Optional[List[str]] = None,
    protocol_dir: Optional[Union[str, Path]] = None,
    section_id: Optional[str] = None,
    verified_protocol_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Institutional protocol baseline for display: exact id/version/section or explicitly unresolved.

    Never substitutes another version, another section, or placeholder text.
    """
    if not protocol_id or protocol_id.lower() in ("none", "null"):
        return {
            "is_match": False,
            "is_resolved": True,
            "protocol_id": None,
            "protocol_version": None,
            "protocol_title": "No Matching Institutional Protocol",
            "section_id": None,
            "section_heading": "No matching section",
            "section_text": "No matching institutional protocol section was identified in the hospital library.",
            "status_message": "Comparison stopped at G3 / committee review required.",
        }

    sec_id = section_id or section_id_from_candidates(candidate_section_ids)
    found = find_protocol_section(protocol_dir or Path("data/protocols"), protocol_id, protocol_version, sec_id)
    if found:
        p_doc, sec = found
        return {
            "is_match": True,
            "is_resolved": True,
            "protocol_id": p_doc.protocol_id,
            "protocol_version": p_doc.protocol_version,
            "protocol_title": p_doc.title or f"Protocol {p_doc.protocol_id}",
            "section_id": sec.section_id,
            "section_heading": sec.section_heading,
            "section_text": sec.section_text,
            "status_message": "Matched institutional protocol section on file.",
        }
    if verified_protocol_text:
        return {
            "is_match": True,
            "is_resolved": True,
            "protocol_id": protocol_id,
            "protocol_version": protocol_version,
            "protocol_title": f"Protocol {protocol_id}",
            "section_id": sec_id,
            "section_heading": f"Section {sec_id}" if sec_id else "Matched section",
            "section_text": verified_protocol_text,
            "status_message": "Verified quotation captured at comparison time.",
        }
    return {
        "is_match": True,
        "is_resolved": False,
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "protocol_title": f"Protocol {protocol_id}",
        "section_id": sec_id,
        "section_heading": "Unresolved",
        "section_text": None,
        "status_message": (
            f"Protocol text for {protocol_id} version {protocol_version} section {sec_id} "
            "could not be resolved. Review against the protocol library before deciding."
        ),
    }


@contextmanager
def _session_scope(session_or_factory: Union[sessionmaker[Session], Session]):
    """Context manager that handles both sessionmaker factory and active Session instances."""
    if hasattr(session_or_factory, "query"):
        yield session_or_factory
    else:
        session = session_or_factory()
        try:
            yield session
        finally:
            session.close()


def get_changes_with_gaps(
    session_factory: Union[sessionmaker[Session], Session],
    config: Optional[AppConfig] = None,
) -> List[Dict[str, Any]]:
    """Retrieve extracted changes joined with comparison gap findings and protocol baselines."""
    cfg = config or load_config()
    with _session_scope(session_factory) as session:
        changes = session.query(ChangeRecord).order_by(desc(ChangeRecord.created_at)).all()
        results = []
        for c in changes:
            doc = c.document
            meta = dict(doc.doc_metadata or {}) if doc else {}
            gap = c.gap_record

            # Resolve protocol details
            proto_id = gap.matched_protocol_id if gap else None
            proto_ver = gap.matched_protocol_version if gap else None
            cand_sec_ids = gap.candidate_protocol_section_ids if gap else []

            proto_info = resolve_protocol_section_details(
                protocol_id=proto_id,
                protocol_version=proto_ver,
                candidate_section_ids=cand_sec_ids,
                protocol_dir=cfg.protocol_dir,
                section_id=gap.matched_section_id if gap else None,
                verified_protocol_text=gap.exact_protocol_text if gap else None,
            )

            # Specific difference & rationale
            brief_payload = {}
            if gap and gap.impact_record and gap.impact_record.change_brief:
                brief_payload = gap.impact_record.change_brief.structured_payload or {}

            spec_diff = (gap.specific_difference if gap else None) or (brief_payload.get("specific_difference") or {}).get("specific_difference")
            if not spec_diff:
                if gap and gap.is_match and gap.difference_type:
                    diff_readable = gap.difference_type.replace("_", " ")
                    spec_diff = f"Identified {diff_readable} between external recommendation and protocol ({gap.matched_protocol_id})."
                elif gap and not gap.is_match:
                    spec_diff = "No matching institutional protocol section found in hospital library."
                else:
                    spec_diff = "Pending comparison"

            diff_type = gap.difference_type if gap else None
            is_contradiction = diff_type in ("conflict", "contradiction")

            previous_doc = (
                session.get(IngestedDocument, doc.previous_source_version_id)
                if doc and doc.previous_source_version_id else None
            )

            results.append({
                "change_id": c.id,
                # Source evolution (external v(n-1) -> v(n)); independent of protocol comparison
                "source_version": (doc.source_version or doc.document_version) if doc else None,
                "previous_source_version": (
                    previous_doc.source_version or previous_doc.document_version
                ) if previous_doc else None,
                "source_change_status": doc.change_status if doc else None,
                "source_diff": meta.get("source_diff"),
                "document_id": c.ingested_document_id,
                "source_identifier": doc.source_identifier if doc else "N/A",
                "source_type": meta.get("source_type", "pdf").upper(),
                "source_url": meta.get("source_url"),
                "source_title": meta.get("title"),
                "resolved_source_url": meta.get("resolved_source_url"),
                "retrieval_timestamp": meta.get("retrieval_timestamp") or str(doc.ingest_timestamp if doc else "N/A"),
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
                # Protocol baseline
                "has_gap": gap is not None,
                "gap_id": gap.id if gap else None,
                "is_match": gap.is_match if gap else False,
                "matched_protocol_id": proto_info["protocol_id"],
                "matched_protocol_version": proto_info["protocol_version"],
                "protocol_title": proto_info["protocol_title"],
                "protocol_section_id": proto_info["section_id"],
                "protocol_section_heading": proto_info["section_heading"],
                "protocol_section_text": proto_info["section_text"],
                "protocol_resolved": proto_info["is_resolved"],
                "protocol_status_message": proto_info["status_message"],
                # Comparison Result
                "comparison_result": gap.comparison_result if gap else "Pending",
                "difference_type": diff_type or "Unspecified",
                "similarity": gap.similarity if gap else None,
                "comparison_confidence": gap.comparison_confidence if gap else None,
                "specific_difference": spec_diff,
                "comparison_rationale": gap.comparison_rationale if gap else None,
                "review_reason": gap.review_reason if gap else None,
                "contradiction_status": "Contradiction detected (opposing intervention)" if is_contradiction else "No contradiction",
                "gap_status": gap.status if gap else "Not compared",
            })
        return results


def get_impact_records(session_factory: Union[sessionmaker[Session], Session]) -> List[Dict[str, Any]]:
    """Retrieve all persisted deterministic impact assessments joined with source and protocol."""
    with _session_scope(session_factory) as session:
        impacts = session.query(ImpactRecord).order_by(desc(ImpactRecord.created_at)).all()
        results = []
        for imp in impacts:
            gap = imp.gap_record
            change = gap.change_record if gap else None
            doc = change.document if change else None
            meta = dict(doc.doc_metadata or {}) if doc else {}

            results.append({
                "id": imp.id,
                "gap_record_id": imp.gap_record_id,
                "source_identifier": doc.source_identifier if doc else "N/A",
                "source_path": doc.source_path if doc else "N/A",
                "source_url": meta.get("source_url"),
                "pmid": meta.get("pmid"),
                "resolved_source_url": meta.get("resolved_source_url"),
                "source_type": meta.get("source_type", "pdf").upper(),
                "verbatim_recommendation": change.verbatim_text if change else "N/A",
                "recommendation_type": change.recommendation_type if change else "N/A",
                "matched_protocol_id": gap.matched_protocol_id if gap else "None",
                "matched_protocol_version": gap.matched_protocol_version if gap else "N/A",
                "comparison_result": gap.comparison_result if gap else "Pending",
                "difference_type": gap.difference_type if gap else "Unspecified",
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
            })
        return results


def get_brief_summaries(
    session_factory: Union[sessionmaker[Session], Session],
    department: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve summary information for all generated ChangeBriefs joined with source and protocol."""
    with _session_scope(session_factory) as session:
        briefs = session.query(ChangeBrief).order_by(desc(ChangeBrief.created_at)).all()
        items = []
        for b in briefs:
            assignment = b.review_assignments[-1] if b.review_assignments else None
            imp = b.impact_record
            gap = imp.gap_record if imp else None
            change = gap.change_record if gap else None
            doc = change.document if change else None
            meta = dict(doc.doc_metadata or {}) if doc else {}
            payload = b.structured_payload or {}

            # Calculate overdue state
            now = datetime.now(timezone.utc)
            is_overdue = False
            if assignment and assignment.due_date:
                due = assignment.due_date if assignment.due_date.tzinfo else assignment.due_date.replace(tzinfo=timezone.utc)
                if due < now and assignment.status in (ReviewAssignmentStatus.ASSIGNED.value, ReviewAssignmentStatus.IN_REVIEW.value):
                    is_overdue = True

            sec_id = None
            if gap and gap.candidate_protocol_section_ids and len(gap.candidate_protocol_section_ids) > 0:
                sec_id = gap.candidate_protocol_section_ids[0].split("__")[-1]

            diff_summary = (gap.specific_difference if gap else None) or (payload.get("specific_difference") or {}).get("specific_difference")
            if not diff_summary and gap:
                diff_summary = f"Difference category: {gap.difference_type or 'unspecified'}"

            items.append({
                "brief_id": b.id,
                "status": b.status,
                "rendered_file_path": b.rendered_file_path,
                "rendered_file_hash": b.rendered_file_hash,
                "tier": imp.tier if imp else "Unassigned",
                "total_score": imp.total_score if imp else None,
                "sla_deadline": imp.sla_deadline if imp else None,
                "source_identifier": doc.source_identifier if doc else "N/A",
                "source_url": meta.get("source_url"),
                "pmid": meta.get("pmid"),
                "resolved_source_url": meta.get("resolved_source_url"),
                "source_type": meta.get("source_type", "pdf").upper(),
                "recommendation_summary": change.verbatim_text[:120] + "..." if change and change.verbatim_text else "N/A",
                "verbatim_recommendation": change.verbatim_text if change else "N/A",
                "matched_protocol": gap.matched_protocol_id if gap else "None",
                "matched_protocol_id": gap.matched_protocol_id if gap else "None",
                "protocol_version": gap.matched_protocol_version if gap else "1.0",
                "protocol_section": sec_id or "SEC-1",
                "protocol_section_id": sec_id or "SEC-1",
                "difference_summary": diff_summary or "Clinical protocol update required",
                "assigned_reviewer": assignment.reviewer_id if assignment else "Unassigned",
                "reviewer_role": assignment.reviewer_role if assignment else "None",
                "assignment_status": assignment.status if assignment else "None",
                "due_date": assignment.due_date if assignment else None,
                "decision": assignment.decision if assignment else None,
                "is_overdue": is_overdue,
                "affected_departments": list(payload.get("affected_departments")
                                             or (imp.affected_departments if imp and imp.affected_departments else [])
                                             or meta.get("departments") or []),
            })
        if department:
            items = [i for i in items if department in i["affected_departments"]]
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


def handle_source_pdf_upload(
    uploaded_name: str,
    uploaded_bytes: bytes,
    session_factory: sessionmaker[Session],
    config: Optional[AppConfig] = None,
    monitoring_agent: Optional[MonitoringAgent] = None,
    pipeline: Optional[ClinicalKnowledgePipeline] = None,
    existing_source_identifier: Optional[str] = None,
) -> Dict[str, Any]:
    """UI entry point for PDF uploads; all orchestration lives in SourceIngestionService."""
    return SourceIngestionService(
        session_factory=session_factory,
        config=config or load_config(),
        monitoring_agent=monitoring_agent,
        pipeline=pipeline,
    ).ingest_pdf_upload(
        uploaded_name=uploaded_name,
        uploaded_bytes=uploaded_bytes,
        existing_source_identifier=existing_source_identifier,
    )


def handle_source_url_upload(
    url: str,
    session_factory: sessionmaker[Session],
    config: Optional[AppConfig] = None,
    url_service: Optional[URLIngestionService] = None,
    monitoring_agent: Optional[MonitoringAgent] = None,
    pipeline: Optional[ClinicalKnowledgePipeline] = None,
) -> Dict[str, Any]:
    """UI entry point for URL sources; all orchestration lives in SourceIngestionService."""
    return SourceIngestionService(
        session_factory=session_factory,
        config=config or load_config(),
        url_service=url_service,
        monitoring_agent=monitoring_agent,
        pipeline=pipeline,
    ).ingest_url(url)


# ==============================================================================
# CLINICIAN QUERY WORKFLOW, WATCHLIST SURVEILLANCE AND DETECTED CHANGES
# ==============================================================================

def build_clinician_workflow(session_factory: sessionmaker[Session], config: AppConfig):
    """Construct the six agents and the clinician Treatment Check workflow."""
    from app.agents.briefing_agent import BriefingAgent
    from app.agents.comparison_agent import ComparisonAgent
    from app.agents.extraction_agent import ExtractionAgent
    from app.agents.impact_agent import ImpactAgent
    from app.orchestration.clinician_query import ClinicianQueryWorkflow
    from app.services.url_ingestion_service import URLIngestionService

    return ClinicianQueryWorkflow(
        session_factory=session_factory,
        config=config,
        monitoring_agent=MonitoringAgent(source_dir=config.source_dir, session_factory=session_factory, config=config),
        extraction_agent=ExtractionAgent(session_factory=session_factory, config=config),
        comparison_agent=ComparisonAgent(session_factory=session_factory, config=config),
        impact_agent=ImpactAgent(session_factory=session_factory, config=config),
        briefing_agent=BriefingAgent(session_factory=session_factory, config=config),
        governance_agent=GovernanceAgent(session_factory=session_factory, config=config),
        url_service=URLIngestionService(config=config),
    )


def run_watchlist_surveillance(
    session_factory: sessionmaker[Session], config: AppConfig, entry_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Background surveillance: check watchlist sources, index new versions, grade and rank changes."""
    from app.agents.extraction_agent import ExtractionAgent
    from app.agents.impact_agent import ImpactAgent
    from app.services.url_ingestion_service import URLIngestionService
    from app.services.watchlist import load_watchlist

    watchlist = load_watchlist()
    entries = [watchlist.get(e) for e in entry_ids] if entry_ids else list(watchlist)
    monitoring = MonitoringAgent(source_dir=config.source_dir, session_factory=session_factory, config=config)
    checks = monitoring.check_watchlist(watchlist, entries, URLIngestionService(config=config))
    doc_ids = [v.document_id for c in checks for v in c.versions]
    ExtractionAgent(session_factory=session_factory, config=config).index_guidance_statements(doc_ids)
    ImpactAgent(session_factory=session_factory, config=config).assess_guidance_changes()
    return [{
        "Source": watchlist.get(c.entry_id).title,
        "New versions": len(c.new_document_ids),
        "Latest": (c.latest.publisher_version or c.latest.internal_version) if c.latest else "-",
        "Published": c.latest.published_date if c.latest else "-",
        "Result": c.error or "OK",
    } for c in checks]


def get_watchlist_rows(session_factory: sessionmaker[Session]) -> List[Dict[str, Any]]:
    """Watchlist entries with their latest stored version."""
    from app.services.taxonomy import get_taxonomy
    from app.services.watchlist import load_watchlist

    taxonomy = get_taxonomy()
    rows = []
    with session_factory() as session:
        for entry in load_watchlist():
            docs = (session.query(IngestedDocument).filter_by(source_identifier=entry.source_identity)
                    .order_by(desc(IngestedDocument.created_at)).all())
            latest = docs[0] if docs else None
            meta = (latest.doc_metadata or {}) if latest else {}
            rows.append({
                "id": entry.id,
                "Source": entry.title,
                "Type": entry.source_type.replace("_", " "),
                "Departments": ", ".join(taxonomy.department_name(d) for d in entry.departments),
                "Quality tier": entry.quality_tier,
                "Latest version": meta.get("publisher_version") or (latest.source_version if latest else "not yet checked"),
                "Published": meta.get("published_date") or "",
                "Versions stored": len(docs),
                "Last checked": str(latest.ingest_timestamp)[:16] if latest else "",
            })
    return rows


def get_detected_changes(
    session_factory: sessionmaker[Session], department: Optional[str] = None, filtered: bool = False,
) -> List[Dict[str, Any]]:
    """Ranked source-evolution changes (latest versions only), or the filtered ones."""
    from app.agents.impact_agent import ImpactAgent
    from app.models.entities import GuidanceStatement
    from app.services.taxonomy import get_taxonomy

    taxonomy = get_taxonomy()
    feed = ImpactAgent(session_factory=session_factory).ranked_feed(department=department, include_filtered=True)
    rows = []
    with session_factory() as session:
        for change in feed:
            is_filtered = change.relevance_status == "filtered_not_practice_changing"
            if is_filtered != filtered:
                continue
            old = session.get(GuidanceStatement, change.from_statement_id) if change.from_statement_id else None
            new = session.get(GuidanceStatement, change.to_statement_id) if change.to_statement_id else None
            doc = session.get(IngestedDocument, change.to_document_id)
            meta = (doc.doc_metadata or {}) if doc else {}
            rows.append({
                "id": change.id,
                "category": change.change_category,
                "source": meta.get("title") or change.source_identity,
                "source_type": change.source_type,
                "previous_version": old.publisher_version if old else None,
                "previous_date": old.published_date if old else None,
                "previous_text": old.verbatim_text if old else None,
                "latest_version": new.publisher_version if new else meta.get("publisher_version"),
                "latest_date": new.published_date if new else meta.get("published_date"),
                "latest_text": new.verbatim_text if new else None,
                "attribute_changes": change.attribute_changes or [],
                "treatments": [taxonomy.treatment_name(t) for t in (change.treatments or [])],
                "departments": [taxonomy.department_name(d) for d in (change.departments or [])],
                "pathways": [taxonomy.pathways[p].name for p in (change.pathways or []) if p in taxonomy.pathways],
                "priority": change.priority_score,
                "urgency": change.urgency,
                "relevance": change.relevance,
                "source_quality": change.source_quality,
                "novelty": change.novelty,
                "duplicate_of": change.duplicate_of_change_id,
                "ranking_basis": change.ranking_basis or {},
                "status": change.relevance_status,
                "filter_reason": change.filter_reason,
            })
    return rows


def restore_filtered_change(session_factory: sessionmaker[Session], config: AppConfig, change_id: str,
                            reviewer_id: str, reason: str):
    from app.agents.impact_agent import ImpactAgent
    from app.services.reviewer_authorization import ReviewerAuthorizationService

    auth = ReviewerAuthorizationService(registry_path=config.reviewer_registry_path)
    return ImpactAgent(session_factory=session_factory, config=config).restore_filtered_change(change_id, reviewer_id, reason, auth)


def get_query_history(session_factory: sessionmaker[Session], limit: int = 15) -> List[Dict[str, Any]]:
    from app.models.entities import ClinicianQueryRecord

    with session_factory() as session:
        rows = session.query(ClinicianQueryRecord).order_by(desc(ClinicianQueryRecord.created_at)).limit(limit).all()
        return [{"id": r.id, "created_at": r.created_at, "department": r.department, "treatment": r.treatment,
                 "verdict": r.verdict, "answer": r.answer} for r in rows]
