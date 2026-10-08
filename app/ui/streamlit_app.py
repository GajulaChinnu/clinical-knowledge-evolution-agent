"""Clinical Knowledge Evolution Agent (CKEA) - Streamlit Governance & Demonstration UI.

Phase 13: Final End-to-End Application Integration & Human Governance Interface.

CORE SAFETY PRINCIPLE:
"System prepares evidence and review material. Authorized clinicians make the final decision."
The application NEVER makes automated clinical decisions, approves, rejects, defers, or closes
briefs without explicit human action.
"""

from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path for direct `streamlit run` execution
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.agents.governance_agent import GovernanceAgent
from app.agents.monitoring_agent import MonitoringAgent, ScanResult
from app.models.database import get_engine, get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    IngestionFailure,
    Notification,
    ReviewAssignment,
)
from app.orchestration.pipeline import ClinicalKnowledgePipeline
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
from app.schemas.orchestration import (
    HumanGate,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)
from app.services.brief_renderer import BriefRenderer
from app.services.config_service import AppConfig, load_config
from app.services.file_hash import compute_sha256
from app.services.pdf_parser import extract_page_texts
from app.services.protocol_index import parse_protocol_file
from app.services.reviewer_authorization import ReviewerAuthorizationService
from app.services.source_documents import UnreadableSourceError
from app.services.source_ingestion_service import (
    SourceIngestionService,
    UnknownSourceError,
    sanitize_filename,
    save_uploaded_pdf,
)
from app.services.url_ingestion_service import (
    ContentChallengeError,
    EmptyContentError,
    HTTPFetchError,
    InvalidURLError,
    URLIngestionError,
    URLIngestionResult,
    URLIngestionService,
)

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


def _retrieval_provider_label(doc_metadata: Optional[Dict[str, Any]]) -> str:
    meta = doc_metadata or {}
    if meta.get("retrieval_provider"):
        return meta["retrieval_provider"]
    return "Direct HTTP" if meta.get("source_url") else "Local Upload"


def get_source_documents(session_factory: sessionmaker[Session]) -> List[Dict[str, Any]]:
    """Retrieve all ingested synthetic clinical documents."""
    with session_factory() as session:
        docs = session.query(IngestedDocument).order_by(desc(IngestedDocument.ingest_timestamp)).all()
        
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
) -> Dict[str, Any]:
    """Retrieve ground-truth institutional protocol baseline section details."""
    if not protocol_id or protocol_id.lower() in ("none", "null"):
        return {
            "is_match": False,
            "protocol_id": None,
            "protocol_version": None,
            "protocol_title": "No Matching Institutional Protocol",
            "section_id": None,
            "section_heading": "No matching section",
            "section_text": "No matching institutional protocol section was identified in the hospital library.",
            "status_message": "Comparison stopped at G3 / committee review required.",
        }

    p_dir = Path(protocol_dir) if protocol_dir else Path("data/protocols")
    target_sec_id = None
    if candidate_section_ids and len(candidate_section_ids) > 0:
        target_sec_id = candidate_section_ids[0].split("__")[-1]

    if p_dir.exists():
        for p_file in p_dir.iterdir():
            if p_file.is_file() and p_file.suffix.lower() in (".json", ".md", ".txt"):
                try:
                    p_doc = parse_protocol_file(p_file)
                    if p_doc.protocol_id == protocol_id:
                        matched_sec = None
                        if target_sec_id:
                            for sec in p_doc.sections:
                                if sec.section_id == target_sec_id:
                                    matched_sec = sec
                                    break
                        if not matched_sec and p_doc.sections:
                            matched_sec = p_doc.sections[0]

                        if matched_sec:
                            return {
                                "is_match": True,
                                "protocol_id": p_doc.protocol_id,
                                "protocol_version": p_doc.protocol_version,
                                "protocol_title": p_doc.title or f"Protocol {p_doc.protocol_id}",
                                "section_id": matched_sec.section_id,
                                "section_heading": matched_sec.section_heading,
                                "section_text": matched_sec.section_text,
                                "status_message": "Matched institutional protocol section on file.",
                            }
                except Exception as e:
                    logger.warning("Protocol file %s could not be parsed: %s", p_file.name, e)

    return {
        "is_match": True,
        "protocol_id": protocol_id,
        "protocol_version": protocol_version or "v1.0",
        "protocol_title": f"Protocol {protocol_id}",
        "section_id": target_sec_id or "SEC-1",
        "section_heading": "Institutional Clinical Protocol Section",
        "section_text": f"Ground-truth text for protocol {protocol_id} (version {protocol_version or 'v1.0'}).",
        "status_message": "Institutional protocol metadata on file.",
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
            )

            # Specific difference & rationale
            brief_payload = {}
            if gap and gap.impact_record and gap.impact_record.change_brief:
                brief_payload = gap.impact_record.change_brief.structured_payload or {}

            spec_diff = brief_payload.get("comparison", {}).get("specific_difference")
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

            results.append({
                "change_id": c.id,
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
                # Comparison Result
                "comparison_result": gap.comparison_result if gap else "Pending",
                "difference_type": diff_type or "Unspecified",
                "similarity": gap.similarity if gap else None,
                "comparison_confidence": gap.comparison_confidence if gap else None,
                "specific_difference": spec_diff,
                "comparison_rationale": spec_diff,
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


def get_brief_summaries(session_factory: Union[sessionmaker[Session], Session]) -> List[Dict[str, Any]]:
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

            diff_summary = payload.get("comparison", {}).get("specific_difference")
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
# STREAMLIT UI PRESENTATION LAYER
# ==============================================================================

ENTERPRISE_CUSTOM_CSS = """
<style>
/* Modern Enterprise SaaS Styling & Design Tokens */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Main Container Spacing */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
}

/* Sidebar Styling & Fix for Dark Mode Text Invisible Issue */
[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Hide Streamlit Header to remove the top white strip */
[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
/* Ensure text is always dark on the light sidebar background */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
    color: #1e293b !important;
}

/* Sidebar Navigation Styling (Overrides Streamlit Radio) */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 0.15rem;
}
[data-testid="stSidebar"] .stRadio label {
    background-color: transparent;
    padding: 0.45rem 0.75rem !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background-color: #e2e8f0 !important;
}
/* Hide the default radio circle */
[data-testid="stSidebar"] .stRadio span[data-baseweb="radio"] div {
    display: none !important;
}
/* Active state styling using :has() selector */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background-color: #e0e7ff !important;
    border-left: 4px solid #2563eb !important;
    padding-left: calc(0.75rem - 4px) !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #1e40af !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stRadio label p {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

/* Typography Hierarchy */
.page-breadcrumb {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.1rem;
}
.page-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
    margin: 0 0 0.2rem 0;
}
.page-description {
    font-size: 0.85rem;
    color: #475569;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}

/* Compact Governance Notice Strip */
.gov-notice-strip {
    display: flex;
    align-items: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #334155;
    font-size: 0.8rem;
    padding: 0.45rem 0.85rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}
.gov-notice-strip strong {
    color: #0f172a;
}

/* Enterprise Metric Tiles */
.ent-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    margin-bottom: 0.75rem;
}
.ent-card-urgent {
    background: #fff5f5;
    border: 1px solid #fed7d7;
    border-left: 4px solid #e53e3e;
}
.ent-card-warning {
    background: #fffaf0;
    border: 1px solid #feebc8;
    border-left: 4px solid #dd6b20;
}
.ent-card-info {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3182ce;
}
.ent-metric-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 0.2rem;
}
.ent-metric-val {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}
.ent-metric-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.15rem;
}

/* Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.18rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
}
.badge-neutral { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
.badge-blue { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.badge-amber { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
.badge-red { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.badge-green { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }

/* Lineage Flow Box */
.lineage-flow-box {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
}

/* Compact Expander */
div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 0.65rem !important;
}
div[data-testid="stExpander"] details summary p {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #1e293b !important;
}

/* Tighten button padding */
button[kind="primary"] {
    background-color: #1e40af !important;
    border-color: #1e40af !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}

/* Clean up markdown spacing */
p {
    margin-bottom: 0.4rem !important;
}
</style>
"""


def render_page_header(breadcrumb: str, title: str, description: str) -> None:
    """Render consistent enterprise page header with restrained typography."""
    st.markdown(
        f"""
        <div class="page-breadcrumb">{html.escape(breadcrumb)}</div>
        <h1 class="page-title">{html.escape(title)}</h1>
        <div class="page-description">{html.escape(description)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(gate: str) -> str:
    """Return a standardized markdown badge for a given gate."""
    badges = {
        "G1": "🔹 **G1 — Extraction Quality Review**",
        "G2": "🔹 **G2 — Ambiguous Comparison Review**",
        "G3": "🔹 **G3 — Unmatched Clinical Finding**",
        "G4": "🔸 **G4 — Authorized Governance Sign-off**",
        "G5": "🔺 **G5 — SLA Escalation (Overdue)**"
    }
    return badges.get(gate.upper(), f"**{gate}**")


def render_processing_chain(
    current_stage: str = "monitoring",
    held_gate: Optional[str] = None,
    pipeline_status: str = "held",
    is_no_match: bool = False,
    gate: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Render a compact horizontal pipeline stepper."""
    eff_gate = (gate or held_gate or "").upper()
    eff_status = (status or pipeline_status).lower()
    stage = (current_stage or "").lower()

    stages = ["Ingestion", "Extraction", "Comparison", "Impact", "Brief", "Governance"]

    # Determine the index of the active stage
    if eff_gate == "G1" or (eff_status == "held" and stage == "extraction"):
        active_idx = 1
    elif eff_gate in ["G2", "G3"] or (eff_status == "held" and stage == "comparison"):
        active_idx = 2
    elif eff_gate == "G4" or stage in ("briefing", "governance", "draft", "assigned", "in_review") or eff_status == "pending":
        active_idx = 5
    elif stage in ("closure", "closed", "decided") or eff_status == "complete":
        active_idx = 6  # all done
    else:
        try:
            active_idx = stages.index(stage.capitalize())
        except ValueError:
            active_idx = 0

    steps = []
    for i, s in enumerate(stages):
        if i < active_idx:
            steps.append(f"✅ ~{s}~")
        elif i == active_idx:
            gate_label = f" ({eff_gate})" if eff_gate else ""
            steps.append(f"**🟢 {s}{gate_label}**")
        else:
            steps.append(f"⏸️ {s}")

    chain_md = " ➔ ".join(steps)
    if eff_gate:
        chain_md += f"  \n\n*Held at:* {render_badge(eff_gate)}"

    st.markdown(chain_md)
    return chain_md


def render_header() -> None:
    """Render compact enterprise clinical governance boundary notice."""
    st.markdown(
        """
        <div class="gov-notice-strip">
            <span style="margin-right: 8px; font-size: 1rem;">🛡️</span>
            <div><strong>Clinical Governance Notice:</strong> System prepares evidence, identifies protocol gaps, and calculates impact. 
            <strong>Authorized clinicians make all governance decisions.</strong> No recommendation or protocol is modified automatically.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gate_legend() -> None:
    """Render compact safety gate reference in sidebar."""
    st.sidebar.markdown(
        """
        <div style="margin-top: 1rem; border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <div style="font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;">Safety Gates</div>
            <div style="display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-neutral" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G1</span>
                    <span style="color: #475569; font-weight: 500;">Extraction Review</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-neutral" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G2</span>
                    <span style="color: #475569; font-weight: 500;">Comparison Review</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-neutral" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G3</span>
                    <span style="color: #475569; font-weight: 500;">Unmatched Protocol</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-amber" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G4</span>
                    <span style="color: #475569; font-weight: 500;">Governance Decision</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem;">
                    <span class="badge-pill badge-red" style="min-width: 28px; text-align: center; padding: 0.1rem 0.3rem;">G5</span>
                    <span style="color: #475569; font-weight: 500;">SLA Escalation</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(session_factory: sessionmaker[Session]) -> None:
    """Render Dashboard overview metrics as an Executive Attention Center."""
    render_page_header(
        breadcrumb="OVERVIEW › ATTENTION CENTER",
        title="Governance & Pipeline Dashboard",
        description="Real-time clinical review triage, SLA compliance, and surveillance operations."
    )
    metrics = get_dashboard_metrics(session_factory)

    # ------------------------------------------------------------------
    # SECTION 1: NEEDS ATTENTION (Top Priority Viewport)
    # ------------------------------------------------------------------
    st.markdown("<h3 style='margin-top: 0; padding-top: 0; font-size: 1.15rem; color: #1e293b;'>⚠️ Needs Attention</h3>", unsafe_allow_html=True)

    col_urgent1, col_urgent2 = st.columns(2)
    with col_urgent1:
        if metrics["overdue_reviews"] > 0:
            st.markdown(
                f"""
                <div class="ent-card ent-card-urgent">
                    <div class="ent-metric-title">Overdue SLA Reviews (G5 Escalated)</div>
                    <div class="ent-metric-val" style="color: #b91c1c;">{metrics['overdue_reviews']}</div>
                    <div class="ent-metric-sub">Immediate committee or CMO review required</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="ent-card ent-card-info">
                    <div class="ent-metric-title">Overdue SLA Reviews (G5)</div>
                    <div class="ent-metric-val" style="color: #15803d;">0</div>
                    <div class="ent-metric-sub">All assigned reviews are within SLA target</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_urgent2:
        if metrics["awaiting_governance"] > 0:
            st.markdown(
                f"""
                <div class="ent-card ent-card-warning">
                    <div class="ent-metric-title">Pending Governance Decisions (G4)</div>
                    <div class="ent-metric-val" style="color: #b45309;">{metrics['awaiting_governance']}</div>
                    <div class="ent-metric-sub">Evidence briefs awaiting clinician sign-off</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="ent-card ent-card-info">
                    <div class="ent-metric-title">Pending Governance Decisions (G4)</div>
                    <div class="ent-metric-val" style="color: #15803d;">0</div>
                    <div class="ent-metric-sub">No briefs currently awaiting formal decision</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Human Review Queues")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Held at G1 (Extraction)</div>
                <div class="ent-metric-val">{metrics['held_g1']}</div>
                <div class="ent-metric-sub">Low confidence or unextracted sources</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q2:
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Held at G2 (Comparison)</div>
                <div class="ent-metric-val">{metrics['held_g2']}</div>
                <div class="ent-metric-sub">Ambiguous diffs requiring human check</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q3:
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Held at G3 (No Match)</div>
                <div class="ent-metric-val">{metrics['held_g3']}</div>
                <div class="ent-metric-sub">New external evidence without baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # SECTION 2: RECENT MONITORED SOURCE CHANGES
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Recent Monitored Source Activity")
    docs = get_source_documents(session_factory)
    if docs:
        recent_df = [
            {
                "Source Identifier": d["source_identifier"],
                "Version": f"v{d['source_version']}",
                "Change Status": d["change_status"].replace("_", " ").title(),
                "Pipeline Status": d["status"].title(),
                "Changes Extracted": d["change_records_count"],
                "Last Retrieved": str(d["last_retrieved"])[:19],
            }
            for d in docs[:5]
        ]
        st.dataframe(
            recent_df,
            use_container_width=True,
            column_config={
                "Source Identifier": st.column_config.TextColumn("Source", width="medium"),
                "Version": st.column_config.TextColumn("Version", width="small"),
                "Change Status": st.column_config.TextColumn("Change", width="small"),
                "Pipeline Status": st.column_config.TextColumn("Pipeline", width="small"),
                "Changes Extracted": st.column_config.NumberColumn("Extracted", format="%d", width="small"),
                "Last Retrieved": st.column_config.TextColumn("Last Retrieved", width="medium"),
            },
        )
    else:
        st.info("No sources have been ingested yet.")

    # ------------------------------------------------------------------
    # SECTION 3: OPERATIONAL & GOVERNANCE SUMMARY
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Operational & System Metrics")
    col_op1, col_op2, col_op3, col_op4 = st.columns(4)
    with col_op1:
        st.metric("Total Monitored Sources", metrics["total_documents"])
    with col_op2:
        st.metric("Sources Processed", metrics["processed_documents"])
    with col_op3:
        st.metric("Decided Briefs", metrics["decided_briefs"])
    with col_op4:
        st.metric("Closed Briefs", metrics["closed_briefs"])


def render_upload_result_card(res: Dict[str, Any]) -> None:
    """Render the structured outcome card after PDF or URL upload and processing."""
    st.markdown("---")
    input_type = res.get("input_type", "URL" if res.get("source_url") else "PDF")

    # Bot-check / Content challenge handling
    if not res.get("usable_clinical_content", True):
        st.error(f"❌ [{input_type}] Ingestion Blocked: Browser Challenge / Bot-Check Detected")
        st.warning(
            "🛑 **Source Content Validation Policy Violation**\n\n"
            f"**Reason:** {res.get('result_message')}\n\n"
            "- **Retrieved (HTTP Connection):** YES\n"
            "- **Usable Clinical Content:** NO\n"
            "- **Action Taken:** Challenge page rejected before normalization. Zero challenge data sent to extraction or LLM.\n"
            "- **Anti-Bot Policy:** CKEA strictly does not bypass bot protection challenges or store cookie wall pages."
        )
        return

    gate = res.get("gate_status")
    success = res.get("success", True)
    msg = res.get("result_message", "")
    change_status = res.get("change_status", "new_source")
    source_ver = res.get("source_version", "1.0")
    pipe_status = res.get("pipeline_status", "Held").capitalize()

    st.markdown("#### Ingestion & Processing Outcome")

    # Status notice treatment:
    # Important: "No source change detected" must NOT look like a green success alert. It is a normal informational state.
    is_no_change = "no source change" in msg.lower() or change_status == "no_change"

    if not success:
        st.error(f"❌ [{input_type}] {msg}")
        if res.get("error"):
            st.caption(f"Details: {res.get('error')}")
    elif is_no_change:
        st.info(f"ℹ️ [{input_type}] {msg}")
    elif gate == "G1":
        st.info(f"🔵 **G1 Safety Gate:** {msg}")
    elif gate == "G2":
        st.warning(f"🟡 **G2 Safety Gate:** {msg}")
    elif gate == "G3":
        st.warning(f"🟠 **G3 Safety Gate:** {msg}")
    elif gate == "G4":
        st.info(f"🔴 **G4 Safety Gate:** {msg}")
    else:
        st.success(f"✅ [{input_type}] {msg}")

    # Compact Badges Strip: RETRIEVAL STATUS | PIPELINE STATUS | SOURCE VERSION | CHANGE STATUS | GATE
    retrieval_status = res.get("ingestion_status", "Complete")
    gate_badge_label = gate if gate else "None"
    change_readable = change_status.replace("_", " ").title()

    st.markdown(
        f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 12px 0;">
            <span class="badge-pill badge-neutral">RETRIEVAL: {html.escape(retrieval_status)}</span>
            <span class="badge-pill badge-blue">PIPELINE: {html.escape(pipe_status)}</span>
            <span class="badge-pill badge-neutral">VERSION: v{html.escape(str(source_ver))}</span>
            <span class="badge-pill badge-neutral">CHANGE: {html.escape(change_readable)}</span>
            <span class="badge-pill {'badge-amber' if gate else 'badge-neutral'}">GATE: {html.escape(gate_badge_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    resolved_url = res.get("resolved_source_url")
    if resolved_url:
        st.caption(f"🔗 **Resolved Full-Text URL:** [{resolved_url}]({resolved_url})")
    elif res.get("source_url"):
        st.caption(f"🔗 **Original Source URL:** `{res['source_url']}`")

    if res.get("title"):
        st.markdown(f"**Document Title:** {res['title']}")

    # Render processing chain stepper
    render_processing_chain(
        current_stage=res.get("current_stage", "extraction"),
        held_gate=res.get("gate_status"),
        pipeline_status=res.get("pipeline_status", "held"),
        is_no_match=(res.get("gate_status") == "G3"),
    )

    # Secondary details in compact expander
    with st.expander("⚙️ Technical Provenance & Document Details", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Document ID", (res.get("document_id") or "N/A")[:12] + "...")
        with c2:
            st.metric("Content Type", res.get("content_classification", "Web Page"))
        with c3:
            st.metric("Extracted Size", f"{res.get('extracted_text_size', 0):,} B")
        with c4:
            st.metric("SHA-256", (res.get("sha256_hash", "")[:12] + "...") if res.get("sha256_hash") else "N/A")

        if gate == "G1":
            st.info(
                "🛑 **Why paused at G1?** "
                f"{res.get('pipeline_reason') or 'No actionable clinical recommendation could be confidently extracted.'} "
                "Source is saved with verified SHA-256 integrity, but extraction requires authorized human review."
            )

    saved_path_str = res.get("saved_path")
    if saved_path_str and Path(saved_path_str).exists():
        try:
            page_texts = extract_page_texts(Path(saved_path_str))
            if page_texts:
                with st.expander("📄 Document Text Preview", expanded=False):
                    for pt in page_texts[:2]:
                        st.markdown(f"**Page {pt.page_number}**")
                        snippet = pt.text[:600] + ("..." if len(pt.text) > 600 else "")
                        st.text(snippet if snippet.strip() else "[Empty or non-text page]")
        except Exception as e:
            logger.debug("Read-only preview generation skipped: %s", e)


def render_sources_view(
    session_factory: sessionmaker[Session],
    config: Optional[AppConfig] = None,
) -> None:
    """Render Document and Monitoring inspection view with inventory prioritization."""
    render_page_header(
        breadcrumb="SURVEILLANCE › SOURCES",
        title="Clinical Source Documents",
        description="Continuous surveillance of external clinical guidelines, literature URLs, and protocol evidence."
    )
    cfg = config or load_config()

    docs = get_source_documents(session_factory)

    # ------------------------------------------------------------------
    # COMPACT TOOLBAR & CONTROLS
    # ------------------------------------------------------------------
    col_tool1, col_tool2, col_tool3 = st.columns([2, 3, 2])
    with col_tool1:
        st.markdown(
            f"""
            <div style="padding-top: 6px;">
                <span class="badge-pill badge-neutral">Total Sources: <strong>{len(docs)}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_tool2:
        search_kw = st.text_input(
            "Filter sources",
            placeholder="Search by identifier or source...",
            label_visibility="collapsed",
            key="source_search_input",
        )
    with col_tool3:
        show_add = st.checkbox("➕ Add Clinical Source", value=False, key="toggle_add_source")

    # ------------------------------------------------------------------
    # ADD CLINICAL SOURCE EXPANDABLE CARD
    # ------------------------------------------------------------------
    if show_add:
        with st.container():
            st.markdown(
                """
                <div class="ent-card" style="margin-top: 8px;">
                    <div class="ent-metric-title" style="margin-bottom: 8px;">Add Clinical Source Document</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            mode = st.radio(
                "Ingestion Mode:",
                options=["Web URL", "Upload PDF"],
                horizontal=True,
                label_visibility="collapsed",
                key="source_mode_radio",
            )

            if mode == "Web URL":
                col_u1, col_u2 = st.columns([4, 1])
                with col_u1:
                    url_input = st.text_input(
                        "Clinical Source Document URL",
                        placeholder="https://example.org/guidelines/diabetes_protocol.pdf or web URL",
                        label_visibility="collapsed",
                        key="source_url_input",
                    )
                with col_u2:
                    fetch_clicked = st.button("Fetch Source", type="primary", use_container_width=True, key="fetch_url_btn")

                if fetch_clicked:
                    if not url_input or not url_input.strip():
                        st.warning("Please provide a valid HTTP or HTTPS clinical source URL.")
                    else:
                        with st.spinner("Fetching URL and processing through pipeline..."):
                            try:
                                res = handle_source_url_upload(
                                    url=url_input.strip(),
                                    session_factory=session_factory,
                                    config=cfg,
                                )
                                st.session_state["upload_result_info"] = res
                            except (URLIngestionError, ValueError) as ue:
                                st.error(f"URL Ingestion Error: {ue}")
                            except Exception as e:
                                logger.exception("Unexpected error in URL ingestion handler: %s", e)
                                st.error("URL retrieval succeeded, but processing failed. Please check system logs.")

            else:  # Upload PDF
                col_p1, col_p2 = st.columns([3, 1])
                with col_p1:
                    uploaded_file = st.file_uploader(
                        "Choose PDF",
                        type=["pdf"],
                        accept_multiple_files=False,
                        label_visibility="collapsed",
                        key="source_pdf_file_uploader",
                    )
                with col_p2:
                    upload_clicked = st.button("Process Document", type="primary", use_container_width=True, key="upload_process_btn")

                # Source identity is never the filename: the reviewer states whether this PDF
                # starts a new source or is a new version of an existing one.
                existing_sources = {
                    d["source_identifier"]: d for d in get_source_documents(session_factory)
                }
                NEW_SOURCE = "New source"
                target_source = st.selectbox(
                    "Source",
                    options=[NEW_SOURCE] + list(existing_sources),
                    format_func=lambda sid: sid if sid == NEW_SOURCE else (
                        f"New version of: {existing_sources[sid].get('title') or sid} "
                        f"(v{existing_sources[sid]['source_version']})"
                    ),
                    key="upload_target_source",
                )

                if upload_clicked:
                    if uploaded_file is None:
                        st.warning("Please select a PDF file to upload.")
                    else:
                        with st.spinner("Processing PDF through clinical pipeline..."):
                            try:
                                res = handle_source_pdf_upload(
                                    uploaded_name=uploaded_file.name,
                                    uploaded_bytes=uploaded_file.getvalue(),
                                    session_factory=session_factory,
                                    config=cfg,
                                    existing_source_identifier=None if target_source == NEW_SOURCE else target_source,
                                )
                                st.session_state["upload_result_info"] = res
                            except UnreadableSourceError as ue:
                                st.error(f"Unreadable PDF (recorded as an ingestion failure): {ue}")
                            except UnknownSourceError as ue:
                                st.error(str(ue))
                            except ValueError as ve:
                                st.error(f"Validation Error: {ve}")
                            except Exception as e:
                                logger.exception("Unexpected error in upload handler: %s", e)
                                st.error("Upload succeeded, but processing failed. Please check system logs.")

        if "upload_result_info" in st.session_state:
            render_upload_result_card(st.session_state["upload_result_info"])

    # ------------------------------------------------------------------
    # PRIMARY: SOURCE INVENTORY TABLE (Above Fold)
    # ------------------------------------------------------------------
    st.markdown("### 📋 Source Inventory")
    if not docs:
        st.info("No clinical sources currently in the database. Use '➕ Add Clinical Source' above to ingest a guideline.")
        return

    # Filter docs by search keyword if provided
    filtered_docs = docs
    if search_kw and search_kw.strip():
        kw = search_kw.strip().lower()
        filtered_docs = [
            d for d in docs
            if kw in d["source_identifier"].lower() or kw in str(d["source_path"]).lower()
        ]

    df_rows = []
    for d in filtered_docs:
        meta_type = "PDF" if str(d.get("source_path", "")).lower().endswith(".pdf") else "URL"
        chg_text = d["change_status"].replace("_", " ").title()
        df_rows.append({
            "Source": d["source_identifier"],
            "Type": meta_type,
            "Version": f"v{d['source_version']}",
            "Change": chg_text,
            "Pipeline": d["status"].title(),
            "Gate": "G1" if d["status"] == "held" else "None",
            "Last Retrieved": str(d["last_retrieved"])[:16],
            "id": d["id"],
        })

    st.dataframe(
        df_rows,
        use_container_width=True,
        column_config={
            "Source": st.column_config.TextColumn("Source Identifier", width="large"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Version": st.column_config.TextColumn("Version", width="small"),
            "Change": st.column_config.TextColumn("Change Status", width="medium"),
            "Pipeline": st.column_config.TextColumn("Pipeline", width="small"),
            "Gate": st.column_config.TextColumn("Gate", width="small"),
            "Last Retrieved": st.column_config.TextColumn("Last Retrieved", width="medium"),
            "id": None,  # Hide internal primary key from table
        },
    )

    # ------------------------------------------------------------------
    # SECONDARY: DETAIL VIEW & PROVENANCE
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 Source Detail & Provenance")

    doc_ids = [d["id"] for d in filtered_docs]
    selected_id = st.selectbox(
        "Select source to inspect provenance:",
        options=doc_ids,
        format_func=lambda x: next(f"{d['source_identifier']} (v{d['source_version']})" for d in filtered_docs if d["id"] == x),
        key="source_detail_selector",
    )

    if selected_id:
        doc_item = next(d for d in filtered_docs if d["id"] == selected_id)

        # Level 1 Hierarchy: PRIMARY (Source title, Version, Change status, Pipeline status, Gate)
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.metric("Source Title", doc_item["source_identifier"][:24])
        with col_p2:
            st.metric("Current Version", f"v{doc_item['source_version']}")
        with col_p3:
            st.metric("Change Status", doc_item["change_status"].replace("_", " ").title())
        with col_p4:
            st.metric("Pipeline Gate", "G1 (Held)" if doc_item["status"] == "held" else doc_item["status"].title())

        # Lineage Timeline
        prev_ver_id = doc_item.get("previous_source_version_id")
        lineage_label = f"v{prev_ver_id or 'None'} (Previous) ➔ v{doc_item['source_version']} (Current)"
        st.markdown(
            f"""
            <div class="lineage-flow-box">
                <strong>Source Lineage:</strong> {html.escape(lineage_label)}
                <span class="badge-pill badge-neutral" style="margin-left: auto;">Status: {html.escape(doc_item['change_status'].replace('_', ' ').title())}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Level 2 Hierarchy: SECONDARY (Collapsible Technical Provenance)
        with st.expander("⚙️ Technical Provenance & Integrity", expanded=False):
            t1, t2 = st.columns(2)
            with t1:
                st.write(f"**Document ID:** `{doc_item['id']}`")
                st.write(f"**Storage Path:** `{doc_item['source_path']}`")
                st.write(f"**First Seen:** `{str(doc_item['first_seen'])[:19]}`")
                st.write(f"**Last Retrieved:** `{str(doc_item['last_retrieved'])[:19]}`")
                st.write(f"**Retrieval Method:** `{doc_item['retrieval_provider']}`")
                if doc_item.get("routing_decision"):
                    st.write(f"**Routing Decision:** `{doc_item['routing_decision']}`")
                if doc_item.get("resolved_source_url"):
                    st.write(f"**Resolved URL:** `{doc_item['resolved_source_url']}`")
                for warning in doc_item.get("provider_warnings") or []:
                    st.warning(f"Retrieval provider warning: {warning}")

        source_diff = doc_item.get("source_diff") or {}
        if source_diff.get("changes"):
            with st.expander("Source Evolution (previous version → this version)", expanded=False):
                st.caption(
                    "What changed in the external source itself. This is separate from the "
                    "institutional protocol comparison."
                )
                for change in source_diff["changes"]:
                    kind = change["change_type"].upper()
                    heading = change["heading"]
                    if change.get("old_heading"):
                        heading = f"{change['old_heading']} → {heading}"
                    st.markdown(f"**{kind}** · {html.escape(heading)}")
                    if change["change_type"] == "removed":
                        st.warning("Removed from the source; not re-extracted. Review whether the protocol relied on it.")
                        st.text(change["old_text"][:1500])
                    elif change["change_type"] == "modified":
                        c_old, c_new = st.columns(2)
                        c_old.text(change["old_text"][:1500])
                        c_new.text(change["new_text"][:1500])
                    else:
                        st.text(change["new_text"][:1500])
            with t2:
                st.write(f"**Current SHA-256:** `{doc_item['sha256_hash']}`")
                prev_sha = doc_item.get("previous_sha256_hash")
                st.write(f"**Previous SHA-256:** `{prev_sha if prev_sha else 'None (First Ingestion)'}`")
                st.write(f"**Extracted Change Records:** `{doc_item['change_records_count']}`")


def render_changes_view(
    session_factory: sessionmaker[Session],
    config: Optional[AppConfig] = None,
) -> None:
    """Render Extracted Changes & Protocol Comparison view with explicit 3-section architecture."""
    cfg = config or load_config()
    render_page_header(
        breadcrumb="CLINICAL ANALYSIS › COMPARISON",
        title="Clinical Recommendations & Protocol Comparison",
        description="Side-by-side verification of external published evidence against institutional hospital baselines."
    )

    changes = get_changes_with_gaps(session_factory, config=cfg)
    if not changes:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Comparison Workspace</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No clinical recommendations are currently undergoing active comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        docs = get_source_documents(session_factory)
        held_g1_docs = [d for d in docs if d["status"] == "held"]
        if held_g1_docs:
            st.warning(
                f"🛑 **Pipeline HELD at Human Gate G1:** {len(held_g1_docs)} monitored source document(s) "
                "could not yield actionable recommendations with sufficient confidence."
            )
            st.markdown(
                """
                **Why has Protocol Comparison not started?**
                - Source retrieval and file normalization succeeded with verified SHA-256 integrity.
                - LLM extraction did **not** produce a reviewable clinical recommendation meeting confidence (>= 0.70) or quote-verification rules.
                - Under CKEA safety policy:
                  - ⏸️ Comparison has **NOT** started
                  - ⏸️ Impact scoring has **NOT** started
                  - ⏸️ No ChangeBrief was created
                  - 🔵 Human extraction review is required at **Gate G1**
                """
            )
        return

    for c in changes:
        status_label = c["change_status"]
        is_held_g1 = status_label == ChangeStatus.HELD_FOR_G1.value
        is_held_g2 = status_label == ChangeStatus.HELD_FOR_G2.value or c["gap_status"] == GapStatus.REVIEW_REQUIRED.value
        is_held_g3 = status_label == ChangeStatus.HELD_FOR_G3.value or c["comparison_result"] == "no_match"

        expander_title = (
            f"Recommendation: {c['verbatim_text'][:70]}... "
            f"[{c['source_type']}: {c['source_identifier']}]"
        )

        with st.expander(expander_title, expanded=True):
            # Traceability Stepper
            gate_tag = "G1" if is_held_g1 else ("G2" if is_held_g2 else ("G3" if is_held_g3 else "G4"))
            render_processing_chain(
                current_stage="comparison" if not is_held_g1 else "extraction",
                held_gate=gate_tag if (is_held_g1 or is_held_g2 or is_held_g3) else None,
                pipeline_status="held" if (is_held_g1 or is_held_g2 or is_held_g3) else "in_progress",
                is_no_match=is_held_g3,
            )

            # ------------------------------------------------------------------
            # 1. SOURCE EVOLUTION (External Lineage Timeline)
            # ------------------------------------------------------------------
            st.markdown("#### 1. Source Evolution Lineage")
            prev_ver = c.get("previous_source_version_id")
            lineage_str = f"v{prev_ver or '1.0'} Previous ➔ v{c.get('source_version', '1.0')} Current"
            st.markdown(
                f"""
                <div class="lineage-flow-box">
                    <strong>Version Lineage:</strong> {html.escape(lineage_str)}
                    <span class="badge-pill badge-neutral" style="margin-left: auto;">Change: {html.escape(c['change_status'].replace('_', ' ').title())}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------------------------
            # 2. INSTITUTIONAL PROTOCOL COMPARISON (Side-by-Side Cards)
            # ------------------------------------------------------------------
            st.markdown("#### 2. Institutional Protocol Comparison")
            card_left, card_right = st.columns(2)

            with card_left:
                st.markdown(
                    f"""
                    <div class="ent-card" style="border-top: 3px solid #2563eb;">
                        <div class="ent-metric-title">External Evidence (Published)</div>
                        <div style="font-size: 0.95rem; font-style: italic; color: #1e293b; margin: 6px 0;">
                            "{html.escape(c['verbatim_text'])}"
                        </div>
                        <div style="font-size: 0.78rem; color: #475569; margin-top: 8px;">
                            • <strong>Type:</strong> {html.escape(str(c['recommendation_type']))}<br>
                            • <strong>Population:</strong> {html.escape(str(c['target_population']))}<br>
                            • <strong>Intervention:</strong> {html.escape(str(c['intervention']))}<br>
                            • <strong>Grade:</strong> {html.escape(str(c['evidence_grade']))} | <strong>Confidence:</strong> {c['confidence']:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_right:
                if not c["has_gap"]:
                    st.info("Institutional comparison pending.")
                elif not c["is_match"] or c["comparison_result"] == "no_match":
                    st.markdown(
                        """
                        <div class="ent-card ent-card-warning" style="border-top: 3px solid #dd6b20;">
                            <div class="ent-metric-title">Institutional Protocol Baseline</div>
                            <div style="font-size: 0.95rem; color: #b45309; margin: 6px 0;">
                                No matching institutional protocol section found in hospital library.
                            </div>
                            <div style="font-size: 0.78rem; color: #78350f;">
                                Gate G3 committee review required to determine if a new protocol must be authored.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="ent-card" style="border-top: 3px solid #059669;">
                            <div class="ent-metric-title">Institutional Protocol Baseline</div>
                            <div style="font-size: 0.95rem; font-style: italic; color: #1e293b; margin: 6px 0;">
                                "{html.escape(str(c['protocol_section_text']))}"
                            </div>
                            <div style="font-size: 0.78rem; color: #475569; margin-top: 8px;">
                                • <strong>Protocol:</strong> {html.escape(str(c['matched_protocol_id']))} (v{html.escape(str(c['matched_protocol_version']))})<br>
                                • <strong>Section:</strong> {html.escape(str(c['protocol_section_id']))} — {html.escape(str(c['protocol_section_heading']))}<br>
                                • <strong>Title:</strong> {html.escape(str(c['protocol_title']))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ------------------------------------------------------------------
            # 3. COMPARISON RESULT (Synthesized Gap Finding)
            # ------------------------------------------------------------------
            st.markdown("#### 3. Comparison Result & Gap Synthesis")
            if not c["has_gap"]:
                st.info("Pending comparison execution.")
            elif not c["is_match"] or c["comparison_result"] == "no_match":
                st.error(f"🟠 **No Matching Protocol Section** {render_badge('G3')} — Committee review required.")
            else:
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.write(f"**Result:** `{c['comparison_result'].upper()}`")
                    st.write(f"**Difference Category:** `{c['difference_type']}`")
                    st.write(f"**Confidence:** `{c['comparison_confidence'] if c['comparison_confidence'] is not None else 'N/A'}`")
                    st.write(f"**Contradiction:** `{c['contradiction_status']}`")
                with col_res2:
                    st.markdown("**Specific Clinical Difference:**")
                    st.markdown(f"> {c['specific_difference']}")

                    if is_held_g2:
                        st.warning("🟡 **G2 Safety Gate:** Ambiguous comparison. Human clinical review required.")
                    elif c["comparison_result"] == "gap":
                        st.info("✅ **Confirmed Gap:** Proceeded to deterministic impact scoring.")
                    elif c["comparison_result"] == "no_gap":
                        st.success("✅ **No Material Gap:** Source evidence aligns with hospital protocol.")


def render_impact_view(session_factory: sessionmaker[Session]) -> None:
    """Render Deterministic Multidimensional Impact Assessments with compact metric cards."""
    render_page_header(
        breadcrumb="RISK TRIAGE › IMPACT ASSESSMENT",
        title="Deterministic Impact Assessments",
        description="Clinical risk scoring evaluated against YAML rule matrix. Zero LLM calculations."
    )

    impacts = get_impact_records(session_factory)
    if not impacts:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Impact Assessment Workspace</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No impact assessment records found.
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                    Impact Assessment executes after Protocol Comparison confirms a clinical gap.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for imp in impacts:
        tier_str = str(imp["tier"])
        badge_class = {
            "Critical": "badge-red",
            "High": "badge-amber",
            "Standard": "badge-blue",
            "Low": "badge-green",
        }.get(tier_str, "badge-neutral")

        with st.expander(f"Impact Tier: {tier_str} (Score: {imp['total_score']}/15) — {imp['source_identifier']} vs {imp['matched_protocol_id']}", expanded=True):
            # Top metadata bar
            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 10px;">
                    <span class="badge-pill {badge_class}">TIER: {html.escape(tier_str.upper())}</span>
                    <span class="badge-pill badge-neutral">SLA: {html.escape(str(imp['sla_deadline'])[:19])}</span>
                    <span class="badge-pill badge-neutral">PROTOCOL: {html.escape(str(imp['matched_protocol_id']))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Compact 5-Metric Tiles Row: Urgency, Evidence, Breadth, Total Score, Impact Tier
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Urgency</div>
                        <div class="ent-metric-val">{imp['clinical_urgency']}<span style="font-size: 0.9rem; color: #64748b;"> / 5</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Evidence</div>
                        <div class="ent-metric-val">{imp['evidence_strength']}<span style="font-size: 0.9rem; color: #64748b;"> / 5</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Breadth</div>
                        <div class="ent-metric-val">{imp['pathway_breadth']}<span style="font-size: 0.9rem; color: #64748b;"> / 5</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Total Score</div>
                        <div class="ent-metric-val">{imp['total_score']}<span style="font-size: 0.9rem; color: #64748b;"> / 15</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m5:
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-metric-title">Impact Tier</div>
                        <div class="ent-metric-val" style="font-size: 1.15rem;">{html.escape(tier_str)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Recommendation Excerpt
            st.markdown(f"**Extracted Recommendation:** \"{imp['verbatim_recommendation']}\"")

            # Expandable Scoring Engine Basis & Traceability
            with st.expander("⚙️ Scoring Engine Basis & Deterministic Rules", expanded=False):
                st.write(f"**Urgency Basis:** {imp['urgency_basis']}")
                st.write(f"**Evidence Basis:** {imp['evidence_basis']}")
                st.write(f"**Breadth Basis:** {imp['breadth_basis']}")
                st.write(f"**Applied Rule IDs:** `{', '.join(imp['rule_ids'])}`")
                st.write(f"**Scoring YAML Version:** `{imp['scoring_yaml_version']}`")


def render_briefs_view(session_factory: sessionmaker[Session]) -> None:
    """Render 7-Section Change Briefs formatted as Executive Review Documents."""
    render_page_header(
        breadcrumb="GOVERNANCE › CHANGE BRIEFS",
        title="Clinical Change Briefs",
        description="Audit-grade evidence briefs prepared for authorized hospital governance review."
    )

    briefs = get_brief_summaries(session_factory)
    if not briefs:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Briefs Console</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No ChangeBriefs generated yet.
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                    Briefs are generated when a confirmed gap passes multidimensional impact scoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    selected_brief_id = st.selectbox(
        "Select Change Brief:",
        options=[b["brief_id"] for b in briefs],
        format_func=lambda b_id: f"{b_id[:8]}... | Status: {next(b['status'] for b in briefs if b['brief_id'] == b_id)} | Tier: {next(b['tier'] for b in briefs if b['brief_id'] == b_id)} | Source: {next(b['source_identifier'] for b in briefs if b['brief_id'] == b_id)}",
    )

    if not selected_brief_id:
        return

    current_brief_info = next(b for b in briefs if b["brief_id"] == selected_brief_id)

    # ------------------------------------------------------------------
    # EXECUTIVE REVIEW DOCUMENT HEADER (4 Primary Summary Metrics)
    # ------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="ent-card" style="border-left: 4px solid #1e40af; margin-top: 8px;">
            <div class="ent-metric-title">Executive Governance Dossier</div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px;">
                <div><strong>Change Detected:</strong> {html.escape(current_brief_info['difference_summary'][:60])}</div>
                <div><strong>Impact Tier:</strong> {html.escape(str(current_brief_info['tier']))}</div>
                <div><strong>Protocol Baseline:</strong> {html.escape(str(current_brief_info['matched_protocol']))}</div>
                <div><strong>Review Status:</strong> {html.escape(str(current_brief_info['status']).title())}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_processing_chain(
        current_stage="briefing",
        held_gate="G4",
        pipeline_status=current_brief_info["status"],
    )

    full_brief = get_brief_full_payload(session_factory, selected_brief_id)
    if not full_brief:
        st.error("Failed to load brief payload.")
        return

    payload = full_brief["structured_payload"]

    tab1, tab2, tab3 = st.tabs(["📑 Seven Required Sections", "🌐 Rendered HTML", "📄 Raw Markdown / JSON"])

    with tab1:
        with st.expander("1. What Changed", expanded=True):
            rec = payload.get("recommendation", {})
            st.write(f"**Verbatim Recommendation:** {rec.get('verbatim_text')}")
            st.write(f"**Target Population:** {rec.get('target_population')}")
            st.write(f"**Intervention:** {rec.get('intervention')}")
            st.write(f"**Evidence Grade:** {rec.get('evidence_grade')}")

        with st.expander("2. Current Protocol", expanded=True):
            prot = payload.get("protocol", {})
            st.write(f"**Protocol ID:** `{prot.get('protocol_id')}` (v{prot.get('protocol_version')})")
            st.write(f"**Section Heading:** {prot.get('section_heading')}")
            st.write(f"**Current Text:** {prot.get('section_text')}")

        with st.expander("3. Specific Difference", expanded=True):
            diff = payload.get("comparison", {})
            st.write(f"**Difference Category:** `{diff.get('difference_type')}`")
            st.write(f"**Clinical Difference:** {diff.get('specific_difference')}")

        with st.expander("4. Impact Assessment", expanded=False):
            imp = payload.get("impact", {})
            st.write(f"**Routing Tier:** `{imp.get('tier')}` | **Total Score:** `{imp.get('total_score')}`")
            st.write(f"**SLA Deadline:** `{imp.get('sla_deadline')}`")
            st.write(f"**Urgency Basis:** {imp.get('urgency_basis')}")

        with st.expander("5. Affected Workflows", expanded=False):
            wf = payload.get("workflow", {})
            st.write(f"**Impacted Care Pathways:** {', '.join(wf.get('affected_departments', [])) or 'General Medical'}")
            st.write(f"**EHR Changes Required:** {wf.get('workflow_description')}")

        with st.expander("6. Proposed Review Actions", expanded=False):
            actions = payload.get("proposed_actions", [])
            for a in actions:
                st.markdown(f"- **{a.get('action_type', 'Action')}:** {a.get('description')}")

        with st.expander("7. Source Excerpt", expanded=False):
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
    """Render Human Governance Decision View as an Enterprise Review Console (Gate G4)."""
    render_page_header(
        breadcrumb="DECISION DESK › GOVERNANCE (G4)",
        title="Clinical Governance Workspace",
        description="Authorized clinicians review evidence briefs and record formal, immutable governance decisions."
    )

    briefs = get_brief_summaries(session_factory)
    if not briefs:
        st.markdown(
            """
            <div class="ent-card ent-card-info">
                <div class="ent-metric-title">Governance Review Desk</div>
                <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">
                    No ChangeBriefs currently awaiting governance review.
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                    Change briefs appear here after passing extraction, comparison, and impact scoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    brief_options = [b["brief_id"] for b in briefs]
    selected_brief_id = st.selectbox(
        "Select Brief for Governance Review:",
        options=brief_options,
        format_func=lambda b_id: f"{b_id[:8]}... | Status: {next(b['status'] for b in briefs if b['brief_id'] == b_id)} | Reviewer: {next(b['assigned_reviewer'] for b in briefs if b['brief_id'] == b_id)} | Overdue: {'🚨 YES' if next(b['is_overdue'] for b in briefs if b['brief_id'] == b_id) else 'NO'}",
    )

    current_brief = next(b for b in briefs if b["brief_id"] == selected_brief_id)

    # ------------------------------------------------------------------
    # ASYMMETRIC SPLIT LAYOUT: LEFT ~70% (Dossier) | RIGHT ~30% (Decision Panel)
    # ------------------------------------------------------------------
    col_dossier, col_panel = st.columns([7, 3])

    with col_dossier:
        st.markdown("### 📋 Clinical Evidence Dossier")

        # Card 1: Source Evidence
        st.markdown(
            f"""
            <div class="ent-card" style="border-top: 3px solid #2563eb;">
                <div class="ent-metric-title">External Source Evidence</div>
                <div style="font-size: 0.95rem; font-style: italic; color: #1e293b; margin: 6px 0;">
                    "{html.escape(current_brief['verbatim_recommendation'])}"
                </div>
                <div style="font-size: 0.78rem; color: #475569;">
                    Source: <strong>{html.escape(current_brief['source_identifier'])}</strong> ({html.escape(current_brief['source_type'])})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Card 2: Hospital Protocol Baseline
        st.markdown(
            f"""
            <div class="ent-card" style="border-top: 3px solid #059669;">
                <div class="ent-metric-title">Institutional Protocol Baseline</div>
                <div style="font-size: 0.85rem; color: #1e293b; margin: 4px 0;">
                    Protocol ID: <strong>{html.escape(str(current_brief['matched_protocol']))}</strong> (v{html.escape(str(current_brief['protocol_version']))})
                </div>
                <div style="font-size: 0.85rem; color: #475569;">
                    Section: <strong>{html.escape(str(current_brief['protocol_section']))}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Card 3: Difference & Impact
        st.markdown(
            f"""
            <div class="ent-card">
                <div class="ent-metric-title">Clinical Difference & Risk Triage</div>
                <div style="font-size: 0.88rem; color: #1e293b; margin: 4px 0;">
                    <strong>Specific Difference:</strong> {html.escape(current_brief['difference_summary'])}
                </div>
                <div style="font-size: 0.8rem; color: #475569; margin-top: 4px;">
                    Impact Tier: <strong>{html.escape(str(current_brief['tier']))}</strong> | SLA Deadline: <strong>{html.escape(str(current_brief['sla_deadline']))}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Deep-dive expanders
        full_brief = get_brief_full_payload(session_factory, selected_brief_id)
        if full_brief and "structured_payload" in full_brief:
            payload = full_brief["structured_payload"]
            with st.expander("🔍 Deep Dive: Source Excerpt", expanded=False):
                st.write(payload.get("source_excerpt", {}).get("excerpt"))
            with st.expander("🏥 Deep Dive: Full Protocol Text", expanded=False):
                st.write(payload.get("protocol", {}).get("section_text"))

    with col_panel:
        st.markdown("### ✍️ Formal Review Panel")

        # Status & Overdue check
        if current_brief["is_overdue"]:
            st.markdown(
                """
                <div class="ent-card ent-card-urgent">
                    <div class="ent-metric-title" style="color: #b91c1c;">Review Status: Overdue</div>
                    <div style="font-size: 0.82rem; color: #7f1d1d;">🚨 G5 SLA Escalation Active</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="ent-card">
                    <div class="ent-metric-title">Reviewer & Status</div>
                    <div style="font-size: 0.85rem; color: #0f172a;">Assigned: <strong>{html.escape(current_brief['assigned_reviewer'])}</strong></div>
                    <div style="font-size: 0.78rem; color: #64748b;">Status: {html.escape(str(current_brief['status']).title())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.form("governance_review_action_form"):
            action = st.radio(
                "Governance Decision:",
                options=["approve", "reject", "defer"],
                format_func=lambda a: {
                    "approve": "Approve",
                    "reject": "Reject",
                    "defer": "Defer",
                }.get(a, a),
                horizontal=True,
            )

            reviewer_id = st.selectbox(
                "Authorized Reviewer Identity:",
                options=AUTHORIZED_REVIEWERS,
                index=0,
            )

            rationale = st.text_area(
                "Clinical Rationale (Mandatory):",
                placeholder="Enter clinical rationale, evidence justification, or committee discussion...",
                height=110,
            )

            follow_up_date_input = st.date_input(
                "Deferral Follow-up Date (If Deferring):",
                value=datetime.now(timezone.utc).date() + timedelta(days=14),
            )

            submit_decision = st.form_submit_button("Record Governance Decision", type="primary", use_container_width=True)

            if submit_decision:
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
                    st.toast(f"Governance decision recorded for {selected_brief_id}")
                    # Never imply protocol update was approved automatically
                    st.success(f"Governance decision recorded: {action.upper()}. Audit log updated.")
                except Exception as e:
                    st.error(f"Governance action rejected: {e}")


def render_audit_view(session_factory: sessionmaker[Session]) -> None:
    """Render Immutable Governance Audit Trail as an Enterprise Data Grid."""
    render_page_header(
        breadcrumb="COMPLIANCE › AUDIT LOG",
        title="Immutable Governance Audit Trail",
        description="Cryptographically auditable and append-only record of all system events and human decisions."
    )

    logs = get_audit_trail(session_factory)
    if not logs:
        st.info("No audit events recorded.")
        return

    df = []
    for log in logs:
        df.append({
            "Timestamp": str(log["timestamp"])[:19],
            "Actor": log["actor"],
            "Action": log["action"].upper(),
            "Entity Type": log["entity_type"].title(),
            "Entity ID": (log["entity_id"] or "")[:12] + "...",
            "Transition": f"{log['previous_status'] or 'None'} ➔ {log['new_status']}",
            "Reason / Rationale": log["reason"] or "N/A",
        })

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "Actor": st.column_config.TextColumn("Actor", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Entity Type": st.column_config.TextColumn("Entity Type", width="small"),
            "Entity ID": st.column_config.TextColumn("Entity ID"),
            "Transition": st.column_config.TextColumn("State Transition", width="medium"),
            "Reason / Rationale": st.column_config.TextColumn("Rationale / Details", width="large"),
        },
    )


def render_evaluation_reports_view() -> None:
    """Render Phase 12 Evaluation Framework Reports."""
    render_page_header(
        breadcrumb="VERIFICATION › EVALUATION",
        title="Evaluation Framework Reports",
        description="Read-only inspection of reproducible evaluation results across the 20-document synthetic corpus."
    )

    reports = load_evaluation_reports()
    report_names = [k for k in reports.keys() if reports[k] is not None]

    if not report_names:
        st.info("No evaluation reports found in data/evaluation/reports/.")
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

    # Inject modern enterprise CSS
    st.markdown(ENTERPRISE_CUSTOM_CSS, unsafe_allow_html=True)

    # Render compact global clinical governance boundary notice
    render_header()

    config = load_config()
    session_factory = get_db_session_factory(config)
    governance_agent = GovernanceAgent(session_factory=session_factory, config=config)

    # Compact Enterprise Product Sidebar Header
    st.sidebar.markdown(
        """
        <div style="padding: 0.2rem 0 0.6rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 0.6rem;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em;">CKEA Governance</div>
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 500;">Clinical Knowledge Evolution</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu = st.sidebar.radio(
        "Navigation",
        options=[
            "Dashboard",
            "Sources",
            "Clinical Changes",
            "Impact",
            "Change Briefs",
            "Governance",
            "Audit Log",
            "Evaluation",
        ],
        label_visibility="collapsed",
    )

    # Compact safety gate legend at bottom of sidebar
    render_gate_legend()

    if menu == "Dashboard":
        render_dashboard(session_factory)
    elif menu == "Sources":
        render_sources_view(session_factory, config=config)
    elif menu == "Clinical Changes":
        render_changes_view(session_factory)
    elif menu == "Impact":
        render_impact_view(session_factory)
    elif menu == "Change Briefs":
        render_briefs_view(session_factory)
    elif menu == "Governance":
        render_governance_view(session_factory, governance_agent)
    elif menu == "Audit Log":
        render_audit_view(session_factory)
    elif menu == "Evaluation":
        render_evaluation_reports_view()


if __name__ == "__main__":
    main()

