"""Integration tests for Phase 11 SLA Scheduler and G5 Escalation.

Demonstrates:
1. End-to-end integration with database persistence:
   - Create unresolved review assignment with expired deadline
   - Execute run_once()
   - Verify G5 escalation
   - Verify Notification and AuditLog existence
   - Verify governance decision remains unset
   - Verify brief remains unresolved
   - Re-execute run_once() to verify idempotency (zero duplicates)
   - Explicitly resolve brief via GovernanceAgent (start review -> approve -> close)
   - Re-execute run_once() to verify resolved work is skipped
2. Scheduler lifecycle verification:
   - start() -> scheduler running and single job registered
   - stop() -> scheduler stopped
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid
import pytest
from sqlalchemy import create_engine

from app.agents.governance_agent import GovernanceAgent
from app.models.database import Base, get_session_factory
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
from app.schemas.briefs import BriefStatus
from app.schemas.governance import ReviewAssignmentStatus, ReviewDecision
from app.schemas.impact import ImpactStatus, ImpactTier
from app.services.config_service import AppConfig
from app.services.reviewer_authorization import ReviewerAuthorizationService
from app.services.sla_scheduler import SLAScheduler


@pytest.fixture
def integration_env():
    """Set up an isolated SQLite in-memory database and real GovernanceAgent and SLAScheduler."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    config = AppConfig(
        database_url="sqlite:///:memory:",
        sla_scheduler_enabled=True,
        sla_check_interval_minutes=30,
        sla_timezone="UTC",
    )

    auth_service = ReviewerAuthorizationService()
    gov_agent = GovernanceAgent(
        session_factory=session_factory,
        auth_service=auth_service,
        config=config,
    )

    scheduler = SLAScheduler(
        session_factory=session_factory,
        governance_agent=gov_agent,
        config=config,
        interval_minutes=30,
        timezone_str="UTC",
        enabled=True,
    )

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "gov_agent": gov_agent,
        "scheduler": scheduler,
        "config": config,
    }

    if scheduler.is_running():
        scheduler.stop()
    engine.dispose()


def test_sla_scheduler_lifecycle_and_job_registration(integration_env):
    """Verify scheduler lifecycle: start -> running -> single job registered -> stop -> stopped."""
    scheduler = integration_env["scheduler"]

    # Initial state: not running
    assert scheduler.is_running() is False
    assert scheduler.get_job() is None

    # Start
    started = scheduler.start()
    assert started is True
    assert scheduler.is_running() is True

    # Verify single job registered
    job = scheduler.get_job()
    assert job is not None
    assert job.id == SLAScheduler.JOB_ID

    # Idempotent start
    assert scheduler.start() is True
    assert scheduler.is_running() is True

    # Stop
    stopped = scheduler.stop()
    assert stopped is True
    assert scheduler.is_running() is False
    assert scheduler.get_job() is None

    # Safe double stop
    assert scheduler.stop() is False
    assert scheduler.is_running() is False


def test_full_sla_escalation_lifecycle_integration(integration_env):
    """Demonstrate the complete SLA escalation lifecycle with real database persistence.

    Steps:
    1. Create unresolved review assignment
    2. Set SLA deadline in the past
    3. Execute run_once()
    4. Verify escalation
    5. Verify Notification exists
    6. Verify AuditLog exists
    7. Verify no governance decision was created
    8. Verify brief remains unresolved
    9. Execute run_once() again
    10. Verify no duplicate escalation records
    11. Close/resolve work through existing explicit GovernanceAgent
    12. Execute run_once() again
    13. Verify resolved work is no longer escalated
    """
    session_factory = integration_env["session_factory"]
    gov_agent = integration_env["gov_agent"]
    scheduler = integration_env["scheduler"]

    now = datetime.now(timezone.utc)
    overdue_deadline = now - timedelta(hours=3)

    # 1. Create unresolved review assignment
    # 2. Set SLA deadline in the past
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="KDIGO-CKD-2026.pdf",
            source_path="/data/sources/KDIGO-CKD-2026.pdf",
            sha256_hash="e" * 64,
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="In patients with CKD, initiate finerenone 20mg daily.",
            recommendation_type="pharmacotherapy",
            target_population="Adults with CKD",
            intervention="Finerenone 20mg",
            page=5,
            section="Pharmacotherapy",
            source_excerpt="In patients with CKD, initiate finerenone 20mg daily.",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            confidence=0.95,
            status="gap_confirmed",
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            candidate_protocol_section_ids=["PROT-CKD__1.0__SEC-3"],
            similarity=0.91,
            comparison_result="gap",
            comparison_confidence=0.94,
            difference_type="intervention_change",
            matched_protocol_id="PROT-CKD",
            matched_protocol_version="1.0",
            is_match=True,
            status="matched",
            schema_version="1.0",
        )
        session.add(gap)
        session.flush()

        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=4,
            evidence_strength=5,
            pathway_breadth=3,
            total_score=12,
            tier=ImpactTier.CRITICAL.value,
            routing_target="Rapid Nephrology Committee",
            sla_deadline=overdue_deadline,
            status=ImpactStatus.CALCULATED.value,
            schema_version="1.0",
        )
        session.add(impact)
        session.flush()

        brief = ChangeBrief(
            id=str(uuid.uuid4()),
            impact_record_id=impact.id,
            status=BriefStatus.ASSIGNED.value,
            rendered_file_path="/data/output/brief_ckd.md",
            rendered_file_hash="f" * 64,
            structured_payload={"protocol_text": "Protocol text unmutated."},
            schema_version="1.0",
        )
        session.add(brief)
        session.flush()

        assignment = ReviewAssignment(
            id=str(uuid.uuid4()),
            change_brief_id=brief.id,
            reviewer_role="Clinical Governance Lead",
            reviewer_id="dr_smith",
            status=ReviewAssignmentStatus.ASSIGNED.value,
            due_date=overdue_deadline,
            decision=None,
            rationale=None,
            schema_version="1.0",
        )
        session.add(assignment)
        session.commit()

        brief_id = brief.id
        assignment_id = assignment.id

    # 3. Execute run_once()
    summary1 = scheduler.run_once(reference_time=now)

    # 4. Verify escalation
    assert summary1.evaluated_count == 1
    assert summary1.overdue_count == 1
    assert summary1.escalated_count == 1

    with session_factory() as session:
        assignment_db = session.get(ReviewAssignment, assignment_id)
        brief_db = session.get(ChangeBrief, brief_id)
        notif = session.query(Notification).filter_by(related_entity_id=brief_id).first()
        audits = session.query(AuditLog).filter_by(entity_id=brief_id).all()
        escalation_audits = [a for a in audits if a.audit_metadata and a.audit_metadata.get("action") == "sla_escalation"]

        assert assignment_db.status == ReviewAssignmentStatus.ESCALATED.value

        # 5. Verify Notification exists
        assert notif is not None
        assert notif.notification_type == "escalation"
        assert notif.recipient == "Rapid Nephrology Committee"

        # 6. Verify AuditLog exists
        assert len(escalation_audits) == 1
        assert escalation_audits[0].actor == "system_sla_monitor"

        # 7. Verify NO governance decision was created
        assert assignment_db.decision is None

        # 8. Verify brief remains unresolved (not closed or decided)
        assert brief_db.status == BriefStatus.ASSIGNED.value

    # 9. Execute run_once() again
    summary2 = scheduler.run_once(reference_time=now)

    # 10. Verify NO duplicate escalation records
    assert summary2.evaluated_count == 1
    assert summary2.overdue_count == 1
    assert summary2.escalated_count == 0
    assert summary2.already_escalated_count == 1

    with session_factory() as session:
        notif_count = session.query(Notification).filter_by(related_entity_id=brief_id).count()
        assert notif_count == 1  # exactly 1, no duplicate

        audits = session.query(AuditLog).filter_by(entity_id=brief_id).all()
        escalation_audits = [a for a in audits if a.audit_metadata and a.audit_metadata.get("action") == "sla_escalation"]
        assert len(escalation_audits) == 1  # exactly 1, no duplicate

    # 11. Close/resolve work through the existing explicit GovernanceAgent
    gov_agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    gov_agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Finerenone evidence validated against institutional nephrology guideline.",
    )
    gov_agent.close_after_decision(
        change_brief_id=brief_id,
        actor="governance_chair",
        closure_note="Approved update deployed to clinic order sets.",
    )

    with session_factory() as session:
        resolved_brief = session.get(ChangeBrief, brief_id)
        assert resolved_brief.status == BriefStatus.CLOSED.value

    # 12. Execute run_once() again
    summary3 = scheduler.run_once(reference_time=now)

    # 13. Verify resolved work is no longer escalated
    assert summary3.evaluated_count == 0
    assert summary3.skipped_count == 1
    assert summary3.escalated_count == 0
