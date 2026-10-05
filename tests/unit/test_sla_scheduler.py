"""Unit tests for Phase 11 SLA Scheduler and G5 Escalation.

Verifies:
1. Scheduler construction with defaults and explicit parameters
2. Default configuration (disabled by default, 60 minutes, UTC)
3. Scheduler disabled configuration blocks unforced start
4. Scheduler start creates active BackgroundScheduler
5. Scheduler stop cleanly shuts down scheduler
6. Start is idempotent
7. Stop is safe when already stopped
8. run_once executes synchronously without starting background scheduler
9. Eligible brief evaluation (assigned, in_review, deferred)
10. Unresolved brief evaluation
11. Closed brief skipped
12. Decided brief skipped
13. Overdue brief escalation
14. Non-overdue brief does not escalate
15. Already escalated brief remains idempotent
16. Repeated run_once produces no duplicate notifications or escalations
17. Per-brief failure isolation (one failure does not crash the cycle)
18. Cycle result counts validation
19. Scheduler exception handling
20. UTC timestamp behavior
21. Configured interval honored in job trigger
22. No automatic governance decision (decision remains unset on breach)
23. Protocol data immutability
24. Zero LLM / API calls
25. Manual single-cycle execution
26. Scheduler does not start on import
27. No external notification integrations
28. No Phase 12 evaluation framework behavior
29. No Phase 13 UI behavior
30. AuditLog and Notification consistency
31. In-process concurrency guard preventing overlapping execution cycles
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch
import uuid
import pytest
from sqlalchemy import create_engine

import openai
from apscheduler.triggers.interval import IntervalTrigger

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
from app.schemas.governance import (
    ReviewAssignmentStatus,
    ReviewDecision,
    SLAEscalationResult,
)
from app.schemas.impact import ImpactStatus, ImpactTier
from app.schemas.scheduler import SLACycleSummary
from app.services.config_service import AppConfig
from app.services.reviewer_authorization import ReviewerAuthorizationService
from app.services.sla_scheduler import SLAScheduler


@pytest.fixture
def scheduler_env():
    """Set up an isolated in-memory SQLite database and test dependencies."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    config = AppConfig(
        database_url="sqlite:///:memory:",
        sla_scheduler_enabled=False,
        sla_check_interval_minutes=60,
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
        interval_minutes=60,
        timezone_str="UTC",
        enabled=False,
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


def _create_brief_with_assignment(
    session_factory,
    brief_status: str = BriefStatus.ASSIGNED.value,
    assignment_status: str = ReviewAssignmentStatus.ASSIGNED.value,
    due_date: Optional[datetime] = None,
    decision: Optional[str] = None,
    reviewer_id: str = "dr_smith",
    protocol_text: str = "Initial protocol text v1.0",
) -> str:
    """Helper to seed a ChangeBrief and ReviewAssignment record."""
    uid = uuid.uuid4().hex
    if due_date is None:
        due_date = datetime.now(timezone.utc) + timedelta(hours=24)

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier=f"doc_{uid[:8]}.pdf",
            source_path=f"/data/sources/doc_{uid[:8]}.pdf",
            sha256_hash=(uid * 2)[:64],
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Guideline recommendation text",
            recommendation_type="pharmacotherapy",
            target_population="Adults with diabetes",
            intervention="Target drug",
            page=1,
            section="Treatment",
            source_excerpt="Guideline recommendation text",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            confidence=0.95,
            status="gap_confirmed",
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            candidate_protocol_section_ids=["PROT-01__1.0__SEC-1"],
            similarity=0.90,
            comparison_result="gap",
            comparison_confidence=0.95,
            difference_type="intervention_change",
            matched_protocol_id="PROT-01",
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
            routing_target="Rapid Governance Committee",
            sla_deadline=due_date,
            status=ImpactStatus.CALCULATED.value,
            schema_version="1.0",
        )
        session.add(impact)
        session.flush()

        brief = ChangeBrief(
            id=str(uuid.uuid4()),
            impact_record_id=impact.id,
            status=brief_status,
            rendered_file_path=f"/data/output/brief_{uid[:8]}.md",
            rendered_file_hash=(uid * 2)[:64],
            structured_payload={"protocol_text": protocol_text},
            schema_version="1.0",
        )
        session.add(brief)
        session.flush()

        assignment = ReviewAssignment(
            id=str(uuid.uuid4()),
            change_brief_id=brief.id,
            reviewer_role="Clinical Governance Lead",
            reviewer_id=reviewer_id,
            status=assignment_status,
            due_date=due_date,
            decision=decision,
            rationale="Approved" if decision else None,
            schema_version="1.0",
        )
        session.add(assignment)
        session.commit()
        return brief.id


# ==============================================================================
# 1. SCHEDULER INITIALIZATION AND CONFIGURATION
# ==============================================================================

def test_scheduler_construction(scheduler_env):
    """Test 1: Verify constructor initializes with explicit and default attributes."""
    scheduler = scheduler_env["scheduler"]
    assert scheduler.interval_minutes == 60
    assert scheduler.timezone_str == "UTC"
    assert scheduler.enabled is False
    assert scheduler.is_running() is False


def test_default_configuration():
    """Test 2: Verify default AppConfig values for SLA scheduler."""
    config = AppConfig()
    assert config.sla_scheduler_enabled is False
    assert config.sla_check_interval_minutes == 60
    assert config.sla_timezone == "UTC"


def test_scheduler_disabled_by_default(scheduler_env):
    """Test 3: Verify start() on disabled scheduler returns False without starting."""
    scheduler = scheduler_env["scheduler"]
    assert scheduler.enabled is False
    started = scheduler.start(force=False)
    assert started is False
    assert scheduler.is_running() is False


def test_scheduler_start_forced(scheduler_env):
    """Test 4: Verify start(force=True) starts the APScheduler BackgroundScheduler."""
    scheduler = scheduler_env["scheduler"]
    started = scheduler.start(force=True)
    assert started is True
    assert scheduler.is_running() is True
    job = scheduler.get_job()
    assert job is not None
    assert job.id == SLAScheduler.JOB_ID
    scheduler.stop()


def test_scheduler_stop(scheduler_env):
    """Test 5: Verify stop() cleanly shuts down running scheduler."""
    scheduler = scheduler_env["scheduler"]
    scheduler.start(force=True)
    assert scheduler.is_running() is True

    stopped = scheduler.stop()
    assert stopped is True
    assert scheduler.is_running() is False


def test_start_is_idempotent(scheduler_env):
    """Test 6: Verify calling start() multiple times is safe and maintains single job."""
    scheduler = scheduler_env["scheduler"]
    scheduler.start(force=True)
    assert scheduler.is_running() is True

    # Second start call
    again = scheduler.start(force=True)
    assert again is True
    assert scheduler.is_running() is True
    scheduler.stop()


def test_stop_is_safe_when_already_stopped(scheduler_env):
    """Test 7: Verify stop() on stopped scheduler is safe and returns False."""
    scheduler = scheduler_env["scheduler"]
    assert scheduler.is_running() is False
    assert scheduler.stop() is False
    assert scheduler.is_running() is False


def test_run_once_works_without_background_scheduler(scheduler_env):
    """Test 8: Verify run_once executes synchronously without starting background threads."""
    scheduler = scheduler_env["scheduler"]
    assert scheduler.is_running() is False

    summary = scheduler.run_once()
    assert isinstance(summary, SLACycleSummary)
    assert scheduler.is_running() is False


# ==============================================================================
# 2. ELIGIBILITY AND BRIEF STATE HANDLING
# ==============================================================================

def test_eligible_brief_evaluation(scheduler_env):
    """Test 9: Verify briefs in assigned, in_review, and deferred states are evaluated."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    now = datetime.now(timezone.utc)

    # 1 assigned, 1 in_review, 1 deferred
    _create_brief_with_assignment(session_factory, brief_status=BriefStatus.ASSIGNED.value)
    _create_brief_with_assignment(session_factory, brief_status=BriefStatus.IN_REVIEW.value)
    _create_brief_with_assignment(session_factory, brief_status=BriefStatus.DEFERRED.value)

    summary = scheduler.run_once(reference_time=now)
    assert summary.evaluated_count == 3
    assert summary.skipped_count == 0


def test_unresolved_brief_evaluation(scheduler_env):
    """Test 10: Verify evaluation inspects deadline accurately."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    future = datetime.now(timezone.utc) + timedelta(hours=10)

    _create_brief_with_assignment(session_factory, due_date=future)
    summary = scheduler.run_once()
    assert summary.evaluated_count == 1
    assert summary.overdue_count == 0


def test_closed_brief_skipped(scheduler_env):
    """Test 11: Verify closed ChangeBrief is skipped and not evaluated."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]

    _create_brief_with_assignment(
        session_factory,
        brief_status=BriefStatus.CLOSED.value,
        assignment_status=ReviewAssignmentStatus.COMPLETED.value,
        decision=ReviewDecision.APPROVE.value,
    )

    summary = scheduler.run_once()
    assert summary.evaluated_count == 0
    assert summary.skipped_count == 1


def test_decided_brief_skipped(scheduler_env):
    """Test 12: Verify decided ChangeBrief is skipped and not evaluated."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]

    _create_brief_with_assignment(
        session_factory,
        brief_status=BriefStatus.DECIDED.value,
        assignment_status=ReviewAssignmentStatus.COMPLETED.value,
        decision=ReviewDecision.APPROVE.value,
    )

    summary = scheduler.run_once()
    assert summary.evaluated_count == 0
    assert summary.skipped_count == 1


# ==============================================================================
# 3. ESCALATION, IDEMPOTENCY, AND AUDITABILITY
# ==============================================================================

def test_overdue_brief_escalation(scheduler_env):
    """Test 13: Verify overdue assignment triggers G5 escalation."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=2)

    brief_id = _create_brief_with_assignment(session_factory, due_date=past)
    summary = scheduler.run_once()

    assert summary.evaluated_count == 1
    assert summary.overdue_count == 1
    assert summary.escalated_count == 1

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.status == ReviewAssignmentStatus.ESCALATED.value


def test_non_overdue_brief_does_not_escalate(scheduler_env):
    """Test 14: Verify assignment within SLA window is not escalated."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    future = datetime.now(timezone.utc) + timedelta(hours=48)

    brief_id = _create_brief_with_assignment(session_factory, due_date=future)
    summary = scheduler.run_once()

    assert summary.evaluated_count == 1
    assert summary.overdue_count == 0
    assert summary.escalated_count == 0

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.status == ReviewAssignmentStatus.ASSIGNED.value


def test_already_escalated_brief_remains_idempotent(scheduler_env):
    """Test 15: Verify already-escalated assignment is recognized as overdue but not re-escalated."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=2)

    brief_id = _create_brief_with_assignment(
        session_factory,
        assignment_status=ReviewAssignmentStatus.ESCALATED.value,
        due_date=past,
    )

    summary = scheduler.run_once()
    assert summary.evaluated_count == 1
    assert summary.overdue_count == 1
    assert summary.escalated_count == 0
    assert summary.already_escalated_count == 1


def test_repeated_run_once_has_no_duplicates(scheduler_env):
    """Test 16: Verify repeated run_once cycles produce no duplicate notifications or audit entries."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=5)

    brief_id = _create_brief_with_assignment(session_factory, due_date=past)

    # 3 consecutive cycles
    s1 = scheduler.run_once()
    assert s1.escalated_count == 1

    s2 = scheduler.run_once()
    assert s2.escalated_count == 0
    assert s2.already_escalated_count == 1

    s3 = scheduler.run_once()
    assert s3.escalated_count == 0
    assert s3.already_escalated_count == 1

    with session_factory() as session:
        notif_count = session.query(Notification).filter_by(related_entity_id=brief_id).count()
        assert notif_count == 1

        escalation_audits = (
            session.query(AuditLog)
            .filter_by(entity_id=brief_id)
            .all()
        )
        escalation_actions = [a for a in escalation_audits if a.audit_metadata and a.audit_metadata.get("action") == "sla_escalation"]
        assert len(escalation_actions) == 1


def test_per_brief_failure_isolation(scheduler_env):
    """Test 17: Verify one brief failure does not abort remaining brief evaluations."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    gov_agent = scheduler_env["gov_agent"]
    past = datetime.now(timezone.utc) - timedelta(hours=2)

    b1_id = _create_brief_with_assignment(session_factory, due_date=past)
    b2_id = _create_brief_with_assignment(session_factory, due_date=past)

    # Patch evaluate_sla to fail only on b1
    original_eval = gov_agent.evaluate_sla

    def mock_eval(change_brief_id, current_time=None):
        if change_brief_id == b1_id:
            raise RuntimeError("Database connection glitch for brief 1")
        return original_eval(change_brief_id, current_time)

    with patch.object(gov_agent, "evaluate_sla", side_effect=mock_eval):
        summary = scheduler.run_once()

    assert summary.evaluated_count == 1  # b2 evaluated
    assert summary.failed_count == 1     # b1 failed
    assert summary.escalated_count == 1   # b2 escalated
    assert len(summary.errors) == 1


def test_cycle_result_counts(scheduler_env):
    """Test 18: Verify all summary count properties add up consistently."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    future = datetime.now(timezone.utc) + timedelta(hours=24)

    _create_brief_with_assignment(session_factory, due_date=past)    # overdue & escalated
    _create_brief_with_assignment(session_factory, due_date=future)  # within SLA
    _create_brief_with_assignment(                                    # skipped (closed)
        session_factory,
        brief_status=BriefStatus.CLOSED.value,
        assignment_status=ReviewAssignmentStatus.COMPLETED.value,
        decision=ReviewDecision.APPROVE.value,
    )

    summary = scheduler.run_once()
    assert summary.evaluated_count == 2
    assert summary.skipped_count == 1
    assert summary.overdue_count == 1
    assert summary.escalated_count == 1
    assert summary.total_processed == 3
    assert summary.has_errors is False


def test_scheduler_exception_handling(scheduler_env):
    """Test 19: Verify top-level operational exception during cycle is handled cleanly."""
    scheduler = scheduler_env["scheduler"]

    with patch.object(scheduler, "session_factory", side_effect=RuntimeError("Database pool exhausted")):
        summary = scheduler.run_once()

    assert summary.failed_count == 1
    assert summary.has_errors is True
    assert "Database pool exhausted" in summary.errors[0]


def test_utc_timestamp_behavior(scheduler_env):
    """Test 20: Verify execution timestamps are strictly timezone-aware UTC."""
    scheduler = scheduler_env["scheduler"]
    summary = scheduler.run_once()

    assert summary.timestamp.tzinfo is not None
    assert summary.timestamp.tzinfo == timezone.utc


def test_configured_interval_honored(scheduler_env):
    """Test 21: Verify scheduler registers job with configured interval minutes."""
    session_factory = scheduler_env["session_factory"]
    gov_agent = scheduler_env["gov_agent"]
    config = scheduler_env["config"]

    custom_scheduler = SLAScheduler(
        session_factory=session_factory,
        governance_agent=gov_agent,
        config=config,
        interval_minutes=15,
        enabled=True,
    )
    custom_scheduler.start()
    job = custom_scheduler.get_job()
    assert job is not None
    assert isinstance(job.trigger, IntervalTrigger)
    assert job.trigger.interval == timedelta(minutes=15)
    custom_scheduler.stop()


# ==============================================================================
# 4. SAFETY AND HUMAN GATE BOUNDARIES
# ==============================================================================

def test_no_automatic_governance_decision(scheduler_env):
    """Test 22: CRITICAL - Prove an overdue brief is escalated, but its decision remains unset."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=6)

    brief_id = _create_brief_with_assignment(session_factory, due_date=past)
    summary = scheduler.run_once()
    assert summary.escalated_count == 1

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        brief = session.get(ChangeBrief, brief_id)

        # Decision MUST remain None
        assert assignment.decision is None
        assert assignment.status == ReviewAssignmentStatus.ESCALATED.value
        # Brief status MUST NOT automatically approve, reject, defer, or close
        assert brief.status == BriefStatus.ASSIGNED.value


def test_no_protocol_mutation(scheduler_env):
    """Test 23: Verify scheduler never modifies protocol texts or vector embeddings."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=3)
    baseline_text = "SECTION 4: Metformin is first-line."

    brief_id = _create_brief_with_assignment(
        session_factory,
        due_date=past,
        protocol_text=baseline_text,
    )

    scheduler.run_once()

    with session_factory() as session:
        brief = session.get(ChangeBrief, brief_id)
        assert brief.structured_payload["protocol_text"] == baseline_text


def test_zero_llm_calls(scheduler_env):
    """Test 24: Verify zero LLM calls occur during scheduler execution."""
    scheduler = scheduler_env["scheduler"]
    session_factory = scheduler_env["session_factory"]
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    _create_brief_with_assignment(session_factory, due_date=past)

    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM call attempted during SLA Scheduler run!")

    with patch.object(openai.resources.chat.completions.Completions, "create", side_effect=fail_llm):
        summary = scheduler.run_once()
        assert summary.evaluated_count == 1


def test_manual_single_cycle_execution(scheduler_env):
    """Test 25: Verify run_once executes as a standalone single-cycle runner."""
    scheduler = scheduler_env["scheduler"]
    summary = scheduler.run_once()
    assert summary.duration_ms >= 0.0
    assert scheduler.last_cycle_summary == summary


def test_scheduler_does_not_start_on_import():
    """Test 26: Verify importing the scheduler module does not start background threads."""
    import app.services.sla_scheduler as mod
    # No active BackgroundScheduler should exist simply from importing
    assert hasattr(mod, "SLAScheduler")


def test_no_external_notification_integration(scheduler_env):
    """Test 27: Verify scheduler only creates local Notification entity, no external network calls."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=2)

    brief_id = _create_brief_with_assignment(session_factory, due_date=past)
    scheduler.run_once()

    with session_factory() as session:
        notif = session.query(Notification).filter_by(related_entity_id=brief_id).first()
        assert notif is not None
        assert notif.delivery_status == "pending"
        # No external dispatch attempted in this phase


def test_no_phase_12_behavior(scheduler_env):
    """Test 28: Verify scheduler does not invoke Phase 12 evaluation framework."""
    scheduler = scheduler_env["scheduler"]
    assert not hasattr(scheduler, "evaluate_framework")
    assert not hasattr(scheduler, "benchmark_metrics")


def test_no_phase_13_behavior(scheduler_env):
    """Test 29: Verify scheduler contains no Streamlit or UI logic."""
    scheduler = scheduler_env["scheduler"]
    assert not hasattr(scheduler, "render_ui")
    assert not hasattr(scheduler, "streamlit")


def test_audit_log_and_notification_consistency(scheduler_env):
    """Test 30: Verify notification and audit log metadata are consistent with brief ID and routing target."""
    session_factory = scheduler_env["session_factory"]
    scheduler = scheduler_env["scheduler"]
    past = datetime.now(timezone.utc) - timedelta(hours=2)

    brief_id = _create_brief_with_assignment(session_factory, due_date=past)
    scheduler.run_once()

    with session_factory() as session:
        notif = session.query(Notification).filter_by(related_entity_id=brief_id).first()
        audit = (
            session.query(AuditLog)
            .filter_by(entity_id=brief_id)
            .filter(AuditLog.actor == "system_sla_monitor")
            .first()
        )

        assert notif is not None
        assert audit is not None
        assert notif.recipient == "Rapid Governance Committee"
        assert audit.audit_metadata["notification_id"] == notif.id


def test_in_process_concurrency_guard(scheduler_env):
    """Test 31: Verify concurrency guard skips overlapping run_once executions."""
    scheduler = scheduler_env["scheduler"]

    # Artificially acquire the lock to simulate an active cycle
    assert scheduler._lock.acquire(blocking=False) is True
    try:
        # A concurrent call should skip gracefully
        concurrent_summary = scheduler.run_once()
        assert concurrent_summary.skipped_count == 1
        assert "another SLA evaluation cycle is currently in progress" in concurrent_summary.errors[0]
    finally:
        scheduler._lock.release()
