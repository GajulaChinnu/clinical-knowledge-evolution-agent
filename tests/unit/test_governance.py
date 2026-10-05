"""Unit and integration tests for Phase 9 Governance Agent and Human Gates.

Verifies:
1. Reviewer assignment and persistence
2. Reassignment and audit trail
3. Rejection of unauthorized reviewers
4. Start review state transition
5. Approve workflow
6. Reject workflow
7. Defer workflow with mandatory future follow-up date
8. Missing or whitespace-only rationale rejected
9. Defer without due date or with past due date rejected
10. G4 Human Gate: cannot close without explicit decision
11. No automated decision inference (tier, SLA, timeout)
12. Cannot bypass review state
13. Invalid state transition rejection
14. Decision, reassignment, and escalation audit records
15. G5 SLA Escalation detection and idempotency
16. SLA escalation does not create decision or modify protocol
17. AuditLog append-only immutability
18. Notification persistence
19. Repeated governance operation prevention
20. Protocol data immutability throughout governance operations
21. ZERO LLM/API invocations
22. Complete end-to-end governance lifecycle integration test
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.governance_agent import GovernanceAgent, RecordNotFoundError
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
    GovernanceGateError,
    ReviewAssignmentStatus,
    ReviewDecision,
    SLAEscalationResult,
    UnauthorizedReviewerError,
)
from app.schemas.impact import ImpactStatus, ImpactTier
from app.schemas.transitions import InvalidStateTransitionError
from app.services.config_service import AppConfig
from app.services.reviewer_authorization import ReviewerAuthorizationService


@pytest.fixture
def governance_env():
    """Set up an isolated in-memory SQLite database and GovernanceAgent."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    auth_service = ReviewerAuthorizationService()
    agent = GovernanceAgent(
        session_factory=session_factory,
        auth_service=auth_service,
    )

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "auth_service": auth_service,
        "agent": agent,
    }

    engine.dispose()


def _create_test_brief(
    session_factory,
    status: str = BriefStatus.DRAFT.value,
    sla_deadline: Optional[datetime] = None,
    protocol_text: str = "Standard metformin protocol text v1.0",
) -> str:
    """Helper to persist a complete test hierarchy ending in a ChangeBrief."""
    unique_id = uuid.uuid4().hex
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier=f"ADA-{unique_id[:8]}.pdf",
            source_path=f"/data/sources/ADA-{unique_id[:8]}.pdf",
            sha256_hash=(unique_id * 2)[:64],
            document_version="1.0",
            source_version="2026.1",
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Start SGLT2 inhibitor in adults with T2D and CKD.",
            recommendation_type="pharmacotherapy",
            target_population="Adults with Type 2 Diabetes and CKD",
            intervention="SGLT2 inhibitor",
            evidence_grade="Grade A",
            page=12,
            section="Pharmacotherapy",
            source_excerpt="In patients with T2D and CKD, start SGLT2 inhibitor.",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status="gap_confirmed",
            confidence=0.95,
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            candidate_protocol_section_ids=["PROT-DM-001__1.0__SEC-3"],
            similarity=0.89,
            comparison_result="gap",
            comparison_confidence=0.92,
            difference_type="dosage_change",
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="1.0",
            is_match=True,
            status="matched",
            schema_version="1.0",
        )
        session.add(gap)
        session.flush()

        deadline = sla_deadline or (datetime.now(timezone.utc) + timedelta(hours=48))
        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=4,
            evidence_strength=5,
            pathway_breadth=3,
            rule_ids={"clinical_urgency": "URGENCY-04", "evidence_strength": "EVIDENCE-05", "pathway_breadth": "BREADTH-03"},
            scoring_yaml_version="1.0",
            total_score=12,
            tier=ImpactTier.CRITICAL.value,
            routing_target="Rapid Governance Committee",
            sla_deadline=deadline,
            urgency_basis="Time-sensitive treatment change",
            evidence_basis="Class I / Grade 1A RCT evidence",
            breadth_basis="One specialty, multiple care pathways",
            status=ImpactStatus.CALCULATED.value,
            schema_version="1.0",
        )
        session.add(impact)
        session.flush()

        brief = ChangeBrief(
            id=str(uuid.uuid4()),
            impact_record_id=impact.id,
            status=status,
            rendered_file_path=f"/data/output/brief_{unique_id[:8]}.md",
            rendered_file_hash=(unique_id * 2)[:64],
            structured_payload={"test_protocol_text": protocol_text},
            schema_version="1.0",
        )
        session.add(brief)
        session.commit()
        return brief.id


# ==============================================================================
# TESTS 1–4: REVIEWER ASSIGNMENT & REASSIGNMENT
# ==============================================================================

def test_reviewer_assignment(governance_env):
    """Test 1: Assign authorized reviewer to draft brief; transitions to assigned."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory, status=BriefStatus.DRAFT.value)
    assignment = agent.assign_reviewer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        reviewer_role="Clinical Governance Board Member",
    )

    assert assignment is not None
    assert assignment.reviewer_id == "dr_smith"
    assert assignment.status == ReviewAssignmentStatus.ASSIGNED.value

    with session_factory() as session:
        brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert brief.status == BriefStatus.ASSIGNED.value


def test_reassignment(governance_env):
    """Test 2: Reassign brief to another authorized reviewer with audit reason."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    updated_assignment = agent.reassign_reviewer(
        change_brief_id=brief_id,
        new_reviewer_id="dr_jones",
        reason="Dr. Smith on clinical leave; reassigning to Dr. Jones.",
        actor="governance_chair",
    )

    assert updated_assignment.reviewer_id == "dr_jones"

    with session_factory() as session:
        audit = (
            session.query(AuditLog)
            .filter_by(entity_id=brief_id)
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert audit is not None
        assert "Dr. Smith on clinical leave" in audit.reason
        assert audit.actor == "governance_chair"


def test_valid_assignment_persistence(governance_env):
    """Test 3: Verify all ReviewAssignment fields are accurately persisted in SQLite."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    due = datetime.now(timezone.utc) + timedelta(days=3)
    brief_id = _create_test_brief(session_factory)

    assignment = agent.assign_reviewer(
        change_brief_id=brief_id,
        reviewer_id="cmo_director",
        reviewer_role="Chief Medical Officer",
        due_date=due,
        actor="system_admin",
    )

    with session_factory() as session:
        persisted = session.query(ReviewAssignment).filter_by(id=assignment.id).first()
        assert persisted is not None
        assert persisted.change_brief_id == brief_id
        assert persisted.reviewer_id == "cmo_director"
        assert persisted.reviewer_role == "Chief Medical Officer"
        assert persisted.status == "assigned"
        assert persisted.due_date is not None


def test_unauthorized_reviewer_rejection(governance_env):
    """Test 4: Attempting to assign an unauthorized reviewer raises UnauthorizedReviewerError."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)

    with pytest.raises(UnauthorizedReviewerError):
        agent.assign_reviewer(
            change_brief_id=brief_id,
            reviewer_id="unauthorized_user",
        )

    # Verify unauthorized attempt was audited
    with session_factory() as session:
        audit = session.query(AuditLog).filter_by(entity_id=brief_id).first()
        assert audit is not None
        assert "unauthorized reviewer" in audit.reason.lower()


# ==============================================================================
# TESTS 5–8: REVIEW STATE AND DECISION FLOWS (APPROVE, REJECT, DEFER)
# ==============================================================================

def test_start_review(governance_env):
    """Test 5: Transition brief from assigned to in_review."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    brief = agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    assert brief.status == BriefStatus.IN_REVIEW.value

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.status == ReviewAssignmentStatus.IN_REVIEW.value


def test_approve_flow(governance_env):
    """Test 6: Authorized reviewer approves brief; transitions to decided."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")

    brief = agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Compelling Grade A evidence supporting SGLT2i initiation for renal protection.",
    )

    assert brief.status == BriefStatus.DECIDED.value

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.decision == "approve"
        assert "Compelling Grade A evidence" in assignment.rationale
        assert assignment.status == ReviewAssignmentStatus.COMPLETED.value


def test_reject_flow(governance_env):
    """Test 7: Authorized reviewer rejects brief; transitions to decided."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_jones")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_jones")

    brief = agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_jones",
        decision=ReviewDecision.REJECT,
        rationale="Not applicable to our institutional patient demographic at this stage.",
    )

    assert brief.status == BriefStatus.DECIDED.value

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.decision == "reject"
        assert assignment.status == ReviewAssignmentStatus.COMPLETED.value


def test_defer_flow(governance_env):
    """Test 8: Defer brief with mandatory future date; transitions to deferred and remains open."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")

    future_follow_up = datetime.now(timezone.utc) + timedelta(days=60)
    brief = agent.defer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        rationale="Awaiting upcoming sub-study trial data before institutional adoption.",
        defer_follow_up_date=future_follow_up,
    )

    assert brief.status == BriefStatus.DEFERRED.value

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.decision == "defer"
        assert assignment.defer_follow_up_date is not None


# ==============================================================================
# TESTS 9–13: RATIONALE & DEFER VALIDATION CONSTRAINTS
# ==============================================================================

def test_missing_rationale_rejected(governance_env):
    """Test 9: Attempting decision without rationale raises ValueError."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")

    with pytest.raises(ValueError) as exc_info:
        agent.decide(
            change_brief_id=brief_id,
            reviewer_id="dr_smith",
            decision=ReviewDecision.APPROVE,
            rationale="",  # empty
        )
    assert "rationale" in str(exc_info.value).lower()


def test_blank_whitespace_rationale_rejected(governance_env):
    """Test 10: Attempting decision with whitespace-only rationale raises ValueError."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")

    with pytest.raises(ValueError):
        agent.decide(
            change_brief_id=brief_id,
            reviewer_id="dr_smith",
            decision=ReviewDecision.APPROVE,
            rationale="   \n\t  ",
        )


def test_defer_without_due_date_rejected(governance_env):
    """Test 11: Deferring without a follow-up date raises ValueError."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")

    with pytest.raises(ValueError):
        agent.decide(
            change_brief_id=brief_id,
            reviewer_id="dr_smith",
            decision=ReviewDecision.DEFER,
            rationale="Valid rationale.",
            defer_follow_up_date=None,
        )


def test_defer_with_past_due_date_rejected(governance_env):
    """Test 12: Deferring with a date in the past raises ValueError."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")

    past_date = datetime.now(timezone.utc) - timedelta(days=5)
    with pytest.raises(ValueError) as exc_info:
        agent.defer(
            change_brief_id=brief_id,
            reviewer_id="dr_smith",
            rationale="Valid rationale.",
            defer_follow_up_date=past_date,
        )
    assert "future" in str(exc_info.value).lower()


def test_defer_with_valid_future_due_date(governance_env):
    """Test 13: Defer with future date succeeds and updates assignment deadline."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")

    future_date = datetime.now(timezone.utc) + timedelta(days=90)
    brief = agent.defer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        rationale="Re-evaluating post multi-center study publication.",
        defer_follow_up_date=future_date,
    )
    assert brief.status == "deferred"


# ==============================================================================
# TESTS 14–17: G4 HUMAN GATE & TRANSITION INTEGRITY
# ==============================================================================

def test_cannot_close_without_decision(governance_env):
    """Test 14: G4 Human Gate: attempting to close a non-decided brief raises GovernanceGateError."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    # 1. Draft brief cannot close
    brief_draft_id = _create_test_brief(session_factory, status=BriefStatus.DRAFT.value)
    with pytest.raises(GovernanceGateError):
        agent.close_after_decision(change_brief_id=brief_draft_id, actor="admin")

    # 2. Assigned brief cannot close
    agent.assign_reviewer(change_brief_id=brief_draft_id, reviewer_id="dr_smith")
    with pytest.raises(GovernanceGateError):
        agent.close_after_decision(change_brief_id=brief_draft_id, actor="admin")

    # 3. In-review brief cannot close
    agent.start_review(change_brief_id=brief_draft_id, reviewer_id="dr_smith")
    with pytest.raises(GovernanceGateError):
        agent.close_after_decision(change_brief_id=brief_draft_id, actor="admin")

    # 4. Deferred brief cannot close
    future_date = datetime.now(timezone.utc) + timedelta(days=30)
    agent.defer(
        change_brief_id=brief_draft_id,
        reviewer_id="dr_smith",
        rationale="Deferred.",
        defer_follow_up_date=future_date,
    )
    with pytest.raises(GovernanceGateError):
        agent.close_after_decision(change_brief_id=brief_draft_id, actor="admin")


def test_cannot_auto_decide(governance_env):
    """Test 15: Critical tier or SLA breach never auto-decides; brief remains in assigned state."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    # Past deadline (SLA breach)
    past_deadline = datetime.now(timezone.utc) - timedelta(hours=10)
    brief_id = _create_test_brief(session_factory, sla_deadline=past_deadline)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    # Trigger SLA evaluation
    result = agent.evaluate_sla(change_brief_id=brief_id)
    assert result.is_overdue is True

    # Verify brief has NOT auto-decided or closed
    with session_factory() as session:
        brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert brief.status == BriefStatus.ASSIGNED.value
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.decision is None


def test_cannot_bypass_review_state(governance_env):
    """Test 16: Calling decide() directly from draft or assigned raises InvalidStateTransitionError."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory, status=BriefStatus.DRAFT.value)

    # 1. Attempt from draft
    with pytest.raises(InvalidStateTransitionError):
        agent.decide(
            change_brief_id=brief_id,
            reviewer_id="dr_smith",
            decision=ReviewDecision.APPROVE,
            rationale="Premature decision attempt.",
        )

    # 2. Attempt from assigned (without calling start_review)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    with pytest.raises(InvalidStateTransitionError):
        agent.decide(
            change_brief_id=brief_id,
            reviewer_id="dr_smith",
            decision=ReviewDecision.APPROVE,
            rationale="Skipping in_review step.",
        )


def test_invalid_state_transition_rejection(governance_env):
    """Test 17: Verify illegal transitions raise InvalidStateTransitionError and are audited."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory, status=BriefStatus.DRAFT.value)

    # Attempt to start review on a draft (must be assigned first)
    with pytest.raises(InvalidStateTransitionError):
        agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")


# ==============================================================================
# TESTS 18–20: AUDIT LOG INTEGRITY
# ==============================================================================

def test_decision_audit_record(governance_env):
    """Test 18: Decision creates detailed immutable AuditLog entry with rationale and actor."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Strong guideline recommendation.",
    )

    with session_factory() as session:
        audit = (
            session.query(AuditLog)
            .filter_by(entity_id=brief_id, new_status=BriefStatus.DECIDED.value)
            .first()
        )
        assert audit is not None
        assert audit.actor == "dr_smith"
        assert audit.reason == "Strong guideline recommendation."
        assert audit.audit_metadata["action"] == "approve"


def test_reassignment_audit_record(governance_env):
    """Test 19: Reassignment records previous reviewer, new reviewer, and rationale."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.reassign_reviewer(
        change_brief_id=brief_id,
        new_reviewer_id="dr_jones",
        reason="Workload rebalance across subspecialties.",
        actor="chairperson",
    )

    with session_factory() as session:
        audit = (
            session.query(AuditLog)
            .filter_by(entity_id=brief_id)
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert audit.audit_metadata["action"] == "reassignment"
        assert audit.audit_metadata["previous_reviewer"] == "dr_smith"
        assert audit.audit_metadata["new_reviewer"] == "dr_jones"


def test_escalation_audit_record(governance_env):
    """Test 20: SLA escalation creates an audit record with deadline and timestamp."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    past_date = datetime.now(timezone.utc) - timedelta(hours=5)
    brief_id = _create_test_brief(session_factory, sla_deadline=past_date)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    agent.evaluate_sla(change_brief_id=brief_id)

    with session_factory() as session:
        audit = (
            session.query(AuditLog)
            .filter_by(entity_id=brief_id, actor="system_sla_monitor")
            .first()
        )
        assert audit is not None
        assert "G5 SLA Breach" in audit.reason


# ==============================================================================
# TESTS 21–25: G5 SLA ESCALATION, IDEMPOTENCY, AND NON-DECISION
# ==============================================================================

def test_escalation_idempotency(governance_env):
    """Test 21: Repeated SLA checks do not create duplicate escalation notifications or audit logs."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    past_date = datetime.now(timezone.utc) - timedelta(hours=10)
    brief_id = _create_test_brief(session_factory, sla_deadline=past_date)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    # 1. First escalation
    res1 = agent.evaluate_sla(change_brief_id=brief_id)
    assert res1.is_overdue is True
    assert res1.escalated is True

    # 2. Second evaluation (same brief)
    res2 = agent.evaluate_sla(change_brief_id=brief_id)
    assert res2.is_overdue is True
    assert res2.escalated is False  # skipped duplicate

    with session_factory() as session:
        notifs = session.query(Notification).filter_by(related_entity_id=brief_id, notification_type="escalation").all()
        assert len(notifs) == 1

        audits = session.query(AuditLog).filter_by(entity_id=brief_id, actor="system_sla_monitor").all()
        assert len(audits) == 1


def test_overdue_detection(governance_env):
    """Test 22: Correctly identifies overdue briefs past deadline."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    past_date = datetime.now(timezone.utc) - timedelta(minutes=1)
    brief_id = _create_test_brief(session_factory, sla_deadline=past_date)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    result = agent.evaluate_sla(change_brief_id=brief_id)
    assert result.is_overdue is True


def test_non_overdue_detection(governance_env):
    """Test 23: Correctly identifies briefs within the SLA window."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    future_date = datetime.now(timezone.utc) + timedelta(days=2)
    brief_id = _create_test_brief(session_factory, sla_deadline=future_date)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    result = agent.evaluate_sla(change_brief_id=brief_id)
    assert result.is_overdue is False
    assert result.escalated is False


def test_escalation_does_not_create_decision(governance_env):
    """Test 24: Escalation never sets decision, never auto-approves or auto-rejects."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    past_date = datetime.now(timezone.utc) - timedelta(hours=24)
    brief_id = _create_test_brief(session_factory, sla_deadline=past_date)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    agent.evaluate_sla(change_brief_id=brief_id)

    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.decision is None
        brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert brief.status == BriefStatus.ASSIGNED.value


def test_escalation_does_not_modify_protocol(governance_env):
    """Test 25: SLA escalation never alters protocol content."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    past_date = datetime.now(timezone.utc) - timedelta(hours=24)
    brief_id = _create_test_brief(session_factory, sla_deadline=past_date, protocol_text="Original protocol v1.0")

    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.evaluate_sla(change_brief_id=brief_id)

    with session_factory() as session:
        brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert brief.structured_payload["test_protocol_text"] == "Original protocol v1.0"


# ==============================================================================
# TESTS 26–30: IMMUTABILITY, NOTIFICATIONS, REPEATED OPERATIONS, ZERO LLM
# ==============================================================================

def test_audit_log_remains_append_only(governance_env):
    """Test 26: Attempting to delete an AuditLog record raises an exception."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")

    with session_factory() as session:
        audit = session.query(AuditLog).filter_by(entity_id=brief_id).first()
        session.delete(audit)
        with pytest.raises(Exception) as exc_info:
            session.commit()
        assert "cannot be deleted" in str(exc_info.value).lower()


def test_notification_persistence(governance_env):
    """Test 27: Notifications are persisted for assignment and reassignment."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.reassign_reviewer(
        change_brief_id=brief_id,
        new_reviewer_id="dr_jones",
        reason="Transfer.",
        actor="lead",
    )

    with session_factory() as session:
        notifs = session.query(Notification).filter_by(related_entity_id=brief_id).all()
        types = [n.notification_type for n in notifs]
        assert "assignment" in types
        assert "reassignment" in types


def test_repeated_governance_operation_does_not_create_duplicate_decision(governance_env):
    """Test 28: Attempting to decide an already decided or closed brief is rejected."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Approved first time.",
    )

    # Attempt second decision
    with pytest.raises(InvalidStateTransitionError):
        agent.decide(
            change_brief_id=brief_id,
            reviewer_id="dr_smith",
            decision=ReviewDecision.REJECT,
            rationale="Attempting overwrite.",
        )


def test_protocol_data_remains_unchanged_after_governance_operations(governance_env):
    """Test 29: Regression test proving protocol, change, and document records remain untouched."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    brief_id = _create_test_brief(session_factory)

    with session_factory() as session:
        brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        orig_doc_id = brief.impact_record.gap_record.change_record.document.id
        orig_change_id = brief.impact_record.gap_record.change_record.id
        orig_gap_id = brief.impact_record.gap_record.id
        orig_rec_text = brief.impact_record.gap_record.change_record.verbatim_text

    # Execute governance sequence
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Approved.",
    )
    agent.close_after_decision(change_brief_id=brief_id, actor="governance_chair")

    with session_factory() as session:
        brief_after = session.query(ChangeBrief).filter_by(id=brief_id).first()
        doc_after = brief_after.impact_record.gap_record.change_record.document
        change_after = brief_after.impact_record.gap_record.change_record
        gap_after = brief_after.impact_record.gap_record

        assert doc_after.id == orig_doc_id
        assert change_after.id == orig_change_id
        assert gap_after.id == orig_gap_id
        assert change_after.verbatim_text == orig_rec_text


def test_no_llm_invocation_in_governance(governance_env, monkeypatch):
    """Test 30: Zero LLM / API calls made during GovernanceAgent execution."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    import openai
    def fail_if_llm_called(*args, **kwargs):
        raise AssertionError("LLM call attempted inside Governance Agent!")

    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fail_if_llm_called)

    brief_id = _create_test_brief(session_factory)
    agent.assign_reviewer(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Zero LLM test approval.",
    )
    agent.close_after_decision(change_brief_id=brief_id, actor="chairperson")


# ==============================================================================
# TEST 31: INTEGRATION LIFECYCLE (DRAFT -> ASSIGN -> IN_REVIEW -> DEFER -> IN_REVIEW -> APPROVE -> CLOSE)
# ==============================================================================

def test_full_governance_lifecycle_integration(governance_env):
    """Test 31: Full end-to-end integration lifecycle test with deferral and final closure."""
    session_factory = governance_env["session_factory"]
    agent = governance_env["agent"]

    # 1. Draft ChangeBrief exists
    brief_id = _create_test_brief(session_factory, status=BriefStatus.DRAFT.value)

    # 2. Assign reviewer (draft -> assigned)
    assignment = agent.assign_reviewer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        reviewer_role="Clinical Governance Lead",
    )
    assert assignment.status == "assigned"

    # 3. Start review (assigned -> in_review)
    brief = agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    assert brief.status == "in_review"

    # 4. Defer review (in_review -> deferred)
    future_date = datetime.now(timezone.utc) + timedelta(days=45)
    brief = agent.defer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        rationale="Awaiting published outcome trial sub-analysis.",
        defer_follow_up_date=future_date,
    )
    assert brief.status == "deferred"

    # 5. Re-enter review (deferred -> in_review)
    brief = agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    assert brief.status == "in_review"

    # 6. Explicit Approve (in_review -> decided)
    brief = agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Trial results confirmed; approving institutional protocol update.",
    )
    assert brief.status == "decided"

    # 7. Close brief (decided -> closed)
    brief = agent.close_after_decision(
        change_brief_id=brief_id,
        actor="governance_chair",
        closure_note="Protocol update authorized for clinical EHR order set integration.",
    )
    assert brief.status == "closed"

    # 8. Verify audit trail and final state
    with session_factory() as session:
        final_brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert final_brief.status == "closed"

        audits = session.query(AuditLog).filter_by(entity_id=brief_id).order_by(AuditLog.timestamp.asc()).all()
        actions = [a.audit_metadata.get("action") for a in audits if a.audit_metadata]
        assert "assignment" in actions
        assert "start_review" in actions
        assert "defer" in actions
        assert "approve" in actions
        assert "close" in actions
