"""Deterministic Clinical Governance Agent implementing human gates G4 and G5.

Responsible for:
1. Reviewer assignment and auditable reassignment
2. Transitioning briefs into review
3. Recording explicit human decisions (approve, reject, defer) with mandatory rationale
4. Enforcing G4 Human Gate (no automated approval/rejection/closure)
5. Enforcing G5 SLA Escalation detection (without automated decisions)
6. Logging all actions and rejected attempts to append-only AuditLog
7. Emitting deterministic Notification records

IMPORTANT: Makes ZERO LLM or external API calls.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union
import uuid

from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeBrief,
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
from app.schemas.transitions import (
    BRIEF_TRANSITIONS,
    InvalidStateTransitionError,
    is_valid_transition,
    validate_transition,
)
from app.services.config_service import AppConfig, load_config
from app.services.reviewer_authorization import ReviewerAuthorizationService

logger = logging.getLogger("ckea.agents.governance_agent")


class RecordNotFoundError(GovernanceGateError):
    """Raised when an expected database record is not found."""
    pass


class GovernanceAgent:
    """Deterministic governance agent orchestrating G4 human decision gates and G5 SLA monitoring."""

    def __init__(
        self,
        session_factory: Optional[sessionmaker] = None,
        auth_service: Optional[ReviewerAuthorizationService] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_config()
        self.session_factory = session_factory or get_session_factory(self.config.database_url)
        self.auth_service = auth_service or ReviewerAuthorizationService(
            registry_path=self.config.reviewer_registry_path
        )

    # ==========================================================================
    # 1. REVIEWER ASSIGNMENT & REASSIGNMENT
    # ==========================================================================

    def assign_reviewer(
        self,
        change_brief_id: str,
        reviewer_id: str,
        reviewer_role: Optional[str] = None,
        due_date: Optional[datetime] = None,
        actor: Optional[str] = None,
    ) -> ReviewAssignment:
        """Assign an authorized clinical reviewer to a ChangeBrief.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            reviewer_id: Named identifier of the clinical reviewer.
            reviewer_role: Role or clinical specialty of the reviewer.
            due_date: Deadline for review completion.
            actor: Identity of actor performing the assignment.

        Returns:
            Persisted ReviewAssignment entity.

        Raises:
            RecordNotFoundError: If ChangeBrief is not found.
            UnauthorizedReviewerError: If reviewer_id is not authorized.
            InvalidStateTransitionError: If brief cannot transition to assigned.
        """
        actor_name = actor or "governance_coordinator"

        # Validate authorization
        if not self.auth_service.is_authorized(reviewer_id):
            self._record_audit(
                entity_id=change_brief_id,
                entity_type="ChangeBrief",
                previous_status=None,
                new_status="assignment_failed",
                actor=actor_name,
                reason=f"Attempted assignment to unauthorized reviewer '{reviewer_id}'.",
                audit_metadata={"action": "assignment_rejected", "reviewer_id": reviewer_id},
            )
            raise UnauthorizedReviewerError(f"Reviewer '{reviewer_id}' is not authorized for governance review.")

        resolved_role = reviewer_role or self.auth_service.get_reviewer_role(reviewer_id)

        with self.session_factory() as session:
            brief = session.query(ChangeBrief).filter_by(id=change_brief_id).first()
            if not brief:
                raise RecordNotFoundError(f"ChangeBrief not found with ID: {change_brief_id}")

            current_status = brief.status

            # Validate brief status transition
            if current_status not in (BriefStatus.DRAFT.value, BriefStatus.DEFERRED.value, BriefStatus.ASSIGNED.value):
                self._record_audit(
                    entity_id=change_brief_id,
                    entity_type="ChangeBrief",
                    previous_status=current_status,
                    new_status=current_status,
                    actor=actor_name,
                    reason=f"Illegal transition to 'assigned' from current status '{current_status}'.",
                )
                raise InvalidStateTransitionError(current_status, BriefStatus.ASSIGNED.value, "ChangeBrief")

            # Determine due date from ImpactRecord SLA if not provided
            target_due_date = due_date
            if not target_due_date:
                if brief.impact_record and brief.impact_record.sla_deadline:
                    target_due_date = brief.impact_record.sla_deadline
                else:
                    from datetime import timedelta
                    target_due_date = datetime.now(timezone.utc) + timedelta(days=7)

            if target_due_date.tzinfo is None:
                target_due_date = target_due_date.replace(tzinfo=timezone.utc)

            # Check for existing active assignment
            assignment = (
                session.query(ReviewAssignment)
                .filter_by(change_brief_id=brief.id)
                .order_by(ReviewAssignment.created_at.desc())
                .first()
            )

            if assignment and assignment.status in (ReviewAssignmentStatus.ASSIGNED.value, ReviewAssignmentStatus.IN_REVIEW.value):
                # Update existing assignment
                assignment.reviewer_id = reviewer_id
                assignment.reviewer_role = resolved_role
                assignment.due_date = target_due_date
                assignment.status = ReviewAssignmentStatus.ASSIGNED.value
            else:
                assignment = ReviewAssignment(
                    id=str(uuid.uuid4()),
                    change_brief_id=brief.id,
                    reviewer_role=resolved_role,
                    reviewer_id=reviewer_id,
                    status=ReviewAssignmentStatus.ASSIGNED.value,
                    due_date=target_due_date,
                    schema_version="1.0",
                )
                session.add(assignment)

            # Update brief status
            brief.status = BriefStatus.ASSIGNED.value
            session.commit()

            # Record AuditLog
            self._record_audit(
                entity_id=brief.id,
                entity_type="ChangeBrief",
                previous_status=current_status,
                new_status=BriefStatus.ASSIGNED.value,
                actor=actor_name,
                reason=f"Assigned to reviewer '{reviewer_id}' ({resolved_role}).",
                audit_metadata={
                    "action": "assignment",
                    "reviewer_id": reviewer_id,
                    "reviewer_role": resolved_role,
                    "due_date": target_due_date.isoformat(),
                    "assignment_id": assignment.id,
                },
            )

            # Record Notification
            self._record_notification(
                related_entity_id=brief.id,
                related_entity_type="ChangeBrief",
                notification_type="assignment",
                recipient=reviewer_id,
                payload={
                    "brief_id": brief.id,
                    "assignment_id": assignment.id,
                    "due_date": target_due_date.isoformat(),
                    "role": resolved_role,
                },
            )

            session.refresh(assignment)
            session.expunge(assignment)
            return assignment

    def reassign_reviewer(
        self,
        change_brief_id: str,
        new_reviewer_id: str,
        reason: str,
        actor: str,
        new_reviewer_role: Optional[str] = None,
        new_due_date: Optional[datetime] = None,
    ) -> ReviewAssignment:
        """Reassign an assigned ChangeBrief to a different authorized reviewer.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            new_reviewer_id: Named identifier of the new reviewer.
            reason: Non-empty rationale explaining the reassignment.
            actor: Identity of actor performing the reassignment.
            new_reviewer_role: Optional new role.
            new_due_date: Optional updated deadline.

        Returns:
            Updated ReviewAssignment entity.
        """
        if not reason or not reason.strip():
            raise ValueError("Reassignment reason cannot be empty or whitespace only.")
        if not actor or not actor.strip():
            raise ValueError("Actor identity cannot be empty.")

        if not self.auth_service.is_authorized(new_reviewer_id):
            self._record_audit(
                entity_id=change_brief_id,
                entity_type="ChangeBrief",
                previous_status=None,
                new_status="reassignment_failed",
                actor=actor,
                reason=f"Attempted reassignment to unauthorized reviewer '{new_reviewer_id}'.",
                audit_metadata={"action": "reassignment_rejected", "new_reviewer_id": new_reviewer_id},
            )
            raise UnauthorizedReviewerError(f"Reviewer '{new_reviewer_id}' is not authorized for governance review.")

        resolved_role = new_reviewer_role or self.auth_service.get_reviewer_role(new_reviewer_id)

        with self.session_factory() as session:
            brief = session.query(ChangeBrief).filter_by(id=change_brief_id).first()
            if not brief:
                raise RecordNotFoundError(f"ChangeBrief not found with ID: {change_brief_id}")

            if brief.status in (BriefStatus.DECIDED.value, BriefStatus.CLOSED.value):
                raise InvalidStateTransitionError(brief.status, BriefStatus.ASSIGNED.value, "ChangeBrief")

            assignment = (
                session.query(ReviewAssignment)
                .filter_by(change_brief_id=brief.id)
                .order_by(ReviewAssignment.created_at.desc())
                .first()
            )
            if not assignment:
                raise RecordNotFoundError(f"No existing ReviewAssignment found for brief {change_brief_id}.")

            old_reviewer = assignment.reviewer_id
            assignment.reviewer_id = new_reviewer_id
            assignment.reviewer_role = resolved_role
            if new_due_date:
                if new_due_date.tzinfo is None:
                    new_due_date = new_due_date.replace(tzinfo=timezone.utc)
                assignment.due_date = new_due_date
            assignment.status = ReviewAssignmentStatus.ASSIGNED.value

            # Update brief status to assigned
            brief.status = BriefStatus.ASSIGNED.value
            session.commit()

            # Record AuditLog
            self._record_audit(
                entity_id=brief.id,
                entity_type="ChangeBrief",
                previous_status=brief.status,
                new_status=BriefStatus.ASSIGNED.value,
                actor=actor,
                reason=reason.strip(),
                audit_metadata={
                    "action": "reassignment",
                    "previous_reviewer": old_reviewer,
                    "new_reviewer": new_reviewer_id,
                    "new_role": resolved_role,
                    "reassignment_reason": reason.strip(),
                },
            )

            # Record Notification
            self._record_notification(
                related_entity_id=brief.id,
                related_entity_type="ChangeBrief",
                notification_type="reassignment",
                recipient=new_reviewer_id,
                payload={
                    "brief_id": brief.id,
                    "previous_reviewer": old_reviewer,
                    "reason": reason.strip(),
                },
            )

            session.refresh(assignment)
            session.expunge(assignment)
            return assignment

    # ==========================================================================
    # 2. START REVIEW
    # ==========================================================================

    def start_review(self, change_brief_id: str, reviewer_id: str) -> ChangeBrief:
        """Transition an assigned ChangeBrief into the active 'in_review' state.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            reviewer_id: Named identifier of the reviewing clinician.

        Returns:
            Updated ChangeBrief entity in 'in_review' status.
        """
        if not self.auth_service.is_authorized(reviewer_id):
            raise UnauthorizedReviewerError(f"Reviewer '{reviewer_id}' is not authorized to start review.")

        with self.session_factory() as session:
            brief = session.query(ChangeBrief).filter_by(id=change_brief_id).first()
            if not brief:
                raise RecordNotFoundError(f"ChangeBrief not found with ID: {change_brief_id}")

            current_status = brief.status
            if current_status not in (BriefStatus.ASSIGNED.value, BriefStatus.DEFERRED.value):
                self._record_audit(
                    entity_id=change_brief_id,
                    entity_type="ChangeBrief",
                    previous_status=current_status,
                    new_status=current_status,
                    actor=reviewer_id,
                    reason=f"Cannot start review from status '{current_status}'.",
                )
                raise InvalidStateTransitionError(current_status, BriefStatus.IN_REVIEW.value, "ChangeBrief")

            assignment = (
                session.query(ReviewAssignment)
                .filter_by(change_brief_id=brief.id)
                .order_by(ReviewAssignment.created_at.desc())
                .first()
            )
            if not assignment:
                raise RecordNotFoundError(f"Cannot start review: no assignment exists for brief {change_brief_id}.")

            brief.status = BriefStatus.IN_REVIEW.value
            assignment.status = ReviewAssignmentStatus.IN_REVIEW.value
            session.commit()

            self._record_audit(
                entity_id=brief.id,
                entity_type="ChangeBrief",
                previous_status=current_status,
                new_status=BriefStatus.IN_REVIEW.value,
                actor=reviewer_id,
                reason="Review commenced by clinician.",
                audit_metadata={"action": "start_review", "reviewer_id": reviewer_id},
            )

            session.refresh(brief)
            session.expunge(brief)
            return brief

    # ==========================================================================
    # 3. G4 DECISION GATES (APPROVE, REJECT, DEFER)
    # ==========================================================================

    def decide(
        self,
        change_brief_id: str,
        reviewer_id: str,
        decision: Union[str, ReviewDecision],
        rationale: str,
        defer_follow_up_date: Optional[datetime] = None,
    ) -> ChangeBrief:
        """Record an explicit human governance decision on a ChangeBrief.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            reviewer_id: Named authorized reviewer making the decision.
            decision: 'approve', 'reject', or 'defer'.
            rationale: Mandatory non-empty clinical reasoning.
            defer_follow_up_date: Required if decision is 'defer'; must be None otherwise.

        Returns:
            Updated ChangeBrief entity.

        Raises:
            GovernanceGateError: If G4 gate requirements are breached.
            UnauthorizedReviewerError: If reviewer is not authorized.
            InvalidStateTransitionError: If brief is not in 'in_review' state.
        """
        # Validate reviewer authorization
        if not self.auth_service.is_authorized(reviewer_id):
            self._record_audit(
                entity_id=change_brief_id,
                entity_type="ChangeBrief",
                previous_status=None,
                new_status="decision_rejected",
                actor=reviewer_id,
                reason=f"Attempted decision '{decision}' by unauthorized reviewer '{reviewer_id}'.",
                audit_metadata={"action": "unauthorized_decision_attempt", "reviewer_id": reviewer_id},
            )
            raise UnauthorizedReviewerError(f"Reviewer '{reviewer_id}' is not authorized to make governance decisions.")

        # Validate non-empty rationale
        if not rationale or not rationale.strip():
            raise ValueError("A non-empty rationale is mandatory when recording a governance decision.")

        # Parse decision enum
        decision_str = str(decision).lower()
        if isinstance(decision, ReviewDecision):
            decision_enum = decision
        else:
            try:
                decision_enum = ReviewDecision(decision_str)
            except ValueError:
                raise ValueError(f"Unsupported decision '{decision}'. Must be approve, reject, or defer.")

        # Defer decision delegates to dedicated method
        if decision_enum == ReviewDecision.DEFER:
            if defer_follow_up_date is None:
                raise ValueError("defer decision requires a mandatory future defer_follow_up_date.")
            return self.defer(
                change_brief_id=change_brief_id,
                reviewer_id=reviewer_id,
                rationale=rationale,
                defer_follow_up_date=defer_follow_up_date,
            )

        # Approve and Reject must not have defer-only fields
        if defer_follow_up_date is not None:
            raise ValueError(f"defer_follow_up_date must not be provided for '{decision_enum.value}' decision.")

        with self.session_factory() as session:
            brief = session.query(ChangeBrief).filter_by(id=change_brief_id).first()
            if not brief:
                raise RecordNotFoundError(f"ChangeBrief not found with ID: {change_brief_id}")

            current_status = brief.status

            # Enforce that decision MUST be made from 'in_review' state
            if current_status != BriefStatus.IN_REVIEW.value:
                self._record_audit(
                    entity_id=change_brief_id,
                    entity_type="ChangeBrief",
                    previous_status=current_status,
                    new_status=current_status,
                    actor=reviewer_id,
                    reason=f"G4 Gate Violation: Cannot decide brief in status '{current_status}'. Must be 'in_review'.",
                )
                raise InvalidStateTransitionError(current_status, BriefStatus.DECIDED.value, "ChangeBrief")

            assignment = (
                session.query(ReviewAssignment)
                .filter_by(change_brief_id=brief.id)
                .order_by(ReviewAssignment.created_at.desc())
                .first()
            )
            if not assignment:
                raise RecordNotFoundError(f"No assignment found for brief {change_brief_id}.")

            now_utc = datetime.now(timezone.utc)

            # Update assignment
            assignment.decision = decision_enum.value
            assignment.rationale = rationale.strip()
            assignment.decision_timestamp = now_utc
            assignment.status = ReviewAssignmentStatus.COMPLETED.value

            # Update brief status to decided
            brief.status = BriefStatus.DECIDED.value
            session.commit()

            # Record AuditLog
            self._record_audit(
                entity_id=brief.id,
                entity_type="ChangeBrief",
                previous_status=current_status,
                new_status=BriefStatus.DECIDED.value,
                actor=reviewer_id,
                reason=rationale.strip(),
                audit_metadata={
                    "action": decision_enum.value,
                    "decision": decision_enum.value,
                    "rationale": rationale.strip(),
                    "reviewer_id": reviewer_id,
                    "decision_timestamp": now_utc.isoformat(),
                    "assignment_id": assignment.id,
                },
            )

            session.refresh(brief)
            session.expunge(brief)
            return brief

    def defer(
        self,
        change_brief_id: str,
        reviewer_id: str,
        rationale: str,
        defer_follow_up_date: datetime,
    ) -> ChangeBrief:
        """Record a defer governance decision with mandatory future follow-up date.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            reviewer_id: Named authorized reviewer.
            rationale: Mandatory non-empty clinical justification for deferral.
            defer_follow_up_date: Mandatory future date for reassessment.

        Returns:
            Updated ChangeBrief entity in 'deferred' status.
        """
        if not self.auth_service.is_authorized(reviewer_id):
            raise UnauthorizedReviewerError(f"Reviewer '{reviewer_id}' is not authorized to defer briefs.")

        if not rationale or not rationale.strip():
            raise ValueError("A non-empty rationale is mandatory when deferring a ChangeBrief.")

        if defer_follow_up_date is None:
            raise ValueError("defer decision requires a mandatory defer_follow_up_date.")

        # Validate that follow_up_date is strictly in the future
        now_utc = datetime.now(timezone.utc)
        follow_up_utc = defer_follow_up_date
        if follow_up_utc.tzinfo is None:
            follow_up_utc = follow_up_utc.replace(tzinfo=timezone.utc)

        if follow_up_utc <= now_utc:
            raise ValueError(f"defer_follow_up_date ({defer_follow_up_date}) must be strictly in the future.")

        with self.session_factory() as session:
            brief = session.query(ChangeBrief).filter_by(id=change_brief_id).first()
            if not brief:
                raise RecordNotFoundError(f"ChangeBrief not found with ID: {change_brief_id}")

            current_status = brief.status
            if current_status != BriefStatus.IN_REVIEW.value:
                raise InvalidStateTransitionError(current_status, BriefStatus.DEFERRED.value, "ChangeBrief")

            assignment = (
                session.query(ReviewAssignment)
                .filter_by(change_brief_id=brief.id)
                .order_by(ReviewAssignment.created_at.desc())
                .first()
            )
            if not assignment:
                raise RecordNotFoundError(f"No assignment found for brief {change_brief_id}.")

            # Update assignment
            assignment.decision = ReviewDecision.DEFER.value
            assignment.rationale = rationale.strip()
            assignment.decision_timestamp = now_utc
            assignment.defer_follow_up_date = follow_up_utc
            assignment.due_date = follow_up_utc
            assignment.status = ReviewAssignmentStatus.ASSIGNED.value

            # Update brief status to deferred
            brief.status = BriefStatus.DEFERRED.value
            session.commit()

            # Record AuditLog
            self._record_audit(
                entity_id=brief.id,
                entity_type="ChangeBrief",
                previous_status=current_status,
                new_status=BriefStatus.DEFERRED.value,
                actor=reviewer_id,
                reason=rationale.strip(),
                audit_metadata={
                    "action": "defer",
                    "decision": "defer",
                    "rationale": rationale.strip(),
                    "reviewer_id": reviewer_id,
                    "defer_follow_up_date": follow_up_utc.isoformat(),
                    "decision_timestamp": now_utc.isoformat(),
                },
            )

            session.refresh(brief)
            session.expunge(brief)
            return brief

    # ==========================================================================
    # 4. G4 CLOSURE (ONLY AFTER EXPLICIT DECISION)
    # ==========================================================================

    def close_after_decision(
        self,
        change_brief_id: str,
        actor: str,
        closure_note: Optional[str] = None,
    ) -> ChangeBrief:
        """Close a ChangeBrief following an explicit human decision.

        G4 Human Gate: Enforces that NO brief can reach 'closed' without first
        reaching 'decided'. Attempts to close draft, assigned, in_review, or
        deferred briefs are strictly blocked.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            actor: Identity of actor performing the closure.
            closure_note: Optional closing audit note.

        Returns:
            Updated ChangeBrief in 'closed' status.

        Raises:
            GovernanceGateError: If attempted before an explicit human decision.
        """
        if not actor or not actor.strip():
            raise ValueError("Actor identity is required to close a ChangeBrief.")

        with self.session_factory() as session:
            brief = session.query(ChangeBrief).filter_by(id=change_brief_id).first()
            if not brief:
                raise RecordNotFoundError(f"ChangeBrief not found with ID: {change_brief_id}")

            current_status = brief.status

            # Enforce G4: ONLY decided briefs may transition to closed
            if current_status != BriefStatus.DECIDED.value:
                reason_msg = (
                    f"G4 Gate Violation: Cannot close ChangeBrief in status '{current_status}'. "
                    f"A brief must reach 'decided' via an explicit human decision before it can be closed."
                )
                self._record_audit(
                    entity_id=change_brief_id,
                    entity_type="ChangeBrief",
                    previous_status=current_status,
                    new_status=current_status,
                    actor=actor,
                    reason=reason_msg,
                )
                raise GovernanceGateError(reason_msg)

            assignment = (
                session.query(ReviewAssignment)
                .filter_by(change_brief_id=brief.id)
                .order_by(ReviewAssignment.created_at.desc())
                .first()
            )
            if not assignment or not assignment.decision:
                raise GovernanceGateError("G4 Gate Violation: No recorded human decision found on assignment.")

            brief.status = BriefStatus.CLOSED.value
            session.commit()

            # Record AuditLog
            self._record_audit(
                entity_id=brief.id,
                entity_type="ChangeBrief",
                previous_status=current_status,
                new_status=BriefStatus.CLOSED.value,
                actor=actor,
                reason=closure_note or f"Brief closed following explicit '{assignment.decision}' decision.",
                audit_metadata={
                    "action": "close",
                    "final_decision": assignment.decision,
                    "reviewer_id": assignment.reviewer_id,
                    "closed_by": actor,
                },
            )

            session.refresh(brief)
            session.expunge(brief)
            return brief

    # ==========================================================================
    # 5. G5 SLA ESCALATION (DETERMINISTIC, IDEMPOTENT, NON-DECISION)
    # ==========================================================================

    def evaluate_sla(
        self,
        change_brief_id: str,
        current_time: Optional[datetime] = None,
    ) -> SLAEscalationResult:
        """Evaluate a ChangeBrief against its SLA deadline.

        If overdue, creates escalation Notification and AuditLog entries.
        IMPORTANT: Escalation NEVER makes decisions or modifies protocols.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            current_time: Optional reference timestamp (defaults to UTC now).

        Returns:
            SLAEscalationResult.
        """
        now_utc = current_time or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        with self.session_factory() as session:
            brief = session.query(ChangeBrief).filter_by(id=change_brief_id).first()
            if not brief:
                raise RecordNotFoundError(f"ChangeBrief not found with ID: {change_brief_id}")

            # If already decided or closed, not overdue
            if brief.status in (BriefStatus.DECIDED.value, BriefStatus.CLOSED.value):
                return SLAEscalationResult(
                    change_brief_id=brief.id,
                    is_overdue=False,
                    sla_deadline=None,
                    evaluated_at=now_utc,
                    escalated=False,
                    reason=f"Brief is already resolved ({brief.status}).",
                )

            # Determine deadline
            assignment = (
                session.query(ReviewAssignment)
                .filter_by(change_brief_id=brief.id)
                .order_by(ReviewAssignment.created_at.desc())
                .first()
            )

            deadline = None
            if assignment and assignment.due_date:
                deadline = assignment.due_date
            elif brief.impact_record and brief.impact_record.sla_deadline:
                deadline = brief.impact_record.sla_deadline

            if not deadline:
                return SLAEscalationResult(
                    change_brief_id=brief.id,
                    is_overdue=False,
                    sla_deadline=None,
                    evaluated_at=now_utc,
                    escalated=False,
                    reason="No SLA deadline assigned to this brief.",
                )

            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)

            # Check if overdue
            is_overdue = now_utc > deadline
            if not is_overdue:
                return SLAEscalationResult(
                    change_brief_id=brief.id,
                    is_overdue=False,
                    sla_deadline=deadline,
                    evaluated_at=now_utc,
                    escalated=False,
                    reason="Brief is within SLA deadline window.",
                )

            # Idempotency check: verify if already escalated
            existing_escalation_notif = (
                session.query(Notification)
                .filter_by(related_entity_id=brief.id, notification_type="escalation")
                .first()
            )
            if existing_escalation_notif or (assignment and assignment.status == ReviewAssignmentStatus.ESCALATED.value):
                return SLAEscalationResult(
                    change_brief_id=brief.id,
                    is_overdue=True,
                    sla_deadline=deadline,
                    evaluated_at=now_utc,
                    escalated=False,
                    notification_id=existing_escalation_notif.id if existing_escalation_notif else None,
                    reason="Brief is overdue but already escalated. Duplicate escalation skipped.",
                )

            # Perform escalation
            if assignment:
                assignment.status = ReviewAssignmentStatus.ESCALATED.value
                session.flush()

            # Record Notification
            tier_label = brief.impact_record.tier if brief.impact_record else "Standard"
            target_committee = brief.impact_record.routing_target if brief.impact_record else "Clinical Governance Board"

            notif = Notification(
                id=str(uuid.uuid4()),
                related_entity_id=brief.id,
                related_entity_type="ChangeBrief",
                notification_type="escalation",
                recipient=target_committee,
                payload={
                    "brief_id": brief.id,
                    "deadline": deadline.isoformat(),
                    "evaluated_at": now_utc.isoformat(),
                    "tier": tier_label,
                    "reviewer_id": assignment.reviewer_id if assignment else None,
                },
                delivery_status="pending",
                schema_version="1.0",
            )
            session.add(notif)
            session.flush()

            # Record AuditLog
            audit = AuditLog(
                id=str(uuid.uuid4()),
                entity_id=brief.id,
                entity_type="ChangeBrief",
                previous_status=brief.status,
                new_status=brief.status,  # Note: Brief status does NOT change automatically
                actor="system_sla_monitor",
                reason=f"G5 SLA Breach: Brief overdue (deadline was {deadline.isoformat()}). Escalated to {target_committee}.",
                audit_metadata={
                    "action": "sla_escalation",
                    "deadline": deadline.isoformat(),
                    "evaluated_at": now_utc.isoformat(),
                    "notification_id": notif.id,
                },
                schema_version="1.0",
            )
            session.add(audit)
            session.commit()

            return SLAEscalationResult(
                change_brief_id=brief.id,
                is_overdue=True,
                sla_deadline=deadline,
                evaluated_at=now_utc,
                escalated=True,
                notification_id=notif.id,
                audit_log_id=audit.id,
                reason="SLA breach detected. Escalation recorded.",
            )

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _record_audit(
        self,
        entity_id: str,
        entity_type: str,
        previous_status: Optional[str],
        new_status: str,
        actor: str,
        reason: Optional[str] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Persist an append-only AuditLog record in a dedicated transaction."""
        with self.session_factory() as session:
            log_entry = AuditLog(
                id=str(uuid.uuid4()),
                entity_id=entity_id,
                entity_type=entity_type,
                previous_status=previous_status,
                new_status=new_status,
                actor=actor,
                reason=reason,
                audit_metadata=audit_metadata,
                schema_version="1.0",
            )
            session.add(log_entry)
            session.commit()
            session.refresh(log_entry)
            session.expunge(log_entry)
            return log_entry

    # ==========================================================================
    # CLINICIAN QUERY MODE: protocol-update review status for the clinician
    # ==========================================================================

    def governance_for_query(self, query: Any, answer: Any, actor: str = "clinician") -> List[Any]:
        """For each out-of-date protocol in the answer, report the related change brief's status.

        If no brief exists yet, the governance team is notified (deduplicated). This never assigns
        a reviewer automatically to an unrelated person and never records a decision.
        """
        from app.schemas.treatment_check import GovernanceStatus

        statuses: List[GovernanceStatus] = []
        for position in answer.protocol_positions:
            if position.status != "out_of_date":
                continue
            brief = self._find_brief_for_protocol(position.protocol_id, list(query.treatment_ids))
            if brief is not None:
                statuses.append(self._brief_status(brief, position.protocol_id))
                continue
            flag = self._flag_protocol_out_of_date(position, query, answer, actor)
            statuses.append(GovernanceStatus(
                protocol_id=position.protocol_id,
                status="flagged_to_governance",
                message=(f"Protocol {position.protocol_id} {position.protocol_version} is out of date for "
                         f"{query.plan.text}; no change brief exists yet. The governance team has been notified "
                         f"(notification {flag.id[:8]})."),
            ))
        return statuses

    def _find_brief_for_protocol(self, protocol_id: str, treatment_ids: List[str]) -> Optional[ChangeBrief]:
        from app.services.taxonomy import TaxonomyError, get_taxonomy

        try:
            taxonomy = get_taxonomy()
        except TaxonomyError:
            taxonomy = None
        with self.session_factory() as session:
            briefs = session.query(ChangeBrief).order_by(ChangeBrief.created_at.desc()).all()
            for brief in briefs:
                payload = brief.structured_payload or {}
                if (payload.get("current_protocol") or {}).get("protocol_id") != protocol_id:
                    continue
                text = (payload.get("what_changed") or {}).get("recommendation_text") or ""
                mentioned = taxonomy.find_treatments(text).treatments if taxonomy else []
                if not treatment_ids or set(mentioned) & set(treatment_ids):
                    session.expunge(brief)
                    return brief
        return None

    def _brief_status(self, brief: ChangeBrief, protocol_id: str) -> Any:
        from app.schemas.treatment_check import GovernanceStatus

        with self.session_factory() as session:
            assignments = session.query(ReviewAssignment).filter_by(change_brief_id=brief.id).all()
            reviewers = [f"{a.reviewer_role} ({a.reviewer_id})" for a in assignments]
            decided = next((a for a in assignments if a.decision), None)
        status = brief.status
        if status in (BriefStatus.ASSIGNED.value, BriefStatus.IN_REVIEW.value):
            message = f"Protocol update pending specialist review ({', '.join(reviewers) or 'assignment pending'})."
        elif status == BriefStatus.DEFERRED.value and decided is not None:
            follow = decided.defer_follow_up_date.date().isoformat() if decided.defer_follow_up_date else "a set date"
            message = f"Protocol update review deferred until {follow}."
        elif status in (BriefStatus.DECIDED.value, BriefStatus.CLOSED.value) and decided is not None:
            when = decided.decision_timestamp.date().isoformat() if decided.decision_timestamp else ""
            message = f"Protocol update {decided.decision}d on {when} by {decided.reviewer_role}."
        elif status == BriefStatus.DRAFT.value:
            message = "Protocol update brief drafted; awaiting routing to a specialist."
        else:
            message = f"Protocol update brief status: {status}."
        return GovernanceStatus(protocol_id=protocol_id, brief_id=brief.id, status=status, message=message,
                                reviewers=reviewers)

    def _flag_protocol_out_of_date(self, position: Any, query: Any, answer: Any, actor: str) -> Notification:
        key = {"protocol_id": position.protocol_id, "protocol_version": position.protocol_version,
               "treatments": sorted(query.treatment_ids)}
        with self.session_factory() as session:
            for existing in session.query(Notification).filter_by(notification_type="protocol_out_of_date",
                                                                    delivery_status="pending"):
                payload = existing.payload or {}
                if all(payload.get(k) == v for k, v in key.items()):
                    session.expunge(existing)
                    return existing
        evidence = [i.latest_citation.statement_id for i in position.items if i.latest_citation]
        return self._record_notification(
            related_entity_id=position.protocol_id,
            related_entity_type="Protocol",
            notification_type="protocol_out_of_date",
            recipient="governance_team",
            payload={**key, "department": query.department, "raised_by": actor,
                     "evidence_statement_ids": evidence,
                     "reasons": [i.reason for i in position.items if i.status != "aligned"]},
        )

    def _record_notification(
        self,
        related_entity_id: str,
        related_entity_type: str,
        notification_type: str,
        recipient: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """Persist a Notification record."""
        with self.session_factory() as session:
            notif = Notification(
                id=str(uuid.uuid4()),
                related_entity_id=related_entity_id,
                related_entity_type=related_entity_type,
                notification_type=notification_type,
                recipient=recipient,
                payload=payload,
                delivery_status="pending",
                schema_version="1.0",
            )
            session.add(notif)
            session.commit()
            session.refresh(notif)
            session.expunge(notif)
            return notif
