"""Pydantic schemas for ReviewAssignment and clinical governance decisions."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GovernanceGateError(Exception):
    """Base exception for G4/G5 clinical governance gate violations."""
    pass


class UnauthorizedReviewerError(GovernanceGateError):
    """Raised when an unauthorized individual attempts a clinical governance action."""
    pass


class ReviewDecision(str, Enum):
    """Explicit human governance decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


class ReviewAssignmentStatus(str, Enum):
    """Controlled lifecycle states for review assignments."""
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ReviewAssignmentBase(BaseModel):
    """Base schema for ReviewAssignment models."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    change_brief_id: str
    reviewer_role: str = Field(..., min_length=1)
    reviewer_id: str = Field(..., min_length=1)
    status: ReviewAssignmentStatus = Field(default=ReviewAssignmentStatus.ASSIGNED)
    due_date: datetime
    decision: Optional[ReviewDecision] = Field(default=None)
    rationale: Optional[str] = Field(default=None)
    decision_timestamp: Optional[datetime] = Field(default=None)
    defer_follow_up_date: Optional[datetime] = Field(default=None)
    schema_version: str = Field(default="1.0", min_length=1)

    @field_validator("change_brief_id")
    @classmethod
    def _validate_brief_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)

    @model_validator(mode="after")
    def _validate_decision_and_rationale(self) -> "ReviewAssignmentBase":
        # 1. Non-empty rationale required if a decision is recorded
        if self.decision is not None:
            if not self.rationale or not self.rationale.strip():
                raise ValueError("A non-empty rationale is required when recording a decision.")

        # 2. Defer decision requires a follow-up due date in the future
        if self.decision == ReviewDecision.DEFER:
            if self.defer_follow_up_date is None:
                raise ValueError("defer decision requires a valid defer_follow_up_date.")

        return self


class ReviewAssignmentCreate(ReviewAssignmentBase):
    """Schema for creating a new ReviewAssignment."""
    pass


class ReviewAssignmentDecisionUpdate(BaseModel):
    """Schema for recording a governance decision on an existing assignment."""
    model_config = ConfigDict(validate_assignment=True)

    decision: ReviewDecision
    rationale: str = Field(..., min_length=1)
    decision_timestamp: Optional[datetime] = Field(default=None)
    defer_follow_up_date: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def _validate_update_rules(self) -> "ReviewAssignmentDecisionUpdate":
        if not self.rationale or not self.rationale.strip():
            raise ValueError("A non-empty rationale is required when recording a decision.")

        if self.decision == ReviewDecision.DEFER:
            if self.defer_follow_up_date is None:
                raise ValueError("defer decision requires a valid defer_follow_up_date.")

        return self


class ReviewAssignmentRead(ReviewAssignmentBase):
    """Schema for reading a ReviewAssignment."""
    id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)


# ==============================================================================
# PHASE 9 GOVERNANCE REQUEST & RESULT SCHEMAS
# ==============================================================================

class ReviewerAssignmentRequest(BaseModel):
    """Schema for assigning an authorized reviewer to a ChangeBrief."""
    model_config = ConfigDict(validate_assignment=True)

    change_brief_id: str
    reviewer_id: str = Field(..., min_length=1)
    reviewer_role: str = Field(default="Clinical Governance Reviewer", min_length=1)
    due_date: Optional[datetime] = Field(default=None)
    actor: Optional[str] = Field(default=None)

    @field_validator("change_brief_id")
    @classmethod
    def _validate_brief_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)

    @field_validator("reviewer_id")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reviewer_id cannot be empty or whitespace only.")
        return v.strip()


class ReassignmentRequest(BaseModel):
    """Schema for reassigning an existing ChangeBrief to a new reviewer."""
    model_config = ConfigDict(validate_assignment=True)

    change_brief_id: str
    new_reviewer_id: str = Field(..., min_length=1)
    new_reviewer_role: Optional[str] = Field(default=None)
    reason: str = Field(..., min_length=1)
    actor: str = Field(..., min_length=1)
    new_due_date: Optional[datetime] = Field(default=None)

    @field_validator("change_brief_id")
    @classmethod
    def _validate_brief_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)

    @field_validator("new_reviewer_id", "reason", "actor")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()


class GovernanceDecisionRequest(BaseModel):
    """Schema for recording an explicit human governance decision."""
    model_config = ConfigDict(validate_assignment=True)

    change_brief_id: str
    reviewer_id: str = Field(..., min_length=1)
    decision: ReviewDecision
    rationale: str = Field(..., min_length=1)
    defer_follow_up_date: Optional[datetime] = Field(default=None)

    @field_validator("change_brief_id")
    @classmethod
    def _validate_brief_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)

    @field_validator("reviewer_id", "rationale")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()

    @model_validator(mode="after")
    def _validate_decision_constraints(self) -> "GovernanceDecisionRequest":
        now = datetime.now(timezone.utc)
        if self.decision == ReviewDecision.DEFER:
            if self.defer_follow_up_date is None:
                raise ValueError("defer decision requires a mandatory defer_follow_up_date.")
            # Verify due date is strictly in the future
            date_to_check = self.defer_follow_up_date
            if date_to_check.tzinfo is None:
                date_to_check = date_to_check.replace(tzinfo=timezone.utc)
            if date_to_check <= now:
                raise ValueError(f"defer_follow_up_date ({self.defer_follow_up_date}) must be strictly in the future.")
        elif self.decision in (ReviewDecision.APPROVE, ReviewDecision.REJECT):
            if self.defer_follow_up_date is not None:
                raise ValueError(f"defer_follow_up_date must not be provided for {self.decision.value} decision.")

        return self


class DeferDecisionRequest(BaseModel):
    """Dedicated schema for deferring a ChangeBrief with mandatory future follow-up date."""
    model_config = ConfigDict(validate_assignment=True)

    change_brief_id: str
    reviewer_id: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    defer_follow_up_date: datetime

    @field_validator("change_brief_id")
    @classmethod
    def _validate_brief_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)

    @field_validator("reviewer_id", "rationale")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()

    @model_validator(mode="after")
    def _validate_future_date(self) -> "DeferDecisionRequest":
        now = datetime.now(timezone.utc)
        date_to_check = self.defer_follow_up_date
        if date_to_check.tzinfo is None:
            date_to_check = date_to_check.replace(tzinfo=timezone.utc)
        if date_to_check <= now:
            raise ValueError(f"defer_follow_up_date ({self.defer_follow_up_date}) must be strictly in the future.")
        return self


class SLAEscalationResult(BaseModel):
    """Result of evaluating a ChangeBrief against its SLA deadline."""
    model_config = ConfigDict(validate_assignment=True)

    change_brief_id: str
    is_overdue: bool
    sla_deadline: Optional[datetime] = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    escalated: bool = False
    notification_id: Optional[str] = None
    audit_log_id: Optional[str] = None
    reason: Optional[str] = None
