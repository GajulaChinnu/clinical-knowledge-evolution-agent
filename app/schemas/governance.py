"""Pydantic schemas for ReviewAssignment and clinical governance decisions."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
            if self.decision_timestamp is None:
                # If decision is recorded, ensure decision_timestamp is set or generated
                pass

        # 2. Defer decision requires a follow-up due date
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
