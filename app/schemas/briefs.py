"""Pydantic schemas for ChangeBrief models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BriefStatus(str, Enum):
    """Controlled lifecycle states for clinical change briefs."""
    DRAFT = "draft"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    DEFERRED = "deferred"
    DECIDED = "decided"
    CLOSED = "closed"


class ChangeBriefBase(BaseModel):
    """Base schema for ChangeBrief models."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    impact_record_id: str
    status: BriefStatus = Field(default=BriefStatus.DRAFT)
    rendered_file_path: Optional[str] = Field(default=None)
    rendered_file_hash: Optional[str] = Field(default=None)
    structured_payload: Optional[Dict[str, Any]] = Field(default=None)
    schema_version: str = Field(default="1.0", min_length=1)

    @field_validator("impact_record_id")
    @classmethod
    def _validate_impact_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)


class ChangeBriefCreate(ChangeBriefBase):
    """Schema for creating a ChangeBrief."""
    pass


class ChangeBriefRead(ChangeBriefBase):
    """Schema for reading a ChangeBrief."""
    id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)
