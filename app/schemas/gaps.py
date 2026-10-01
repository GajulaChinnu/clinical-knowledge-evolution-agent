"""Pydantic schemas for GapRecord protocol comparison results."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator


class GapStatus(str, Enum):
    """Controlled lifecycle states for protocol gap records."""
    PENDING = "pending"
    MATCHED = "matched"
    NO_MATCH = "no_match"
    REVIEW_REQUIRED = "review_required"
    RESOLVED = "resolved"


class ComparisonResult(str, Enum):
    """Comparison conclusion between recommendation and protocol."""
    GAP = "gap"
    NO_GAP = "no_gap"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


class DifferenceType(str, Enum):
    """Specific nature of the clinical difference detected."""
    NEW_RECOMMENDATION = "new_recommendation"
    CONFLICT = "conflict"
    DOSAGE_CHANGE = "dosage_change"
    SCOPE_EXPANSION = "scope_expansion"
    NONE = "none"


class GapRecordBase(BaseModel):
    """Base schema for GapRecord models."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    change_record_id: str
    candidate_protocol_section_ids: Optional[List[str]] = Field(default_factory=list)
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    comparison_result: ComparisonResult
    comparison_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    difference_type: Optional[DifferenceType] = Field(default=None)
    matched_protocol_id: Optional[str] = Field(default=None)
    matched_protocol_version: Optional[str] = Field(default=None)
    is_match: bool = Field(default=True)
    reviewer_resolution: Optional[str] = Field(default=None)
    status: GapStatus = Field(default=GapStatus.PENDING)
    schema_version: str = Field(default="1.0", min_length=1)

    @field_validator("change_record_id")
    @classmethod
    def _validate_change_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)


class GapRecordCreate(GapRecordBase):
    """Schema for creating a GapRecord."""
    pass


class GapRecordRead(GapRecordBase):
    """Schema for reading a GapRecord."""
    id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)
