"""Pydantic schemas for ChangeRecord entities."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChangeStatus(str, Enum):
    """Controlled lifecycle states for extracted recommendations."""
    EXTRACTED = "extracted"
    COMPARISON_PENDING = "comparison_pending"
    GAP_CONFIRMED = "gap_confirmed"
    NO_GAP = "no_gap"
    HELD_FOR_G1 = "held_for_G1"
    HELD_FOR_G2 = "held_for_G2"
    HELD_FOR_G3 = "held_for_G3"


class ChangeRecordBase(BaseModel):
    """Base schema for ChangeRecord models."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    ingested_document_id: str
    verbatim_text: str = Field(..., min_length=1)
    recommendation_type: str = Field(..., min_length=1)
    target_population: str = Field(..., min_length=1)
    intervention: str = Field(..., min_length=1)
    evidence_grade: Optional[str] = Field(default=None)
    confidence: float = Field(..., ge=0.0, le=1.0)
    page: Optional[int] = Field(default=None, ge=1)
    section: Optional[str] = Field(default=None)
    source_excerpt: Optional[str] = Field(default=None)
    extraction_model_version: str = Field(..., min_length=1)
    extraction_prompt_version: str = Field(..., min_length=1)
    status: ChangeStatus = Field(default=ChangeStatus.EXTRACTED)
    schema_version: str = Field(default="1.0", min_length=1)

    @field_validator("ingested_document_id")
    @classmethod
    def _validate_doc_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)


class ChangeRecordCreate(ChangeRecordBase):
    """Schema for creating a ChangeRecord."""
    pass


class ChangeRecordRead(ChangeRecordBase):
    """Schema for reading a ChangeRecord."""
    id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)
