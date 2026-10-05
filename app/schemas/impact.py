"""Pydantic schemas for ImpactRecord multidimensional scoring."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ImpactTier(str, Enum):
    """Routing urgency tiers based on total impact score."""
    CRITICAL = "Critical"      # 12-15 (48 hours)
    HIGH = "High"              # 8-11 (7 days)
    STANDARD = "Standard"      # 4-7 (30 days)
    LOW = "Low"                # 3 (90 days)


class ImpactStatus(str, Enum):
    """Controlled lifecycle states for impact evaluation records."""
    PENDING = "pending"
    CALCULATED = "calculated"
    ROUTED = "routed"
    INCOMPLETE = "incomplete"


class DimensionScore(BaseModel):
    """Auditable result for a single scored impact dimension."""
    model_config = ConfigDict(validate_assignment=True)

    dimension: str = Field(..., description="Dimension name: clinical_urgency, evidence_strength, or pathway_breadth.")
    score: int = Field(..., ge=1, le=5, description="Resolved score between 1 and 5.")
    rule_id: str = Field(..., description="Stable rule identifier, e.g. URGENCY-05.")
    rule_version: str = Field(..., description="Version of the scoring rules applied.")
    basis: str = Field(..., description="Written rationale explaining the clinical or evidentiary basis.")


class ScoringResult(BaseModel):
    """Deterministic result produced by the ScoringEngine."""
    model_config = ConfigDict(validate_assignment=True)

    clinical_urgency: Optional[int] = Field(default=None, ge=1, le=5)
    evidence_strength: Optional[int] = Field(default=None, ge=1, le=5)
    pathway_breadth: Optional[int] = Field(default=None, ge=1, le=5)
    rule_ids: Dict[str, str] = Field(default_factory=dict)
    scoring_yaml_version: str = Field(default="1.0")
    total_score: Optional[int] = Field(default=None, ge=3, le=15)
    tier: Optional[ImpactTier] = Field(default=None)
    routing_target: Optional[str] = Field(default=None)
    sla_deadline: Optional[datetime] = Field(default=None)
    sla_hours: Optional[int] = Field(default=None)
    urgency_basis: Optional[str] = Field(default=None)
    evidence_basis: Optional[str] = Field(default=None)
    breadth_basis: Optional[str] = Field(default=None)
    status: ImpactStatus = Field(default=ImpactStatus.PENDING)
    incomplete_reason: Optional[str] = Field(default=None)
    dimension_scores: Dict[str, DimensionScore] = Field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check whether all three dimensions scored successfully."""
        return (
            self.clinical_urgency is not None
            and self.evidence_strength is not None
            and self.pathway_breadth is not None
        )


class ImpactRecordBase(BaseModel):
    """Base schema for ImpactRecord models."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    gap_record_id: str
    clinical_urgency: Optional[int] = Field(default=None, ge=1, le=5)
    evidence_strength: Optional[int] = Field(default=None, ge=1, le=5)
    pathway_breadth: Optional[int] = Field(default=None, ge=1, le=5)
    rule_ids: Optional[Any] = Field(default_factory=list)
    scoring_yaml_version: str = Field(default="1.0", min_length=1)
    total_score: Optional[int] = Field(default=None, ge=3, le=15)
    tier: Optional[ImpactTier] = Field(default=None)
    routing_target: Optional[str] = Field(default=None)
    sla_deadline: Optional[datetime] = Field(default=None)
    urgency_basis: Optional[str] = Field(default=None)
    evidence_basis: Optional[str] = Field(default=None)
    breadth_basis: Optional[str] = Field(default=None)
    status: ImpactStatus = Field(default=ImpactStatus.PENDING)
    incomplete_reason: Optional[str] = Field(default=None)
    schema_version: str = Field(default="1.0", min_length=1)

    @field_validator("gap_record_id")
    @classmethod
    def _validate_gap_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)

    @model_validator(mode="after")
    def _validate_total_score_consistency(self) -> "ImpactRecordBase":
        u = self.clinical_urgency
        e = self.evidence_strength
        b = self.pathway_breadth
        # If incomplete status, tier and SLA must be unset
        if self.status == ImpactStatus.INCOMPLETE:
            if self.tier is not None:
                raise ValueError("Incomplete score must not have an assigned tier.")
            if self.total_score is not None:
                raise ValueError("Incomplete score must not have a total_score.")
        # If all 3 dimensions are present, total_score must equal their sum
        if u is not None and e is not None and b is not None:
            expected_sum = u + e + b
            if self.total_score is not None and self.total_score != expected_sum:
                raise ValueError(
                    f"total_score ({self.total_score}) must equal the sum of dimensions ({expected_sum})"
                )
        return self


class ImpactRecordCreate(ImpactRecordBase):
    """Schema for creating an ImpactRecord."""
    pass


class ImpactRecordRead(ImpactRecordBase):
    """Schema for reading an ImpactRecord."""
    id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)
