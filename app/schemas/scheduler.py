"""Pydantic schemas for the Phase 11 SLA Scheduler and Escalation Cycle."""

from datetime import datetime, timezone
from typing import List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.governance import SLAEscalationResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SLACycleSummary(BaseModel):
    """Deterministic structured outcome of an SLA evaluation cycle."""

    model_config = ConfigDict(validate_assignment=True)

    cycle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=_utc_now)
    evaluated_count: int = Field(default=0, ge=0)
    overdue_count: int = Field(default=0, ge=0)
    escalated_count: int = Field(default=0, ge=0)
    already_escalated_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    results: List[SLAEscalationResult] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0, ge=0.0)
    schema_version: str = Field(default="1.0", min_length=1)

    @property
    def has_errors(self) -> bool:
        """Whether any brief evaluation raised an unexpected error."""
        return self.failed_count > 0 or len(self.errors) > 0

    @property
    def total_processed(self) -> int:
        """Total number of briefs evaluated or skipped."""
        return self.evaluated_count + self.skipped_count + self.failed_count
