"""Pydantic schemas for CKEA Pipeline Orchestration and State Machine.

Defines strongly validated models for:
1. PipelineStage, PipelineStatus, HumanGate enums
2. ArtifactsManifest for generated entity identifiers
3. PipelineResult for stage-aware execution outcomes
4. BatchPipelineResult for multi-document batch orchestration
5. Resolution requests for human gates G1, G2, G3
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PipelineStage(str, Enum):
    """Controlled processing stages of the end-to-end clinical pipeline."""
    MONITORING = "monitoring"
    EXTRACTION = "extraction"
    COMPARISON = "comparison"
    IMPACT = "impact"
    BRIEFING = "briefing"
    GOVERNANCE = "governance"
    CLOSURE = "closure"


class PipelineStatus(str, Enum):
    """Aggregate lifecycle outcomes for pipeline execution."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SUCCESS = "completed"
    HELD = "held"
    FAILED = "failed"
    SKIPPED = "skipped"


class HumanGate(str, Enum):
    """Explicit human oversight checkpoints defined in AGENTS.md."""
    G1 = "G1"  # Extraction review (low confidence or provenance failure)
    G2 = "G2"  # Comparison review (ambiguous, conflicting, or low confidence)
    G3 = "G3"  # Confirmed no-match protocol review
    G4 = "G4"  # Mandatory governance decision (approve/reject/defer)
    G5 = "G5"  # SLA escalation diagnostic check


class ArtifactsManifest(BaseModel):
    """Persistent entity IDs produced or associated with this pipeline run."""
    model_config = ConfigDict(validate_assignment=True)

    document_id: Optional[str] = None
    change_record_ids: List[str] = Field(default_factory=list)
    gap_record_ids: List[str] = Field(default_factory=list)
    impact_record_ids: List[str] = Field(default_factory=list)
    change_brief_ids: List[str] = Field(default_factory=list)
    review_assignment_ids: List[str] = Field(default_factory=list)

    @property
    def change_record_id(self) -> Optional[str]:
        return self.change_record_ids[0] if self.change_record_ids else None

    @property
    def gap_record_id(self) -> Optional[str]:
        return self.gap_record_ids[0] if self.gap_record_ids else None

    @property
    def impact_record_id(self) -> Optional[str]:
        return self.impact_record_ids[0] if self.impact_record_ids else None

    @property
    def brief_id(self) -> Optional[str]:
        return self.change_brief_ids[0] if self.change_brief_ids else None

    @property
    def assignment_id(self) -> Optional[str]:
        return self.review_assignment_ids[0] if self.review_assignment_ids else None


class PipelineResult(BaseModel):
    """Structured, deterministic outcome of pipeline orchestration."""
    model_config = ConfigDict(validate_assignment=True)

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    source_identifier: Optional[str] = None
    status: PipelineStatus = Field(default=PipelineStatus.IN_PROGRESS)
    current_stage: Optional[PipelineStage] = None
    completed_stages: List[PipelineStage] = Field(default_factory=list)
    blocked_stage: Optional[PipelineStage] = None
    held_gate: Optional[HumanGate] = None
    reason: Optional[str] = None
    artifacts: ArtifactsManifest = Field(default_factory=ArtifactsManifest)
    retry_count: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    execution_timestamp: datetime = Field(default_factory=_utc_now)
    schema_version: str = Field(default="1.0", min_length=1)

    @property
    def blocked_gate(self) -> Optional[HumanGate]:
        """Alias for held_gate."""
        return self.held_gate

    @property
    def is_held(self) -> bool:
        """Whether execution stopped at a human gate."""
        return self.status == PipelineStatus.HELD

    @property
    def is_completed(self) -> bool:
        """Whether all applicable pipeline stages completed successfully."""
        return self.status in (PipelineStatus.COMPLETED, PipelineStatus.SUCCESS)

    @property
    def is_failed(self) -> bool:
        """Whether pipeline encountered an unrecoverable failure."""
        return self.status == PipelineStatus.FAILED


class BatchPipelineResult(BaseModel):
    """Aggregate execution summary for batch document processing."""
    model_config = ConfigDict(validate_assignment=True)

    total_documents: int = 0
    completed_count: int = 0
    held_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    results: List[PipelineResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utc_now)


# ==============================================================================
# HUMAN GATE RESOLUTION MODELS
# ==============================================================================

class G1ResolutionRequest(BaseModel):
    """Input payload to resolve a G1 extraction human gate."""
    model_config = ConfigDict(validate_assignment=True)

    action: str = Field(..., description="'proceed' or 'reject'")
    reviewer: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    corrected_text: Optional[str] = None


class G2ResolutionRequest(BaseModel):
    """Input payload to resolve a G2 comparison human gate."""
    model_config = ConfigDict(validate_assignment=True)

    action: str = Field(..., description="'confirm_gap' or 'no_gap'")
    reviewer: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    difference_type: Optional[str] = None


class G3ResolutionRequest(BaseModel):
    """Input payload to resolve a G3 no-match human review."""
    model_config = ConfigDict(validate_assignment=True)

    action: str = Field(default="confirm_no_match", description="'confirm_no_match' or 'reject'")
    reviewer: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    urgency_input: Optional[Any] = None
    evidence_input: Optional[Any] = None
    breadth_input: Optional[Any] = None
