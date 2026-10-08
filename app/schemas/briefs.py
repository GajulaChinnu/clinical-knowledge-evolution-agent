"""Pydantic schemas for ChangeBrief models and Structured Brief Payload."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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


class BriefCompletenessError(Exception):
    """Raised when a ChangeBrief fails required completeness validation."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


# ==============================================================================
# SECTION PAYLOAD MODELS
# ==============================================================================

class SourceMetadataPayload(BaseModel):
    """Provenance metadata identifying the external clinical guideline source."""

    model_config = ConfigDict(validate_assignment=True)

    source_identifier: str = Field(..., min_length=1, description="External document identifier (e.g. ADA-2026.pdf)")
    source_path: str = Field(..., min_length=1, description="Path where document was located")
    source_version: str = Field(default="1.0", min_length=1, description="Version of external source document")
    document_version: str = Field(default="1.0", min_length=1, description="Version of ingested document")
    sha256_hash: Optional[str] = Field(default=None, description="SHA-256 hash of external document")


class RecommendationSectionPayload(BaseModel):
    """Section 1: What changed (verbatim recommendation from external evidence)."""

    model_config = ConfigDict(validate_assignment=True)

    recommendation_text: str = Field(..., min_length=1, description="Verbatim text of clinical recommendation")
    source_identifier: str = Field(..., min_length=1, description="Source document identifier")
    page: Optional[int] = Field(default=None, description="Page number where recommendation occurs")
    section: Optional[str] = Field(default=None, description="Section heading in source document")
    recommendation_type: str = Field(..., min_length=1, description="Category of recommendation (pharmacotherapy, etc.)")
    target_population: str = Field(..., min_length=1, description="Target patient population")
    intervention: str = Field(..., min_length=1, description="Specific clinical intervention or test")
    evidence_grade: Optional[str] = Field(default=None, description="Source guideline evidence level/grade")
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ProtocolSectionPayload(BaseModel):
    """Section 2: What our protocol currently says (exact protocol section or explicit no-match)."""

    model_config = ConfigDict(validate_assignment=True)

    is_match: bool = Field(default=True, description="True if a candidate protocol section matched; False for no-match")
    protocol_id: Optional[str] = Field(default=None, description="Institutional protocol identifier")
    protocol_version: Optional[str] = Field(default=None, description="Version of matched institutional protocol")
    section_id: Optional[str] = Field(default=None, description="Section identifier within protocol")
    section_heading: Optional[str] = Field(default=None, description="Heading of matched protocol section")
    exact_protocol_text: Optional[str] = Field(default=None, description="Verbatim quotation of protocol text")
    no_match_statement: Optional[str] = Field(default=None, description="Explicit statement when no protocol matches")


class ComparisonSectionPayload(BaseModel):
    """Section 3: Specific difference between recommendation and protocol."""

    model_config = ConfigDict(validate_assignment=True)

    specific_difference: str = Field(..., min_length=1, description="One-sentence specific difference statement")
    difference_type: str = Field(..., min_length=1, description="Controlled category of difference detected")
    comparison_result: str = Field(..., min_length=1, description="gap, no_gap, ambiguous, or no_match")
    comparison_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    candidate_section_ids: List[str] = Field(default_factory=list, description="IDs of candidate protocol sections evaluated")


class ImpactSectionPayload(BaseModel):
    """Section 4: Impact assessment across the three scoring dimensions."""

    model_config = ConfigDict(validate_assignment=True)

    status: str = Field(default="completed", description="completed or incomplete")
    is_complete: bool = Field(default=True, description="True if all three dimensions resolved deterministically")
    clinical_urgency: Optional[int] = Field(default=None, ge=1, le=5)
    urgency_basis: Optional[str] = Field(default=None)
    evidence_strength: Optional[int] = Field(default=None, ge=1, le=5)
    evidence_basis: Optional[str] = Field(default=None)
    pathway_breadth: Optional[int] = Field(default=None, ge=1, le=5)
    breadth_basis: Optional[str] = Field(default=None)
    total_score: Optional[int] = Field(default=None, ge=3, le=15)
    tier: Optional[str] = Field(default=None, description="Critical, High, Standard, Low")
    routing_target: Optional[str] = Field(default=None, description="Governance committee or body assigned")
    sla_hours: Optional[int] = Field(default=None)
    sla_deadline: Optional[str] = Field(default=None, description="ISO-8601 formatted deadline timestamp")
    rule_ids: Dict[str, str] = Field(default_factory=dict, description="Audit rule IDs per dimension")
    scoring_yaml_version: str = Field(default="1.0")
    incomplete_reason: Optional[str] = Field(default=None, description="Reason if scoring is incomplete")


class WorkflowSectionPayload(BaseModel):
    """Section 5: Affected workflows."""

    model_config = ConfigDict(validate_assignment=True)

    affected_workflows: List[str] = Field(default_factory=list, description="List of affected clinical workflows")
    workflow_summary: Optional[str] = Field(default=None, description="Summary description of workflow impact")
    is_available: bool = Field(default=True, description="True if workflow information is available")
    unavailability_reason: Optional[str] = Field(default=None, description="Explicit statement if unavailable")


class ProposedActionItem(BaseModel):
    """Section 6: Single proposed review action for clinical governance."""

    model_config = ConfigDict(validate_assignment=True)

    step: int = Field(..., ge=1, le=3)
    action: str = Field(..., min_length=1, description="Title of proposed action")
    description: str = Field(..., min_length=1, description="Description of review action")


class SourceExcerptSectionPayload(BaseModel):
    """Section 7: Source excerpt (full source paragraph containing the recommendation)."""

    model_config = ConfigDict(validate_assignment=True)

    source_excerpt: str = Field(..., min_length=1, description="Full verified verbatim source paragraph")
    source_identifier: str = Field(..., min_length=1, description="Source document identifier")
    page: Optional[int] = Field(default=None)
    section: Optional[str] = Field(default=None)


# ==============================================================================
# STRUCTURED BRIEF PAYLOAD
# ==============================================================================

class StructuredBriefPayload(BaseModel):
    """Complete, self-contained, serializable payload representing all seven brief sections."""

    model_config = ConfigDict(validate_assignment=True)

    brief_id: str = Field(..., description="Unique brief identifier (UUID or derived)")
    impact_record_id: str = Field(..., description="Foreign key to parent ImpactRecord")
    gap_record_id: str = Field(..., description="Foreign key to parent GapRecord")
    change_record_id: str = Field(..., description="Foreign key to parent ChangeRecord")
    schema_version: str = Field(default="1.0", min_length=1)
    generated_at: str = Field(..., description="ISO-8601 generation timestamp")

    # The Seven Required Sections
    source_metadata: SourceMetadataPayload
    what_changed: RecommendationSectionPayload
    current_protocol: ProtocolSectionPayload
    specific_difference: ComparisonSectionPayload
    impact_assessment: ImpactSectionPayload
    affected_workflows: WorkflowSectionPayload
    proposed_actions: List[ProposedActionItem] = Field(..., min_length=3, max_length=3)
    source_excerpt: SourceExcerptSectionPayload

    # Review aids (background and query mode); never replace the seven sections above.
    summary: Optional[str] = Field(default=None, description="Concise summary of the change (at most 5 lines)")
    change_category: Optional[str] = Field(default=None)
    priority_score: Optional[float] = Field(default=None)
    affected_departments: List[str] = Field(default_factory=list)
    affected_pathways: List[str] = Field(default_factory=list)


# ==============================================================================
# COMPLETENESS VALIDATION
# ==============================================================================

def validate_brief_completeness(payload: StructuredBriefPayload) -> Tuple[bool, List[str]]:
    """Validate that all required sections and provenance fields are present and complete.

    Args:
        payload: StructuredBriefPayload to validate.

    Returns:
        Tuple of (is_valid, list_of_error_strings).
    """
    errors: List[str] = []

    # 1. Recommendation quote & provenance
    if not payload.what_changed.recommendation_text or not payload.what_changed.recommendation_text.strip():
        errors.append("Section 1: Missing recommendation text quote.")
    if not payload.source_metadata.source_identifier or not payload.source_metadata.source_identifier.strip():
        errors.append("Section 1: Missing source identifier in provenance.")
    if payload.what_changed.page is None:
        errors.append("Section 1: Missing source page number.")
    if not payload.what_changed.section or not payload.what_changed.section.strip():
        errors.append("Section 1: Missing source section heading.")

    # 2. Protocol quote OR explicit no-match statement
    if payload.current_protocol.is_match:
        if not payload.current_protocol.exact_protocol_text or not payload.current_protocol.exact_protocol_text.strip():
            errors.append("Section 2: Matched protocol is missing exact protocol text quote.")
        if not payload.current_protocol.protocol_version or not payload.current_protocol.protocol_version.strip():
            errors.append("Section 2: Matched protocol is missing protocol version.")
        if not payload.current_protocol.protocol_id or not payload.current_protocol.protocol_id.strip():
            errors.append("Section 2: Matched protocol is missing protocol ID.")
    else:
        if not payload.current_protocol.no_match_statement or not payload.current_protocol.no_match_statement.strip():
            errors.append("Section 2: No-match protocol finding is missing explicit no-match statement.")

    # 3. Specific difference
    if not payload.specific_difference.specific_difference or not payload.specific_difference.specific_difference.strip():
        errors.append("Section 3: Missing specific difference statement.")
    if not payload.specific_difference.difference_type or not payload.specific_difference.difference_type.strip():
        errors.append("Section 3: Missing difference type.")

    # 4. Impact assessment
    if payload.impact_assessment.is_complete:
        if payload.impact_assessment.clinical_urgency is None:
            errors.append("Section 4: Completed impact assessment is missing clinical urgency score.")
        if not payload.impact_assessment.urgency_basis or not payload.impact_assessment.urgency_basis.strip():
            errors.append("Section 4: Completed impact assessment is missing clinical urgency written basis.")
        if payload.impact_assessment.evidence_strength is None:
            errors.append("Section 4: Completed impact assessment is missing evidence strength score.")
        if not payload.impact_assessment.evidence_basis or not payload.impact_assessment.evidence_basis.strip():
            errors.append("Section 4: Completed impact assessment is missing evidence strength written basis.")
        if payload.impact_assessment.pathway_breadth is None:
            errors.append("Section 4: Completed impact assessment is missing pathway breadth score.")
        if not payload.impact_assessment.breadth_basis or not payload.impact_assessment.breadth_basis.strip():
            errors.append("Section 4: Completed impact assessment is missing pathway breadth written basis.")
        if payload.impact_assessment.total_score is None:
            errors.append("Section 4: Completed impact assessment is missing total score.")
        if not payload.impact_assessment.tier or not payload.impact_assessment.tier.strip():
            errors.append("Section 4: Completed impact assessment is missing assigned routing tier.")
    else:
        if not payload.impact_assessment.incomplete_reason or not payload.impact_assessment.incomplete_reason.strip():
            errors.append("Section 4: Incomplete impact assessment is missing explicit incomplete reason.")
        if payload.impact_assessment.tier is not None:
            errors.append("Section 4: Incomplete impact assessment must NOT have an assigned tier.")

    # 5. Affected workflows field
    if not payload.affected_workflows.is_available:
        if not payload.affected_workflows.unavailability_reason or not payload.affected_workflows.unavailability_reason.strip():
            errors.append("Section 5: Missing explicit reason for unavailable workflow information.")
    else:
        if not payload.affected_workflows.affected_workflows and not payload.affected_workflows.workflow_summary:
            errors.append("Section 5: Affected workflows field is empty without explicit status.")

    # 6. Three proposed review actions
    if len(payload.proposed_actions) != 3:
        errors.append(f"Section 6: Expected exactly 3 proposed review actions, got {len(payload.proposed_actions)}.")
    else:
        expected_actions = ["Confirm applicability", "Update protocol if adopted", "Record rationale either way"]
        for expected in expected_actions:
            if not any(expected.lower() in a.action.lower() for a in payload.proposed_actions):
                errors.append(f"Section 6: Missing required proposed action '{expected}'.")

    # 7. Source excerpt
    if not payload.source_excerpt.source_excerpt or not payload.source_excerpt.source_excerpt.strip():
        errors.append("Section 7: Missing verbatim source excerpt.")
    if not payload.source_excerpt.source_identifier or not payload.source_excerpt.source_identifier.strip():
        errors.append("Section 7: Missing source identifier on source excerpt.")

    return len(errors) == 0, errors


def assert_brief_completeness(payload: StructuredBriefPayload) -> None:
    """Raise BriefCompletenessError if the payload fails completeness checks."""
    is_valid, errors = validate_brief_completeness(payload)
    if not is_valid:
        error_summary = "; ".join(errors)
        raise BriefCompletenessError(f"Brief completeness validation failed: {error_summary}", errors=errors)


# ==============================================================================
# DATABASE ENTITY PYDANTIC SCHEMAS
# ==============================================================================

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
