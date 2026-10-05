"""CKEA versioned Pydantic schemas package."""

from app.schemas.audit import (
    AuditLogBase,
    AuditLogCreate,
    AuditLogRead,
    NotificationBase,
    NotificationCreate,
    NotificationRead,
)
from app.schemas.briefs import (
    BriefCompletenessError,
    BriefStatus,
    ChangeBriefBase,
    ChangeBriefCreate,
    ChangeBriefRead,
    StructuredBriefPayload,
    assert_brief_completeness,
    validate_brief_completeness,
)
from app.schemas.changes import (
    ChangeRecordBase,
    ChangeRecordCreate,
    ChangeRecordRead,
    ChangeStatus,
)
from app.schemas.comparison import (
    ComparisonResponse,
)
from app.schemas.extraction import (
    ExtractedRecommendation,
    ExtractionResponse,
)
from app.schemas.protocol import (
    CandidateProtocolSection,
    ProtocolDocument,
    ProtocolIndexingSummary,
    ProtocolSection,
)
from app.schemas.documents import (
    DocumentStatus,
    IngestedDocumentBase,
    IngestedDocumentCreate,
    IngestedDocumentRead,
    IngestionFailureBase,
    IngestionFailureCreate,
    IngestionFailureRead,
)
from app.schemas.gaps import (
    ComparisonResult,
    DifferenceType,
    GapRecordBase,
    GapRecordCreate,
    GapRecordRead,
    GapStatus,
)
from app.schemas.governance import (
    DeferDecisionRequest,
    GovernanceDecisionRequest,
    GovernanceGateError,
    ReassignmentRequest,
    ReviewAssignmentBase,
    ReviewAssignmentCreate,
    ReviewAssignmentDecisionUpdate,
    ReviewAssignmentRead,
    ReviewAssignmentStatus,
    ReviewDecision,
    ReviewerAssignmentRequest,
    SLAEscalationResult,
    UnauthorizedReviewerError,
)
from app.schemas.scheduler import SLACycleSummary
from app.schemas.impact import (
    DimensionScore,
    ImpactRecordBase,
    ImpactRecordCreate,
    ImpactRecordRead,
    ImpactStatus,
    ImpactTier,
    ScoringResult,
)
from app.schemas.transitions import (
    BRIEF_TRANSITIONS,
    CHANGE_TRANSITIONS,
    DOCUMENT_TRANSITIONS,
    InvalidStateTransitionError,
    is_valid_transition,
    validate_transition,
)
from app.schemas.orchestration import (
    ArtifactsManifest,
    BatchPipelineResult,
    G1ResolutionRequest,
    G2ResolutionRequest,
    G3ResolutionRequest,
    HumanGate,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)

__all__ = [
    # Documents
    "DocumentStatus",
    "IngestedDocumentBase",
    "IngestedDocumentCreate",
    "IngestedDocumentRead",
    "IngestionFailureBase",
    "IngestionFailureCreate",
    "IngestionFailureRead",
    # Changes
    "ChangeStatus",
    "ChangeRecordBase",
    "ChangeRecordCreate",
    "ChangeRecordRead",
    # Extraction
    "ExtractedRecommendation",
    "ExtractionResponse",
    # Protocols
    "ProtocolSection",
    "ProtocolDocument",
    "CandidateProtocolSection",
    "ProtocolIndexingSummary",
    # Comparison
    "ComparisonResponse",
    # Gaps
    "GapStatus",
    "ComparisonResult",
    "DifferenceType",
    "GapRecordBase",
    "GapRecordCreate",
    "GapRecordRead",
    # Impact
    "ImpactTier",
    "ImpactStatus",
    "DimensionScore",
    "ScoringResult",
    "ImpactRecordBase",
    "ImpactRecordCreate",
    "ImpactRecordRead",
    # Briefs
    "BriefStatus",
    "ChangeBriefBase",
    "ChangeBriefCreate",
    "ChangeBriefRead",
    "BriefCompletenessError",
    "StructuredBriefPayload",
    "validate_brief_completeness",
    "assert_brief_completeness",
    # Governance
    "GovernanceGateError",
    "UnauthorizedReviewerError",
    "ReviewDecision",
    "ReviewAssignmentStatus",
    "ReviewAssignmentBase",
    "ReviewAssignmentCreate",
    "ReviewAssignmentDecisionUpdate",
    "ReviewAssignmentRead",
    "ReviewerAssignmentRequest",
    "ReassignmentRequest",
    "GovernanceDecisionRequest",
    "DeferDecisionRequest",
    "SLAEscalationResult",
    # Audit & Notification
    "AuditLogBase",
    "AuditLogCreate",
    "AuditLogRead",
    "NotificationBase",
    "NotificationCreate",
    "NotificationRead",
    # Transitions
    "DOCUMENT_TRANSITIONS",
    "CHANGE_TRANSITIONS",
    "BRIEF_TRANSITIONS",
    "InvalidStateTransitionError",
    "is_valid_transition",
    "validate_transition",
    # Orchestration
    "PipelineStage",
    "PipelineStatus",
    "HumanGate",
    "ArtifactsManifest",
    "PipelineResult",
    "BatchPipelineResult",
    "G1ResolutionRequest",
    "G2ResolutionRequest",
    "G3ResolutionRequest",
    # Scheduler (Phase 11)
    "SLACycleSummary",
]
