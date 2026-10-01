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
    BriefStatus,
    ChangeBriefBase,
    ChangeBriefCreate,
    ChangeBriefRead,
)
from app.schemas.changes import (
    ChangeRecordBase,
    ChangeRecordCreate,
    ChangeRecordRead,
    ChangeStatus,
)
from app.schemas.extraction import (
    ExtractedRecommendation,
    ExtractionResponse,
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
    ReviewAssignmentBase,
    ReviewAssignmentCreate,
    ReviewAssignmentDecisionUpdate,
    ReviewAssignmentRead,
    ReviewAssignmentStatus,
    ReviewDecision,
)
from app.schemas.impact import (
    ImpactRecordBase,
    ImpactRecordCreate,
    ImpactRecordRead,
    ImpactStatus,
    ImpactTier,
)
from app.schemas.transitions import (
    BRIEF_TRANSITIONS,
    CHANGE_TRANSITIONS,
    DOCUMENT_TRANSITIONS,
    InvalidStateTransitionError,
    is_valid_transition,
    validate_transition,
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
    "ImpactRecordBase",
    "ImpactRecordCreate",
    "ImpactRecordRead",
    # Briefs
    "BriefStatus",
    "ChangeBriefBase",
    "ChangeBriefCreate",
    "ChangeBriefRead",
    # Governance
    "ReviewDecision",
    "ReviewAssignmentStatus",
    "ReviewAssignmentBase",
    "ReviewAssignmentCreate",
    "ReviewAssignmentDecisionUpdate",
    "ReviewAssignmentRead",
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
]
