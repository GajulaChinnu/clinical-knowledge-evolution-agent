"""Unit tests for Pydantic schema validation rules."""

from datetime import datetime, timezone
import uuid
from pydantic import ValidationError
import pytest

from app.schemas.briefs import BriefStatus, ChangeBriefCreate
from app.schemas.changes import ChangeRecordCreate, ChangeStatus
from app.schemas.documents import DocumentStatus, IngestedDocumentCreate
from app.schemas.gaps import ComparisonResult, DifferenceType, GapRecordCreate, GapStatus
from app.schemas.governance import (
    ReviewAssignmentCreate,
    ReviewAssignmentDecisionUpdate,
    ReviewAssignmentStatus,
    ReviewDecision,
)
from app.schemas.impact import (
    ImpactRecordCreate,
    ImpactStatus,
    ImpactTier,
)


def test_required_field_validation():
    """Test 5: Verify missing required fields raise ValidationError."""
    # IngestedDocumentCreate missing sha256_hash, source_identifier, source_path
    with pytest.raises(ValidationError) as exc:
        IngestedDocumentCreate()
    assert "source_identifier" in str(exc.value)
    assert "source_path" in str(exc.value)
    assert "sha256_hash" in str(exc.value)

    # ChangeRecordCreate missing required fields
    with pytest.raises(ValidationError):
        ChangeRecordCreate(
            ingested_document_id=str(uuid.uuid4()),
            verbatim_text="Sample text",
            # missing target_population, intervention, confidence, etc.
        )


def test_status_enum_validation():
    """Test 6: Verify status enums reject invalid status strings."""
    valid_uuid = str(uuid.uuid4())

    # Invalid document status
    with pytest.raises(ValidationError):
        IngestedDocumentCreate(
            source_identifier="DOC-01",
            source_path="data/test.pdf",
            sha256_hash="d" * 64,
            status="invalid_status",
        )

    # Invalid brief status
    with pytest.raises(ValidationError):
        ChangeBriefCreate(
            impact_record_id=valid_uuid,
            status="unknown_brief_state",
        )


def test_score_range_and_consistency_validation():
    """Test 7: Verify score range (1-5) and total score consistency."""
    gap_id = str(uuid.uuid4())

    # Score below 1 rejected
    with pytest.raises(ValidationError):
        ImpactRecordCreate(
            gap_record_id=gap_id,
            clinical_urgency=0,
            evidence_strength=3,
            pathway_breadth=3,
        )

    # Score above 5 rejected
    with pytest.raises(ValidationError):
        ImpactRecordCreate(
            gap_record_id=gap_id,
            clinical_urgency=6,
            evidence_strength=3,
            pathway_breadth=3,
        )

    # Total score inconsistent with sum of dimensions
    with pytest.raises(ValidationError) as exc:
        ImpactRecordCreate(
            gap_record_id=gap_id,
            clinical_urgency=3,
            evidence_strength=4,
            pathway_breadth=2,
            total_score=15,  # 3+4+2 = 9 != 15
        )
    assert "total_score" in str(exc.value)

    # Valid scores and consistent total score passes
    valid_impact = ImpactRecordCreate(
        gap_record_id=gap_id,
        clinical_urgency=4,
        evidence_strength=5,
        pathway_breadth=3,
        total_score=12,
        tier=ImpactTier.CRITICAL,
    )
    assert valid_impact.total_score == 12
    assert valid_impact.tier == ImpactTier.CRITICAL


def test_decision_validation():
    """Test 8: Verify valid and invalid decision values."""
    brief_id = str(uuid.uuid4())

    # Valid decisions
    for dec in [ReviewDecision.APPROVE, ReviewDecision.REJECT]:
        update = ReviewAssignmentDecisionUpdate(
            decision=dec,
            rationale="Clinical evidence thoroughly reviewed and accepted.",
        )
        assert update.decision == dec

    # Invalid decision string
    with pytest.raises(ValidationError):
        ReviewAssignmentDecisionUpdate(
            decision="auto_approve",  # Invalid
            rationale="Some reason",
        )


def test_non_empty_rationale_validation():
    """Test 9: Verify non-empty rationale is enforced when recording a decision."""
    brief_id = str(uuid.uuid4())

    # Missing/empty rationale with decision raises error
    with pytest.raises(ValidationError) as exc:
        ReviewAssignmentCreate(
            change_brief_id=brief_id,
            reviewer_role="Clinical Specialist",
            reviewer_id="REV-100",
            due_date=datetime.now(timezone.utc),
            decision=ReviewDecision.APPROVE,
            rationale="   ",  # Blank/whitespace
        )
    assert "rationale" in str(exc.value)

    with pytest.raises(ValidationError):
        ReviewAssignmentDecisionUpdate(
            decision=ReviewDecision.REJECT,
            rationale="",  # Empty
        )


def test_defer_follow_up_date_validation():
    """Test 10: Verify defer decision requires a follow-up due date."""
    brief_id = str(uuid.uuid4())

    # Defer without follow-up date raises error
    with pytest.raises(ValidationError) as exc:
        ReviewAssignmentCreate(
            change_brief_id=brief_id,
            reviewer_role="Chief Medical Officer",
            reviewer_id="REV-CMO",
            due_date=datetime.now(timezone.utc),
            decision=ReviewDecision.DEFER,
            rationale="Awaiting upcoming trial publication next quarter.",
            defer_follow_up_date=None,  # Missing
        )
    assert "defer_follow_up_date" in str(exc.value)

    # Defer with follow-up date passes
    valid_defer = ReviewAssignmentCreate(
        change_brief_id=brief_id,
        reviewer_role="Chief Medical Officer",
        reviewer_id="REV-CMO",
        due_date=datetime.now(timezone.utc),
        decision=ReviewDecision.DEFER,
        rationale="Awaiting upcoming trial publication next quarter.",
        defer_follow_up_date=datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert valid_defer.decision == ReviewDecision.DEFER
    assert valid_defer.defer_follow_up_date is not None
