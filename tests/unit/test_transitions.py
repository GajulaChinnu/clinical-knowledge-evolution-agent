"""Unit tests for status transition validation primitives."""

import pytest

from app.schemas.transitions import (
    BRIEF_TRANSITIONS,
    CHANGE_TRANSITIONS,
    DOCUMENT_TRANSITIONS,
    InvalidStateTransitionError,
    is_valid_transition,
    validate_transition,
)


def test_valid_state_transitions():
    """Test 13: Verify valid state transitions for document, change, and brief lifecycles."""
    # Document transitions
    assert validate_transition("discovered", "parsed", DOCUMENT_TRANSITIONS, "Document") is True
    assert validate_transition("parsed", "processing", DOCUMENT_TRANSITIONS, "Document") is True
    assert validate_transition("processing", "complete", DOCUMENT_TRANSITIONS, "Document") is True
    assert validate_transition("processing", "failed", DOCUMENT_TRANSITIONS, "Document") is True
    assert validate_transition("failed", "discovered", DOCUMENT_TRANSITIONS, "Document") is True  # retry
    assert validate_transition("held", "processing", DOCUMENT_TRANSITIONS, "Document") is True

    # Change / recommendation transitions
    assert validate_transition("extracted", "comparison_pending", CHANGE_TRANSITIONS, "ChangeRecord") is True
    assert validate_transition("extracted", "held_for_G1", CHANGE_TRANSITIONS, "ChangeRecord") is True
    assert validate_transition("comparison_pending", "gap_confirmed", CHANGE_TRANSITIONS, "ChangeRecord") is True
    assert validate_transition("comparison_pending", "no_gap", CHANGE_TRANSITIONS, "ChangeRecord") is True
    assert validate_transition("comparison_pending", "held_for_G2", CHANGE_TRANSITIONS, "ChangeRecord") is True
    assert validate_transition("held_for_G1", "comparison_pending", CHANGE_TRANSITIONS, "ChangeRecord") is True

    # Brief transitions
    assert validate_transition("draft", "assigned", BRIEF_TRANSITIONS, "ChangeBrief") is True
    assert validate_transition("assigned", "in_review", BRIEF_TRANSITIONS, "ChangeBrief") is True
    assert validate_transition("in_review", "decided", BRIEF_TRANSITIONS, "ChangeBrief") is True
    assert validate_transition("in_review", "deferred", BRIEF_TRANSITIONS, "ChangeBrief") is True
    assert validate_transition("decided", "closed", BRIEF_TRANSITIONS, "ChangeBrief") is True


def test_invalid_state_transitions():
    """Test 14: Verify invalid state transitions raise InvalidStateTransitionError."""
    # Terminal document state cannot transition back
    assert is_valid_transition("complete", "discovered", DOCUMENT_TRANSITIONS) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition("complete", "discovered", DOCUMENT_TRANSITIONS, "Document")

    # Terminal recommendation cannot reopen
    assert is_valid_transition("gap_confirmed", "extracted", CHANGE_TRANSITIONS) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition("gap_confirmed", "extracted", CHANGE_TRANSITIONS, "ChangeRecord")

    assert is_valid_transition("no_gap", "gap_confirmed", CHANGE_TRANSITIONS) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition("no_gap", "gap_confirmed", CHANGE_TRANSITIONS, "ChangeRecord")

    # Closed brief cannot transition back to draft or in_review
    assert is_valid_transition("closed", "draft", BRIEF_TRANSITIONS) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition("closed", "draft", BRIEF_TRANSITIONS, "ChangeBrief")

    assert is_valid_transition("draft", "decided", BRIEF_TRANSITIONS) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition("draft", "decided", BRIEF_TRANSITIONS, "ChangeBrief")
