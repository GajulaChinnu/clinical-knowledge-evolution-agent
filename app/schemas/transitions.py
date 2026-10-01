"""CKEA State Transition Validation Primitives.

Defines legal status transitions and validation primitives for:
1. Document lifecycle
2. Recommendation/Change record lifecycle
3. Change Brief lifecycle
"""

from typing import Dict, Set


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid entity lifecycle status transition is attempted."""

    def __init__(self, current_status: str, new_status: str, entity_name: str = "Entity"):
        self.current_status = current_status
        self.new_status = new_status
        self.entity_name = entity_name
        super().__init__(
            f"Illegal {entity_name} status transition from '{current_status}' to '{new_status}'."
        )


# 1. Document Lifecycle Transitions
DOCUMENT_TRANSITIONS: Dict[str, Set[str]] = {
    "discovered": {"parsed", "failed", "held"},
    "parsed": {"processing", "failed", "held"},
    "processing": {"complete", "failed", "held"},
    "held": {"discovered", "parsed", "processing", "failed"},
    "failed": {"discovered", "processing"},
    "complete": set(),  # Terminal state
}

# 2. Recommendation / Change Record Lifecycle Transitions
CHANGE_TRANSITIONS: Dict[str, Set[str]] = {
    "extracted": {"comparison_pending", "held_for_G1", "no_gap"},
    "held_for_G1": {"comparison_pending", "no_gap"},
    "comparison_pending": {"gap_confirmed", "no_gap", "held_for_G2", "held_for_G3"},
    "held_for_G2": {"gap_confirmed", "no_gap"},
    "held_for_G3": {"gap_confirmed", "no_gap"},
    "gap_confirmed": set(),  # Handoff to impact scoring
    "no_gap": set(),  # Terminal state (no action needed)
}

# 3. Change Brief Lifecycle Transitions
BRIEF_TRANSITIONS: Dict[str, Set[str]] = {
    "draft": {"assigned", "deferred", "closed"},
    "assigned": {"in_review", "deferred", "closed"},
    "in_review": {"decided", "deferred", "closed"},
    "deferred": {"assigned", "in_review", "closed"},
    "decided": {"closed"},
    "closed": set(),  # Terminal state
}


def is_valid_transition(
    current_status: str,
    new_status: str,
    transition_table: Dict[str, Set[str]],
) -> bool:
    """Check whether a status transition is permitted by the given transition table.

    Args:
        current_status: Current status string.
        new_status: Target status string.
        transition_table: Mapping of status to allowed target statuses.

    Returns:
        True if the transition is legal, False otherwise.
    """
    if current_status == new_status:
        return True  # Idempotent no-op
    allowed = transition_table.get(current_status, set())
    return new_status in allowed


def validate_transition(
    current_status: str,
    new_status: str,
    transition_table: Dict[str, Set[str]],
    entity_name: str = "Entity",
) -> bool:
    """Validate a status transition or raise InvalidStateTransitionError.

    Args:
        current_status: Current status string.
        new_status: Target status string.
        transition_table: Mapping of status to allowed target statuses.
        entity_name: Human-readable entity name for error messages.

    Returns:
        True if transition is valid.

    Raises:
        InvalidStateTransitionError: If the transition is illegal.
    """
    if not is_valid_transition(current_status, new_status, transition_table):
        raise InvalidStateTransitionError(current_status, new_status, entity_name)
    return True
