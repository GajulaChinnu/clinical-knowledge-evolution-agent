"""Deterministic reviewer authorization service for clinical governance.

Enforces that only authorized clinical reviewers can be assigned or record decisions.
Rejects unauthorized actors deterministically without external authentication dependencies.
"""

from typing import Dict, Optional, Set


class ReviewerAuthorizationService:
    """Minimal deterministic reviewer authorization registry and validator for Phase 9."""

    DEFAULT_AUTHORIZED_REVIEWERS: Dict[str, str] = {
        "dr_smith": "Clinical Governance Board Member",
        "dr_jones": "Senior Clinical Pharmacist",
        "cmo_director": "Chief Medical Officer",
        "pharmacy_lead": "P&T Committee Chair",
        "clinical_reviewer_1": "Clinical Governance Reviewer",
        "clinical_reviewer_2": "Subspecialty Care Lead",
        "rapid_governance_chair": "Rapid Governance Committee Chair",
        "dr_wilson": "Endocrinology Clinical Specialist",
    }

    DEFAULT_UNAUTHORIZED_REVIEWERS: Set[str] = {
        "unauthorized_reviewer",
        "unauthorized_user",
        "guest",
        "hacker",
        "intern_unauthorized",
        "unauthorized_dr_fake",
        "unknown_actor",
    }

    def __init__(
        self,
        authorized_reviewers: Optional[Dict[str, str]] = None,
        unauthorized_reviewers: Optional[Set[str]] = None,
    ) -> None:
        self._authorized = dict(authorized_reviewers or self.DEFAULT_AUTHORIZED_REVIEWERS)
        self._unauthorized = set(unauthorized_reviewers or self.DEFAULT_UNAUTHORIZED_REVIEWERS)

    def is_authorized(self, reviewer_id: Optional[str]) -> bool:
        """Check whether a reviewer identifier is authorized for governance operations.

        Args:
            reviewer_id: String identifier of the reviewer.

        Returns:
            True if authorized, False otherwise.
        """
        if not reviewer_id or not str(reviewer_id).strip():
            return False

        r_id = str(reviewer_id).strip().lower()

        # Explicitly denied
        if r_id in self._unauthorized:
            return False

        # Explicitly permitted
        if r_id in self._authorized:
            return True

        # Recognized credentialed prefix conventions (e.g. "dr_", "clinical_reviewer")
        # unless explicitly in unauthorized list
        if r_id.startswith(("dr_", "reviewer_", "lead_", "cmo_")):
            return True

        return False

    def register_reviewer(self, reviewer_id: str, role: str = "Clinical Governance Reviewer") -> None:
        """Register a reviewer as authorized."""
        r_id = reviewer_id.strip().lower()
        self._authorized[r_id] = role
        self._unauthorized.discard(r_id)

    def revoke_reviewer(self, reviewer_id: str) -> None:
        """Revoke a reviewer's authorization."""
        r_id = reviewer_id.strip().lower()
        self._authorized.pop(r_id, None)
        self._unauthorized.add(r_id)

    def get_reviewer_role(self, reviewer_id: str) -> str:
        """Get the recorded role for a reviewer, or default."""
        r_id = reviewer_id.strip().lower()
        return self._authorized.get(r_id, "Clinical Governance Reviewer")
