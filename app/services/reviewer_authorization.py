"""Reviewer registry and authorization for clinical governance.

Authorization is deny-by-default: only reviewers listed in the registry
(config/reviewers.yaml) may be assigned, review or record decisions. There is no
identifier-prefix convention; typing "dr_anything" grants nothing.

Identity: when SSO is configured, the UI resolves the acting reviewer from the signed-in
user's email via `reviewer_for_email`. Without SSO the UI runs in an explicitly flagged
demo mode (see app.ui.streamlit_app.resolve_acting_reviewer).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

import yaml

logger = logging.getLogger("ckea.services.reviewer_authorization")

DEFAULT_REGISTRY_PATH = Path("./config/reviewers.yaml")


class ReviewerRegistryError(ValueError):
    """Raised when the reviewer registry file is missing or malformed."""


@dataclass(frozen=True)
class Reviewer:
    """A named, authorized clinical governance reviewer."""
    reviewer_id: str
    name: str
    role: str
    email: Optional[str] = None


def load_reviewer_registry(path: Union[str, Path]) -> List[Reviewer]:
    """Load and validate the reviewer registry YAML."""
    path = Path(path)
    if not path.exists():
        raise ReviewerRegistryError(f"Reviewer registry not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ReviewerRegistryError(f"Reviewer registry '{path}' is not valid YAML: {e}") from e

    reviewers: List[Reviewer] = []
    seen: Set[str] = set()
    for entry in data.get("reviewers") or []:
        if not isinstance(entry, dict) or not str(entry.get("id") or "").strip() or not entry.get("role"):
            raise ReviewerRegistryError(f"Invalid reviewer entry in {path}: {entry!r} (id and role are required)")
        r_id = str(entry["id"]).strip().lower()
        if r_id in seen:
            raise ReviewerRegistryError(f"Duplicate reviewer id '{r_id}' in {path}")
        seen.add(r_id)
        email = str(entry["email"]).strip().lower() if entry.get("email") else None
        reviewers.append(Reviewer(r_id, str(entry.get("name") or r_id), str(entry["role"]), email))
    return reviewers


class ReviewerAuthorizationService:
    """Deny-by-default reviewer registry and validator."""

    def __init__(
        self,
        authorized_reviewers: Optional[Dict[str, str]] = None,
        unauthorized_reviewers: Optional[Set[str]] = None,
        registry_path: Optional[Union[str, Path]] = None,
        reviewers: Optional[Iterable[Reviewer]] = None,
    ) -> None:
        """Build the registry.

        Precedence: explicit `reviewers`, then `authorized_reviewers` ({id: role}),
        then the registry file (`registry_path` or config/reviewers.yaml).
        `unauthorized_reviewers` is an optional explicit deny list on top of deny-by-default.
        """
        if reviewers is not None:
            entries = list(reviewers)
        elif authorized_reviewers is not None:
            entries = [Reviewer(k.strip().lower(), k, v) for k, v in authorized_reviewers.items()]
        else:
            entries = load_reviewer_registry(registry_path or DEFAULT_REGISTRY_PATH)

        self._reviewers: Dict[str, Reviewer] = {r.reviewer_id: r for r in entries}
        self._denied: Set[str] = {r.strip().lower() for r in (unauthorized_reviewers or set())}

    @staticmethod
    def _norm(reviewer_id: Optional[str]) -> str:
        return str(reviewer_id or "").strip().lower()

    def is_authorized(self, reviewer_id: Optional[str]) -> bool:
        """True only for a non-denied reviewer listed in the registry."""
        r_id = self._norm(reviewer_id)
        return bool(r_id) and r_id not in self._denied and r_id in self._reviewers

    def get_reviewer(self, reviewer_id: Optional[str]) -> Optional[Reviewer]:
        r_id = self._norm(reviewer_id)
        return self._reviewers.get(r_id) if self.is_authorized(r_id) else None

    def reviewer_for_email(self, email: Optional[str]) -> Optional[Reviewer]:
        """Map an authenticated (SSO) email to its registered reviewer, if any."""
        target = self._norm(email)
        if not target:
            return None
        for reviewer in self._reviewers.values():
            if reviewer.email == target and self.is_authorized(reviewer.reviewer_id):
                return reviewer
        return None

    def list_reviewers(self) -> List[Reviewer]:
        return [r for r in self._reviewers.values() if self.is_authorized(r.reviewer_id)]

    def register_reviewer(self, reviewer_id: str, role: str = "Clinical Governance Reviewer") -> None:
        """Register a reviewer as authorized."""
        r_id = self._norm(reviewer_id)
        self._reviewers[r_id] = Reviewer(r_id, reviewer_id, role)
        self._denied.discard(r_id)

    def revoke_reviewer(self, reviewer_id: str) -> None:
        """Revoke a reviewer's authorization."""
        r_id = self._norm(reviewer_id)
        self._reviewers.pop(r_id, None)
        self._denied.add(r_id)

    def get_reviewer_role(self, reviewer_id: str) -> str:
        """Recorded role for an authorized reviewer."""
        reviewer = self.get_reviewer(reviewer_id)
        if reviewer is None:
            raise KeyError(f"Reviewer '{reviewer_id}' is not in the reviewer registry.")
        return reviewer.role
