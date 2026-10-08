"""Tests for the deny-by-default reviewer registry and governance identity resolution."""

from pathlib import Path

import pytest

from app.services.config_service import AppConfig
from app.services.reviewer_authorization import (
    Reviewer,
    ReviewerAuthorizationService,
    ReviewerRegistryError,
    load_reviewer_registry,
)
from app.ui.streamlit_app import resolve_acting_reviewer

REGISTRY = """
reviewers:
  - id: Dr_Adams
    name: Dr. Adams
    role: Governance Chair
    email: Adams@Hospital.example
  - id: pharm_lead
    name: Pharmacy Lead
    role: P&T Chair
"""


@pytest.fixture
def registry_file(tmp_path: Path) -> Path:
    path = tmp_path / "reviewers.yaml"
    path.write_text(REGISTRY, encoding="utf-8")
    return path


def test_prefix_convention_no_longer_grants_authority(registry_file):
    auth = ReviewerAuthorizationService(registry_path=registry_file)
    for impostor in ("dr_anyone", "reviewer_x", "lead_y", "cmo_z", "DR_ANYONE"):
        assert auth.is_authorized(impostor) is False


def test_only_registered_reviewers_are_authorized(registry_file):
    auth = ReviewerAuthorizationService(registry_path=registry_file)
    assert auth.is_authorized("dr_adams")
    assert auth.is_authorized("  PHARM_LEAD ")
    assert not auth.is_authorized("")
    assert not auth.is_authorized(None)
    assert auth.get_reviewer_role("dr_adams") == "Governance Chair"
    with pytest.raises(KeyError):
        auth.get_reviewer_role("dr_unknown")


def test_revocation_and_registration(registry_file):
    auth = ReviewerAuthorizationService(registry_path=registry_file)
    auth.revoke_reviewer("dr_adams")
    assert not auth.is_authorized("dr_adams")
    assert auth.reviewer_for_email("adams@hospital.example") is None
    auth.register_reviewer("new_reviewer", "Locum Reviewer")
    assert auth.is_authorized("new_reviewer")


def test_email_lookup_is_case_insensitive(registry_file):
    auth = ReviewerAuthorizationService(registry_path=registry_file)
    assert auth.reviewer_for_email("ADAMS@hospital.example").reviewer_id == "dr_adams"
    assert auth.reviewer_for_email("someone@else.example") is None


@pytest.mark.parametrize(
    "content, match",
    [
        ("reviewers:\n  - name: No Id\n    role: X\n", "id and role are required"),
        ("reviewers:\n  - id: a\n    role: X\n  - id: A\n    role: Y\n", "Duplicate"),
        ("reviewers: [", "not valid YAML"),
    ],
)
def test_malformed_registry_rejected(tmp_path, content, match):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReviewerRegistryError, match=match):
        load_reviewer_registry(path)


def test_missing_registry_rejected(tmp_path):
    with pytest.raises(ReviewerRegistryError):
        ReviewerAuthorizationService(registry_path=tmp_path / "missing.yaml")


def test_project_registry_loads():
    assert ReviewerAuthorizationService().is_authorized("dr_smith")


# ------------------------------------------------------------------------------
# Acting-reviewer resolution in the governance workspace
# ------------------------------------------------------------------------------

class FakeUser(dict):
    def __init__(self, logged_in, email=None):
        super().__init__({"email": email} if email else {})
        self.is_logged_in = logged_in


def _auth():
    return ReviewerAuthorizationService(reviewers=[Reviewer("dr_adams", "Dr. Adams", "Chair", "adams@h.example")])


def test_demo_mode_is_explicitly_flagged_when_sso_disabled():
    reviewer, mode, message = resolve_acting_reviewer(_auth(), AppConfig(sso_enabled=False), FakeUser(False))
    assert reviewer is None and mode == "demo"
    assert "unauthenticated" in message


def test_sso_requires_sign_in():
    reviewer, mode, _ = resolve_acting_reviewer(_auth(), AppConfig(sso_enabled=True), FakeUser(False))
    assert reviewer is None and mode == "signin_required"


def test_sso_user_mapped_to_registered_reviewer():
    reviewer, mode, message = resolve_acting_reviewer(
        _auth(), AppConfig(sso_enabled=True), FakeUser(True, "Adams@H.example")
    )
    assert mode == "sso" and message is None
    assert reviewer.reviewer_id == "dr_adams"


def test_sso_user_not_in_registry_is_blocked():
    reviewer, mode, message = resolve_acting_reviewer(
        _auth(), AppConfig(sso_enabled=True), FakeUser(True, "intruder@h.example")
    )
    assert reviewer is None and mode == "sso"
    assert "not in the reviewer registry" in message
