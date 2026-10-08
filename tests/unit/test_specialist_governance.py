"""Phase 6: specialist routing and protocol-update governance from clinician queries."""

from datetime import datetime, timedelta, timezone

import pytest

from app.agents.briefing_agent import BriefingAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.impact_agent import ImpactAgent
from app.models.entities import AuditLog, ChangeBrief, Notification, ReviewAssignment
from app.orchestration.clinician_query import ClinicianQueryWorkflow
from app.schemas.clinician_query import ClinicianQueryInput, PatientContext
from app.services.reviewer_authorization import ReviewerAuthorizationService, ReviewerRegistryError, load_reviewer_registry
from tests.corpus_env import PROTOCOLS, TAXONOMY, WATCHLIST, build_corpus_env


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    corpus = build_corpus_env(tmp_path_factory.mktemp("gov"))
    corpus.governance = GovernanceAgent(session_factory=corpus.session_factory, config=corpus.config)
    corpus.workflow = ClinicianQueryWorkflow(
        session_factory=corpus.session_factory, config=corpus.config,
        monitoring_agent=corpus.monitoring, extraction_agent=corpus.extraction, comparison_agent=corpus.comparison,
        impact_agent=ImpactAgent(session_factory=corpus.session_factory, config=corpus.config),
        briefing_agent=BriefingAgent(session_factory=corpus.session_factory, config=corpus.config),
        governance_agent=corpus.governance, watchlist=WATCHLIST, taxonomy=TAXONOMY, protocols=PROTOCOLS,
    )
    yield corpus
    corpus.engine.dispose()


def ask(env, department, treatment, **context):
    return env.workflow.run(ClinicianQueryInput(department=department, treatment=treatment, context=PatientContext(**context)))


def _assignments(env, brief_id):
    with env.session_factory() as session:
        return session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).all()


# ------------------------------------------------------------------------------ registry
def test_registry_roles_and_specialties():
    auth = ReviewerAuthorizationService()
    for dept in ["endocrinology", "cardiology", "nephrology", "infectious-diseases", "emergency-medicine"]:
        assert auth.specialists_for(dept), dept
    assert auth.chairs() and auth.governance_pool()


@pytest.mark.parametrize("entry, match", [
    ("  - {id: x, role: R, governance_role: wizard}\n", "governance_role"),
    ("  - {id: x, role: R, governance_role: specialist}\n", "at least one specialty"),
    ("  - {id: x, role: R, governance_role: specialist, specialties: [astrology]}\n", "Unknown department"),
])
def test_invalid_registry_entries(tmp_path, entry, match):
    path = tmp_path / "r.yaml"
    path.write_text("reviewers:\n" + entry, encoding="utf-8")
    with pytest.raises(ReviewerRegistryError, match=match):
        load_reviewer_registry(path)


# ------------------------------------------------------------------------------ query-created briefs
def test_out_of_date_protocol_creates_brief_routed_to_department_specialist(env):
    answer = ask(env, "Endocrinology", "metformin 500 mg once daily", egfr_band=">=60")
    status = next(g for g in answer.governance if g.protocol_id == "PROT-DM-001")
    assert status.brief_id and status.status == "assigned"
    assert "pending specialist review" in status.message
    assignments = _assignments(env, status.brief_id)
    assert [a.reviewer_id for a in assignments] == ["dr_wilson"]
    with env.session_factory() as session:
        brief = session.get(ChangeBrief, status.brief_id)
        payload = brief.structured_payload
        assert payload["current_protocol"]["protocol_id"] == "PROT-DM-001"
        assert payload["current_protocol"]["exact_protocol_text"].startswith("Metformin 500mg once daily")
        assert payload["what_changed"]["recommendation_text"].startswith("Start metformin 500 mg twice daily")
        assert payload["change_category"] == "dose_change"
        assert payload["affected_departments"] == ["endocrinology"]
        assert 1 <= len(payload["summary"].splitlines()) <= 5
        assert brief.status == "assigned"  # routing never decides
        routed = session.query(AuditLog).filter_by(entity_id=brief.id, new_status="routed").one()
        assert routed.audit_metadata["reviewer_id"] == "dr_wilson"


def test_repeat_query_reuses_the_existing_brief(env):
    first = ask(env, "Endocrinology", "metformin 500 mg once daily")
    second = ask(env, "Endocrinology", "metformin 500 mg once daily")
    assert first.governance[0].brief_id == second.governance[0].brief_id
    with env.session_factory() as session:
        briefs = [b for b in session.query(ChangeBrief)
                  if (b.structured_payload or {}).get("current_protocol", {}).get("protocol_id") == "PROT-DM-001"]
        assert len(briefs) == 1


@pytest.mark.parametrize("department, treatment, protocol, specialist", [
    ("Nephrology", "ramipril 2.5 mg once daily", "PROT-NEPH-001", "clinical_reviewer_2"),
    ("Emergency Medicine", "intramuscular adrenaline 0.5 mg", "PROT-EM-001", "emergency_specialist"),
])
def test_each_department_routes_to_its_specialist(env, department, treatment, protocol, specialist):
    answer = ask(env, department, treatment)
    status = next(g for g in answer.governance if g.protocol_id == protocol)
    assert [a.reviewer_id for a in _assignments(env, status.brief_id)] == [specialist]


def test_clinician_sees_human_decision_status(env):
    answer = ask(env, "Endocrinology", "metformin 500 mg once daily")
    brief_id = answer.governance[0].brief_id
    env.governance.start_review(change_brief_id=brief_id, reviewer_id="dr_wilson")
    env.governance.decide(change_brief_id=brief_id, reviewer_id="dr_wilson", decision="defer",
                          rationale="Awaiting renal pharmacist input.",
                          defer_follow_up_date=datetime.now(timezone.utc) + timedelta(days=14))
    deferred = ask(env, "Endocrinology", "metformin 500 mg once daily")
    assert "deferred until" in deferred.governance[0].message


def test_aligned_protocol_creates_no_brief(env):
    answer = ask(env, "Cardiology", "apixaban 5 mg twice daily")
    assert answer.governance == []


# ------------------------------------------------------------------------------ routing rules
def _make_brief(env, department, treatment):
    answer = ask(env, department, treatment)
    return answer.governance[0].brief_id


def test_unroutable_brief_stays_unassigned_and_is_flagged(env):
    # A registry with no nephrology specialist: routing must not fall back to someone else.
    auth = ReviewerAuthorizationService()
    auth.revoke_reviewer("clinical_reviewer_2")
    governance = GovernanceAgent(session_factory=env.session_factory, config=env.config, auth_service=auth)
    with env.session_factory() as session:
        brief = next(b for b in session.query(ChangeBrief)
                     if (b.structured_payload or {}).get("current_protocol", {}).get("protocol_id") == "PROT-NEPH-001")
        for a in session.query(ReviewAssignment).filter_by(change_brief_id=brief.id):
            session.delete(a)
        brief.status = "draft"
        impact = brief.impact_record
        impact.tier = "High"
        session.commit()
        brief_id = brief.id
    result = governance.route_brief(brief_id)
    assert result["status"] == "unroutable" and result["reviewer_id"] is None
    assert _assignments(env, brief_id) == []
    with env.session_factory() as session:
        assert session.query(Notification).filter_by(related_entity_id=brief_id, notification_type="unroutable_brief").count() == 1
        assert session.query(AuditLog).filter_by(entity_id=brief_id, new_status="unroutable").count() == 1
        assert session.get(ChangeBrief, brief_id).status == "draft"


def test_standard_tier_goes_to_governance_pool(env):
    governance = env.governance
    with env.session_factory() as session:
        brief = next(b for b in session.query(ChangeBrief)
                     if (b.structured_payload or {}).get("current_protocol", {}).get("protocol_id") == "PROT-EM-001")
        for a in session.query(ReviewAssignment).filter_by(change_brief_id=brief.id):
            session.delete(a)
        brief.status = "draft"
        brief.impact_record.tier = "Standard"
        session.commit()
        brief_id = brief.id
    result = governance.route_brief(brief_id)
    assert result["reviewer_id"] == ReviewerAuthorizationService().governance_pool()[0].reviewer_id
