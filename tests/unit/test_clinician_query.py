"""Phase 5: the six-agent clinician Treatment Check workflow and deterministic answer brief."""

import json
from unittest.mock import MagicMock

import pytest

from app.agents.briefing_agent import BriefingAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.impact_agent import ImpactAgent
from app.models.entities import AuditLog, ClinicianQueryRecord, Notification
from app.orchestration.clinician_query import ClinicianQueryWorkflow
from app.schemas.clinician_query import ClinicianQueryInput, PatientContext, PatientIdentifierError
from app.schemas.treatment_check import DISCLAIMER
from tests.corpus_env import PROTOCOLS, TAXONOMY, WATCHLIST, build_corpus_env


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    corpus = build_corpus_env(tmp_path_factory.mktemp("query"))
    workflow = ClinicianQueryWorkflow(
        session_factory=corpus.session_factory, config=corpus.config,
        monitoring_agent=corpus.monitoring, extraction_agent=corpus.extraction,
        comparison_agent=corpus.comparison,
        impact_agent=ImpactAgent(session_factory=corpus.session_factory, config=corpus.config),
        briefing_agent=BriefingAgent(session_factory=corpus.session_factory, config=corpus.config),
        governance_agent=GovernanceAgent(session_factory=corpus.session_factory, config=corpus.config),
        watchlist=WATCHLIST, taxonomy=TAXONOMY, protocols=PROTOCOLS,
    )
    corpus.workflow = workflow
    yield corpus
    corpus.engine.dispose()


def ask(env, department, treatment, condition=None, **context):
    return env.workflow.run(ClinicianQueryInput(department=department, treatment=treatment, condition=condition,
                                                context=PatientContext(**context)))


def test_guidance_updated_answer_shows_old_and_new_version(env):
    answer = ask(env, "Endocrinology", "metformin 500 mg once daily", egfr_band=">=60")
    assert answer.verdict == "guidance_updated_follow_new_version"
    assert answer.latest_guidance[0].excerpt.startswith("Start metformin 500 mg twice daily")
    assert answer.latest_guidance[0].version == "2.0" and answer.latest_guidance[0].published_date == "2026-02-01"
    change = next(v for v in answer.what_changed if v.plan_matches_previous)
    assert change.previous.version == "1.0" and change.previous.published_date == "2025-01-15"
    assert change.previous.excerpt.startswith("Start metformin 500 mg once daily")
    assert "Follow the latest version" in answer.recommended_actions[0]
    assert any(p.protocol_id == "PROT-DM-001" and p.status == "out_of_date" for p in answer.protocol_positions)


def test_consistent_answer(env):
    answer = ask(env, "Endocrinology", "metformin 500 mg twice daily", egfr_band=">=60")
    assert answer.verdict == "consistent_with_latest_guidance"
    assert answer.check_before_prescribing == []  # both contraindications ruled out by the eGFR band


def test_consistent_answer_lists_checks_when_context_missing(env):
    answer = ask(env, "Endocrinology", "metformin 500 mg twice daily")
    assert answer.verdict == "consistent_with_latest_guidance"
    assert answer.check_before_prescribing  # eGFR-dependent contraindications to confirm


@pytest.mark.parametrize("department, treatment, condition, context, expected", [
    ("Cardiology", "apixaban 5 mg twice daily", None, {"comorbidities": ["mechanical heart valve"]}, "conflicts_with_latest_guidance"),
    ("Infectious Diseases", "levofloxacin 500 mg once daily", "low-severity community-acquired pneumonia", {}, "conflicts_with_latest_guidance"),
    ("Nephrology", "dabigatran 110 mg twice daily", None, {"egfr_band": "15-29"}, "conflicts_with_latest_guidance"),
    ("Endocrinology", "metformin 3000 mg daily", None, {}, "conflicts_with_latest_guidance"),
    ("Endocrinology", "gliclazide 40 mg once daily", None, {}, "guidance_updated_follow_new_version"),
    ("Emergency Medicine", "intramuscular adrenaline 0.5 mg, repeat after 10 minutes", None, {}, "guidance_updated_follow_new_version"),
    ("Cardiology", "unicorn extract 5 mg", None, {}, "insufficient_grounded_evidence"),
])
def test_verdicts(env, department, treatment, condition, context, expected):
    answer = ask(env, department, treatment, condition, **context)
    assert answer.verdict == expected, answer.verdict_basis


def test_safety_notice_conflict_is_cited_and_ranked_first(env):
    answer = ask(env, "Infectious Diseases", "levofloxacin 500 mg once daily", "low-severity community-acquired pneumonia")
    assert answer.safety_notices and answer.safety_notices[0].citation.source_type == "safety_notice"
    assert answer.findings[0].kind == "contraindication" and answer.findings[0].relation == "applies"


def test_insufficient_answer_has_no_citations(env):
    answer = ask(env, "Cardiology", "unicorn extract 5 mg")
    assert answer.citations == [] and answer.latest_guidance == []


def test_every_verdict_is_grounded_in_verified_citations(env):
    for dept, treatment in [("Endocrinology", "metformin 500 mg once daily"), ("Cardiology", "apixaban 5 mg twice daily"),
                            ("Nephrology", "ramipril 2.5 mg once daily"), ("Infectious Diseases", "amoxicillin 500 mg three times daily for 5 days")]:
        answer = ask(env, dept, treatment)
        assert answer.verdict != "insufficient_grounded_evidence"
        assert answer.citations and all(c.verified for c in answer.citations)


def test_answer_structure(env):
    answer = ask(env, "Nephrology", "ramipril 2.5 mg once daily")
    assert [s.agent for s in answer.steps] == ["Monitoring", "Extraction", "Comparison", "Impact", "Briefing", "Governance"]
    assert len(answer.summary) <= 5 and answer.disclaimer == DISCLAIMER
    assert "Nephrology" in answer.affected_departments
    assert answer.sources_checked and all(s.checked_at for s in answer.sources_checked)
    neph = next(s for s in answer.sources_checked if s.watchlist_id == "neph-ckd-guideline")
    assert (neph.latest_version, neph.previous_version) == ("2.0", "1.0")


def test_query_is_persisted_and_audited_without_patient_context(env):
    answer = ask(env, "Endocrinology", "metformin 500 mg twice daily", egfr_band="30-44",
                 comorbidities=["heart failure"], current_medications=["simvastatin"])
    with env.session_factory() as session:
        record = session.get(ClinicianQueryRecord, answer.query_id)
        assert record.verdict == answer.verdict and record.cited_statement_ids
        stored = json.dumps(record.answer)
        assert "30-44" not in stored and "heart failure" not in stored and "simvastatin" not in stored
        audit = session.query(AuditLog).filter_by(entity_id=record.id).one()
        assert audit.entity_type == "ClinicianQuery" and "heart failure" not in json.dumps(audit.audit_metadata)


def test_identifiers_rejected_before_anything_runs(env):
    with env.session_factory() as session:
        before = session.query(ClinicianQueryRecord).count()
    with pytest.raises(PatientIdentifierError):
        ask(env, "Cardiology", "apixaban 5 mg twice daily for Mr Patel MRN 1234567")
    with env.session_factory() as session:
        assert session.query(ClinicianQueryRecord).count() == before


def test_out_of_date_protocol_flags_governance_once(env):
    first = ask(env, "Emergency Medicine", "intramuscular adrenaline 0.5 mg")
    second = ask(env, "Emergency Medicine", "intramuscular adrenaline 0.5 mg")
    assert first.governance and first.governance[0].status == "flagged_to_governance"
    with env.session_factory() as session:
        flags = [n for n in session.query(Notification).filter_by(notification_type="protocol_out_of_date")
                 if n.payload["protocol_id"] == "PROT-EM-001"]
        assert len(flags) == 1
    assert first.governance[0].message == second.governance[0].message


def test_aligned_protocol_needs_no_governance(env):
    answer = ask(env, "Cardiology", "apixaban 5 mg twice daily")
    assert answer.governance == []


def test_degraded_mode_when_comparison_fails(env):
    broken = MagicMock(side_effect=RuntimeError("comparison offline"))
    original = env.workflow.comparison.compare_treatment_plan
    env.workflow.comparison.compare_treatment_plan = broken
    try:
        answer = ask(env, "Cardiology", "apixaban 5 mg twice daily")
    finally:
        env.workflow.comparison.compare_treatment_plan = original
    assert answer.verdict is None and answer.degraded_mode
    assert next(s for s in answer.steps if s.agent == "Comparison").status == "failed"


def test_conditional_dose_reduction_is_consistent(env):
    answer = ask(env, "Cardiology", "apixaban 2.5 mg twice daily")
    assert answer.verdict == "consistent_with_latest_guidance"


def test_each_plan_part_is_assessed_separately(env):
    corpus_query, outcome = env.compare("Emergency Medicine", "intramuscular adrenaline 0.5 mg, repeat after 10 minutes")
    parts = {c.attribute: c for c in outcome.plan_components}
    assert parts["dose"].status == "supported" and parts["route"].status == "supported"
    assert parts["interval"].status == "unsupported"
    assert parts["interval"].superseded_support.version == "1.0"
    assert "after 5 minutes" in parts["interval"].contradicted_by[0].excerpt
