"""Phase 8: labelled evaluation of the clinician Treatment Check and surveillance (SYNTHETIC corpus)."""

import pytest

from app.agents.briefing_agent import BriefingAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.impact_agent import ImpactAgent
from app.orchestration.clinician_query import ClinicianQueryWorkflow
from app.services.clinician_query_evaluation import run_clinician_evaluation, write_report
from tests.corpus_env import PROTOCOLS, TAXONOMY, WATCHLIST, build_corpus_env


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    env = build_corpus_env(tmp)
    impact = ImpactAgent(session_factory=env.session_factory, config=env.config)
    impact.assess_guidance_changes()
    workflow = ClinicianQueryWorkflow(
        session_factory=env.session_factory, config=env.config, monitoring_agent=env.monitoring,
        extraction_agent=env.extraction, comparison_agent=env.comparison, impact_agent=impact,
        briefing_agent=BriefingAgent(session_factory=env.session_factory, config=env.config),
        governance_agent=GovernanceAgent(session_factory=env.session_factory, config=env.config),
        watchlist=WATCHLIST, taxonomy=TAXONOMY, protocols=PROTOCOLS,
    )
    result = run_clinician_evaluation(workflow, env.session_factory, impact)
    write_report(result, tmp / "reports")
    yield result
    env.engine.dispose()


def test_at_least_twenty_labelled_cases_covering_every_verdict(report):
    assert report["cases"] >= 20
    assert set(report["verdict_accuracy_by_class"]) >= {
        "consistent_with_latest_guidance", "guidance_updated_follow_new_version",
        "conflicts_with_latest_guidance", "insufficient_grounded_evidence", "rejected"}


def test_every_case_passes(report):
    failed = [(r["case_id"], r["expected"], r["actual"], r["checks"]) for r in report["results"] if not r["passed"]]
    assert failed == []


def test_grounding_guarantees(report):
    assert report["citations_verified_rate"] == 1.0
    assert report["ungrounded_answers"] == 0
    assert report["latest_version_identified_rate"] == 1.0


def test_surveillance_quality(report):
    s = report["surveillance"]
    assert s["change_category_accuracy"] == 1.0, s["change_category_by_source"]
    assert s["filter_precision"] == 1.0 and s["filter_recall"] == 1.0
    assert s["filtered_items_retained"] > 0
    assert s["duplicate_linked_to_higher_quality_source"]
    assert s["safety_and_threshold_ranked_above_case_series"]
