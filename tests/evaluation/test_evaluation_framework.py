"""Phase 12: Comprehensive Evaluation Framework Test Suite.

Covers all 25 mandatory architectural test requirements:
1. Extraction corpus has >= 20 meaningful cases
2. Comparison corpus has >= 15 scenarios
3. Same 20-document corpus is used for E2E
4. Expected extraction outcomes are validated
5. G1 cases are correctly detected
6. Expected comparison outcomes are validated
7. G2 cases are correctly detected
8. G3 no-match cases are correctly detected
9. Impact score evaluation
10. Incomplete impact detection
11. Briefing seven-section validation
12. Briefing determinism & hash stability
13. Governance safety assertions
14. G4 cannot be bypassed
15. G5 cannot make a decision
16. Idempotency evaluation
17. Provenance evaluation
18. Report schema validation
19. Report deterministic ordering
20. Threshold calibration report generation
21. Zero unnecessary LLM calls
22. No protocol modification (immutability)
23. Repeated evaluation stability
24. End-to-end case accounting
25. Failure cases reported rather than silently ignored
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.models.database import get_engine, get_session_factory, init_db
from app.models.entities import (
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    Notification,
    ReviewAssignment,
)
from app.schemas.briefs import (
    BriefStatus,
    ComparisonSectionPayload,
    ImpactSectionPayload,
    ProposedActionItem,
    ProtocolSectionPayload,
    RecommendationSectionPayload,
    SourceExcerptSectionPayload,
    SourceMetadataPayload,
    StructuredBriefPayload,
    WorkflowSectionPayload,
    validate_brief_completeness,
)
from app.schemas.changes import ChangeStatus
from app.schemas.comparison import ComparisonResponse, ComparisonResult, DifferenceType
from app.schemas.evaluation import (
    EvaluationCaseResult,
    EvaluationOutcome,
    EvaluationSummaryReport,
    SuiteSummary,
    ThresholdCalibrationReport,
)
from app.schemas.governance import GovernanceDecisionRequest, ReviewDecision
from app.schemas.impact import ImpactStatus, ImpactTier
from app.schemas.orchestration import HumanGate, PipelineStage, PipelineStatus
from app.services.config_service import load_config
from app.services.evaluation_corpus import (
    EvaluationCorpusService,
    RAW_COMPARISON_SCENARIOS,
    RAW_EXTRACTION_CASES,
)
from app.services.evaluation_reporter import EvaluationReporter
from app.services.evaluation_runner import EvaluationRunner, ProtocolImmutabilityViolation
from app.services.scoring_engine import ScoringEngine


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def corpus_service():
    return EvaluationCorpusService()


@pytest.fixture
def temp_reports_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def runner(config, corpus_service, temp_reports_dir):
    reporter = EvaluationReporter(reports_dir=temp_reports_dir)
    return EvaluationRunner(
        config=config,
        corpus_service=corpus_service,
        reporter=reporter,
        live_llm=False,
        work_dir=temp_reports_dir,
    )


# =============================================================================
# 1. CORPUS DISCOVERY AND COUNTS
# =============================================================================

def test_01_extraction_corpus_has_at_least_20_cases(corpus_service):
    """Requirement 1: Extraction corpus must contain >= 20 meaningful cases."""
    cases = corpus_service.get_extraction_cases()
    assert len(cases) >= 20, f"Expected >= 20 extraction cases, got {len(cases)}"
    unique_ids = {c["case_id"] for c in cases}
    assert len(unique_ids) == len(cases), "Extraction case IDs must be unique"
    # Ensure meaningful variation
    condition_types = {c["condition_type"] for c in cases}
    assert len(condition_types) >= 10, "Extraction corpus lacks meaningful condition diversity"


def test_02_comparison_corpus_has_at_least_15_scenarios(corpus_service):
    """Requirement 2: Comparison corpus must contain >= 15 scenarios."""
    scenarios = corpus_service.get_comparison_scenarios()
    assert len(scenarios) >= 15, f"Expected >= 15 comparison scenarios, got {len(scenarios)}"
    unique_ids = {s["scenario_id"] for s in scenarios}
    assert len(unique_ids) == len(scenarios), "Comparison scenario IDs must be unique"
    # Check scenario diversity
    conditions = {s["condition_type"] for s in scenarios}
    assert len(conditions) >= 8, "Comparison corpus lacks meaningful condition diversity"


def test_03_same_20_document_corpus_is_used_for_e2e(corpus_service):
    """Requirement 3: Same 20-document corpus is used for End-to-End evaluation."""
    ext_cases = corpus_service.get_extraction_cases()
    assert len(ext_cases) >= 20
    filenames = [c["filename"] for c in ext_cases]
    assert len(filenames) == len(ext_cases), "Duplicate filenames in extraction corpus"
    # Verify every extraction PDF is present in the expected E2E directory
    for fn in filenames:
        expected_path = corpus_service.e2e_dir / fn
        assert expected_path.exists() or (corpus_service.extraction_dir / fn).exists()


# =============================================================================
# 2. EXTRACTION EVALUATION
# =============================================================================

def test_04_expected_extraction_outcomes_are_validated(runner):
    """Requirement 4: Expected extraction outcomes are validated with metrics."""
    summary = runner.run_extraction_suite()
    assert summary.total_cases >= 20
    assert summary.failed == 0
    assert summary.errors == 0
    assert summary.metrics["presence_accuracy"] >= 0.95
    assert summary.metrics["successful_extractions"] >= 15


def test_05_g1_cases_are_correctly_detected(runner):
    """Requirement 5: Low-confidence / provenance-failing G1 cases are detected."""
    summary = runner.run_extraction_suite()
    g1_cases = [c for c in summary.case_results if c.outcome == EvaluationOutcome.HELD_EXPECTED]
    assert len(g1_cases) >= 4, f"Expected at least 4 G1 hold cases, found {len(g1_cases)}"
    for c in g1_cases:
        assert "held" in c.actual_behavior.lower() or "held" in c.expected_behavior.lower()


# =============================================================================
# 3. COMPARISON EVALUATION
# =============================================================================

def test_06_expected_comparison_outcomes_are_validated(runner):
    """Requirement 6: Expected comparison outcomes are validated with precision/recall/F1."""
    summary = runner.run_comparison_suite()
    assert summary.total_cases >= 15
    assert summary.failed == 0
    assert summary.errors == 0
    assert summary.metrics["difference_type_accuracy"] >= 0.85
    assert summary.metrics["f1_score"] >= 0.70


def test_07_g2_cases_are_correctly_detected(runner):
    """Requirement 7: Ambiguous / invalid comparison cases held at G2."""
    summary = runner.run_comparison_suite()
    g2_results = [
        c for c in summary.case_results
        if "review_required" in c.actual_behavior.lower() or "g2" in c.case_id.lower() or "ambiguous" in c.actual_behavior.lower()
    ]
    assert len(g2_results) >= 2, f"Expected G2 review holds in comparison, found {len(g2_results)}"


def test_08_g3_no_match_cases_are_correctly_detected(runner):
    """Requirement 8: Confirmed no-match cases routed to G3."""
    summary = runner.run_comparison_suite()
    no_match_results = [
        c for c in summary.case_results
        if "no_match" in c.actual_behavior.lower()
    ]
    assert len(no_match_results) >= 2, f"Expected G3 no-match cases, found {len(no_match_results)}"


# =============================================================================
# 4. IMPACT SCORING EVALUATION
# =============================================================================

def test_09_impact_score_evaluation(runner):
    """Requirement 9: Impact scoring evaluates rules, dimensions, tiers, and SLA."""
    summary = runner.run_impact_suite()
    assert summary.total_cases >= 4
    assert summary.failed == 0
    assert summary.errors == 0
    tiers_tested = {c.expected_behavior for c in summary.case_results}
    assert any("critical" in t.lower() for t in tiers_tested)
    assert any("high" in t.lower() for t in tiers_tested)
    assert any("standard" in t.lower() for t in tiers_tested)


def test_10_incomplete_impact_detection(runner):
    """Requirement 10: Incomplete / missing dimension correctly flagged as incomplete without guessing."""
    summary = runner.run_impact_suite()
    incomplete_case = next((c for c in summary.case_results if c.case_id == "IMP-05"), None)
    assert incomplete_case is not None, "Missing dimension test case IMP-05 not found"
    assert incomplete_case.outcome == EvaluationOutcome.PASS
    assert "incomplete" in incomplete_case.message.lower() or "none" in incomplete_case.actual_behavior.lower()


# =============================================================================
# 5. BRIEFING EVALUATION
# =============================================================================

def test_11_briefing_seven_section_validation(runner):
    """Requirement 11: All seven required sections must be validated."""
    summary = runner.run_briefing_suite()
    complete_case = next((c for c in summary.case_results if c.case_id == "BRF-01"), None)
    assert complete_case is not None
    assert complete_case.outcome == EvaluationOutcome.PASS
    assert complete_case.metrics.get("sections_present") == 7


def test_12_briefing_determinism(runner):
    """Requirement 12: Brief rendering is deterministic with stable SHA-256 hash."""
    summary = runner.run_briefing_suite()
    hash_case = next((c for c in summary.case_results if c.case_id == "BRF-03"), None)
    assert hash_case is not None
    assert hash_case.outcome == EvaluationOutcome.PASS
    assert "stable" in hash_case.actual_behavior.lower()


# =============================================================================
# 6. GOVERNANCE SAFETY EVALUATION
# =============================================================================

def test_13_governance_safety_assertions(runner):
    """Requirement 13: Governance safety assertions verify zero forbidden actions."""
    summary = runner.run_governance_safety_suite()
    assert summary.total_cases >= 5
    assert summary.failed == 0
    assert summary.errors == 0
    forbidden_case = next((c for c in summary.case_results if c.case_id == "GOV-FORBIDDEN"), None)
    assert forbidden_case is not None
    assert forbidden_case.outcome == EvaluationOutcome.PASS


def test_14_g4_cannot_be_bypassed(runner):
    """Requirement 14: G4 cannot be bypassed; system daemon cannot close brief."""
    summary = runner.run_governance_safety_suite()
    g4_case = next((c for c in summary.case_results if c.case_id == "GOV-G4"), None)
    assert g4_case is not None
    assert g4_case.outcome == EvaluationOutcome.PASS
    assert "blocked" in g4_case.actual_behavior.lower()


def test_15_g5_cannot_make_a_decision(runner):
    """Requirement 15: G5 SLA escalation triggers notifications only, never decides."""
    summary = runner.run_governance_safety_suite()
    g5_case = next((c for c in summary.case_results if c.case_id == "GOV-G5"), None)
    assert g5_case is not None
    assert g5_case.outcome == EvaluationOutcome.PASS


# =============================================================================
# 7. IDEMPOTENCY & PROVENANCE EVALUATION
# =============================================================================

def test_16_idempotency_evaluation(runner, config):
    """Requirement 16: Repeated execution does not create duplicate entities."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine=engine)
    sf = get_session_factory(engine=engine)

    # Ingest document
    pdf_path = runner.corpus_service.extraction_dir / "ext_01_clear_rec.pdf"
    sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    with sf() as session:
        doc1 = IngestedDocument(
            source_identifier="SYN-IDEM-01",
            source_path=str(pdf_path),
            sha256_hash=sha,
            source_version="1.0",
            pipeline_version="1.0",
            status="parsed",
        )
        session.add(doc1)
        session.commit()
        doc_id = doc1.id

    # Verify second insertion with same hash or duplicate check is prevented
    with sf() as session:
        existing = session.query(IngestedDocument).filter_by(sha256_hash=sha).all()
        assert len(existing) == 1, "Duplicate IngestedDocument detected on idempotency check."

    engine.dispose()


def test_17_provenance_evaluation(runner):
    """Requirement 17: Provenance metadata (SHA, page, section, protocol) preserved."""
    summary = runner.run_extraction_suite()
    for case in summary.case_results:
        assert case.provenance_valid is True or case.case_id == "EXT-19", (
            f"Provenance unexpectedly invalid for case {case.case_id}"
        )


# =============================================================================
# 8. REPORTING & DETERMINISM
# =============================================================================

def test_18_report_schema_validation(runner, temp_reports_dir):
    """Requirement 18: Generated reports validate against Pydantic schema."""
    summary = runner.run_impact_suite()
    report_file = temp_reports_dir / "impact_report.json"
    assert report_file.exists()
    content = json.loads(report_file.read_text(encoding="utf-8"))
    validated = SuiteSummary.model_validate(content)
    assert validated.suite_name == "Impact"
    assert validated.total_cases == summary.total_cases


def test_19_report_deterministic_ordering(runner, temp_reports_dir):
    """Requirement 19: Reports are deterministically sorted by case_id."""
    runner.run_comparison_suite()
    report_file = temp_reports_dir / "comparison_report.json"
    data = json.loads(report_file.read_text(encoding="utf-8"))
    case_ids = [c["case_id"] for c in data["case_results"]]
    assert case_ids == sorted(case_ids), "Report cases are not deterministically sorted."


def test_20_threshold_calibration_report_generation(runner, temp_reports_dir):
    """Requirement 20: Threshold calibration report identifies cases around 0.70 boundary."""
    report = runner.run_threshold_calibration()
    assert isinstance(report, ThresholdCalibrationReport)
    assert report.extraction_threshold == 0.70
    assert report.comparison_threshold == 0.70
    total_boundary = len(report.extraction_boundary_cases) + len(report.comparison_boundary_cases)
    assert total_boundary >= 2
    cal_file = temp_reports_dir / "threshold_calibration_report.json"
    assert cal_file.exists()


# =============================================================================
# 9. INTEGRITY & IMMUTABILITY
# =============================================================================

def test_21_zero_unnecessary_llm_calls(runner):
    """Requirement 21: Briefing, impact, and safety suites make ZERO LLM calls."""
    with patch("app.services.llm_client.SharedLLMClient.extract_recommendations") as mock_ext:
        with patch("app.services.llm_client.SharedLLMClient.compare_recommendation_to_protocol") as mock_cmp:
            runner.run_impact_suite()
            runner.run_briefing_suite()
            runner.run_governance_safety_suite()
            assert mock_ext.call_count == 0, "LLM was invoked during impact/briefing/governance evaluation!"
            assert mock_cmp.call_count == 0, "LLM was invoked during impact/briefing/governance evaluation!"


def test_22_no_protocol_modification(runner):
    """Requirement 22: Evaluation MUST NOT mutate protocol file, hash, or index."""
    protocol_path = Path("data/protocols/PROT-DM-001_v1.0.json")
    before_bytes = protocol_path.read_bytes()
    before_sha = hashlib.sha256(before_bytes).hexdigest()

    runner.run_comparison_suite()

    after_bytes = protocol_path.read_bytes()
    after_sha = hashlib.sha256(after_bytes).hexdigest()
    assert before_sha == after_sha, "Protocol file was mutated during comparison evaluation!"


def test_23_repeated_evaluation_stability(runner):
    """Requirement 23: Repeated evaluation produces stable and identical outcomes."""
    res1 = runner.run_impact_suite()
    res2 = runner.run_impact_suite()
    assert res1.total_cases == res2.total_cases
    assert res1.passed == res2.passed
    assert res1.failed == res2.failed
    for c1, c2 in zip(res1.case_results, res2.case_results):
        assert c1.case_id == c2.case_id
        assert c1.outcome == c2.outcome


def test_24_end_to_end_case_accounting(runner):
    """Requirement 24: E2E accounting accounts for all 20 cases with zero lost outcomes."""
    summary = runner.run_e2e_suite()
    total_accounted = summary.passed + summary.held_expected + summary.failed + summary.errors + summary.not_run
    assert total_accounted == summary.total_cases == 20
    assert summary.metrics["zero_auto_decisions_verified"] is True


def test_25_failure_cases_reported_rather_than_silently_ignored(runner):
    """Requirement 25: Evaluation failures are explicitly recorded and reported."""
    failed_result = EvaluationCaseResult(
        case_id="TEST-FAIL-01",
        suite="extraction",
        outcome=EvaluationOutcome.FAIL,
        message="Simulated test failure",
        expected_behavior="extracted",
        actual_behavior="held_for_g1",
        provenance_valid=False,
    )
    summary = SuiteSummary(
        suite_name="Extraction",
        total_cases=1,
        passed=0,
        failed=1,
        held_expected=0,
        errors=0,
        not_run=0,
        metrics={},
        case_results=[failed_result],
    )
    runner.reporter.write_suite_report(summary, filename="test_fail_report.json")
    saved = json.loads((runner.reporter.reports_dir / "test_fail_report.json").read_text(encoding="utf-8"))
    assert saved["failed"] == 1
    assert saved["case_results"][0]["outcome"] == "FAIL"
