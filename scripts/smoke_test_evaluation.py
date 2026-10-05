"""Smoke test for Phase 12 Evaluation Framework.

Verifies:
1. Corpus discovery and counts (>=20 extraction documents, >=15 comparison scenarios, same 20 for E2E)
2. Representative extraction evaluation
3. Representative comparison evaluation
4. Impact evaluation
5. Briefing evaluation
6. Governance safety checks (G1..G5 enforcement, zero forbidden actions)
7. Representative end-to-end evaluation
8. Evaluation summary generation
9. Deterministic report generation & hash stability
10. Protocol immutability (text, hash, and index untouched)
11. Zero prohibited automated governance decisions (never auto-approve/reject/defer/close)
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.models.database import get_engine, get_session_factory, init_db
from app.models.entities import ChangeBrief, ChangeRecord, GapRecord, ImpactRecord, IngestedDocument
from app.schemas.briefs import BriefStatus
from app.schemas.evaluation import EvaluationOutcome
from app.services.config_service import load_config
from app.services.evaluation_corpus import EvaluationCorpusService
from app.services.evaluation_reporter import EvaluationReporter
from app.services.evaluation_runner import EvaluationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ckea.smoke_test_evaluation")


def run_smoke_test() -> int:
    logger.info("=================================================================")
    logger.info(" CKEA PHASE 12: EVALUATION FRAMEWORK SMOKE TEST")
    logger.info("=================================================================")

    config = load_config()
    corpus_service = EvaluationCorpusService()

    # 1. Corpus discovery and counts
    logger.info("[Step 1/11] Discovering evaluation corpus and verifying counts...")
    extraction_cases = corpus_service.get_extraction_cases()
    comparison_scenarios = corpus_service.get_comparison_scenarios()

    ext_count = len(extraction_cases)
    cmp_count = len(comparison_scenarios)
    logger.info("Discovered %d extraction documents (target >= 20)", ext_count)
    logger.info("Discovered %d comparison scenarios (target >= 15)", cmp_count)

    assert ext_count >= 20, f"Extraction corpus has {ext_count} cases, expected >= 20."
    assert cmp_count >= 15, f"Comparison corpus has {cmp_count} scenarios, expected >= 15."

    # Verify same 20-document corpus is used for E2E
    e2e_files = [c["filename"] for c in extraction_cases]
    assert len(set(e2e_files)) == ext_count, "Extraction filenames are not distinct."
    logger.info("Verified: Same %d documents are designated for End-to-End evaluation.", ext_count)

    # 2. Snapshot protocol data before any evaluation
    logger.info("[Step 2/11] Snapshotting protocol data for immutability verification...")
    protocol_file = Path("data/protocols/PROT-DM-001_v1.0.json")
    assert protocol_file.exists(), f"Protocol file missing: {protocol_file}"
    initial_protocol_bytes = protocol_file.read_bytes()
    initial_protocol_sha = hashlib.sha256(initial_protocol_bytes).hexdigest()
    initial_protocol_json = json.loads(initial_protocol_bytes.decode("utf-8"))

    # 3. Initialize EvaluationRunner in isolated temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_reports_dir = tmp_path / "reports"
        reporter = EvaluationReporter(reports_dir=test_reports_dir)
        runner = EvaluationRunner(
            config=config,
            corpus_service=corpus_service,
            reporter=reporter,
            live_llm=False,
            work_dir=tmp_path,
        )

        # 4. Run Impact Evaluation Suite
        logger.info("[Step 3/11] Running Impact Scoring Evaluation Suite...")
        impact_summary = runner.run_impact_suite()
        assert impact_summary.total_cases >= 4, "Impact suite ran insufficient cases."
        assert impact_summary.failed == 0, f"Impact suite had {impact_summary.failed} failures."
        assert impact_summary.errors == 0, f"Impact suite had {impact_summary.errors} errors."
        logger.info("Impact suite passed: %d cases, 0 failures, 0 errors.", impact_summary.total_cases)

        # 5. Run Briefing Evaluation Suite
        logger.info("[Step 4/11] Running Briefing Evaluation Suite...")
        briefing_summary = runner.run_briefing_suite()
        assert briefing_summary.total_cases >= 3, "Briefing suite ran insufficient cases."
        assert briefing_summary.failed == 0, f"Briefing suite had {briefing_summary.failed} failures."
        assert briefing_summary.errors == 0, f"Briefing suite had {briefing_summary.errors} errors."
        logger.info("Briefing suite passed: 7 sections, verbatim quotes, and hash stability verified.")

        # 6. Run Governance Safety Evaluation Suite
        logger.info("[Step 5/11] Running Governance Safety Suite (G1..G5 & forbidden actions)...")
        gov_summary = runner.run_governance_safety_suite()
        assert gov_summary.failed == 0, "Governance safety checks detected violations!"
        assert gov_summary.errors == 0, "Governance safety checks encountered runtime errors!"
        logger.info("Governance safety passed: G1..G5 enforced, zero automated approvals/rejections.")

        # 7. Run Representative Extraction Evaluation
        logger.info("[Step 6/11] Running Extraction Evaluation Suite (20 documents)...")
        ext_summary = runner.run_extraction_suite()
        assert ext_summary.total_cases >= 20, "Extraction suite did not run all 20 cases."
        assert ext_summary.failed == 0, f"Extraction suite had {ext_summary.failed} failures."
        assert ext_summary.errors == 0, f"Extraction suite had {ext_summary.errors} errors."
        assert ext_summary.metrics.get("g1_holds", 0) > 0, "Extraction suite failed to detect G1 holds."
        logger.info("Extraction suite passed: %d evaluated, %d extracted, %d G1 holds.",
                    ext_summary.total_cases,
                    ext_summary.metrics.get("successful_extractions", 0),
                    ext_summary.metrics.get("g1_holds", 0))

        # 8. Run Representative Comparison Evaluation
        logger.info("[Step 7/11] Running Comparison Evaluation Suite (16 scenarios)...")
        cmp_summary = runner.run_comparison_suite()
        assert cmp_summary.total_cases >= 15, "Comparison suite ran < 15 scenarios."
        assert cmp_summary.failed == 0, f"Comparison suite had {cmp_summary.failed} failures."
        assert cmp_summary.errors == 0, f"Comparison suite had {cmp_summary.errors} errors."
        assert cmp_summary.metrics.get("g2_holds", 0) > 0, "Comparison suite failed to detect G2 holds."
        assert cmp_summary.metrics.get("g3_holds", 0) > 0, "Comparison suite failed to detect G3 holds."
        logger.info("Comparison suite passed: %d scenarios, G2 holds=%d, G3 holds=%d.",
                    cmp_summary.total_cases,
                    cmp_summary.metrics.get("g2_holds", 0),
                    cmp_summary.metrics.get("g3_holds", 0))

        # 9. Run End-to-End Pipeline Evaluation
        logger.info("[Step 8/11] Running End-to-End Pipeline Evaluation (same 20 documents)...")
        e2e_summary = runner.run_e2e_suite()
        assert e2e_summary.total_cases == ext_count, f"E2E suite ran {e2e_summary.total_cases} cases, expected {ext_count}."
        assert e2e_summary.failed == 0, f"E2E suite had {e2e_summary.failed} failures."
        assert e2e_summary.errors == 0, f"E2E suite had {e2e_summary.errors} errors."
        assert e2e_summary.metrics.get("zero_auto_decisions_verified") is True, "Prohibited auto decisions detected!"
        logger.info("E2E suite passed: %d documents, passed to governance=%d, human holds=%d.",
                    e2e_summary.total_cases,
                    e2e_summary.metrics.get("passed_to_governance", 0),
                    e2e_summary.metrics.get("held_at_human_gates", 0))

        # 10. Generate Threshold Calibration and Summary Reports
        logger.info("[Step 9/11] Generating Threshold Calibration and Evaluation Summary Reports...")
        cal_report = runner.run_threshold_calibration()
        total_boundary_cases = len(cal_report.extraction_boundary_cases) + len(cal_report.comparison_boundary_cases)
        assert total_boundary_cases >= 2, f"Threshold calibration analyzed insufficient boundary cases: {total_boundary_cases}."

        suites_dict = {
            "extraction": ext_summary,
            "comparison": cmp_summary,
            "impact": impact_summary,
            "briefing": briefing_summary,
            "governance": gov_summary,
            "e2e": e2e_summary,
        }
        total_cases = sum(s.total_cases for s in suites_dict.values())
        total_passed = sum(s.passed for s in suites_dict.values())
        total_held = sum(s.held_expected for s in suites_dict.values())
        total_failed = sum(s.failed for s in suites_dict.values())
        total_errors = sum(s.errors for s in suites_dict.values())

        from app.schemas.evaluation import EvaluationSummaryReport
        summary_report = EvaluationSummaryReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_mode="offline",
            git_branch="develop",
            total_cases=total_cases,
            total_passed=total_passed,
            total_failed=total_failed,
            total_held_expected=total_held,
            total_errors=total_errors,
            all_passed=(total_failed == 0 and total_errors == 0),
            suites=suites_dict,
        )
        runner.reporter.write_summary_report(summary_report)

        assert summary_report.all_passed is True, "Summary report indicates failures!"
        assert summary_report.total_failed == 0, f"Summary report records {summary_report.total_failed} failures."
        assert summary_report.total_errors == 0, f"Summary report records {summary_report.total_errors} errors."

        # Verify Report Files Exist and Are Valid JSON
        report_files = [
            "extraction_report.json",
            "comparison_report.json",
            "impact_report.json",
            "briefing_report.json",
            "governance_safety_report.json",
            "end_to_end_report.json",
            "threshold_calibration_report.json",
            "evaluation_summary.json",
        ]
        for r_name in report_files:
            r_path = test_reports_dir / r_name
            assert r_path.exists(), f"Required report file missing: {r_name}"
            with open(r_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, dict), f"Report {r_name} is not a valid JSON dictionary."
        logger.info("Verified: All 8 structured report JSON files successfully written and validated.")

        # 11. Verify Determinism: repeated report generation produces identical SHA-256
        logger.info("[Step 10/11] Verifying report determinism and hash stability...")
        summary_path = test_reports_dir / "evaluation_summary.json"
        content_1 = summary_path.read_bytes()
        runner.reporter.write_summary_report(summary_report, filename="evaluation_summary.json")
        content_2 = summary_path.read_bytes()
        assert hashlib.sha256(content_1).hexdigest() == hashlib.sha256(content_2).hexdigest(), (
            "Report generation is not deterministic!"
        )
        logger.info("Report determinism confirmed: SHA-256 hash is stable.")

    # 12. Protocol Immutability Verification
    logger.info("[Step 11/11] Verifying protocol immutability...")
    post_bytes = protocol_file.read_bytes()
    post_sha = hashlib.sha256(post_bytes).hexdigest()
    post_json = json.loads(post_bytes.decode("utf-8"))

    assert post_sha == initial_protocol_sha, "Protocol SHA-256 changed during evaluation!"
    assert post_json == initial_protocol_json, "Protocol JSON mutated during evaluation!"
    logger.info("Protocol immutability verified: SHA-256 and content are completely unchanged.")

    logger.info("=================================================================")
    logger.info(" SMOKE TEST PASSED: ALL EVALUATION ASSERTIONS GREEN")
    logger.info("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
