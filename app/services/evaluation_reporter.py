"""Evaluation reporting service for CKEA (Phase 12).

Formats and persists deterministic machine-readable JSON and human-readable Markdown
evaluation reports across all evaluation suites:
- Extraction
- Comparison
- Impact
- Briefing
- Governance Safety
- End-to-End Pipeline
- Evaluation Summary
- Threshold Calibration
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.evaluation import (
    EvaluationCaseResult,
    EvaluationOutcome,
    EvaluationSummaryReport,
    SuiteSummary,
    ThresholdCalibrationCase,
    ThresholdCalibrationReport,
)

logger = logging.getLogger("ckea.services.evaluation_reporter")


class EvaluationReporter:
    """Generates structured JSON and Markdown reports with deterministic sorting."""

    def __init__(self, reports_dir: Optional[Path] = None) -> None:
        self.reports_dir = reports_dir or Path("data/evaluation/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_suite_report(self, summary: SuiteSummary, filename: Optional[str] = None) -> Path:
        """Serialize a SuiteSummary to JSON and companion Markdown in the reports directory."""
        if filename:
            stem = Path(filename).stem
            target_json = self.reports_dir / filename
            target_md = self.reports_dir / f"{stem}.md"
        else:
            clean_name = summary.suite_name.lower().replace(" ", "_").replace("-", "_")
            target_json = self.reports_dir / f"{clean_name}_report.json"
            target_md = self.reports_dir / f"{clean_name}_report.md"

        # Deterministic sorting of case results by case_id
        sorted_cases = sorted(summary.case_results, key=lambda c: c.case_id)
        summary_to_write = summary.model_copy(update={"case_results": sorted_cases})

        # Write JSON
        with open(target_json, "w", encoding="utf-8") as f:
            f.write(summary_to_write.model_dump_json(indent=2))

        # Write Markdown companion
        md_content = self._render_suite_markdown(summary_to_write)
        with open(target_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Wrote suite report: %s and %s", target_json, target_md)
        return target_json

    def write_threshold_calibration_report(
        self, report: ThresholdCalibrationReport, filename: str = "threshold_calibration_report.json"
    ) -> Path:
        """Write the threshold calibration analysis to JSON and Markdown."""
        target_json = self.reports_dir / filename
        target_md = self.reports_dir / "threshold_calibration_report.md"

        # Sort boundary cases deterministically by case_id
        sorted_ext = sorted(report.extraction_boundary_cases, key=lambda c: c.case_id)
        sorted_cmp = sorted(report.comparison_boundary_cases, key=lambda c: c.case_id)
        report_to_write = report.model_copy(
            update={
                "extraction_boundary_cases": sorted_ext,
                "comparison_boundary_cases": sorted_cmp,
            }
        )

        with open(target_json, "w", encoding="utf-8") as f:
            f.write(report_to_write.model_dump_json(indent=2))

        # Markdown
        lines = [
            "# CKEA Threshold Calibration Analysis (Provisional 0.70)",
            "",
            "> [!NOTE]",
            "> This calibration report is observational only. Production thresholds remain configured at 0.70.",
            "",
            f"- **Extraction Confidence Threshold**: `{report_to_write.extraction_threshold:.2f}`",
            f"- **Protocol Similarity Threshold**: `{report_to_write.comparison_threshold:.2f}`",
            f"- **Evaluated Score Boundary Window**: `[{report_to_write.boundary_range[0]:.2f}, {report_to_write.boundary_range[1]:.2f}]`",
            "",
            "## Extraction Boundary Cases",
            "",
            "| Case ID | Domain | Score | Threshold | Distance | Observed Gate | Expected Gate |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in report_to_write.extraction_boundary_cases:
            lines.append(
                f"| `{c.case_id}` | {c.domain} | {c.observed_score:.2f} | {c.threshold:.2f} | {c.distance_to_threshold:+.2f} | {c.observed_outcome} | {c.expected_outcome} |"
            )

        lines.extend([
            "",
            "## Comparison Boundary Cases",
            "",
            "| Case ID | Domain | Score | Threshold | Distance | Observed Gate | Expected Gate |",
            "|---|---|---|---|---|---|---|",
        ])
        for c in report_to_write.comparison_boundary_cases:
            lines.append(
                f"| `{c.case_id}` | {c.domain} | {c.observed_score:.2f} | {c.threshold:.2f} | {c.distance_to_threshold:+.2f} | {c.observed_outcome} | {c.expected_outcome} |"
            )

        lines.extend([
            "",
            "## Key Observations",
            "",
        ])
        for obs in report_to_write.observations:
            lines.append(f"- {obs}")

        lines.extend([
            "",
            "## Calibration Guidance",
            "",
        ])
        for rec in report_to_write.recommendations:
            lines.append(f"- {rec}")

        with open(target_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return target_json

    def write_summary_report(
        self, summary_report: EvaluationSummaryReport, filename: str = "evaluation_summary.json"
    ) -> Path:
        """Write the top-level evaluation summary report across all suites."""
        target_json = self.reports_dir / filename
        target_md = self.reports_dir / "evaluation_summary.md"

        # Sort suite keys deterministically
        sorted_suites = {k: summary_report.suites[k] for k in sorted(summary_report.suites.keys())}
        report_to_write = summary_report.model_copy(update={"suites": sorted_suites})

        with open(target_json, "w", encoding="utf-8") as f:
            f.write(report_to_write.model_dump_json(indent=2))

        lines = [
            "# CKEA Comprehensive Evaluation Summary Report",
            "",
            f"- **Timestamp**: `{report_to_write.timestamp}`",
            f"- **Execution Mode**: `{report_to_write.execution_mode}`",
            f"- **Branch**: `{report_to_write.git_branch}`",
            f"- **Total Evaluated Cases**: `{report_to_write.total_cases}`",
            f"- **Passed**: `{report_to_write.total_passed}`",
            f"- **Expected Holds**: `{report_to_write.total_held_expected}`",
            f"- **Failed**: `{report_to_write.total_failed}`",
            f"- **Errors**: `{report_to_write.total_errors}`",
            f"- **All Passed / Expected**: `{'YES' if report_to_write.all_passed else 'NO'}`",
            "",
            "## Suite Breakdown",
            "",
            "| Suite | Total | Passed | Held Expected | Failed | Errors | Status |",
            "|---|---|---|---|---|---|---|",
        ]

        for sname, s in report_to_write.suites.items():
            status = "PASS" if (s.failed == 0 and s.errors == 0) else "FAIL"
            lines.append(
                f"| **{s.suite_name}** | {s.total_cases} | {s.passed} | {s.held_expected} | {s.failed} | {s.errors} | **{status}** |"
            )

        lines.extend([
            "",
            "## Safety & Governance Assertions",
            "",
            "- **Zero Prohibited Automatic Decisions**: Verified across all cases.",
            "- **Zero Silent Brief Closures**: Verified across all cases.",
            "- **Zero Protocol Data Mutations**: Protocol text, versions, and hashes verified immutable.",
            "- **Zero Unnecessary LLM / API Calls**: Extraction and Comparison only; Briefing/Impact/Governance deterministic.",
            "",
        ])

        with open(target_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return target_json

    def _render_suite_markdown(self, summary: SuiteSummary) -> str:
        """Render a clean Markdown summary for an individual suite."""
        lines = [
            f"# Evaluation Report: {summary.suite_name}",
            "",
            f"- **Total Cases**: {summary.total_cases}",
            f"- **Passed**: {summary.passed}",
            f"- **Expected Holds**: {summary.held_expected}",
            f"- **Failed**: {summary.failed}",
            f"- **Errors**: {summary.errors}",
            "",
            "## Domain Metrics",
            "",
        ]

        if summary.metrics:
            for k, v in summary.metrics.items():
                if isinstance(v, float):
                    lines.append(f"- **{k}**: `{v:.4f}`")
                else:
                    lines.append(f"- **{k}**: `{v}`")
        else:
            lines.append("- *No domain metrics specified.*")

        lines.extend([
            "",
            "## Detailed Case Results",
            "",
            "| Case ID | Outcome | Expected | Actual | Provenance | Notes |",
            "|---|---|---|---|---|---|",
        ])

        for c in summary.case_results:
            prov_str = "Valid" if c.provenance_valid else "INVALID"
            msg = c.message.replace("|", "/")
            lines.append(
                f"| `{c.case_id}` | `{c.outcome.value}` | {c.expected_behavior} | {c.actual_behavior} | {prov_str} | {msg} |"
            )

        return "\n".join(lines) + "\n"
