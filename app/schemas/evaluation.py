"""Pydantic schemas for the CKEA Evaluation Framework (Phase 12).

Defines structured models for evaluation results, suite summaries,
case accounting, threshold calibration, and machine-readable reports.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvaluationOutcome(str, Enum):
    """Discrete accounting outcome for each evaluation case."""

    PASS = "PASS"
    FAIL = "FAIL"
    HELD_EXPECTED = "HELD_EXPECTED"
    ERROR = "ERROR"


class EvaluationCaseResult(BaseModel):
    """Structured result for a single evaluated case or scenario."""

    model_config = ConfigDict(validate_assignment=True)

    case_id: str = Field(..., description="Unique identifier for the evaluation case (e.g. EXT-01, CMP-01)")
    suite: str = Field(..., description="Evaluation suite name (extraction, comparison, impact, briefing, governance, e2e)")
    outcome: EvaluationOutcome = Field(..., description="Evaluation outcome: PASS, FAIL, HELD_EXPECTED, ERROR")
    message: str = Field(..., description="Summary explanation of evaluation outcome")
    expected_behavior: str = Field(..., description="Expected behavior specification")
    actual_behavior: str = Field(..., description="Observed behavior during evaluation")
    provenance_valid: bool = Field(default=True, description="Whether all required provenance fields are intact")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific case metrics")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed diagnostic or assertion attributes")


class SuiteSummary(BaseModel):
    """Aggregate summary statistics and results for an evaluation suite."""

    model_config = ConfigDict(validate_assignment=True)

    suite_name: str = Field(..., description="Name of the evaluation suite")
    total_cases: int = Field(..., ge=0, description="Total number of cases evaluated in this suite")
    passed: int = Field(0, ge=0, description="Number of PASS outcomes")
    failed: int = Field(0, ge=0, description="Number of FAIL outcomes (including safety violations)")
    held_expected: int = Field(0, ge=0, description="Number of HELD_EXPECTED outcomes (correctly held by human gates)")
    errors: int = Field(0, ge=0, description="Number of ERROR outcomes (unexpected runtime exceptions)")
    not_run: int = Field(0, ge=0, description="Number of cases skipped or not run")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Aggregated domain metrics (precision, recall, etc.)")
    case_results: List[EvaluationCaseResult] = Field(default_factory=list, description="Ordered list of case results")


class ThresholdCalibrationCase(BaseModel):
    """Detailed observation for cases near provisional decision boundaries (e.g. 0.70)."""

    model_config = ConfigDict(validate_assignment=True)

    case_id: str = Field(..., description="Evaluation case identifier")
    domain: str = Field(..., description="Domain: extraction or comparison")
    score_type: str = Field(..., description="Type of score: extraction_confidence or similarity")
    observed_score: float = Field(..., ge=0.0, le=1.0, description="Observed score")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Applied threshold (e.g. 0.70)")
    distance_to_threshold: float = Field(..., description="Distance = observed_score - threshold")
    observed_outcome: str = Field(..., description="Observed gate or progression outcome")
    expected_outcome: str = Field(..., description="Expected gate or progression outcome")
    notes: str = Field(..., description="Clinical and operational observation notes")


class ThresholdCalibrationReport(BaseModel):
    """Report detailing boundary-sensitive evaluation evidence around provisional thresholds."""

    model_config = ConfigDict(validate_assignment=True)

    extraction_threshold: float = Field(default=0.70, description="Current provisional extraction confidence threshold")
    comparison_threshold: float = Field(default=0.70, description="Current provisional comparison similarity threshold")
    boundary_range: List[float] = Field(default_factory=lambda: [0.65, 0.75], description="Score window evaluated [min, max]")
    extraction_boundary_cases: List[ThresholdCalibrationCase] = Field(default_factory=list)
    comparison_boundary_cases: List[ThresholdCalibrationCase] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list, description="Key empirical observations")
    recommendations: List[str] = Field(default_factory=list, description="Informational-only calibration guidance")


class EvaluationSummaryReport(BaseModel):
    """Comprehensive summary of all evaluated suites."""

    model_config = ConfigDict(validate_assignment=True)

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_mode: str = Field(default="offline", description="Execution mode: offline or live_llm")
    git_branch: str = Field(default="develop")
    total_cases: int = Field(0, ge=0)
    total_passed: int = Field(0, ge=0)
    total_failed: int = Field(0, ge=0)
    total_held_expected: int = Field(0, ge=0)
    total_errors: int = Field(0, ge=0)
    all_passed: bool = Field(default=False)
    suites: Dict[str, SuiteSummary] = Field(default_factory=dict)
