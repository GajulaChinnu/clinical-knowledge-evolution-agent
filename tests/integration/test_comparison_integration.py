"""Opt-in integration test verifying live Groq API protocol comparison.

This test is strictly opt-in and is skipped by default during normal test runs.
To execute, set RUN_GROQ_INTEGRATION_TESTS=1 and provide a valid GROQ_API_KEY in the environment.
"""

import os
import pytest

from app.schemas.comparison import ComparisonResponse, ComparisonResult
from app.services.config_service import load_config
from app.services.llm_client import SharedLLMClient

RUN_INTEGRATION = os.environ.get("RUN_GROQ_INTEGRATION_TESTS") == "1"
HAS_API_KEY = bool(os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_API_KEY").strip())


@pytest.mark.skipif(
    not (RUN_INTEGRATION and HAS_API_KEY),
    reason="Opt-in test: set RUN_GROQ_INTEGRATION_TESTS=1 and GROQ_API_KEY to run live Groq tests.",
)
def test_live_groq_structured_comparison():
    """Execute a single live comparison inference against the Groq API when explicitly requested."""
    config = load_config()
    client = SharedLLMClient(config=config)

    recommendation_text = (
        "Adult patients diagnosed with type 2 diabetes should be initiated on metformin "
        "at 1000mg twice daily with meals if tolerated."
    )
    candidate_heading = "Metformin Monotherapy Starting Dose"
    candidate_text = (
        "Start metformin 500mg once daily with evening meals for non-pregnant adults."
    )

    response = client.compare_recommendation_to_protocol(
        recommendation_text=recommendation_text,
        target_population="Adults with Type 2 Diabetes",
        intervention="Metformin 1000mg twice daily",
        candidate_section_heading=candidate_heading,
        candidate_section_text=candidate_text,
        candidate_protocol_id="PROT-DM-001",
        candidate_protocol_version="1.0",
        candidate_section_id="SEC-2",
        evidence_grade="Grade A",
    )

    assert isinstance(response, ComparisonResponse)
    assert response.comparison_result in (ComparisonResult.GAP, ComparisonResult.NO_GAP, ComparisonResult.AMBIGUOUS)
    assert response.protocol_id == "PROT-DM-001"
    assert response.protocol_version == "1.0"
    assert response.section_id == "SEC-2"
    assert response.confidence >= 0.0
    assert response.rationale
    assert client.last_usage is not None
    assert client.last_usage["model"] == "openai/gpt-oss-20b"
