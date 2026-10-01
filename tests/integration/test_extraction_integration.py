"""Opt-in integration test verifying live Groq API recommendation extraction.

This test is strictly opt-in and is skipped by default during normal test runs.
To execute, set RUN_GROQ_INTEGRATION_TESTS=1 and provide a valid GROQ_API_KEY in the environment.
"""

import os
import pytest

from app.schemas.extraction import ExtractionResponse
from app.services.config_service import load_config
from app.services.llm_client import SharedLLMClient

RUN_INTEGRATION = os.environ.get("RUN_GROQ_INTEGRATION_TESTS") == "1"
HAS_API_KEY = bool(os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_API_KEY").strip())


@pytest.mark.skipif(
    not (RUN_INTEGRATION and HAS_API_KEY),
    reason="Opt-in test: set RUN_GROQ_INTEGRATION_TESTS=1 and GROQ_API_KEY to run live Groq tests.",
)
def test_live_groq_structured_extraction():
    """Execute a single live inference against the Groq API when explicitly requested."""
    config = load_config()
    client = SharedLLMClient(config=config)

    section_heading = "Glycemic Control in Type 2 Diabetes"
    section_text = (
        "Adult patients diagnosed with type 2 diabetes should be started on metformin "
        "as first-line monotherapy alongside lifestyle interventions. Target HbA1c "
        "should be below 7.0% for most non-pregnant adults (Grade A)."
    )

    response = client.extract_recommendations(
        section_heading=section_heading,
        section_text=section_text,
        page_number=1,
        document_identifier="live-integration-test",
    )

    assert isinstance(response, ExtractionResponse)
    assert len(response.recommendations) >= 1

    rec = response.recommendations[0]
    assert rec.verbatim_text
    assert rec.intervention
    assert rec.target_population
    assert rec.confidence >= 0.0
