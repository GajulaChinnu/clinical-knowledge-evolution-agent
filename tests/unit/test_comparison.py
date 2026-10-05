"""Unit tests for Phase 6 Comparison Agent.

All tests strictly mock the SharedLLMClient (Groq) to ensure:
1. Zero live Groq calls during pytest runs
2. Complete verification of embedding/retrieval integration
3. Top-3 candidate retrieval
4. Clear protocol match with gap
5. Difference type extraction
6. Protocol version persistence
7. Protocol quotation verification
8. Protocol quotation mismatch routing to G2
9. Comparison confidence >= 0.70 routing
10. Comparison confidence < 0.70 routing to G2
11. Contradictory output routing to G2
12. Invalid structured output routing to G2
13. No candidate reaching similarity threshold detection
14. No-match routing to G3 without calling Groq (Token optimization)
15. Idempotent retries (no duplicate GapRecord)
16. Only focused candidate sections passed to Groq
17. Full protocol repository excluded from Groq input
18. Groq provider and model configuration
19. Token usage metadata handled safely without leaking secrets
"""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.comparison_agent import ComparisonAgent
from app.models.database import Base, get_session_factory
from app.models.entities import ChangeRecord, GapRecord
from app.schemas.changes import ChangeStatus
from app.schemas.comparison import ComparisonResponse, ComparisonResult
from app.schemas.gaps import DifferenceType, GapStatus
from app.schemas.protocol import ProtocolDocument, ProtocolSection
from app.services.config_service import AppConfig
from app.services.llm_client import SharedLLMClient
from app.services.protocol_index import ProtocolIndexService


@pytest.fixture
def comparison_env(tmp_path):
    """Set up an isolated SQLite in-memory database and ChromaDB collection for comparison tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        groq_api_key="mock-groq-key",
        protocol_similarity_threshold=0.70,
        comparison_confidence_threshold=0.70,
        groq_model="openai/gpt-oss-20b",
    )

    protocol_service = ProtocolIndexService(chroma_dir=chroma_dir, config=config)

    # Index 10 distinct synthetic protocol sections (simulating institutional repository)
    protocol_sections = [
        ProtocolSection(
            protocol_id="PROT-DM-001",
            protocol_version="1.0",
            section_id="SEC-1",
            section_heading="Diabetes Screening Criteria",
            section_text="Fasting plasma glucose >= 126 mg/dL confirms diagnosis of type 2 diabetes mellitus.",
        ),
        ProtocolSection(
            protocol_id="PROT-DM-001",
            protocol_version="1.0",
            section_id="SEC-2",
            section_heading="Metformin Monotherapy Starting Dose",
            section_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        ),
        ProtocolSection(
            protocol_id="PROT-DM-001",
            protocol_version="1.0",
            section_id="SEC-3",
            section_heading="Glycemic Target HbA1c",
            section_text="Target HbA1c for non-pregnant adult patients is < 7.0% to minimize complications.",
        ),
        ProtocolSection(
            protocol_id="PROT-DM-001",
            protocol_version="1.0",
            section_id="SEC-4",
            section_heading="Renal Monitoring and Discontinuation",
            section_text="Monitor eGFR annually. Discontinue metformin if eGFR falls below 30 mL/min/1.73m2.",
        ),
        ProtocolSection(
            protocol_id="PROT-CARD-002",
            protocol_version="1.0",
            section_id="SEC-CARD-1",
            section_heading="Hypertension Blood Pressure Targets",
            section_text="Target blood pressure is systolic < 130 mmHg and diastolic < 80 mmHg.",
        ),
        ProtocolSection(
            protocol_id="PROT-CARD-002",
            protocol_version="1.0",
            section_id="SEC-CARD-2",
            section_heading="Statin Intensity in ASCVD",
            section_text="Initiate high-intensity atorvastatin 40mg to 80mg daily for clinical ASCVD.",
        ),
        ProtocolSection(
            protocol_id="PROT-RESP-003",
            protocol_version="1.0",
            section_id="SEC-RESP-1",
            section_heading="Asthma Controller Therapy",
            section_text="Low-dose inhaled fluticasone is recommended as daily controller therapy.",
        ),
        ProtocolSection(
            protocol_id="PROT-RESP-003",
            protocol_version="1.0",
            section_id="SEC-RESP-2",
            section_heading="Acute Asthma Exacerbation",
            section_text="Albuterol 2-4 puffs every 20 minutes for up to 1 hour for acute bronchospasm.",
        ),
        ProtocolSection(
            protocol_id="PROT-ONC-004",
            protocol_version="1.0",
            section_id="SEC-ONC-1",
            section_heading="Colorectal Cancer Screening",
            section_text="Begin routine colonoscopy screening at age 45 for average-risk individuals.",
        ),
        ProtocolSection(
            protocol_id="PROT-ONC-004",
            protocol_version="1.0",
            section_id="SEC-ONC-2",
            section_heading="Breast Cancer Mammography",
            section_text="Biennial screening mammography is recommended for women aged 40 to 74 years.",
        ),
    ]

    doc = ProtocolDocument(
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        sections=protocol_sections,
    )
    protocol_service.index_protocol(doc)

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "protocol_service": protocol_service,
        "config": config,
        "tmp_path": tmp_path,
    }

    engine.dispose()


def _create_sample_change_record(
    session_factory,
    verbatim_text: str,
    target_population: str = "Adults with Type 2 Diabetes",
    intervention: str = "Metformin",
    evidence_grade: str = "Grade A",
) -> str:
    """Helper to persist an initial ChangeRecord in 'extracted' status."""
    from app.models.entities import IngestedDocument

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="guideline-2026.pdf",
            source_path="/data/sources/guideline-2026.pdf",
            sha256_hash="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            status="processed",
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text=verbatim_text,
            recommendation_type="pharmacotherapy",
            target_population=target_population,
            intervention=intervention,
            evidence_grade=evidence_grade,
            page=12,
            section="Pharmacologic Therapy",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
            confidence=0.95,
        )
        session.add(change)
        session.commit()
        return change.id


def test_embedding_retrieval_integration(comparison_env):
    """Test 1: Verify ComparisonAgent leverages Phase 5 ChromaDB & embedding service."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Target HbA1c for non-pregnant adult patients is < 7.0%.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Glycemic Target HbA1c",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-3",
        exact_protocol_text="Target HbA1c for non-pregnant adult patients is < 7.0% to minimize complications.",
        specific_difference="Guideline matches existing institutional target of HbA1c < 7.0%.",
        difference_type=DifferenceType.NO_MATERIAL_DIFFERENCE,
        confidence=0.95,
        rationale="Both guideline and protocol advise an HbA1c target below 7.0%.",
    )

    agent = ComparisonAgent(
        session_factory=session_factory,
        protocol_index_service=protocol_service,
        llm_client=mock_llm,
        config=config,
    )
    gap = agent.process_change_record(change_id)

    assert gap is not None
    assert gap.is_match is True
    assert gap.comparison_result == ComparisonResult.NO_GAP.value
    assert gap.matched_protocol_id == "PROT-DM-001"
    assert gap.matched_protocol_version == "1.0"


def test_top_3_candidate_retrieval(comparison_env):
    """Test 2: Verify top-3 candidate protocol sections are retrieved from repository."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 500mg once daily with evening meals.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        specific_difference="Dosage and timing completely align.",
        difference_type=DifferenceType.NO_MATERIAL_DIFFERENCE,
        confidence=0.98,
        rationale="Exact dose match.",
    )

    agent = ComparisonAgent(
        session_factory=session_factory,
        protocol_index_service=protocol_service,
        llm_client=mock_llm,
        config=config,
        top_k=3,
    )
    gap = agent.process_change_record(change_id)

    assert len(gap.candidate_protocol_section_ids) == 3
    # Top candidate should be the metformin section
    assert "SEC-2" in gap.candidate_protocol_section_ids[0]


def test_clear_protocol_match_with_gap(comparison_env):
    """Test 3: Verify clear protocol match with clinical gap transitions to gap_confirmed."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    # Guideline recommends 1000mg BID, protocol states 500mg daily
    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 1000mg twice daily with evening meals for non-pregnant adults.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        specific_difference="Starting dose increased from 500mg daily to 1000mg twice daily.",
        difference_type=DifferenceType.DOSAGE_CHANGE,
        confidence=0.92,
        rationale="The recommendation specifies a higher initial starting dose.",
    )

    agent = ComparisonAgent(
        session_factory=session_factory,
        protocol_index_service=protocol_service,
        llm_client=mock_llm,
        config=config,
    )
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.GAP_CONFIRMED.value
        assert gap.status == GapStatus.MATCHED.value
        assert gap.comparison_result == "gap"
        assert gap.difference_type == "dosage_change"


def test_specific_difference_extraction(comparison_env):
    """Test 4: Verify specific difference type is captured accurately in GapRecord."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Monitor eGFR annually. Discontinue metformin if eGFR falls below 45 mL/min/1.73m2.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Renal Monitoring and Discontinuation",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-4",
        exact_protocol_text="Monitor eGFR annually. Discontinue metformin if eGFR falls below 30 mL/min/1.73m2.",
        specific_difference="Cutoff threshold raised from eGFR 30 to eGFR 45.",
        difference_type=DifferenceType.THRESHOLD_CHANGE,
        confidence=0.88,
        rationale="Stricter renal threshold recommended.",
    )

    agent = ComparisonAgent(
        session_factory=session_factory,
        protocol_index_service=protocol_service,
        llm_client=mock_llm,
        config=config,
    )
    gap = agent.process_change_record(change_id)

    assert gap.difference_type == "threshold_change"


def test_protocol_version_persistence(comparison_env):
    """Test 5: Verify exact protocol version is persisted in GapRecord."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Target HbA1c for non-pregnant adult patients is < 7.0%.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Glycemic Target HbA1c",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-3",
        exact_protocol_text="Target HbA1c for non-pregnant adult patients is < 7.0% to minimize complications.",
        specific_difference="Targets align.",
        difference_type=DifferenceType.NO_MATERIAL_DIFFERENCE,
        confidence=0.95,
        rationale="Match.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    assert gap.matched_protocol_id == "PROT-DM-001"
    assert gap.matched_protocol_version == "1.0"
    assert gap.matched_protocol_version is not None


def test_protocol_quote_verification_success(comparison_env):
    """Test 6: Verify protocol quote verification succeeds when exact text is present in candidate."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Target blood pressure is systolic < 130 mmHg and diastolic < 80 mmHg.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Hypertension Blood Pressure Targets",
        protocol_id="PROT-CARD-002",
        protocol_version="1.0",
        section_id="SEC-CARD-1",
        exact_protocol_text="Target blood pressure is systolic < 130 mmHg and diastolic < 80 mmHg.",
        specific_difference="Targets match.",
        difference_type=DifferenceType.NO_MATERIAL_DIFFERENCE,
        confidence=0.96,
        rationale="Exact match.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.NO_GAP.value
        assert gap.status == GapStatus.MATCHED.value


def test_protocol_quote_mismatch_routes_to_g2(comparison_env):
    """Test 7: Verify invented/hallucinated protocol quotation routes to held_for_G2."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Target blood pressure is systolic < 130 mmHg and diastolic < 80 mmHg.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    # The model quotes text that does NOT exist in the candidate section
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Hypertension Blood Pressure Targets",
        protocol_id="PROT-CARD-002",
        protocol_version="1.0",
        section_id="SEC-CARD-1",
        exact_protocol_text="Invented quotation that is not anywhere in the section text!",
        specific_difference="Difference reported.",
        difference_type=DifferenceType.CONFLICT,
        confidence=0.95,
        rationale="Hallucinated text.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.HELD_FOR_G2.value
        assert gap.status == GapStatus.REVIEW_REQUIRED.value


def test_comparison_confidence_ge_70(comparison_env):
    """Test 8: Verify confidence >= 0.70 allows normal progression to gap_confirmed."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 850mg once daily with evening meals for non-pregnant adults.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        specific_difference="Dose change 500mg to 850mg.",
        difference_type=DifferenceType.DOSAGE_CHANGE,
        confidence=0.75,
        rationale="Confident comparison.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.GAP_CONFIRMED.value
        assert gap.status == GapStatus.MATCHED.value


def test_comparison_confidence_lt_70_routes_to_g2(comparison_env):
    """Test 9: Verify confidence < 0.70 routes to held_for_G2."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 850mg once daily with evening meals for non-pregnant adults.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        specific_difference="Dose change.",
        difference_type=DifferenceType.DOSAGE_CHANGE,
        confidence=0.65,  # Below 0.70 threshold
        rationale="Uncertain comparison.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.HELD_FOR_G2.value
        assert gap.status == GapStatus.REVIEW_REQUIRED.value


def test_contradictory_output_routes_to_g2(comparison_env):
    """Test 10: Verify contradictory/inconsistent model output routes to held_for_G2."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    # Contradiction: comparison_result says NO_GAP, but difference_type says CONFLICT!
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        specific_difference="Conflict in therapy.",
        difference_type=DifferenceType.CONFLICT,
        confidence=0.90,
        rationale="Contradictory output.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.HELD_FOR_G2.value
        assert gap.status == GapStatus.REVIEW_REQUIRED.value


def test_invalid_structured_output_routes_to_g2(comparison_env):
    """Test 11: Verify invalid or unparseable structured output routes to held_for_G2."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.side_effect = ValueError("Invalid JSON schema structure")

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.HELD_FOR_G2.value
        assert gap.status == GapStatus.REVIEW_REQUIRED.value


def test_no_candidate_reaches_similarity_threshold(comparison_env):
    """Test 12: Verify no candidate reaching similarity threshold is recognized as no-match."""
    protocol_service = comparison_env["protocol_service"]

    # Recommendation unrelated to any protocol section in database
    recommendation = "Infants born before 30 weeks gestation require ophthalmologic screening for retinopathy of prematurity."
    candidates = protocol_service.retrieve(recommendation_text=recommendation, top_k=3)

    matching = [c for c in candidates if c.similarity >= 0.70]
    assert len(matching) == 0


def test_no_match_routes_to_g3_and_skips_groq(comparison_env):
    """Test 13 (TOKEN OPTIMIZATION): Verify no-match routes to held_for_G3 and NEVER calls Groq."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Neonatal intensive care phototherapy guidelines for extreme hyperbilirubinemia.",
        target_population="Neonates",
        intervention="Phototherapy",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)

    agent = ComparisonAgent(
        session_factory=session_factory,
        protocol_index_service=protocol_service,
        llm_client=mock_llm,
        config=config,
    )
    gap = agent.process_change_record(change_id)

    # Groq must NOT have been called!
    assert mock_llm.compare_recommendation_to_protocol.call_count == 0

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.HELD_FOR_G3.value
        assert gap.status == GapStatus.NO_MATCH.value
        assert gap.is_match is False
        assert gap.comparison_result == "no_match"


def test_low_confidence_routes_to_g2(comparison_env):
    """Test 14: Verify low confidence comparison (< 0.70) routes to held_for_G2."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Target HbA1c for non-pregnant adult patients is < 7.0%.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Glycemic Target HbA1c",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-3",
        exact_protocol_text="Target HbA1c for non-pregnant adult patients is < 7.0% to minimize complications.",
        specific_difference="Possible gap.",
        difference_type=DifferenceType.SCOPE_EXPANSION,
        confidence=0.55,
        rationale="Low confidence comparison.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    gap = agent.process_change_record(change_id)

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.HELD_FOR_G2.value
        assert gap.status == GapStatus.REVIEW_REQUIRED.value


def test_retry_does_not_create_duplicate_gap_record(comparison_env):
    """Test 15: Verify retrying comparison does not create duplicate GapRecord for a ChangeRecord."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Target HbA1c for non-pregnant adult patients is < 7.0%.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Glycemic Target HbA1c",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-3",
        exact_protocol_text="Target HbA1c for non-pregnant adult patients is < 7.0% to minimize complications.",
        specific_difference="Aligned.",
        difference_type=DifferenceType.NO_MATERIAL_DIFFERENCE,
        confidence=0.95,
        rationale="Match.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)

    # First run
    gap1 = agent.process_change_record(change_id)

    # Reset ChangeRecord status to extracted for retry simulation
    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        change.status = ChangeStatus.EXTRACTED.value
        session.commit()

    # Second run (retry)
    gap2 = agent.process_change_record(change_id)

    assert gap1.id == gap2.id

    # Verify strictly 1 GapRecord exists in database
    with session_factory() as session:
        all_gaps = session.query(GapRecord).filter_by(change_record_id=change_id).all()
        assert len(all_gaps) == 1


def test_only_retrieved_candidate_sections_sent_to_groq(comparison_env):
    """Test 16: Verify only candidate section text is passed into Groq call."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        specific_difference="Exact match.",
        difference_type=DifferenceType.NO_MATERIAL_DIFFERENCE,
        confidence=0.98,
        rationale="Same dose.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    agent.process_change_record(change_id)

    assert mock_llm.compare_recommendation_to_protocol.call_count == 1
    call_kwargs = mock_llm.compare_recommendation_to_protocol.call_args.kwargs

    # Focused section sent
    assert "Start metformin 500mg" in call_kwargs["candidate_section_text"]
    assert call_kwargs["candidate_protocol_id"] == "PROT-DM-001"
    assert call_kwargs["candidate_section_id"] == "SEC-2"


def test_full_protocol_repository_not_sent_to_groq(comparison_env):
    """Test 17 (TOKEN-SAFETY TEST): Verify full protocol repository is NOT passed to Groq."""
    session_factory = comparison_env["session_factory"]
    protocol_service = comparison_env["protocol_service"]
    config = comparison_env["config"]

    # Target is metformin
    change_id = _create_sample_change_record(
        session_factory=session_factory,
        verbatim_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
    )

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.NO_GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily with evening meals for non-pregnant adults.",
        specific_difference="Exact match.",
        difference_type=DifferenceType.NO_MATERIAL_DIFFERENCE,
        confidence=0.98,
        rationale="Same dose.",
    )

    agent = ComparisonAgent(session_factory=session_factory, protocol_index_service=protocol_service, llm_client=mock_llm, config=config)
    agent.process_change_record(change_id)

    call_kwargs = mock_llm.compare_recommendation_to_protocol.call_args.kwargs
    sent_text = call_kwargs["candidate_section_text"]

    # Unrelated sections from the 10-section repository must NOT be in the sent payload!
    assert "Colonoscopy screening at age 45" not in sent_text
    assert "Mammography is recommended" not in sent_text
    assert "Albuterol 2-4 puffs" not in sent_text
    assert "High-intensity atorvastatin" not in sent_text
    assert "Target blood pressure is systolic" not in sent_text


def test_groq_client_uses_correct_provider_and_model():
    """Test 18: Verify SharedLLMClient uses Groq base_url and openai/gpt-oss-20b model."""
    config = AppConfig(groq_api_key="mock-key-12345")
    client = SharedLLMClient(config=config)

    assert client.base_url == "https://api.groq.com/openai/v1"
    assert client.model == "openai/gpt-oss-20b"


def test_token_usage_metadata_handled_safely():
    """Test 19: Verify token usage observability records metadata without leaking secrets."""
    from openai import OpenAI

    mock_openai = MagicMock(spec=OpenAI)
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "comparison_result": "gap",
        "matched_protocol_section": "Dosing",
        "protocol_id": "PROT-1",
        "protocol_version": "1.0",
        "section_id": "SEC-1",
        "exact_protocol_text": "500mg daily",
        "specific_difference": "Dose increased",
        "difference_type": "dosage_change",
        "confidence": 0.90,
        "rationale": "Clear dose change.",
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = MagicMock(prompt_tokens=180, completion_tokens=90, total_tokens=270)
    mock_openai.chat.completions.create.return_value = mock_completion

    config = AppConfig(groq_api_key="super-secret-key-123", groq_model="openai/gpt-oss-20b")
    client = SharedLLMClient(config=config, client=mock_openai)

    response = client.compare_recommendation_to_protocol(
        recommendation_text="1000mg daily",
        target_population="Adults",
        intervention="Drug",
        candidate_section_heading="Dosing",
        candidate_section_text="500mg daily",
        candidate_protocol_id="PROT-1",
        candidate_protocol_version="1.0",
        candidate_section_id="SEC-1",
    )

    assert isinstance(response, ComparisonResponse)
    assert client.request_count == 1
    assert client.last_usage is not None
    assert client.last_usage["model"] == "openai/gpt-oss-20b"
    assert client.last_usage["prompt_tokens"] == 180
    assert client.last_usage["completion_tokens"] == 90
    assert client.last_usage["total_tokens"] == 270
    assert "super-secret-key-123" not in str(client.last_usage)


def test_dosage_change_supported_and_unsupported_rejected():
    """Test 20: Regression test proving dosage_change is a supported DifferenceType and unsupported values are rejected."""
    # 1. Verify dosage_change is an official enum value
    assert DifferenceType.DOSAGE_CHANGE.value == "dosage_change"
    assert "dosage_change" in [d.value for d in DifferenceType]

    # 2. Verify ComparisonResponse validates dosage_change successfully
    resp = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily.",
        specific_difference="Dose changed from 500mg daily to 1000mg twice daily.",
        difference_type=DifferenceType.DOSAGE_CHANGE,
        confidence=0.95,
        rationale="Supported dosage change difference.",
    )
    assert resp.difference_type == DifferenceType.DOSAGE_CHANGE

    # Also string literal assignment succeeds via Pydantic coercion
    resp_str = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Metformin Monotherapy Starting Dose",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="SEC-2",
        exact_protocol_text="Start metformin 500mg once daily.",
        specific_difference="Dose changed from 500mg daily to 1000mg twice daily.",
        difference_type="dosage_change",
        confidence=0.95,
        rationale="Supported dosage change difference.",
    )
    assert resp_str.difference_type == DifferenceType.DOSAGE_CHANGE

    # 3. Verify Pydantic rejects an unsupported difference type
    with pytest.raises(ValidationError):
        ComparisonResponse(
            comparison_result=ComparisonResult.GAP,
            matched_protocol_section="Metformin Monotherapy Starting Dose",
            protocol_id="PROT-DM-001",
            protocol_version="1.0",
            section_id="SEC-2",
            exact_protocol_text="Start metformin 500mg once daily.",
            specific_difference="Dose changed from 500mg daily to 1000mg twice daily.",
            difference_type="unsupported_invented_type",
            confidence=0.95,
            rationale="Invalid type.",
        )


def test_all_difference_types_consistent_across_schema_and_prompt():
    """Test 21: Verify all 15 DifferenceType values are consistently defined and accepted."""
    from app.services.llm_client import COMPARISON_SYSTEM_PROMPT

    expected_types = {
        "new_recommendation",
        "conflict",
        "dosage_change",
        "scope_expansion",
        "population_expansion",
        "population_restriction",
        "threshold_change",
        "intervention_change",
        "contraindication",
        "monitoring_change",
        "frequency_change",
        "no_material_difference",
        "no_match",
        "other_supported_change",
        "none",
    }

    actual_types = {d.value for d in DifferenceType}
    assert actual_types == expected_types

    # Verify each difference type is listed in COMPARISON_SYSTEM_PROMPT
    for dt_val in expected_types:
        assert f"'{dt_val}'" in COMPARISON_SYSTEM_PROMPT

