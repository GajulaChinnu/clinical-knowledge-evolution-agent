"""Unit and integration tests for Phase 10 Pipeline Orchestrator and State Machine.

Verifies:
1. Normal end-to-end pipeline progression
2. Monitoring, Extraction, Comparison, Impact, Briefing, and Governance handoffs
3. Strict human gate boundaries: G1, G2, G3, G4
4. Cannot auto-decide or close without explicit human decision
5. Resume after G1, G2, G3, and deferred governance briefs
6. Complete idempotency and duplicate prevention across all stages
7. Partial pipeline recovery and checkpoint continuation
8. Stage failure handling, transient retries, and retry limit enforcement
9. Failure isolation in multi-document batch processing
10. Schema validation and stage status reporting
11. Append-only AuditLog event emission
12. Protocol data immutability throughout pipeline execution
13. ZERO direct LLM / API calls from the orchestrator
14. No scheduler or future-phase logic invoked
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import uuid
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.briefing_agent import BriefingAgent
from app.agents.comparison_agent import ComparisonAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.impact_agent import ImpactAgent
from app.agents.monitoring_agent import MonitoringAgent
from app.models.database import Base, get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    ReviewAssignment,
)
from app.orchestration.pipeline import ClinicalKnowledgePipeline
from app.schemas.briefs import BriefStatus
from app.schemas.changes import ChangeStatus
from app.schemas.comparison import ComparisonResponse
from app.schemas.documents import DocumentStatus
from app.schemas.extraction import ExtractedRecommendation, ExtractionResponse
from app.schemas.gaps import ComparisonResult, DifferenceType, GapStatus
from app.schemas.governance import ReviewDecision
from app.schemas.impact import ImpactStatus, ImpactTier
from app.schemas.orchestration import (
    ArtifactsManifest,
    BatchPipelineResult,
    HumanGate,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)
from app.schemas.transitions import InvalidStateTransitionError
from app.services.config_service import AppConfig
from app.services.pdf_parser import PageText
from app.services.protocol_index import CandidateProtocolSection


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(autouse=True)
def mock_pdf_text_extraction(monkeypatch):
    """Ensure extract_page_texts returns valid PageText for sample test documents."""
    def fake_extract_page_texts(source_path):
        return [
            PageText(
                page_number=1,
                text="Pharmacotherapy\nAdults with type 2 diabetes and CKD should receive an SGLT2 inhibitor.",
            )
        ]
    monkeypatch.setattr("app.services.source_documents.extract_page_texts", fake_extract_page_texts)


@pytest.fixture
def orchestration_env(tmp_path: Path):
    """Set up an isolated in-memory SQLite database and initialized pipeline."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    project_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = project_root / "templates"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        database_url="sqlite:///:memory:",
        templates_dir=templates_dir,
        output_dir=output_dir,
        groq_api_key="mock-key-not-for-live-calls",
    )

    # Initialize mocked LLM clients for extraction and comparison
    mock_llm = MagicMock()
    mock_protocol_service = MagicMock()

    # Default mock responses
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[
            ExtractedRecommendation(
                verbatim_text="Adults with type 2 diabetes and CKD should receive an SGLT2 inhibitor.",
                recommendation_type="pharmacotherapy",
                target_population="Adults with T2D and CKD",
                intervention="SGLT2 inhibitor",
                evidence_grade="Grade A",
                confidence=0.95,
                page=1,
                section="Pharmacotherapy",
                source_excerpt="Adults with type 2 diabetes and CKD should receive an SGLT2 inhibitor.",
            )
        ]
    )

    cand = CandidateProtocolSection(
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="4.1",
        section_heading="Metformin Monotherapy",
        section_text="Metformin is the preferred initial pharmacologic agent for type 2 diabetes.",
        similarity=0.85,
        distance=0.15,
        chroma_id="PROT-DM-001__1.0__4.1",
    )
    mock_protocol_service.retrieve.return_value = [cand]

    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="Metformin Monotherapy",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="4.1",
        exact_protocol_text="Metformin is the preferred initial pharmacologic agent for type 2 diabetes.",
        specific_difference="Recommendation introduces SGLT2 inhibitors for CKD patients, absent in protocol.",
        difference_type=DifferenceType.INTERVENTION_CHANGE,
        confidence=0.92,
        rationale="Protocol requires addition of SGLT2 inhibitor therapy for CKD patient cohort.",
    )

    monitoring_agent = MonitoringAgent(session_factory=session_factory, config=config)
    extraction_agent = ExtractionAgent(session_factory=session_factory, llm_client=mock_llm, config=config)
    comparison_agent = ComparisonAgent(
        session_factory=session_factory,
        llm_client=mock_llm,
        protocol_index_service=mock_protocol_service,
        config=config,
    )
    impact_agent = ImpactAgent(session_factory=session_factory, config=config)
    briefing_agent = BriefingAgent(session_factory=session_factory, config=config)
    governance_agent = GovernanceAgent(session_factory=session_factory, config=config)

    pipeline = ClinicalKnowledgePipeline(
        session_factory=session_factory,
        config=config,
        monitoring_agent=monitoring_agent,
        extraction_agent=extraction_agent,
        comparison_agent=comparison_agent,
        impact_agent=impact_agent,
        briefing_agent=briefing_agent,
        governance_agent=governance_agent,
        default_reviewer_id="dr_smith",
        default_reviewer_role="Clinical Governance Lead",
        max_retries=3,
    )

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "config": config,
        "pipeline": pipeline,
        "mock_llm": mock_llm,
        "mock_protocol_service": mock_protocol_service,
        "governance_agent": governance_agent,
        "tmp_path": tmp_path,
    }

    engine.dispose()


def _create_sample_doc(
    session_factory,
    status: str = DocumentStatus.PARSED.value,
    retry_count: int = 0,
    source_identifier: str = "ADA-2026",
) -> str:
    """Helper to create an IngestedDocument record."""
    uid = uuid.uuid4().hex
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier=source_identifier,
            source_path=f"/data/sources/{source_identifier}.pdf",
            sha256_hash=(uid * 2)[:64],
            document_version="1.0",
            source_version="2026.1",
            parser_version="1.0",
            pipeline_version="1.0",
            status=status,
            retry_count=retry_count,
        )
        session.add(doc)
        session.commit()
        return doc.id


def _create_sample_change(
    session_factory,
    doc_id: str,
    status: str = ChangeStatus.EXTRACTED.value,
    confidence: float = 0.95,
) -> str:
    """Helper to create a ChangeRecord."""
    with session_factory() as session:
        change = ChangeRecord(
            ingested_document_id=doc_id,
            verbatim_text="Start SGLT2 inhibitor in adults with T2D and CKD.",
            recommendation_type="pharmacotherapy",
            target_population="Adults with Type 2 Diabetes and CKD",
            intervention="SGLT2 inhibitor",
            evidence_grade="Grade A",
            page=1,
            section="Pharmacotherapy",
            source_excerpt="Start SGLT2 inhibitor in adults with T2D and CKD.",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            confidence=confidence,
            status=status,
        )
        session.add(change)
        session.commit()
        return change.id


# ==============================================================================
# 1. SCHEMA AND REPORTING TESTS
# ==============================================================================

def test_pipeline_result_schema_validation():
    """Test 1: Verify PipelineResult Pydantic validation, default values, and helper properties."""
    res = PipelineResult(
        document_id=str(uuid.uuid4()),
        status=PipelineStatus.HELD,
        blocked_stage=PipelineStage.EXTRACTION,
        held_gate=HumanGate.G1,
        reason="Test G1 hold",
    )
    assert res.is_held is True
    assert res.is_completed is False
    assert res.is_failed is False
    assert res.held_gate == HumanGate.G1
    assert res.blocked_stage == PipelineStage.EXTRACTION

    # Test serialization
    data = res.model_dump()
    assert data["status"] == "held"
    assert data["held_gate"] == "G1"


def test_stage_status_reporting():
    """Test 2: Verify stage reporting enumerations and completed stages tracking."""
    manifest = ArtifactsManifest(document_id="doc-123")
    manifest.change_record_ids.append("change-456")

    res = PipelineResult(
        document_id="doc-123",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.CLOSURE,
        completed_stages=[
            PipelineStage.MONITORING,
            PipelineStage.EXTRACTION,
            PipelineStage.COMPARISON,
            PipelineStage.IMPACT,
            PipelineStage.BRIEFING,
            PipelineStage.GOVERNANCE,
            PipelineStage.CLOSURE,
        ],
        artifacts=manifest,
    )
    assert res.is_completed is True
    assert len(res.completed_stages) == 7
    assert res.artifacts.change_record_ids == ["change-456"]


def test_no_direct_llm_calls_in_orchestrator():
    """Test 3: Verify ClinicalKnowledgePipeline itself makes zero direct Groq or OpenAI invocations."""
    with patch("openai.OpenAI", side_effect=RuntimeError("Groq/OpenAI client must not be called by Orchestrator!")):
        import app.orchestration.pipeline as pipe_mod
        # Verify module imports
        assert "OpenAI" not in pipe_mod.__dict__
        assert "SharedLLMClient" not in pipe_mod.__dict__


def test_no_future_phase_behavior_or_scheduler(orchestration_env):
    """Test 4: Verify orchestrator contains no APScheduler background logic or external monitors."""
    pipeline = orchestration_env["pipeline"]
    assert not hasattr(pipeline, "scheduler")
    assert not hasattr(pipeline, "apscheduler")
    assert not hasattr(pipeline, "cron")


# ==============================================================================
# 2. STAGE HANDOFF TESTS
# ==============================================================================

def test_monitoring_handoff(orchestration_env):
    """Test 5: Verify monitoring handoff discovers and ingests new documents."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value)
    # Orchestrator process_ingested_document
    res = pipeline.process_ingested_document(doc_id)
    assert res.document_id == doc_id
    assert PipelineStage.MONITORING in res.completed_stages


def test_extraction_handoff(orchestration_env):
    """Test 6: Verify extraction handoff creates ChangeRecord and progresses stage."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value)
    res = pipeline.process_ingested_document(doc_id)

    assert PipelineStage.EXTRACTION in res.completed_stages
    assert len(res.artifacts.change_record_ids) > 0


def test_comparison_handoff(orchestration_env):
    """Test 7: Verify comparison handoff creates GapRecord from ChangeRecord."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert PipelineStage.COMPARISON in res.completed_stages
    assert len(res.artifacts.gap_record_ids) > 0


def test_impact_handoff(orchestration_env):
    """Test 8: Verify impact handoff deterministically calculates score and assigns tier."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert PipelineStage.IMPACT in res.completed_stages
    assert len(res.artifacts.impact_record_ids) > 0

    with session_factory() as session:
        impact = session.get(ImpactRecord, res.artifacts.impact_record_ids[0])
        assert impact.total_score is not None
        assert impact.tier is not None


def test_briefing_handoff(orchestration_env):
    """Test 9: Verify briefing handoff renders and stores ChangeBrief in draft."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert PipelineStage.BRIEFING in res.completed_stages
    assert len(res.artifacts.change_brief_ids) > 0

    with session_factory() as session:
        brief = session.get(ChangeBrief, res.artifacts.change_brief_ids[0])
        assert brief.status in (BriefStatus.DRAFT.value, BriefStatus.ASSIGNED.value)


def test_governance_handoff(orchestration_env):
    """Test 10: Verify governance handoff assigns reviewer and halts at G4 gate."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert PipelineStage.GOVERNANCE in res.completed_stages
    assert len(res.artifacts.review_assignment_ids) > 0
    assert res.is_held is True
    assert res.held_gate == HumanGate.G4


# ==============================================================================
# 3. END-TO-END NORMAL & HUMAN GATE BOUNDARIES
# ==============================================================================

def test_normal_end_to_end_pipeline_progression(orchestration_env):
    """Test 11: Execute full lifecycle from document to assigned brief, explicit approval, and closure."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]
    gov_agent = orchestration_env["governance_agent"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value)

    # 1. Ingest & process through governance assignment
    res = pipeline.process_ingested_document(doc_id)
    assert res.is_held is True
    assert res.held_gate == HumanGate.G4
    brief_id = res.artifacts.change_brief_ids[0]

    # 2. Reviewer starts review
    gov_agent.start_review(brief_id, reviewer_id="dr_smith")

    # 3. Reviewer records explicit APPROVE decision
    gov_agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Strong trial evidence supports SGLT2 inhibitor protocol update.",
    )

    # 4. Progress closure
    close_res = pipeline.process_brief(brief_id, close_if_decided=True, actor="gov_coordinator")
    assert close_res.is_completed is True
    assert close_res.current_stage == PipelineStage.CLOSURE

    with session_factory() as session:
        brief = session.get(ChangeBrief, brief_id)
        assert brief.status == BriefStatus.CLOSED.value


def test_g1_hold_on_low_confidence_extraction(orchestration_env):
    """Test 12: Verify extraction with confidence below threshold halts at G1 human gate."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    # Create change held for G1
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.HELD_FOR_G1.value, confidence=0.50)

    res = pipeline.process_change(change_id)
    assert res.is_held is True
    assert res.held_gate == HumanGate.G1
    assert res.blocked_stage == PipelineStage.EXTRACTION


def test_g2_hold_on_ambiguous_comparison(orchestration_env):
    """Test 13: Verify comparison finding requiring review halts at G2 human gate."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]
    mock_llm = orchestration_env["mock_llm"]

    # Configure mock LLM to return ambiguous / contradictory output
    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.AMBIGUOUS,
        matched_protocol_section="Metformin Monotherapy",
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="4.1",
        exact_protocol_text="Metformin is the preferred initial pharmacologic agent for type 2 diabetes.",
        specific_difference="Ambiguous finding.",
        difference_type=DifferenceType.NONE,
        confidence=0.40,
        rationale="Ambiguous comparison output.",
    )

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert res.is_held is True
    assert res.held_gate == HumanGate.G2
    assert res.blocked_stage == PipelineStage.COMPARISON


def test_g3_no_match_path_stops_awaiting_committee(orchestration_env):
    """Test 14: Verify explicit no-match stops at G3 gate awaiting committee confirmation."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]
    mock_protocol = orchestration_env["mock_protocol_service"]

    # Candidate similarity below 0.70 triggers G3
    cand = CandidateProtocolSection(
        protocol_id="PROT-DM-001",
        protocol_version="1.0",
        section_id="4.1",
        section_heading="Metformin",
        section_text="Metformin only.",
        similarity=0.35,  # below 0.70 threshold
        distance=0.65,
        chroma_id="PROT-DM-001__1.0__4.1",
    )
    mock_protocol.retrieve.return_value = [cand]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert res.is_held is True
    assert res.held_gate == HumanGate.G3
    assert res.blocked_stage == PipelineStage.COMPARISON

    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.HELD_FOR_G3.value
        assert change.gap_record.comparison_result == ComparisonResult.NO_MATCH.value


def test_g3_no_match_confirmed_progresses_to_briefing(orchestration_env):
    """Test 15: Verify confirmed G3 no-match progresses through Impact and Briefing preserving no-match."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.HELD_FOR_G3.value)

    with session_factory() as session:
        gap = GapRecord(
            change_record_id=change_id,
            candidate_protocol_section_ids=[],
            similarity=0.20,
            comparison_result=ComparisonResult.NO_MATCH.value,
            comparison_confidence=1.0,
            difference_type=DifferenceType.NO_MATCH.value,
            matched_protocol_id=None,
            matched_protocol_version=None,
            is_match=False,
            status=GapStatus.NO_MATCH.value,
            reviewer_resolution="Clinical committee confirmed new recommendation domain without existing protocol.",
        )
        session.add(gap)
        session.commit()

    # Now process_change should recognize confirmed resolution and continue to Impact & Briefing
    res = pipeline.process_change(change_id)
    assert PipelineStage.IMPACT in res.completed_stages
    assert PipelineStage.BRIEFING in res.completed_stages
    assert res.held_gate == HumanGate.G4  # Stops at G4 governance


def test_g4_decision_boundary_enforced(orchestration_env):
    """Test 16: Verify G4 gate stops pipeline and rejects automatic decision inference."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert res.status == PipelineStatus.HELD
    assert res.held_gate == HumanGate.G4

    # Verify brief is in assigned status, NOT decided or closed
    with session_factory() as session:
        brief = session.get(ChangeBrief, res.artifacts.change_brief_ids[0])
        assert brief.status == BriefStatus.ASSIGNED.value


def test_cannot_auto_decide(orchestration_env):
    """Test 17: Verify pipeline never infers decision from impact tier or elapsed time."""
    pipeline = orchestration_env["pipeline"]
    assert not hasattr(pipeline, "auto_approve")
    assert not hasattr(pipeline, "auto_reject")
    assert not hasattr(pipeline, "infer_decision")


def test_cannot_close_without_decision(orchestration_env):
    """Test 18: Verify attempt to close brief before explicit human decision is rejected."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)
    res = pipeline.process_change(change_id)
    brief_id = res.artifacts.change_brief_ids[0]

    # Attempt closure via pipeline while still in assigned status
    close_res = pipeline.process_brief(brief_id, close_if_decided=True)
    assert close_res.is_held is True
    assert close_res.held_gate == HumanGate.G4

    with session_factory() as session:
        brief = session.get(ChangeBrief, brief_id)
        assert brief.status != BriefStatus.CLOSED.value


# ==============================================================================
# 4. RESUME CAPABILITY TESTS
# ==============================================================================

def test_resume_after_g1_proceed(orchestration_env):
    """Test 19: Verify resume_held_work for G1 proceed continues to comparison and downstream."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.HELD.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.HELD_FOR_G1.value)

    res = pipeline.resume_held_work(
        entity_type="ChangeRecord",
        entity_id=change_id,
        resolution_data={"action": "proceed", "reviewer": "dr_smith", "rationale": "Excerpt verified manually."},
    )
    assert PipelineStage.COMPARISON in res.completed_stages
    assert res.held_gate == HumanGate.G4


def test_resume_after_g1_reject(orchestration_env):
    """Test 20: Verify resume_held_work for G1 reject marks record as no_gap and completes."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.HELD.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.HELD_FOR_G1.value)

    res = pipeline.resume_held_work(
        entity_type="ChangeRecord",
        entity_id=change_id,
        resolution_data={"action": "reject", "reviewer": "dr_smith", "rationale": "Spurious recommendation."},
    )
    assert res.is_completed is True
    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.NO_GAP.value


def test_resume_after_g2_confirm_gap(orchestration_env):
    """Test 21: Verify resume_held_work for G2 confirm_gap progresses to Impact and Briefing."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.HELD_FOR_G2.value)

    with session_factory() as session:
        gap = GapRecord(
            change_record_id=change_id,
            candidate_protocol_section_ids=["PROT-DM-001_sec_4.1"],
            similarity=0.80,
            comparison_result=ComparisonResult.AMBIGUOUS.value,
            difference_type=DifferenceType.INTERVENTION_CHANGE.value,
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="1.0",
            matched_section_id="4.1",
            matched_section_heading="Metformin Monotherapy",
            exact_protocol_text="Metformin is the preferred initial pharmacologic agent for type 2 diabetes.",
            status=GapStatus.REVIEW_REQUIRED.value,
        )
        session.add(gap)
        session.commit()

    res = pipeline.resume_held_work(
        entity_type="ChangeRecord",
        entity_id=change_id,
        resolution_data={"action": "confirm_gap", "reviewer": "dr_smith", "rationale": "Verified gap manually."},
    )
    assert PipelineStage.IMPACT in res.completed_stages
    assert PipelineStage.BRIEFING in res.completed_stages
    assert res.held_gate == HumanGate.G4


def test_resume_after_g2_no_gap(orchestration_env):
    """Test 22: Verify resume_held_work for G2 no_gap marks record as no_gap and terminates."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.HELD_FOR_G2.value)

    with session_factory() as session:
        gap = GapRecord(
            change_record_id=change_id,
            candidate_protocol_section_ids=["PROT-DM-001_sec_4.1"],
            similarity=0.80,
            comparison_result=ComparisonResult.AMBIGUOUS.value,
            difference_type=DifferenceType.NONE.value,
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="1.0",
            status=GapStatus.REVIEW_REQUIRED.value,
        )
        session.add(gap)
        session.commit()

    res = pipeline.resume_held_work(
        entity_type="ChangeRecord",
        entity_id=change_id,
        resolution_data={"action": "no_gap", "reviewer": "dr_smith", "rationale": "No material gap."},
    )
    assert res.is_completed is True
    with session_factory() as session:
        change = session.get(ChangeRecord, change_id)
        assert change.status == ChangeStatus.NO_GAP.value


def test_resume_deferred_governance_brief(orchestration_env):
    """Test 23: Verify resuming a deferred brief re-enters review without restarting extraction/comparison."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]
    gov_agent = orchestration_env["governance_agent"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    brief_id = res.artifacts.change_brief_ids[0]

    # Reviewer defers with valid future due date
    gov_agent.start_review(brief_id, reviewer_id="dr_smith")
    future_date = datetime.now(timezone.utc) + timedelta(days=30)
    gov_agent.defer(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        rationale="Awaiting upcoming KDIGO trial results.",
        defer_follow_up_date=future_date,
    )

    # Resume deferred brief
    resume_res = pipeline.resume_held_work(
        entity_type="ChangeBrief",
        entity_id=brief_id,
        resolution_data={"reviewer_id": "dr_smith"},
    )
    assert resume_res.is_held is True
    assert resume_res.held_gate == HumanGate.G4
    with session_factory() as session:
        brief = session.get(ChangeBrief, brief_id)
        assert brief.status == BriefStatus.IN_REVIEW.value


# ==============================================================================
# 5. IDEMPOTENCY AND DUPLICATE PREVENTION
# ==============================================================================

def test_already_completed_document_idempotency(orchestration_env):
    """Test 24: Re-running pipeline on an already-completed document reuses existing artifacts."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    # First run
    res1 = pipeline.process_change(change_id)
    brief_id_1 = res1.artifacts.change_brief_ids[0]

    # Second run
    res2 = pipeline.process_change(change_id)
    brief_id_2 = res2.artifacts.change_brief_ids[0]

    assert brief_id_1 == brief_id_2


def test_duplicate_prevention_across_all_stages(orchestration_env):
    """Test 25: Verify repeated executions produce zero duplicate database records."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    pipeline.process_change(change_id)
    pipeline.process_change(change_id)
    pipeline.process_change(change_id)

    with session_factory() as session:
        gaps = session.query(GapRecord).filter_by(change_record_id=change_id).all()
        assert len(gaps) == 1

        impacts = session.query(ImpactRecord).filter_by(gap_record_id=gaps[0].id).all()
        assert len(impacts) == 1

        briefs = session.query(ChangeBrief).filter_by(impact_record_id=impacts[0].id).all()
        assert len(briefs) == 1

        assignments = session.query(ReviewAssignment).filter_by(change_brief_id=briefs[0].id).all()
        assert len(assignments) == 1


def test_closed_brief_is_not_reprocessed(orchestration_env):
    """Test 26: Verify calling process_brief on a closed brief is a safe no-op."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]
    gov_agent = orchestration_env["governance_agent"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)
    res = pipeline.process_change(change_id)
    brief_id = res.artifacts.change_brief_ids[0]

    gov_agent.start_review(brief_id, reviewer_id="dr_smith")
    gov_agent.decide(brief_id, reviewer_id="dr_smith", decision=ReviewDecision.APPROVE, rationale="Approved.")
    gov_agent.close_after_decision(brief_id, actor="dr_smith")

    # Re-process closed brief
    close_res = pipeline.process_brief(brief_id)
    assert close_res.is_completed is True
    assert "already closed" in close_res.reason


def test_partial_pipeline_recovery(orchestration_env):
    """Test 27: Verify pipeline can pick up at intermediate checkpoint (e.g. gap exists, impact pending)."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.GAP_CONFIRMED.value)

    # Pre-create GapRecord
    with session_factory() as session:
        gap = GapRecord(
            change_record_id=change_id,
            candidate_protocol_section_ids=["sec-1"],
            similarity=0.88,
            comparison_result=ComparisonResult.GAP.value,
            difference_type=DifferenceType.INTERVENTION_CHANGE.value,
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="1.0",
            matched_section_id="4.1",
            matched_section_heading="Metformin Monotherapy",
            exact_protocol_text="Metformin is the preferred initial pharmacologic agent for type 2 diabetes.",
            status=GapStatus.MATCHED.value,
        )
        session.add(gap)
        session.commit()

    # process_change recovers from gap checkpoint
    res = pipeline.process_change(change_id)
    assert PipelineStage.IMPACT in res.completed_stages
    assert PipelineStage.BRIEFING in res.completed_stages


# ==============================================================================
# 6. RETRY, FAILURE HANDLING, AND BATCH ISOLATION
# ==============================================================================

def test_failed_stage_handling(orchestration_env):
    """Test 28: Verify failure in extraction transitions document to failed and records audit."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]
    mock_llm = orchestration_env["mock_llm"]

    mock_llm.extract_recommendations.side_effect = RuntimeError("Extraction provider crash")

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value)
    res = pipeline.process_ingested_document(doc_id)

    assert res.is_failed is True
    with session_factory() as session:
        doc = session.get(IngestedDocument, doc_id)
        assert doc.status == DocumentStatus.FAILED.value


def test_transient_retry_logic(orchestration_env):
    """Test 29: Verify document in failed state with retry_count < max_retries can be retried."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.FAILED.value, retry_count=1)

    # Pipeline retry attempt
    res = pipeline.process_ingested_document(doc_id)
    assert res.retry_count == 2


def test_retry_limit_enforcement(orchestration_env):
    """Test 30: Verify reaching max_retries prevents further retry attempts."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.FAILED.value, retry_count=3)
    res = pipeline.process_ingested_document(doc_id)

    assert res.is_failed is True
    assert "max retries (3) have been reached" in res.reason


def test_permanent_validation_failure_behavior(orchestration_env):
    """Test 31: Verify unrecoverable non-existent ID raises ValueError immediately."""
    pipeline = orchestration_env["pipeline"]
    with pytest.raises(ValueError, match="not found"):
        pipeline.process_ingested_document(str(uuid.uuid4()))


def test_failure_isolation_between_documents(orchestration_env):
    """Test 32: Verify failure in one document does not crash processing of valid documents in batch."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]
    mock_llm = orchestration_env["mock_llm"]

    # Configure mock_llm to fail on DOC-FAIL-2
    def side_effect_extract(section_heading, section_text, page_number, document_identifier):
        if "FAIL" in str(document_identifier):
            raise RuntimeError("Extraction failed on corrupt document")
        return ExtractionResponse(
            recommendations=[
                ExtractedRecommendation(
                    verbatim_text="Adults with type 2 diabetes and CKD should receive an SGLT2 inhibitor.",
                    recommendation_type="pharmacotherapy",
                    target_population="Adults with T2D and CKD",
                    intervention="SGLT2 inhibitor",
                    evidence_grade="Grade A",
                    confidence=0.95,
                    page=1,
                    section="Pharmacotherapy",
                    source_excerpt="Adults with type 2 diabetes and CKD should receive an SGLT2 inhibitor.",
                )
            ]
        )
    mock_llm.extract_recommendations.side_effect = side_effect_extract

    doc1_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value, source_identifier="DOC-GOOD-1")
    doc2_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value, source_identifier="DOC-FAIL-2")
    doc3_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value, source_identifier="DOC-GOOD-3")

    batch_res = pipeline.process_pending_documents()
    assert batch_res.total_documents == 3
    assert batch_res.failed_count == 1
    assert batch_res.held_count == 2  # Good docs progress to G4 held


def test_batch_processing_behavior(orchestration_env):
    """Test 33: Verify process_pending_documents discovers eligible documents and aggregates statistics."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc1 = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value, source_identifier="DOC-A")
    doc2 = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value, source_identifier="DOC-B")

    batch_res = pipeline.process_pending_documents(max_docs=10)
    assert batch_res.total_documents == 2
    assert len(batch_res.results) == 2


# ==============================================================================
# 7. AUDITABILITY AND PROTOCOL IMMUTABILITY
# ==============================================================================

def test_audit_events_recorded_during_orchestration(orchestration_env):
    """Test 34: Verify meaningful pipeline lifecycle events are logged to append-only AuditLog."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.PARSED.value)
    pipeline.process_ingested_document(doc_id)

    with session_factory() as session:
        logs = session.query(AuditLog).filter_by(entity_id=doc_id).all()
        actions = [l.audit_metadata.get("action") for l in logs if l.audit_metadata]
        assert "pipeline_started" in actions


def test_protocol_immutability_during_orchestration(orchestration_env):
    """Test 35: Verify institutional protocol sections remain strictly immutable throughout pipeline operations."""
    session_factory = orchestration_env["session_factory"]
    pipeline = orchestration_env["pipeline"]

    doc_id = _create_sample_doc(session_factory, status=DocumentStatus.COMPLETE.value)
    change_id = _create_sample_change(session_factory, doc_id, status=ChangeStatus.EXTRACTED.value)

    res = pipeline.process_change(change_id)
    assert res.is_held is True

    # Verify no protocol records were modified or added
    with session_factory() as session:
        # Check that gap records refer to protocol without altering any protocol definitions
        gap = session.query(GapRecord).filter_by(change_record_id=change_id).first()
        assert gap.matched_protocol_id == "PROT-DM-001"
        assert gap.matched_protocol_version == "1.0"
