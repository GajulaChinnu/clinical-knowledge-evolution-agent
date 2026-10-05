"""Integration tests for end-to-end CKEA orchestration lifecycle.

Demonstrates:
1. Full normal successful lifecycle:
   document -> monitoring -> extraction -> comparison -> impact -> briefing -> governance assignment -> review -> explicit approve -> close
2. Blocked lifecycle with G1 human gate hold and resume:
   document -> extraction (low confidence) -> G1 hold -> human resolution -> resume -> comparison -> impact -> briefing -> governance
3. Human decision is never faked automatically; human operations are explicitly called.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from sqlalchemy import create_engine

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
from app.schemas.comparison import ComparisonResponse, ComparisonResult, DifferenceType
from app.schemas.documents import DocumentStatus
from app.schemas.extraction import ExtractedRecommendation, ExtractionResponse
from app.schemas.gaps import GapStatus
from app.schemas.governance import ReviewDecision
from app.schemas.orchestration import (
    HumanGate,
    PipelineStage,
    PipelineStatus,
)
from app.schemas.protocol import CandidateProtocolSection
from app.services.config_service import AppConfig
from app.services.pdf_parser import PageText


@pytest.fixture(autouse=True)
def mock_pdf_text_extraction(monkeypatch):
    """Bypass binary PDF reading during tests."""
    def fake_extract_page_texts(pdf_path):
        return [
            PageText(
                page_number=1,
                text=(
                    "SECTION 4: RECOMMENDATIONS\n"
                    "Adults with type 2 diabetes and CKD should receive an SGLT2 inhibitor.\n"
                    "Consider mineralocorticoid receptor antagonists in selected patients."
                ),
            )
        ]
    monkeypatch.setattr("app.agents.extraction_agent.extract_page_texts", fake_extract_page_texts)


@pytest.fixture
def lifecycle_env(tmp_path: Path):
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

    mock_llm = MagicMock()
    mock_protocol_service = MagicMock()

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
                section="Recommendations",
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

    return {
        "engine": engine,
        "session_factory": session_factory,
        "config": config,
        "pipeline": pipeline,
        "mock_llm": mock_llm,
        "mock_protocol_service": mock_protocol_service,
        "governance_agent": governance_agent,
    }


def test_full_normal_lifecycle_integration(lifecycle_env, tmp_path):
    """Demonstrate a complete, realistic lifecycle from document to closure."""
    session_factory = lifecycle_env["session_factory"]
    pipeline = lifecycle_env["pipeline"]
    governance_agent = lifecycle_env["governance_agent"]

    # 1. Create source document in DB
    dummy_pdf = tmp_path / "KDIGO-Diabetes-CKD-2026.pdf"
    dummy_pdf.write_text("dummy")

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="KDIGO-Diabetes-CKD-2026.pdf",
            source_path=str(dummy_pdf),
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            status=DocumentStatus.DISCOVERED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    # 2. Execute pipeline through document ingestion
    result = pipeline.process_ingested_document(doc_id)

    # Verify pipeline completed up to governance assignment and held at G4
    assert result.status == PipelineStatus.HELD
    assert result.held_gate == HumanGate.G4
    assert result.current_stage == PipelineStage.GOVERNANCE
    assert PipelineStage.EXTRACTION in result.completed_stages
    assert PipelineStage.COMPARISON in result.completed_stages
    assert PipelineStage.IMPACT in result.completed_stages
    assert PipelineStage.BRIEFING in result.completed_stages
    assert PipelineStage.GOVERNANCE in result.completed_stages

    # Verify created artifacts in DB
    brief_id = result.artifacts.brief_id
    assert brief_id is not None
    assert result.artifacts.change_record_id is not None
    assert result.artifacts.gap_record_id is not None
    assert result.artifacts.impact_record_id is not None
    assert result.artifacts.assignment_id is not None

    with session_factory() as session:
        brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert brief is not None
        assert brief.status == BriefStatus.ASSIGNED.value

        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment is not None
        assert assignment.reviewer_id == "dr_smith"
        assert assignment.status == "assigned"

        impact = session.query(ImpactRecord).filter_by(id=result.artifacts.impact_record_id).first()
        assert impact is not None
        assert impact.total_score is not None

    # 3. Explicit human reviewer action: Start review
    brief = governance_agent.start_review(change_brief_id=brief_id, reviewer_id="dr_smith")
    assert brief.status == BriefStatus.IN_REVIEW.value

    # 4. Explicit human decision: Approve
    brief = governance_agent.decide(
        change_brief_id=brief_id,
        reviewer_id="dr_smith",
        decision=ReviewDecision.APPROVE,
        rationale="Strong cardiorenal protection evidence validated against active CKD pathway.",
    )
    assert brief.status == BriefStatus.DECIDED.value
    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment.decision == ReviewDecision.APPROVE.value

    # 5. Formal closure after decision via pipeline
    close_res = pipeline.process_brief(
        change_brief_id=brief_id,
        close_if_decided=True,
        actor="governance_chair",
        closure_note="Approved protocol update scheduled for next guideline release cycle.",
    )
    assert close_res.is_completed is True
    assert close_res.current_stage == PipelineStage.CLOSURE

    with session_factory() as session:
        final_brief = session.query(ChangeBrief).filter_by(id=brief_id).first()
        assert final_brief.status == BriefStatus.CLOSED.value

    # 6. Verify audit logs record pipeline and governance events
    with session_factory() as session:
        audits = session.query(AuditLog).all()
        assert len(audits) >= 5
        pipeline_audits = [a for a in audits if a.actor == "pipeline_orchestrator"]
        assert len(pipeline_audits) >= 1


def test_blocked_g1_lifecycle_and_resume_integration(lifecycle_env, tmp_path):
    """Demonstrate a blocked lifecycle when extraction produces low confidence (G1 hold),

    followed by explicit human resolution and pipeline resume.
    """
    session_factory = lifecycle_env["session_factory"]
    pipeline = lifecycle_env["pipeline"]
    mock_llm = lifecycle_env["mock_llm"]
    governance_agent = lifecycle_env["governance_agent"]

    # Configure mock LLM to return low confidence (below 0.70 threshold)
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[
            ExtractedRecommendation(
                verbatim_text="Consider mineralocorticoid receptor antagonists in selected patients.",
                recommendation_type="pharmacotherapy",
                target_population="Unclear diabetic population",
                intervention="MRA therapy",
                evidence_grade="Grade C",
                confidence=0.55,  # Below G1 threshold
                page=1,
                section="Recommendations",
                source_excerpt="Consider mineralocorticoid receptor antagonists in selected patients.",
            )
        ]
    )

    dummy_pdf = tmp_path / "Ambiguous-Guideline-2026.pdf"
    dummy_pdf.write_text("dummy")

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="Ambiguous-Guideline-2026.pdf",
            source_path=str(dummy_pdf),
            sha256_hash="a" * 64,
            status=DocumentStatus.DISCOVERED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    # 1. Process document -> must STOP at G1
    result = pipeline.process_ingested_document(doc_id)

    assert result.status == PipelineStatus.HELD
    assert result.blocked_gate == HumanGate.G1
    assert result.current_stage == PipelineStage.EXTRACTION
    change_id = result.artifacts.change_record_id
    assert change_id is not None

    with session_factory() as session:
        change = session.query(ChangeRecord).filter_by(id=change_id).first()
        assert change.status == "held_for_G1"
        # Ensure no downstream gap or brief was created prematurely
        assert session.query(GapRecord).count() == 0
        assert session.query(ChangeBrief).count() == 0

    # 2. Simulate human resolution of G1
    resume_result = pipeline.resume_held_work(
        entity_type="ChangeRecord",
        entity_id=change_id,
        resolution_data={
            "action": "proceed",
            "reviewer": "dr_smith",
            "rationale": "Clinician verified recommendation text and clinical applicability despite low confidence.",
        },
    )

    # 3. Pipeline resumed and completed downstream stages up to G4
    assert resume_result.status == PipelineStatus.HELD
    assert resume_result.held_gate == HumanGate.G4
    assert resume_result.current_stage == PipelineStage.GOVERNANCE
    assert PipelineStage.COMPARISON in resume_result.completed_stages
    assert PipelineStage.IMPACT in resume_result.completed_stages
    assert PipelineStage.BRIEFING in resume_result.completed_stages
    assert PipelineStage.GOVERNANCE in resume_result.completed_stages

    # 4. Verify brief is assigned and can be reviewed normally
    brief_id = resume_result.artifacts.brief_id
    assert brief_id is not None

    governance_agent.start_review(brief_id, "dr_smith")
    brief = governance_agent.decide(
        brief_id,
        "dr_smith",
        ReviewDecision.APPROVE,
        rationale="Human-in-the-loop approved following G1 verification.",
    )
    assert brief.status == BriefStatus.DECIDED.value

    close_res = pipeline.process_brief(
        change_brief_id=brief_id,
        close_if_decided=True,
        actor="governance_chair",
    )
    assert close_res.is_completed is True
    assert close_res.current_stage == PipelineStage.CLOSURE
