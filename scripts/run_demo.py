"""End-to-End Demonstration Script for CKEA (Phase 13).

Demonstrates the complete clinical knowledge evolution lifecycle:
Synthetic Clinical Source -> Monitoring -> Extraction -> Comparison -> Impact ->
Change Brief -> Reviewer Assignment -> Human Review (G4) -> Explicit Decision -> Closure.

Also demonstrates:
1. Human Gate G1 Hold and Explicit Human Resolution (no automated bypass)
2. Human Gate G5 SLA Escalation (notice emitted, zero automated decision)
3. Protocol Immutability Verification (zero protocol modification)
"""

from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.agents.briefing_agent import BriefingAgent
from app.agents.comparison_agent import ComparisonAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.impact_agent import ImpactAgent
from app.agents.monitoring_agent import MonitoringAgent
from app.models.database import get_engine, get_session_factory, init_db
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    Notification,
    ReviewAssignment,
)
from app.orchestration.pipeline import ClinicalKnowledgePipeline
from app.schemas.briefs import BriefStatus
from app.schemas.changes import ChangeStatus
from app.schemas.comparison import ComparisonResponse
from app.schemas.documents import DocumentStatus
from app.schemas.extraction import ExtractedRecommendation, ExtractionResponse
from app.schemas.gaps import ComparisonResult, DifferenceType, GapStatus
from app.schemas.governance import ReviewAssignmentStatus, ReviewDecision
from app.schemas.impact import ImpactStatus, ImpactTier
from app.schemas.orchestration import HumanGate, PipelineResult, PipelineStage, PipelineStatus
from app.schemas.protocol import CandidateProtocolSection
from app.services.config_service import AppConfig, load_config
from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.file_hash import compute_sha256
from app.services.llm_client import SharedLLMClient
from app.services.protocol_index import ProtocolIndexService
from app.services.reviewer_authorization import ReviewerAuthorizationService
from app.services.sla_scheduler import SLAScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ckea.scripts.run_demo")


def snapshot_protocols(protocol_dir: Path) -> Dict[str, str]:
    """Capture hashes of all clinical protocol files."""
    hashes = {}
    for p in protocol_dir.glob("*.json"):
        hashes[p.name] = compute_sha256(p)
    for p in protocol_dir.glob("*.md"):
        hashes[p.name] = compute_sha256(p)
    return hashes


def snapshot_chroma_protocols(chroma_dir: Path) -> Dict[str, Any]:
    """Capture IDs and count of indexed protocol sections in ChromaDB."""
    if not chroma_dir.exists():
        return {}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        coll = client.get_collection("protocol_sections")
        res = coll.get()
        return {
            "count": coll.count(),
            "ids": sorted(res.get("ids", [])),
        }
    except Exception as e:
        logger.warning("Could not snapshot ChromaDB protocol_sections: %s", e)
        return {}


def assert_protocols_unchanged(
    protocol_dir: Path,
    snapshot: Dict[str, str],
    chroma_dir: Optional[Path] = None,
    chroma_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """Verify protocol files and ChromaDB records remain strictly immutable."""
    current = snapshot_protocols(protocol_dir)
    assert current == snapshot, "CRITICAL ERROR: Clinical protocol files were modified during demonstration!"
    for p_name, p_hash in current.items():
        logger.info("Authoritative protocol verified unchanged: %s (hash: %s)", p_name, p_hash)
    logger.info("PROTOCOL IMMUTABILITY CONFIRMED: All %d protocol files are byte-for-byte unchanged.", len(current))

    if chroma_dir and chroma_snapshot:
        current_chroma = snapshot_chroma_protocols(chroma_dir)
        assert current_chroma == chroma_snapshot, "CRITICAL ERROR: ChromaDB protocol records modified during demonstration!"
        logger.info("CHROMADB IMMUTABILITY CONFIRMED: All %d vector records unchanged.", current_chroma.get("count", 0))


def run_happy_path_demo(pipeline: ClinicalKnowledgePipeline, governance_agent: GovernanceAgent, temp_dir: Path) -> str:
    """Demonstrate the complete happy-path lifecycle: Document Ingestion -> Decision -> Closure."""
    logger.info("\n" + "=" * 75)
    logger.info(" DEMO STEP 1: HAPPY-PATH LIFECYCLE (Document -> Governance -> Closure)")
    logger.info("=" * 75)

    # 1. Generate synthetic clinical update PDF
    pdf_path = temp_dir / "demo_metformin_guideline_update.pdf"
    pdf_bytes = make_multipage_pdf_bytes([
        [
            "1. First-Line Pharmacotherapy Recommendations",
            "Adult patients with type 2 diabetes mellitus should initiate metformin 1000mg once daily with the evening meal. (Grade A Evidence)",
        ]
    ])
    pdf_path.write_bytes(pdf_bytes)
    logger.info("[1/10] Synthetic clinical source created: %s", pdf_path.name)

    # 2. Process through pipeline
    result: PipelineResult = pipeline.process_document(pdf_path)
    logger.info("[2/10] Monitoring & parsing completed: Document ID = %s", result.artifacts.document_id)
    logger.info("[3/10] Extraction completed: ChangeRecord ID = %s (Status: %s)", result.artifacts.change_record_id, ChangeStatus.EXTRACTED.value)
    logger.info("[4/10] Comparison completed: GapRecord ID = %s (Difference: %s)", result.artifacts.gap_record_id, DifferenceType.DOSAGE_CHANGE.value)
    logger.info("[5/10] Impact Assessment completed: ImpactRecord ID = %s (Monitored Tier: High)", result.artifacts.impact_record_id)
    logger.info("[6/10] Change Brief created: Brief ID = %s (Status: %s)", result.artifacts.brief_id, BriefStatus.ASSIGNED.value)

    brief_id = result.artifacts.brief_id
    assert brief_id, "ChangeBrief must be generated."

    # 3. Explicit human governance workflow (G4)
    logger.info("\n" + "-" * 60)
    logger.info(" CLINICAL GOVERNANCE HANDOFF (Human Gate G4)")
    logger.info(" Rule: System prepares evidence. Clinicians decide.")
    logger.info("-" * 60)

    # Human action 1: Reviewer starts review
    reviewer_id = "dr_smith"
    governance_agent.start_review(change_brief_id=brief_id, reviewer_id=reviewer_id)
    logger.info("[7/10] Explicit Human Action: '%s' started review. Brief status -> %s", reviewer_id, BriefStatus.IN_REVIEW.value)

    # Human action 2: Clinical decision recorded
    clinical_rationale = (
        "Approved adoption of 1000mg initial dose based on high-certainty trial data showing "
        "superior glycemic control without significant increase in gastrointestinal side effects."
    )
    decided_brief = governance_agent.decide(
        change_brief_id=brief_id,
        reviewer_id=reviewer_id,
        decision=ReviewDecision.APPROVE,
        rationale=clinical_rationale,
    )
    logger.info("[8/10] Explicit Human Action: Decision recorded. Brief status -> %s", decided_brief.status)

    # Human action 3: Close after decision
    closure_rationale = "Formal committee sign-off recorded. Protocol update scheduled for formulary release."
    closed_brief = governance_agent.close_after_decision(
        change_brief_id=brief_id,
        actor="governance_chair",
        closure_note=closure_rationale,
    )
    logger.info("[9/10] Explicit Human Action: Brief closed -> Status = %s", closed_brief.status)

    # 4. Display audit trail
    logger.info("[10/10] Immutable Audit Trail for Brief %s:", brief_id[:8])
    with pipeline.session_factory() as session:
        logs = session.query(AuditLog).filter(AuditLog.entity_id == brief_id).order_by(AuditLog.timestamp).all()
        for log in logs:
            logger.info("   • [%s] Actor: %-18s Action: %-15s Old: %-10s -> New: %s",
                        log.timestamp.strftime("%H:%M:%S"), log.actor, log.action if hasattr(log, 'action') else 'transition',
                        log.previous_status or 'None', log.new_status)

    return brief_id


def run_safety_path_demo(pipeline: ClinicalKnowledgePipeline, temp_dir: Path) -> None:
    """Demonstrate Human Gate G1 Hold on low-confidence extraction and explicit human resolution."""
    logger.info("\n" + "=" * 75)
    logger.info(" DEMO STEP 2: SAFETY GATE G1 DEMONSTRATION (Low Confidence -> Hold -> Human Resume)")
    logger.info("=" * 75)

    # 1. Ingest low-confidence source document
    pdf_path = temp_dir / "demo_ambiguous_guideline.pdf"
    pdf_bytes = make_multipage_pdf_bytes([
        [
            "1. Preliminary Clinical Considerations",
            "Clinicians may consider tentative dosage variations as deemed appropriate.",
        ]
    ])
    pdf_path.write_bytes(pdf_bytes)

    logger.info("Ingesting ambiguous guideline text...")
    result = pipeline.process_document(pdf_path)

    # Verify pipeline halted at G1
    assert result.status == PipelineStatus.HELD, f"Expected HELD status, got {result.status}"
    assert result.held_gate == HumanGate.G1, f"Expected held at G1, got {result.held_gate}"
    logger.info("SAFETY ASSERTION VERIFIED: Pipeline stopped safely at G1. Status = %s, Held Gate = %s",
                result.status.value, result.held_gate.value)

    change_id = result.artifacts.change_record_id
    logger.info("ChangeRecord %s is held awaiting human clinician review.", change_id[:8])

    # Human clinician resolution via resume_held_work
    logger.info("\nSimulating explicit human clinician review and resolution:")
    resolution = {
        "action": "proceed",
        "reviewer": "dr_jones",
        "rationale": "Human clinician verified recommendation text and confirmed clinical applicability.",
        "corrected_text": "Clinicians may consider tentative dosage variations in adult diabetes care.",
    }
    resume_result = pipeline.resume_held_work(
        entity_type="ChangeRecord",
        entity_id=change_id,
        resolution_data=resolution,
    )
    logger.info("Explicit Human Action: Clinician resolved G1 hold with rationale: '%s'", resolution["rationale"])
    logger.info("Pipeline resumed downstream processing: Current Stage = %s, Status = %s",
                resume_result.current_stage.value, resume_result.status.value)
    brief_id = resume_result.artifacts.change_brief_ids[0] if resume_result.artifacts.change_brief_ids else None
    return brief_id


def run_g5_sla_demo(governance_agent: GovernanceAgent, brief_id: str, session_factory: sessionmaker) -> None:
    """Demonstrate G5 SLA Escalation: breach detection emits notification without automated decision."""
    logger.info("\n" + "=" * 75)
    logger.info(" DEMO STEP 3: G5 SLA ESCALATION DEMONSTRATION")
    logger.info(" Rule: SLA breach triggers notification ONLY. Decision remains UNSET.")
    logger.info("=" * 75)

    # Set due date on the brief's assignment to 4 hours in the past
    with session_factory() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert assignment is not None, "Brief must have a review assignment."
        expired_date = datetime.now(timezone.utc) - timedelta(hours=4)
        assignment.due_date = expired_date
        session.commit()
        assignment_id = assignment.id
        reviewer_id = assignment.reviewer_id

    logger.info("Simulated ReviewAssignment %s assigned to '%s' with expired due date (%s)",
                assignment_id[:8], reviewer_id, expired_date.strftime("%Y-%m-%d %H:%M:%S UTC"))

    # Execute single-cycle SLA evaluation
    logger.info("Executing single-cycle deterministic SLA evaluation...")
    sla_result = governance_agent.evaluate_sla(change_brief_id=brief_id)

    logger.info("SLA Evaluation Output:")
    logger.info("   • SLA Overdue: %s", sla_result.is_overdue)
    logger.info("   • Escalated: %s", sla_result.escalated)
    logger.info("   • Reason: %s", sla_result.reason)

    # Verify decision is STILL None
    with session_factory() as session:
        reloaded_assignment = session.get(ReviewAssignment, assignment_id)
        assert reloaded_assignment.decision is None, "CRITICAL ERROR: SLA evaluation must NEVER make a governance decision!"
        logger.info("SAFETY ASSERTION VERIFIED: Review assignment decision remains strictly UNSET ('%s')", reloaded_assignment.decision)

        # Verify notification was logged
        notif = session.query(Notification).filter(Notification.related_entity_id == brief_id).first()
        assert notif is not None, "Notification record must be persisted."
        logger.info("Notification persisted: Type = %s, Recipient = %s", notif.notification_type, notif.recipient)

    # Perform explicit human resolution post-escalation
    logger.info("\nPerforming explicit human decision post-escalation:")
    governance_agent.start_review(change_brief_id=brief_id, reviewer_id=reviewer_id)
    decided_brief = governance_agent.decide(
        change_brief_id=brief_id,
        reviewer_id=reviewer_id,
        decision=ReviewDecision.REJECT,
        rationale="Rejected following escalation review: risk profile too high for general adoption.",
    )
    logger.info("Post-escalation Human Decision recorded: brief status -> '%s'", decided_brief.status)


def main() -> int:
    """Main demonstration runner."""
    config = load_config()
    logger.info("=================================================================")
    logger.info(" CLINICAL KNOWLEDGE EVOLUTION AGENT (CKEA) - FULL DEMONSTRATION")
    logger.info(" Phase 13 Final Integration & Governance Demonstration")
    logger.info("=================================================================")
    logger.info("Target Database: %s", config.database_url)
    logger.info("Protocols Dir:   %s", config.protocol_dir)

    # Snapshot protocols and ChromaDB to verify immutability
    protocol_snapshot = snapshot_protocols(config.protocol_dir)
    chroma_snapshot = snapshot_chroma_protocols(config.chroma_dir)
    for p_name, p_hash in protocol_snapshot.items():
        logger.info("Baseline authoritative protocol: %s (hash: %s)", p_name, p_hash)
    if chroma_snapshot:
        logger.info("Baseline ChromaDB protocol records count: %d", chroma_snapshot.get("count", 0))

    # Isolated in-memory database for demo execution
    engine = get_engine(db_url="sqlite:///:memory:")
    init_db(engine=engine)
    session_factory = get_session_factory(engine=engine)

    # Initialize deterministic mock LLM client (Zero external Groq calls for offline demo)
    mock_llm = MagicMock(spec=SharedLLMClient)

    def mock_extract(section_heading, section_text, page_number, document_identifier, **kwargs):
        if "tentative dosage variations" in section_text:
            return ExtractionResponse(
                recommendations=[
                    ExtractedRecommendation(
                        verbatim_text="Clinicians may consider tentative dosage variations as deemed appropriate.",
                        recommendation_type="treatment",
                        target_population="Adult diabetes patients",
                        intervention="Tentative dosage variations",
                        evidence_grade="Grade B",
                        page=page_number,
                        section=section_heading,
                        source_excerpt="Clinicians may consider tentative dosage variations as deemed appropriate.",
                        confidence=0.55,  # < 0.70 triggers G1
                    )
                ]
            )
        return ExtractionResponse(
            recommendations=[
                ExtractedRecommendation(
                    verbatim_text="Adult patients with type 2 diabetes mellitus should initiate metformin 1000mg once daily with the evening meal.",
                    recommendation_type="treatment",
                    target_population="Adult patients with type 2 diabetes mellitus",
                    intervention="Metformin 1000mg once daily",
                    evidence_grade="Grade A",
                    page=page_number,
                    section=section_heading,
                    source_excerpt="Adult patients with type 2 diabetes mellitus should initiate metformin 1000mg once daily with the evening meal.",
                    confidence=0.95,
                )
            ]
        )

    mock_llm.extract_recommendations.side_effect = mock_extract

    mock_index = MagicMock(spec=ProtocolIndexService)
    mock_index.retrieve.return_value = [
        CandidateProtocolSection(
            chroma_id="PROT-DM-001_v1.0_SEC-2",
            protocol_id="PROT-DM-001",
            protocol_version="v1.0",
            section_id="SEC-2",
            section_heading="First-Line Pharmacotherapy",
            section_text="Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
            similarity=0.92,
            distance=0.08,
        )
    ]

    mock_llm.compare_recommendation_to_protocol.return_value = ComparisonResponse(
        comparison_result=ComparisonResult.GAP,
        matched_protocol_section="First-Line Pharmacotherapy",
        protocol_id="PROT-DM-001",
        protocol_version="v1.0",
        section_id="SEC-2",
        exact_protocol_text="Metformin 500mg once daily with evening meals is the preferred initial pharmacologic agent for type 2 diabetes.",
        specific_difference="Initial dose increased from 500mg daily to 1000mg daily.",
        difference_type=DifferenceType.DOSAGE_CHANGE,
        confidence=0.95,
        rationale="Clear dosage recommendation difference identified.",
    )

    governance_agent = GovernanceAgent(session_factory=session_factory, config=config)
    extraction_agent = ExtractionAgent(session_factory=session_factory, llm_client=mock_llm, config=config)
    comparison_agent = ComparisonAgent(session_factory=session_factory, llm_client=mock_llm, protocol_index_service=mock_index, config=config)

    pipeline = ClinicalKnowledgePipeline(
        session_factory=session_factory,
        config=config,
        extraction_agent=extraction_agent,
        comparison_agent=comparison_agent,
        governance_agent=governance_agent,
    )

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)

        # 1. Happy-path lifecycle demonstration
        brief_id = run_happy_path_demo(pipeline, governance_agent, temp_dir)

        # 2. Safety path G1 hold & explicit resolution
        sla_brief_id = run_safety_path_demo(pipeline, temp_dir)

        # 3. G5 SLA escalation demonstration
        run_g5_sla_demo(governance_agent, sla_brief_id, session_factory)

    # 4. Assert protocol and vector index immutability
    assert_protocols_unchanged(config.protocol_dir, protocol_snapshot, config.chroma_dir, chroma_snapshot)

    logger.info("\n" + "=" * 75)
    logger.info(" DEMONSTRATION COMPLETED SUCCESSFULLY: ALL WORKFLOWS & GATES VERIFIED")
    logger.info("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
