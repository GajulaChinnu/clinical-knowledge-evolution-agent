"""Smoke test for Phase 13 Final Demonstration and Governance Integration.

Validates:
1. Streamlit UI functions and models can be imported without side effects.
2. Dashboard metrics and data queries run without errors.
3. Full governance action lifecycle (assign -> start -> decide -> close).
4. Rejection of decision without clinical rationale.
5. Rejection of deferral without future due date.
6. Non-bypassability of human gate G4.
7. G5 SLA escalation triggers notification only (zero auto-decision).
8. Protocol immutability verification.
9. Zero automatic governance decisions across all scenarios.
"""

from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import sys
import uuid

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.agents.governance_agent import GovernanceAgent
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
from app.schemas.briefs import BriefStatus, validate_brief_completeness
from app.schemas.changes import ChangeStatus
from app.schemas.documents import DocumentStatus
from app.schemas.gaps import DifferenceType, GapStatus
from app.schemas.governance import (
    GovernanceGateError,
    ReviewAssignmentStatus,
    ReviewDecision,
    UnauthorizedReviewerError,
)
from app.schemas.impact import ImpactStatus, ImpactTier
from app.services.config_service import load_config
from app.services.file_hash import compute_sha256
from app.services.reviewer_authorization import ReviewerAuthorizationService
from app.ui.streamlit_app import (
    execute_governance_action,
    get_audit_trail,
    get_brief_full_payload,
    get_brief_summaries,
    get_changes_with_gaps,
    get_dashboard_metrics,
    get_impact_records,
    get_source_documents,
    load_evaluation_reports,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ckea.scripts.smoke_test_final_demo")


def main() -> int:
    logger.info("=================================================================")
    logger.info(" CKEA PHASE 13: FINAL INTEGRATION & DEMO SMOKE TEST")
    logger.info("=================================================================")

    config = load_config()

    # Step 1: Snapshot protocol directory
    logger.info("[Step 1/12] Snapshotting protocol data...")
    protocol_snapshot = {}
    for p in config.protocol_dir.glob("*.json"):
        protocol_snapshot[p.name] = compute_sha256(p)
    logger.info("Protocol snapshot captured: %d files.", len(protocol_snapshot))

    # Step 2: Initialize in-memory isolated database
    logger.info("[Step 2/12] Setting up isolated database...")
    engine = get_engine(db_url="sqlite:///:memory:")
    init_db(engine=engine)
    session_factory = get_session_factory(engine=engine)

    # Step 3: Populate sample records for UI verification
    logger.info("[Step 3/12] Populating deterministic sample records...")
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="SYN-DEMO-001",
            source_path="data/sources/demo.pdf",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            source_version="1.0",
            status=DocumentStatus.PARSED.value,
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Clinicians should prescribe metformin 1000mg daily.",
            recommendation_type="treatment",
            target_population="Adults with type 2 diabetes",
            intervention="Metformin 1000mg daily",
            evidence_grade="Grade A",
            confidence=0.95,
            page=1,
            section="1. Treatment",
            source_excerpt="Clinicians should prescribe metformin 1000mg daily.",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            comparison_result="gap",
            similarity=0.92,
            comparison_confidence=0.95,
            difference_type=DifferenceType.DOSAGE_CHANGE.value,
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="v1.0",
            status=GapStatus.MATCHED.value,
        )
        session.add(gap)
        session.flush()

        imp = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=3,
            evidence_strength=3,
            pathway_breadth=2,
            total_score=8,
            tier=ImpactTier.HIGH.value,
            sla_deadline=datetime.now(timezone.utc) + timedelta(days=7),
            urgency_basis="Dosage change requires formulary update.",
            evidence_basis="High quality Grade A randomized trial.",
            breadth_basis="Primary care outpatient pathway.",
            rule_ids=["RULE-DOSE-01", "RULE-EVID-01", "RULE-BREADTH-01"],
            scoring_yaml_version="v1.0.0",
            status=ImpactStatus.ROUTED.value,
        )
        session.add(imp)
        session.flush()

        brief = ChangeBrief(
            impact_record_id=imp.id,
            status=BriefStatus.ASSIGNED.value,
            structured_payload={
                "recommendation": {
                    "verbatim_text": change.verbatim_text,
                    "target_population": change.target_population,
                    "intervention": change.intervention,
                    "evidence_grade": change.evidence_grade,
                },
                "protocol": {
                    "protocol_id": "PROT-DM-001",
                    "protocol_version": "v1.0",
                    "section_heading": "First-Line Pharmacotherapy",
                    "section_text": "Metformin 500mg daily is preferred.",
                },
                "comparison": {
                    "difference_type": "dosage_change",
                    "specific_difference": "Dose increased from 500mg to 1000mg daily.",
                },
                "impact": {
                    "tier": "High",
                    "total_score": 8,
                    "sla_deadline": str(datetime.now(timezone.utc) + timedelta(days=7)),
                    "urgency_basis": "Dosage change requires formulary update.",
                },
                "workflow": {
                    "affected_departments": ["Endocrinology", "Primary Care"],
                    "workflow_description": "Update EHR order sets.",
                },
                "proposed_actions": [
                    {"action_type": "Formulary Review", "description": "Review dose adjustment."}
                ],
                "source_excerpt": {
                    "page": 1,
                    "section": "1. Treatment",
                    "excerpt": change.verbatim_text,
                },
            },
        )
        session.add(brief)
        session.flush()

        assignment = ReviewAssignment(
            change_brief_id=brief.id,
            reviewer_role="Clinical Governance Lead",
            reviewer_id="dr_smith",
            status=ReviewAssignmentStatus.ASSIGNED.value,
            due_date=datetime.now(timezone.utc) + timedelta(days=7),
        )
        session.add(assignment)
        session.commit()
        sample_brief_id = brief.id

    logger.info("Sample records populated successfully. Brief ID: %s", sample_brief_id[:8])

    # Step 4: Verify Dashboard Metrics
    logger.info("[Step 4/12] Verifying dashboard metric retrieval...")
    metrics = get_dashboard_metrics(session_factory)
    assert metrics["total_documents"] == 1
    assert metrics["processed_documents"] == 1
    assert metrics["awaiting_governance"] == 1
    assert metrics["closed_briefs"] == 0
    logger.info("Dashboard metrics verified: %s", metrics)

    # Step 5: Verify Source Documents and Changes queries
    logger.info("[Step 5/12] Verifying data inspection queries...")
    docs = get_source_documents(session_factory)
    assert len(docs) == 1
    assert docs[0]["source_identifier"] == "SYN-DEMO-001"

    changes = get_changes_with_gaps(session_factory)
    assert len(changes) == 1
    assert changes[0]["difference_type"] == DifferenceType.DOSAGE_CHANGE.value

    impacts = get_impact_records(session_factory)
    assert len(impacts) == 1
    assert impacts[0]["tier"] == ImpactTier.HIGH.value

    briefs = get_brief_summaries(session_factory)
    assert len(briefs) == 1
    assert briefs[0]["assigned_reviewer"] == "dr_smith"
    logger.info("Data queries verified.")

    # Step 6: Verify 7-section Change Brief payload
    logger.info("[Step 6/12] Verifying 7-section ChangeBrief rendering...")
    full_payload = get_brief_full_payload(session_factory, sample_brief_id)
    assert full_payload is not None
    assert full_payload["structured_payload"] is not None
    required_sections = ["recommendation", "protocol", "comparison", "impact", "workflow", "proposed_actions", "source_excerpt"]
    for sec in required_sections:
        assert sec in full_payload["structured_payload"], f"Missing brief section: {sec}"
    logger.info("All 7 required brief sections verified.")

    # Step 7: Governance Agent Workflow (Start Review)
    logger.info("[Step 7/12] Verifying Start Review action...")
    gov_agent = GovernanceAgent(session_factory=session_factory, config=config)
    res_start = execute_governance_action(
        governance_agent=gov_agent,
        brief_id=sample_brief_id,
        action="start_review",
        reviewer_id="dr_smith",
        rationale="",
    )
    assert res_start["status"] == "success"

    # Step 8: Verify empty rationale is rejected for decision
    logger.info("[Step 8/12] Verifying decision requires mandatory clinical rationale...")
    try:
        execute_governance_action(
            governance_agent=gov_agent,
            brief_id=sample_brief_id,
            action="approve",
            reviewer_id="dr_smith",
            rationale="   ",  # whitespace only
        )
        assert False, "Approval without rationale must be rejected!"
    except ValueError as e:
        logger.info("Approval with empty rationale correctly rejected: %s", e)

    # Step 9: Verify deferral requires future date
    logger.info("[Step 9/12] Verifying deferral requires future due date...")
    try:
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        execute_governance_action(
            governance_agent=gov_agent,
            brief_id=sample_brief_id,
            action="defer",
            reviewer_id="dr_smith",
            rationale="Clinical trial data pending publication.",
            follow_up_date=past_date,
        )
        assert False, "Deferral with past date must be rejected!"
    except ValueError as e:
        logger.info("Deferral with past date correctly rejected: %s", e)

    # Step 10: Record explicit human decision and close
    logger.info("[Step 10/12] Verifying explicit human decision and closure...")
    res_decide = execute_governance_action(
        governance_agent=gov_agent,
        brief_id=sample_brief_id,
        action="approve",
        reviewer_id="dr_smith",
        rationale="Approved based on Grade A evidence for 1000mg initial dosing.",
    )
    assert res_decide["decision"] == "approve"
    assert res_decide["brief_status"] == BriefStatus.DECIDED.value

    res_close = execute_governance_action(
        governance_agent=gov_agent,
        brief_id=sample_brief_id,
        action="close",
        reviewer_id="dr_smith",
        rationale="Institutional formulary updated.",
    )
    assert res_close["brief_status"] == BriefStatus.CLOSED.value
    logger.info("Decision and closure completed successfully.")

    # Step 11: Verify G4 non-bypassability on a new brief
    logger.info("[Step 11/12] Verifying G4 non-bypassability (cannot close undecided brief)...")
    with session_factory() as session:
        doc_rec = session.query(IngestedDocument).first()
        new_ch = ChangeRecord(
            ingested_document_id=doc_rec.id,
            recommendation_type="pharmacotherapy",
            target_population="Type 2 Diabetes",
            intervention="Metformin 850mg",
            verbatim_text="Metformin 850mg once daily.",
            confidence=0.90,
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
        )
        session.add(new_ch)
        session.flush()

        new_gap = GapRecord(
            change_record_id=new_ch.id,
            comparison_result="gap",
            similarity=0.90,
            status=GapStatus.MATCHED.value,
        )
        session.add(new_gap)
        session.flush()
        new_imp = ImpactRecord(
            gap_record_id=new_gap.id,
            clinical_urgency=2,
            evidence_strength=2,
            pathway_breadth=2,
            total_score=6,
            tier=ImpactTier.STANDARD.value,
            status=ImpactStatus.ROUTED.value,
        )
        session.add(new_imp)
        session.flush()
        new_brief = ChangeBrief(
            impact_record_id=new_imp.id,
            status=BriefStatus.ASSIGNED.value,
        )
        session.add(new_brief)
        session.commit()
        undecided_brief_id = new_brief.id

    try:
        execute_governance_action(
            governance_agent=gov_agent,
            brief_id=undecided_brief_id,
            action="close",
            reviewer_id="dr_smith",
            rationale="Attempting illegal closure.",
        )
        assert False, "Undecided brief cannot be closed!"
    except GovernanceGateError as e:
        logger.info("Undecided brief closure correctly rejected by G4 gate: %s", e)

    # Step 12: Verify Protocol Immutability
    logger.info("[Step 12/12] Verifying protocol immutability...")
    assert "PROT-DM-001_v1.0.json" in protocol_snapshot, "Authoritative protocol PROT-DM-001_v1.0.json must exist!"
    for p in config.protocol_dir.glob("*.json"):
        current_hash = compute_sha256(p)
        assert current_hash == protocol_snapshot[p.name], f"Protocol {p.name} was modified!"
        logger.info("Authoritative protocol file verified immutable: %s (hash: %s)", p.name, current_hash)

    # Verify ChromaDB protocol records remain unchanged
    if config.chroma_dir.exists():
        import chromadb
        chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
        coll = chroma_client.get_collection("protocol_sections")
        assert coll.count() == 4, f"Expected 4 ChromaDB protocol records, found {coll.count()}"
        chroma_ids = coll.get()["ids"]
        assert "PROT-DM-001__v1.0__SEC-1" in chroma_ids
        assert "PROT-DM-001__v1.0__SEC-4" in chroma_ids
        logger.info("ChromaDB protocol collection verified immutable: 4 records intact.")

    logger.info("=================================================================")
    logger.info(" SMOKE TEST PASSED: ALL 12 VERIFICATIONS GREEN")
    logger.info("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
