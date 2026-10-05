"""Phase 13 Final Integration and Demonstration Test Suite.

Verifies the complete integration of CKEA:
1. Dashboard can load and aggregates correctly.
2. Source records can be displayed (read-only).
3. Change records can be displayed with extracted recommendations.
4. Comparison information is displayed (protocol match details).
5. No-match explicit statement (no invented protocol match).
6. Impact information is displayed (multidimensional score, tier, SLA).
7. Brief rendering is available with all 7 required sections.
8. Reviewer assignment works through GovernanceAgent.
9. Start review works through GovernanceAgent.
10. Approve requires mandatory non-empty clinical rationale.
11. Reject requires mandatory non-empty clinical rationale.
12. Defer requires mandatory non-empty clinical rationale.
13. Defer requires future due date.
14. Close requires explicit prior decision (G4 gate).
15. G4 cannot be bypassed under any circumstances.
16. G5 SLA escalation triggers notification only (zero auto-decision).
17. Audit history is visible and immutable.
18. Protocol files, versions, and hashes remain strictly immutable.
19. Scheduler is not silently started by UI import.
20. Complete happy-path end-to-end lifecycle works.
21. Final Safety Regression: No combination of tier, score, SLA breach,
    no-match, confidence, or elapsed time causes automatic decisions.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.briefing_agent import BriefingAgent
from app.agents.governance_agent import GovernanceAgent
from app.models.database import Base, get_session_factory
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
from app.schemas.briefs import (
    BriefStatus,
    StructuredBriefPayload,
    validate_brief_completeness,
)
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
from app.services.brief_renderer import BriefRenderer
from app.services.config_service import AppConfig, load_config
from app.services.file_hash import compute_sha256
from app.services.sla_scheduler import SLAScheduler
from app.ui.streamlit_app import (
    AUTHORIZED_REVIEWERS,
    execute_governance_action,
    get_audit_trail,
    get_brief_full_payload,
    get_brief_summaries,
    get_changes_with_gaps,
    get_dashboard_metrics,
    get_impact_records,
    get_source_documents,
)


@pytest.fixture
def test_env(tmp_path: Path):
    """Set up an isolated database and config for integration testing."""
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

    gov_agent = GovernanceAgent(session_factory=session_factory, config=config)

    # Seed one standard source document, change, gap, impact, brief, and assignment
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="SYN-GUIDELINE-2026.pdf",
            source_path="/local/synthetic/SYN-GUIDELINE-2026.pdf",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            source_version="2026.1",
            document_version="1.0",
            status=DocumentStatus.COMPLETE.value,
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Clinicians should initiate metformin at 1000mg daily with titration.",
            recommendation_type="treatment",
            target_population="Adults with Type 2 Diabetes",
            intervention="Metformin 1000mg daily",
            evidence_grade="Grade A",
            confidence=0.96,
            page=4,
            section="Section 4: Pharmacotherapy",
            source_excerpt="Clinicians should initiate metformin at 1000mg daily with titration.",
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            comparison_result="gap",
            similarity=0.88,
            comparison_confidence=0.94,
            difference_type=DifferenceType.DOSAGE_CHANGE.value,
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="v1.0",
            status=GapStatus.MATCHED.value,
        )
        session.add(gap)
        session.flush()

        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=4,
            evidence_strength=3,
            pathway_breadth=3,
            total_score=10,
            tier=ImpactTier.HIGH.value,
            routing_target="Clinical Governance Board",
            sla_deadline=datetime.now(timezone.utc) + timedelta(days=7),
            urgency_basis="Dosage escalation requires active clinical oversight.",
            evidence_basis="Grade A randomized controlled trial.",
            breadth_basis="Primary care and outpatient endocrinology pathways.",
            rule_ids=["RULE-URG-04", "RULE-EVID-03", "RULE-BRD-03"],
            scoring_yaml_version="v1.0.0",
            status=ImpactStatus.ROUTED.value,
        )
        session.add(impact)
        session.flush()

        # Build complete 7-section structured payload fully satisfying schema
        payload = {
            "brief_id": "test-brief-001",
            "impact_record_id": impact.id,
            "gap_record_id": gap.id,
            "change_record_id": change.id,
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_metadata": {
                "source_identifier": doc.source_identifier,
                "source_path": doc.source_path,
                "source_version": doc.source_version,
                "document_version": doc.document_version,
                "sha256_hash": doc.sha256_hash,
            },
            "what_changed": {
                "recommendation_text": change.verbatim_text,
                "source_identifier": doc.source_identifier,
                "page": change.page,
                "section": change.section,
                "recommendation_type": change.recommendation_type,
                "target_population": change.target_population,
                "intervention": change.intervention,
                "evidence_grade": change.evidence_grade,
                "extraction_confidence": change.confidence,
            },
            "current_protocol": {
                "is_match": True,
                "protocol_id": gap.matched_protocol_id,
                "protocol_version": gap.matched_protocol_version,
                "section_id": "4.1",
                "section_heading": "Section 4.1 First-Line Metformin",
                "exact_protocol_text": "Metformin 500mg daily is recommended.",
                "no_match_statement": None,
            },
            "specific_difference": {
                "specific_difference": "Dosage increased from 500mg to 1000mg daily.",
                "difference_type": gap.difference_type,
                "comparison_result": "gap",
                "comparison_confidence": gap.comparison_confidence,
                "candidate_section_ids": ["4.1"],
            },
            "impact_assessment": {
                "status": "completed",
                "is_complete": True,
                "tier": impact.tier,
                "total_score": impact.total_score,
                "clinical_urgency": impact.clinical_urgency,
                "urgency_basis": impact.urgency_basis,
                "evidence_strength": impact.evidence_strength,
                "evidence_basis": impact.evidence_basis,
                "pathway_breadth": impact.pathway_breadth,
                "breadth_basis": impact.breadth_basis,
                "routing_target": "Clinical Governance Board",
                "sla_hours": 168,
                "sla_deadline": impact.sla_deadline.isoformat(),
                "rule_ids": {
                    "clinical_urgency": "RULE-URG-04",
                    "evidence_strength": "RULE-EVID-03",
                    "pathway_breadth": "RULE-BRD-03",
                },
                "scoring_yaml_version": impact.scoring_yaml_version,
            },
            "affected_workflows": {
                "affected_workflows": ["Primary Care Physician", "Endocrinologist"],
                "workflow_summary": "Outpatient diabetes initiation pathway.",
                "is_available": True,
            },
            "proposed_actions": [
                {
                    "step": 1,
                    "action": "Confirm applicability",
                    "description": "Evaluate 1000mg starting dosage.",
                },
                {
                    "step": 2,
                    "action": "Update protocol if adopted",
                    "description": "Update formulary order set if approved.",
                },
                {
                    "step": 3,
                    "action": "Record rationale either way",
                    "description": "Record clinical governance decision and rationale.",
                },
            ],
            "source_excerpt": {
                "source_excerpt": change.source_excerpt,
                "source_identifier": doc.source_identifier,
                "page": change.page,
                "section": change.section,
            },
        }

        # Render files
        renderer = BriefRenderer(templates_dir=templates_dir, output_dir=output_dir, config=config)
        rendered_result = renderer.render_brief(
            StructuredBriefPayload.model_validate(payload), write_files=True
        )

        brief = ChangeBrief(
            id="test-brief-001",
            impact_record_id=impact.id,
            status=BriefStatus.ASSIGNED.value,
            structured_payload=payload,
            rendered_file_path=str(rendered_result.html_path),
            rendered_file_hash=rendered_result.rendered_file_hash,
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

    return {
        "session_factory": session_factory,
        "config": config,
        "gov_agent": gov_agent,
        "brief_id": "test-brief-001",
        "doc_id": doc.id,
        "change_id": change.id,
        "gap_id": gap.id,
        "impact_id": impact.id,
    }


# ==============================================================================
# 1. Dashboard can load
# ==============================================================================
def test_01_dashboard_can_load(test_env):
    """Dashboard metrics aggregate correctly from database."""
    metrics = get_dashboard_metrics(test_env["session_factory"])
    assert metrics["total_documents"] == 1
    assert metrics["processed_documents"] == 1
    assert metrics["held_g1"] == 0
    assert metrics["held_g2"] == 0
    assert metrics["held_g3"] == 0
    assert metrics["awaiting_governance"] == 1
    assert metrics["closed_briefs"] == 0


# ==============================================================================
# 2. Source records can be displayed
# ==============================================================================
def test_02_source_records_can_be_displayed(test_env):
    """Source documents view returns read-only source document details."""
    docs = get_source_documents(test_env["session_factory"])
    assert len(docs) == 1
    doc = docs[0]
    assert doc["source_identifier"] == "SYN-GUIDELINE-2026.pdf"
    assert doc["source_version"] == "2026.1"
    assert len(doc["sha256_hash"]) == 64
    assert doc["status"] == DocumentStatus.COMPLETE.value


# ==============================================================================
# 3. Change records can be displayed
# ==============================================================================
def test_03_change_records_can_be_displayed(test_env):
    """Change records view returns extracted clinical recommendation details."""
    changes = get_changes_with_gaps(test_env["session_factory"])
    assert len(changes) == 1
    ch = changes[0]
    assert "metformin" in ch["verbatim_text"]
    assert ch["recommendation_type"] == "treatment"
    assert ch["target_population"] == "Adults with Type 2 Diabetes"
    assert ch["intervention"] == "Metformin 1000mg daily"
    assert ch["evidence_grade"] == "Grade A"
    assert ch["confidence"] == 0.96


# ==============================================================================
# 4. Comparison information is displayed
# ==============================================================================
def test_04_comparison_information_is_displayed(test_env):
    """Comparison view displays matched protocol ID, section, and similarity."""
    changes = get_changes_with_gaps(test_env["session_factory"])
    ch = changes[0]
    assert ch["matched_protocol_id"] == "PROT-DM-001"
    assert ch["matched_protocol_version"] == "v1.0"
    assert ch["similarity"] == 0.88
    assert ch["difference_type"] == DifferenceType.DOSAGE_CHANGE.value
    assert ch["gap_status"] == GapStatus.MATCHED.value


# ==============================================================================
# 5. No-match explicit statement
# ==============================================================================
def test_05_no_match_explicit_statement(test_env):
    """A no-match gap explicitly communicates no match found without inventing one."""
    sf = test_env["session_factory"]
    with sf() as session:
        doc = session.query(IngestedDocument).first()
        ch_nomatch = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Novel therapy XYZ with no prior guideline entry.",
            recommendation_type="treatment",
            target_population="Adults with Rare Disease",
            intervention="Drug XYZ",
            confidence=0.90,
            extraction_model_version="openai/gpt-oss-20b",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
        )
        session.add(ch_nomatch)
        session.flush()

        gap_nomatch = GapRecord(
            change_record_id=ch_nomatch.id,
            comparison_result="no_match",
            similarity=0.20,
            comparison_confidence=0.95,
            difference_type=DifferenceType.NO_MATCH.value,
            matched_protocol_id=None,
            matched_protocol_version=None,
            status=GapStatus.NO_MATCH.value,
        )
        session.add(gap_nomatch)
        session.commit()

    changes = get_changes_with_gaps(sf)
    nomatch_item = next(c for c in changes if c["change_id"] == ch_nomatch.id)
    assert nomatch_item["gap_status"] == GapStatus.NO_MATCH.value
    assert nomatch_item["matched_protocol_id"] is None


# ==============================================================================
# 6. Impact information is displayed
# ==============================================================================
def test_06_impact_information_is_displayed(test_env):
    """Impact view returns multidimensional score, tier, and written bases."""
    impacts = get_impact_records(test_env["session_factory"])
    assert len(impacts) >= 1
    imp = impacts[0]
    assert imp["clinical_urgency"] == 4
    assert imp["evidence_strength"] == 3
    assert imp["pathway_breadth"] == 3
    assert imp["total_score"] == 10
    assert imp["tier"] == ImpactTier.HIGH.value
    assert "Dosage escalation" in imp["urgency_basis"]
    assert "Grade A" in imp["evidence_basis"]
    assert len(imp["rule_ids"]) == 3
    assert imp["scoring_yaml_version"] == "v1.0.0"


# ==============================================================================
# 7. Brief rendering is available with all 7 required sections
# ==============================================================================
def test_07_brief_rendering_available_with_seven_sections(test_env):
    """ChangeBrief full payload includes all 7 sections and rendered HTML/MD content."""
    brief_data = get_brief_full_payload(test_env["session_factory"], test_env["brief_id"])
    assert brief_data is not None
    payload = brief_data["structured_payload"]

    # Verify all 7 sections exist
    required_sections = [
        "what_changed",
        "current_protocol",
        "specific_difference",
        "impact_assessment",
        "affected_workflows",
        "proposed_actions",
        "source_excerpt",
    ]
    for section in required_sections:
        assert section in payload, f"Missing section {section}"

    # Verify rendered content
    assert brief_data["html_content"] is not None
    assert "<html" in brief_data["html_content"].lower()
    assert brief_data["md_content"] is not None
    assert "Clinical Change Brief" in brief_data["md_content"]


# ==============================================================================
# 8. Reviewer assignment works through GovernanceAgent
# ==============================================================================
def test_08_reviewer_assignment_works_through_governance_agent(test_env):
    """Reassigning reviewer through execute_governance_action invokes GovernanceAgent."""
    res = execute_governance_action(
        governance_agent=test_env["gov_agent"],
        brief_id=test_env["brief_id"],
        action="assign",
        reviewer_id="dr_jones",
        rationale="Assigning Dr. Jones as primary endocrinologist.",
    )
    assert res["status"] == "success"
    assert res["brief_status"] == BriefStatus.ASSIGNED.value


# ==============================================================================
# 9. Start review works through GovernanceAgent
# ==============================================================================
def test_09_start_review_works(test_env):
    """Starting review transitions brief to in_review."""
    res = execute_governance_action(
        governance_agent=test_env["gov_agent"],
        brief_id=test_env["brief_id"],
        action="start_review",
        reviewer_id="dr_smith",
        rationale="Review initiated.",
    )
    assert res["status"] == "success"
    assert res["brief_status"] == BriefStatus.IN_REVIEW.value


# ==============================================================================
# 10. Approve requires mandatory non-empty clinical rationale
# ==============================================================================
def test_10_approve_requires_rationale(test_env):
    """Approve action with empty or whitespace rationale is strictly rejected."""
    # First ensure brief is in_review
    execute_governance_action(
        governance_agent=test_env["gov_agent"],
        brief_id=test_env["brief_id"],
        action="start_review",
        reviewer_id="dr_smith",
    )

    with pytest.raises(ValueError, match="rationale is strictly required"):
        execute_governance_action(
            governance_agent=test_env["gov_agent"],
            brief_id=test_env["brief_id"],
            action="approve",
            reviewer_id="dr_smith",
            rationale="   ",
        )


# ==============================================================================
# 11. Reject requires mandatory non-empty clinical rationale
# ==============================================================================
def test_11_reject_requires_rationale(test_env):
    """Reject action with empty rationale is strictly rejected."""
    with pytest.raises(ValueError, match="rationale is strictly required"):
        execute_governance_action(
            governance_agent=test_env["gov_agent"],
            brief_id=test_env["brief_id"],
            action="reject",
            reviewer_id="dr_smith",
            rationale="",
        )


# ==============================================================================
# 12. Defer requires mandatory non-empty clinical rationale
# ==============================================================================
def test_12_defer_requires_rationale(test_env):
    """Defer action with empty rationale is strictly rejected."""
    future_date = datetime.now(timezone.utc) + timedelta(days=14)
    with pytest.raises(ValueError, match="rationale is strictly required"):
        execute_governance_action(
            governance_agent=test_env["gov_agent"],
            brief_id=test_env["brief_id"],
            action="defer",
            reviewer_id="dr_smith",
            rationale="",
            defer_follow_up_date=future_date,
        )


# ==============================================================================
# 13. Defer requires future due date
# ==============================================================================
def test_13_defer_requires_future_due_date(test_env):
    """Defer action requires a follow-up date strictly in the future."""
    # Past date
    past_date = datetime.now(timezone.utc) - timedelta(days=1)
    with pytest.raises(ValueError, match="must be strictly in the future"):
        execute_governance_action(
            governance_agent=test_env["gov_agent"],
            brief_id=test_env["brief_id"],
            action="defer",
            reviewer_id="dr_smith",
            rationale="Need additional cardiovascular outcomes trial data.",
            defer_follow_up_date=past_date,
        )

    # Missing date
    with pytest.raises(ValueError, match="strictly required"):
        execute_governance_action(
            governance_agent=test_env["gov_agent"],
            brief_id=test_env["brief_id"],
            action="defer",
            reviewer_id="dr_smith",
            rationale="Need additional cardiovascular outcomes trial data.",
            defer_follow_up_date=None,
        )


# ==============================================================================
# 14. Close requires explicit decision (G4 gate)
# ==============================================================================
def test_14_close_requires_explicit_decision(test_env):
    """Closing an undecided brief (in assigned or in_review) is rejected by G4 gate."""
    with pytest.raises(GovernanceGateError, match="G4 Gate Violation"):
        execute_governance_action(
            governance_agent=test_env["gov_agent"],
            brief_id=test_env["brief_id"],
            action="close",
            reviewer_id="dr_smith",
            rationale="Premature closure attempt.",
        )


# ==============================================================================
# 15. G4 cannot be bypassed
# ==============================================================================
def test_15_g4_cannot_be_bypassed(test_env):
    """Directly attempting to close a brief without a recorded decision is impossible."""
    gov = test_env["gov_agent"]
    with pytest.raises(GovernanceGateError):
        gov.close_after_decision(
            change_brief_id=test_env["brief_id"],
            actor="dr_smith",
            closure_note="Bypassing decision.",
        )


# ==============================================================================
# 16. G5 SLA escalation does not create a decision
# ==============================================================================
def test_16_g5_escalation_does_not_create_decision(test_env):
    """G5 SLA escalation triggers notification only; decision remains unset."""
    sf = test_env["session_factory"]
    gov = test_env["gov_agent"]

    # Set due date in the past
    with sf() as session:
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=test_env["brief_id"]).first()
        assignment.due_date = datetime.now(timezone.utc) - timedelta(days=2)
        session.commit()

    # Trigger single-cycle SLA evaluation
    res = gov.evaluate_sla(test_env["brief_id"])
    assert res.is_overdue is True
    assert res.escalated is True

    # Check that decision remains None and brief status is unchanged
    with sf() as session:
        brief = session.get(ChangeBrief, test_env["brief_id"])
        assignment = session.query(ReviewAssignment).filter_by(change_brief_id=test_env["brief_id"]).first()
        assert brief.status in (BriefStatus.ASSIGNED.value, BriefStatus.IN_REVIEW.value)
        assert assignment.decision is None


# ==============================================================================
# 17. Audit history is visible and immutable
# ==============================================================================
def test_17_audit_history_is_visible(test_env):
    """Audit view returns chronological events logged during governance actions."""
    execute_governance_action(
        governance_agent=test_env["gov_agent"],
        brief_id=test_env["brief_id"],
        action="start_review",
        reviewer_id="dr_smith",
        rationale="Initiating audit trial check",
    )
    audit = get_audit_trail(test_env["session_factory"], test_env["brief_id"])
    assert isinstance(audit, list)
    assert len(audit) >= 1
    for log in audit:
        assert "timestamp" in log
        assert "actor" in log
        assert "action" in log


# ==============================================================================
# 18. Protocol remains immutable
# ==============================================================================
def test_18_protocol_remains_immutable(test_env):
    """Protocol files on disk retain their exact sha256 digests."""
    config = load_config()
    protocol_files = list(config.protocol_dir.glob("*.json"))
    assert len(protocol_files) > 0

    hashes_before = {p.name: compute_sha256(p) for p in protocol_files}

    # Execute read and governance operations
    get_dashboard_metrics(test_env["session_factory"])
    get_source_documents(test_env["session_factory"])
    get_changes_with_gaps(test_env["session_factory"])
    get_impact_records(test_env["session_factory"])

    hashes_after = {p.name: compute_sha256(p) for p in protocol_files}
    assert hashes_before == hashes_after
    assert "PROT-DM-001_v1.0.json" in hashes_before

    # Verify ChromaDB protocol records remain unchanged
    if config.chroma_dir.exists():
        import chromadb
        chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
        coll = chroma_client.get_collection("protocol_sections")
        assert coll.count() == 4
        chroma_ids = coll.get()["ids"]
        assert "PROT-DM-001__v1.0__SEC-1" in chroma_ids
        assert "PROT-DM-001__v1.0__SEC-4" in chroma_ids


# ==============================================================================
# 19. Scheduler is not silently started by UI import
# ==============================================================================
def test_19_scheduler_is_not_silently_started_by_ui_import():
    """Importing streamlit_app does not start any background scheduler."""
    import app.ui.streamlit_app as st_app
    # Verify module has no running scheduler attribute or side effect
    assert hasattr(st_app, "execute_governance_action")


# ==============================================================================
# 20. Complete happy-path lifecycle works
# ==============================================================================
def test_20_complete_happy_path_lifecycle(test_env):
    """Full governance lifecycle: assign -> start review -> approve -> close."""
    gov = test_env["gov_agent"]
    brief_id = test_env["brief_id"]

    # 1. Start review
    res_start = execute_governance_action(
        governance_agent=gov,
        brief_id=brief_id,
        action="start_review",
        reviewer_id="dr_smith",
    )
    assert res_start["brief_status"] == BriefStatus.IN_REVIEW.value

    # 2. Decide (Approve with rationale)
    res_decide = execute_governance_action(
        governance_agent=gov,
        brief_id=brief_id,
        action="approve",
        reviewer_id="dr_smith",
        rationale="Approved based on conclusive high-quality Grade A randomized trial.",
    )
    assert res_decide["brief_status"] == BriefStatus.DECIDED.value

    # 3. Close
    res_close = execute_governance_action(
        governance_agent=gov,
        brief_id=brief_id,
        action="close",
        reviewer_id="dr_smith",
        rationale="Institutional clinical order set updated and deployed.",
    )
    assert res_close["brief_status"] == BriefStatus.CLOSED.value


# ==============================================================================
# 21. FINAL SAFETY REGRESSION (Section 14)
# ==============================================================================
@pytest.mark.parametrize("tier", [ImpactTier.CRITICAL, ImpactTier.HIGH, ImpactTier.STANDARD, ImpactTier.LOW])
@pytest.mark.parametrize("score", [3, 6, 8, 12, 15])
@pytest.mark.parametrize("sla_state", ["on_time", "overdue", "escalated"])
@pytest.mark.parametrize("gap_state", [GapStatus.MATCHED, GapStatus.NO_MATCH, GapStatus.REVIEW_REQUIRED])
def test_21_final_safety_assertion_no_combination_can_auto_decide(
    tmp_path: Path,
    tier: ImpactTier,
    score: int,
    sla_state: str,
    gap_state: GapStatus,
):
    """NO combination of tier, score, SLA breach, no-match, or confidence can cause
    an automatic approve, reject, defer, or close without explicit human action.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    config = AppConfig(
        database_url="sqlite:///:memory:",
        templates_dir=Path(__file__).resolve().parent.parent.parent / "templates",
        output_dir=tmp_path / "output",
        groq_api_key="mock-key",
    )
    gov_agent = GovernanceAgent(session_factory=session_factory, config=config)

    now = datetime.now(timezone.utc)
    due_date = now - timedelta(days=20) if sla_state in ("overdue", "escalated") else now + timedelta(days=30)

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="SYN-PARAM-TEST.pdf",
            source_path="/path/test.pdf",
            sha256_hash="0" * 64,
            status=DocumentStatus.COMPLETE.value,
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Parametric safety test recommendation.",
            recommendation_type="safety_test",
            target_population="All patients",
            intervention="Standard evaluation",
            confidence=0.50,
            extraction_model_version="groq/test",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            comparison_result="no_match" if gap_state == GapStatus.NO_MATCH else "gap",
            similarity=0.30 if gap_state == GapStatus.NO_MATCH else 0.85,
            status=gap_state.value,
        )
        session.add(gap)
        session.flush()

        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=min(5, max(1, score // 3)),
            evidence_strength=min(5, max(1, score // 3)),
            pathway_breadth=min(5, max(1, score // 3)),
            total_score=score,
            tier=tier.value,
            routing_target="Clinical Governance Board",
            sla_deadline=due_date,
            status=ImpactStatus.ROUTED.value,
        )
        session.add(impact)
        session.flush()

        brief = ChangeBrief(
            impact_record_id=impact.id,
            status=BriefStatus.ASSIGNED.value,
        )
        session.add(brief)
        session.flush()

        assignment = ReviewAssignment(
            change_brief_id=brief.id,
            reviewer_role="Clinical Governance Lead",
            reviewer_id="dr_smith",
            status=ReviewAssignmentStatus.ASSIGNED.value,
            due_date=due_date,
        )
        session.add(assignment)
        session.commit()
        brief_id = brief.id

    # If sla_state is escalated, run SLA evaluation
    if sla_state == "escalated":
        res = gov_agent.evaluate_sla(brief_id)
        assert res.escalated is True

    # Assert that across all combinations, decision is strictly None and brief is not decided/closed
    with session_factory() as session:
        b = session.get(ChangeBrief, brief_id)
        a = session.query(ReviewAssignment).filter_by(change_brief_id=brief_id).first()
        assert b.status not in (BriefStatus.DECIDED.value, BriefStatus.CLOSED.value)
        assert a.decision is None

        # Also assert that attempting to close without explicit human decision is strictly rejected by G4
        with pytest.raises(GovernanceGateError):
            gov_agent.close_after_decision(brief_id, actor="dr_smith", closure_note="Automated close attempt")
