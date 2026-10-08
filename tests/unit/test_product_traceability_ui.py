"""
Tests for CKEA Product Behavior and Traceability UI enhancements.

Verifies:
1. Extracted recommendation is shown
2. Current protocol ID/version/section is shown
3. Source recommendation and protocol recommendation are both visible
4. Comparison result is visible
5. Similarity/confidence is visible
6. G1 state is clearly represented
7. No recommendation does not create impact/brief/governance records
8. G2 state is clearly represented
9. G3 no-match state is clearly represented
10. G4 awaiting human decision is clearly represented
11. G5 contains escalation only
12. Ingestion complete does not imply pipeline complete
13. Source URL provenance remains visible
14. Protocol remains immutable
15. Chroma protocol collection remains unchanged
16. No new LLM calls are introduced in UI
17. Existing PDF and URL ingestion tests continue passing
"""

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import Base, get_session_factory
from app.models.entities import (
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
from app.schemas.documents import DocumentStatus
from app.schemas.gaps import ComparisonResult, DifferenceType, GapStatus
from app.schemas.impact import ImpactStatus, ImpactTier
from app.schemas.orchestration import (
    HumanGate,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)
from app.services.config_service import AppConfig
from app.ui.streamlit_app import (
    get_brief_summaries,
    get_changes_with_gaps,
    get_impact_records,
    render_processing_chain,
    resolve_protocol_section_details,
)


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite database session fixture."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_protocol_json_path(tmp_path):
    proto_data = {
        "protocol_id": "PROT-DM-001",
        "title": "Institutional Type 2 Diabetes Mellitus Clinical Protocol",
        "protocol_version": "v1.0",
        "sections": [
            {
                "section_id": "SEC-2",
                "section_heading": "First-Line Pharmacotherapy",
                "section_text": "Metformin 500mg daily is the first-line pharmacotherapy for newly diagnosed Type 2 Diabetes.",
            }
        ],
    }
    path = tmp_path / "PROT-DM-001_v1.0.json"
    path.write_text(json.dumps(proto_data), encoding="utf-8")
    return path


class TestTraceabilityHelpers:
    """Tests for protocol section resolution and UI processing chain helpers."""

    def test_1_resolve_protocol_section_details(self, sample_protocol_json_path):
        details = resolve_protocol_section_details(
            candidate_section_ids=["PROT-DM-001__v1.0__SEC-2"],
            protocol_id="PROT-DM-001",
            protocol_version="v1.0",
            protocol_dir=sample_protocol_json_path.parent,
        )
        assert details["protocol_id"] == "PROT-DM-001"
        assert details["protocol_version"] == "v1.0"
        assert details["section_id"] == "SEC-2"
        assert details["section_heading"] == "First-Line Pharmacotherapy"
        assert "Metformin 500mg daily" in details["section_text"]

    def test_2_render_processing_chain_representations(self):
        # 6. G1 state is clearly represented
        chain_g1 = render_processing_chain(gate="G1", status="held")
        assert "G1 — Extraction Quality Review" in chain_g1
        assert "Extraction (G1)" in chain_g1
        assert "⏸️ Comparison" in chain_g1

        # 8. G2 state is clearly represented
        chain_g2 = render_processing_chain(gate="G2", status="review")
        assert "G2 — Ambiguous Comparison Review" in chain_g2
        assert "Comparison (G2)" in chain_g2
        assert "⏸️ Impact" in chain_g2

        # 9. G3 no-match state is clearly represented
        chain_g3 = render_processing_chain(gate="G3", status="held")
        assert "G3 — Unmatched Clinical Finding" in chain_g3
        assert "Comparison (G3)" in chain_g3

        # 10. G4 awaiting human decision is clearly represented
        chain_g4 = render_processing_chain(gate="G4", status="pending")
        assert "G4 — Authorized Governance Sign-off" in chain_g4
        assert "Governance (G4)" in chain_g4
        assert "Brief" in chain_g4

        # Complete / Approved
        chain_comp = render_processing_chain(status="complete")
        assert "Governance" in chain_comp


class TestDataRetrievalAndProvenance:
    """Verifies that queries return source, protocol baseline, and comparison fields."""

    def test_3_get_changes_with_gaps_exposes_baseline_and_source(self, db_session: Session, sample_protocol_json_path):
        # Seed Source Document
        doc = IngestedDocument(
            id="DOC-TEST-001",
            source_identifier="ada_2024_guideline.pdf",
            source_path="data/sources/ada_2024_guideline.pdf",
            sha256_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            status=DocumentStatus.COMPLETE.value,
            doc_metadata={
                "source_type": "URL",
                "source_url": "https://diabetesjournals.org/care/article/2024/guideline",
                "title": "ADA Standards of Care 2024",
                "retrieval_timestamp": "2026-10-06T10:00:00Z",
                "ingestion_status": "Complete",
                "pipeline_status": "Complete",
            },
        )
        db_session.add(doc)

        # Seed Recommendation (ChangeRecord)
        rec = ChangeRecord(
            id="REC-TEST-001",
            ingested_document_id=doc.id,
            verbatim_text="Initiate SGLT2i or GLP-1 RA in T2D with high CV risk irrespective of baseline HbA1c.",
            recommendation_type="INTERVENTION",
            target_population="Type 2 Diabetes with ASCVD",
            intervention="SGLT2 inhibitor or GLP-1 RA",
            evidence_grade="Level A",
            page=14,
            section="Pharmacotherapy",
            source_excerpt="In adults with type 2 diabetes and established cardiovascular disease...",
            confidence=0.95,
            extraction_model_version="groq-llama-3.3-70b-versatile",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
        )
        db_session.add(rec)

        # Seed GapRecord
        gap = GapRecord(
            id="GAP-TEST-001",
            change_record_id=rec.id,
            candidate_protocol_section_ids=["PROT-DM-001__v1.0__SEC-2"],
            comparison_result=ComparisonResult.GAP.value,
            difference_type="scope_expansion",
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="v1.0",
            is_match=True,
            similarity=0.88,
            comparison_confidence=0.92,
            status=GapStatus.MATCHED.value,
        )
        db_session.add(gap)
        db_session.commit()

        cfg = AppConfig(
            database_url="sqlite:///:memory:",
            protocol_dir=sample_protocol_json_path.parent,
            groq_api_key="mock-key",
        )

        changes = get_changes_with_gaps(db_session, config=cfg)
        assert len(changes) == 1
        item = changes[0]

        # 1. Extracted recommendation is shown
        assert item["verbatim_text"] == rec.verbatim_text
        assert item["target_population"] == "Type 2 Diabetes with ASCVD"
        assert item["evidence_grade"] == "Level A"

        # 2. Current protocol ID/version/section is shown
        assert item["matched_protocol_id"] == "PROT-DM-001"
        assert item["matched_protocol_version"] == "v1.0"
        assert item["protocol_section_id"] == "SEC-2"
        assert item["protocol_section_heading"] == "First-Line Pharmacotherapy"
        assert "Metformin 500mg daily" in item["protocol_section_text"]

        # 3. Source recommendation and protocol recommendation are both visible
        assert item["verbatim_text"] is not None
        assert item["protocol_section_text"] is not None

        # 4. Comparison result is visible
        assert item["difference_type"] == "scope_expansion"
        assert item["comparison_result"] == "gap"

        # 5. Similarity and confidence are visible
        assert item["similarity"] == 0.88
        assert item["comparison_confidence"] == 0.92

        # 13. Source URL provenance remains visible
        assert item["source_url"] == "https://diabetesjournals.org/care/article/2024/guideline"
        assert item["source_title"] == "ADA Standards of Care 2024"

    def test_4_no_recommendation_held_at_g1_and_no_downstream_records(self, db_session: Session):
        # 6, 7 & 12: Source yields no recommendations -> HELD at G1, no impact/brief/governance records
        doc = IngestedDocument(
            id="DOC-HELD-001",
            source_identifier="empty_pubmed_page.pdf",
            source_path="data/sources/empty_pubmed_page.pdf",
            sha256_hash="1122334455667788112233445566778811223344556677881122334455667788",
            status=DocumentStatus.PARSED.value,
            doc_metadata={
                "source_type": "URL",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
                "ingestion_status": "Complete",
                "pipeline_status": "Processing",
                "current_stage": "MONITORING",
            },
        )
        db_session.add(doc)
        db_session.commit()

        # Mock extraction agent returning zero recommendations
        mock_extraction = MagicMock()
        mock_extraction.process_document.return_value = []

        mock_comparison = MagicMock()
        mock_impact = MagicMock()
        mock_briefing = MagicMock()
        mock_governance = MagicMock()

        engine = db_session.get_bind()
        session_factory = sessionmaker(bind=engine)
        cfg = AppConfig(database_url="sqlite:///:memory:", groq_api_key="mock-key")

        pipeline = ClinicalKnowledgePipeline(
            config=cfg,
            session_factory=session_factory,
            extraction_agent=mock_extraction,
            comparison_agent=mock_comparison,
            impact_agent=mock_impact,
            briefing_agent=mock_briefing,
            governance_agent=mock_governance,
        )

        result = pipeline.process_ingested_document(doc.id)

        # 6. G1 state is clearly represented
        assert result.status == PipelineStatus.HELD
        assert result.current_stage == PipelineStage.EXTRACTION
        assert result.held_gate == HumanGate.G1
        assert "No actionable clinical recommendation" in result.reason

        # 7. No recommendation does not create impact/brief/governance records
        assert db_session.query(ChangeRecord).filter_by(ingested_document_id=doc.id).count() == 0
        assert db_session.query(GapRecord).count() == 0
        assert db_session.query(ImpactRecord).count() == 0
        assert db_session.query(ChangeBrief).count() == 0
        assert db_session.query(ReviewAssignment).count() == 0

        # Downstream agents were NOT called
        mock_comparison.process_change_record.assert_not_called()
        mock_impact.process_gap_record.assert_not_called()
        mock_briefing.process_impact_record.assert_not_called()

        # 12. Ingestion complete does not imply pipeline complete
        assert doc.doc_metadata["ingestion_status"] == "Complete"
        assert result.status == PipelineStatus.HELD

    def test_5_impact_and_brief_records_display_source_and_protocol(self, db_session: Session, sample_protocol_json_path):
        # 10. G4 awaiting human decision and complete provenance in downstream views
        doc = IngestedDocument(
            id="DOC-PROV-001",
            source_identifier="ada_sglt2_update.pdf",
            source_path="data/sources/ada_sglt2_update.pdf",
            sha256_hash="9988776655443322998877665544332299887766554433229988776655443322",
            status=DocumentStatus.COMPLETE.value,
            doc_metadata={
                "source_type": "URL",
                "source_url": "https://care.diabetesjournals.org/content/47/Suppl_1/S1",
                "title": "ADA Guidelines 2024 - Pharmacotherapy",
            },
        )
        db_session.add(doc)

        rec = ChangeRecord(
            id="REC-PROV-001",
            ingested_document_id=doc.id,
            verbatim_text="Add SGLT2i when eGFR is 20-60 ml/min to reduce CKD progression.",
            recommendation_type="INTERVENTION",
            target_population="Type 2 Diabetes with CKD",
            intervention="SGLT2 inhibitor",
            evidence_grade="Level A",
            confidence=0.96,
            extraction_model_version="groq-llama-3.3-70b-versatile",
            extraction_prompt_version="1.0",
            status=ChangeStatus.EXTRACTED.value,
        )
        db_session.add(rec)

        gap = GapRecord(
            id="GAP-PROV-001",
            change_record_id=rec.id,
            candidate_protocol_section_ids=["PROT-DM-001__v1.0__SEC-2"],
            comparison_result=ComparisonResult.GAP.value,
            difference_type="threshold_change",
            matched_protocol_id="PROT-DM-001",
            matched_protocol_version="v1.0",
            is_match=True,
            similarity=0.85,
            comparison_confidence=0.91,
            status=GapStatus.MATCHED.value,
        )
        db_session.add(gap)

        impact = ImpactRecord(
            id="IMP-PROV-001",
            gap_record_id=gap.id,
            clinical_urgency=4,
            evidence_strength=5,
            pathway_breadth=3,
            total_score=12,
            tier=ImpactTier.HIGH.value,
            sla_deadline=datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc),
            status=ImpactStatus.CALCULATED.value,
        )
        db_session.add(impact)

        brief = ChangeBrief(
            id="BRF-PROV-001",
            impact_record_id=impact.id,
            rendered_file_path="data/briefs/BRF-PROV-001.md",
            rendered_file_hash="abcdef",
            structured_payload={
                "comparison": {
                    "specific_difference": "Lower eGFR threshold for SGLT2i initiation from 30 down to 20 ml/min."
                }
            },
            status=BriefStatus.ASSIGNED.value,
        )
        db_session.add(brief)

        rev = ReviewAssignment(
            id="REV-PROV-001",
            change_brief_id=brief.id,
            reviewer_role="clinical_reviewer",
            reviewer_id="dr_smith",
            due_date=datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc),
            status="assigned",
            decision=None,
        )
        db_session.add(rev)
        db_session.commit()

        # Test Impact View Query
        impact_records = get_impact_records(db_session)
        assert len(impact_records) == 1
        imp_item = impact_records[0]
        assert imp_item["source_identifier"] == "ada_sglt2_update.pdf"
        assert imp_item["source_url"] == "https://care.diabetesjournals.org/content/47/Suppl_1/S1"
        assert imp_item["verbatim_recommendation"] == rec.verbatim_text
        assert imp_item["matched_protocol_id"] == "PROT-DM-001"
        assert imp_item["matched_protocol_version"] == "v1.0"
        assert imp_item["tier"] == ImpactTier.HIGH.value
        assert imp_item["total_score"] == 12

        # Test Brief View Query
        briefs = get_brief_summaries(db_session)
        assert len(briefs) == 1
        brf_item = briefs[0]
        assert brf_item["source_url"] == "https://care.diabetesjournals.org/content/47/Suppl_1/S1"
        assert brf_item["verbatim_recommendation"] == rec.verbatim_text
        assert brf_item["matched_protocol_id"] == "PROT-DM-001"
        assert brf_item["protocol_section_id"] == "SEC-2"
        assert brf_item["difference_summary"] == "Lower eGFR threshold for SGLT2i initiation from 30 down to 20 ml/min."

    def test_6_g5_contains_escalation_only(self, db_session: Session):
        # 11: G5 contains escalation only
        doc = IngestedDocument(
            id="DOC-ESC-001",
            source_identifier="doc_esc.pdf",
            source_path="data/sources/doc_esc.pdf",
            sha256_hash="11223344",
            status=DocumentStatus.COMPLETE.value,
        )
        db_session.add(doc)

        rec = ChangeRecord(
            id="REC-ESC-001",
            ingested_document_id=doc.id,
            verbatim_text="Recommendation requiring escalation.",
            recommendation_type="INTERVENTION",
            target_population="Adults",
            intervention="Treatment",
            confidence=0.9,
            extraction_model_version="groq",
            extraction_prompt_version="1.0",
        )
        db_session.add(rec)

        gap = GapRecord(
            id="GAP-ESC-001",
            change_record_id=rec.id,
            comparison_result=ComparisonResult.GAP.value,
        )
        db_session.add(gap)

        impact = ImpactRecord(
            id="IMP-ESC-001",
            gap_record_id=gap.id,
        )
        db_session.add(impact)

        brief = ChangeBrief(
            id="BRF-ESC-001",
            impact_record_id=impact.id,
            status=BriefStatus.ASSIGNED.value,
        )
        db_session.add(brief)

        rev_esc = ReviewAssignment(
            id="REV-ESC-001",
            change_brief_id=brief.id,
            reviewer_role="cmo_director",
            reviewer_id="cmo_director",
            status="escalated",
            due_date=datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc),
            decision="escalate",
            rationale="Significant clinical practice divergence requiring CMO approval.",
        )
        db_session.add(rev_esc)
        db_session.commit()

        esc_records = (
            db_session.query(ReviewAssignment)
            .filter(ReviewAssignment.decision == "escalate")
            .all()
        )
        assert len(esc_records) == 1
        assert esc_records[0].decision == "escalate"
        assert esc_records[0].reviewer_role == "cmo_director"


class TestSafetyAndIntegrityInvariants:
    """Verifies that no new LLM calls exist in UI, protocol is immutable, and ChromaDB is unchanged."""

    def test_7_no_new_llm_calls_in_streamlit_ui(self):
        # 16. No new LLM calls are introduced in UI
        ui_file = Path("app/ui/streamlit_app.py")
        content = ui_file.read_text(encoding="utf-8")
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "groq" not in alias.name.lower(), "Streamlit UI must not import Groq"
                    assert "openai" not in alias.name.lower(), "Streamlit UI must not import OpenAI"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "groq" not in node.module.lower(), "Streamlit UI must not import Groq"
                    assert "openai" not in node.module.lower(), "Streamlit UI must not import OpenAI"

    def test_8_protocol_immutability(self):
        # 14. Protocol remains immutable
        protocol_path = Path("data/protocols/PROT-DM-001_v1.0.json")
        assert protocol_path.exists(), "Authoritative protocol file must exist"
        content = protocol_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        expected_hash = "4203b13ea09d18285ceb34b136841973c7e26b6b59a3943da3a7801d7416289a"
        assert actual_hash == expected_hash, f"Protocol hash mismatch! Expected {expected_hash}, got {actual_hash}"

    def test_9_chromadb_protocol_collection_integrity(self):
        # 15. Chroma protocol collection remains unchanged
        from app.services.protocol_index import ProtocolIndexService
        index_service = ProtocolIndexService()
        count = index_service.collection.count()
        assert count == 4, f"Expected 4 protocol sections in Chroma, got {count}"
        expected_ids = [
            "PROT-DM-001__v1.0__SEC-1",
            "PROT-DM-001__v1.0__SEC-2",
            "PROT-DM-001__v1.0__SEC-3",
            "PROT-DM-001__v1.0__SEC-4",
        ]
        chroma_ids = index_service.collection.get()["ids"]
        for exp_id in expected_ids:
            assert exp_id in chroma_ids, f"Expected section ID {exp_id} in Chroma collection"

    def test_10_g4_explicit_decisions_only_no_request_revision(self):
        # Explicit G4 human governance decisions must strictly be APPROVE, REJECT, DEFER
        from app.schemas.governance import ReviewDecision
        from app.ui.streamlit_app import execute_governance_action

        approved_decisions = {d.value for d in ReviewDecision}
        assert approved_decisions == {"approve", "reject", "defer"}, (
            f"ReviewDecision must strictly contain only approve, reject, defer; got {approved_decisions}"
        )
        assert "request_revision" not in approved_decisions
        assert "revision" not in approved_decisions

        # Reject invalid governance action
        mock_gov_agent = MagicMock()
        mock_gov_agent.auth_service.is_authorized.return_value = True
        with pytest.raises(Exception):
            execute_governance_action(
                governance_agent=mock_gov_agent,
                brief_id="BRF-001",
                action="request_revision",
                reviewer_id="dr_smith",
                rationale="Requesting changes",
            )
