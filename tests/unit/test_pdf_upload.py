"""Focused unit and integration tests for Streamlit PDF upload workflow.

Verifies:
1. PDF accepted
2. Non-PDF rejected
3. Upload saved to data/sources/
4. Filename traversal blocked
5. MonitoringAgent invoked
6. Existing pipeline invoked
7. Duplicate upload uses existing idempotency
8. Uploaded document appears in source listing
9. G1 remains a human hold
10. G2 remains a human hold
11. G3 remains explicit no-match
12. No automatic governance decision
13. Protocol unchanged
14. data/protocols not modified
15. data/chroma not modified
16. Streamlit UI adds no LLM calls
"""

import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.monitoring_agent import MonitoringAgent, ScanResult
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
from app.schemas.gaps import ComparisonResult, GapStatus
from app.schemas.orchestration import (
    ArtifactsManifest,
    HumanGate,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)
from app.services.config_service import AppConfig, load_config
from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.file_hash import compute_sha256
from app.ui.streamlit_app import (
    get_source_documents,
    handle_source_pdf_upload,
    render_upload_result_card,
    sanitize_filename,
    save_uploaded_pdf,
)


@pytest.fixture
def isolated_env(tmp_path: Path):
    """Provide isolated SQLite database, sources directory, and config."""
    db_path = tmp_path / "test_upload.db"
    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    # Copy or create sample protocol file
    sample_proto = protocol_dir / "PROT-DM-001_v1.0.json"
    proto_content = Path("data/protocols/PROT-DM-001_v1.0.json").read_text(encoding="utf-8")
    sample_proto.write_text(proto_content, encoding="utf-8")

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine=engine)

    config = AppConfig(
        database_url=f"sqlite:///{db_path}",
        source_dir=source_dir,
        protocol_dir=protocol_dir,
        groq_api_key="mock-key-no-live-calls",
    )

    valid_pdf_bytes = make_multipage_pdf_bytes([
        [
            "1. Clinical Guideline Section",
            "Adult patients with type 2 diabetes should initiate metformin 1000mg once daily.",
        ]
    ])

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "config": config,
        "source_dir": source_dir,
        "protocol_dir": protocol_dir,
        "sample_proto": sample_proto,
        "valid_pdf_bytes": valid_pdf_bytes,
    }

    engine.dispose()


# ==============================================================================
# TEST 1: PDF ACCEPTED
# ==============================================================================

def test_pdf_accepted(isolated_env):
    """Test that valid PDF filenames and bytes are accepted."""
    safe_name = sanitize_filename("metformin_update_2026.pdf")
    assert safe_name == "metformin_update_2026.pdf"

    saved = save_uploaded_pdf(
        file_bytes=isolated_env["valid_pdf_bytes"],
        filename="metformin_update_2026.pdf",
        source_dir=isolated_env["source_dir"],
    )
    assert saved.exists()
    assert saved.name == "metformin_update_2026.pdf"
    assert saved.stat().st_size == len(isolated_env["valid_pdf_bytes"])


# ==============================================================================
# TEST 2: NON-PDF REJECTED
# ==============================================================================

def test_non_pdf_rejected(isolated_env):
    """Test that non-PDF files are strictly rejected."""
    invalid_filenames = [
        "guideline.exe",
        "script.sh",
        "document.docx",
        "notes.txt",
        "archive.zip",
        "no_extension",
        "",
        "   ",
    ]
    for fn in invalid_filenames:
        with pytest.raises(ValueError):
            sanitize_filename(fn)

        with pytest.raises(ValueError):
            save_uploaded_pdf(
                file_bytes=b"invalid content",
                filename=fn,
                source_dir=isolated_env["source_dir"],
            )


# ==============================================================================
# TEST 3: UPLOAD SAVED TO DATA/SOURCES
# ==============================================================================

def test_upload_saved_to_data_sources(isolated_env):
    """Test that uploaded PDF is saved strictly in the designated source directory."""
    saved = save_uploaded_pdf(
        file_bytes=isolated_env["valid_pdf_bytes"],
        filename="diabetes_guideline.pdf",
        source_dir=isolated_env["source_dir"],
    )
    assert saved.parent.resolve() == isolated_env["source_dir"].resolve()
    assert saved.is_file()

    # Verify not saved in disallowed directories
    for disallowed in [
        isolated_env["protocol_dir"],
        isolated_env["config"].chroma_dir,
        isolated_env["config"].output_dir,
    ]:
        assert not (disallowed / "diabetes_guideline.pdf").exists()


# ==============================================================================
# TEST 4: FILENAME TRAVERSAL BLOCKED
# ==============================================================================

def test_filename_traversal_blocked(isolated_env):
    """Test that path traversal attempts are neutralized and confined to source directory."""
    traversal_attacks = [
        "../../evil.pdf",
        "..\\..\\evil.pdf",
        "/etc/passwd.pdf",
        "C:\\Windows\\System32\\calc.pdf",
        "....//....//evil.pdf",
        "nested/../../traversal.pdf",
    ]
    for attack in traversal_attacks:
        safe_name = sanitize_filename(attack)
        assert "/" not in safe_name
        assert "\\" not in safe_name
        assert ".." not in safe_name
        assert safe_name.endswith(".pdf")

        saved = save_uploaded_pdf(
            file_bytes=isolated_env["valid_pdf_bytes"],
            filename=attack,
            source_dir=isolated_env["source_dir"],
        )
        assert saved.parent.resolve() == isolated_env["source_dir"].resolve()
        assert saved.exists()


# ==============================================================================
# TEST 5: MONITORING AGENT INVOKED
# ==============================================================================

def test_monitoring_agent_invoked(isolated_env):
    """Test that MonitoringAgent is explicitly invoked during upload processing."""
    mock_monitoring = MagicMock(spec=MonitoringAgent)
    mock_monitoring.scan.return_value = ScanResult(discovered=1, new_documents=1, skipped=0, failed=0)

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-123",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    handle_source_pdf_upload(
        uploaded_name="new_source.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        monitoring_agent=mock_monitoring,
        pipeline=mock_pipeline,
    )

    assert mock_monitoring.scan.called


# ==============================================================================
# TEST 6: EXISTING PIPELINE INVOKED
# ==============================================================================

def test_existing_pipeline_invoked(isolated_env):
    """Test that ClinicalKnowledgePipeline is invoked with the saved document path."""
    mock_monitoring = MagicMock(spec=MonitoringAgent)
    mock_monitoring.scan.return_value = ScanResult(discovered=1, new_documents=1, skipped=0, failed=0)

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-456",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    handle_source_pdf_upload(
        uploaded_name="pipeline_test.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        monitoring_agent=mock_monitoring,
        pipeline=mock_pipeline,
    )

    assert mock_pipeline.process_document.called
    called_path = mock_pipeline.process_document.call_args[0][0]
    assert Path(called_path).name == "pipeline_test.pdf"


# ==============================================================================
# TEST 7: DUPLICATE UPLOAD USES EXISTING IDEMPOTENCY
# ==============================================================================

def test_duplicate_upload_uses_existing_idempotency(isolated_env):
    """Test that uploading the same document twice leverages MonitoringAgent SHA-256 idempotency."""
    # First upload using actual MonitoringAgent against SQLite
    m_agent = MonitoringAgent(
        source_dir=isolated_env["source_dir"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
    )

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-dup-1",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    res1 = handle_source_pdf_upload(
        uploaded_name="idempotent_doc.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        monitoring_agent=m_agent,
        pipeline=mock_pipeline,
    )
    assert res1["scan_result"].new_documents == 1
    assert res1["scan_result"].skipped == 0

    # Second upload with identical bytes
    res2 = handle_source_pdf_upload(
        uploaded_name="idempotent_doc.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        monitoring_agent=m_agent,
        pipeline=mock_pipeline,
    )
    # Existing MonitoringAgent recognized unchanged SHA-256 and skipped re-ingestion
    assert res2["scan_result"].skipped == 1

    # Verify database has exactly 1 document record
    with isolated_env["session_factory"]() as session:
        count = session.query(IngestedDocument).count()
        assert count == 1


# ==============================================================================
# TEST 8: UPLOADED DOCUMENT APPEARS IN SOURCE LISTING
# ==============================================================================

def test_uploaded_document_appears_in_source_listing(isolated_env):
    """Test that newly ingested document appears in get_source_documents()."""
    m_agent = MonitoringAgent(
        source_dir=isolated_env["source_dir"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
    )

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-listing-test",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    res = handle_source_pdf_upload(
        uploaded_name="clinical_listing.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        monitoring_agent=m_agent,
        pipeline=mock_pipeline,
    )

    docs = get_source_documents(isolated_env["session_factory"])
    assert len(docs) == 1
    assert docs[0]["source_identifier"] == "clinical_listing"
    assert docs[0]["sha256_hash"] == res["sha256_hash"]
    assert docs[0]["status"] == DocumentStatus.PARSED.value


# ==============================================================================
# TEST 9: G1 REMAINS A HUMAN HOLD
# ==============================================================================

def test_g1_remains_human_hold(isolated_env):
    """Test that G1 low-confidence extraction pauses pipeline and reports G1 hold."""
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-g1",
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.EXTRACTION,
        blocked_stage=PipelineStage.EXTRACTION,
        held_gate=HumanGate.G1,
        reason="Document held for G1 extraction review.",
    )

    res = handle_source_pdf_upload(
        uploaded_name="g1_doc.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        pipeline=mock_pipeline,
    )

    assert res["gate_status"] == "G1"
    assert "paused at G1" in res["result_message"]
    assert "human extraction review required" in res["result_message"]


# ==============================================================================
# TEST 10: G2 REMAINS A HUMAN HOLD
# ==============================================================================

def test_g2_remains_human_hold(isolated_env):
    """Test that G2 ambiguous comparison pauses pipeline and reports G2 hold."""
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-g2",
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.COMPARISON,
        blocked_stage=PipelineStage.COMPARISON,
        held_gate=HumanGate.G2,
        reason="Comparison output requires G2 human review.",
    )

    res = handle_source_pdf_upload(
        uploaded_name="g2_doc.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        pipeline=mock_pipeline,
    )

    assert res["gate_status"] == "G2"
    assert "paused at G2" in res["result_message"]
    assert "comparison review required" in res["result_message"]


# ==============================================================================
# TEST 11: G3 REMAINS EXPLICIT NO-MATCH
# ==============================================================================

def test_g3_remains_explicit_no_match(isolated_env):
    """Test that G3 no-match pauses pipeline and reports committee review required."""
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-g3",
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.COMPARISON,
        blocked_stage=PipelineStage.COMPARISON,
        held_gate=HumanGate.G3,
        reason="Explicit no-match finding requires G3 clinical committee confirmation.",
    )

    res = handle_source_pdf_upload(
        uploaded_name="g3_doc.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        pipeline=mock_pipeline,
    )

    assert res["gate_status"] == "G3"
    assert "No matching institutional protocol section found" in res["result_message"]
    assert "G3 committee review required" in res["result_message"]


# ==============================================================================
# TEST 12: NO AUTOMATIC GOVERNANCE DECISION
# ==============================================================================

def test_no_automatic_governance_decision(isolated_env):
    """Test that upload handler NEVER approves, rejects, defers, or closes a brief."""
    # Seed a draft change brief in SQLite
    with isolated_env["session_factory"]() as session:
        doc = IngestedDocument(
            source_identifier="gov_test",
            source_path="/path/gov_test.pdf",
            sha256_hash="dummy_hash_123",
            status=DocumentStatus.COMPLETE.value,
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Test recommendation",
            recommendation_type="treatment",
            target_population="Adults",
            intervention="Test intervention",
            confidence=0.95,
            extraction_model_version="test",
            extraction_prompt_version="1.0",
        )
        session.add(change)
        session.flush()

        gap = GapRecord(
            change_record_id=change.id,
            comparison_result="gap",
            status=GapStatus.MATCHED.value,
        )
        session.add(gap)
        session.flush()

        impact = ImpactRecord(
            gap_record_id=gap.id,
            clinical_urgency=3,
            evidence_strength=3,
            pathway_breadth=3,
            total_score=9,
            tier="High",
        )
        session.add(impact)
        session.flush()

        brief = ChangeBrief(
            impact_record_id=impact.id,
            status=BriefStatus.ASSIGNED.value,
            schema_version="1.0",
        )
        session.add(brief)
        session.commit()
        brief_id = brief.id

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id=doc.id,
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.GOVERNANCE,
        blocked_stage=PipelineStage.GOVERNANCE,
        held_gate=HumanGate.G4,
        reason="ChangeBrief requires an explicit human governance decision.",
        artifacts=ArtifactsManifest(
            document_id=doc.id,
            change_brief_ids=[brief_id],
        ),
    )

    res = handle_source_pdf_upload(
        uploaded_name="gov_test.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        pipeline=mock_pipeline,
    )

    assert res["gate_status"] == "G4"
    assert "awaiting G4 authorized clinical governance decision" in res["result_message"]

    # Verify brief status in database remains ASSIGNED (unaltered)
    with isolated_env["session_factory"]() as session:
        b = session.get(ChangeBrief, brief_id)
        assert b.status == BriefStatus.ASSIGNED.value
        assert b.status not in (
            BriefStatus.DECIDED.value,
            BriefStatus.CLOSED.value,
            BriefStatus.DEFERRED.value,
        )


# ==============================================================================
# TEST 13: PROTOCOL UNCHANGED
# ==============================================================================

def test_protocol_unchanged(isolated_env):
    """Test that PROT-DM-001_v1.0.json remains byte-for-byte identical before and after upload."""
    proto_file = Path("data/protocols/PROT-DM-001_v1.0.json")
    before_hash = compute_sha256(proto_file)
    before_bytes = proto_file.read_bytes()

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-proto-test",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    handle_source_pdf_upload(
        uploaded_name="check_protocol_immutability.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        pipeline=mock_pipeline,
    )

    after_hash = compute_sha256(proto_file)
    after_bytes = proto_file.read_bytes()

    assert before_hash == after_hash
    assert before_bytes == after_bytes


# ==============================================================================
# TEST 14: DATA/PROTOCOLS NOT MODIFIED
# ==============================================================================

def test_data_protocols_not_modified(isolated_env):
    """Test that no files are added, removed, or changed in data/protocols/."""
    proto_dir = Path("data/protocols")
    before_files = sorted([p.name for p in proto_dir.iterdir()])
    before_hashes = {p.name: compute_sha256(p) for p in proto_dir.iterdir()}

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-proto-dir-test",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    handle_source_pdf_upload(
        uploaded_name="proto_dir_check.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        pipeline=mock_pipeline,
    )

    after_files = sorted([p.name for p in proto_dir.iterdir()])
    after_hashes = {p.name: compute_sha256(p) for p in proto_dir.iterdir()}

    assert before_files == after_files
    assert before_hashes == after_hashes


# ==============================================================================
# TEST 15: DATA/CHROMA NOT MODIFIED
# ==============================================================================

def test_data_chroma_not_modified(isolated_env):
    """Test that ChromaDB collections and vector records remain unchanged."""
    chroma_dir = Path("data/chroma")
    if not chroma_dir.exists():
        pytest.skip("Chroma directory does not exist.")

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        coll = client.get_collection("protocol_sections")
        before_count = coll.count()
        before_ids = sorted(coll.get().get("ids", []))
    except Exception:
        pytest.skip("ChromaDB collection protocol_sections not available.")

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-chroma-test",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    handle_source_pdf_upload(
        uploaded_name="chroma_check.pdf",
        uploaded_bytes=isolated_env["valid_pdf_bytes"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        pipeline=mock_pipeline,
    )

    after_count = coll.count()
    after_ids = sorted(coll.get().get("ids", []))

    assert before_count == after_count
    assert before_ids == after_ids


# ==============================================================================
# TEST 16: STREAMLIT UI ADDS NO LLM CALLS
# ==============================================================================

def test_streamlit_ui_adds_no_llm_calls():
    """Verify Streamlit UI contains zero LLM client instantiation or API calls."""
    import app.ui.streamlit_app as ui_module

    source_code = inspect.getsource(ui_module)

    # UI must not instantiate or directly import Groq client or OpenAI client
    assert "Groq(" not in source_code
    assert "OpenAI(" not in source_code
    assert "import groq" not in source_code
    assert "from groq" not in source_code
    assert "import openai" not in source_code
    assert "from openai" not in source_code
    assert "chat.completions" not in source_code
    assert "SharedLLMClient" not in source_code

    # The upload logic delegates solely to MonitoringAgent and ClinicalKnowledgePipeline
    assert "MonitoringAgent" in source_code
    assert "ClinicalKnowledgePipeline" in source_code
