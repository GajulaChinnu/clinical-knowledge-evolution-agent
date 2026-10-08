"""Focused tests for the CKEA URL Ingestion Workflow and Dual-Ingestion Integration.

Covers all 23 URL requirements:
1. HTTPS URL accepted
2. HTTP URL accepted
3. Malformed URL rejected
4. Unsupported scheme rejected
5. Direct PDF URL handled
6. HTML URL handled
7. URL-derived file saved only under data/sources
8. Unsafe URL-derived filename prevented
9. HTTP failure handled
10. Timeout handled
11. Empty content handled
12. Source URL provenance preserved
13. SHA-256/idempotency preserved
14. MonitoringAgent invoked
15. ClinicalKnowledgePipeline invoked
16. G1 remains human-gated
17. G2 remains human-gated
18. G3 remains explicit
19. G4 has no automatic decision
20. G5 does not create a decision
21. Protocol file unchanged
22. Chroma protocol collection unchanged
23. No LLM calls from UI/ingestion layer
"""

from datetime import datetime, timezone
import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import httpx
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
    Notification,
    ReviewAssignment,
)
from app.orchestration.pipeline import ClinicalKnowledgePipeline
from app.schemas.briefs import BriefStatus
from app.schemas.changes import ChangeStatus
from app.schemas.documents import DocumentStatus
from app.schemas.gaps import GapStatus
from app.schemas.orchestration import (
    ArtifactsManifest,
    HumanGate,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)
from app.services.config_service import AppConfig
from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.file_hash import compute_sha256
from app.services.jina_reader_service import JinaReaderService, JinaResult
from app.services.url_ingestion_service import (
    ContentChallengeError,
    EmptyContentError,
    HTTPFetchError,
    InvalidURLError,
    URLIngestionResult,
    URLIngestionService,
    derive_safe_filename_from_url,
    detect_content_challenge,
    fetch_url_content,
    validate_source_url,
)
from app.ui.streamlit_app import (
    get_source_documents,
    handle_source_url_upload,
    render_upload_result_card,
)


@pytest.fixture
def isolated_env(tmp_path: Path):
    """Provide isolated SQLite database, sources directory, and config."""
    db_path = tmp_path / "test_url.db"
    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
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

    sample_html = """<!DOCTYPE html>
    <html>
    <head><title>Diabetes Pharmacotherapy Guidelines 2026</title></head>
    <body>
        <h1>1. Clinical Recommendations for Type 2 Diabetes</h1>
        <p>Adult patients diagnosed with type 2 diabetes should initiate metformin at 1000mg daily.</p>
        <h2>Section 2: Renal Safety Adjustments</h2>
        <p>Monitor eGFR regularly and titrate dosages as indicated by clinical evidence.</p>
    </body>
    </html>
    """

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "config": config,
        "source_dir": source_dir,
        "protocol_dir": protocol_dir,
        "sample_proto": sample_proto,
        "valid_pdf_bytes": valid_pdf_bytes,
        "sample_html": sample_html,
    }

    engine.dispose()




PUBLIC_RESOLVER = lambda host: ["93.184.216.34"]


class FakeJina:
    """Stand-in Jina provider recording every invocation."""

    def __init__(self, content="", title="", warnings=None, error=None):
        self.calls = []
        self._content, self._title, self._warnings, self._error = content, title, warnings or [], error

    is_configured = True

    def retrieve(self, url):
        self.calls.append(url)
        if self._error:
            raise self._error
        return JinaResult(content=self._content, title=self._title, resolved_url=url,
                          status_code=200, attempts=1, warnings=list(self._warnings))


def make_url_service(config, handler, jina=None, resolver=PUBLIC_RESOLVER, **overrides):
    """URLIngestionService with mocked HTTP transport, resolver and Jina provider."""
    cfg = config.model_copy(update=overrides) if overrides else config
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return URLIngestionService(
        config=cfg,
        jina_service=jina or FakeJina(),
        http_client=client,
        host_resolver=resolver,
    )


def pdf_handler(pdf_bytes):
    def handler(request):
        return httpx.Response(200, content=b"" if request.method == "HEAD" else pdf_bytes,
                              headers={"content-type": "application/pdf"})
    return handler


def html_handler(html_bytes, status=200):
    seen = []
    def handler(request):
        seen.append((request.method, str(request.url)))
        return httpx.Response(status, content=b"" if request.method == "HEAD" else html_bytes,
                              headers={"content-type": "text/html; charset=utf-8"})
    handler.seen = seen
    return handler

# ==============================================================================
# TEST 1: HTTPS URL ACCEPTED
# ==============================================================================

def test_https_url_accepted():
    parsed = validate_source_url("https://clinicalguidelines.org/t2d_update.pdf")
    assert parsed.scheme == "https"
    assert parsed.netloc == "clinicalguidelines.org"


# ==============================================================================
# TEST 2: HTTP URL ACCEPTED
# ==============================================================================

def test_http_url_accepted():
    parsed = validate_source_url("http://internal-health.org/protocols/guideline.html")
    assert parsed.scheme == "http"
    assert parsed.netloc == "internal-health.org"


# ==============================================================================
# TEST 3: MALFORMED URL REJECTED
# ==============================================================================

def test_malformed_url_rejected():
    malformed_urls = [
        "",
        "   ",
        "not_a_valid_url",
        "https://",
        "http://",
        "http://has space.com/doc",
        "://missing-scheme.org",
    ]
    for url in malformed_urls:
        with pytest.raises(InvalidURLError):
            validate_source_url(url)


# ==============================================================================
# TEST 4: UNSUPPORTED SCHEME REJECTED
# ==============================================================================

def test_unsupported_scheme_rejected():
    unsupported_urls = [
        "ftp://guidelines.org/doc.pdf",
        "file:///etc/passwd",
        "data:text/plain;base64,SGVsbG8=",
        "javascript:alert('xss')",
        "gopher://ancient.org/file",
    ]
    for url in unsupported_urls:
        with pytest.raises(InvalidURLError):
            validate_source_url(url)


# ==============================================================================
# TEST 5: DIRECT PDF URL HANDLED
# ==============================================================================

def test_direct_pdf_url_handled(isolated_env):
    pdf_url = "https://health.org/guidelines/metformin_2026.pdf"
    jina = FakeJina()
    service = make_url_service(isolated_env["config"], pdf_handler(isolated_env["valid_pdf_bytes"]), jina=jina)

    res = service.ingest_url(pdf_url, isolated_env["source_dir"])
    assert res.is_direct_pdf is True
    assert res.routing_decision == "pdf_direct"
    assert res.retrieval_provider == "Direct HTTP"
    assert jina.calls == []
    assert res.saved_path.exists()
    assert res.saved_path.suffix == ".pdf"
    assert res.saved_path.parent.resolve() == isolated_env["source_dir"].resolve()
    assert res.content_sha256 == hashlib.sha256(isolated_env["valid_pdf_bytes"]).hexdigest()


# ==============================================================================
# TEST 6: HTML URL HANDLED
# ==============================================================================

def test_html_url_handled(isolated_env):
    html_url = "https://guidelines.org/diabetes/recommendations"
    jina = FakeJina(content="Mocked markdown content from Jina Reader for testing", title="Mocked HTML Page")
    service = make_url_service(isolated_env["config"], html_handler(isolated_env["sample_html"].encode()), jina=jina)

    res = service.ingest_url(html_url, isolated_env["source_dir"])
    assert res.is_direct_pdf is False
    assert res.routing_decision == "jina"
    assert res.retrieval_provider == "Jina Reader"
    assert jina.calls == [html_url]
    assert res.saved_path.exists()
    assert res.saved_path.suffix == ".md"
    assert res.artifact_format == "markdown"
    assert res.saved_path.read_text(encoding="utf-8") == "Mocked markdown content from Jina Reader for testing"
    assert res.file_size_bytes > 0


# ==============================================================================
# TEST 7: URL-DERIVED FILE SAVED ONLY UNDER DATA/SOURCES
# ==============================================================================

def test_url_derived_file_saved_only_under_data_sources(isolated_env):
    url = "https://org.org/guideline.pdf"
    service = make_url_service(isolated_env["config"], pdf_handler(isolated_env["valid_pdf_bytes"]))
    res = service.ingest_url(url, isolated_env["source_dir"])
    assert res.saved_path.parent.resolve() == isolated_env["source_dir"].resolve()

    # Verify not present in restricted directories
    assert not (isolated_env["protocol_dir"] / res.filename).exists()
    assert not (isolated_env["config"].chroma_dir / res.filename).exists()
    assert not (isolated_env["config"].output_dir / res.filename).exists()


# ==============================================================================
# TEST 8: UNSAFE URL-DERIVED FILENAME PREVENTED
# ==============================================================================

def test_unsafe_url_derived_filename_prevented(isolated_env):
    unsafe_urls = [
        "https://evil.org/../../etc/passwd.pdf",
        "https://evil.org/..%2f..%2fevil.pdf",
        "https://evil.org/subdir/../../../root.pdf",
    ]
    for u in unsafe_urls:
        safe_fn = derive_safe_filename_from_url(u, content_hash="mock_hash", is_pdf=True)
        assert "/" not in safe_fn
        assert "\\" not in safe_fn
        assert ".." not in safe_fn
        assert safe_fn.endswith(".pdf")


# ==============================================================================
# TEST 9: HTTP FAILURE HANDLED
# ==============================================================================

def test_http_failure_handled():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(HTTPFetchError) as exc_info:
        fetch_url_content("https://site.org/404", client=client, resolver=PUBLIC_RESOLVER)
    assert "404" in str(exc_info.value)


# ==============================================================================
# TEST 10: TIMEOUT HANDLED
# ==============================================================================

def test_timeout_handled():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(HTTPFetchError) as exc_info:
        fetch_url_content("https://slow.org/doc.pdf", timeout_seconds=1.0, client=client, resolver=PUBLIC_RESOLVER)
    assert "timed out" in str(exc_info.value).lower()


# ==============================================================================
# TEST 11: EMPTY CONTENT HANDLED
# ==============================================================================

def test_empty_content_handled(isolated_env):
    service = make_url_service(isolated_env["config"], html_handler(b"<html></html>"), jina=FakeJina(content=""))
    with pytest.raises(EmptyContentError):
        service.ingest_url("https://empty.org", isolated_env["source_dir"])


# ==============================================================================
# TEST 12: SOURCE URL PROVENANCE PRESERVED
# ==============================================================================

def test_source_url_provenance_preserved(isolated_env):
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-url-prov",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "provenance_test.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    file_hash = compute_sha256(test_file)

    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://guidelines.nih.gov/t2d.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=file_hash,
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=len(isolated_env["valid_pdf_bytes"]),
        content_type="application/pdf",
    )

    res = handle_source_url_upload(
        url="https://guidelines.nih.gov/t2d.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )

    assert res["source_url"] == "https://guidelines.nih.gov/t2d.pdf"
    assert res["input_type"] == "URL"

    # Verify provenance attached to database record
    with isolated_env["session_factory"]() as session:
        doc = session.query(IngestedDocument).filter_by(sha256_hash=file_hash).first()
        assert doc is not None
        assert doc.doc_metadata is not None
        assert doc.doc_metadata.get("source_url") == "https://guidelines.nih.gov/t2d.pdf"
        assert doc.doc_metadata.get("source_type") == "url"
        assert "retrieval_timestamp" in doc.doc_metadata


# ==============================================================================
# TEST 13: SHA-256 / IDEMPOTENCY PRESERVED
# ==============================================================================

def test_sha256_idempotency_preserved(isolated_env):
    m_agent = MonitoringAgent(
        source_dir=isolated_env["source_dir"],
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
    )
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-dup-url",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    test_file = isolated_env["source_dir"] / "idempotent_url.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    file_hash = compute_sha256(test_file)

    mock_service = MagicMock(spec=URLIngestionService)
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://site.org/doc.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=file_hash,
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=len(isolated_env["valid_pdf_bytes"]),
        content_type="application/pdf",
    )

    # First ingestion
    res1 = handle_source_url_upload(
        url="https://site.org/doc.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        monitoring_agent=m_agent,
        pipeline=mock_pipeline,
    )

    # Second ingestion of same document
    res2 = handle_source_url_upload(
        url="https://site.org/doc.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        monitoring_agent=m_agent,
        pipeline=mock_pipeline,
    )
    # Existing MonitoringAgent recognized unchanged SHA-256 and skipped re-ingestion
    assert res2["gate_status"] == "No Change"
    assert "No source change detected" in res2["result_message"]

    with isolated_env["session_factory"]() as session:
        count = session.query(IngestedDocument).count()
        assert count == 1


# ==============================================================================
# TEST 14: MONITORING AGENT INVOKED
# ==============================================================================

def test_monitoring_agent_invoked(isolated_env):
    mock_monitoring = MagicMock(spec=MonitoringAgent)
    mock_monitoring.ingest_known_file.return_value = ScanResult(discovered=1, new_documents=1, skipped=0, failed=0, ingested_document_ids=["doc_id"])

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "agent_invoked.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://nih.gov/update.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=len(isolated_env["valid_pdf_bytes"]),
        content_type="application/pdf",
    )

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-invoke-test",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    handle_source_url_upload(
        url="https://nih.gov/update.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        monitoring_agent=mock_monitoring,
        pipeline=mock_pipeline,
    )
    assert mock_monitoring.ingest_known_file.called


# ==============================================================================
# TEST 15: CLINICAL KNOWLEDGE PIPELINE INVOKED
# ==============================================================================

def test_clinical_knowledge_pipeline_invoked(isolated_env):
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-pipe-test",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "pipeline_call.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://who.int/guideline.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=len(isolated_env["valid_pdf_bytes"]),
        content_type="application/pdf",
    )

    handle_source_url_upload(
        url="https://who.int/guideline.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )
    assert mock_pipeline.process_document.called
    called_doc = mock_pipeline.process_document.call_args[0][0]
    assert isinstance(called_doc, str)  # should be doc_id


# ==============================================================================
# TEST 16: G1 REMAINS HUMAN-GATED
# ==============================================================================

def test_g1_remains_human_gated(isolated_env):
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-g1-url",
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.EXTRACTION,
        blocked_stage=PipelineStage.EXTRACTION,
        held_gate=HumanGate.G1,
        reason="Extraction confidence below threshold.",
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "g1_url.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://site.org/g1.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=100,
        content_type="application/pdf",
    )

    res = handle_source_url_upload(
        url="https://site.org/g1.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )
    assert res["gate_status"] == "G1"
    assert "human extraction review required" in res["result_message"]


# ==============================================================================
# TEST 17: G2 REMAINS HUMAN-GATED
# ==============================================================================

def test_g2_remains_human_gated(isolated_env):
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-g2-url",
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.COMPARISON,
        blocked_stage=PipelineStage.COMPARISON,
        held_gate=HumanGate.G2,
        reason="Comparison ambiguity.",
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "g2_url.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://site.org/g2.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=100,
        content_type="application/pdf",
    )

    res = handle_source_url_upload(
        url="https://site.org/g2.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )
    assert res["gate_status"] == "G2"
    assert "comparison review required" in res["result_message"]


# ==============================================================================
# TEST 18: G3 REMAINS EXPLICIT
# ==============================================================================

def test_g3_remains_explicit(isolated_env):
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-g3-url",
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.COMPARISON,
        blocked_stage=PipelineStage.COMPARISON,
        held_gate=HumanGate.G3,
        reason="No matching protocol section found.",
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "g3_url.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://site.org/g3.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=100,
        content_type="application/pdf",
    )

    res = handle_source_url_upload(
        url="https://site.org/g3.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )
    assert res["gate_status"] == "G3"
    assert "No matching institutional protocol section found" in res["result_message"]


# ==============================================================================
# TEST 19: G4 HAS NO AUTOMATIC DECISION
# ==============================================================================

def test_g4_has_no_automatic_decision(isolated_env):
    with isolated_env["session_factory"]() as session:
        doc = IngestedDocument(
            source_identifier="url_g4",
            source_path="/path/url_g4.pdf",
            sha256_hash="dummy_url_g4_hash",
            status=DocumentStatus.COMPLETE.value,
        )
        session.add(doc)
        session.flush()

        change = ChangeRecord(
            ingested_document_id=doc.id,
            verbatim_text="Recommendation text",
            recommendation_type="treatment",
            target_population="Adults",
            intervention="Intervention",
            confidence=0.98,
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
            clinical_urgency=4,
            evidence_strength=4,
            pathway_breadth=4,
            total_score=12,
            tier="Critical",
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

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "url_g4.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://site.org/g4.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=100,
        content_type="application/pdf",
    )

    res = handle_source_url_upload(
        url="https://site.org/g4.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )
    assert res["gate_status"] == "G4"
    assert "awaiting G4 authorized clinical governance decision" in res["result_message"]

    # Brief status in database remains ASSIGNED (unaltered)
    with isolated_env["session_factory"]() as session:
        b = session.get(ChangeBrief, brief_id)
        assert b.status == BriefStatus.ASSIGNED.value


# ==============================================================================
# TEST 20: G5 DOES NOT CREATE A DECISION
# ==============================================================================

def test_g5_does_not_create_a_decision(isolated_env):
    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-g5-url",
        status=PipelineStatus.HELD,
        current_stage=PipelineStage.GOVERNANCE,
        blocked_stage=PipelineStage.GOVERNANCE,
        held_gate=HumanGate.G5,
        reason="SLA escalation active.",
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "url_g5.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://site.org/g5.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=100,
        content_type="application/pdf",
    )

    res = handle_source_url_upload(
        url="https://site.org/g5.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )
    assert res["gate_status"] == "G5"
    assert "Review SLA escalation active (G5 notice)" in res["result_message"]


# ==============================================================================
# TEST 21: PROTOCOL FILE UNCHANGED
# ==============================================================================

def test_protocol_file_unchanged(isolated_env):
    proto_file = Path("data/protocols/PROT-DM-001_v1.0.json")
    before_hash = compute_sha256(proto_file)
    before_bytes = proto_file.read_bytes()

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-url-proto-immutability",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "url_proto_test.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://proto.org/guide.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=100,
        content_type="application/pdf",
    )

    handle_source_url_upload(
        url="https://proto.org/guide.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )

    after_hash = compute_sha256(proto_file)
    after_bytes = proto_file.read_bytes()

    assert before_hash == after_hash
    assert before_bytes == after_bytes


# ==============================================================================
# TEST 22: CHROMA PROTOCOL COLLECTION UNCHANGED
# ==============================================================================

def test_chroma_protocol_collection_unchanged(isolated_env):
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
        pytest.skip("ChromaDB collection not available.")

    mock_pipeline = MagicMock(spec=ClinicalKnowledgePipeline)
    mock_pipeline.process_document.return_value = PipelineResult(
        document_id="doc-url-chroma-test",
        status=PipelineStatus.COMPLETED,
        current_stage=PipelineStage.EXTRACTION,
    )

    mock_service = MagicMock(spec=URLIngestionService)
    test_file = isolated_env["source_dir"] / "url_chroma.pdf"
    test_file.write_bytes(isolated_env["valid_pdf_bytes"])
    mock_service.ingest_url.return_value = URLIngestionResult(
        source_url="https://chroma.org/doc.pdf",
        saved_path=test_file,
        filename=test_file.name,
        content_sha256=compute_sha256(test_file),
        retrieval_timestamp=datetime.now(timezone.utc),
        is_direct_pdf=True,
        file_size_bytes=100,
        content_type="application/pdf",
    )

    handle_source_url_upload(
        url="https://chroma.org/doc.pdf",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
        pipeline=mock_pipeline,
    )

    after_count = coll.count()
    after_ids = sorted(coll.get().get("ids", []))

    assert before_count == after_count
    assert before_ids == after_ids


# ==============================================================================
# TEST 23: NO LLM CALLS FROM UI OR INGESTION LAYER
# ==============================================================================

def test_no_llm_calls_from_ui_or_ingestion_layer():
    import app.services.source_ingestion_service as ingest_mod
    import app.services.url_ingestion_service as url_mod

    sources = [inspect.getsource(url_mod), inspect.getsource(ingest_mod)]
    sources += [f.read_text(encoding="utf-8") for f in Path("app/ui").rglob("*.py")]

    for src in sources:
        assert "Groq(" not in src
        assert "OpenAI(" not in src
        assert "import groq" not in src
        assert "from groq" not in src
        assert "import openai" not in src
        assert "from openai" not in src
        assert "chat.completions" not in src
        assert "SharedLLMClient" not in src





def test_cookie_and_bot_challenge_detection():
    # 8 & 9. Detection of cookie challenge, cloudflare, captcha, login error
    assert detect_content_challenge("Cookies must be enabled for pubmed.ncbi.nlm.nih.gov and reload") is not None
    assert detect_content_challenge("Please enable cookies to continue.") is not None
    assert detect_content_challenge("Checking your browser before accessing.") is not None
    assert detect_content_challenge("Please verify you are human by solving captcha.") is not None
    assert detect_content_challenge("Access Denied: 403 Forbidden") is not None
    assert detect_content_challenge("Login required to view this document") is not None
    assert detect_content_challenge("Just a moment... Attention Required! | Cloudflare") is not None

    # Legitimate clinical content does not trigger challenge detection
    assert detect_content_challenge("Clinical Practice Guideline: Type 2 Diabetes Management") is None
    assert detect_content_challenge("Initial therapy with Metformin 500mg daily.") is None


def test_content_challenge_raises_error_before_pdf(isolated_env, tmp_path: Path):
    out_dir = tmp_path / "challenge_out"
    out_dir.mkdir()
    # Challenge text returned by the retrieval provider raises and writes no file
    jina = FakeJina(content="Cookies must be enabled to view this page.", title="Challenge")
    service = make_url_service(isolated_env["config"], html_handler(b"<html></html>"), jina=jina)

    with pytest.raises(ContentChallengeError) as exc_info:
        service.ingest_url("https://example.com/blocked", source_dir=out_dir)

    assert "challenge" in str(exc_info.value).lower()
    assert list(out_dir.iterdir()) == []


def test_handle_source_url_upload_challenge_blocked_state(isolated_env):
    # 8, 9. handle_source_url_upload returns structured blocked state when challenge detected
    mock_service = MagicMock(spec=URLIngestionService)
    mock_service.ingest_url.side_effect = ContentChallengeError(
        "Source returned browser/access challenge ('enable cookies') instead of article content."
    )

    res = handle_source_url_upload(
        url="https://pubmed.ncbi.nlm.nih.gov/blocked",
        session_factory=isolated_env["session_factory"],
        config=isolated_env["config"],
        url_service=mock_service,
    )

    assert res["success"] is False
    assert res["usable_clinical_content"] is False
    assert res["pipeline_status"] == "Blocked"
    assert "challenge" in res["result_message"].lower()


def test_render_upload_result_card_blocked_state_no_nameerror():
    # Verify that rendering the blocked state does not raise a NameError for input_type
    with patch("app.ui.components.st") as mock_st:
        res = {
            "success": False,
            "usable_clinical_content": False,
            "pipeline_status": "Blocked",
            "result_message": "Browser challenge detected",
            "source_url": "https://example.com/blocked"
        }
        # Should execute cleanly without NameError
        render_upload_result_card(res)
        
        # Verify st.error was called with [URL] in the message since source_url is present
        mock_st.error.assert_called_once()
        assert "[URL]" in mock_st.error.call_args[0][0]



