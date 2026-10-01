"""Unit tests for CKEA Monitoring Agent, file hashing, and idempotency."""

import hashlib
import logging
from pathlib import Path
import pypdfium2 as pdfium
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.monitoring_agent import MonitoringAgent, ScanResult
from app.models.database import get_engine, get_session_factory, init_db
from app.models.entities import IngestedDocument, IngestionFailure
from app.services.config_service import AppConfig
from app.services.file_hash import compute_bytes_sha256, compute_sha256


def create_sample_pdf(file_path: Path, pages: int = 1) -> Path:
    """Helper to generate a valid test PDF file."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(width=200, height=200)
    pdf.save(str(file_path))
    pdf.close()
    return file_path


@pytest.fixture
def temp_env(tmp_path: Path):
    """Set up temporary source directory and SQLite database for isolated testing."""
    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "test_monitoring.db"
    engine = get_engine(db_url=f"sqlite:///{db_path}")
    init_db(engine=engine)
    session_factory = get_session_factory(engine=engine)

    config = AppConfig(
        source_dir=source_dir,
        database_url=f"sqlite:///{db_path}",
        xai_api_key="secret-key-do-not-log-12345",
    )

    agent = MonitoringAgent(
        source_dir=source_dir,
        session_factory=session_factory,
        config=config,
    )

    yield {
        "source_dir": source_dir,
        "engine": engine,
        "session_factory": session_factory,
        "config": config,
        "agent": agent,
    }

    engine.dispose()


def test_sha256_is_calculated_correctly(tmp_path: Path):
    """Test 2: Verify SHA-256 hash calculation against known digest."""
    test_file = tmp_path / "test_data.bin"
    raw_content = b"Clinical Guideline Content For SHA Verification"
    test_file.write_bytes(raw_content)

    expected_hash = hashlib.sha256(raw_content).hexdigest().lower()
    computed_file_hash = compute_sha256(test_file)
    computed_bytes_hash = compute_bytes_sha256(raw_content)

    assert computed_file_hash == expected_hash
    assert computed_bytes_hash == expected_hash


def test_new_pdf_discovered_and_persisted(temp_env):
    """Test 1 & 3: Verify new PDF discovery, metadata persistence, and parsed state."""
    source_dir = temp_env["source_dir"]
    agent = temp_env["agent"]
    session_factory = temp_env["session_factory"]

    pdf_file = source_dir / "guideline_diabetes.pdf"
    create_sample_pdf(pdf_file, pages=2)

    result = agent.scan()

    assert result.discovered == 1
    assert result.new == 1
    assert result.skipped == 0
    assert result.failed == 0
    assert len(result.ingested_document_ids) == 1

    with session_factory() as session:
        doc = session.get(IngestedDocument, result.ingested_document_ids[0])
        assert doc is not None
        assert doc.source_identifier == "guideline_diabetes"
        assert doc.document_version == "1.0"
        assert doc.source_version == "1.0"
        assert doc.status == "parsed"
        assert doc.doc_metadata is not None
        assert doc.doc_metadata["page_count"] == 2
        assert doc.doc_metadata["file_size_bytes"] > 0


def test_unchanged_pdf_is_skipped(temp_env):
    """Test 4: Verify identical/unchanged PDF is skipped without duplicating."""
    source_dir = temp_env["source_dir"]
    agent = temp_env["agent"]

    pdf_file = source_dir / "nice_guideline.pdf"
    create_sample_pdf(pdf_file, pages=1)

    # First run: new document
    res1 = agent.scan()
    assert res1.new == 1
    assert res1.skipped == 0

    # Second run: file untouched -> skipped
    res2 = agent.scan()
    assert res2.discovered == 1
    assert res2.new == 0
    assert res2.skipped == 1
    assert res2.failed == 0


def test_changed_pdf_creates_new_version(temp_env):
    """Test 5: Verify modifying a PDF creates a new version while preserving the previous."""
    source_dir = temp_env["source_dir"]
    agent = temp_env["agent"]
    session_factory = temp_env["session_factory"]

    pdf_file = source_dir / "protocol_update.pdf"

    # Version 1: 1 page
    create_sample_pdf(pdf_file, pages=1)
    res1 = agent.scan()
    assert res1.new == 1

    # Modify file: 3 pages (new content and new hash)
    create_sample_pdf(pdf_file, pages=3)
    res2 = agent.scan()
    assert res2.discovered == 1
    assert res2.new == 1
    assert res2.skipped == 0

    # Verify both versions exist in database with provenance intact
    with session_factory() as session:
        docs = session.scalars(
            select(IngestedDocument)
            .where(IngestedDocument.source_identifier == "protocol_update")
            .order_by(IngestedDocument.document_version.asc())
        ).all()

        assert len(docs) == 2
        v1, v2 = docs[0], docs[1]
        assert v1.document_version == "1.0"
        assert v1.doc_metadata["page_count"] == 1
        assert v2.document_version == "2.0"
        assert v2.doc_metadata["page_count"] == 3
        assert v1.sha256_hash != v2.sha256_hash


def test_multiple_pdfs_processed(temp_env):
    """Test 6: Verify multiple PDFs are all processed in a single scan."""
    source_dir = temp_env["source_dir"]
    agent = temp_env["agent"]

    create_sample_pdf(source_dir / "doc_a.pdf", pages=1)
    create_sample_pdf(source_dir / "doc_b.pdf", pages=2)
    create_sample_pdf(source_dir / "doc_c.pdf", pages=1)

    result = agent.scan()
    assert result.discovered == 3
    assert result.new == 3
    assert result.skipped == 0
    assert result.failed == 0


def test_failed_pdf_does_not_stop_other_files_and_persists_failure(temp_env):
    """Test 7 & 8: Verify corrupted PDF failure is isolated and recorded as IngestionFailure."""
    source_dir = temp_env["source_dir"]
    agent = temp_env["agent"]
    session_factory = temp_env["session_factory"]

    # Valid PDFs
    create_sample_pdf(source_dir / "01_valid.pdf", pages=1)
    create_sample_pdf(source_dir / "03_valid.pdf", pages=1)

    # Invalid / Corrupt PDF
    corrupt_file = source_dir / "02_corrupted.pdf"
    corrupt_file.write_bytes(b"This is definitely not a valid PDF file format.")

    result = agent.scan()

    assert result.discovered == 3
    assert result.new == 2
    assert result.failed == 1
    assert len(result.failure_ids) == 1

    # Verify IngestionFailure record in database
    with session_factory() as session:
        failure = session.get(IngestionFailure, result.failure_ids[0])
        assert failure is not None
        assert "02_corrupted.pdf" in failure.source_path
        assert failure.error_category in ["PdfminerException", "Exception", "ValueError"]
        assert failure.retry_status == "pending"
        assert failure.operator_status == "unresolved"


def test_missing_source_directory_handled_visibly(tmp_path: Path, temp_env):
    """Test 9: Verify missing source directory raises FileNotFoundError."""
    missing_dir = tmp_path / "non_existent_sources"
    agent = MonitoringAgent(
        source_dir=missing_dir,
        session_factory=temp_env["session_factory"],
        config=temp_env["config"],
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        agent.scan()

    assert "Configured source directory does not exist" in str(exc_info.value)


def test_repeated_monitoring_is_idempotent(temp_env):
    """Test 10 (MANDATORY): Verify repeated monitoring produces zero duplicates."""
    source_dir = temp_env["source_dir"]
    agent = temp_env["agent"]
    session_factory = temp_env["session_factory"]

    create_sample_pdf(source_dir / "idempotent_1.pdf", pages=1)
    create_sample_pdf(source_dir / "idempotent_2.pdf", pages=2)

    # Run 1
    run1 = agent.scan()
    assert run1.discovered == 2
    assert run1.new == 2
    assert run1.skipped == 0
    assert run1.failed == 0

    with session_factory() as session:
        count_after_run1 = len(session.scalars(select(IngestedDocument)).all())
        assert count_after_run1 == 2

    # Run 2: Exact same files
    run2 = agent.scan()
    assert run2.discovered == 2
    assert run2.new == 0
    assert run2.skipped == 2
    assert run2.failed == 0

    # Verify zero duplicate rows created in database
    with session_factory() as session:
        all_docs = session.scalars(select(IngestedDocument)).all()
        assert len(all_docs) == 2


def test_sensitive_configuration_not_logged(temp_env, caplog):
    """Test 11: Verify sensitive credentials (API keys) are never logged."""
    source_dir = temp_env["source_dir"]
    agent = temp_env["agent"]
    secret_key = temp_env["config"].xai_api_key

    create_sample_pdf(source_dir / "sensitive_test.pdf", pages=1)

    with caplog.at_level(logging.DEBUG):
        agent.scan()

    assert secret_key not in caplog.text, "Secret API key was found in log output!"
