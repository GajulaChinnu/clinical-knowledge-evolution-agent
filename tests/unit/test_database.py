"""Unit tests for CKEA SQLite database initialization, tables, and foreign keys."""

from pathlib import Path
import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.database import Base, get_engine, init_db, session_scope
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    IngestionFailure,
    Notification,
    ReviewAssignment,
)


@pytest.fixture
def temp_db_engine(tmp_path: Path):
    """Provide a fresh SQLite database engine for testing."""
    test_db_path = tmp_path / "test_ckea.db"
    db_url = f"sqlite:///{test_db_path}"
    engine = get_engine(db_url=db_url)
    init_db(engine=engine)
    yield engine
    engine.dispose()


def test_database_initialization(temp_db_engine):
    """Test 1: Verify database initialization creates tables successfully."""
    inspector = inspect(temp_db_engine)
    table_names = inspector.get_table_names()

    # 9 core CKEA tables + guidance_statements + guidance_changes + clinician_queries + alembic_version
    assert len(table_names) == 13
    assert "alembic_version" in table_names


def test_all_required_tables_exist(temp_db_engine):
    """Test 2: Verify all 9 required CKEA SQLite tables exist and protocol_sections does NOT."""
    inspector = inspect(temp_db_engine)
    table_names = set(inspector.get_table_names())

    expected_tables = {
        "ingested_documents",
        "change_records",
        "gap_records",
        "impact_records",
        "change_briefs",
        "review_assignments",
        "audit_log",
        "ingestion_failures",
        "notifications",
    }

    assert expected_tables.issubset(table_names)
    # Explicit constraint: protocol_sections must NOT be in SQLite
    assert "protocol_sections" not in table_names


def test_foreign_key_enforcement(temp_db_engine):
    """Test foreign key enforcement is active in SQLite connection."""
    with pytest.raises(IntegrityError):
        with session_scope(lambda: Session(bind=temp_db_engine)) as session:
            # Attempt to create a ChangeRecord with a non-existent parent document ID
            orphan_change = ChangeRecord(
                ingested_document_id="00000000-0000-0000-0000-000000000000",
                verbatim_text="Test recommendation",
                recommendation_type="treatment",
                target_population="Adults with Type 2 Diabetes",
                intervention="Metformin",
                confidence=0.95,
                extraction_model_version="openai/gpt-oss-20b",
                extraction_prompt_version="1.0",
                status="extracted",
            )
            session.add(orphan_change)
