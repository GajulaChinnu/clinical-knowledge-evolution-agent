"""Unit tests for SQLAlchemy entity models, relationships, idempotency, and persistence."""

from datetime import datetime, timezone
from pathlib import Path
import uuid
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.database import get_engine, init_db, session_scope
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    ImmutableEntityError,
    IngestedDocument,
    ReviewAssignment,
)
from app.repositories.base import BaseRepository


@pytest.fixture
def temp_session(tmp_path: Path):
    """Provide a transactional session bound to a temporary SQLite database."""
    test_db_path = tmp_path / "entities_test.db"
    engine = get_engine(db_url=f"sqlite:///{test_db_path}")
    init_db(engine=engine)

    session = Session(bind=engine)
    yield session
    session.close()
    engine.dispose()


def test_uuid_and_utc_timestamps_creation(temp_session: Session):
    """Test 3 & 4: Verify UUID generation and UTC timezone awareness on creation."""
    doc = IngestedDocument(
        source_identifier="NICE-NG28",
        source_path="data/sources/nice_ng28.pdf",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        document_version="1.0",
        source_version="2026.1",
        pipeline_version="1.0",
    )
    temp_session.add(doc)
    temp_session.commit()

    # Verify valid UUID v4
    val = uuid.UUID(doc.id)
    assert str(val) == doc.id

    # Verify UTC timezone
    assert doc.created_at.tzinfo is not None
    assert doc.created_at.tzinfo == timezone.utc
    assert doc.updated_at.tzinfo == timezone.utc
    assert doc.ingest_timestamp.tzinfo == timezone.utc


def test_document_idempotency_uniqueness(temp_session: Session):
    """Test 11: Verify idempotency unique constraint (sha256_hash + source_version + pipeline_version)."""
    sha = "a" * 64
    doc1 = IngestedDocument(
        source_identifier="ADA-2026",
        source_path="data/sources/ada_2026.pdf",
        sha256_hash=sha,
        source_version="1.0",
        pipeline_version="1.0",
    )
    temp_session.add(doc1)
    temp_session.commit()

    # Attempting to insert identical hash + source_version + pipeline_version must fail
    doc2 = IngestedDocument(
        source_identifier="ADA-2026-Duplicate",
        source_path="data/sources/ada_2026_copy.pdf",
        sha256_hash=sha,
        source_version="1.0",
        pipeline_version="1.0",
    )
    temp_session.add(doc2)
    with pytest.raises(IntegrityError):
        temp_session.commit()
    temp_session.rollback()

    # Different source version or pipeline version succeeds
    doc3 = IngestedDocument(
        source_identifier="ADA-2026-V2",
        source_path="data/sources/ada_2026_v2.pdf",
        sha256_hash=sha,
        source_version="2.0",  # Different version
        pipeline_version="1.0",
    )
    temp_session.add(doc3)
    temp_session.commit()
    assert doc3.id != doc1.id


def test_parent_child_pipeline_relationships(temp_session: Session):
    """Test 12: Verify full parent-child relational pipeline linkage."""
    # 1. IngestedDocument
    doc = IngestedDocument(
        source_identifier="GUIDELINE-101",
        source_path="data/sources/g101.pdf",
        sha256_hash="b" * 64,
        source_version="1.0",
        pipeline_version="1.0",
    )
    temp_session.add(doc)
    temp_session.commit()

    # 2. ChangeRecord (Child of IngestedDocument)
    change = ChangeRecord(
        ingested_document_id=doc.id,
        verbatim_text="Target HbA1c < 7.0% for most adults.",
        recommendation_type="glycemic_control",
        target_population="Adults with Type 2 Diabetes",
        intervention="HbA1c target",
        confidence=0.92,
        extraction_model_version="grok-4.7",
        extraction_prompt_version="1.0",
    )
    temp_session.add(change)
    temp_session.commit()

    # 3. GapRecord (Child of ChangeRecord - 1 to 1)
    gap = GapRecord(
        change_record_id=change.id,
        candidate_protocol_section_ids=["PROT-DM-SEC-4"],
        similarity=0.88,
        comparison_result="gap",
        comparison_confidence=0.85,
        difference_type="conflict",
        matched_protocol_id="PROT-DM-V1",
        is_match=True,
    )
    temp_session.add(gap)
    temp_session.commit()

    # 4. ImpactRecord (Child of GapRecord - 1 to 1)
    impact = ImpactRecord(
        gap_record_id=gap.id,
        clinical_urgency=4,
        evidence_strength=5,
        pathway_breadth=3,
        total_score=12,
        tier="Critical",
        routing_target="Endocrinology Governance Committee",
        scoring_yaml_version="1.0",
    )
    temp_session.add(impact)
    temp_session.commit()

    # 5. ChangeBrief (Child of ImpactRecord - 1 to 1)
    brief = ChangeBrief(
        impact_record_id=impact.id,
        status="assigned",
        structured_payload={"summary": "HbA1c target conflict with PROT-DM-V1"},
    )
    temp_session.add(brief)
    temp_session.commit()

    # 6. ReviewAssignments (Multiple children of ChangeBrief)
    review1 = ReviewAssignment(
        change_brief_id=brief.id,
        reviewer_role="Lead Endocrinologist",
        reviewer_id="REV-001",
        due_date=datetime.now(timezone.utc),
        status="assigned",
    )
    review2 = ReviewAssignment(
        change_brief_id=brief.id,
        reviewer_role="Clinical Pharmacist",
        reviewer_id="REV-002",
        due_date=datetime.now(timezone.utc),
        status="assigned",
    )
    temp_session.add_all([review1, review2])
    temp_session.commit()

    # Verify relationships from top-down and bottom-up
    temp_session.refresh(doc)
    assert len(doc.change_records) == 1
    assert doc.change_records[0].id == change.id
    assert change.document.id == doc.id
    assert change.gap_record.id == gap.id
    assert gap.change_record.id == change.id
    assert gap.impact_record.id == impact.id
    assert impact.gap_record.id == gap.id
    assert impact.change_brief.id == brief.id
    assert brief.impact_record.id == impact.id
    assert len(brief.review_assignments) == 2
    assert review1.brief.id == brief.id


def test_audit_record_structure(temp_session: Session):
    """Test 15: Verify append-only audit record structure and attributes."""
    audit = AuditLog(
        entity_id="11111111-1111-1111-1111-111111111111",
        entity_type="ChangeBrief",
        previous_status="draft",
        new_status="assigned",
        actor="governance_orchestrator",
        reason="Assigned to primary clinical reviewers according to SLA.",
        audit_metadata={"sla_hours": 48},
        schema_version="1.0",
    )
    temp_session.add(audit)
    temp_session.commit()

    fetched = temp_session.get(AuditLog, audit.id)
    assert fetched is not None
    assert fetched.entity_type == "ChangeBrief"
    assert fetched.previous_status == "draft"
    assert fetched.new_status == "assigned"
    assert fetched.actor == "governance_orchestrator"
    assert fetched.timestamp.tzinfo == timezone.utc


def test_database_persistence_and_retrieval(temp_session: Session):
    """Test 16: Verify database persistence and retrieval using BaseRepository."""
    repo = BaseRepository(temp_session, IngestedDocument)

    doc = IngestedDocument(
        source_identifier="REPO-TEST-01",
        source_path="data/sources/repo_test.pdf",
        sha256_hash="c" * 64,
        source_version="1.0",
        pipeline_version="1.0",
        status="parsed",
    )

    created = repo.add(doc)
    temp_session.commit()

    retrieved = repo.get_by_id(created.id)
    assert retrieved is not None
    assert retrieved.source_identifier == "REPO-TEST-01"
    assert retrieved.status == "parsed"

    all_docs = repo.list_all()
    assert len(all_docs) >= 1


def test_audit_log_cannot_be_deleted(temp_session: Session):
    """Verify that AuditLog records are append-only and cannot be deleted via repository or session."""
    repo = BaseRepository(temp_session, AuditLog)

    audit = AuditLog(
        entity_id="22222222-2222-2222-2222-222222222222",
        entity_type="ChangeRecord",
        previous_status="extracted",
        new_status="comparison_pending",
        actor="system",
        reason="Handoff to Comparison Agent",
        schema_version="1.0",
    )
    repo.add(audit)
    temp_session.commit()

    # 1. Attempting deletion via BaseRepository.delete() must raise ImmutableEntityError
    with pytest.raises(ImmutableEntityError, match="AuditLog records are append-only and cannot be deleted"):
        repo.delete(audit)

    # 2. Attempting direct deletion via session.delete() must also raise ImmutableEntityError
    with pytest.raises(ImmutableEntityError, match="AuditLog records are append-only and cannot be deleted"):
        temp_session.delete(audit)
        temp_session.flush()

    temp_session.rollback()

    # Verify the audit log record still exists intact in the database
    fetched = repo.get_by_id(audit.id)
    assert fetched is not None
    assert fetched.id == audit.id
    assert fetched.new_status == "comparison_pending"
