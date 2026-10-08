"""SQLAlchemy entity models for the CKEA SQLite persistence layer.

Defines the exact tables required for Phase 1 and beyond:
1. IngestedDocument
2. ChangeRecord
3. GapRecord
4. ImpactRecord
5. ChangeBrief
6. ReviewAssignment
7. AuditLog
8. IngestionFailure
9. Notification
"""

from datetime import datetime, timezone
import uuid
from typing import Any, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import relationship

from app.models.database import Base


class ImmutableEntityError(ValueError):
    """Raised when an attempt is made to delete or modify an append-only/immutable entity."""
    pass


class UTCDateTime(TypeDecorator):
    """Ensures datetime values are stored and retrieved as timezone-aware UTC datetimes."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is not None:
            if not isinstance(value, datetime):
                raise TypeError(f"Expected datetime object, got {type(value)}")
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
        return value

    def process_result_value(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is not None:
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
        return value


def _utc_now() -> datetime:
    """Generate timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _gen_uuid() -> str:
    """Generate string representation of standard UUID v4."""
    return str(uuid.uuid4())


class IngestedDocument(Base):
    """Represents an ingested clinical source document version."""

    __tablename__ = "ingested_documents"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    source_identifier: str = Column(String(255), nullable=False, index=True)
    source_path: str = Column(String(1024), nullable=False)
    sha256_hash: str = Column(String(64), nullable=False, index=True)
    document_version: str = Column(String(32), nullable=False, default="1.0")
    source_version: str = Column(String(32), nullable=False, default="1.0")
    previous_source_version_id: Optional[str] = Column(String(36), nullable=True)
    previous_sha256_hash: Optional[str] = Column(String(64), nullable=True)
    change_status: str = Column(String(32), nullable=False, default="first_seen")
    parser_version: str = Column(String(32), nullable=False, default="1.0")
    pipeline_version: str = Column(String(32), nullable=False, default="1.0")
    ingest_timestamp: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    status: str = Column(String(32), nullable=False, default="discovered")
    doc_metadata: Optional[Any] = Column(JSON, nullable=True)
    retry_count: int = Column(Integer, nullable=False, default=0)
    error_message: Optional[str] = Column(Text, nullable=True)

    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")

    __table_args__ = (
        UniqueConstraint(
            "sha256_hash",
            "source_version",
            "pipeline_version",
            name="uq_doc_hash_source_pipeline",
        ),
    )

    # Relationships
    change_records = relationship(
        "ChangeRecord",
        back_populates="document",
        cascade="all, save-update",
    )


class ChangeRecord(Base):
    """Represents a discrete clinical recommendation extracted from an ingested document."""

    __tablename__ = "change_records"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    ingested_document_id: str = Column(
        String(36),
        ForeignKey("ingested_documents.id"),
        nullable=False,
        index=True,
    )
    verbatim_text: str = Column(Text, nullable=False)
    recommendation_type: str = Column(String(64), nullable=False)
    target_population: str = Column(Text, nullable=False)
    intervention: str = Column(Text, nullable=False)
    evidence_grade: Optional[str] = Column(String(32), nullable=True)
    confidence: float = Column(Float, nullable=False)
    page: Optional[int] = Column(Integer, nullable=True)
    section: Optional[str] = Column(String(255), nullable=True)
    source_excerpt: Optional[str] = Column(Text, nullable=True)
    extraction_model_version: str = Column(String(64), nullable=False)
    extraction_prompt_version: str = Column(String(64), nullable=False)
    status: str = Column(String(32), nullable=False, default="extracted")
    change_category: Optional[str] = Column(String(48), nullable=True)
    departments: Optional[Any] = Column(JSON, nullable=True)
    treatments: Optional[Any] = Column(JSON, nullable=True)

    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")

    # Relationships
    document = relationship("IngestedDocument", back_populates="change_records")
    gap_record = relationship(
        "GapRecord",
        back_populates="change_record",
        uselist=False,
    )


class GapRecord(Base):
    """Represents protocol comparison findings (match or explicit no-match) for a recommendation."""

    __tablename__ = "gap_records"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    change_record_id: str = Column(
        String(36),
        ForeignKey("change_records.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    candidate_protocol_section_ids: Optional[Any] = Column(JSON, nullable=True)
    similarity: Optional[float] = Column(Float, nullable=True)
    comparison_result: str = Column(String(32), nullable=False)  # gap, no_gap, ambiguous, no_match
    comparison_confidence: Optional[float] = Column(Float, nullable=True)
    difference_type: Optional[str] = Column(String(64), nullable=True)
    matched_protocol_id: Optional[str] = Column(String(128), nullable=True)
    matched_protocol_version: Optional[str] = Column(String(64), nullable=True)
    # Verified comparison evidence (persisted so briefs never reconstruct or invent it)
    matched_section_id: Optional[str] = Column(String(128), nullable=True)
    matched_section_heading: Optional[str] = Column(String(512), nullable=True)
    exact_protocol_text: Optional[str] = Column(Text, nullable=True)  # only when verified verbatim
    specific_difference: Optional[str] = Column(Text, nullable=True)
    comparison_rationale: Optional[str] = Column(Text, nullable=True)
    review_reason: Optional[str] = Column(Text, nullable=True)  # why G2/G3 review is required
    is_match: bool = Column(Boolean, nullable=False, default=True)
    reviewer_resolution: Optional[str] = Column(Text, nullable=True)
    status: str = Column(String(32), nullable=False, default="pending")

    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")

    # Relationships
    change_record = relationship("ChangeRecord", back_populates="gap_record")
    impact_record = relationship(
        "ImpactRecord",
        back_populates="gap_record",
        uselist=False,
    )


class ImpactRecord(Base):
    """Represents deterministic multidimensional impact scoring for a confirmed gap."""

    __tablename__ = "impact_records"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    gap_record_id: str = Column(
        String(36),
        ForeignKey("gap_records.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    clinical_urgency: Optional[int] = Column(Integer, nullable=True)
    evidence_strength: Optional[int] = Column(Integer, nullable=True)
    pathway_breadth: Optional[int] = Column(Integer, nullable=True)
    rule_ids: Optional[Any] = Column(JSON, nullable=True)
    scoring_yaml_version: str = Column(String(32), nullable=False, default="1.0")
    total_score: Optional[int] = Column(Integer, nullable=True)
    tier: Optional[str] = Column(String(32), nullable=True)  # Critical, High, Standard, Low
    routing_target: Optional[str] = Column(String(64), nullable=True)
    sla_deadline: Optional[datetime] = Column(UTCDateTime(), nullable=True)
    urgency_basis: Optional[str] = Column(Text, nullable=True)
    evidence_basis: Optional[str] = Column(Text, nullable=True)
    breadth_basis: Optional[str] = Column(Text, nullable=True)
    status: str = Column(String(32), nullable=False, default="pending")
    # Ranking outputs (order the governance queue only; never change the tier)
    relevance: Optional[int] = Column(Integer, nullable=True)
    source_quality: Optional[int] = Column(Integer, nullable=True)
    novelty: Optional[str] = Column(String(32), nullable=True)
    priority_score: Optional[float] = Column(Float, nullable=True)
    affected_departments: Optional[Any] = Column(JSON, nullable=True)
    affected_pathways: Optional[Any] = Column(JSON, nullable=True)
    ranking_basis: Optional[Any] = Column(JSON, nullable=True)

    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")

    # Relationships
    gap_record = relationship("GapRecord", back_populates="impact_record")
    change_brief = relationship(
        "ChangeBrief",
        back_populates="impact_record",
        uselist=False,
    )


class ChangeBrief(Base):
    """Represents a structured evidence brief prepared for human clinical governance."""

    __tablename__ = "change_briefs"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    impact_record_id: str = Column(
        String(36),
        ForeignKey("impact_records.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    status: str = Column(String(32), nullable=False, default="draft")
    rendered_file_path: Optional[str] = Column(String(1024), nullable=True)
    rendered_file_hash: Optional[str] = Column(String(64), nullable=True)
    structured_payload: Optional[Any] = Column(JSON, nullable=True)

    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")

    # Relationships
    impact_record = relationship("ImpactRecord", back_populates="change_brief")
    review_assignments = relationship(
        "ReviewAssignment",
        back_populates="brief",
        cascade="all, save-update",
    )


class ReviewAssignment(Base):
    """Represents an assigned clinical reviewer and their recorded governance decision."""

    __tablename__ = "review_assignments"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    change_brief_id: str = Column(
        String(36),
        ForeignKey("change_briefs.id"),
        nullable=False,
        index=True,
    )
    reviewer_role: str = Column(String(64), nullable=False)
    reviewer_id: str = Column(String(64), nullable=False)
    status: str = Column(String(32), nullable=False, default="assigned")
    due_date: datetime = Column(UTCDateTime(), nullable=False)
    decision: Optional[str] = Column(String(32), nullable=True)  # approve, reject, defer
    rationale: Optional[str] = Column(Text, nullable=True)
    decision_timestamp: Optional[datetime] = Column(UTCDateTime(), nullable=True)
    defer_follow_up_date: Optional[datetime] = Column(UTCDateTime(), nullable=True)

    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")

    # Relationships
    brief = relationship("ChangeBrief", back_populates="review_assignments")


class AuditLog(Base):
    """Append-only audit trail recording all state transitions and system actions."""

    __tablename__ = "audit_log"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    entity_id: str = Column(String(36), nullable=False, index=True)
    entity_type: str = Column(String(64), nullable=False, index=True)
    previous_status: Optional[str] = Column(String(32), nullable=True)
    new_status: str = Column(String(32), nullable=False)
    actor: str = Column(String(128), nullable=False)
    timestamp: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    reason: Optional[str] = Column(Text, nullable=True)
    audit_metadata: Optional[Any] = Column(JSON, nullable=True)
    schema_version: str = Column(String(16), nullable=False, default="1.0")


@event.listens_for(AuditLog, "before_delete")
def _prevent_audit_log_delete(mapper: Any, connection: Any, target: Any) -> None:
    """Enforce append-only immutability by preventing AuditLog deletion in SQLite."""
    raise ImmutableEntityError("AuditLog records are append-only and cannot be deleted.")


class IngestionFailure(Base):
    """Records document ingestion or parsing errors with retry telemetry."""

    __tablename__ = "ingestion_failures"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    source_path: str = Column(String(1024), nullable=False)
    timestamp: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    error_category: str = Column(String(64), nullable=False)
    error_message: str = Column(Text, nullable=False)
    retry_count: int = Column(Integer, nullable=False, default=0)
    retry_status: str = Column(String(32), nullable=False, default="pending")
    operator_status: str = Column(String(32), nullable=False, default="unresolved")

    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")


class Notification(Base):
    """Records governance or SLA notifications prepared for delivery."""

    __tablename__ = "notifications"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    related_entity_id: Optional[str] = Column(String(36), nullable=True)
    related_entity_type: Optional[str] = Column(String(64), nullable=True)
    notification_type: str = Column(String(64), nullable=False)
    recipient: str = Column(String(255), nullable=False)
    payload: Optional[Any] = Column(JSON, nullable=True)
    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    delivery_status: str = Column(String(32), nullable=False, default="pending")
    schema_version: str = Column(String(16), nullable=False, default="1.0")


class GuidanceStatement(Base):
    """A verbatim, sentence-level statement from one version of a monitored source.

    Built deterministically on ingestion (no LLM). Statements are the grounding unit for
    clinician queries: every citation CKEA shows is one of these rows, located by character
    offsets in the stored source artifact. Rows are immutable.
    """

    __tablename__ = "guidance_statements"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    ingested_document_id: str = Column(String(36), ForeignKey("ingested_documents.id"), nullable=False, index=True)
    source_identity: str = Column(String(255), nullable=False, index=True)
    watchlist_id: Optional[str] = Column(String(128), nullable=True, index=True)
    source_type: Optional[str] = Column(String(32), nullable=True)
    publisher_version: Optional[str] = Column(String(32), nullable=True)
    published_date: Optional[str] = Column(String(32), nullable=True)
    section_heading: Optional[str] = Column(String(512), nullable=True)
    char_start: Optional[int] = Column(Integer, nullable=True)
    char_end: Optional[int] = Column(Integer, nullable=True)
    sequence: int = Column(Integer, nullable=False, default=0)
    verbatim_text: str = Column(Text, nullable=False)
    statement_type: str = Column(String(32), nullable=False)
    treatments: Optional[Any] = Column(JSON, nullable=True)
    treatment_classes: Optional[Any] = Column(JSON, nullable=True)
    departments: Optional[Any] = Column(JSON, nullable=True)
    pathways: Optional[Any] = Column(JSON, nullable=True)
    parsed: Optional[Any] = Column(JSON, nullable=True)
    evidence_level: Optional[str] = Column(String(8), nullable=True)
    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")


class GuidanceChange(Base):
    """A detected change between consecutive versions of a source (or a new source's content).

    Source evolution only (external v(n-1) -> v(n)); institutional protocol comparison is a
    separate step. Filtered changes are kept with a reason and can be restored by a reviewer.
    """

    __tablename__ = "guidance_changes"

    id: str = Column(String(36), primary_key=True, default=_gen_uuid)
    source_identity: str = Column(String(255), nullable=False, index=True)
    watchlist_id: Optional[str] = Column(String(128), nullable=True, index=True)
    source_type: Optional[str] = Column(String(32), nullable=True)
    from_document_id: Optional[str] = Column(String(36), ForeignKey("ingested_documents.id"), nullable=True)
    to_document_id: str = Column(String(36), ForeignKey("ingested_documents.id"), nullable=False, index=True)
    from_statement_id: Optional[str] = Column(String(36), ForeignKey("guidance_statements.id"), nullable=True)
    to_statement_id: Optional[str] = Column(String(36), ForeignKey("guidance_statements.id"), nullable=True)
    change_category: str = Column(String(48), nullable=False)
    attribute_changes: Optional[Any] = Column(JSON, nullable=True)
    treatments: Optional[Any] = Column(JSON, nullable=True)
    departments: Optional[Any] = Column(JSON, nullable=True)
    pathways: Optional[Any] = Column(JSON, nullable=True)
    relevance_status: str = Column(String(48), nullable=False, default="active")
    filter_reason: Optional[str] = Column(Text, nullable=True)
    relevance: Optional[int] = Column(Integer, nullable=True)
    urgency: Optional[int] = Column(Integer, nullable=True)
    source_quality: Optional[int] = Column(Integer, nullable=True)
    novelty: Optional[str] = Column(String(32), nullable=True)
    duplicate_of_change_id: Optional[str] = Column(String(36), nullable=True)
    priority_score: Optional[float] = Column(Float, nullable=True)
    ranking_basis: Optional[Any] = Column(JSON, nullable=True)
    created_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now)
    updated_at: datetime = Column(UTCDateTime(), nullable=False, default=_utc_now, onupdate=_utc_now)
    schema_version: str = Column(String(16), nullable=False, default="1.0")


@event.listens_for(GuidanceStatement, "before_update")
def _block_statement_update(mapper, connection, target) -> None:
    raise ImmutableEntityError("GuidanceStatement rows are immutable provenance records.")


@event.listens_for(GuidanceStatement, "before_delete")
def _block_statement_delete(mapper, connection, target) -> None:
    raise ImmutableEntityError("GuidanceStatement rows are immutable provenance records.")
