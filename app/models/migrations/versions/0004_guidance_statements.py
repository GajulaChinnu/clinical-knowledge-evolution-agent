"""Statement-level guidance index and detected source changes.

Revision ID: 0004_guidance_statements
Revises: 0003_gap_evidence
Create Date: 2026-10-08
"""

from alembic import op
import sqlalchemy as sa

from app.models.entities import UTCDateTime

revision = "0004_guidance_statements"
down_revision = "0003_gap_evidence"
branch_labels = None
depends_on = None


def _has(table: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table)


def upgrade() -> None:
    if not _has("guidance_statements"):
        op.create_table(
            "guidance_statements",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("ingested_document_id", sa.String(36), sa.ForeignKey("ingested_documents.id"), nullable=False, index=True),
            sa.Column("source_identity", sa.String(255), nullable=False, index=True),
            sa.Column("watchlist_id", sa.String(128), nullable=True, index=True),
            sa.Column("source_type", sa.String(32)),
            sa.Column("publisher_version", sa.String(32)),
            sa.Column("published_date", sa.String(32)),
            sa.Column("section_heading", sa.String(512)),
            sa.Column("char_start", sa.Integer()),
            sa.Column("char_end", sa.Integer()),
            sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("verbatim_text", sa.Text(), nullable=False),
            sa.Column("statement_type", sa.String(32), nullable=False),
            sa.Column("treatments", sa.JSON()),
            sa.Column("treatment_classes", sa.JSON()),
            sa.Column("departments", sa.JSON()),
            sa.Column("pathways", sa.JSON()),
            sa.Column("parsed", sa.JSON()),
            sa.Column("evidence_level", sa.String(8)),
            sa.Column("created_at", UTCDateTime(), nullable=False),
            sa.Column("schema_version", sa.String(16), nullable=False, server_default="1.0"),
        )
    if not _has("guidance_changes"):
        op.create_table(
            "guidance_changes",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("source_identity", sa.String(255), nullable=False, index=True),
            sa.Column("watchlist_id", sa.String(128), nullable=True, index=True),
            sa.Column("source_type", sa.String(32)),
            sa.Column("from_document_id", sa.String(36), sa.ForeignKey("ingested_documents.id")),
            sa.Column("to_document_id", sa.String(36), sa.ForeignKey("ingested_documents.id"), nullable=False, index=True),
            sa.Column("from_statement_id", sa.String(36), sa.ForeignKey("guidance_statements.id")),
            sa.Column("to_statement_id", sa.String(36), sa.ForeignKey("guidance_statements.id")),
            sa.Column("change_category", sa.String(48), nullable=False),
            sa.Column("attribute_changes", sa.JSON()),
            sa.Column("treatments", sa.JSON()),
            sa.Column("departments", sa.JSON()),
            sa.Column("pathways", sa.JSON()),
            sa.Column("relevance_status", sa.String(48), nullable=False, server_default="active"),
            sa.Column("filter_reason", sa.Text()),
            sa.Column("relevance", sa.Integer()),
            sa.Column("urgency", sa.Integer()),
            sa.Column("source_quality", sa.Integer()),
            sa.Column("novelty", sa.String(32)),
            sa.Column("duplicate_of_change_id", sa.String(36)),
            sa.Column("priority_score", sa.Float()),
            sa.Column("ranking_basis", sa.JSON()),
            sa.Column("created_at", UTCDateTime(), nullable=False),
            sa.Column("updated_at", UTCDateTime(), nullable=False),
            sa.Column("schema_version", sa.String(16), nullable=False, server_default="1.0"),
        )


def downgrade() -> None:
    if _has("guidance_changes"):
        op.drop_table("guidance_changes")
    if _has("guidance_statements"):
        op.drop_table("guidance_statements")
