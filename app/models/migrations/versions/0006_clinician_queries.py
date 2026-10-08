"""Clinician Treatment Check records (no patient context stored).

Revision ID: 0006_clinician_queries
Revises: 0005_ranking_and_classification
Create Date: 2026-10-08
"""

from alembic import op
import sqlalchemy as sa

from app.models.entities import UTCDateTime

revision = "0006_clinician_queries"
down_revision = "0005_ranking_and_classification"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("clinician_queries"):
        return
    op.create_table(
        "clinician_queries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("actor", sa.String(128), nullable=False, server_default="clinician"),
        sa.Column("department", sa.String(64), nullable=False, index=True),
        sa.Column("treatment", sa.String(255), nullable=False),
        sa.Column("condition", sa.String(255)),
        sa.Column("verdict", sa.String(64), index=True),
        sa.Column("cited_statement_ids", sa.JSON()),
        sa.Column("answer", sa.JSON()),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("schema_version", sa.String(16), nullable=False, server_default="1.0"),
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("clinician_queries"):
        op.drop_table("clinician_queries")
