"""Source version lineage columns on ingested_documents.

Revision ID: 0002_source_lineage
Revises: 0001_baseline
Create Date: 2026-10-08
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_source_lineage"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

COLUMNS = (
    sa.Column("previous_source_version_id", sa.String(36), nullable=True),
    sa.Column("previous_sha256_hash", sa.String(64), nullable=True),
    sa.Column("change_status", sa.String(32), nullable=True, server_default="first_seen"),
)


def _existing(table: str) -> set:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(table):
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    existing = _existing("ingested_documents")
    if not existing:
        return
    with op.batch_alter_table("ingested_documents") as batch:
        for column in COLUMNS:
            if column.name not in existing:
                batch.add_column(column.copy())


def downgrade() -> None:
    existing = _existing("ingested_documents")
    with op.batch_alter_table("ingested_documents") as batch:
        for column in COLUMNS:
            if column.name in existing:
                batch.drop_column(column.name)
