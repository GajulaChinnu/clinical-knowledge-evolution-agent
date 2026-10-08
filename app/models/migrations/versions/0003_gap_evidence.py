"""Persist verified comparison evidence on gap_records.

Revision ID: 0003_gap_evidence
Revises: 0002_source_lineage
Create Date: 2026-10-08
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_gap_evidence"
down_revision = "0002_source_lineage"
branch_labels = None
depends_on = None

COLUMNS = (
    sa.Column("matched_section_id", sa.String(128), nullable=True),
    sa.Column("matched_section_heading", sa.String(512), nullable=True),
    sa.Column("exact_protocol_text", sa.Text(), nullable=True),
    sa.Column("specific_difference", sa.Text(), nullable=True),
    sa.Column("comparison_rationale", sa.Text(), nullable=True),
    sa.Column("review_reason", sa.Text(), nullable=True),
)


def _existing(table: str) -> set:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(table):
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    existing = _existing("gap_records")
    if not existing:
        return
    with op.batch_alter_table("gap_records") as batch:
        for column in COLUMNS:
            if column.name not in existing:
                batch.add_column(column.copy())


def downgrade() -> None:
    existing = _existing("gap_records")
    with op.batch_alter_table("gap_records") as batch:
        for column in COLUMNS:
            if column.name in existing:
                batch.drop_column(column.name)
