"""Ranking outputs on impact records; category/department classification on change records.

Revision ID: 0005_ranking_and_classification
Revises: 0004_guidance_statements
Create Date: 2026-10-08
"""

from alembic import op
import sqlalchemy as sa

revision = "0005_ranking_and_classification"
down_revision = "0004_guidance_statements"
branch_labels = None
depends_on = None

IMPACT_COLUMNS = (
    sa.Column("relevance", sa.Integer(), nullable=True),
    sa.Column("source_quality", sa.Integer(), nullable=True),
    sa.Column("novelty", sa.String(32), nullable=True),
    sa.Column("priority_score", sa.Float(), nullable=True),
    sa.Column("affected_departments", sa.JSON(), nullable=True),
    sa.Column("affected_pathways", sa.JSON(), nullable=True),
    sa.Column("ranking_basis", sa.JSON(), nullable=True),
)
CHANGE_COLUMNS = (
    sa.Column("change_category", sa.String(48), nullable=True),
    sa.Column("departments", sa.JSON(), nullable=True),
    sa.Column("treatments", sa.JSON(), nullable=True),
)


def _existing(table: str) -> set:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(table):
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def _add(table: str, columns) -> None:
    existing = _existing(table)
    if not existing:
        return
    with op.batch_alter_table(table) as batch:
        for column in columns:
            if column.name not in existing:
                batch.add_column(column.copy())


def _drop(table: str, columns) -> None:
    existing = _existing(table)
    with op.batch_alter_table(table) as batch:
        for column in columns:
            if column.name in existing:
                batch.drop_column(column.name)


def upgrade() -> None:
    _add("impact_records", IMPACT_COLUMNS)
    _add("change_records", CHANGE_COLUMNS)


def downgrade() -> None:
    _drop("impact_records", IMPACT_COLUMNS)
    _drop("change_records", CHANGE_COLUMNS)
