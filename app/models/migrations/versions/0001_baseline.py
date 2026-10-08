"""Baseline: CKEA schema as created by Phases 0-13 (before source lineage columns).

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-08

Existing databases without an alembic_version table are stamped at this revision and
then upgraded. New databases are created from the ORM metadata and stamped at head.
"""

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
