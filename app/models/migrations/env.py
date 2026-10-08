"""Alembic environment for CKEA.

Uses a connection passed via `config.attributes["connection"]` (application / tests,
including in-memory SQLite), otherwise the configured DATABASE_URL (CLI usage).
"""

from alembic import context
from sqlalchemy import engine_from_config, pool

import app.models.entities  # noqa: F401  (register tables on Base.metadata)
from app.models.database import Base
from app.services.config_service import load_config

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
        return

    url = config.get_main_option("sqlalchemy.url") or load_config().database_url
    engine = engine_from_config({"sqlalchemy.url": url}, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url") or load_config().database_url
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
