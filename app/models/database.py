"""CKEA SQLite and SQLAlchemy 2.x persistence initialization module.

Provides the declarative Base, engine factory, session factory, and database initialization
routines with SQLite foreign key enforcement enabled.
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.services.config_service import load_config


class Base(DeclarativeBase):
    """Declarative Base class for all CKEA SQLAlchemy entities."""
    pass


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """Ensure SQLite enforces foreign key constraints on every connection."""
    # Check if connection is from sqlite3 driver
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    except Exception:  # Non-SQLite DBAPI drivers reject PRAGMA; foreign keys are then enforced natively.
        pass
    finally:
        cursor.close()


def get_engine(db_url: Optional[str] = None, echo: bool = False) -> Engine:
    """Create and return a configured SQLAlchemy Engine.

    Args:
        db_url: Database connection string. If None, loaded from configuration.
        echo: If True, log SQL statements.

    Returns:
        SQLAlchemy Engine instance.
    """
    if db_url is None:
        db_url = load_config().database_url

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # If it is a file-based sqlite URL, ensure the parent directory exists
        if "///" in db_url and not db_url.startswith("sqlite:///:memory:"):
            db_path_str = db_url.split("///", 1)[1]
            db_path = Path(db_path_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        db_url,
        echo=echo,
        connect_args=connect_args,
        future=True,
    )
    return engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker[Session]:
    """Create a thread-safe session factory bound to the provided or default engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        sessionmaker configured for Session instances.
    """
    if engine is None:
        engine = get_engine()
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

def apply_migrations(engine: Engine) -> None:
    """Apply safe schema migrations for existing databases."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if not inspector.has_table("ingested_documents"):
        return
    columns = [col["name"] for col in inspector.get_columns("ingested_documents")]

    with engine.begin() as conn:
        if "previous_source_version_id" not in columns:
            conn.execute(text("ALTER TABLE ingested_documents ADD COLUMN previous_source_version_id VARCHAR(36)"))
        if "previous_sha256_hash" not in columns:
            conn.execute(text("ALTER TABLE ingested_documents ADD COLUMN previous_sha256_hash VARCHAR(64)"))
        if "change_status" not in columns:
            conn.execute(text("ALTER TABLE ingested_documents ADD COLUMN change_status VARCHAR(32) DEFAULT 'first_seen'"))

        if inspector.has_table("gap_records"):
            gap_columns = {col["name"] for col in inspector.get_columns("gap_records")}
            for name, ddl in (
                ("matched_section_id", "VARCHAR(128)"),
                ("matched_section_heading", "VARCHAR(512)"),
                ("exact_protocol_text", "TEXT"),
                ("specific_difference", "TEXT"),
                ("comparison_rationale", "TEXT"),
                ("review_reason", "TEXT"),
            ):
                if name not in gap_columns:
                    conn.execute(text(f"ALTER TABLE gap_records ADD COLUMN {name} {ddl}"))

def init_db(engine: Optional[Engine] = None, db_url: Optional[str] = None) -> Engine:
    """Explicitly initialize the database schema by creating all registered tables.

    Args:
        engine: Existing Engine to initialize.
        db_url: Optional URL to construct a new engine if engine is not provided.

    Returns:
        The Engine bound to the initialized database.
    """
    # Import entities so they register with Base.metadata before create_all
    import app.models.entities  # noqa: F401

    if engine is None:
        engine = get_engine(db_url=db_url)

    Base.metadata.create_all(bind=engine)
    apply_migrations(engine)
    return engine

@contextmanager
def session_scope(
    session_factory: Optional[sessionmaker[Session]] = None,
) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    if session_factory is None:
        session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
