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
    except Exception:
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
