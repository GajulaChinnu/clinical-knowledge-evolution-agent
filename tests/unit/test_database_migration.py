import pytest
from sqlalchemy import create_engine, inspect, text
from app.models.database import apply_migrations, Base
from app.models.entities import IngestedDocument

def test_database_migration_fresh_db():
    """Prove that a fresh database receives all columns through Base.metadata.create_all."""
    engine = create_engine("sqlite:///:memory:")
    # Initialize fresh DB
    Base.metadata.create_all(engine)
    
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("ingested_documents")]
    
    assert "previous_source_version_id" in columns
    assert "previous_sha256_hash" in columns
    assert "change_status" in columns
    
    # Running migration on fresh db should be idempotent
    apply_migrations(engine)
    columns_after = [col["name"] for col in inspector.get_columns("ingested_documents")]
    assert len(columns) == len(columns_after)

def test_database_migration_existing_db():
    """Prove that an existing DB missing the new columns gets them added by apply_migrations."""
    engine = create_engine("sqlite:///:memory:")
    
    # Manually create the old schema
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE ingested_documents (
                id VARCHAR(36) PRIMARY KEY,
                source_identifier VARCHAR(1024) NOT NULL,
                source_path VARCHAR(1024) NOT NULL,
                sha256_hash VARCHAR(64) NOT NULL,
                document_version VARCHAR(32) NOT NULL,
                source_version VARCHAR(32) NOT NULL,
                parser_version VARCHAR(32) NOT NULL,
                pipeline_version VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL,
                doc_metadata JSON,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        '''))
        # Insert a dummy record
        conn.execute(text('''
            INSERT INTO ingested_documents 
            (id, source_identifier, source_path, sha256_hash, document_version, source_version, parser_version, pipeline_version, status, created_at, updated_at) 
            VALUES ('test-id', 'test', 'test', 'hash', '1.0', '1.0', '1.0', '1.0', 'parsed', '2023-01-01', '2023-01-01')
        '''))
        
    inspector = inspect(engine)
    columns_before = [col["name"] for col in inspector.get_columns("ingested_documents")]
    assert "previous_source_version_id" not in columns_before
    assert "previous_sha256_hash" not in columns_before
    assert "change_status" not in columns_before
    
    # Apply migrations
    apply_migrations(engine)
    
    inspector = inspect(engine)
    columns_after = [col["name"] for col in inspector.get_columns("ingested_documents")]
    assert "previous_source_version_id" in columns_after
    assert "previous_sha256_hash" in columns_after
    assert "change_status" in columns_after
    
    # Prove data is preserved
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, change_status FROM ingested_documents WHERE id = 'test-id'")).fetchone()
        assert row is not None
        assert row.id == 'test-id'
        assert row.change_status == 'first_seen' # DEFAULT value


def _revision(engine):
    from alembic.runtime.migration import MigrationContext

    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def test_legacy_database_is_tracked_by_alembic_and_gets_gap_evidence_columns():
    from app.models.database import apply_migrations

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:  # simulate a pre-evidence gap_records table
        conn.execute(text("DROP TABLE gap_records"))
        conn.execute(text("CREATE TABLE gap_records (id VARCHAR(36) PRIMARY KEY, change_record_id VARCHAR(36) NOT NULL, comparison_result VARCHAR(32) NOT NULL, is_match BOOLEAN NOT NULL, status VARCHAR(32) NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, schema_version VARCHAR(16) NOT NULL)"))
    assert _revision(engine) is None

    apply_migrations(engine)

    cols = {c["name"] for c in inspect(engine).get_columns("gap_records")}
    assert {"exact_protocol_text", "matched_section_id", "review_reason"} <= cols
    assert _revision(engine) == "0003_gap_evidence"
    apply_migrations(engine)  # idempotent
    assert _revision(engine) == "0003_gap_evidence"


def test_init_db_on_new_database_is_stamped_at_head(tmp_path):
    from app.models.database import get_engine, init_db

    engine = get_engine(db_url=f"sqlite:///{tmp_path / 'fresh.db'}")
    init_db(engine=engine)
    assert _revision(engine) == "0003_gap_evidence"
    engine.dispose()
