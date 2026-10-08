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
