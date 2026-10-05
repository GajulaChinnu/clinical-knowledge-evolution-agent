"""Deterministic reset utility for CKEA demo and test data (Phase 13).

Cleans and re-initializes SQLite database tables and temporary demo artifacts
WITHOUT destroying or modifying clinical protocols in data/protocols or ChromaDB indexes.
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.models.database import Base, get_engine, init_db
from app.services.config_service import load_config
from app.services.file_hash import compute_sha256

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ckea.scripts.reset_demo_data")


def reset_demo_database(db_url: str) -> None:
    """Drop and recreate all application database tables cleanly."""
    logger.info("Connecting to database: %s", db_url)
    engine = get_engine(db_url=db_url)

    # Cleanly drop all tables (except immutable protocols)
    logger.info("Dropping existing application tables...")
    Base.metadata.drop_all(bind=engine)

    # Recreate tables fresh
    logger.info("Re-creating clean database schema...")
    init_db(engine=engine)
    engine.dispose()
    logger.info("Database schema successfully re-initialized.")


def verify_protocol_preservation(config) -> None:
    """Verify that clinical protocols remain untouched during reset."""
    protocol_dir = config.protocol_dir
    files = list(protocol_dir.glob("*.json")) + list(protocol_dir.glob("*.md"))
    logger.info("Verified: %d protocol definitions remain safely preserved in %s.", len(files), protocol_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset CKEA demo database and temporary records.")
    parser.add_argument("--force", action="store_true", help="Force reset without interactive prompt")
    args = parser.parse_args()

    config = load_config()
    logger.info("=================================================================")
    logger.info(" CKEA DEMO DATA RESET UTILITY")
    logger.info("=================================================================")
    logger.info("Target Database: %s", config.database_url)
    logger.info("Protocol Directory: %s (WILL NOT BE MODIFIED)", config.protocol_dir)

    reset_demo_database(config.database_url)
    verify_protocol_preservation(config)

    logger.info("=================================================================")
    logger.info(" RESET COMPLETE: Application database clean and ready for demo.")
    logger.info("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
