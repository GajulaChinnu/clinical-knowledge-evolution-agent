"""Operational entrypoint for executing a single CKEA Monitoring Agent scan."""

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.agents.monitoring_agent import MonitoringAgent
from app.models.database import init_db
from app.services.config_service import load_config


def main() -> int:
    """Run one monitoring scan and display the operational summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("ckea.scripts.run_monitoring")

    try:
        config = load_config()
        logger.info("Initializing persistence layer at %s...", config.database_url)
        init_db()

        logger.info("Starting Monitoring Agent scan on: %s", config.source_dir)
        agent = MonitoringAgent(config=config)
        result = agent.scan()

        print("\n==========================================")
        print("     CKEA MONITORING SCAN SUMMARY")
        print("==========================================")
        print(f"Discovered: {result.discovered}")
        print(f"New: {result.new}")
        print(f"Skipped: {result.skipped}")
        print(f"Failed: {result.failed}")
        print("==========================================\n")

        return 0 if result.failed == 0 else 1

    except Exception as e:
        logger.error("Monitoring scan execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
