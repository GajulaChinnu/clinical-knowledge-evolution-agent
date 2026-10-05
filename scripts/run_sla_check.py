"""Manual single-cycle CLI runner for CKEA SLA Escalation evaluation.

Executes exactly one synchronous SLA evaluation cycle without running
a continuous background scheduler loop. Useful for operations, testing,
and cron triggers.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.database import get_engine, get_session_factory, init_db
from app.services.config_service import load_config
from app.services.sla_scheduler import SLAScheduler


def main() -> int:
    config = load_config()
    engine = get_engine(config.database_url)
    init_db(engine=engine)
    session_factory = get_session_factory(engine=engine)

    print("=" * 65)
    print(" CKEA SLA CHECK: MANUAL SINGLE-CYCLE RUNNER")
    print("=" * 65)
    print(f"Timestamp:    {datetime.now(timezone.utc).isoformat()}")
    print(f"Database:     {config.database_url}")
    print(f"Timezone:     {config.sla_timezone}")
    print("-" * 65)

    scheduler = SLAScheduler(
        session_factory=session_factory,
        config=config,
    )

    summary = scheduler.run_once()

    print(f"Cycle ID:               {summary.cycle_id}")
    print(f"Execution Duration:     {summary.duration_ms:.2f} ms")
    print(f"Evaluated Briefs:       {summary.evaluated_count}")
    print(f"Overdue Briefs:         {summary.overdue_count}")
    print(f"Newly Escalated:        {summary.escalated_count}")
    print(f"Already Escalated:      {summary.already_escalated_count}")
    print(f"Skipped (Resolved):     {summary.skipped_count}")
    print(f"Failed / Errored:       {summary.failed_count}")

    if summary.errors:
        print("\nErrors encountered:")
        for err in summary.errors:
            print(f"  - {err}")

    print("=" * 65)
    return 1 if summary.failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
