"""Deterministic recurring SLA scheduler and escalation coordinator for Phase 11.

Periodically evaluates unresolved ChangeBrief review assignments and triggers
the Phase 9 G5 escalation mechanism without performing automated decisions.

Core Guarantees:
1. Zero LLM / Groq / OpenAI calls.
2. Protocol data and clinical decisions remain strictly unmutated.
3. No external notification side effects (email, Slack, SMS, webhooks).
4. Full idempotency on repeated executions.
5. Per-brief failure isolation (one failure does not abort the cycle).
6. In-process concurrency guard preventing overlapping execution cycles.
7. Background scheduler does not start on module import.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from app.agents.governance_agent import GovernanceAgent

from app.models.database import get_engine, get_session_factory
from app.models.entities import AuditLog, ChangeBrief
from app.schemas.briefs import BriefStatus
from app.schemas.governance import SLAEscalationResult
from app.schemas.scheduler import SLACycleSummary
from app.services.config_service import AppConfig, load_config

logger = logging.getLogger("ckea.services.sla_scheduler")


class SLAScheduler:
    """Recurring, in-process SLA scheduler coordinating G5 review escalation."""

    JOB_ID: str = "ckea_sla_periodic_check"

    def __init__(
        self,
        session_factory: Optional[sessionmaker[Session]] = None,
        governance_agent: Optional[Any] = None,
        config: Optional[AppConfig] = None,
        interval_minutes: Optional[int] = None,
        timezone_str: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Initialize the SLA Scheduler.

        Args:
            session_factory: SQLAlchemy sessionmaker factory.
            governance_agent: GovernanceAgent instance (created if not provided).
            config: AppConfig instance.
            interval_minutes: Checking interval in minutes (defaults to config).
            timezone_str: Timezone string for scheduling (defaults to UTC).
            enabled: Explicit enable/disable flag (defaults to config.sla_scheduler_enabled).
        """
        self.config = config or load_config()
        if session_factory is not None:
            self.session_factory = session_factory
        else:
            engine = get_engine(self.config.database_url)
            self.session_factory = get_session_factory(engine)
        if governance_agent is not None:
            self.governance_agent = governance_agent
        else:
            from app.agents.governance_agent import GovernanceAgent
            self.governance_agent = GovernanceAgent(
                session_factory=self.session_factory,
                config=self.config,
            )

        self.interval_minutes = (
            interval_minutes
            if interval_minutes is not None
            else getattr(self.config, "sla_check_interval_minutes", 60)
        )
        self.timezone_str = (
            timezone_str
            if timezone_str is not None
            else getattr(self.config, "sla_timezone", "UTC")
        )
        self.enabled = (
            enabled
            if enabled is not None
            else getattr(self.config, "sla_scheduler_enabled", False)
        )

        self._scheduler: Optional[BackgroundScheduler] = None
        self._lock = threading.Lock()
        self._last_cycle_summary: Optional[SLACycleSummary] = None

    # ==========================================================================
    # LIFECYCLE MANAGEMENT
    # ==========================================================================

    def is_running(self) -> bool:
        """Check whether the recurring background scheduler is currently active."""
        return bool(self._scheduler is not None and self._scheduler.running)

    def start(self, force: bool = False) -> bool:
        """Start the background recurring SLA scheduler.

        Idempotent: calling start() while already running is safe and no-op.

        Args:
            force: If True, starts even if self.enabled is False.

        Returns:
            bool: True if scheduler is running after call, False if disabled and not forced.
        """
        if self.is_running():
            logger.info("SLAScheduler is already running (job: %s).", self.JOB_ID)
            return True

        if not self.enabled and not force:
            logger.info(
                "SLAScheduler is disabled by configuration (sla_scheduler_enabled=False). "
                "Call start(force=True) to override."
            )
            return False

        logger.info(
            "Starting SLAScheduler with interval of %d minutes (timezone: %s).",
            self.interval_minutes,
            self.timezone_str,
        )

        scheduler = BackgroundScheduler(timezone=self.timezone_str)
        scheduler.add_job(
            func=self.run_once,
            trigger=IntervalTrigger(
                minutes=self.interval_minutes,
                timezone=self.timezone_str,
            ),
            id=self.JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            name="CKEA Periodic SLA Escalation Check",
        )
        scheduler.start()
        self._scheduler = scheduler

        self._record_cycle_audit(
            action="sla_scheduler_started",
            reason=f"SLA scheduler started with interval of {self.interval_minutes} minutes.",
            metadata={
                "interval_minutes": self.interval_minutes,
                "timezone": self.timezone_str,
                "job_id": self.JOB_ID,
            },
        )
        return True

    def stop(self, wait: bool = False) -> bool:
        """Stop the background recurring SLA scheduler.

        Idempotent: calling stop() when not running is safe and no-op.

        Args:
            wait: Whether to wait for running jobs to complete before shutdown.

        Returns:
            bool: True if a running scheduler was stopped, False if was not running.
        """
        if self._scheduler is not None and self._scheduler.running:
            logger.info("Stopping SLAScheduler (wait=%s)...", wait)
            self._scheduler.shutdown(wait=wait)
            self._scheduler = None

            self._record_cycle_audit(
                action="sla_scheduler_stopped",
                reason="SLA scheduler stopped cleanly.",
            )
            return True

        self._scheduler = None
        return False

    def get_job(self) -> Optional[Any]:
        """Return the active APScheduler Job instance if scheduled, else None."""
        if self._scheduler is not None and self._scheduler.running:
            return self._scheduler.get_job(self.JOB_ID)
        return None

    @property
    def last_cycle_summary(self) -> Optional[SLACycleSummary]:
        """Return the most recent cycle execution summary."""
        return self._last_cycle_summary

    # ==========================================================================
    # EVALUATION CYCLE EXECUTION
    # ==========================================================================

    def run_once(self, reference_time: Optional[datetime] = None) -> SLACycleSummary:
        """Synchronously execute one SLA evaluation cycle across eligible briefs.

        This method:
        1. Acquires an in-process concurrency lock to prevent overlapping runs.
        2. Identifies all eligible unresolved ChangeBriefs (assigned, in_review, deferred).
        3. Skips resolved briefs (decided, closed).
        4. Calls existing Phase 9 GovernanceAgent.evaluate_sla() for each eligible brief.
        5. Preserves per-brief failure isolation (one failure does not crash cycle).
        6. Appends cycle audit log entry.
        7. Returns a strongly-validated SLACycleSummary.

        Args:
            reference_time: Optional reference UTC timestamp (defaults to current UTC time).

        Returns:
            SLACycleSummary detailing counts, outcomes, and any operational errors.
        """
        now_utc = reference_time or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        # In-process concurrency guard
        if not self._lock.acquire(blocking=False):
            logger.warning("SLA evaluation cycle already in progress; skipping overlapping invocation.")
            return SLACycleSummary(
                timestamp=now_utc,
                skipped_count=1,
                errors=["Cycle skipped: another SLA evaluation cycle is currently in progress."],
            )

        start_time = time.perf_counter()
        summary = SLACycleSummary(timestamp=now_utc)

        try:
            logger.info("Starting SLA evaluation cycle %s at %s...", summary.cycle_id, now_utc.isoformat())

            # Query all briefs to partition into eligible vs skipped
            with self.session_factory() as session:
                all_briefs = session.query(ChangeBrief).all()

                eligible_brief_ids: List[str] = []
                for b in all_briefs:
                    if b.status in (
                        BriefStatus.ASSIGNED.value,
                        BriefStatus.IN_REVIEW.value,
                        BriefStatus.DEFERRED.value,
                    ):
                        eligible_brief_ids.append(b.id)
                    elif b.status in (
                        BriefStatus.DECIDED.value,
                        BriefStatus.CLOSED.value,
                    ):
                        summary.skipped_count += 1
                    else:
                        # Draft without assignment or undefined status
                        summary.skipped_count += 1

            # Process eligible briefs with strict failure isolation
            for brief_id in eligible_brief_ids:
                try:
                    res: SLAEscalationResult = self.governance_agent.evaluate_sla(
                        change_brief_id=brief_id,
                        current_time=now_utc,
                    )
                    summary.evaluated_count += 1
                    summary.results.append(res)

                    if res.is_overdue:
                        summary.overdue_count += 1
                        if res.escalated:
                            summary.escalated_count += 1
                            logger.warning(
                                "G5 SLA Escalation triggered for ChangeBrief %s (deadline: %s).",
                                brief_id,
                                res.sla_deadline,
                            )
                        else:
                            summary.already_escalated_count += 1
                            logger.debug(
                                "ChangeBrief %s is overdue but already escalated.",
                                brief_id,
                            )
                except Exception as exc:
                    summary.failed_count += 1
                    err_msg = f"Evaluation failed for ChangeBrief '{brief_id}': {exc}"
                    summary.errors.append(err_msg)
                    logger.error(err_msg, exc_info=True)

            summary.duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._last_cycle_summary = summary

            # Audit record for cycle completion
            self._record_cycle_audit(
                action="sla_cycle_completed",
                reason=(
                    f"SLA cycle completed: {summary.evaluated_count} evaluated, "
                    f"{summary.escalated_count} escalated, "
                    f"{summary.already_escalated_count} already escalated, "
                    f"{summary.skipped_count} skipped, "
                    f"{summary.failed_count} failed."
                ),
                metadata={
                    "cycle_id": summary.cycle_id,
                    "evaluated_count": summary.evaluated_count,
                    "overdue_count": summary.overdue_count,
                    "escalated_count": summary.escalated_count,
                    "already_escalated_count": summary.already_escalated_count,
                    "skipped_count": summary.skipped_count,
                    "failed_count": summary.failed_count,
                    "duration_ms": summary.duration_ms,
                },
            )

            logger.info(
                "SLA evaluation cycle %s finished in %.1fms (Evaluated: %d, Escalated: %d, Already: %d, Failed: %d).",
                summary.cycle_id,
                summary.duration_ms,
                summary.evaluated_count,
                summary.escalated_count,
                summary.already_escalated_count,
                summary.failed_count,
            )
            return summary

        except Exception as cycle_exc:
            summary.failed_count += 1
            summary.errors.append(f"Fatal error during SLA cycle: {cycle_exc}")
            summary.duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("Unexpected exception in SLA evaluation cycle: %s", cycle_exc, exc_info=True)

            self._record_cycle_audit(
                action="sla_cycle_failed",
                reason=f"SLA cycle failed unexpectedly: {cycle_exc}",
                metadata={"error": str(cycle_exc), "cycle_id": summary.cycle_id},
            )
            return summary

        finally:
            self._lock.release()

    # ==========================================================================
    # AUDIT LOGGING HELPER
    # ==========================================================================

    def _record_cycle_audit(
        self,
        action: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a cycle or scheduler lifecycle event in the immutable AuditLog."""
        try:
            with self.session_factory() as session:
                entry = AuditLog(
                    id=str(uuid.uuid4()),
                    entity_id=metadata.get("cycle_id", str(uuid.uuid4())) if metadata else str(uuid.uuid4()),
                    entity_type="SLAScheduler",
                    previous_status=None,
                    new_status=action,
                    actor="system_sla_scheduler",
                    reason=reason,
                    audit_metadata={"action": action, **(metadata or {})},
                    schema_version="1.0",
                )
                session.add(entry)
                session.commit()
        except Exception as audit_exc:
            logger.error("Failed to persist AuditLog for SLAScheduler action '%s': %s", action, audit_exc)
