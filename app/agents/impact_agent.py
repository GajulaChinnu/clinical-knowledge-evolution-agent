"""CKEA Impact Assessment Agent for deterministic multidimensional protocol gap evaluation."""

import logging
from typing import Any, List, Optional
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import AuditLog, ChangeRecord, GapRecord, ImpactRecord
from app.schemas.changes import ChangeStatus
from app.schemas.gaps import GapStatus
from app.schemas.impact import ImpactStatus, ImpactTier, ScoringResult
from app.services.config_service import AppConfig, load_config
from app.services.scoring_engine import (
    ScoringEngine,
    TIER_SEVERITY_ORDER,
    TierDowngradeBlockedError,
    UnresolvedComparisonError,
)

logger = logging.getLogger("ckea.impact")


class ImpactAgent:
    """Specialized agent responsible for calculating and persisting deterministic impact scores for confirmed gaps."""

    def __init__(
        self,
        session_factory: Optional[sessionmaker[Session]] = None,
        scoring_engine: Optional[ScoringEngine] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_config()
        self.session_factory = session_factory or get_session_factory()
        self.scoring_engine = scoring_engine or ScoringEngine(config=self.config)

    def process_gap_record(
        self,
        gap_record_id: str,
        urgency_input: Optional[Any] = None,
        evidence_input: Optional[Any] = None,
        breadth_input: Optional[Any] = None,
        reviewer: Optional[str] = None,
        correction_rationale: Optional[str] = None,
    ) -> ImpactRecord:
        """Score the clinical impact of a confirmed gap or G3-confirmed no-match.

        Args:
            gap_record_id: Unique UUID of the GapRecord.
            urgency_input: Optional explicit urgency input; if None, derived from gap.difference_type.
            evidence_input: Optional explicit evidence input; if None, derived from change.evidence_grade.
            breadth_input: Optional explicit pathway breadth input.
            reviewer: Optional reviewer identity required if a manual tier downgrade is performed.
            correction_rationale: Optional clinical reasoning required if a manual tier downgrade is performed.

        Returns:
            Persisted and committed ImpactRecord.

        Raises:
            ValueError: If GapRecord does not exist.
            UnresolvedComparisonError: If the finding is held for review or incomplete.
            TierDowngradeBlockedError: If an existing tier is automatically downgraded without reviewer authorization.
        """
        session: Session = self.session_factory()
        try:
            gap = session.get(GapRecord, gap_record_id)
            if not gap:
                raise ValueError(f"GapRecord '{gap_record_id}' not found.")

            change = session.get(ChangeRecord, gap.change_record_id)
            if not change:
                raise ValueError(f"ChangeRecord '{gap.change_record_id}' not found for GapRecord '{gap_record_id}'.")

            # 1. Validate that the finding is scoreable
            self._validate_scoreable(gap=gap, change=change)

            # 2. Derive dimension inputs from data when not explicitly provided
            resolved_urgency = urgency_input if urgency_input is not None else gap.difference_type
            resolved_evidence = evidence_input if evidence_input is not None else change.evidence_grade
            resolved_breadth = breadth_input

            # 3. Deterministically compute multidimensional score
            result: ScoringResult = self.scoring_engine.calculate_impact(
                urgency_input=resolved_urgency,
                evidence_input=resolved_evidence,
                breadth_input=resolved_breadth,
            )

            # 4. Check for existing record and enforce Tier Immutability (Section 8)
            existing = session.query(ImpactRecord).filter_by(gap_record_id=gap.id).first()
            if existing and existing.tier:
                self._check_tier_immutability(
                    session=session,
                    existing=existing,
                    new_result=result,
                    reviewer=reviewer,
                    correction_rationale=correction_rationale,
                )

            # 5. Persist ImpactRecord idempotently
            rule_id_list = list(result.rule_ids.values())
            tier_val = result.tier.value if result.tier else None

            if existing:
                existing.clinical_urgency = result.clinical_urgency
                existing.evidence_strength = result.evidence_strength
                existing.pathway_breadth = result.pathway_breadth
                existing.rule_ids = rule_id_list
                existing.scoring_yaml_version = result.scoring_yaml_version
                existing.total_score = result.total_score
                existing.tier = tier_val
                existing.routing_target = result.routing_target
                existing.sla_deadline = result.sla_deadline
                existing.urgency_basis = result.urgency_basis
                existing.evidence_basis = result.evidence_basis
                existing.breadth_basis = result.breadth_basis
                existing.status = result.status.value
                impact_record = existing
            else:
                impact_record = ImpactRecord(
                    gap_record_id=gap.id,
                    clinical_urgency=result.clinical_urgency,
                    evidence_strength=result.evidence_strength,
                    pathway_breadth=result.pathway_breadth,
                    rule_ids=rule_id_list,
                    scoring_yaml_version=result.scoring_yaml_version,
                    total_score=result.total_score,
                    tier=tier_val,
                    routing_target=result.routing_target,
                    sla_deadline=result.sla_deadline,
                    urgency_basis=result.urgency_basis,
                    evidence_basis=result.evidence_basis,
                    breadth_basis=result.breadth_basis,
                    status=result.status.value,
                    schema_version="1.0",
                )
                session.add(impact_record)

            session.commit()
            session.refresh(impact_record)
            session.expunge(impact_record)
            return impact_record

        except Exception as e:
            session.rollback()
            logger.error("Impact assessment failed for GapRecord %s: %s", gap_record_id, e)
            raise
        finally:
            session.close()

    def _validate_scoreable(self, gap: GapRecord, change: ChangeRecord) -> None:
        """Verify that comparison is finalized and not held for G1/G2 review."""
        # Unresolved G2 review
        if gap.status == GapStatus.REVIEW_REQUIRED.value or change.status == ChangeStatus.HELD_FOR_G2.value:
            raise UnresolvedComparisonError(
                f"Cannot score unresolved comparison finding (GapRecord {gap.id}) held for G2 review."
            )

        # Unresolved pipeline states
        if change.status in (
            ChangeStatus.EXTRACTED.value,
            ChangeStatus.HELD_FOR_G1.value,
            "comparison_pending",
        ):
            raise UnresolvedComparisonError(
                f"Cannot score finding before comparison is completed (current change status: '{change.status}')."
            )

        # G3 findings: must be confirmed as a no-match finding
        if gap.comparison_result == "no_match" or change.status == ChangeStatus.HELD_FOR_G3.value:
            # G3 findings are valid for scoring
            return

        # Confirmed gap or matched finding
        if change.status == ChangeStatus.GAP_CONFIRMED.value or gap.status == GapStatus.MATCHED.value:
            return

        # Any other unscoreable state
        raise UnresolvedComparisonError(
            f"Finding {gap.id} with status '{gap.status}' and change status '{change.status}' is not scoreable."
        )

    def _check_tier_immutability(
        self,
        session: Session,
        existing: ImpactRecord,
        new_result: ScoringResult,
        reviewer: Optional[str],
        correction_rationale: Optional[str],
    ) -> None:
        """Prevent automatic tier downgrade unless authorized by named reviewer and rationale."""
        old_tier = existing.tier
        new_tier = new_result.tier.value if new_result.tier else None

        if not old_tier:
            return

        old_rank = TIER_SEVERITY_ORDER.get(old_tier, 0)
        new_rank = TIER_SEVERITY_ORDER.get(new_tier, 0) if new_tier else 0

        # If new score results in a lower tier (or unsets tier to incomplete)
        if new_rank < old_rank:
            if not reviewer or not str(reviewer).strip() or not correction_rationale or not str(correction_rationale).strip():
                raise TierDowngradeBlockedError(
                    f"Automatic tier downgrade from '{old_tier}' to '{new_tier or 'incomplete'}' is blocked. "
                    "Manual correction requires a named reviewer and clinical rationale."
                )

            # Record authorized tier downgrade in append-only AuditLog
            audit = AuditLog(
                entity_id=existing.id,
                entity_type="ImpactRecord",
                previous_status=old_tier,
                new_status=new_tier or "incomplete",
                actor=reviewer.strip(),
                reason=correction_rationale.strip(),
                audit_metadata={
                    "action": "tier_downgrade_correction",
                    "previous_tier": old_tier,
                    "new_tier": new_tier,
                    "previous_score": existing.total_score,
                    "new_score": new_result.total_score,
                },
                schema_version="1.0",
            )
            session.add(audit)
            logger.info(
                "Authorized tier downgrade for ImpactRecord %s from '%s' to '%s' by '%s'",
                existing.id,
                old_tier,
                new_tier,
                reviewer,
            )
