"""CKEA Impact Assessment Agent for deterministic multidimensional protocol gap evaluation."""

import logging
from typing import Any, Dict, List, Optional, Sequence
import uuid
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeRecord,
    GapRecord,
    GuidanceChange,
    GuidanceStatement,
    ImpactRecord,
    IngestedDocument,
)
from app.schemas.treatment_check import Finding
from app.services.clinical_statements import similarity, skeleton
from app.services.ranking import (
    CATEGORY_URGENCY_RULE,
    RankingRules,
    finding_urgency_rule,
    get_ranking_rules,
    priority,
    source_quality,
)
from app.services.taxonomy import Taxonomy, get_taxonomy
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

            self._rank_impact(session, impact_record, gap, change)
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


    # ==========================================================================
    # RELEVANCE, EVIDENCE GRADING AND RANKING (orders work; never changes a tier)
    # ==========================================================================

    def _rank_impact(self, session: Session, impact: ImpactRecord, gap: GapRecord, change: ChangeRecord) -> None:
        """Record relevance, source quality, novelty, priority and affected scope on an ImpactRecord."""
        rules = get_ranking_rules()
        taxonomy = get_taxonomy()
        doc = session.get(IngestedDocument, change.ingested_document_id)
        meta = (doc.doc_metadata or {}) if doc else {}
        treatments = list(change.treatments or taxonomy.find_treatments(change.verbatim_text).treatments)
        departments = list(change.departments or meta.get("departments") or taxonomy.departments_for_treatments(treatments))
        if gap.matched_protocol_id:
            legacy = taxonomy.legacy_metadata_for(gap.matched_protocol_id)
            if legacy and legacy.department not in departments:
                departments.append(legacy.department)
        rel_rule = rules.relevance_rule("REL-01" if treatments and departments else "REL-03" if departments else "REL-05")
        quality, quality_basis = source_quality(rules, meta.get("quality_tier"), change.evidence_grade[-1:] if change.evidence_grade else None)
        urg_rule = rules.urgency_rule(CATEGORY_URGENCY_RULE.get(change.change_category or "", "URG-04"))
        novelty = "revision_of_known" if (gap.is_match or change.change_category in ("dose_change", "threshold_change", "revised_recommendation")) else "new"
        impact.relevance = rel_rule.score
        impact.source_quality = quality
        impact.novelty = novelty
        impact.priority_score = priority(rules, urg_rule.score, rel_rule.score, quality, novelty)
        impact.affected_departments = sorted(set(departments))
        impact.affected_pathways = taxonomy.pathways_for(treatments, departments)
        impact.ranking_basis = {
            "rules_version": rules.version,
            "relevance": {"rule_id": rel_rule.id, "basis": rel_rule.basis},
            "urgency": {"rule_id": urg_rule.id, "basis": urg_rule.basis},
            "source_quality": {"score": quality, "basis": quality_basis},
            "novelty": novelty,
            "note": "Priority orders the governance queue only; it does not change the impact tier.",
        }

    def score_findings(
        self,
        department: str,
        plan_treatments: Sequence[str],
        findings: List[Finding],
        rules: Optional[RankingRules] = None,
    ) -> List[Finding]:
        """Clinician query mode: grade and rank findings (relevance, urgency, source quality, novelty)."""
        rules = rules or get_ranking_rules()
        tiers: Dict[str, Optional[int]] = {}
        with self.session_factory() as session:
            for f in findings:
                if f.citation.document_id not in tiers:
                    doc = session.get(IngestedDocument, f.citation.document_id)
                    tiers[f.citation.document_id] = (doc.doc_metadata or {}).get("quality_tier") if doc else None

        for f in findings:
            direct = bool(set(f.treatments) & set(plan_treatments))
            if f.relation == "not_applicable":
                rel_id = "REL-04"
            elif not direct:
                rel_id = "REL-03"
            elif department in f.departments:
                rel_id = "REL-01"
            else:
                rel_id = "REL-02"
            rel = rules.relevance_rule(rel_id)
            urg = rules.urgency_rule(finding_urgency_rule(f.kind, f.relation, f.change_category))
            quality, quality_basis = source_quality(rules, tiers.get(f.citation.document_id), _evidence_level(f.citation.excerpt))
            f.relevance, f.urgency, f.source_quality = rel.score, urg.score, quality
            f.novelty = "revision_of_known" if (f.previous_citation or f.change_category) else "new"
            f.ranking_basis = {
                "rules_version": rules.version,
                "relevance": {"rule_id": rel.id, "basis": rel.basis},
                "urgency": {"rule_id": urg.id, "basis": urg.basis},
                "source_quality": {"score": quality, "basis": quality_basis},
            }

        # Duplicates across sources: the higher-quality source is canonical; others link to it.
        ordered = sorted(findings, key=lambda f: (-(f.source_quality or 0), f.citation.published_date or ""))
        for i, f in enumerate(ordered):
            for canonical in ordered[:i]:
                if (canonical.citation.source_identity != f.citation.source_identity
                        and set(canonical.treatments) == set(f.treatments)
                        and similarity(skeleton(canonical.citation.excerpt), skeleton(f.citation.excerpt)) >= rules.duplicate_similarity):
                    f.novelty = "duplicate_of_existing"
                    f.duplicate_of = canonical.citation.statement_id
                    break
        for f in findings:
            f.priority_score = priority(rules, f.urgency, f.relevance, f.source_quality, f.novelty)
            f.ranking_basis["novelty"] = f.novelty
        return sorted(findings, key=lambda f: -(f.priority_score or 0))

    def assess_guidance_changes(self, document_ids: Optional[Sequence[str]] = None) -> int:
        """Background mode: relevance-filter, grade and rank detected source changes."""
        rules = get_ranking_rules()
        with self.session_factory() as session:
            q = session.query(GuidanceChange)
            if document_ids is not None:
                q = q.filter(GuidanceChange.to_document_id.in_(list(document_ids)))
            changes = q.all()
            docs: Dict[str, IngestedDocument] = {}
            statements: Dict[str, GuidanceStatement] = {}

            def doc_of(doc_id):
                if doc_id not in docs:
                    docs[doc_id] = session.get(IngestedDocument, doc_id)
                return docs[doc_id]

            def stmt_of(stmt_id):
                if stmt_id and stmt_id not in statements:
                    statements[stmt_id] = session.get(GuidanceStatement, stmt_id)
                return statements.get(stmt_id) if stmt_id else None

            for change in changes:
                meta = (doc_of(change.to_document_id).doc_metadata or {})
                stmt = stmt_of(change.to_statement_id) or stmt_of(change.from_statement_id)
                if change.change_category == "no_practice_change":
                    if change.relevance_status == "active":
                        change.relevance_status = "filtered_not_practice_changing"
                        change.filter_reason = f"FLT-01: {rules.filters.get('FLT-01', 'No practice change')}"
                    rel = rules.relevance_rule("REL-05")
                    urg = rules.urgency_rule("URG-06")
                else:
                    classes = list((stmt.treatment_classes or []) if stmt else [])
                    rel = rules.relevance_rule("REL-01" if change.treatments else "REL-03" if classes else "REL-05")
                    urg = rules.urgency_rule(CATEGORY_URGENCY_RULE.get(change.change_category, "URG-04"))
                quality, quality_basis = source_quality(rules, meta.get("quality_tier"), stmt.evidence_level if stmt else None)
                change.relevance, change.urgency, change.source_quality = rel.score, urg.score, quality
                change.novelty = "revision_of_known" if change.from_statement_id else "new"
                change.ranking_basis = {
                    "rules_version": rules.version,
                    "relevance": {"rule_id": rel.id, "basis": rel.basis},
                    "urgency": {"rule_id": urg.id, "basis": urg.basis},
                    "source_quality": {"score": quality, "basis": quality_basis},
                }

            # Duplicates: same treatment statement published by several sources.
            active = [c for c in changes if c.relevance_status != "filtered_not_practice_changing" and c.to_statement_id]
            ordered = sorted(active, key=lambda c: (-(c.source_quality or 0), c.created_at))
            for i, change in enumerate(ordered):
                change.duplicate_of_change_id = None
                text = stmt_of(change.to_statement_id).verbatim_text
                for canonical in ordered[:i]:
                    if canonical.source_identity == change.source_identity or set(canonical.treatments or []) != set(change.treatments or []):
                        continue
                    if similarity(skeleton(stmt_of(canonical.to_statement_id).verbatim_text), skeleton(text)) >= rules.duplicate_similarity:
                        change.novelty = "duplicate_of_existing"
                        change.duplicate_of_change_id = canonical.id
                        break
            for change in changes:
                change.priority_score = priority(rules, change.urgency or 1, change.relevance or 1,
                                                 change.source_quality or 1, change.novelty or "new")
                change.ranking_basis = {**(change.ranking_basis or {}), "novelty": change.novelty}
            session.commit()
            return len(changes)

    def ranked_feed(self, department: Optional[str] = None, include_filtered: bool = False) -> List[GuidanceChange]:
        """Changes into the latest version of each source, highest priority first."""
        with self.session_factory() as session:
            superseded = {d.previous_source_version_id for d in session.query(IngestedDocument) if d.previous_source_version_id}
            q = session.query(GuidanceChange).filter(~GuidanceChange.to_document_id.in_(superseded))
            rows = [c for c in q.all() if include_filtered or c.relevance_status != "filtered_not_practice_changing"]
            if department:
                rows = [c for c in rows if department in (c.departments or [])]
            for c in rows:
                session.expunge(c)
            return sorted(rows, key=lambda c: (-(c.priority_score or 0), c.created_at))

    def restore_filtered_change(self, change_id: str, reviewer_id: str, reason: str, auth_service: Any) -> GuidanceChange:
        """Human override: restore a filtered change to the active queue (authorised reviewer, audited)."""
        if not auth_service.is_authorized(reviewer_id):
            raise PermissionError(f"Reviewer '{reviewer_id}' is not authorised to restore filtered changes.")
        if not reason or not reason.strip():
            raise ValueError("A reason is required to restore a filtered change.")
        with self.session_factory() as session:
            change = session.get(GuidanceChange, change_id)
            if change is None:
                raise ValueError(f"GuidanceChange '{change_id}' not found.")
            previous = change.relevance_status
            if previous == "active":
                raise ValueError("Change is not filtered.")
            change.relevance_status = "restored_by_reviewer"
            session.add(AuditLog(
                id=str(uuid.uuid4()), entity_id=change.id, entity_type="GuidanceChange",
                previous_status=previous, new_status="restored_by_reviewer", actor=reviewer_id,
                reason=reason.strip(), audit_metadata={"action": "restore_filtered_change", "filter_reason": change.filter_reason},
                schema_version="1.0",
            ))
            session.commit()
            session.refresh(change)
            session.expunge(change)
            return change


def _evidence_level(text: str) -> Optional[str]:
    import re

    m = re.search(r"\(\s*evidence level\s+([A-D])\s*\)", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else None
