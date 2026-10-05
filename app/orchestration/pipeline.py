"""CKEA Pipeline Orchestrator and Controlled State Machine.

Coordinates the end-to-end document-to-governance lifecycle:
Monitoring -> Extraction -> Comparison -> Impact -> Briefing -> Governance

Enforces:
1. State-aware execution and checkpoints (no blind re-runs)
2. Strict human gate boundaries: G1, G2, G3, G4
3. Complete idempotency across all stages (zero duplicate entities)
4. Failure isolation in document and batch processing
5. Zero direct LLM/API calls from the orchestrator
6. Append-only audit logging and protocol immutability
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.briefing_agent import BriefingAgent
from app.agents.comparison_agent import ComparisonAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.impact_agent import ImpactAgent
from app.agents.monitoring_agent import MonitoringAgent, ScanResult
from app.models.database import get_session_factory
from app.models.entities import (
    AuditLog,
    ChangeBrief,
    ChangeRecord,
    GapRecord,
    ImpactRecord,
    IngestedDocument,
    ReviewAssignment,
)
from app.schemas.briefs import BriefStatus
from app.schemas.changes import ChangeStatus
from app.schemas.documents import DocumentStatus
from app.schemas.gaps import ComparisonResult, DifferenceType, GapStatus
from app.schemas.impact import ImpactStatus
from app.schemas.orchestration import (
    ArtifactsManifest,
    BatchPipelineResult,
    HumanGate,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
)
from app.schemas.transitions import (
    CHANGE_TRANSITIONS,
    DOCUMENT_TRANSITIONS,
    validate_transition,
)
from app.services.config_service import AppConfig, load_config
from app.services.file_hash import compute_sha256

logger = logging.getLogger("ckea.orchestration.pipeline")


class ClinicalKnowledgePipeline:
    """Deterministic orchestrator managing the CKEA clinical knowledge evolution lifecycle."""

    def __init__(
        self,
        session_factory: Optional[sessionmaker[Session]] = None,
        config: Optional[AppConfig] = None,
        monitoring_agent: Optional[MonitoringAgent] = None,
        extraction_agent: Optional[ExtractionAgent] = None,
        comparison_agent: Optional[ComparisonAgent] = None,
        impact_agent: Optional[ImpactAgent] = None,
        briefing_agent: Optional[BriefingAgent] = None,
        governance_agent: Optional[GovernanceAgent] = None,
        default_reviewer_id: str = "dr_smith",
        default_reviewer_role: str = "Clinical Governance Lead",
        default_breadth_input: str = "one_specialty",
        default_g3_urgency_input: str = "time_sensitive_treatment",
        max_retries: int = 3,
    ) -> None:
        """Initialize the Clinical Knowledge Pipeline.

        All agents and factories can be injected for deterministic isolated testing.
        """
        self.config = config or load_config()
        self.session_factory = session_factory or get_session_factory(self.config.database_url)
        self.default_reviewer_id = default_reviewer_id
        self.default_reviewer_role = default_reviewer_role
        self.default_breadth_input = default_breadth_input
        self.default_g3_urgency_input = default_g3_urgency_input
        self.max_retries = max_retries

        # Agent dependencies (lazily initialized if not provided)
        self.monitoring_agent = monitoring_agent or MonitoringAgent(
            session_factory=self.session_factory,
            config=self.config,
        )
        self.extraction_agent = extraction_agent or ExtractionAgent(
            session_factory=self.session_factory,
            config=self.config,
        )
        self.comparison_agent = comparison_agent or ComparisonAgent(
            session_factory=self.session_factory,
            config=self.config,
        )
        self.impact_agent = impact_agent or ImpactAgent(
            session_factory=self.session_factory,
            config=self.config,
        )
        self.briefing_agent = briefing_agent or BriefingAgent(
            session_factory=self.session_factory,
            config=self.config,
        )
        self.governance_agent = governance_agent or GovernanceAgent(
            session_factory=self.session_factory,
            config=self.config,
        )

    # ==========================================================================
    # AUDIT LOGGING HELPER
    # ==========================================================================

    def _record_audit(
        self,
        session: Session,
        entity_id: str,
        entity_type: str,
        previous_status: Optional[str],
        new_status: str,
        action: str,
        reason: Optional[str] = None,
        stage: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        actor: str = "pipeline_orchestrator",
    ) -> AuditLog:
        """Append an auditable event to the immutable AuditLog."""
        meta = {"action": action}
        if stage:
            meta["stage"] = stage
        if extra_metadata:
            meta.update(extra_metadata)

        entry = AuditLog(
            entity_id=entity_id,
            entity_type=entity_type,
            previous_status=previous_status,
            new_status=new_status,
            actor=actor,
            reason=reason,
            audit_metadata=meta,
            schema_version="1.0",
        )
        session.add(entry)
        session.commit()
        return entry

    # ==========================================================================
    # ENTRY POINT 1: PROCESS DOCUMENT (FILE PATH OR UUID)
    # ==========================================================================

    def process_document(
        self,
        source_path_or_id: Union[str, Path],
    ) -> PipelineResult:
        """Process a clinical guideline document from discovery through governance assignment.

        Args:
            source_path_or_id: Either an existing IngestedDocument UUID or a path to a source PDF.

        Returns:
            PipelineResult communicating final status, completed stages, and produced artifacts.
        """
        # 1. Check if source_path_or_id is an existing IngestedDocument ID
        str_val = str(source_path_or_id).strip()
        try:
            uuid.UUID(str_val)
            with self.session_factory() as session:
                existing_doc = session.get(IngestedDocument, str_val)
                if existing_doc:
                    return self.process_ingested_document(existing_doc.id)
        except (ValueError, TypeError):
            pass

        # 2. Treat as file path
        file_path = Path(source_path_or_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Source document file not found: {file_path}")

        file_hash = compute_sha256(file_path)
        source_id = file_path.stem

        with self.session_factory() as session:
            # Check for existing document by hash (idempotency check)
            existing_doc = (
                session.query(IngestedDocument)
                .filter_by(sha256_hash=file_hash)
                .order_by(IngestedDocument.created_at.desc())
                .first()
            )
            if existing_doc and existing_doc.status != DocumentStatus.FAILED.value:
                logger.info(
                    "Document '%s' already ingested (ID: %s, status: %s); resuming pipeline from persisted state.",
                    file_path.name,
                    existing_doc.id,
                    existing_doc.status,
                )
                return self.process_ingested_document(existing_doc.id)

            # Ingest via monitoring agent
            scan_res = ScanResult()
            self.monitoring_agent._process_file(file_path, session, scan_res)

            if scan_res.failed > 0:
                logger.error("Monitoring failed to ingest document '%s'", file_path.name)
                return PipelineResult(
                    source_identifier=source_id,
                    status=PipelineStatus.FAILED,
                    current_stage=PipelineStage.MONITORING,
                    reason=f"Failed to ingest document '{file_path.name}'",
                    errors=[f"Ingestion failure recorded: {scan_res.failure_ids}"],
                )

            if scan_res.ingested_document_ids:
                doc_id = scan_res.ingested_document_ids[0]
                return self.process_ingested_document(doc_id)

            # Skipped due to unchanged content
            existing_doc = (
                session.query(IngestedDocument)
                .filter_by(source_identifier=source_id)
                .order_by(IngestedDocument.created_at.desc())
                .first()
            )
            if existing_doc:
                return self.process_ingested_document(existing_doc.id)

            raise RuntimeError(f"Unexpected monitoring outcome for file: {file_path.name}")

    # ==========================================================================
    # ENTRY POINT 2: PROCESS INGESTED DOCUMENT
    # ==========================================================================

    def process_ingested_document(
        self,
        document_id: str,
    ) -> PipelineResult:
        """Execute state-aware pipeline progression for an ingested document.

        Args:
            document_id: UUID string of the IngestedDocument record.

        Returns:
            PipelineResult communicating final status, held gates, and artifacts.
        """
        with self.session_factory() as session:
            doc = session.get(IngestedDocument, document_id)
            if not doc:
                raise ValueError(f"IngestedDocument with ID '{document_id}' not found.")

            res = PipelineResult(
                document_id=doc.id,
                source_identifier=doc.source_identifier,
                retry_count=doc.retry_count,
            )
            res.artifacts.document_id = doc.id

            # Audit start
            self._record_audit(
                session=session,
                entity_id=doc.id,
                entity_type="IngestedDocument",
                previous_status=None,
                new_status=doc.status,
                action="pipeline_started",
                stage=PipelineStage.MONITORING.value,
                reason=f"Pipeline started for document '{doc.source_identifier}'.",
            )
            res.completed_stages.append(PipelineStage.MONITORING)

            # Check if document is already in terminal or failed state
            if doc.status == DocumentStatus.FAILED.value:
                if doc.retry_count >= self.max_retries:
                    res.status = PipelineStatus.FAILED
                    res.current_stage = PipelineStage.MONITORING
                    res.reason = f"Document is in failed state and max retries ({self.max_retries}) have been reached."
                    res.errors.append(doc.error_message or "Max retries exceeded.")
                    return res
                # Otherwise, increment retry and re-attempt
                doc.retry_count += 1
                doc.status = DocumentStatus.DISCOVERED.value
                session.commit()
                res.retry_count = doc.retry_count

            # Check if document is currently held (G1)
            if doc.status == DocumentStatus.HELD.value:
                res.status = PipelineStatus.HELD
                res.current_stage = PipelineStage.EXTRACTION
                res.blocked_stage = PipelineStage.EXTRACTION
                res.held_gate = HumanGate.G1
                res.reason = "Document is held for G1 extraction review."
                # Collect existing change records
                res.artifacts.change_record_ids = [c.id for c in doc.change_records]
                return res

            # Document is ready for extraction
            # 1. Check if extraction already completed (idempotency: avoid re-running extraction)
            existing_changes = list(doc.change_records)
            if doc.status == DocumentStatus.COMPLETE.value and existing_changes:
                logger.info("Extraction already complete for document %s; using %d existing change record(s).", doc.id, len(existing_changes))
                changes = existing_changes
            elif existing_changes:
                # Changes already extracted but document not yet marked complete
                changes = existing_changes
            else:
                # Run extraction agent
                try:
                    changes = self.extraction_agent.process_document(doc.id)
                except Exception as e:
                    logger.error("ExtractionAgent failed on document %s: %s", doc.id, e)
                    doc.status = DocumentStatus.FAILED.value
                    doc.error_message = str(e)
                    session.commit()
                    self._record_audit(
                        session=session,
                        entity_id=doc.id,
                        entity_type="IngestedDocument",
                        previous_status=DocumentStatus.PARSED.value,
                        new_status=DocumentStatus.FAILED.value,
                        action="stage_failed",
                        stage=PipelineStage.EXTRACTION.value,
                        reason=f"Extraction failed: {e}",
                    )
                    res.status = PipelineStatus.FAILED
                    res.current_stage = PipelineStage.EXTRACTION
                    res.reason = f"Extraction failed: {e}"
                    res.errors.append(str(e))
                    return res

            # Re-read doc state after extraction
            session.refresh(doc)
            res.completed_stages.append(PipelineStage.EXTRACTION)
            res.artifacts.change_record_ids = [c.id for c in changes]

            # Check if any change is held for G1
            held_g1_changes = [c for c in changes if c.status == ChangeStatus.HELD_FOR_G1.value]
            if held_g1_changes:
                self._record_audit(
                    session=session,
                    entity_id=doc.id,
                    entity_type="IngestedDocument",
                    previous_status=doc.status,
                    new_status=DocumentStatus.HELD.value,
                    action="stage_held",
                    stage=PipelineStage.EXTRACTION.value,
                    reason=f"{len(held_g1_changes)} recommendation(s) held for G1 extraction review.",
                )
                res.status = PipelineStatus.HELD
                res.current_stage = PipelineStage.EXTRACTION
                res.blocked_stage = PipelineStage.EXTRACTION
                res.held_gate = HumanGate.G1
                res.reason = f"{len(held_g1_changes)} recommendation(s) require G1 extraction review."
                return res

            if not changes:
                # Document extraction complete, but no recommendations found
                res.status = PipelineStatus.COMPLETED
                res.current_stage = PipelineStage.EXTRACTION
                res.reason = "Document extraction completed; zero clinical recommendations found."
                return res

        # Process each ChangeRecord downstream (failure isolation per change)
        any_held = False
        any_failed = False
        for change_id in res.artifacts.change_record_ids:
            try:
                change_res = self.process_change(change_id)
                # Merge artifacts
                res.artifacts.gap_record_ids.extend(
                    [g for g in change_res.artifacts.gap_record_ids if g not in res.artifacts.gap_record_ids]
                )
                res.artifacts.impact_record_ids.extend(
                    [i for i in change_res.artifacts.impact_record_ids if i not in res.artifacts.impact_record_ids]
                )
                res.artifacts.change_brief_ids.extend(
                    [b for b in change_res.artifacts.change_brief_ids if b not in res.artifacts.change_brief_ids]
                )
                res.artifacts.review_assignment_ids.extend(
                    [a for a in change_res.artifacts.review_assignment_ids if a not in res.artifacts.review_assignment_ids]
                )

                for st in change_res.completed_stages:
                    if st not in res.completed_stages:
                        res.completed_stages.append(st)

                if change_res.is_held:
                    any_held = True
                    res.blocked_stage = change_res.blocked_stage
                    res.held_gate = change_res.held_gate
                    res.reason = change_res.reason
                elif change_res.is_failed:
                    any_failed = True
                    res.errors.extend(change_res.errors)

            except Exception as e:
                logger.error("Processing failed for ChangeRecord %s: %s", change_id, e)
                any_failed = True
                res.errors.append(f"ChangeRecord {change_id} error: {e}")

        if any_held:
            res.status = PipelineStatus.HELD
            res.current_stage = res.blocked_stage
        elif any_failed:
            res.status = PipelineStatus.FAILED
            res.current_stage = PipelineStage.COMPARISON
            res.reason = "One or more change records failed processing."
        else:
            res.status = PipelineStatus.COMPLETED
            res.current_stage = res.completed_stages[-1] if res.completed_stages else PipelineStage.EXTRACTION
            res.reason = "Document processing successfully reached completion."

        return res

    # ==========================================================================
    # ENTRY POINT 3: PROCESS CHANGE RECORD
    # ==========================================================================

    def process_change(
        self,
        change_record_id: str,
    ) -> PipelineResult:
        """Process a single clinical recommendation through Comparison, Impact, Briefing, and Governance.

        Args:
            change_record_id: UUID string of the ChangeRecord.

        Returns:
            PipelineResult tracking progression for this discrete change.
        """
        with self.session_factory() as session:
            change = session.get(ChangeRecord, change_record_id)
            if not change:
                raise ValueError(f"ChangeRecord with ID '{change_record_id}' not found.")

            res = PipelineResult(
                document_id=change.ingested_document_id,
            )
            res.artifacts.change_record_ids.append(change.id)
            res.completed_stages.append(PipelineStage.EXTRACTION)

            # 1. G1 Gate Check
            if change.status == ChangeStatus.HELD_FOR_G1.value:
                res.status = PipelineStatus.HELD
                res.current_stage = PipelineStage.EXTRACTION
                res.blocked_stage = PipelineStage.EXTRACTION
                res.held_gate = HumanGate.G1
                res.reason = "ChangeRecord is held for G1 extraction review."
                return res

            # 2. Check if change is terminal no_gap
            if change.status == ChangeStatus.NO_GAP.value:
                res.status = PipelineStatus.COMPLETED
                res.current_stage = PipelineStage.COMPARISON
                res.completed_stages.append(PipelineStage.COMPARISON)
                res.reason = "Recommendation resolved as no material protocol gap."
                return res

            # 3. Protocol Comparison Stage
            gap = session.query(GapRecord).filter_by(change_record_id=change.id).first()
            if gap is None or change.status in (ChangeStatus.EXTRACTED.value, ChangeStatus.COMPARISON_PENDING.value):
                # Run ComparisonAgent
                self.comparison_agent.process_change_record(change.id)
                session.refresh(change)
                gap = session.query(GapRecord).filter_by(change_record_id=change.id).first()

            res.artifacts.gap_record_ids.append(gap.id)

            # 4. G2 Gate Check (Ambiguity / Low Confidence / Inconsistency)
            if change.status == ChangeStatus.HELD_FOR_G2.value or gap.status == GapStatus.REVIEW_REQUIRED.value:
                self._record_audit(
                    session=session,
                    entity_id=gap.id,
                    entity_type="GapRecord",
                    previous_status=None,
                    new_status=GapStatus.REVIEW_REQUIRED.value,
                    action="stage_held",
                    stage=PipelineStage.COMPARISON.value,
                    reason="Comparison result is ambiguous or below confidence threshold; held for G2 review.",
                )
                res.status = PipelineStatus.HELD
                res.current_stage = PipelineStage.COMPARISON
                res.blocked_stage = PipelineStage.COMPARISON
                res.held_gate = HumanGate.G2
                res.reason = "Comparison output requires G2 human review."
                return res

            # 5. G3 Gate Check (Confirmed No-Match Finding)
            is_no_match = gap.comparison_result == ComparisonResult.NO_MATCH.value or change.status == ChangeStatus.HELD_FOR_G3.value
            if is_no_match:
                # If committee resolution has not yet been documented, stop at G3 gate
                if not gap.reviewer_resolution:
                    self._record_audit(
                        session=session,
                        entity_id=gap.id,
                        entity_type="GapRecord",
                        previous_status=None,
                        new_status=gap.status,
                        action="stage_held",
                        stage=PipelineStage.COMPARISON.value,
                        reason="No institutional protocol match found (G3); awaiting clinical committee confirmation.",
                    )
                    res.status = PipelineStatus.HELD
                    res.current_stage = PipelineStage.COMPARISON
                    res.blocked_stage = PipelineStage.COMPARISON
                    res.held_gate = HumanGate.G3
                    res.reason = "Explicit no-match finding requires G3 clinical committee confirmation."
                    return res
                # If reviewer_resolution is present, committee has confirmed the no-match path
                logger.info("G3 confirmed no-match finding confirmed by committee for GapRecord %s.", gap.id)

            # 6. Check if comparison resulted in no_gap
            if change.status == ChangeStatus.NO_GAP.value:
                res.status = PipelineStatus.COMPLETED
                res.current_stage = PipelineStage.COMPARISON
                res.completed_stages.append(PipelineStage.COMPARISON)
                res.reason = "Protocol comparison confirmed no material gap; no governance action needed."
                return res

            res.completed_stages.append(PipelineStage.COMPARISON)

            # 7. Impact Assessment Stage
            impact = session.query(ImpactRecord).filter_by(gap_record_id=gap.id).first()
            if impact is None:
                # Run ImpactAgent
                g3_urgency = (
                    self.default_g3_urgency_input
                    if is_no_match and (not gap.difference_type or gap.difference_type == DifferenceType.NO_MATCH.value)
                    else None
                )
                self.impact_agent.process_gap_record(
                    gap_record_id=gap.id,
                    urgency_input=g3_urgency,
                    breadth_input=self.default_breadth_input,
                )
                impact = session.query(ImpactRecord).filter_by(gap_record_id=gap.id).first()

            res.artifacts.impact_record_ids.append(impact.id)
            res.completed_stages.append(PipelineStage.IMPACT)

            if impact.status == ImpactStatus.INCOMPLETE.value:
                res.status = PipelineStatus.HELD
                res.current_stage = PipelineStage.IMPACT
                res.blocked_stage = PipelineStage.IMPACT
                res.reason = "Impact assessment incomplete (unmapped clinical rules or missing dimensions)."
                return res

            # 8. Brief Generation Stage
            brief = session.query(ChangeBrief).filter_by(impact_record_id=impact.id).first()
            if brief is None:
                # Run BriefingAgent
                self.briefing_agent.process_impact_record(impact.id)
                brief = session.query(ChangeBrief).filter_by(impact_record_id=impact.id).first()

            res.artifacts.change_brief_ids.append(brief.id)
            res.completed_stages.append(PipelineStage.BRIEFING)

            # 9. Governance Assignment Stage
            assignments = session.query(ReviewAssignment).filter_by(change_brief_id=brief.id).all()
            if not assignments:
                # Assign default reviewer if configured
                if self.default_reviewer_id:
                    assignment = self.governance_agent.assign_reviewer(
                        change_brief_id=brief.id,
                        reviewer_id=self.default_reviewer_id,
                        reviewer_role=self.default_reviewer_role,
                        actor="pipeline_orchestrator",
                    )
                    res.artifacts.review_assignment_ids.append(assignment.id)
            else:
                res.artifacts.review_assignment_ids.extend([a.id for a in assignments])

            brief = session.get(ChangeBrief, brief.id)
            res.completed_stages.append(PipelineStage.GOVERNANCE)

            # 10. G4 Human Gate Check (Unconditional Human Governance Decision)
            # The orchestrator MUST NEVER auto-approve, auto-reject, or auto-close!
            if brief.status in (BriefStatus.DRAFT.value, BriefStatus.ASSIGNED.value, BriefStatus.IN_REVIEW.value):
                res.status = PipelineStatus.HELD
                res.current_stage = PipelineStage.GOVERNANCE
                res.blocked_stage = PipelineStage.GOVERNANCE
                res.held_gate = HumanGate.G4
                res.reason = f"ChangeBrief '{brief.id}' is in status '{brief.status}' and requires an explicit human governance decision."
                return res

            if brief.status == BriefStatus.DEFERRED.value:
                res.status = PipelineStatus.HELD
                res.current_stage = PipelineStage.GOVERNANCE
                res.blocked_stage = PipelineStage.GOVERNANCE
                res.held_gate = HumanGate.G4
                res.reason = f"ChangeBrief '{brief.id}' is deferred and remains open awaiting follow-up review."
                return res

            if brief.status == BriefStatus.DECIDED.value:
                # Explicit decision has been made; awaiting closure
                res.status = PipelineStatus.IN_PROGRESS
                res.current_stage = PipelineStage.CLOSURE
                res.reason = f"ChangeBrief '{brief.id}' has been decided by a human reviewer; ready for final closure."
                return res

            if brief.status == BriefStatus.CLOSED.value:
                res.status = PipelineStatus.COMPLETED
                res.current_stage = PipelineStage.CLOSURE
                res.completed_stages.append(PipelineStage.CLOSURE)
                res.reason = f"ChangeBrief '{brief.id}' review cycle is fully completed and closed."
                return res

            return res

    # ==========================================================================
    # ENTRY POINT 4: PROCESS BRIEF (GOVERNANCE LIFECYCLE)
    # ==========================================================================

    def process_brief(
        self,
        change_brief_id: str,
        close_if_decided: bool = False,
        actor: Optional[str] = None,
        closure_note: Optional[str] = None,
    ) -> PipelineResult:
        """Inspect and progress a ChangeBrief through the governance stage.

        Args:
            change_brief_id: UUID of the target ChangeBrief.
            close_if_decided: If True and brief has already received an explicit human decision,
                             authorizes closure progression.
            actor: Identity of actor performing the closure.
            closure_note: Optional note documenting closure authorization.

        Returns:
            PipelineResult tracking governance stage status.
        """
        with self.session_factory() as session:
            brief = session.get(ChangeBrief, change_brief_id)
            if not brief:
                raise ValueError(f"ChangeBrief with ID '{change_brief_id}' not found.")

            res = PipelineResult(
                current_stage=PipelineStage.GOVERNANCE,
            )
            res.artifacts.change_brief_ids.append(brief.id)

            # Ensure reviewer assignment if still draft
            if brief.status == BriefStatus.DRAFT.value:
                if self.default_reviewer_id:
                    self.governance_agent.assign_reviewer(
                        change_brief_id=brief.id,
                        reviewer_id=self.default_reviewer_id,
                        reviewer_role=self.default_reviewer_role,
                        actor=actor or "pipeline_orchestrator",
                    )
                    session.refresh(brief)

            # Check G4 Gate
            if brief.status in (BriefStatus.ASSIGNED.value, BriefStatus.IN_REVIEW.value):
                res.status = PipelineStatus.HELD
                res.blocked_stage = PipelineStage.GOVERNANCE
                res.held_gate = HumanGate.G4
                res.reason = "ChangeBrief is awaiting an explicit human governance decision (approve/reject/defer)."
                return res

            if brief.status == BriefStatus.DEFERRED.value:
                res.status = PipelineStatus.HELD
                res.blocked_stage = PipelineStage.GOVERNANCE
                res.held_gate = HumanGate.G4
                res.reason = "ChangeBrief is deferred and remains open."
                return res

            if brief.status == BriefStatus.DECIDED.value:
                if close_if_decided:
                    # Authorize administrative closure following explicit human decision
                    self.governance_agent.close_after_decision(
                        change_brief_id=brief.id,
                        actor=actor or "governance_coordinator",
                        closure_note=closure_note,
                    )
                    session.refresh(brief)
                    res.status = PipelineStatus.COMPLETED
                    res.current_stage = PipelineStage.CLOSURE
                    res.completed_stages.extend([PipelineStage.GOVERNANCE, PipelineStage.CLOSURE])
                    res.reason = "ChangeBrief successfully closed following explicit human decision."
                    return res
                else:
                    res.status = PipelineStatus.IN_PROGRESS
                    res.current_stage = PipelineStage.CLOSURE
                    res.reason = "ChangeBrief is decided; awaiting closure authorization."
                    return res

            if brief.status == BriefStatus.CLOSED.value:
                res.status = PipelineStatus.COMPLETED
                res.current_stage = PipelineStage.CLOSURE
                res.completed_stages.extend([PipelineStage.GOVERNANCE, PipelineStage.CLOSURE])
                res.reason = "ChangeBrief is already closed."
                return res

            return res

    # ==========================================================================
    # ENTRY POINT 5: RESUME HELD WORK (HUMAN GATE RESOLUTION)
    # ==========================================================================

    def resume_held_work(
        self,
        entity_type: str,
        entity_id: str,
        resolution_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Resume execution from a persisted checkpoint after human gate review.

        Args:
            entity_type: Entity type to resume ('ChangeRecord', 'IngestedDocument', 'ChangeBrief').
            entity_id: UUID identifier of the held entity.
            resolution_data: Human review outcome, overrides, or decisions.

        Returns:
            PipelineResult resuming downstream execution.
        """
        entity_norm = entity_type.lower().replace("_", "")
        resolution = resolution_data or {}
        actor = resolution.get("reviewer") or "human_reviewer"

        # ----------------------------------------------------------------------
        # CASE A: RESUME HELD CHANGE RECORD (G1 / G2 / G3)
        # ----------------------------------------------------------------------
        if entity_norm in ("changerecord", "change"):
            with self.session_factory() as session:
                change = session.get(ChangeRecord, entity_id)
                if not change:
                    raise ValueError(f"ChangeRecord with ID '{entity_id}' not found.")

                old_status = change.status

                # G1 Gate Resolution
                if change.status == ChangeStatus.HELD_FOR_G1.value:
                    action = str(resolution.get("action", "proceed")).lower()
                    rationale = resolution.get("rationale") or "Human clinician approved extracted recommendation."

                    if action in ("proceed", "approve", "confirm", "override"):
                        if resolution.get("corrected_text"):
                            change.verbatim_text = resolution["corrected_text"]
                        validate_transition(change.status, ChangeStatus.COMPARISON_PENDING.value, CHANGE_TRANSITIONS, "ChangeRecord")
                        change.status = ChangeStatus.COMPARISON_PENDING.value
                        session.commit()

                        self._record_audit(
                            session=session,
                            entity_id=change.id,
                            entity_type="ChangeRecord",
                            previous_status=old_status,
                            new_status=change.status,
                            action="g1_resolved",
                            stage=PipelineStage.EXTRACTION.value,
                            reason=rationale,
                            actor=actor,
                        )

                        # Check if parent document can transition from held to processing
                        doc = session.get(IngestedDocument, change.ingested_document_id)
                        if doc and doc.status == DocumentStatus.HELD.value:
                            remaining_g1 = [
                                c for c in doc.change_records
                                if c.id != change.id and c.status == ChangeStatus.HELD_FOR_G1.value
                            ]
                            if not remaining_g1:
                                doc.status = DocumentStatus.PROCESSING.value
                                session.commit()

                        # Resume downstream from comparison boundary
                        return self.process_change(change.id)

                    elif action in ("reject", "no_gap"):
                        validate_transition(change.status, ChangeStatus.NO_GAP.value, CHANGE_TRANSITIONS, "ChangeRecord")
                        change.status = ChangeStatus.NO_GAP.value
                        session.commit()

                        self._record_audit(
                            session=session,
                            entity_id=change.id,
                            entity_type="ChangeRecord",
                            previous_status=old_status,
                            new_status=change.status,
                            action="g1_rejected",
                            stage=PipelineStage.EXTRACTION.value,
                            reason=rationale,
                            actor=actor,
                        )
                        return PipelineResult(
                            status=PipelineStatus.COMPLETED,
                            current_stage=PipelineStage.EXTRACTION,
                            reason="G1 extraction rejected by human reviewer; marked as no_gap.",
                        )

                # G2 Gate Resolution
                elif change.status == ChangeStatus.HELD_FOR_G2.value:
                    action = str(resolution.get("action", "confirm_gap")).lower()
                    rationale = resolution.get("rationale") or "Human clinician verified protocol gap."

                    gap = change.gap_record
                    if not gap:
                        raise ValueError(f"No GapRecord found for ChangeRecord '{change.id}' held for G2.")

                    if action in ("confirm_gap", "gap_confirmed", "proceed", "approve"):
                        validate_transition(change.status, ChangeStatus.GAP_CONFIRMED.value, CHANGE_TRANSITIONS, "ChangeRecord")
                        change.status = ChangeStatus.GAP_CONFIRMED.value
                        gap.status = GapStatus.RESOLVED.value
                        gap.reviewer_resolution = rationale
                        if resolution.get("difference_type"):
                            gap.difference_type = resolution["difference_type"]
                        session.commit()

                        self._record_audit(
                            session=session,
                            entity_id=change.id,
                            entity_type="ChangeRecord",
                            previous_status=old_status,
                            new_status=change.status,
                            action="g2_resolved",
                            stage=PipelineStage.COMPARISON.value,
                            reason=rationale,
                            actor=actor,
                        )
                        # Resume downstream to impact and briefing
                        return self.process_change(change.id)

                    elif action in ("no_gap", "reject"):
                        validate_transition(change.status, ChangeStatus.NO_GAP.value, CHANGE_TRANSITIONS, "ChangeRecord")
                        change.status = ChangeStatus.NO_GAP.value
                        gap.status = GapStatus.RESOLVED.value
                        gap.reviewer_resolution = rationale
                        session.commit()

                        self._record_audit(
                            session=session,
                            entity_id=change.id,
                            entity_type="ChangeRecord",
                            previous_status=old_status,
                            new_status=change.status,
                            action="g2_rejected",
                            stage=PipelineStage.COMPARISON.value,
                            reason=rationale,
                            actor=actor,
                        )
                        return PipelineResult(
                            status=PipelineStatus.COMPLETED,
                            current_stage=PipelineStage.COMPARISON,
                            reason="G2 comparison resolved as no material gap.",
                        )

                # G3 Gate Resolution (Confirmed No-Match Committee Review)
                elif change.status == ChangeStatus.HELD_FOR_G3.value:
                    gap = change.gap_record
                    if not gap:
                        raise ValueError(f"No GapRecord found for ChangeRecord '{change.id}' held for G3.")

                    rationale = resolution.get("rationale") or "Clinical committee confirmed no-match finding."
                    gap.reviewer_resolution = rationale
                    session.commit()

                    self._record_audit(
                        session=session,
                        entity_id=gap.id,
                        entity_type="GapRecord",
                        previous_status=gap.status,
                        new_status=gap.status,
                        action="g3_confirmed",
                        stage=PipelineStage.COMPARISON.value,
                        reason=rationale,
                        actor=actor,
                    )
                    # Resume downstream: G3 confirmed no-match proceeds to impact scoring & briefing
                    return self.process_change(change.id)

                else:
                    # Change record is not held at a known gate; continue normal execution
                    return self.process_change(change.id)

        # ----------------------------------------------------------------------
        # CASE B: RESUME HELD INGESTED DOCUMENT
        # ----------------------------------------------------------------------
        elif entity_norm in ("ingesteddocument", "document", "doc"):
            with self.session_factory() as session:
                doc = session.get(IngestedDocument, entity_id)
                if not doc:
                    raise ValueError(f"IngestedDocument with ID '{entity_id}' not found.")

                if doc.status == DocumentStatus.HELD.value:
                    # Check if all child changes are resolved
                    held_children = [c for c in doc.change_records if c.status == ChangeStatus.HELD_FOR_G1.value]
                    if held_children:
                        return PipelineResult(
                            document_id=doc.id,
                            status=PipelineStatus.HELD,
                            blocked_stage=PipelineStage.EXTRACTION,
                            held_gate=HumanGate.G1,
                            reason=f"Cannot resume document '{doc.id}'; {len(held_children)} ChangeRecord(s) still require G1 resolution.",
                        )
                    # If all children resolved, transition document to complete and resume
                    doc.status = DocumentStatus.COMPLETE.value
                    session.commit()

                return self.process_ingested_document(doc.id)

        # ----------------------------------------------------------------------
        # CASE C: RESUME HELD CHANGE BRIEF (DEFERRED / DECIDED)
        # ----------------------------------------------------------------------
        elif entity_norm in ("changebrief", "brief"):
            with self.session_factory() as session:
                brief = session.get(ChangeBrief, entity_id)
                if not brief:
                    raise ValueError(f"ChangeBrief with ID '{entity_id}' not found.")

                # If brief was deferred, resume review only (do NOT restart extraction/comparison)
                if brief.status == BriefStatus.DEFERRED.value:
                    reviewer_id = resolution.get("reviewer_id") or self.default_reviewer_id
                    self.governance_agent.start_review(
                        change_brief_id=brief.id,
                        reviewer_id=reviewer_id,
                    )
                    session.refresh(brief)
                    return PipelineResult(
                        status=PipelineStatus.HELD,
                        current_stage=PipelineStage.GOVERNANCE,
                        blocked_stage=PipelineStage.GOVERNANCE,
                        held_gate=HumanGate.G4,
                        reason="ChangeBrief re-entered review after deferral; awaiting explicit human decision.",
                    )

                # If brief was decided, allow closure
                if brief.status == BriefStatus.DECIDED.value:
                    return self.process_brief(brief.id, close_if_decided=True, actor=actor)

                return self.process_brief(brief.id)

        else:
            raise ValueError(f"Unsupported entity_type for resume: '{entity_type}'")

    # ==========================================================================
    # ENTRY POINT 6: BATCH PROCESSING OF PENDING DOCUMENTS
    # ==========================================================================

    def process_pending_documents(
        self,
        max_docs: Optional[int] = None,
    ) -> BatchPipelineResult:
        """Batch process all eligible pending documents with strict failure isolation.

        Eligible documents include statuses: 'discovered', 'parsed', 'processing',
        or 'failed' where retry_count < max_retries.

        Args:
            max_docs: Optional maximum number of documents to process in this run.

        Returns:
            BatchPipelineResult aggregating individual outcomes.
        """
        eligible_statuses = [
            DocumentStatus.DISCOVERED.value,
            DocumentStatus.PARSED.value,
            DocumentStatus.PROCESSING.value,
        ]

        with self.session_factory() as session:
            stmt = (
                select(IngestedDocument.id)
                .where(
                    (IngestedDocument.status.in_(eligible_statuses))
                    | (
                        (IngestedDocument.status == DocumentStatus.FAILED.value)
                        & (IngestedDocument.retry_count < self.max_retries)
                    )
                )
                .order_by(IngestedDocument.created_at.asc())
            )
            if max_docs is not None and max_docs > 0:
                stmt = stmt.limit(max_docs)
            eligible_ids = session.scalars(stmt).all()

        batch_result = BatchPipelineResult(total_documents=len(eligible_ids))
        logger.info("Batch processing started for %d pending document(s).", len(eligible_ids))

        for doc_id in eligible_ids:
            try:
                res = self.process_ingested_document(doc_id)
                batch_result.results.append(res)

                if res.is_completed:
                    batch_result.completed_count += 1
                elif res.is_held:
                    batch_result.held_count += 1
                elif res.is_failed:
                    batch_result.failed_count += 1
                elif res.status == PipelineStatus.SKIPPED:
                    batch_result.skipped_count += 1

            except Exception as e:
                # Failure isolation: one failing document does not abort the batch
                logger.error("Batch document execution failed for document %s: %s", doc_id, e)
                batch_result.failed_count += 1
                batch_result.results.append(
                    PipelineResult(
                        document_id=doc_id,
                        status=PipelineStatus.FAILED,
                        errors=[str(e)],
                        reason=f"Batch execution exception: {e}",
                    )
                )

        logger.info(
            "Batch processing completed: %d total, %d completed, %d held, %d failed.",
            batch_result.total_documents,
            batch_result.completed_count,
            batch_result.held_count,
            batch_result.failed_count,
        )
        return batch_result
