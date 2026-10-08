"""Deterministic Briefing Agent for assembling, validating, and persisting ChangeBriefs.

Assembles data from IngestedDocument, ChangeRecord, GapRecord, and ImpactRecord,
populates the seven required brief sections, invokes BriefRenderer for Jinja2
rendering of HTML, Markdown, and JSON, and persists ChangeBrief entities.

IMPORTANT: Makes ZERO LLM or external API calls.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import ChangeBrief, ChangeRecord, GapRecord, ImpactRecord, IngestedDocument
from app.schemas.briefs import (
    BriefCompletenessError,
    BriefStatus,
    ComparisonSectionPayload,
    ImpactSectionPayload,
    ProposedActionItem,
    ProtocolSectionPayload,
    RecommendationSectionPayload,
    SourceExcerptSectionPayload,
    SourceMetadataPayload,
    StructuredBriefPayload,
    WorkflowSectionPayload,
    assert_brief_completeness,
    validate_brief_completeness,
)
from app.schemas.gaps import ComparisonResult, DifferenceType
from app.schemas.impact import ImpactStatus
from app.services.brief_renderer import BriefRenderer, RenderedBriefResult
from app.services.config_service import AppConfig, load_config
from app.services.protocol_lookup import find_protocol_section

logger = logging.getLogger("ckea.agents.briefing_agent")


class BriefingError(Exception):
    """Base exception for BriefingAgent errors."""
    pass


class RecordNotFoundError(BriefingError):
    """Raised when an expected database record is not found."""
    pass


class BriefingAgent:
    """Deterministic, auditable briefing agent for clinical change briefs."""

    def __init__(
        self,
        session_factory: Optional[sessionmaker] = None,
        renderer: Optional[BriefRenderer] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_config()
        self.session_factory = session_factory or get_session_factory(self.config.database_url)
        self.renderer = renderer or BriefRenderer(
            templates_dir=self.config.templates_dir,
            output_dir=self.config.output_dir,
            config=self.config,
        )

    def process_impact_record(
        self,
        impact_record_id: str,
        exact_protocol_text: Optional[str] = None,
        specific_difference: Optional[str] = None,
        affected_workflows: Optional[List[str]] = None,
        workflow_summary: Optional[str] = None,
        is_workflow_available: bool = True,
        unavailability_reason: Optional[str] = None,
        write_files: bool = True,
        validate: bool = True,
    ) -> ChangeBrief:
        """Assemble, render, and persist a ChangeBrief from an existing ImpactRecord.

        Args:
            impact_record_id: Primary key UUID of the parent ImpactRecord.
            exact_protocol_text: Optional explicit quotation of matched protocol text.
            specific_difference: Optional explicit specific difference statement.
            affected_workflows: Optional list of affected workflow names.
            workflow_summary: Optional workflow summary statement.
            is_workflow_available: Whether workflow information is available (default: True).
            unavailability_reason: Optional explanation if workflows are unavailable.
            write_files: Whether to persist rendered files to disk.
            validate: Whether to enforce strict brief completeness validation.

        Returns:
            Persisted ChangeBrief entity in 'draft' status.

        Raises:
            RecordNotFoundError: If ImpactRecord or any required parent entity is missing.
            BriefCompletenessError: If validation fails and validate=True.
        """
        with self.session_factory() as session:
            # 1. Load ImpactRecord and verify hierarchy
            impact = session.query(ImpactRecord).filter_by(id=impact_record_id).first()
            if not impact:
                raise RecordNotFoundError(f"ImpactRecord not found with ID: {impact_record_id}")

            gap = session.query(GapRecord).filter_by(id=impact.gap_record_id).first()
            if not gap:
                raise RecordNotFoundError(f"GapRecord not found with ID: {impact.gap_record_id}")

            change = session.query(ChangeRecord).filter_by(id=gap.change_record_id).first()
            if not change:
                raise RecordNotFoundError(f"ChangeRecord not found with ID: {gap.change_record_id}")

            doc = session.query(IngestedDocument).filter_by(id=change.ingested_document_id).first()
            if not doc:
                raise RecordNotFoundError(f"IngestedDocument not found with ID: {change.ingested_document_id}")

            # 2. Check for existing ChangeBrief (idempotency)
            existing_brief = session.query(ChangeBrief).filter_by(impact_record_id=impact.id).first()
            brief_id = existing_brief.id if existing_brief else str(uuid.uuid4())

            # 3. Assemble Section 1: What changed
            what_changed = RecommendationSectionPayload(
                recommendation_text=change.verbatim_text,
                source_identifier=doc.source_identifier,
                page=change.page if change.page is not None else 1,
                section=change.section or "Clinical Recommendation",
                recommendation_type=change.recommendation_type,
                target_population=change.target_population,
                intervention=change.intervention,
                evidence_grade=change.evidence_grade,
                extraction_confidence=change.confidence,
            )

            # 4. Assemble Section 2: What our protocol currently says
            is_match = bool(gap.is_match and gap.comparison_result != ComparisonResult.NO_MATCH.value)
            resolved_protocol_text = exact_protocol_text or gap.exact_protocol_text
            section_id = gap.matched_section_id or (
                gap.candidate_protocol_section_ids[0].split("__")[-1]
                if gap.candidate_protocol_section_ids else None
            )
            section_heading = gap.matched_section_heading

            if is_match and (not resolved_protocol_text or not section_heading):
                sec = self._resolve_protocol_section(
                    protocol_id=gap.matched_protocol_id,
                    protocol_version=gap.matched_protocol_version,
                    section_id=section_id,
                )
                if sec is not None:
                    resolved_protocol_text = resolved_protocol_text or sec.section_text
                    section_heading = section_heading or sec.section_heading
                if not resolved_protocol_text:
                    raise BriefingError(
                        f"Exact protocol text for {gap.matched_protocol_id} version "
                        f"{gap.matched_protocol_version} section {section_id} could not be resolved. "
                        "The brief was not generated rather than show invented or substituted protocol content."
                    )

            current_protocol = ProtocolSectionPayload(
                is_match=is_match,
                protocol_id=gap.matched_protocol_id if is_match else None,
                protocol_version=gap.matched_protocol_version if is_match else None,
                section_id=section_id if is_match else None,
                section_heading=(section_heading or f"Section {section_id}") if is_match else None,
                exact_protocol_text=resolved_protocol_text if is_match else None,
                no_match_statement=(
                    "No matching protocol section was identified in the current institutional protocol library."
                    if not is_match else None
                ),
            )

            # 5. Assemble Section 3: Specific difference
            resolved_difference = specific_difference or gap.specific_difference
            if not resolved_difference:
                if is_match:
                    diff_type_readable = (gap.difference_type or "difference").replace("_", " ")
                    resolved_difference = (
                        f"Identified {diff_type_readable} between external recommendation and "
                        f"institutional protocol ({gap.matched_protocol_id})."
                    )
                else:
                    resolved_difference = (
                        "No matching institutional protocol section exists; recommendation represents "
                        "an unaddressed clinical area."
                    )

            diff_payload = ComparisonSectionPayload(
                specific_difference=resolved_difference,
                difference_type=gap.difference_type or (DifferenceType.NO_MATCH.value if not is_match else "unspecified"),
                comparison_result=gap.comparison_result,
                comparison_confidence=gap.comparison_confidence,
                candidate_section_ids=gap.candidate_protocol_section_ids or [],
            )

            # 6. Assemble Section 4: Impact assessment
            is_impact_complete = bool(
                impact.status in (ImpactStatus.CALCULATED.value, ImpactStatus.ROUTED.value, "completed")
                and impact.total_score is not None
                and impact.tier is not None
                and impact.clinical_urgency is not None
                and impact.evidence_strength is not None
                and impact.pathway_breadth is not None
            )

            rule_ids_dict: Dict[str, str] = {}
            if isinstance(impact.rule_ids, dict):
                rule_ids_dict = impact.rule_ids
            elif isinstance(impact.rule_ids, list):
                for rid in impact.rule_ids:
                    if str(rid).startswith("URGENCY"):
                        rule_ids_dict["clinical_urgency"] = str(rid)
                    elif str(rid).startswith("EVIDENCE"):
                        rule_ids_dict["evidence_strength"] = str(rid)
                    elif str(rid).startswith("BREADTH"):
                        rule_ids_dict["pathway_breadth"] = str(rid)

            impact_payload = ImpactSectionPayload(
                status=impact.status,
                is_complete=is_impact_complete,
                clinical_urgency=impact.clinical_urgency if is_impact_complete else impact.clinical_urgency,
                urgency_basis=impact.urgency_basis,
                evidence_strength=impact.evidence_strength if is_impact_complete else impact.evidence_strength,
                evidence_basis=impact.evidence_basis,
                pathway_breadth=impact.pathway_breadth if is_impact_complete else impact.pathway_breadth,
                breadth_basis=impact.breadth_basis,
                total_score=impact.total_score if is_impact_complete else None,
                tier=impact.tier if is_impact_complete else None,
                routing_target=impact.routing_target if is_impact_complete else "Governance Review Committee",
                sla_deadline=impact.sla_deadline.isoformat() if (is_impact_complete and impact.sla_deadline) else None,
                rule_ids=rule_ids_dict,
                scoring_yaml_version=impact.scoring_yaml_version,
                incomplete_reason=(
                    "Impact assessment is incomplete due to missing or unmapped dimension scores. "
                    "No tier or SLA deadline assigned."
                    if not is_impact_complete else None
                ),
            )

            # 7. Assemble Section 5: Affected workflows
            if not is_workflow_available:
                workflow_payload = WorkflowSectionPayload(
                    affected_workflows=[],
                    workflow_summary=None,
                    is_available=False,
                    unavailability_reason=unavailability_reason or "Workflow information unavailable in source metadata.",
                )
            else:
                resolved_wf_list = affected_workflows
                resolved_wf_summary = workflow_summary
                if resolved_wf_list is None:
                    # Deterministically construct standard workflow items from change record metadata
                    resolved_wf_list = [
                        f"{change.target_population} care pathway",
                        f"{change.recommendation_type.replace('_', ' ').title()} ordering workflow",
                    ]
                workflow_payload = WorkflowSectionPayload(
                    affected_workflows=resolved_wf_list,
                    workflow_summary=resolved_wf_summary,
                    is_available=True,
                    unavailability_reason=None,
                )

            # 8. Assemble Section 6: Proposed review actions (The approved three-action structure)
            proposed_actions = [
                ProposedActionItem(
                    step=1,
                    action="Confirm applicability",
                    description="Confirm clinical applicability of the external recommendation to the target patient population and institutional setting.",
                ),
                ProposedActionItem(
                    step=2,
                    action="Update protocol if adopted",
                    description="Update institutional clinical protocol and corresponding EHR clinical decision support rules / order sets if adopted.",
                ),
                ProposedActionItem(
                    step=3,
                    action="Record rationale either way",
                    description="Document clinical governance rationale and formal review decision (adopt, defer, or reject) in the immutable audit log.",
                ),
            ]

            # 9. Assemble Section 7: Source excerpt
            source_excerpt_text = change.source_excerpt or change.verbatim_text
            source_excerpt = SourceExcerptSectionPayload(
                source_excerpt=source_excerpt_text,
                source_identifier=doc.source_identifier,
                page=change.page if change.page is not None else 1,
                section=change.section or "Clinical Recommendation",
            )

            # 10. Assemble top-level StructuredBriefPayload
            source_meta = SourceMetadataPayload(
                source_identifier=doc.source_identifier,
                source_path=doc.source_path,
                source_version=doc.source_version,
                document_version=doc.document_version,
                sha256_hash=doc.sha256_hash,
            )

            payload = StructuredBriefPayload(
                brief_id=brief_id,
                impact_record_id=impact.id,
                gap_record_id=gap.id,
                change_record_id=change.id,
                schema_version="1.0",
                generated_at=datetime.now(timezone.utc).isoformat(),
                source_metadata=source_meta,
                what_changed=what_changed,
                current_protocol=current_protocol,
                specific_difference=diff_payload,
                impact_assessment=impact_payload,
                affected_workflows=workflow_payload,
                proposed_actions=proposed_actions,
                source_excerpt=source_excerpt,
                summary=self._concise_summary(doc, change, gap, impact, current_protocol, resolved_difference),
                change_category=change.change_category,
                priority_score=impact.priority_score,
                affected_departments=list(impact.affected_departments or []),
                affected_pathways=list(impact.affected_pathways or []),
            )

            # 11. Validate brief completeness
            if validate:
                assert_brief_completeness(payload)

            # 12. Render HTML, Markdown, and JSON
            render_result = self.renderer.render_brief(payload, write_files=write_files, validate=False)

            # 13. Persist or idempotently update ChangeBrief entity
            if existing_brief:
                existing_brief.status = BriefStatus.DRAFT.value
                existing_brief.rendered_file_path = str(render_result.primary_file_path)
                existing_brief.rendered_file_hash = render_result.rendered_file_hash
                existing_brief.structured_payload = payload.model_dump()
                brief_entity = existing_brief
            else:
                brief_entity = ChangeBrief(
                    id=brief_id,
                    impact_record_id=impact.id,
                    status=BriefStatus.DRAFT.value,
                    rendered_file_path=str(render_result.primary_file_path),
                    rendered_file_hash=render_result.rendered_file_hash,
                    structured_payload=payload.model_dump(),
                    schema_version="1.0",
                )
                session.add(brief_entity)

            session.commit()
            session.refresh(brief_entity)
            session.expunge(brief_entity)
            return brief_entity

    @staticmethod
    def _concise_summary(doc, change, gap, impact, current_protocol, difference) -> str:
        """At most five lines: what changed, where, what our protocol says, why it matters."""
        meta = doc.doc_metadata or {}
        source = meta.get("title") or doc.source_identifier
        version = meta.get("publisher_version") or doc.source_version
        lines = [
            f"What changed: {change.verbatim_text}",
            f"Source: {source} version {version}" + (f" ({meta['published_date']})" if meta.get("published_date") else ""),
        ]
        if current_protocol.is_match:
            lines.append(f"Our protocol ({current_protocol.protocol_id} {current_protocol.protocol_version}): "
                         f"{current_protocol.exact_protocol_text}")
        else:
            lines.append("Our protocol: no matching section.")
        lines.append(f"Difference: {difference}")
        if impact.tier:
            lines.append(f"Why it matters: {impact.tier} impact ({impact.total_score}/15)"
                         + (f", {change.change_category.replace('_', ' ')}" if change.change_category else "") + ".")
        return "\n".join(lines[:5])

    def build_clinician_answer(self, query, outcome, ranked_findings, sources_checked, steps, taxonomy):
        """Clinician query mode: deterministic verdict and answer brief from verified comparison results."""
        from app.services.clinician_answer import build_answer

        return build_answer(query, outcome, ranked_findings, sources_checked, steps, taxonomy)

    def _resolve_protocol_section(
        self,
        protocol_id: Optional[str],
        protocol_version: Optional[str],
        section_id: Optional[str],
    ):
        """Exact protocol section (same id, version and section) or None. Never substitutes."""
        found = find_protocol_section(self.config.protocol_dir, protocol_id, protocol_version, section_id)
        return found[1] if found else None
