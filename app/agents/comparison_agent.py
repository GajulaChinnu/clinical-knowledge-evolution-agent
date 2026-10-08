"""CKEA Comparison Agent for evaluating extracted recommendations against institutional protocols."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import ChangeRecord, GapRecord, GuidanceChange, GuidanceStatement, IngestedDocument
from app.schemas.changes import ChangeStatus
from app.schemas.comparison import ComparisonResponse
from app.schemas.gaps import ComparisonResult, DifferenceType, GapStatus
from app.schemas.protocol import CandidateProtocolSection
from app.schemas.transitions import CHANGE_TRANSITIONS, validate_transition
from app.services.config_service import AppConfig, load_config
from app.services.llm_client import SharedLLMClient
from app.services.protocol_index import ProtocolIndexService
from app.schemas.clinician_query import ClinicianQuery
from app.schemas.protocol import ProtocolDocument
from app.schemas.treatment_check import ComparisonOutcome, Finding, PlanComponent, VersionComparison
from app.services.clinical_statements import parse_statement
from app.services.taxonomy import Taxonomy, get_taxonomy
from app.services.treatment_comparison import (
    PLAN_ATTRIBUTES,
    attribute_supported,
    attribute_value,
    GUIDANCE_SOURCE_TYPES,
    StatementRef,
    build_citation,
    compare_protocol,
    contraindication_applicability,
    egfr_condition,
    initiation_score,
    mentions_plan,
    plan_differences,
    to_diffs,
)
from app.services.clinical_statements import attribute_differences

_GAP_EVIDENCE_FIELDS = (
    "matched_section_id",
    "matched_section_heading",
    "exact_protocol_text",
    "specific_difference",
    "comparison_rationale",
    "review_reason",
)

logger = logging.getLogger("ckea.comparison")


@dataclass
class SourceScope:
    """Latest and previous stored versions of one monitored source, for a clinician query."""
    watchlist_id: str
    latest_document_id: Optional[str]
    previous_document_id: Optional[str] = None


class ComparisonAgent:
    """Specialized agent responsible for retrieving matching protocol sections and comparing recommendations."""

    def __init__(
        self,
        session_factory: Optional[sessionmaker[Session]] = None,
        llm_client: Optional[SharedLLMClient] = None,
        protocol_index_service: Optional[ProtocolIndexService] = None,
        config: Optional[AppConfig] = None,
        top_k: int = 3,
    ) -> None:
        self.config = config or load_config()
        self.session_factory = session_factory or get_session_factory()
        self.llm_client = llm_client or SharedLLMClient(config=self.config)
        self._protocol_index_service = protocol_index_service
        self.top_k = top_k

    @property
    def protocol_index_service(self) -> ProtocolIndexService:
        """Chroma protocol index, created on first use (the clinician query path does not need it)."""
        if self._protocol_index_service is None:
            self._protocol_index_service = ProtocolIndexService(config=self.config)
        return self._protocol_index_service

    @protocol_index_service.setter
    def protocol_index_service(self, value: ProtocolIndexService) -> None:
        self._protocol_index_service = value

    # ==========================================================================
    # CLINICIAN QUERY MODE: three-way comparison for a planned treatment
    # ==========================================================================

    def compare_treatment_plan(
        self,
        query: ClinicianQuery,
        scopes: Sequence[SourceScope],
        protocols: Sequence[ProtocolDocument],
        taxonomy: Optional[Taxonomy] = None,
    ) -> ComparisonOutcome:
        """(a) plan vs latest guidance, (b) previous vs latest version, (c) protocol vs latest.

        Deterministic; every finding cites a verbatim GuidanceStatement. Conditions the
        de-identified context cannot settle are reported as check_applicability, never guessed.
        """
        taxonomy = taxonomy or get_taxonomy()
        outcome = ComparisonOutcome()
        if not query.treatment_ids:
            outcome.retrieval = "none"
            outcome.notes.append(
                "The planned treatment is not in the controlled treatment vocabulary, so no grounded "
                "guidance could be matched."
            )
            return outcome

        plan = query.plan.parsed
        with self.session_factory() as session:
            latest_refs: List[StatementRef] = []
            change_refs: List[Tuple[GuidanceChange, Optional[StatementRef], Optional[StatementRef]]] = []
            for scope in scopes:
                if not scope.latest_document_id:
                    continue
                doc = session.get(IngestedDocument, scope.latest_document_id)
                if doc is None:
                    continue
                for stmt in (session.query(GuidanceStatement)
                             .filter_by(ingested_document_id=doc.id).order_by(GuidanceStatement.sequence)):
                    latest_refs.append(StatementRef(stmt, doc, parse_statement(stmt.verbatim_text, taxonomy, stmt.source_type)))
                for change in session.query(GuidanceChange).filter_by(to_document_id=doc.id):
                    change_refs.append((change, self._ref(session, change.from_statement_id, taxonomy),
                                        self._ref(session, change.to_statement_id, taxonomy)))

            findings = self._plan_findings(query, latest_refs, change_refs, taxonomy)
            outcome.findings = findings
            outcome.plan_components = self._plan_components(query, latest_refs, change_refs, taxonomy)
            outcome.version_changes = self._version_changes(query, change_refs, taxonomy)
            outcome.protocol_positions = [
                compare_protocol(p, query, [r for r in latest_refs if r.stmt.source_type in GUIDANCE_SOURCE_TYPES],
                                 [c for c in change_refs if c[0].source_type in GUIDANCE_SOURCE_TYPES], taxonomy)
                for p in protocols
            ]
        return outcome

    @staticmethod
    def _ref(session: Session, statement_id: Optional[str], taxonomy: Taxonomy) -> Optional[StatementRef]:
        if not statement_id:
            return None
        stmt = session.get(GuidanceStatement, statement_id)
        if stmt is None:
            return None
        doc = session.get(IngestedDocument, stmt.ingested_document_id)
        return StatementRef(stmt, doc, parse_statement(stmt.verbatim_text, taxonomy, stmt.source_type))

    def _plan_findings(self, query, latest_refs, change_refs, taxonomy) -> List[Finding]:
        plan = query.plan.parsed
        findings: List[Finding] = []
        withdrawn_from: Dict[str, StatementRef] = {
            new.stmt.id: old for change, old, new in change_refs
            if change.change_category == "withdrawn" and old is not None and new is not None
        }
        recommendation_findings: List[Tuple[Finding, StatementRef, float]] = []

        for ref in latest_refs:
            parsed, stmt = ref.parsed, ref.stmt
            direct, via_class = mentions_plan(parsed, query, taxonomy)
            if not (direct or via_class):
                continue
            meta = dict(treatments=parsed.treatments, departments=list(stmt.departments or []),
                        pathways=list(stmt.pathways or []))
            citation = build_citation(stmt, ref.doc)

            if parsed.statement_type == "contraindication":
                relation, basis = contraindication_applicability(parsed, query, taxonomy)
                findings.append(Finding(
                    kind="contraindication", relation=relation, citation=citation,
                    explanation={"applies": "Contraindication applies to this patient.",
                                 "not_applicable": "Contraindication does not apply on the information provided.",
                                 "check_applicability": "Contraindication may apply: confirm before prescribing."}[relation],
                    applicability_basis=basis, **meta,
                ))
            elif parsed.statement_type == "safety_warning":
                findings.append(Finding(kind="safety_warning", relation="informational", citation=citation,
                                        explanation="Safety warning relevant to this treatment.", **meta))
            elif parsed.statement_type == "withdrawal" and direct:
                old = withdrawn_from.get(stmt.id)
                relation = "informational"
                prev_cit = None
                if old is not None:
                    prev_cit = build_citation(old.stmt, old.doc)
                    diffs, compared = plan_differences(plan, old.parsed)
                    if not diffs:
                        relation = "withdrawn_matches_plan"
                findings.append(Finding(
                    kind="withdrawal", relation=relation, citation=citation, previous_citation=prev_cit,
                    change_category="withdrawn",
                    explanation=("The recommendation this plan follows has been withdrawn in the latest version."
                                 if relation == "withdrawn_matches_plan" else "A related recommendation was withdrawn."),
                    **meta,
                ))
            elif parsed.statement_type == "evidence":
                findings.append(Finding(kind="evidence", relation="informational", citation=citation,
                                        explanation="Published evidence about this treatment.", **meta))
            elif parsed.statement_type == "recommendation" and direct:
                condition, basis = egfr_condition(parsed, query)
                diffs, compared = plan_differences(plan, parsed)
                if condition == "not_applicable":
                    relation, explanation = "not_applicable", f"Conditional recommendation not applicable: {basis}."
                elif not compared:
                    relation, explanation = "informational", "Current recommendation for this treatment."
                elif diffs:
                    relation, explanation = "differs_from_plan", "Planned treatment differs from this current recommendation."
                else:
                    relation, explanation = "matches_plan", "Planned treatment matches this current recommendation."
                finding = Finding(kind="supporting_recommendation", relation=relation, citation=citation,
                                  explanation=explanation, differences=diffs, applicability_basis=basis or None, **meta)
                is_guidance = stmt.source_type in GUIDANCE_SOURCE_TYPES
                score = initiation_score(parsed, plan) + (10.0 if is_guidance else 0.0)
                if relation in ("matches_plan", "differs_from_plan") and compared:
                    recommendation_findings.append((finding, ref, score))
                findings.append(finding)

        # Governing recommendation: the best match if the plan matches any current recommendation,
        # otherwise the closest current recommendation it differs from.
        matches = [x for x in recommendation_findings if x[0].relation == "matches_plan"]
        pool = matches or recommendation_findings
        if pool:
            governing = max(pool, key=lambda x: x[2])[0]
            governing.kind = "governing_recommendation"
            for change, old, new in change_refs:
                if new is not None and new.stmt.id == governing.citation.statement_id and old is not None:
                    governing.previous_citation = build_citation(old.stmt, old.doc)
                    governing.change_category = change.change_category
        return findings

    CATEGORY_TO_DIFFERENCE = {
        "dose_change": "dosage_change",
        "threshold_change": "threshold_change",
        "contraindication_added": "contraindication",
        "safety_warning": "safety_warning",
        "withdrawn": "intervention_change",
        "new_recommendation": "intervention_change",
        "revised_recommendation": "intervention_change",
    }

    def record_protocol_gap(self, change_record_id: str, position: Any, item: Any, change_category: str) -> str:
        """Record the deterministic protocol comparison for a query as a GapRecord (idempotent)."""
        with self.session_factory() as session:
            change = session.get(ChangeRecord, change_record_id)
            if change is None:
                raise ValueError(f"ChangeRecord '{change_record_id}' not found.")
            existing = session.query(GapRecord).filter_by(change_record_id=change.id).first()
            if existing is not None:
                return existing.id
            for target in (ChangeStatus.COMPARISON_PENDING.value, ChangeStatus.GAP_CONFIRMED.value):
                validate_transition(change.status, target, CHANGE_TRANSITIONS, "ChangeRecord")
                change.status = target
            diffs = "; ".join(f"{d.kind}: protocol {d.before} vs guidance {d.after}" for d in item.differences)
            gap = GapRecord(
                change_record_id=change.id,
                candidate_protocol_section_ids=[f"{position.protocol_id}__{position.protocol_version}__{item.section_id}"],
                similarity=None,
                comparison_result=ComparisonResult.GAP.value,
                comparison_confidence=1.0,
                difference_type=self.CATEGORY_TO_DIFFERENCE.get(change_category, "intervention_change"),
                matched_protocol_id=position.protocol_id,
                matched_protocol_version=position.protocol_version,
                matched_section_id=item.section_id,
                matched_section_heading=item.section_heading,
                exact_protocol_text=item.section_text,
                specific_difference=item.reason + (f" ({diffs})" if diffs else ""),
                comparison_rationale="Deterministic comparison of verbatim guidance and protocol statements (clinician query).",
                is_match=True,
                status=GapStatus.MATCHED.value,
                schema_version="1.0",
            )
            session.add(gap)
            session.commit()
            return gap.id

    def _plan_components(self, query, latest_refs, change_refs, taxonomy) -> List[PlanComponent]:
        """Assess each part of the plan separately against current guideline/notice recommendations.

        A part is supported if any applicable current recommendation states the same value,
        unsupported if current recommendations state only other values, and not_addressed if none
        states it. For unsupported parts, a superseded version that stated the plan's value is
        recorded (that is what makes the verdict 'guidance updated' rather than 'conflicts').
        """
        plan = query.plan.parsed
        candidates = [
            r for r in latest_refs
            if r.stmt.source_type in GUIDANCE_SOURCE_TYPES and r.parsed.statement_type == "recommendation"
            and mentions_plan(r.parsed, query, taxonomy)[0]
            and egfr_condition(r.parsed, query)[0] != "not_applicable"
        ]
        components: List[PlanComponent] = []
        for attribute in PLAN_ATTRIBUTES:
            plan_value = attribute_value(plan, attribute)
            if plan_value is None:
                continue
            supporters, contradictors = [], []
            for ref in candidates:
                verdict = attribute_supported(plan, ref.parsed, attribute)
                if verdict is True:
                    supporters.append(ref)
                elif verdict is False:
                    contradictors.append(ref)
            status = "supported" if supporters else "unsupported" if contradictors else "not_addressed"
            superseded = None
            if status == "unsupported":
                for change, old, new in change_refs:
                    if old is None or old.stmt.source_type not in GUIDANCE_SOURCE_TYPES:
                        continue
                    if not mentions_plan(old.parsed, query, taxonomy)[0]:
                        continue
                    if attribute_supported(plan, old.parsed, attribute) is True and (
                        new is None or new.stmt.statement_type == "withdrawal"
                        or attribute_supported(plan, new.parsed, attribute) is not True
                    ):
                        superseded = build_citation(old.stmt, old.doc)
                        break
            components.append(PlanComponent(
                attribute=attribute, plan_value=plan_value, status=status,
                supported_by=[r.stmt.id for r in supporters],
                contradicted_by=[build_citation(r.stmt, r.doc) for r in contradictors],
                superseded_support=superseded,
            ))
        return components

    def _version_changes(self, query, change_refs, taxonomy) -> List[VersionComparison]:
        plan = query.plan.parsed
        comparisons: List[VersionComparison] = []
        for change, old, new in change_refs:
            if change.change_category in ("no_practice_change", "unchanged"):
                continue
            refs = [r for r in (old, new) if r is not None]
            if not any(mentions_plan(r.parsed, query, taxonomy)[0] for r in refs):
                continue
            if old is None and change.from_document_id is None:
                continue  # first version of a source: nothing changed relative to an earlier version
            plan_matches_previous = False
            if old is not None:
                diffs_old, compared = plan_differences(plan, old.parsed)
                if not diffs_old and (new is None or new.stmt.statement_type == "withdrawal"
                                      or plan_differences(plan, new.parsed)[0]):
                    plan_matches_previous = compared or new is None or new.stmt.statement_type == "withdrawal"
            meta = (new or old).doc.doc_metadata or {}
            comparisons.append(VersionComparison(
                source_identity=change.source_identity,
                source_title=meta.get("title"),
                change_category=change.change_category,
                previous=build_citation(old.stmt, old.doc) if old else None,
                latest=build_citation(new.stmt, new.doc) if new else None,
                differences=to_diffs(change_diffs(old, new)),
                plan_matches_previous=plan_matches_previous,
            ))
        return comparisons

    def process_change_record(self, change_record_id: str) -> GapRecord:
        """Compare an extracted clinical recommendation against active protocols in ChromaDB.

        Args:
            change_record_id: UUID string of the ChangeRecord.

        Returns:
            Persisted GapRecord instance.

        Raises:
            ValueError: If the ChangeRecord is not found or in an illegal lifecycle state.
        """
        session = self.session_factory()
        try:
            change = session.get(ChangeRecord, change_record_id)
            if change is None:
                raise ValueError(f"ChangeRecord with ID '{change_record_id}' not found.")

            logger.info("Starting protocol comparison for ChangeRecord ID: %s", change.id)

            # 1. State transition: extracted -> comparison_pending (if currently extracted)
            if change.status == ChangeStatus.EXTRACTED.value:
                validate_transition(
                    change.status,
                    ChangeStatus.COMPARISON_PENDING.value,
                    CHANGE_TRANSITIONS,
                    "ChangeRecord",
                )
                change.status = ChangeStatus.COMPARISON_PENDING.value
                session.commit()

            # 2. Local semantic retrieval from ChromaDB via Phase 5 service
            candidates: List[CandidateProtocolSection] = self.protocol_index_service.retrieve(
                recommendation_text=change.verbatim_text,
                top_k=self.top_k,
            )
            candidate_ids = [c.chroma_id for c in candidates]
            logger.info(
                "ChangeRecord %s: retrieved %d candidate protocol section(s).",
                change.id,
                len(candidates),
            )

            # 3. Check similarity threshold (0.70)
            threshold = self.config.protocol_similarity_threshold
            matching_candidates = [c for c in candidates if c.similarity >= threshold]

            # 4. Handle NO-MATCH / G3
            if not matching_candidates:
                top_similarity = candidates[0].similarity if candidates else 0.0
                logger.warning(
                    "ChangeRecord %s: no protocol candidate reached similarity threshold %.2f (top: %.2f) -> routing to G3.",
                    change.id,
                    threshold,
                    top_similarity,
                )

                validate_transition(
                    change.status,
                    ChangeStatus.HELD_FOR_G3.value,
                    CHANGE_TRANSITIONS,
                    "ChangeRecord",
                )
                change.status = ChangeStatus.HELD_FOR_G3.value

                gap = self._upsert_gap_record(
                    session=session,
                    change_id=change.id,
                    candidate_section_ids=candidate_ids,
                    similarity=top_similarity,
                    comparison_result=ComparisonResult.NO_MATCH.value,
                    comparison_confidence=1.0,
                    difference_type=DifferenceType.NO_MATCH.value,
                    matched_protocol_id=None,
                    matched_protocol_version=None,
                    is_match=False,
                    status=GapStatus.NO_MATCH.value,
                    details={
                        "review_reason": (
                            f"No protocol section reached similarity threshold {threshold:.2f} "
                            f"(top similarity {top_similarity:.2f})."
                        ),
                    },
                )
                session.commit()
                session.refresh(gap)
                session.expunge(gap)
                return gap

            # 5. Focus on best matching candidate for Groq comparison
            best_candidate = matching_candidates[0]
            logger.info(
                "ChangeRecord %s: matching candidate found: protocol=%s version=%s section=%s (similarity=%.2f)",
                change.id,
                best_candidate.protocol_id,
                best_candidate.protocol_version,
                best_candidate.section_id,
                best_candidate.similarity,
            )

            # 6. Groq structured comparison call
            comparison_resp: Optional[ComparisonResponse] = None
            invalid_error_reason: Optional[str] = None

            try:
                comparison_resp = self.llm_client.compare_recommendation_to_protocol(
                    recommendation_text=change.verbatim_text,
                    target_population=change.target_population,
                    intervention=change.intervention,
                    candidate_section_heading=best_candidate.section_heading,
                    candidate_section_text=best_candidate.section_text,
                    candidate_protocol_id=best_candidate.protocol_id,
                    candidate_protocol_version=best_candidate.protocol_version,
                    candidate_section_id=best_candidate.section_id,
                    evidence_grade=change.evidence_grade,
                )
            except ValueError as e:
                # Unparseable / schema-invalid model output (LLMOutputError, ValidationError) -> G2.
                # Provider outages and missing credentials are not ambiguity: they propagate as failures.
                logger.warning("Groq comparison output invalid for ChangeRecord %s: %s", change.id, e)
                invalid_error_reason = f"Model structured output error: {e}"

            # 7. Verification & Routing logic
            if comparison_resp is None:
                # Invalid structured output -> G2
                validate_transition(change.status, ChangeStatus.HELD_FOR_G2.value, CHANGE_TRANSITIONS, "ChangeRecord")
                change.status = ChangeStatus.HELD_FOR_G2.value
                gap = self._upsert_gap_record(
                    session=session,
                    change_id=change.id,
                    candidate_section_ids=candidate_ids,
                    similarity=best_candidate.similarity,
                    comparison_result=ComparisonResult.AMBIGUOUS.value,
                    comparison_confidence=0.0,
                    difference_type=DifferenceType.NONE.value,
                    matched_protocol_id=best_candidate.protocol_id,
                    matched_protocol_version=best_candidate.protocol_version,
                    is_match=True,
                    status=GapStatus.REVIEW_REQUIRED.value,
                    details={
                        "matched_section_id": best_candidate.section_id,
                        "matched_section_heading": best_candidate.section_heading,
                        "review_reason": invalid_error_reason,
                    },
                )
                session.commit()
                session.refresh(gap)
                session.expunge(gap)
                return gap

            # 8. Protocol quotation verification
            # Ensure exact_protocol_text exists verbatim in the candidate section text
            quote_in_section = comparison_resp.exact_protocol_text in best_candidate.section_text
            metadata_matches = (
                comparison_resp.protocol_id == best_candidate.protocol_id
                and comparison_resp.protocol_version == best_candidate.protocol_version
                and comparison_resp.section_id == best_candidate.section_id
            )
            quote_verified = quote_in_section and metadata_matches

            # 9. Internal consistency check
            is_consistent = comparison_resp.is_internally_consistent()

            # 10. Confidence threshold check
            is_confident = comparison_resp.confidence >= self.config.comparison_confidence_threshold

            # 11. State determination
            reasons: List[str] = []
            if not quote_verified or not is_consistent or not is_confident:
                if not quote_verified:
                    reasons.append("Protocol quotation or metadata mismatch")
                if not is_consistent:
                    reasons.append("Contradictory comparison output")
                if not is_confident:
                    reasons.append(f"Confidence {comparison_resp.confidence:.2f} < threshold {self.config.comparison_confidence_threshold:.2f}")

                logger.warning(
                    "ChangeRecord %s held for G2: %s",
                    change.id,
                    "; ".join(reasons),
                )
                validate_transition(change.status, ChangeStatus.HELD_FOR_G2.value, CHANGE_TRANSITIONS, "ChangeRecord")
                change.status = ChangeStatus.HELD_FOR_G2.value
                gap_status = GapStatus.REVIEW_REQUIRED.value
            else:
                if comparison_resp.comparison_result == ComparisonResult.GAP:
                    validate_transition(change.status, ChangeStatus.GAP_CONFIRMED.value, CHANGE_TRANSITIONS, "ChangeRecord")
                    change.status = ChangeStatus.GAP_CONFIRMED.value
                    gap_status = GapStatus.MATCHED.value
                elif comparison_resp.comparison_result == ComparisonResult.NO_GAP:
                    validate_transition(change.status, ChangeStatus.NO_GAP.value, CHANGE_TRANSITIONS, "ChangeRecord")
                    change.status = ChangeStatus.NO_GAP.value
                    gap_status = GapStatus.MATCHED.value
                else:
                    validate_transition(change.status, ChangeStatus.HELD_FOR_G2.value, CHANGE_TRANSITIONS, "ChangeRecord")
                    change.status = ChangeStatus.HELD_FOR_G2.value
                    gap_status = GapStatus.REVIEW_REQUIRED.value

            gap = self._upsert_gap_record(
                session=session,
                change_id=change.id,
                candidate_section_ids=candidate_ids,
                similarity=best_candidate.similarity,
                comparison_result=comparison_resp.comparison_result.value,
                comparison_confidence=comparison_resp.confidence,
                difference_type=comparison_resp.difference_type.value,
                matched_protocol_id=best_candidate.protocol_id,
                matched_protocol_version=best_candidate.protocol_version,
                is_match=True,
                status=gap_status,
                details={
                    "matched_section_id": best_candidate.section_id,
                    "matched_section_heading": best_candidate.section_heading,
                    # An unverified quotation is never stored as protocol text.
                    "exact_protocol_text": comparison_resp.exact_protocol_text if quote_verified else None,
                    "specific_difference": comparison_resp.specific_difference,
                    "comparison_rationale": comparison_resp.rationale,
                    "review_reason": "; ".join(reasons) or None,
                },
            )
            session.commit()
            session.refresh(gap)
            session.expunge(gap)
            return gap

        except Exception as e:
            session.rollback()
            logger.error("ComparisonAgent failed for ChangeRecord %s: %s", change_record_id, e)
            raise
        finally:
            session.close()

    def _upsert_gap_record(
        self,
        session: Session,
        change_id: str,
        candidate_section_ids: List[str],
        similarity: float,
        comparison_result: str,
        comparison_confidence: float,
        difference_type: str,
        matched_protocol_id: Optional[str],
        matched_protocol_version: Optional[str],
        is_match: bool,
        status: str,
        details: Optional[dict] = None,
    ) -> GapRecord:
        """Idempotently create or update the single GapRecord for a ChangeRecord."""
        evidence = {key: None for key in _GAP_EVIDENCE_FIELDS}
        evidence.update(details or {})
        existing = session.query(GapRecord).filter_by(change_record_id=change_id).first()
        if existing:
            for key, value in evidence.items():
                setattr(existing, key, value)
            existing.candidate_protocol_section_ids = candidate_section_ids
            existing.similarity = similarity
            existing.comparison_result = comparison_result
            existing.comparison_confidence = comparison_confidence
            existing.difference_type = difference_type
            existing.matched_protocol_id = matched_protocol_id
            existing.matched_protocol_version = matched_protocol_version
            existing.is_match = is_match
            existing.status = status
            return existing
        else:
            gap = GapRecord(
                change_record_id=change_id,
                candidate_protocol_section_ids=candidate_section_ids,
                similarity=similarity,
                comparison_result=comparison_result,
                comparison_confidence=comparison_confidence,
                difference_type=difference_type,
                matched_protocol_id=matched_protocol_id,
                matched_protocol_version=matched_protocol_version,
                is_match=is_match,
                status=status,
                schema_version="1.0",
                **evidence,
            )
            session.add(gap)
            return gap


def change_diffs(old: Optional[StatementRef], new: Optional[StatementRef]):
    if old is None or new is None or new.stmt.statement_type == "withdrawal":
        return []
    return attribute_differences(old.parsed, new.parsed)
