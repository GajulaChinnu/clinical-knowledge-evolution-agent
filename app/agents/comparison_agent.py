"""CKEA Comparison Agent for evaluating extracted recommendations against institutional protocols."""

import logging
from typing import Any, List, Optional
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import ChangeRecord, GapRecord
from app.schemas.changes import ChangeStatus
from app.schemas.comparison import ComparisonResponse
from app.schemas.gaps import ComparisonResult, DifferenceType, GapStatus
from app.schemas.protocol import CandidateProtocolSection
from app.schemas.transitions import CHANGE_TRANSITIONS, validate_transition
from app.services.config_service import AppConfig, load_config
from app.services.llm_client import SharedLLMClient
from app.services.protocol_index import ProtocolIndexService

logger = logging.getLogger("ckea.comparison")


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
        self.protocol_index_service = protocol_index_service or ProtocolIndexService(config=self.config)
        self.top_k = top_k

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
            except Exception as e:
                logger.warning("Groq comparison call or parsing failed for ChangeRecord %s: %s", change.id, e)
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
            if not quote_verified or not is_consistent or not is_confident:
                reasons = []
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
    ) -> GapRecord:
        """Idempotently create or update the single GapRecord for a ChangeRecord."""
        existing = session.query(GapRecord).filter_by(change_record_id=change_id).first()
        if existing:
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
            )
            session.add(gap)
            return gap
