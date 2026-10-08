"""CKEA Extraction Agent for structured clinical recommendation extraction."""

import logging
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import ChangeRecord, IngestedDocument
from app.schemas.changes import ChangeStatus
from app.schemas.documents import DocumentStatus
from app.schemas.extraction import ExtractionResponse
from app.schemas.transitions import DOCUMENT_TRANSITIONS, validate_transition
from app.services.config_service import AppConfig, load_config
from app.services.llm_client import LLMOutputError, SharedLLMClient
from app.services.pdf_parser import filter_candidate_sections, DocumentSection
from app.services.source_documents import load_source_sections
from app.services.source_verifier import verify_recommendation_provenance
from app.services.source_version_diff import SourceVersionDiffService

logger = logging.getLogger("ckea.extraction")


class ExtractionAgent:
    """Specialized agent responsible for extracting structured recommendations from ingested PDFs."""

    def __init__(
        self,
        session_factory: Optional[sessionmaker[Session]] = None,
        llm_client: Optional[SharedLLMClient] = None,
        config: Optional[AppConfig] = None,
        prompt_version: str = "1.0.0",
    ) -> None:
        self.config = config or load_config()
        self.session_factory = session_factory or get_session_factory()
        self.llm_client = llm_client or SharedLLMClient(config=self.config)
        self.prompt_version = prompt_version

    def index_guidance_statements(self, document_ids: List[str]) -> List["IndexResult"]:
        """Query/surveillance mode: build the verbatim statement index and version change log.

        Deterministic (no LLM): statements are exact substrings of the stored artifact, so every
        citation shown to a clinician is grounded by construction. Idempotent per document.
        """
        from app.services.guidance_index import GuidanceIndexService

        service = GuidanceIndexService(self.session_factory)
        return [service.index_document(doc_id) for doc_id in document_ids]

    def record_change_from_statement(self, statement_id: str, change_category: str) -> str:
        """Record a verified guidance statement as a ChangeRecord (query mode, deterministic, idempotent).

        The statement is verbatim from the stored source, so provenance needs no LLM verification.
        """
        from app.models.entities import GuidanceStatement
        from app.services.taxonomy import get_taxonomy

        with self.session_factory() as session:
            stmt = session.get(GuidanceStatement, statement_id)
            if stmt is None:
                raise ValueError(f"GuidanceStatement '{statement_id}' not found.")
            existing = (session.query(ChangeRecord)
                        .filter_by(ingested_document_id=stmt.ingested_document_id, verbatim_text=stmt.verbatim_text)
                        .first())
            if existing is not None:
                return existing.id
            taxonomy = get_taxonomy()
            names = [taxonomy.treatment_name(t) for t in (stmt.treatments or [])]
            change = ChangeRecord(
                ingested_document_id=stmt.ingested_document_id,
                verbatim_text=stmt.verbatim_text,
                recommendation_type=stmt.statement_type,
                target_population="As stated in the source excerpt",
                intervention=", ".join(names) or "As stated in the source excerpt",
                evidence_grade=f"Grade {stmt.evidence_level}" if stmt.evidence_level else None,
                confidence=1.0,
                page=1,
                section=stmt.section_heading,
                source_excerpt=stmt.verbatim_text,
                extraction_model_version="deterministic-statement-index",
                extraction_prompt_version="n/a",
                status=ChangeStatus.EXTRACTED.value,
                schema_version="1.0",
                change_category=change_category,
                departments=list(stmt.departments or []),
                treatments=list(stmt.treatments or []),
            )
            session.add(change)
            session.commit()
            return change.id

    def _classify_recommendation(self, session: Session, doc: IngestedDocument, verbatim: str) -> dict:
        """Department, treatment and change-category classification (taxonomy + statement index)."""
        from app.models.entities import GuidanceChange, GuidanceStatement
        from app.services.clinical_statements import similarity
        from app.services.taxonomy import TaxonomyError, get_taxonomy

        try:
            taxonomy = get_taxonomy()
        except TaxonomyError:
            return {}
        treatments = taxonomy.find_treatments(verbatim).treatments
        meta = doc.doc_metadata or {}
        departments = list(meta.get("departments") or taxonomy.departments_for_treatments(treatments))
        category = None
        for change in session.query(GuidanceChange).filter_by(to_document_id=doc.id):
            stmt = session.get(GuidanceStatement, change.to_statement_id) if change.to_statement_id else None
            if stmt is not None and similarity(" ".join(stmt.verbatim_text.split()), " ".join(verbatim.split())) >= 0.85:
                category = change.change_category
                break
        if category is None:
            category = "new_recommendation" if not doc.previous_source_version_id else "revised_recommendation"
        return {"change_category": category, "departments": sorted(set(departments)), "treatments": treatments}

    def process_document(self, document_id: str) -> List[ChangeRecord]:
        """Extract clinical recommendations from an ingested document.

        Args:
            document_id: UUID of the IngestedDocument record.

        Returns:
            List of persisted ChangeRecord instances.

        Raises:
            ValueError: If the document is not found or in an invalid state.
        """
        session = self.session_factory()
        try:
            doc = session.get(IngestedDocument, document_id)
            if doc is None:
                raise ValueError(f"IngestedDocument with ID '{document_id}' not found.")

            logger.info("Starting recommendation extraction for document ID: %s", doc.id)

            # 1. Transition document: parsed -> processing (if currently parsed)
            if doc.status == DocumentStatus.PARSED.value:
                validate_transition(
                    doc.status,
                    DocumentStatus.PROCESSING.value,
                    DOCUMENT_TRANSITIONS,
                    "IngestedDocument",
                )
                doc.status = DocumentStatus.PROCESSING.value
                session.commit()

            # 2. Load sections from the stored artifact (PDF or normalized web text)
            sections = load_source_sections(doc.source_path)

            # 3. Handle version differencing if previous version exists
            if doc.previous_source_version_id:
                old_doc = session.get(IngestedDocument, doc.previous_source_version_id)
                if old_doc:
                    logger.info("Previous version found (%s), performing diff.", old_doc.id)
                    old_sections = load_source_sections(old_doc.source_path)

                    diff_service = SourceVersionDiffService()
                    changes = diff_service.diff_sections(old_sections, sections)

                    # Source-evolution artifact (v(n-1) -> v(n)); kept separate from the
                    # institutional protocol comparison. Removals are recorded, not dropped.
                    doc.doc_metadata = {
                        **(doc.doc_metadata or {}),
                        "source_diff": {
                            "previous_document_id": old_doc.id,
                            "changes": [c.to_dict() for c in changes],
                        },
                    }
                    session.commit()

                    sec_map = {s.section_heading: s for s in sections}
                    diffed_sections = []
                    for c in changes:
                        if c.change_type in ("added", "modified"):
                            original_sec = sec_map.get(c.heading)
                            diffed_sections.append(DocumentSection(
                                page_number=original_sec.page_number if original_sec else 1,
                                section_heading=c.heading,
                                text=c.new_text,
                                char_start=original_sec.char_start if original_sec else None,
                                char_end=original_sec.char_end if original_sec else None,
                            ))

                    candidate_sections = filter_candidate_sections(diffed_sections)
                    logger.info(
                        "Document %s: %d total section(s), %d changed section(s), %d candidate section(s) selected for extraction.",
                        doc.id,
                        len(sections),
                        len(diffed_sections),
                        len(candidate_sections),
                    )
                else:
                    candidate_sections = filter_candidate_sections(sections)
                    logger.info(
                        "Document %s: %d total section(s), %d candidate section(s) selected for extraction.",
                        doc.id,
                        len(sections),
                        len(candidate_sections),
                    )
            else:
                candidate_sections = filter_candidate_sections(sections)
                logger.info(
                    "Document %s: %d total section(s), %d candidate section(s) selected for extraction.",
                    doc.id,
                    len(sections),
                    len(candidate_sections),
                )

            created_records: List[ChangeRecord] = []
            extraction_issues: List[dict] = []

            # 4. Process each candidate section via Groq
            for section in candidate_sections:
                logger.info(
                    "Extracting from candidate section '%s' on page %d for document %s",
                    section.section_heading,
                    section.page_number,
                    doc.id,
                )

                # Send ONLY the candidate section text to Groq. Invalid model output for a
                # section is a G1 matter (human extraction review), not a document failure;
                # provider outages (LLMTransportError) still propagate as failures.
                try:
                    extraction_resp: ExtractionResponse = self.llm_client.extract_recommendations(
                        section_heading=section.section_heading,
                        section_text=section.text,
                        page_number=section.page_number,
                        document_identifier=doc.source_identifier,
                    )
                except LLMOutputError as e:
                    logger.warning(
                        "Invalid extraction output for section '%s' (page %d) of document %s -> G1: %s",
                        section.section_heading, section.page_number, doc.id, e,
                    )
                    extraction_issues.append({
                        "section": section.section_heading,
                        "page": section.page_number,
                        "reason": str(e)[:500],
                    })
                    continue

                for rec in extraction_resp.recommendations:
                    # 5. Source verification: ensure exact quotation exists in the source section
                    verification = verify_recommendation_provenance(
                        verbatim_text=rec.verbatim_text,
                        source_excerpt=rec.source_excerpt,
                        source_section_text=section.text,
                        page=rec.page,
                        expected_page=section.page_number,
                    )

                    # 6. Check confidence threshold
                    is_confident = rec.confidence >= self.config.extraction_confidence_threshold

                    # 7. Determine initial ChangeRecord state
                    if verification.is_verified and is_confident:
                        record_status = ChangeStatus.EXTRACTED.value
                        logger.info(
                            "Extraction succeeded for document %s (page %d, confidence %.2f)",
                            doc.id,
                            rec.page,
                            rec.confidence,
                        )
                    else:
                        record_status = ChangeStatus.HELD_FOR_G1.value
                        reason = verification.reason if not verification.is_verified else f"Confidence {rec.confidence:.2f} < threshold {self.config.extraction_confidence_threshold:.2f}"
                        logger.warning(
                            "Extraction held for G1 review for document %s (page %d): %s",
                            doc.id,
                            rec.page,
                            reason,
                        )

                    # 8. Persist ChangeRecord
                    change = ChangeRecord(
                        ingested_document_id=doc.id,
                        verbatim_text=rec.verbatim_text,
                        recommendation_type=rec.recommendation_type,
                        target_population=rec.target_population,
                        intervention=rec.intervention,
                        evidence_grade=rec.evidence_grade,
                        confidence=rec.confidence,
                        page=rec.page,
                        section=rec.section,
                        source_excerpt=rec.source_excerpt,
                        extraction_model_version=self.config.groq_model,
                        extraction_prompt_version=self.prompt_version,
                        status=record_status,
                        schema_version="1.0",
                        **self._classify_recommendation(session, doc, rec.verbatim_text),
                    )
                    session.add(change)
                    session.commit()
                    created_records.append(change)

            # 9. Update document lifecycle state
            if extraction_issues:
                doc.doc_metadata = {**(doc.doc_metadata or {}), "extraction_issues": extraction_issues}
            if (
                any(c.status == ChangeStatus.HELD_FOR_G1.value for c in created_records)
                or not created_records
                or extraction_issues
            ):
                if not created_records:
                    logger.warning(
                        "Document %s held for G1: no recommendation extracted from %d section(s) / %d candidate(s).",
                        doc.id,
                        len(sections),
                        len(candidate_sections),
                    )
                if doc.status == DocumentStatus.PROCESSING.value:
                    validate_transition(
                        doc.status,
                        DocumentStatus.HELD.value,
                        DOCUMENT_TRANSITIONS,
                        "IngestedDocument",
                    )
                    doc.status = DocumentStatus.HELD.value
            else:
                if doc.status == DocumentStatus.PROCESSING.value:
                    validate_transition(
                        doc.status,
                        DocumentStatus.COMPLETE.value,
                        DOCUMENT_TRANSITIONS,
                        "IngestedDocument",
                    )
                    doc.status = DocumentStatus.COMPLETE.value

            session.commit()
            return created_records

        except Exception as e:
            session.rollback()
            logger.error("Document extraction failed for document %s: %s", document_id, e)
            raise
        finally:
            session.close()
