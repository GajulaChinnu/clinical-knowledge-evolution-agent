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
from app.services.llm_client import SharedLLMClient
from app.services.pdf_parser import (
    detect_document_sections,
    extract_page_texts,
    filter_candidate_sections,
)
from app.services.source_verifier import verify_recommendation_provenance

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

            # 2. Extract text and detect sections locally
            page_texts = extract_page_texts(doc.source_path)
            sections = detect_document_sections(page_texts)

            # 3. Filter candidate sections (token reduction heuristic)
            candidate_sections = filter_candidate_sections(sections)
            logger.info(
                "Document %s: %d total section(s), %d candidate section(s) selected for extraction.",
                doc.id,
                len(sections),
                len(candidate_sections),
            )

            created_records: List[ChangeRecord] = []

            # 4. Process each candidate section via Groq
            for section in candidate_sections:
                logger.info(
                    "Extracting from candidate section '%s' on page %d for document %s",
                    section.section_heading,
                    section.page_number,
                    doc.id,
                )

                # Send ONLY the candidate section text to Groq
                extraction_resp: ExtractionResponse = self.llm_client.extract_recommendations(
                    section_heading=section.section_heading,
                    section_text=section.text,
                    page_number=section.page_number,
                    document_identifier=doc.source_identifier,
                )

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
                    )
                    session.add(change)
                    session.commit()
                    created_records.append(change)

            # 9. Update document lifecycle state
            if any(c.status == ChangeStatus.HELD_FOR_G1.value for c in created_records):
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
