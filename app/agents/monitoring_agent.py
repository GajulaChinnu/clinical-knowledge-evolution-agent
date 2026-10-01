"""CKEA Monitoring Agent for local synthetic PDF discovery, hashing, and ingestion."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pdfplumber
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import get_session_factory
from app.models.entities import IngestedDocument, IngestionFailure
from app.schemas.documents import DocumentStatus
from app.schemas.transitions import DOCUMENT_TRANSITIONS, validate_transition
from app.services.config_service import AppConfig, load_config
from app.services.file_hash import compute_sha256

logger = logging.getLogger("ckea.monitoring")


@dataclass
class ScanResult:
    """Operational summary of a monitoring scan execution."""
    discovered: int = 0
    new_documents: int = 0
    skipped: int = 0
    failed: int = 0
    ingested_document_ids: List[str] = field(default_factory=list)
    failure_ids: List[str] = field(default_factory=list)

    @property
    def new(self) -> int:
        """Alias for new_documents."""
        return self.new_documents


class MonitoringAgent:
    """Specialized agent responsible for discovering and ingesting source PDFs."""

    def __init__(
        self,
        source_dir: Optional[Union[str, Path]] = None,
        session_factory: Optional[sessionmaker[Session]] = None,
        config: Optional[AppConfig] = None,
        pipeline_version: str = "1.0",
        parser_version: str = "1.0",
    ) -> None:
        """Initialize the Monitoring Agent.

        Args:
            source_dir: Directory to scan for PDFs. Defaults to config.source_dir.
            session_factory: SQLAlchemy sessionmaker factory. Defaults to global session factory.
            config: Application configuration. Defaults to loaded configuration.
            pipeline_version: Pipeline execution version for idempotency tracking.
            parser_version: PDF parser version for provenance tracking.
        """
        self.config = config or load_config()
        self.source_dir = Path(source_dir) if source_dir is not None else self.config.source_dir
        self.session_factory = session_factory or get_session_factory()
        self.pipeline_version = pipeline_version
        self.parser_version = parser_version

    @staticmethod
    def _get_next_version(existing_versions: List[str]) -> str:
        """Compute the next major version string (e.g., '1.0' -> '2.0')."""
        if not existing_versions:
            return "1.0"
        max_major = 0
        for v in existing_versions:
            try:
                parts = v.split(".")
                major = int(parts[0])
                if major > max_major:
                    max_major = major
            except (ValueError, IndexError):
                pass
        return f"{max_major + 1}.0"

    def scan(self) -> ScanResult:
        """Scan the configured source directory and ingest discovered PDF files.

        Returns:
            ScanResult containing counts of discovered, new, skipped, and failed files.

        Raises:
            FileNotFoundError: If the configured source directory does not exist.
            NotADirectoryError: If the configured source path is not a directory.
        """
        if not self.source_dir.exists():
            logger.error("Configured source directory does not exist: %s", self.source_dir)
            raise FileNotFoundError(f"Configured source directory does not exist: {self.source_dir}")

        if not self.source_dir.is_dir():
            logger.error("Configured source path is not a directory: %s", self.source_dir)
            raise NotADirectoryError(f"Configured source path is not a directory: {self.source_dir}")

        pdf_files = sorted(
            [p for p in self.source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
            key=lambda p: p.name,
        )

        result = ScanResult(discovered=len(pdf_files))
        logger.info("Discovered %d PDF file(s) in %s", len(pdf_files), self.source_dir)

        session = self.session_factory()
        try:
            for file_path in pdf_files:
                self._process_file(file_path, session, result)
        finally:
            session.close()

        return result

    def _process_file(
        self,
        file_path: Path,
        session: Session,
        result: ScanResult,
    ) -> None:
        """Process a single PDF file with isolated error handling."""
        logger.info("File discovered: %s", file_path.name)
        source_identifier = file_path.stem

        try:
            # 1. Compute SHA-256 hash of file contents
            file_hash = compute_sha256(file_path)

            # 2. Query existing documents for this source_identifier
            stmt = (
                select(IngestedDocument)
                .where(IngestedDocument.source_identifier == source_identifier)
                .order_by(IngestedDocument.created_at.desc())
            )
            existing_docs = session.scalars(stmt).all()

            # 3. Check for unchanged content (idempotency check)
            matching_existing = [
                d for d in existing_docs
                if d.sha256_hash == file_hash and d.status != DocumentStatus.FAILED.value
            ]
            if matching_existing:
                logger.info(
                    "File skipped (unchanged content): %s (SHA-256: %s)",
                    file_path.name,
                    file_hash[:8],
                )
                result.skipped += 1
                return

            # 4. Determine document version
            if not existing_docs:
                next_version = "1.0"
            else:
                next_version = self._get_next_version([d.document_version for d in existing_docs])

            # 5. Validate PDF readability via pdfplumber
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                if page_count == 0:
                    raise ValueError(f"PDF file contains 0 pages: {file_path.name}")
                raw_meta = pdf.metadata or {}
                clean_meta: Dict[str, Any] = {
                    str(k): str(v) for k, v in raw_meta.items() if v is not None
                }

            doc_metadata = {
                "page_count": page_count,
                "file_size_bytes": file_path.stat().st_size,
                "pdf_metadata": clean_meta,
            }

            # 6. Instantiate and transition record
            doc = IngestedDocument(
                source_identifier=source_identifier,
                source_path=str(file_path),
                sha256_hash=file_hash,
                document_version=next_version,
                source_version=next_version,
                parser_version=self.parser_version,
                pipeline_version=self.pipeline_version,
                status=DocumentStatus.DISCOVERED.value,
                doc_metadata=doc_metadata,
            )

            # Reusable transition primitive: discovered -> parsed
            validate_transition(
                doc.status,
                DocumentStatus.PARSED.value,
                DOCUMENT_TRANSITIONS,
                "IngestedDocument",
            )
            doc.status = DocumentStatus.PARSED.value

            session.add(doc)
            session.commit()

            result.new_documents += 1
            result.ingested_document_ids.append(doc.id)
            logger.info(
                "File ingested: %s (version: %s, pages: %d, SHA-256: %s...)",
                file_path.name,
                next_version,
                page_count,
                file_hash[:8],
            )

        except Exception as e:
            session.rollback()
            err_category = type(e).__name__
            err_message = str(e)
            logger.error(
                "File failed: %s [%s: %s]",
                file_path.name,
                err_category,
                err_message,
            )

            failure = IngestionFailure(
                source_path=str(file_path),
                error_category=err_category,
                error_message=err_message,
                retry_count=0,
                retry_status="pending",
                operator_status="unresolved",
                schema_version="1.0",
            )
            session.add(failure)
            session.commit()

            result.failed += 1
            result.failure_ids.append(failure.id)
