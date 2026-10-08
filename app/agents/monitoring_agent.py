"""CKEA Monitoring Agent: source discovery, hashing, versioning and ingestion.

Versioning is driven by the SHA-256 of the canonical normalized text, so re-exported PDFs
or re-fetched pages with identical content are no-ops, while any content change against
the latest version creates a new immutable version linked to its predecessor. The raw
artifact SHA-256 is kept for integrity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from app.services.source_documents import artifact_format, content_sha256, load_source_text
from app.services.watchlist import Watchlist, WatchlistEntry, WatchlistError, version_key

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


@dataclass
class SourceVersionInfo:
    """One stored version of a monitored source."""
    document_id: str
    internal_version: str
    publisher_version: Optional[str]
    published_date: Optional[str]
    sha256: str
    ingested_at: Optional[datetime]


@dataclass
class SourceCheckResult:
    """Outcome of checking one watchlist source for new versions."""
    entry_id: str
    source_identity: str
    checked_at: datetime
    new_document_ids: List[str] = field(default_factory=list)
    unchanged: int = 0
    versions: List[SourceVersionInfo] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def latest(self) -> Optional[SourceVersionInfo]:
        return self.versions[-1] if self.versions else None

    @property
    def previous(self) -> Optional[SourceVersionInfo]:
        return self.versions[-2] if len(self.versions) > 1 else None


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

    def ingest_known_file(
        self,
        file_path: Path,
        source_identifier: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> ScanResult:
        """Explicitly ingest a file with a known, stable source identifier.

        Args:
            file_path: Path to the target PDF file.
            source_identifier: Stable canonical identifier for the source (e.g. URL).
            doc_metadata: Optional metadata to attach to the IngestedDocument.
            session: Optional SQLAlchemy session. If not provided, a new one is created.

        Returns:
            ScanResult detailing the outcome.
        """
        result = ScanResult(discovered=1)
        owns_session = False
        if session is None:
            session = self.session_factory()
            owns_session = True

        try:
            self._process_file_with_id(
                file_path=file_path,
                source_identifier=source_identifier,
                doc_metadata=doc_metadata,
                session=session,
                result=result,
            )
            if owns_session:
                session.commit()
        except Exception:
            if owns_session:
                session.rollback()
            raise
        finally:
            if owns_session:
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
        self._process_file_with_id(
            file_path=file_path,
            source_identifier=source_identifier,
            doc_metadata=None,
            session=session,
            result=result,
        )

    def _process_file_with_id(
        self,
        file_path: Path,
        source_identifier: str,
        doc_metadata: Optional[Dict[str, Any]],
        session: Session,
        result: ScanResult,
    ) -> None:
        """Core logic to process a file given a specific source identifier."""

        try:
            # 1. Compute SHA-256 of the raw artifact (integrity) and of the canonical
            #    normalized text (versioning). Unreadable sources raise here.
            file_hash = compute_sha256(file_path)
            fmt = artifact_format(file_path)
            normalized_hash = content_sha256(load_source_text(file_path))

            # 2. Query existing documents for this source_identifier
            stmt = (
                select(IngestedDocument)
                .where(IngestedDocument.source_identifier == source_identifier)
                .order_by(IngestedDocument.created_at.desc())
            )
            existing_docs = session.scalars(stmt).all()

            # 3. Idempotency: unchanged relative to the latest non-failed version
            latest_valid = next(
                (d for d in existing_docs if d.status != DocumentStatus.FAILED.value), None
            )
            if latest_valid is not None and (
                latest_valid.sha256_hash == file_hash
                or (latest_valid.doc_metadata or {}).get("normalized_content_sha256") == normalized_hash
            ):
                logger.info(
                    "File skipped (unchanged content): %s (SHA-256: %s, normalized: %s)",
                    file_path.name,
                    file_hash[:8],
                    normalized_hash[:8],
                )
                result.skipped += 1
                return

            # 4. Determine document version and previous version linkage
            previous_source_version_id = None
            previous_sha256_hash = None
            change_status = "first_seen"

            if not existing_docs:
                next_version = "1.0"
            else:
                next_version = self._get_next_version([d.document_version for d in existing_docs])
                latest_existing = latest_valid or existing_docs[0]
                previous_source_version_id = latest_existing.id
                previous_sha256_hash = latest_existing.sha256_hash
                change_status = "changed"

            # 5. Artifact metadata (PDF structure when applicable)
            page_count = 1
            clean_meta: Dict[str, Any] = {}
            if fmt == "pdf":
                with pdfplumber.open(file_path) as pdf:
                    page_count = len(pdf.pages)
                    if page_count == 0:
                        raise ValueError(f"PDF file contains 0 pages: {file_path.name}")
                    raw_meta = pdf.metadata or {}
                    clean_meta = {str(k): str(v) for k, v in raw_meta.items() if v is not None}

            combined_metadata = {
                "page_count": page_count,
                "file_size_bytes": file_path.stat().st_size,
                "pdf_metadata": clean_meta,
                "artifact_format": fmt,
                "original_bytes_sha256": file_hash,
                "normalized_content_sha256": normalized_hash,
            }
            if doc_metadata:
                combined_metadata.update(
                    {k: v for k, v in doc_metadata.items()
                     if k not in ("original_bytes_sha256", "normalized_content_sha256", "artifact_format")}
                )

            # 6. Instantiate and transition record
            doc = IngestedDocument(
                source_identifier=source_identifier,
                source_path=str(file_path),
                sha256_hash=file_hash,
                document_version=next_version,
                source_version=next_version,
                previous_source_version_id=previous_source_version_id,
                previous_sha256_hash=previous_sha256_hash,
                change_status=change_status,
                parser_version=self.parser_version,
                pipeline_version=self.pipeline_version,
                status=DocumentStatus.DISCOVERED.value,
                doc_metadata=combined_metadata,
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

    # ==========================================================================
    # WATCHLIST MONITORING (defined source list; corpus or URL locations)
    # ==========================================================================

    def version_history(self, source_identity: str, session: Optional[Session] = None) -> List[SourceVersionInfo]:
        """Stored versions of a source, oldest first (failed records excluded)."""
        owns = session is None
        session = session or self.session_factory()
        try:
            docs = (
                session.query(IngestedDocument)
                .filter(IngestedDocument.source_identifier == source_identity)
                .filter(IngestedDocument.status != DocumentStatus.FAILED.value)
                .order_by(IngestedDocument.created_at)
                .all()
            )
            return [
                SourceVersionInfo(
                    document_id=d.id,
                    internal_version=d.source_version,
                    publisher_version=(d.doc_metadata or {}).get("publisher_version"),
                    published_date=(d.doc_metadata or {}).get("published_date"),
                    sha256=d.sha256_hash,
                    ingested_at=d.ingest_timestamp,
                )
                for d in docs
            ]
        finally:
            if owns:
                session.close()

    def check_watchlist(
        self,
        watchlist: Watchlist,
        entries: Optional[List[WatchlistEntry]] = None,
        url_service: Any = None,
    ) -> List[SourceCheckResult]:
        """Check watchlist sources (all, or the given entries) and ingest new versions."""
        return [self.check_watchlist_entry(e, watchlist, url_service) for e in (entries or list(watchlist))]

    def check_watchlist_entry(
        self,
        entry: WatchlistEntry,
        watchlist: Watchlist,
        url_service: Any = None,
    ) -> SourceCheckResult:
        """Ingest any versions of one watchlist source that are newer than what is stored.

        Corpus sources publish versioned files; every newer published version is ingested in
        order so the lineage v1 -> v2 is preserved. URL sources are fetched once per check.
        Failures are recorded as IngestionFailure and returned (never silently dropped).
        """
        result = SourceCheckResult(entry.id, entry.source_identity, datetime.now(timezone.utc))
        try:
            if entry.is_corpus:
                self._check_corpus_entry(entry, watchlist, result)
            elif entry.is_url:
                self._check_url_entry(entry, url_service, result)
            else:
                raise WatchlistError(f"Unsupported location for '{entry.id}': {entry.location}")
        except (WatchlistError, OSError, ValueError) as e:
            result.error = f"{type(e).__name__}: {e}"
            self._record_watchlist_failure(entry, e)
        except Exception as e:  # URL ingestion raises a family of provider errors; record, never drop
            result.error = f"{type(e).__name__}: {e}"
            self._record_watchlist_failure(entry, e)
        result.versions = self.version_history(entry.source_identity)
        return result

    def _check_corpus_entry(self, entry: WatchlistEntry, watchlist: Watchlist, result: SourceCheckResult) -> None:
        stored = self.version_history(entry.source_identity)
        stored_publisher = {v.publisher_version for v in stored if v.publisher_version}
        latest_key = max((version_key(v) for v in stored_publisher), default=None)
        self.source_dir.mkdir(parents=True, exist_ok=True)

        for version in watchlist.corpus_versions(entry):
            if version.version in stored_publisher:
                continue
            if latest_key is not None and version.version_key < latest_key:
                logger.info("Skipping older corpus version %s of %s (latest stored is newer).", version.version, entry.id)
                continue
            artifact = (self.source_dir / f"{entry.id}__v{version.version}.md").resolve()
            if artifact.parent != self.source_dir.resolve():
                raise WatchlistError(f"Unsafe artifact path for '{entry.id}'")
            artifact.write_text(version.body, encoding="utf-8", newline="\n")
            metadata = {
                **entry.metadata(),
                "publisher_version": version.version,
                "published_date": version.published_date,
                "input_type": "watchlist",
                "retrieval_provider": "Watchlist corpus",
                "routing_decision": "corpus",
                "source_location": entry.location,
                "source_type_label": entry.source_type,
            }
            scan = self.ingest_known_file(artifact, entry.source_identity, metadata)
            result.new_document_ids.extend(scan.ingested_document_ids)
            result.unchanged += scan.skipped
            if scan.failed:
                raise WatchlistError(f"Ingestion failed for {entry.id} v{version.version} (see ingestion failures).")
            latest_key = version.version_key

    def _check_url_entry(self, entry: WatchlistEntry, url_service: Any, result: SourceCheckResult) -> None:
        if url_service is None:
            raise WatchlistError(f"Watchlist entry '{entry.id}' is a URL source but no URL ingestion service was provided.")
        ingestion = url_service.ingest_url(url=entry.location, source_dir=self.source_dir)
        metadata = {
            **entry.metadata(),
            **ingestion.provenance_metadata(),
            "input_type": "watchlist",
            "source_url": entry.location,
            "retrieval_timestamp": ingestion.retrieval_timestamp.isoformat(),
        }
        scan = self.ingest_known_file(ingestion.saved_path, entry.source_identity, metadata)
        result.new_document_ids.extend(scan.ingested_document_ids)
        result.unchanged += scan.skipped

    def _record_watchlist_failure(self, entry: WatchlistEntry, error: Exception) -> None:
        logger.error("Watchlist check failed for %s: %s", entry.id, error)
        with self.session_factory() as session:
            session.add(IngestionFailure(
                source_path=f"watchlist:{entry.id} ({entry.location})",
                error_category=type(error).__name__,
                error_message=str(error),
                retry_count=0,
                retry_status="pending",
                operator_status="unresolved",
                schema_version="1.0",
            ))
            session.commit()
