"""Source ingestion orchestration for all CKEA entry points (PDF upload, PDF URL, web URL).

The UI only calls this service. Every entry point produces a `NormalizedSource`, is
versioned by the MonitoringAgent against a stable source identity, and is then handed to
the existing ClinicalKnowledgePipeline. Human gates (G1-G5) are untouched: no decision is
made here, and no LLM is called here.

Identity rules:
- URL sources: hash of the canonical URL.
- PDF uploads: either an existing source the user picks ("new version of ...") or, for a
  new source, an identity derived from the normalized content. Never the filename.
"""

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.agents.monitoring_agent import MonitoringAgent
from app.models.entities import IngestedDocument, IngestionFailure
from app.orchestration.pipeline import ClinicalKnowledgePipeline
from app.schemas.orchestration import HumanGate, PipelineResult
from app.services.config_service import AppConfig, load_config
from app.services.source_documents import (
    NormalizedSource,
    UnreadableSourceError,
    artifact_format,
    bytes_sha256,
    canonicalize_url,
    content_source_identity,
    content_sha256,
    load_source_text,
    url_source_identity,
)
from app.services.url_ingestion_service import (
    ContentChallengeError,
    URLIngestionError,
    URLIngestionService,
)

logger = logging.getLogger("ckea.services.source_ingestion")


class UnknownSourceError(ValueError):
    """Raised when a PDF upload targets a source identity that does not exist."""


# ==============================================================================
# FILE SAFETY
# ==============================================================================

def sanitize_filename(filename: str) -> str:
    """Safely sanitize an uploaded filename to prevent traversal and preserve .pdf."""
    if not filename or not filename.strip():
        raise ValueError("Uploaded filename cannot be empty.")

    base_name = Path(filename).name.strip()
    base_name = base_name.replace("/", "").replace("\\", "").replace("..", "")

    if not base_name.lower().endswith(".pdf"):
        raise ValueError(f"Only PDF files are accepted. Invalid filename: '{filename}'.")

    stem = base_name[:-4]
    clean_stem = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", stem).strip("._-")
    if not clean_stem:
        clean_stem = "clinical_source"

    return f"{clean_stem}.pdf"


def save_uploaded_pdf(file_bytes: bytes, filename: str, source_dir: Union[str, Path]) -> Path:
    """Save uploaded PDF bytes into the authorized source directory (traversal-safe, no clobbering)."""
    if not file_bytes:
        raise ValueError("Uploaded PDF file is empty (0 bytes).")

    safe_name = sanitize_filename(filename)
    target_dir = Path(source_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dest_path = (target_dir / safe_name).resolve()
    if dest_path.parent != target_dir:
        raise ValueError(f"Security error: path traversal detected for filename '{filename}'.")

    if dest_path.exists():
        new_hash = hashlib.sha256(file_bytes).hexdigest()
        if bytes_sha256(dest_path.read_bytes()) != new_hash:
            dest_path = (target_dir / f"{dest_path.stem}_{new_hash[:8]}.pdf").resolve()
            if dest_path.parent != target_dir:
                raise ValueError("Security error: path traversal detected.")

    dest_path.write_bytes(file_bytes)
    return dest_path


# ==============================================================================
# SERVICE
# ==============================================================================

_GATE_MESSAGES = {
    HumanGate.G1: "Processing paused at G1 (Pipeline is HELD) — human extraction review required "
                  "(No actionable clinical recommendation could be confidently extracted).",
    HumanGate.G2: "Processing paused at G2 — comparison review required.",
    HumanGate.G3: "No matching institutional protocol section found. G3 committee review required.",
    HumanGate.G4: "Evidence brief prepared; awaiting G4 authorized clinical governance decision.",
    HumanGate.G5: "Review SLA escalation active (G5 notice).",
}


class SourceIngestionService:
    """Single orchestration point for PDF and URL source ingestion."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        config: Optional[AppConfig] = None,
        url_service: Optional[URLIngestionService] = None,
        monitoring_agent: Optional[MonitoringAgent] = None,
        pipeline: Optional[ClinicalKnowledgePipeline] = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config or load_config()
        self._url_service = url_service
        self._monitoring_agent = monitoring_agent
        self._pipeline = pipeline

    # Lazily constructed collaborators keep construction cheap for callers that only need one path.
    @property
    def url_service(self) -> URLIngestionService:
        if self._url_service is None:
            self._url_service = URLIngestionService(config=self.config)
        return self._url_service

    @property
    def monitoring_agent(self) -> MonitoringAgent:
        if self._monitoring_agent is None:
            self._monitoring_agent = MonitoringAgent(
                source_dir=self.config.source_dir,
                session_factory=self.session_factory,
                config=self.config,
            )
        return self._monitoring_agent

    @property
    def pipeline(self) -> ClinicalKnowledgePipeline:
        if self._pipeline is None:
            self._pipeline = ClinicalKnowledgePipeline(
                session_factory=self.session_factory,
                config=self.config,
                monitoring_agent=self.monitoring_agent,
            )
        return self._pipeline

    # --------------------------------------------------------------------------
    # Failure recording
    # --------------------------------------------------------------------------

    def record_failure(self, source_ref: str, error: Exception) -> None:
        """Persist an ingestion failure so it is visible to operators (never silently dropped)."""
        try:
            with self.session_factory() as session:
                session.add(IngestionFailure(
                    source_path=source_ref,
                    error_category=type(error).__name__,
                    error_message=str(error),
                    retry_count=0,
                    retry_status="not_retried",
                    operator_status="unresolved",
                    schema_version="1.0",
                ))
                session.commit()
        except SQLAlchemyError:
            logger.exception("Could not record ingestion failure for %s", source_ref)

    # --------------------------------------------------------------------------
    # URL entry point (PDF URL or web page)
    # --------------------------------------------------------------------------

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Retrieve, normalize, version and process a URL source.

        Returns a structured result; a walled source returns a "Blocked" result. Other
        ingestion errors are recorded as IngestionFailure and re-raised.
        """
        try:
            ingestion_res = self.url_service.ingest_url(url=url, source_dir=self.config.source_dir)
        except (URLIngestionError, ValueError) as error:
            self.record_failure(url, error)
            if not isinstance(error, ContentChallengeError):
                raise
            logger.warning("URL ingestion rejected due to bot challenge: %s", error)
            return {
                "success": False,
                "input_type": "URL",
                "source_url": url,
                "ingestion_status": "Failed (Challenge Detected)",
                "pipeline_status": "Blocked",
                "gate_status": "Ingestion Failed",
                "result_message": str(error),
                "usable_clinical_content": False,
                "pipeline_reason": "Source returned browser/access challenge instead of article content.",
                "error": "The source website presented an anti-bot challenge, captcha, or cookie wall. "
                         "Content was rejected by clinical ingestion policy.",
            }

        saved_path = ingestion_res.saved_path
        try:
            text = load_source_text(saved_path)
        except UnreadableSourceError as error:
            self.record_failure(url, error)
            raise

        normalized = NormalizedSource(
            source_identity=url_source_identity(url),
            canonical_source_ref=canonicalize_url(url),
            input_type="pdf_url" if ingestion_res.is_direct_pdf else "web_url",
            retrieval_provider=ingestion_res.retrieval_provider,
            routing_decision=ingestion_res.routing_decision,
            artifact_path=str(saved_path),
            artifact_format=artifact_format(saved_path),
            normalized_content_sha256=content_sha256(text),
            original_bytes_sha256=ingestion_res.content_sha256,
            normalized_text_chars=len(text),
            title=ingestion_res.page_title,
            retrieved_at=ingestion_res.retrieval_timestamp,
            warnings=list(ingestion_res.provider_warnings),
        )
        doc_metadata = {
            **normalized.to_metadata(),
            **ingestion_res.provenance_metadata(),
            "source_url": url,
            "original_source_url": url,
            "source_type": "url",
            "is_direct_pdf": ingestion_res.is_direct_pdf,
            "retrieval_timestamp": ingestion_res.retrieval_timestamp.isoformat(),
            "content_sha256": ingestion_res.content_sha256,
            "extracted_text_size": ingestion_res.extracted_text_size,
            "content_classification": ingestion_res.content_classification,
        }

        outcome = self._version_and_process(saved_path, normalized.source_identity, doc_metadata)
        if outcome["skipped"]:
            return {
                "success": True,
                "input_type": "URL",
                "source_url": url,
                "ingestion_status": "Success",
                "pipeline_status": "Complete",
                "gate_status": "No Change",
                "result_message": "No source change detected. Previous and current content hashes are identical.",
                "usable_clinical_content": True,
                "pipeline_reason": "No source change detected. Previous and current content hashes are identical.",
                "document_id": None,
                "source_identifier": normalized.source_identity,
            }

        pipe_result: PipelineResult = outcome["pipeline_result"]
        return {
            **self._result_common(pipe_result, outcome, "Document retrieved from URL", "URL retrieval succeeded"),
            "input_type": "URL",
            "source_url": url,
            "title": ingestion_res.page_title,
            "filename": saved_path.name,
            "saved_path": str(saved_path),
            "sha256_hash": ingestion_res.content_sha256,
            "source_identifier": normalized.source_identity,
            "content_classification": ingestion_res.content_classification,
            "extracted_text_size": ingestion_res.extracted_text_size,
            "resolved_source_url": ingestion_res.resolved_source_url,
            "retrieval_provider": ingestion_res.retrieval_provider,
            "routing_decision": ingestion_res.routing_decision,
            "usable_clinical_content": True,
        }

    # --------------------------------------------------------------------------
    # PDF upload entry point
    # --------------------------------------------------------------------------

    def ingest_pdf_upload(
        self,
        uploaded_name: str,
        uploaded_bytes: bytes,
        existing_source_identifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save, normalize, version and process an uploaded PDF.

        Args:
            uploaded_name: Original filename (used only for safe storage and display).
            uploaded_bytes: PDF bytes.
            existing_source_identifier: Attach this upload as a new version of an existing
                source. When None, the upload starts (or matches) a content-identified source.
        """
        saved_path = save_uploaded_pdf(uploaded_bytes, uploaded_name, self.config.source_dir)
        try:
            text = load_source_text(saved_path)
        except (UnreadableSourceError, ValueError) as error:
            self.record_failure(str(saved_path), error)
            raise

        normalized_hash = content_sha256(text)
        if existing_source_identifier:
            self._require_existing_source(existing_source_identifier)
            identity = existing_source_identifier
        else:
            identity = content_source_identity(normalized_hash)

        normalized = NormalizedSource(
            source_identity=identity,
            canonical_source_ref=f"upload:{identity}",
            input_type="pdf_upload",
            retrieval_provider="Local Upload",
            routing_decision="local_upload",
            artifact_path=str(saved_path),
            artifact_format="pdf",
            normalized_content_sha256=normalized_hash,
            original_bytes_sha256=bytes_sha256(uploaded_bytes),
            normalized_text_chars=len(text),
            title=Path(sanitize_filename(uploaded_name)).stem.replace("_", " ").title(),
            retrieved_at=datetime.now(timezone.utc),
        )
        doc_metadata = {
            **normalized.to_metadata(),
            "source_type": "pdf_upload",
            "original_filename": uploaded_name,
        }

        outcome = self._version_and_process(saved_path, identity, doc_metadata)
        scan_result = outcome["scan_result"]
        if outcome["skipped"]:
            return {
                "success": True,
                "input_type": "PDF",
                "source_url": None,
                "title": normalized.title,
                "filename": saved_path.name,
                "saved_path": str(saved_path),
                "document_id": None,
                "sha256_hash": normalized.original_bytes_sha256,
                "source_identifier": identity,
                "ingestion_status": "Success",
                "pipeline_status": "Complete",
                "gate_status": "No Change",
                "result_message": "No source change detected. Previous and current content hashes are identical.",
                "pipeline_reason": "No source change detected.",
                "scan_result": scan_result,
                "pipeline_result": None,
                "error": None,
            }

        pipe_result: PipelineResult = outcome["pipeline_result"]
        return {
            **self._result_common(pipe_result, outcome, "Document uploaded", "Upload succeeded"),
            "input_type": "PDF",
            "source_url": None,
            "title": normalized.title,
            "filename": saved_path.name,
            "saved_path": str(saved_path),
            "sha256_hash": normalized.original_bytes_sha256,
            "source_identifier": identity,
            "content_classification": "Direct Clinical PDF",
            "extracted_text_size": len(text),
        }

    # --------------------------------------------------------------------------
    # Shared
    # --------------------------------------------------------------------------

    def _require_existing_source(self, source_identifier: str) -> None:
        with self.session_factory() as session:
            exists = (
                session.query(IngestedDocument.id)
                .filter(IngestedDocument.source_identifier == source_identifier)
                .first()
            )
        if not exists:
            raise UnknownSourceError(f"Source '{source_identifier}' does not exist; cannot add a new version to it.")

    def _version_and_process(self, saved_path: Path, identity: str, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        scan_result = self.monitoring_agent.ingest_known_file(
            file_path=saved_path,
            source_identifier=identity,
            doc_metadata=doc_metadata,
        )
        if scan_result.skipped > 0:
            return {"skipped": True, "scan_result": scan_result}

        doc_id = scan_result.ingested_document_ids[0] if scan_result.ingested_document_ids else None
        pipe_result = self.pipeline.process_document(doc_id or saved_path)

        doc_id = doc_id or pipe_result.artifacts.document_id
        doc_status, doc_version = "unknown", "1.0"
        if doc_id:
            with self.session_factory() as session:
                doc = session.get(IngestedDocument, doc_id)
                if doc:
                    doc_status = doc.status
                    doc_version = doc.source_version or doc.document_version
        return {
            "skipped": False,
            "scan_result": scan_result,
            "pipeline_result": pipe_result,
            "document_id": doc_id,
            "document_status": doc_status,
            "source_version": doc_version,
        }

    @staticmethod
    def _result_common(pipe_result: PipelineResult, outcome: Dict[str, Any], ok_prefix: str, fail_prefix: str) -> Dict[str, Any]:
        held_gate = pipe_result.held_gate
        gate_str = held_gate.value if held_gate else None
        if pipe_result.is_held:
            msg = f"{ok_prefix}. " + _GATE_MESSAGES.get(held_gate, f"Processing paused at {gate_str}.")
            if held_gate == HumanGate.G1:
                msg = "Document ingestion complete. " + _GATE_MESSAGES[HumanGate.G1]
            elif held_gate == HumanGate.G3:
                msg = _GATE_MESSAGES[HumanGate.G3]
        elif pipe_result.is_completed:
            msg = f"{ok_prefix} and processed successfully."
        elif pipe_result.is_failed:
            msg = f"{fail_prefix}, but processing failed."
        else:
            msg = f"{ok_prefix}. Status: {pipe_result.status.value}."

        return {
            "success": not pipe_result.is_failed,
            "document_id": outcome["document_id"],
            "source_version": outcome["source_version"],
            "document_status": outcome["document_status"],
            "ingestion_status": "Complete",
            "pipeline_status": pipe_result.status.value.capitalize(),
            "current_stage": pipe_result.current_stage.value if pipe_result.current_stage else "monitoring",
            "gate_status": gate_str,
            "pipeline_reason": pipe_result.reason,
            "result_message": msg,
            "pipeline_result": pipe_result,
            "scan_result": outcome["scan_result"],
            "error": pipe_result.errors[0] if pipe_result.errors else None,
        }
