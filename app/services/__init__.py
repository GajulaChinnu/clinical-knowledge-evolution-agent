"""CKEA business services package."""

from app.services.config_service import (
    AppConfig,
    ConfigError,
    MissingAPIKeyError,
    load_config,
)
from app.services.embeddings import (
    EmbeddingService,
    get_embedding_service,
)
from app.services.file_hash import compute_bytes_sha256, compute_sha256
from app.services.llm_client import SharedLLMClient
from app.services.pdf_parser import (
    DocumentSection,
    PageText,
    detect_document_sections,
    extract_page_texts,
    filter_candidate_sections,
)
from app.services.protocol_index import (
    EmptyProtocolError,
    ProtocolError,
    ProtocolIndexService,
    ProtocolParseError,
    parse_protocol_file,
)
from app.services.source_verifier import (
    VerificationResult,
    verify_recommendation_provenance,
)

__all__ = [
    "AppConfig",
    "ConfigError",
    "MissingAPIKeyError",
    "load_config",
    "compute_sha256",
    "compute_bytes_sha256",
    "SharedLLMClient",
    "PageText",
    "DocumentSection",
    "extract_page_texts",
    "detect_document_sections",
    "filter_candidate_sections",
    "VerificationResult",
    "verify_recommendation_provenance",
    "EmbeddingService",
    "get_embedding_service",
    "ProtocolIndexService",
    "parse_protocol_file",
    "ProtocolError",
    "ProtocolParseError",
    "EmptyProtocolError",
]
