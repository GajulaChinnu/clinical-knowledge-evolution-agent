"""CKEA business services package.

Re-exports are resolved lazily: importing any `app.services.<module>` no longer loads
sentence-transformers / chromadb up front (that made every import take over a minute).
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AppConfig": "app.services.config_service",
    "ConfigError": "app.services.config_service",
    "MissingAPIKeyError": "app.services.config_service",
    "load_config": "app.services.config_service",
    "EmbeddingService": "app.services.embeddings",
    "get_embedding_service": "app.services.embeddings",
    "compute_bytes_sha256": "app.services.file_hash",
    "compute_sha256": "app.services.file_hash",
    "SharedLLMClient": "app.services.llm_client",
    "DocumentSection": "app.services.pdf_parser",
    "PageText": "app.services.pdf_parser",
    "detect_document_sections": "app.services.pdf_parser",
    "extract_page_texts": "app.services.pdf_parser",
    "filter_candidate_sections": "app.services.pdf_parser",
    "EmptyProtocolError": "app.services.protocol_index",
    "ProtocolError": "app.services.protocol_index",
    "ProtocolIndexService": "app.services.protocol_index",
    "ProtocolParseError": "app.services.protocol_index",
    "parse_protocol_file": "app.services.protocol_index",
    "VerificationResult": "app.services.source_verifier",
    "verify_recommendation_provenance": "app.services.source_verifier",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'app.services' has no attribute '{name}'")
    return getattr(import_module(module), name)
