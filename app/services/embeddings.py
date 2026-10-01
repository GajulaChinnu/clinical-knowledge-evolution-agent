"""Local embedding service using sentence-transformers and all-MiniLM-L6-v2."""

import logging
from typing import Any, List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("ckea.services.embeddings")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingService:
    """Thread-safe singleton/cached embedding service for all-MiniLM-L6-v2."""

    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name

    def _ensure_model_loaded(self) -> SentenceTransformer:
        """Lazily load and cache the SentenceTransformer model on first invocation."""
        if EmbeddingService._model is None:
            logger.info("Initializing sentence-transformers model '%s' locally...", self.model_name)
            EmbeddingService._model = SentenceTransformer(self.model_name)
            logger.info("Model '%s' initialized successfully (dimension=%d).", self.model_name, EMBEDDING_DIMENSION)
        return EmbeddingService._model

    @property
    def model(self) -> SentenceTransformer:
        return self._ensure_model_loaded()

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a 384-dimensional normalized vector."""
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIMENSION
        vector = self.model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into a list of 384-dimensional normalized vectors."""
        if not texts:
            return []
        cleaned = [t.strip() if t and t.strip() else "" for t in texts]
        vectors = self.model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    def as_chroma_embedding_function(self) -> Any:
        """Return a ChromaDB-compatible embedding function using this service."""
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

        service = self

        class _ChromaLocalEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self) -> None:
                pass

            def name(self) -> str:
                return "all-MiniLM-L6-v2"

            def get_config(self) -> dict:
                return {"model_name": service.model_name}

            def __call__(self, input: Documents) -> Embeddings:
                return service.embed_batch(list(input))

        return _ChromaLocalEmbeddingFunction()


_global_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingService:
    """Return a shared singleton instance of EmbeddingService to avoid redundant model loads."""
    global _global_embedding_service
    if _global_embedding_service is None or _global_embedding_service.model_name != model_name:
        _global_embedding_service = EmbeddingService(model_name=model_name)
    return _global_embedding_service
