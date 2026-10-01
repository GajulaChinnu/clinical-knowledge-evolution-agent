"""Protocol indexing and semantic retrieval service using local ChromaDB and all-MiniLM-L6-v2."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.schemas.protocol import (
    CandidateProtocolSection,
    ProtocolDocument,
    ProtocolIndexingSummary,
    ProtocolSection,
)
from app.services.config_service import AppConfig, load_config
from app.services.embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger("ckea.services.protocol_index")

COLLECTION_NAME = "protocol_sections"


class ProtocolError(Exception):
    """Base exception for protocol-related operations."""
    pass


class ProtocolParseError(ProtocolError):
    """Raised when a protocol file has invalid format or missing required fields."""
    pass


class EmptyProtocolError(ProtocolError):
    """Raised when a protocol file contains no extractable sections."""
    pass


def parse_protocol_file(file_path: Union[str, Path]) -> ProtocolDocument:
    """Deterministically parse a local protocol file (.json or .md/.txt).

    Args:
        file_path: Absolute or relative path to the protocol file.

    Returns:
        ProtocolDocument containing metadata and discrete ProtocolSection objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ProtocolParseError: If file content is invalid or malformed.
        EmptyProtocolError: If the protocol contains no sections.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        raise ProtocolParseError(f"Failed to read protocol file '{path.name}': {e}") from e

    if not content.strip():
        raise EmptyProtocolError(f"Protocol file '{path.name}' is empty.")

    if path.suffix.lower() == ".json":
        return _parse_json_protocol(content, path.name)
    else:
        return _parse_markdown_protocol(content, path.name)


def _parse_json_protocol(content: str, filename: str) -> ProtocolDocument:
    """Parse JSON protocol structure."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ProtocolParseError(f"Invalid JSON in '{filename}': {e}") from e

    if not isinstance(data, dict):
        raise ProtocolParseError(f"Root JSON in '{filename}' must be an object.")

    protocol_id = data.get("protocol_id")
    protocol_version = data.get("protocol_version")
    title = data.get("title")
    raw_sections = data.get("sections")

    if not protocol_id or not str(protocol_id).strip():
        raise ProtocolParseError(f"Missing or empty 'protocol_id' in '{filename}'.")
    if not protocol_version or not str(protocol_version).strip():
        raise ProtocolParseError(f"Missing or empty 'protocol_version' in '{filename}'.")
    if not isinstance(raw_sections, list):
        raise ProtocolParseError(f"Missing or invalid 'sections' list in '{filename}'.")
    if len(raw_sections) == 0:
        raise EmptyProtocolError(f"Protocol file '{filename}' contains an empty 'sections' list.")

    sections: List[ProtocolSection] = []
    for idx, s in enumerate(raw_sections, 1):
        if not isinstance(s, dict):
            raise ProtocolParseError(f"Section {idx} in '{filename}' must be a dictionary.")

        sec_id = s.get("section_id")
        heading = s.get("section_heading")
        text = s.get("section_text")

        if not sec_id or not str(sec_id).strip():
            raise ProtocolParseError(f"Section {idx} in '{filename}' missing 'section_id'.")
        if not heading or not str(heading).strip():
            raise ProtocolParseError(f"Section {idx} in '{filename}' missing 'section_heading'.")
        if not text or not str(text).strip():
            raise ProtocolParseError(f"Section {idx} in '{filename}' missing 'section_text'.")

        sections.append(
            ProtocolSection(
                protocol_id=str(protocol_id).strip(),
                protocol_version=str(protocol_version).strip(),
                section_id=str(sec_id).strip(),
                section_heading=str(heading).strip(),
                section_text=str(text).strip(),
            )
        )

    return ProtocolDocument(
        protocol_id=str(protocol_id).strip(),
        protocol_version=str(protocol_version).strip(),
        title=str(title).strip() if title else None,
        sections=sections,
    )


def _parse_markdown_protocol(content: str, filename: str) -> ProtocolDocument:
    """Parse Markdown / plain text protocol structure using deterministic heading matching."""
    lines = content.splitlines()

    protocol_id: Optional[str] = None
    protocol_version: Optional[str] = None
    title: Optional[str] = None

    # Infer default protocol_id and version from filename if patterned: <protocol_id>_<version>.<ext>
    stem = Path(filename).stem
    if "_" in stem:
        parts = stem.split("_", 1)
        protocol_id = parts[0].strip()
        protocol_version = parts[1].strip()

    sections: List[ProtocolSection] = []
    current_sec_id: Optional[str] = None
    current_heading: Optional[str] = None
    current_text_lines: List[str] = []

    def flush_current_section():
        nonlocal current_sec_id, current_heading, current_text_lines
        if current_heading and current_text_lines:
            text = "\n".join(current_text_lines).strip()
            if text:
                sec_id = current_sec_id or f"SEC-{len(sections) + 1}"
                sections.append(
                    ProtocolSection(
                        protocol_id=protocol_id or "UNKNOWN-PROTOCOL",
                        protocol_version=protocol_version or "v1.0",
                        section_id=sec_id,
                        section_heading=current_heading,
                        section_text=text,
                    )
                )
        current_sec_id = None
        current_heading = None
        current_text_lines = []

    for line in lines:
        stripped = line.strip()

        # Metadata header parsing
        if stripped.lower().startswith("protocol_id:") or stripped.lower().startswith("# protocol:"):
            protocol_id = stripped.split(":", 1)[1].strip()
            continue
        if stripped.lower().startswith("protocol_version:") or stripped.lower().startswith("version:"):
            protocol_version = stripped.split(":", 1)[1].strip()
            continue
        if stripped.lower().startswith("title:") or stripped.lower().startswith("# title:"):
            title = stripped.split(":", 1)[1].strip()
            continue

        # Top-level heading (# ...) could be title or protocol ID if not set
        if stripped.startswith("# ") and not stripped.startswith("## "):
            h_text = stripped[2:].strip()
            if not title:
                title = h_text
            continue

        # Section headings (## ... or ### ...)
        heading_match = re.match(r"^#{2,3}\s+(.*)$", stripped)
        if heading_match:
            flush_current_section()
            raw_h = heading_match.group(1).strip()
            # Check if heading has "SEC-X: Heading" or "Section X: Heading" pattern
            sec_match = re.match(r"^(?:Section\s+)?([A-Za-z0-9_-]+)\s*[:.-]\s*(.+)$", raw_h, re.IGNORECASE)
            if sec_match:
                current_sec_id = sec_match.group(1).strip()
                current_heading = sec_match.group(2).strip()
            else:
                current_sec_id = f"SEC-{len(sections) + 1}"
                current_heading = raw_h
            continue

        if current_heading:
            current_text_lines.append(line)

    flush_current_section()

    if not protocol_id:
        protocol_id = stem
    if not protocol_version:
        protocol_version = "v1.0"

    # Propagate resolved protocol_id & version to sections
    for s in sections:
        object.__setattr__(s, "protocol_id", protocol_id)
        object.__setattr__(s, "protocol_version", protocol_version)

    if not sections:
        raise EmptyProtocolError(f"Protocol file '{filename}' contains no extractable sections.")

    return ProtocolDocument(
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        title=title,
        sections=sections,
    )


class ProtocolIndexService:
    """Manages local embedded ChromaDB persistence, indexing, and semantic retrieval for protocol sections."""

    def __init__(
        self,
        chroma_dir: Optional[Union[str, Path]] = None,
        config: Optional[AppConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chroma_client: Optional[ClientAPI] = None,
    ) -> None:
        self.config = config or load_config()
        self.chroma_dir = Path(chroma_dir or self.config.chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_service = embedding_service or get_embedding_service()
        self._client = chroma_client or chromadb.PersistentClient(path=str(self.chroma_dir))
        self._collection = self._init_collection()

    def _init_collection(self) -> Collection:
        """Initialize or retrieve the protocol sections ChromaDB collection with cosine space."""
        return self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_service.as_chroma_embedding_function(),
        )

    @property
    def collection(self) -> Collection:
        """Active ChromaDB collection."""
        return self._collection

    def index_protocol(self, protocol_doc: ProtocolDocument) -> int:
        """Index sections from a ProtocolDocument into ChromaDB.

        Idempotent: Unchanged sections sharing identical stable IDs are skipped.
        Historical versions are preserved as distinct ChromaDB records.

        Args:
            protocol_doc: Validated ProtocolDocument instance.

        Returns:
            Count of newly indexed sections.
        """
        if not protocol_doc.sections:
            return 0

        # Check existing stable document IDs to ensure idempotency
        all_ids = [s.document_id for s in protocol_doc.sections]
        existing = self.collection.get(ids=all_ids)
        existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()

        new_sections = [s for s in protocol_doc.sections if s.document_id not in existing_ids]

        if not new_sections:
            logger.info(
                "Protocol %s (version %s): all %d section(s) already indexed; skipping.",
                protocol_doc.protocol_id,
                protocol_doc.protocol_version,
                len(protocol_doc.sections),
            )
            return 0

        # Generate vectors locally
        texts_to_embed = [f"{s.section_heading}\n{s.section_text}" for s in new_sections]
        embeddings = self.embedding_service.embed_batch(texts_to_embed)

        ids = [s.document_id for s in new_sections]
        documents = [s.section_text for s in new_sections]
        metadatas: List[Dict[str, Any]] = [
            {
                "protocol_id": s.protocol_id,
                "protocol_version": s.protocol_version,
                "section_id": s.section_id,
                "section_heading": s.section_heading,
                "section_text": s.section_text,
            }
            for s in new_sections
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        logger.info(
            "Protocol %s (version %s): indexed %d new section(s) (total in store: %d).",
            protocol_doc.protocol_id,
            protocol_doc.protocol_version,
            len(new_sections),
            self.collection.count(),
        )
        return len(new_sections)

    def index_directory(self, protocol_dir: Optional[Union[str, Path]] = None) -> ProtocolIndexingSummary:
        """Discover and index all protocol files in the specified or configured directory.

        Args:
            protocol_dir: Path to directory containing protocol files.

        Returns:
            ProtocolIndexingSummary detailing discovered, indexed, skipped, and failed counts.
        """
        target_dir = Path(protocol_dir or self.config.protocol_dir)
        summary = ProtocolIndexingSummary()

        if not target_dir.exists() or not target_dir.is_dir():
            logger.warning("Protocol directory '%s' does not exist or is not a directory.", target_dir)
            return summary

        # Discover candidate files (.json, .md, .txt) excluding hidden files / keep files
        files = [
            p for p in target_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".json", ".md", ".txt"] and not p.name.startswith(".")
        ]
        files.sort(key=lambda p: p.name)

        summary.protocols_discovered = len(files)
        logger.info("Discovered %d protocol file(s) in '%s'.", len(files), target_dir)

        for file_path in files:
            try:
                protocol_doc = parse_protocol_file(file_path)
                total_sections = len(protocol_doc.sections)
                indexed_count = self.index_protocol(protocol_doc)
                skipped_count = total_sections - indexed_count

                summary.sections_indexed += indexed_count
                summary.sections_skipped += skipped_count

            except Exception as e:
                summary.protocols_failed += 1
                err_msg = f"Failed to index protocol file '{file_path.name}': {e}"
                logger.error(err_msg)
                summary.errors.append(err_msg)

        return summary

    def retrieve(
        self,
        recommendation_text: str,
        top_k: int = 3,
        protocol_id: Optional[str] = None,
        protocol_version: Optional[str] = None,
    ) -> List[CandidateProtocolSection]:
        """Retrieve top-k candidate protocol sections matching recommendation text.

        Args:
            recommendation_text: Verbatim clinical recommendation text.
            top_k: Number of candidate sections to retrieve (default: 3).
            protocol_id: Optional filter for a specific protocol.
            protocol_version: Optional filter for a specific protocol version.

        Returns:
            List of CandidateProtocolSection sorted by similarity.
        """
        if not recommendation_text or not recommendation_text.strip():
            return []

        total_count = self.collection.count()
        if total_count == 0:
            return []

        # Bound k by total available items
        k = max(1, min(top_k, total_count))

        # Generate query vector locally
        query_vector = self.embedding_service.embed_text(recommendation_text)

        # Build optional Chroma filter
        where: Optional[Dict[str, Any]] = None
        if protocol_id and protocol_version:
            where = {"$and": [{"protocol_id": protocol_id}, {"protocol_version": protocol_version}]}
        elif protocol_id:
            where = {"protocol_id": protocol_id}
        elif protocol_version:
            where = {"protocol_version": protocol_version}

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB retrieval failed: %s", e)
            raise

        candidates: List[CandidateProtocolSection] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return candidates

        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc_text = results["documents"][0][i] if results["documents"] else ""
            dist = float(results["distances"][0][i]) if results["distances"] else 0.0

            # Cosine distance to cosine similarity: sim = 1.0 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))

            candidates.append(
                CandidateProtocolSection(
                    protocol_id=str(meta.get("protocol_id", "")),
                    protocol_version=str(meta.get("protocol_version", "")),
                    section_id=str(meta.get("section_id", "")),
                    section_heading=str(meta.get("section_heading", "")),
                    section_text=str(meta.get("section_text", doc_text)),
                    similarity=similarity,
                    distance=dist,
                    chroma_id=doc_id,
                )
            )

        return candidates
