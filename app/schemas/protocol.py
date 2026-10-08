"""Pydantic schemas for protocol definitions, sections, and semantic retrieval results."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProtocolSection(BaseModel):
    """A discrete clinical protocol section to be indexed into ChromaDB."""

    model_config = ConfigDict(validate_assignment=True)

    protocol_id: str = Field(..., min_length=1, description="Unique identifier for the protocol.")
    protocol_version: str = Field(..., min_length=1, description="Immutable protocol version string.")
    section_id: str = Field(..., min_length=1, description="Identifier for this section within the protocol.")
    section_heading: str = Field(..., min_length=1, description="Section heading or title.")
    section_text: str = Field(..., min_length=1, description="Verbatim text of the protocol section.")

    @property
    def document_id(self) -> str:
        """Deterministic stable Chroma document ID derived from immutable components."""
        return f"{self.protocol_id}__{self.protocol_version}__{self.section_id}"

    @field_validator("protocol_id", "protocol_version", "section_id", "section_heading", "section_text")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()


class ProtocolDocument(BaseModel):
    """A full synthetic protocol document containing multiple sections."""

    model_config = ConfigDict(validate_assignment=True)

    protocol_id: str = Field(..., min_length=1)
    protocol_version: str = Field(..., min_length=1)
    title: Optional[str] = Field(default=None)
    sections: List[ProtocolSection] = Field(default_factory=list)
    # Clinical ownership metadata (taxonomy ids). Optional for legacy protocol files, whose
    # metadata comes from config/taxonomy.yaml (legacy_protocol_metadata) instead.
    department: Optional[str] = Field(default=None)
    pathways: List[str] = Field(default_factory=list)
    treatments: List[str] = Field(default_factory=list)
    owner: Optional[str] = Field(default=None)
    effective_date: Optional[str] = Field(default=None)

    @field_validator("protocol_id", "protocol_version")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()


class CandidateProtocolSection(BaseModel):
    """A candidate protocol section retrieved via semantic similarity search from ChromaDB."""

    model_config = ConfigDict(validate_assignment=True)

    protocol_id: str = Field(..., description="Protocol identifier.")
    protocol_version: str = Field(..., description="Immutable version of the protocol.")
    section_id: str = Field(..., description="Section identifier.")
    section_heading: str = Field(..., description="Heading/title of the candidate section.")
    section_text: str = Field(..., description="Ground truth text of the candidate section.")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score [0.0, 1.0].")
    distance: float = Field(..., ge=0.0, description="Raw ChromaDB vector distance.")
    chroma_id: str = Field(..., description="Deterministic ChromaDB document ID.")


class ProtocolIndexingSummary(BaseModel):
    """Summary metrics of a protocol indexing execution."""

    model_config = ConfigDict(validate_assignment=True)

    protocols_discovered: int = 0
    sections_indexed: int = 0
    sections_skipped: int = 0
    protocols_failed: int = 0
    errors: List[str] = Field(default_factory=list)
