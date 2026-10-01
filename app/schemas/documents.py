"""Pydantic schemas for Ingested Documents and Ingestion Failures."""

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentStatus(str, Enum):
    """Controlled lifecycle states for ingested documents."""
    DISCOVERED = "discovered"
    PARSED = "parsed"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    HELD = "held"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IngestedDocumentBase(BaseModel):
    """Base schema for IngestedDocument records."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    source_identifier: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    sha256_hash: str = Field(..., min_length=64, max_length=64)
    document_version: str = Field(default="1.0", min_length=1)
    source_version: str = Field(default="1.0", min_length=1)
    parser_version: str = Field(default="1.0", min_length=1)
    pipeline_version: str = Field(default="1.0", min_length=1)
    status: DocumentStatus = Field(default=DocumentStatus.DISCOVERED)
    doc_metadata: Optional[Dict[str, Any]] = Field(default=None)
    retry_count: int = Field(default=0, ge=0)
    error_message: Optional[str] = Field(default=None)
    schema_version: str = Field(default="1.0", min_length=1)

    @field_validator("sha256_hash")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if not re.fullmatch(r"[a-fA-F0-9]{64}", v):
            raise ValueError("sha256_hash must be a 64-character hexadecimal string")
        return v.lower()


class IngestedDocumentCreate(IngestedDocumentBase):
    """Schema for creating a new IngestedDocument record."""
    ingest_timestamp: datetime = Field(default_factory=_utc_now)


class IngestedDocumentRead(IngestedDocumentBase):
    """Schema for reading an IngestedDocument record."""
    id: str
    ingest_timestamp: datetime
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)


class IngestionFailureBase(BaseModel):
    """Base schema for IngestionFailure records."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    source_path: str = Field(..., min_length=1)
    error_category: str = Field(..., min_length=1)
    error_message: str = Field(..., min_length=1)
    retry_count: int = Field(default=0, ge=0)
    retry_status: str = Field(default="pending")
    operator_status: str = Field(default="unresolved")
    schema_version: str = Field(default="1.0", min_length=1)


class IngestionFailureCreate(IngestionFailureBase):
    """Schema for creating an IngestionFailure record."""
    timestamp: datetime = Field(default_factory=_utc_now)


class IngestionFailureRead(IngestionFailureBase):
    """Schema for reading an IngestionFailure record."""
    id: str
    timestamp: datetime
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)
