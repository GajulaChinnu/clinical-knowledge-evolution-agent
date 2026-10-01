"""Pydantic schemas for AuditLog and Notification entities."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AuditLogBase(BaseModel):
    """Base schema for AuditLog records."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    entity_id: str
    entity_type: str = Field(..., min_length=1)
    previous_status: Optional[str] = Field(default=None)
    new_status: str = Field(..., min_length=1)
    actor: str = Field(..., min_length=1)
    reason: Optional[str] = Field(default=None)
    audit_metadata: Optional[Dict[str, Any]] = Field(default=None)
    schema_version: str = Field(default="1.0", min_length=1)

    @field_validator("entity_id")
    @classmethod
    def _validate_entity_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)


class AuditLogCreate(AuditLogBase):
    """Schema for appending an audit log entry."""
    timestamp: datetime = Field(default_factory=_utc_now)


class AuditLogRead(AuditLogBase):
    """Schema for reading an audit log entry."""
    id: str
    timestamp: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)


class NotificationBase(BaseModel):
    """Base schema for Notification records."""
    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    related_entity_id: Optional[str] = Field(default=None)
    related_entity_type: Optional[str] = Field(default=None)
    notification_type: str = Field(..., min_length=1)
    recipient: str = Field(..., min_length=1)
    payload: Optional[Dict[str, Any]] = Field(default=None)
    delivery_status: str = Field(default="pending")
    schema_version: str = Field(default="1.0", min_length=1)


class NotificationCreate(NotificationBase):
    """Schema for creating a Notification."""
    created_at: datetime = Field(default_factory=_utc_now)


class NotificationRead(NotificationBase):
    """Schema for reading a Notification."""
    id: str
    created_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        uuid.UUID(str(v))
        return str(v)
