"""Minimal generic repository helpers for CKEA database operations."""

from typing import Generic, List, Optional, Type, TypeVar
from sqlalchemy.orm import Session

from app.models.entities import AuditLog, ImmutableEntityError

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Generic repository providing basic CRUD operations for SQLAlchemy models."""

    def __init__(self, session: Session, model_cls: Type[T]):
        self.session = session
        self.model_cls = model_cls

    def add(self, entity: T) -> T:
        """Add a new entity to the session and flush."""
        self.session.add(entity)
        self.session.flush()
        return entity

    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Retrieve an entity by its UUID string."""
        return self.session.get(self.model_cls, entity_id)

    def list_all(self) -> List[T]:
        """List all records of this entity type."""
        return self.session.query(self.model_cls).all()

    def delete(self, entity: T) -> None:
        """Mark an entity for deletion and flush. AuditLog records cannot be deleted."""
        if isinstance(entity, AuditLog) or self.model_cls is AuditLog:
            raise ImmutableEntityError("AuditLog records are append-only and cannot be deleted.")
        self.session.delete(entity)
        self.session.flush()
