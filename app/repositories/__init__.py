"""CKEA data repositories package."""

from app.repositories.base import BaseRepository, ImmutableEntityError

__all__ = [
    "BaseRepository",
    "ImmutableEntityError",
]
