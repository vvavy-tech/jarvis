"""Core memory models and the provider abstraction.

The provider is deliberately policy-free: it stores, retrieves, searches,
updates, and forgets records. Whether a record *may* be stored, how it is
deduplicated, and what secrets are rejected is the job of
:class:`memory.policy.MemoryPolicy`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

VALID_CATEGORIES = (
    "preference",
    "project",
    "business",
    "person",
    "decision",
    "workflow",
    "technical",
    "integration",
    "general",
)


@dataclass
class MemoryEntry:
    """One durable memory record (Phase 1: local SQLite)."""

    content: str
    category: str = "general"
    id: str = ""
    project: str | None = None
    importance: str = "low"
    source: str = "voice"
    confidence: str = "high"
    tags: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""
    updated_at: str = ""
    last_accessed: str = ""
    access_count: int = 0

    @property
    def is_valid(self) -> bool:
        return bool(self.content and self.content.strip())


@dataclass(frozen=True)
class MemoryStoreResult:
    """Outcome of one policy-aware store attempt."""

    accepted: bool
    entry: MemoryEntry | None = None
    message: str = ""

    @property
    def rejected(self) -> bool:
        return not self.accepted


class MemoryProvider(abc.ABC):
    """Durable long-term memory store. Policy-free by design."""

    name: str = "memory.base"
    persistent: bool = True
    semantic_search: bool = False

    @property
    @abc.abstractmethod
    def db_path(self) -> str | None:
        """The on-disk location of this store, if any."""

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Create the store and its schema (idempotent)."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the store cleanly (idempotent, never raises)."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether the store is usable right now."""

    @abc.abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """A short, safe health probe; never raises."""

    @abc.abstractmethod
    async def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Persist a record; returns it with an assigned id."""

    @abc.abstractmethod
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Fetch one record by id, or None."""

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        category: str | None = None,
        project: str | None = None,
    ) -> list[MemoryEntry]:
        """Text search over content, newest first, bounded by ``limit``."""

    @abc.abstractmethod
    async def update(
        self, memory_id: str, *, content: str | None = None, **fields: Any
    ) -> MemoryEntry | None:
        """Update fields of one record; returns the refreshed record."""

    @abc.abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Remove one record; True if it existed."""

    @abc.abstractmethod
    async def find_exact(self, content: str) -> MemoryEntry | None:
        """Return the record whose stored content matches exactly, if any."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Total number of records."""

    @abc.abstractmethod
    async def list_all(
        self,
        *,
        category: str | None = None,
        project: str | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Recent records, optionally filtered, bounded by ``limit``."""
