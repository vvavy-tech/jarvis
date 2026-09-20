"""Durable long-term memory backed by :mod:`sqlite3` (Python stdlib).

Design notes
------------
* No external dependencies: ``sqlite3`` ships with CPython.
* Every synchronous SQLite operation runs inside ``asyncio.to_thread`` so the
  realtime voice loop is never blocked.
* Connections are opened fresh per operation and closed by context manager, so
  a connection is never shared across event loops.
* The schema is versioned via ``PRAGMA user_version`` (v1 today), leaving a
  clean path for future migrations.
* The store is policy-free: all policy decisions live in :mod:`memory.policy`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.base import MemoryEntry, MemoryProvider

logger = logging.getLogger("agent")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LocalSQLiteMemoryProvider(MemoryProvider):
    """A durable, policy-free SQLite memory store."""

    name = "local_sqlite"
    persistent = True
    semantic_search = False

    _SCHEMA_VERSION = 1

    def __init__(self, *, path: str | None = None) -> None:
        self._path = path
        self._available = path is not None

    # ------------------------------------------------------------------ #
    # core provider surface
    # ------------------------------------------------------------------ #

    @property
    def db_path(self) -> str | None:
        return self._path

    def is_available(self) -> bool:
        return self._available

    async def initialize(self) -> None:
        if not self._path:
            self._available = False
            return

        def _init() -> None:
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with self._connect():
                pass

        try:
            await asyncio.to_thread(_init)
        except Exception:
            self._available = False
            return
        self._available = True

    async def close(self) -> None:
        """Idempotent no-op: connections are per-operation, never held."""

    async def health_check(self) -> dict[str, Any]:
        def _probe() -> int:
            with self._connect() as conn:
                return conn.execute("SELECT count(*) FROM memories").fetchone()[0]

        try:
            await asyncio.to_thread(_probe)
        except Exception as exc:
            return {
                "name": self.name,
                "ok": False,
                "status": "error",
                "detail": "memory store unavailable",
                "error": type(exc).__name__,
            }
        return {
            "name": self.name,
            "ok": True,
            "status": "healthy",
            "detail": "memory store connected",
            "persistent": self.persistent,
            "semantic_search": self.semantic_search,
            "db_path": self._path,
        }

    async def store(self, entry: MemoryEntry) -> MemoryEntry:
        if not entry.content or not entry.content.strip():
            raise ValueError("refusing empty memory content")
        now = _now_iso()
        if not entry.id:
            entry.id = uuid.uuid4().hex
        if not entry.created_at:
            entry.created_at = now
        entry.updated_at = now

        def _insert() -> None:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO memories (id, content, category, project, "
                    "importance, source, confidence, tags, created_at, "
                    "updated_at, last_accessed, access_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry.id,
                        entry.content,
                        entry.category,
                        entry.project,
                        entry.importance,
                        entry.source,
                        entry.confidence,
                        ",".join(entry.tags),
                        entry.created_at,
                        entry.updated_at,
                        entry.last_accessed,
                        entry.access_count,
                    ),
                )

        logger.info(
            "[MEMORY-DEBUG] provider store requested: id=%s content=%r category=%s",
            entry.id,
            entry.content,
            entry.category,
        )
        await asyncio.to_thread(_insert)
        logger.info(
            "[MEMORY-DEBUG] provider store completed: id=%s content=%r",
            entry.id,
            entry.content,
        )
        return entry

    async def get(self, memory_id: str) -> MemoryEntry | None:
        def _fetch() -> MemoryEntry | None:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
            return self._from_row(row) if row else None

        entry = await asyncio.to_thread(_fetch)
        if entry is not None:
            await self._bump_access(memory_id)
        return entry

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        category: str | None = None,
        project: str | None = None,
    ) -> list[MemoryEntry]:
        terms = query.lower().split()
        if not terms:
            return []

        def _select() -> list[MemoryEntry]:
            clauses = ["content LIKE ?"]
            params: list[str] = [f"%{terms[0]}%"]
            for term in terms[1:]:
                clauses.append("content LIKE ?")
                params.append(f"%{term}%")
            if category:
                clauses.append("category = ?")
                params.append(category)
            if project:
                clauses.append("project = ?")
                params.append(project)
            sql = (
                f"SELECT * FROM memories WHERE {' OR '.join(clauses)} "
                "ORDER BY updated_at DESC LIMIT ?"
            )
            params.append(str(int(limit)))
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [self._from_row(row) for row in rows]

        return await asyncio.to_thread(_select)

    async def update(
        self, memory_id: str, *, content: str | None = None, **fields: Any
    ) -> MemoryEntry | None:
        updates = dict(fields)
        if content is not None:
            updates["content"] = content
        if not updates:
            return await self.get(memory_id)

        def _apply() -> MemoryEntry | None:
            sets = "updated_at = ?"
            params: list[Any] = [_now_iso()]
            allowed = {
                "content",
                "category",
                "project",
                "importance",
                "source",
                "confidence",
                "last_accessed",
            }
            for key, value in updates.items():
                if key not in allowed:
                    continue
                if key == "tags":
                    value = ",".join(value)
                sets += f", {key} = ?"
                params.append(value)
            params.append(memory_id)
            with self._connect() as conn:
                cur = conn.execute(f"UPDATE memories SET {sets} WHERE id = ?", params)
                if cur.rowcount == 0:
                    return None
                row = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
            return self._from_row(row) if row else None

        return await asyncio.to_thread(_apply)

    async def delete(self, memory_id: str) -> bool:
        def _remove() -> bool:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cur.rowcount > 0

        return await asyncio.to_thread(_remove)

    async def find_exact(self, content: str) -> MemoryEntry | None:
        def _find() -> MemoryEntry | None:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM memories WHERE lower(content) = lower(?) "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (content,),
                ).fetchone()
            return self._from_row(row) if row else None

        return await asyncio.to_thread(_find)

    async def count(self) -> int:
        def _count() -> int:
            with self._connect() as conn:
                return conn.execute("SELECT count(*) FROM memories").fetchone()[0]

        return await asyncio.to_thread(_count)

    async def list_all(
        self,
        *,
        category: str | None = None,
        project: str | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        def _list() -> list[MemoryEntry]:
            clauses = ["1 = 1"]
            params: list[str] = []
            if category:
                clauses.append("category = ?")
                params.append(category)
            if project:
                clauses.append("project = ?")
                params.append(project)
            params.append(str(int(limit)))
            sql = (
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
                "ORDER BY updated_at DESC LIMIT ?"
            )
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [self._from_row(row) for row in rows]

        return await asyncio.to_thread(_list)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        if not self._path:
            raise OSError("no database path configured")
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < self._SCHEMA_VERSION:
            self._create_schema(conn)
            conn.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
            conn.commit()
        return conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id TEXT PRIMARY KEY, "
            "content TEXT NOT NULL, "
            "category TEXT NOT NULL DEFAULT 'general', "
            "project TEXT, "
            "importance TEXT NOT NULL DEFAULT 'low', "
            "source TEXT NOT NULL DEFAULT 'voice', "
            "confidence TEXT NOT NULL DEFAULT 'high', "
            "tags TEXT NOT NULL DEFAULT '', "
            "created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, "
            "last_accessed TEXT, "
            "access_count INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories "
            "(updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category)"
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryEntry:
        tags = tuple(t for t in row["tags"].split(",") if t)
        return MemoryEntry(
            content=row["content"] or "",
            category=row["category"] or "general",
            id=row["id"] or "",
            project=row["project"],
            importance=row["importance"] or "low",
            source=row["source"] or "voice",
            confidence=row["confidence"] or "high",
            tags=tags,
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            last_accessed=row["last_accessed"] or "",
            access_count=int(row["access_count"] or 0),
        )

    async def _bump_access(self, memory_id: str) -> None:
        def _bump() -> None:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE memories SET last_accessed = ?, "
                        "access_count = access_count + 1 WHERE id = ?",
                        (_now_iso(), memory_id),
                    )
            except Exception:
                return

        await asyncio.to_thread(_bump)
