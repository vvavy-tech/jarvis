"""Long-term memory for JARVIS (Phase 1, LOCAL ONLY).

This package provides an abstract :class:`MemoryProvider` and a concrete,
durable :class:`LocalSQLiteMemoryProvider` backed by the Python standard
library ``sqlite3``. All synchronous database work is dispatched through
``asyncio.to_thread`` so the realtime voice loop is never blocked.

Policy (what may be stored, secret protection, duplication) lives in
:mod:`memory.policy` — never inside a provider. The provider is a dumb,
bounded store; the policy decides.
"""

from memory.auto import (
    ACTION_RECALL,
    ACTION_SKIP,
    ACTION_STORE,
    AutoMemoryController,
    AutoMemoryPipeline,
    AutoMemoryResult,
    auto_recall_query,
    classify_candidate,
    clean_candidate,
    stage1_skip_reason,
)
from memory.base import MemoryEntry, MemoryProvider, MemoryStoreResult
from memory.config import (
    DEFAULT_MEMORY_DB,
    PROJECT_ROOT,
    default_memory_db_path,
)
from memory.local_sqlite import LocalSQLiteMemoryProvider
from memory.manager import MemoryManager
from memory.policy import MemoryDecision, MemoryPolicy
from memory.tools import MemoryIntegration

__all__ = [
    "ACTION_RECALL",
    "ACTION_SKIP",
    "ACTION_STORE",
    "DEFAULT_MEMORY_DB",
    "PROJECT_ROOT",
    "AutoMemoryController",
    "AutoMemoryPipeline",
    "AutoMemoryResult",
    "LocalSQLiteMemoryProvider",
    "MemoryDecision",
    "MemoryEntry",
    "MemoryIntegration",
    "MemoryManager",
    "MemoryPolicy",
    "MemoryProvider",
    "MemoryStoreResult",
    "auto_recall_query",
    "classify_candidate",
    "clean_candidate",
    "default_memory_db_path",
    "stage1_skip_reason",
]
