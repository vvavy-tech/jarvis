"""Runtime configuration for persistent memory.

The database path is resolved against the stable project/application root
(``PROJECT_ROOT``), never against the current working directory, so that
write and recall across full process restarts always target the exact same
SQLite file no matter where the agent process is launched from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MEMORY_DIR = PROJECT_ROOT / "data"
DEFAULT_MEMORY_DB = DEFAULT_MEMORY_DIR / "jarvis_memory.db"


def default_memory_db_path() -> str:
    """Absolute path of the runtime memory database, launch-dir independent."""
    return str(DEFAULT_MEMORY_DB)


__all__ = [
    "DEFAULT_MEMORY_DB",
    "DEFAULT_MEMORY_DIR",
    "PROJECT_ROOT",
    "default_memory_db_path",
]
