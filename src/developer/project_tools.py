"""Sandboxed access to the JARVIS project for self-improvement tasks.

All file operations go through :class:`ProjectAccess`, which confines reads and
writes to the project root, blocks internal/service directories, and refuses to
touch credentials and other secret-shaped files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_BLOCKED_DIRS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "improvement_queue",
        "logs",
        "node_modules",
    }
)

_SECRET_FILE_RE = re.compile(r"^\.env(?:\.|$)", re.IGNORECASE)
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_SECRET_WORDS = ("secret", "credential", "token", "password")
_PATH_TOKEN_RE = re.compile(r"[\w./\\-]+\.[a-zA-Z0-9]{1,6}")

MAX_READ_BYTES = 150_000
MAX_LIST_FILES = 500
MAX_SEARCH_FILE_BYTES = 200_000


class ProjectAccessError(RuntimeError):
    """Raised when a sandboxed path is missing, blocked, or escapes the root."""


def _is_secret_name(name: str) -> bool:
    if _SECRET_FILE_RE.match(name):
        return True
    lowered = name.lower()
    if lowered.endswith(_SECRET_SUFFIXES):
        return True
    stem = Path(name).stem.lower()
    for word in _SECRET_WORDS:
        if stem == word:
            return True
        if stem.startswith(word + "s"):
            return True
        if stem.startswith(word + "_") or stem.startswith(word + "-"):
            return True
    return False


class ProjectAccess:
    """Confined read/search/write helpers over the project root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or PROJECT_ROOT).resolve()

    # ------------------------------------------------------------------ #
    # path safety
    # ------------------------------------------------------------------ #

    def resolve_project_path(self, path: str | os.PathLike) -> Path:
        candidate = Path(self.root, path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ProjectAccessError(
                f"Path escapes the project root: {str(path)!r}"
            ) from exc
        return candidate

    def has_path(self, path: str | os.PathLike) -> bool:
        try:
            return self.resolve_project_path(path).is_file()
        except ProjectAccessError:
            return False

    def _check_readable(self, path: str | os.PathLike) -> Path:
        resolved = self.resolve_project_path(path)
        relative = resolved.relative_to(self.root)
        for part in relative.parts:
            if part in _BLOCKED_DIRS:
                raise ProjectAccessError(
                    f"Path is inside a blocked directory: {str(path)!r}"
                )
        if not resolved.exists():
            raise ProjectAccessError(f"Path does not exist: {str(path)!r}")
        if resolved.is_dir():
            raise ProjectAccessError(f"Path is a directory: {str(path)!r}")
        if _is_secret_name(resolved.name):
            raise ProjectAccessError("Refusing to read a secret-shaped file")
        return resolved

    def _check_writable(self, path: str | os.PathLike) -> Path:
        resolved = self.resolve_project_path(path)
        relative = resolved.relative_to(self.root)
        for part in relative.parts:
            if part in _BLOCKED_DIRS:
                raise ProjectAccessError(
                    f"Path is inside a blocked directory: {str(path)!r}"
                )
        if _is_secret_name(resolved.name):
            raise ProjectAccessError("Refusing to write a secret-shaped file")
        if resolved.is_dir():
            raise ProjectAccessError(f"Path is a directory: {str(path)!r}")
        return resolved

    # ------------------------------------------------------------------ #
    # listing and search
    # ------------------------------------------------------------------ #

    def list_files(self, start: str = ".") -> list[str]:
        start_dir = self.resolve_project_path(start)
        files: list[str] = []
        for current, dirs, names in os.walk(start_dir):
            dirs[:] = [
                d for d in dirs if d not in _BLOCKED_DIRS and not _is_secret_name(d)
            ]
            for name in sorted(names):
                if _is_secret_name(name):
                    continue
                full_path = Path(current, name)
                if not full_path.is_file():
                    continue
                relative = full_path.relative_to(self.root).as_posix()
                files.append(relative)
                if len(files) >= MAX_LIST_FILES:
                    return files
        return files

    def read_file(self, path: str | os.PathLike) -> dict[str, str]:
        resolved = self._check_readable(path)
        relative = resolved.relative_to(self.root).as_posix()
        data = resolved.read_bytes()
        truncated = len(data) > MAX_READ_BYTES
        content = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        if truncated:
            content += f"\n[... truncated at {MAX_READ_BYTES} bytes ...]"
        return {"path": relative, "content": content}

    def search_files(
        self,
        pattern: str,
        *,
        start: str = ".",
        max_results: int = 200,
    ) -> list[dict[str, str | int]]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ProjectAccessError(f"Invalid regex: {exc}") from exc
        matches: list[dict[str, str | int]] = []
        start_dir = self.resolve_project_path(start)
        for current, dirs, names in os.walk(start_dir):
            dirs[:] = [d for d in dirs if d not in _BLOCKED_DIRS]
            for name in sorted(names):
                if _is_secret_name(name):
                    continue
                full_path = Path(current, name)
                if not full_path.is_file():
                    continue
                if full_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                try:
                    lines = full_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                except OSError:
                    continue
                relative = full_path.relative_to(self.root).as_posix()
                for line_number, line in enumerate(lines, start=1):
                    if regex.search(line):
                        matches.append(
                            {"path": relative, "line": line_number, "text": line}
                        )
                        if len(matches) >= max_results:
                            return matches
        return matches

    # ------------------------------------------------------------------ #
    # writes (deterministic, confined)
    # ------------------------------------------------------------------ #

    def write_file(
        self,
        path: str | os.PathLike,
        content: str,
        *,
        overwrite: bool = True,
    ) -> dict[str, str | bool]:
        resolved = self._check_writable(path)
        relative = resolved.relative_to(self.root).as_posix()
        existed = resolved.exists()
        if existed and not overwrite:
            raise ProjectAccessError(f"File already exists: {relative!r}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8", newline="")
        return {"path": relative, "created": not existed, "overwritten": existed}

    def delete_file(self, path: str | os.PathLike) -> dict[str, str | bool]:
        resolved = self._check_writable(path)
        relative = resolved.relative_to(self.root).as_posix()
        if not resolved.exists():
            raise ProjectAccessError(f"File does not exist: {relative!r}")
        resolved.unlink()
        return {"path": relative, "deleted": True}


def extract_path_tokens(description: str) -> list[str]:
    """Pull likely file-path tokens out of a free-text task description."""
    return [token for token in _PATH_TOKEN_RE.findall(description) if token]
