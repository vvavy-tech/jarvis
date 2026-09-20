"""Git operations for development tasks.

Every task runs on its own ``jarvis-dev/<task_id>`` branch. Commits only ever
stage the exact files a task changed; merges back to ``main`` happen with
``--no-ff`` so they can be reverted cleanly.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from developer.project_tools import PROJECT_ROOT

BRANCH_PREFIX = "jarvis-dev/"

_TASK_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")

_HIGH_IMPACT_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^pyproject\.toml$", re.I),
        "project dependencies and build configuration",
    ),
    (re.compile(r"^uv\.lock$", re.I), "dependency lockfile"),
    (re.compile(r"^requirements[^/\\]*\.txt$", re.I), "dependency manifest"),
    (re.compile(r"(^|[\\/])src[\\/]agent\.py$", re.I), "core agent boot"),
    (
        re.compile(r"realtime|voice|inference|stt|tts|synthesi", re.I),
        "voice, model, or inference pipeline",
    ),
    (
        re.compile(r"(^|[\\/])(auth|login|credential|secret|token|password)", re.I),
        "credentials or authentication",
    ),
    (
        re.compile(r"payment|billing|stripe|checkout|subscription|purchase", re.I),
        "financial logic",
    ),
    (re.compile(r"(^|[\\/])(delete|remove|drop)", re.I), "destructive behavior"),
    (
        re.compile(r"security|firewall|permission|sudo|chmod|selinux|sandbox", re.I),
        "security or permissions",
    ),
    (re.compile(r"\.(pem|key|p12|pfx|crt)$", re.I), "credential material"),
]


class GitError(RuntimeError):
    """Raised when a git command fails or is unavailable."""


def task_branch(task_id: str) -> str:
    if not _TASK_ID_RE.match(task_id):
        raise GitError(f"Invalid task id: {task_id!r}")
    return f"{BRANCH_PREFIX}{task_id}"


class GitManager:
    """Thin, safe wrapper around git for a single project root."""

    def __init__(self, root: Path | None = None, timeout_s: float = 60.0) -> None:
        self.root = (root or PROJECT_ROOT).resolve()
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ #
    # low-level
    # ------------------------------------------------------------------ #

    def _git(
        self,
        *args: str,
        check: bool = True,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s or self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise GitError("git is not available on this system") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitError(
                f"git {' '.join(shlex.quote(a) for a in args)} timed out"
            ) from exc
        if check and result.returncode != 0:
            raise GitError(
                result.stderr.strip()
                or f"git {' '.join(args)} exited {result.returncode}"
            )
        return result

    def available(self) -> bool:
        try:
            self._git("--version", timeout_s=10.0)
            return True
        except GitError:
            return False

    def is_repo(self) -> bool:
        result = self._git("rev-parse", "--is-inside-work-tree", check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"

    # ------------------------------------------------------------------ #
    # read-only state
    # ------------------------------------------------------------------ #

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def head_commit(self, ref: str = "HEAD") -> str:
        return self._git("rev-parse", ref).stdout.strip()

    def branch_exists(self, name: str) -> bool:
        result = self._git("branch", "--list", name, check=False)
        return bool(result.stdout.strip())

    def status_porcelain(self) -> str:
        result = self._git("status", "--porcelain")
        return result.stdout

    def tracked_files(self) -> set[str]:
        result = self._git("ls-files")
        return {line.replace("\\", "/") for line in result.stdout.splitlines() if line}

    def diff(
        self,
        base: str | None = None,
        *,
        paths: list[str] | None = None,
        max_chars: int = 12_000,
    ) -> str:
        args = ["diff"]
        if base:
            args.append(base)
        if paths:
            args += ["--", *[p.replace("\\", "/") for p in paths]]
        output = self._git(*args).stdout[:max_chars]
        if len(output) >= max_chars:
            output += "\n[... diff truncated ...]"
        return output

    # ------------------------------------------------------------------ #
    # branch lifecycle
    # ------------------------------------------------------------------ #

    def create_task_branch(self, task_id: str) -> str:
        branch = task_branch(task_id)
        if not self.branch_exists(branch):
            self._git("checkout", "-b", branch)
        else:
            self._git("checkout", branch)
        return branch

    def checkout(self, ref: str) -> None:
        self._git("checkout", ref)

    def delete_branch(self, task_id: str) -> None:
        branch = task_branch(task_id)
        if not self.branch_exists(branch):
            return
        if self.current_branch() == branch:
            self._git("checkout", "main")
        self._git("branch", "-D", branch)

    # ------------------------------------------------------------------ #
    # commits and merges
    # ------------------------------------------------------------------ #

    def commit_paths(self, message: str, paths: list[str]) -> str:
        if not paths:
            raise GitError("Nothing to commit: no files were changed")
        safe_paths = [path.replace("\\", "/") for path in paths]
        self._git("add", "--", *safe_paths)
        self._git("commit", "-m", message)
        return self.head_commit()

    def is_merged(self, task_id: str) -> bool:
        branch = task_branch(task_id)
        if not self.branch_exists(branch):
            return True
        result = self._git(
            "log",
            "--merges",
            "--oneline",
            "--grep",
            re.escape(branch),
            "--max-count=1",
            check=False,
        )
        return bool(result.stdout.strip())

    def merge_main(self, task_id: str, message: str) -> str:
        self.checkout("main")
        branch = task_branch(task_id)
        self._git("merge", "--no-ff", branch, "-m", message)
        return self.head_commit()

    def revert_merge(self, merge_commit: str) -> None:
        self._git("revert", "-m", "1", merge_commit, "--no-edit")

    def restore_from_index(self, paths: list[str]) -> None:
        if not paths:
            return
        self._git("checkout", "--", *[p.replace("\\", "/") for p in paths])

    # ------------------------------------------------------------------ #
    # high-impact analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def high_impact_report(paths: list[str]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for path in paths:
            normalized = path.replace("\\", "/")
            for pattern, label in _HIGH_IMPACT_RULES:
                if pattern.search(normalized):
                    reasons.append(f"{normalized}: {label}")
        return (bool(reasons), reasons)

    @staticmethod
    def dirty_paths_from_porcelain(porcelain: str) -> set[str]:
        dirty: set[str] = set()
        for line in porcelain.splitlines():
            if len(line) < 4:
                continue
            status, path = line[:2].strip(), line[3:].strip()
            if status == "??":
                continue
            dirty.add(path.replace("\\", "/"))
        return dirty
