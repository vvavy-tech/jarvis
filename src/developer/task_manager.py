"""Development task storage.

Tasks live in ``improvement_queue/tasks/`` as one JSON document per task. The
queue is service state, not source code, and is Git-ignored.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from developer.project_tools import PROJECT_ROOT

DEFAULT_QUEUE_ROOT = PROJECT_ROOT / "improvement_queue"

TASK_STATUSES = frozenset(
    {
        "suggestion",
        "queued",
        "in_progress",
        "needs_approval",
        "completed",
        "rejected",
        "cancelled",
        "rolled_back",
        "failed",
    }
)

_PENDING_STATUSES = frozenset({"suggestion", "queued", "in_progress", "needs_approval"})

_TASK_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")


class TaskNotFoundError(KeyError):
    """Raised when a task id does not exist in the queue."""


class TaskActionError(RuntimeError):
    """Raised for invalid task state transitions."""


@dataclass
class DevTask:
    id: str
    title: str
    description: str = ""
    status: str = "suggestion"
    category: str = "manual"
    created_by: str = "user"
    created_at: str = ""
    updated_at: str = ""
    branch: str | None = None
    base_commit: str | None = None
    baseline_status: str = ""
    files: list[str] = field(default_factory=list)
    high_impact: bool = False
    caution: bool = False
    next_action: str | None = None
    notes: str = ""
    summary: str = ""
    merge_commit: str | None = None
    proposal: dict[str, Any] | None = None
    repairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DevTask:
        known = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in known}
        task = cls(**clean)
        created_at = data.get("created_at")
        if created_at:
            task.created_at = created_at
        return task


def _now_stamp() -> str:
    micros = int(time.monotonic() * 1_000_000) % 1_000_000
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S')}.{micros:06d}"


class TaskManager:
    """Persists and lists development tasks as JSON files."""

    def __init__(self, queue_root: Path | None = None) -> None:
        self.queue_root = (queue_root or DEFAULT_QUEUE_ROOT).resolve()
        self.tasks_dir = self.queue_root / "tasks"
        self.backups_dir = self.queue_root / "backups"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def new_task_id(self) -> str:
        return f"dev{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def _task_path(self, task_id: str) -> Path:
        if not _TASK_ID_RE.match(task_id):
            raise TaskNotFoundError(f"Invalid task id: {task_id!r}")
        return self.tasks_dir / f"{task_id}.json"

    def create_task(
        self,
        *,
        title: str,
        description: str,
        category: str = "manual",
        created_by: str = "user",
        status: str = "suggestion",
    ) -> DevTask:
        now = _now_stamp()
        if status not in TASK_STATUSES:
            raise TaskActionError(f"Unknown task status: {status!r}")
        task = DevTask(
            id=self.new_task_id(),
            title=title,
            description=description,
            category=category,
            created_by=created_by,
            status=status,
            created_at=now,
            updated_at=now,
        )
        self.save(task)
        return task

    def save(self, task: DevTask) -> None:
        if task.status not in TASK_STATUSES:
            raise TaskActionError(f"Unknown task status: {task.status!r}")
        if not _TASK_ID_RE.match(task.id):
            raise TaskActionError(f"Invalid task id: {task.id!r}")
        task.updated_at = _now_stamp()
        self._task_path(task.id).write_text(
            json.dumps(task.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, task_id: str) -> DevTask:
        path = self._task_path(task_id)
        if not path.exists():
            raise TaskNotFoundError(f"Task not found: {task_id!r}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return DevTask.from_dict(data)

    def delete(self, task_id: str) -> None:
        path = self._task_path(task_id)
        if path.exists():
            path.unlink()

    def list_tasks(self, status: str | None = None) -> list[DevTask]:
        tasks: list[DevTask] = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            try:
                task = DevTask.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if status is None or task.status == status:
                tasks.append(task)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def latest(self) -> DevTask | None:
        tasks = self.list_tasks()
        return tasks[0] if tasks else None

    def set_status(self, task: DevTask, status: str, notes: str = "") -> DevTask:
        if status not in TASK_STATUSES:
            raise TaskActionError(f"Unknown task status: {status!r}")
        task.status = status
        if notes:
            task.notes = notes
        self.save(task)
        return task
