"""Turns failure-log statistics into suggested improvement tasks."""

from __future__ import annotations

from pathlib import Path

from developer.failures import (
    DEFAULT_FAILURE_LOG,
    FailureAnalyzer,
)
from developer.task_manager import DevTask, TaskManager


class ImprovementAnalyzer:
    """Materializes analyzer suggestions as real suggestion tasks."""

    def __init__(
        self,
        thresholds: dict[str, int] | None = None,
        *,
        tasks: TaskManager | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._failures = FailureAnalyzer(thresholds)
        self._tasks = tasks
        self._log_path = log_path or DEFAULT_FAILURE_LOG

    @property
    def tasks(self) -> TaskManager:
        return self._tasks or TaskManager()

    def run(
        self,
        tasks: TaskManager | None = None,
        *,
        log_path: Path | None = None,
    ) -> list[DevTask]:
        manager = tasks or self.tasks
        suggestions = self._failures.analyze(log_path or self._log_path)
        existing = manager.list_tasks()
        created: list[DevTask] = []
        for suggestion in suggestions:
            if any(
                task.title == suggestion["title"] and task.status != "failed"
                for task in existing
            ):
                continue
            task = manager.create_task(
                title=suggestion["title"],
                description=suggestion["description"],
                category=suggestion["category"],
                created_by="analyzer",
            )
            created.append(task)
        return created
