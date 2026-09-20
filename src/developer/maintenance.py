"""Conservative, opt-in autonomous maintenance.

Maintenance is always off by default and is controlled by a small state file in
``improvement_queue/maintenance.json``. Even when enabled, ``run_once`` only
turns analyzer statistics into suggestion tasks; it never applies changes on
its own. The maintenance interval is configurable and the scheduler polls
slowly (no busy loop), only acting when enabled and due.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path

from developer.improvement_analyzer import ImprovementAnalyzer
from developer.project_tools import PROJECT_ROOT
from developer.task_manager import DevTask, TaskManager

DEFAULT_STATE_FILE = PROJECT_ROOT / "improvement_queue" / "maintenance.json"

DEFAULT_INTERVAL_MINUTES = int(
    os.environ.get("JARVIS_MAINTENANCE_INTERVAL_MINUTES", "60") or 60
)
POLL_SECONDS = 60
MIN_INTERVAL_MINUTES = 5
MAX_INTERVAL_MINUTES = 1440


def _clamp_interval(minutes: int) -> int:
    try:
        value = int(minutes)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_MINUTES
    return min(max(value, MIN_INTERVAL_MINUTES), MAX_INTERVAL_MINUTES)


class MaintenanceController:
    """Persistence gate for the maintenance loop."""

    def __init__(
        self,
        state_file: Path | None = None,
        *,
        interval_minutes: int | None = None,
    ) -> None:
        self._state_file = (state_file or DEFAULT_STATE_FILE).resolve()
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        stored = self._state.get("interval_minutes", DEFAULT_INTERVAL_MINUTES)
        self._interval_minutes = _clamp_interval(
            interval_minutes if interval_minutes is not None else stored
        )

    def _load(self) -> dict:
        defaults = {
            "enabled": False,
            "mode": "suggest",
            "updated_at": "",
            "last_run": "",
            "interval_minutes": DEFAULT_INTERVAL_MINUTES,
        }
        if not self._state_file.exists():
            return dict(defaults)
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return dict(defaults)
        if not isinstance(data, dict):
            return dict(defaults)
        for key, value in defaults.items():
            data.setdefault(key, value)
        return data

    def _save(self) -> None:
        self._state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._state["interval_minutes"] = self._interval_minutes
        self._state_file.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_enabled(self) -> bool:
        return bool(self._state.get("enabled"))

    @property
    def mode(self) -> str:
        return str(self._state.get("mode", "suggest"))

    @property
    def state_file(self) -> Path:
        return self._state_file

    @property
    def interval_minutes(self) -> int:
        return self._interval_minutes

    @property
    def last_run(self) -> str:
        return str(self._state.get("last_run", ""))

    def set_enabled(self, enabled: bool) -> None:
        self._state["enabled"] = bool(enabled)
        self._save()

    def set_interval(self, minutes: int) -> int:
        self._interval_minutes = _clamp_interval(minutes)
        self._save()
        return self._interval_minutes

    def due(self, now_unix: float | None = None) -> bool:
        """Whether an enabled run is due based on the configured interval."""
        if not self.is_enabled():
            return False
        last_run = self._state.get("last_run", "")
        if not last_run:
            return True
        try:
            last_unix = time.mktime(time.strptime(last_run, "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            return True
        now = now_unix or time.time()
        return now - last_unix >= self._interval_minutes * 60

    def run_once(
        self,
        analyzer: ImprovementAnalyzer,
        tasks: TaskManager | None = None,
    ) -> list[DevTask]:
        """Create suggestion tasks from current failure statistics."""
        if not self.is_enabled():
            return []
        manager = tasks or analyzer.tasks
        created = analyzer.run(manager)
        self._state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save()
        return created


class MaintenanceScheduler:
    """Background maintenance loop with a slow poll and a configurable interval."""

    def __init__(
        self,
        controller: MaintenanceController,
        analyzer: ImprovementAnalyzer | None = None,
        tasks: TaskManager | None = None,
    ) -> None:
        self.controller = controller
        self.analyzer = analyzer or ImprovementAnalyzer(tasks=tasks)
        self.tasks = tasks
        self._task: asyncio.Task | None = None

    async def run_tick(self) -> list[DevTask]:
        if self.controller.is_enabled() and self.controller.due():
            return self.controller.run_once(self.analyzer, self.tasks)
        return []

    async def run_loop(self, poll_seconds: float = POLL_SECONDS) -> None:
        while True:
            await asyncio.sleep(poll_seconds)
            with contextlib.suppress(Exception):
                await self.run_tick()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_loop())

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
