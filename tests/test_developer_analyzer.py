"""Tests for the improvement analyzer and the maintenance controller."""

import asyncio

from developer.improvement_analyzer import ImprovementAnalyzer
from developer.maintenance import (
    MaintenanceController,
    MaintenanceScheduler,
)
from developer.task_manager import TaskManager


def _seed_failures(log_path, tool, reason, count):
    from developer.failures import FailureLog

    log = FailureLog(log_path)
    for _ in range(count):
        log.append(tool=tool, reason=reason)


def _analyzer(log_path, tmp_path, thresholds=None):
    tasks = TaskManager(tmp_path / "improvement_queue")
    return tasks, ImprovementAnalyzer(thresholds, tasks=tasks, log_path=log_path)


class TestImprovementAnalyzer:
    def test_creates_suggestion_tasks(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        created = analyzer.run(tasks)
        assert len(created) == 1
        assert created[0].status == "suggestion"
        assert created[0].category == "browser.click"
        assert created[0].created_by == "analyzer"

    def test_no_duplicates_for_active_suggestions(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        first = analyzer.run(tasks)
        second = analyzer.run(tasks)
        assert len(first) == 1
        assert second == []

    def test_recreates_after_suggestion_closed(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        created = analyzer.run(tasks)
        tasks.set_status(created[0], "rejected")
        assert analyzer.run(tasks) == []  # rejected still counts as handled

    def test_resuggests_after_failed_attempt(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        created = analyzer.run(tasks)
        tasks.set_status(created[0], "failed")
        recreated = analyzer.run(tasks)
        assert len(recreated) == 1


class TestMaintenanceController:
    def test_disabled_by_default(self, tmp_path):
        controller = MaintenanceController(
            tmp_path / "improvement_queue" / "maintenance.json"
        )
        assert controller.is_enabled() is False
        assert controller.mode == "suggest"

    def test_enable_persists(self, tmp_path):
        path = tmp_path / "improvement_queue" / "maintenance.json"
        controller = MaintenanceController(path)
        controller.set_enabled(True)
        reloaded = MaintenanceController(path)
        assert reloaded.is_enabled() is True

    def test_run_once_returns_nothing_when_disabled(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        controller = MaintenanceController(
            tmp_path / "improvement_queue" / "maintenance.json"
        )
        assert controller.run_once(analyzer, tasks) == []

    def test_run_once_creates_suggestions_when_enabled(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        controller = MaintenanceController(
            tmp_path / "improvement_queue" / "maintenance.json"
        )
        controller.set_enabled(True)
        created = controller.run_once(analyzer, tasks)
        assert len(created) == 1
        assert created[0].status == "suggestion"

    def test_interval_default_and_clamping(self, tmp_path):
        controller = MaintenanceController(
            tmp_path / "improvement_queue" / "maintenance.json"
        )
        assert controller.interval_minutes == 60
        assert controller.set_interval(1) == 5
        assert controller.set_interval(99999) == 1440
        assert controller.set_interval(30) == 30

    def test_interval_persists_to_reload(self, tmp_path):
        path = tmp_path / "improvement_queue" / "maintenance.json"
        controller = MaintenanceController(path)
        controller.set_interval(90)
        assert MaintenanceController(path).interval_minutes == 90

    def test_due_when_disabled_is_never(self, tmp_path):
        controller = MaintenanceController(
            tmp_path / "improvement_queue" / "maintenance.json"
        )
        controller.set_enabled(True)
        controller._state["last_run"] = ""
        assert controller.due() is True
        controller.set_enabled(False)
        assert controller.due() is False

    def test_due_respects_interval(self, tmp_path):
        path = tmp_path / "improvement_queue" / "maintenance.json"
        controller = MaintenanceController(path)
        controller.set_interval(60)
        controller.set_enabled(True)
        controller._state["last_run"] = "2026-01-01T00:00:00"
        import time as _time

        now = _time.mktime(_time.strptime("2026-01-01T00:59:00", "%Y-%m-%dT%H:%M:%S"))
        assert controller.due(now) is False
        later = _time.mktime(_time.strptime("2026-01-01T01:00:00", "%Y-%m-%dT%H:%M:%S"))
        assert controller.due(later) is True


class TestMaintenanceScheduler:
    def test_tick_returns_nothing_when_disabled(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        controller = MaintenanceController(
            tmp_path / "improvement_queue" / "maintenance.json"
        )
        scheduler = MaintenanceScheduler(controller, analyzer, tasks)
        assert asyncio.run(scheduler.run_tick()) == []

    def test_tick_creates_suggestions_when_enabled_and_due(self, tmp_path):
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        controller = MaintenanceController(
            tmp_path / "improvement_queue" / "maintenance.json"
        )
        controller.set_enabled(True)
        scheduler = MaintenanceScheduler(controller, analyzer, tasks)
        created = asyncio.run(scheduler.run_tick())
        assert len(created) == 1

    def test_tick_skips_when_not_due(self, tmp_path):
        path = tmp_path / "improvement_queue" / "maintenance.json"
        log_path = tmp_path / "logs" / "failures.jsonl"
        _seed_failures(log_path, "click", "stale element", 3)
        tasks, analyzer = _analyzer(log_path, tmp_path, {"browser.click": 3})
        controller = MaintenanceController(path)
        controller.set_interval(60)
        controller.set_enabled(True)
        controller._state["last_run"] = "2099-01-01T00:00:00"
        scheduler = MaintenanceScheduler(controller, analyzer, tasks)
        assert asyncio.run(scheduler.run_tick()) == []
