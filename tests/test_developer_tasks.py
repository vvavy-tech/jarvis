"""Tests for the task queue persistence layer."""

import pytest

from developer.task_manager import (
    TASK_STATUSES,
    DevTask,
    TaskActionError,
    TaskManager,
    TaskNotFoundError,
)


@pytest.fixture
def manager(tmp_path):
    return TaskManager(tmp_path / "improvement_queue")


class TestCreateLoadSave:
    def test_create_task_defaults_to_suggestion(self, manager):
        task = manager.create_task(title="T", description="D")
        assert task.status == "suggestion"
        assert task.id.startswith("dev")
        assert task.title == "T"

    def test_round_trip(self, manager):
        created = manager.create_task(title="T", description="D", status="queued")
        loaded = manager.load(created.id)
        assert loaded.id == created.id
        assert loaded.status == "queued"
        assert loaded.title == "T"

    def test_save_persists_edits(self, manager):
        task = manager.create_task(title="T", description="D")
        task.files = ["src/a.py", "src/b.py"]
        task.proposal = {"edits": [{"op": "create", "path": "src/a.py"}]}
        manager.save(task)
        loaded = manager.load(task.id)
        assert loaded.files == ["src/a.py", "src/b.py"]
        assert loaded.proposal["edits"][0]["path"] == "src/a.py"

    def test_load_missing_raises(self, manager):
        with pytest.raises(TaskNotFoundError):
            manager.load("devdoesnotexist")

    def test_invalid_task_id_rejected(self, manager):
        with pytest.raises(TaskNotFoundError):
            manager._task_path("../escape")

    def test_unknown_status_rejected(self, manager):
        with pytest.raises(TaskActionError):
            manager.create_task(title="T", description="D", status="bogus")

    def test_save_with_bad_status_rejected(self, manager):
        task = manager.create_task(title="T", description="D")
        task.status = "bogus"
        with pytest.raises(TaskActionError):
            manager.save(task)


class TestListing:
    def test_latest_is_most_recent(self, manager):
        first = manager.create_task(title="first", description="D")
        second = manager.create_task(title="second", description="D")
        assert manager.latest().id == second.id
        assert manager.latest().id != first.id

    def test_list_filter_by_status(self, manager):
        manager.create_task(title="a", description="D", status="suggestion")
        manager.create_task(title="b", description="D", status="queued")
        queued = manager.list_tasks(status="queued")
        assert len(queued) == 1
        assert queued[0].title == "b"

    def test_delete(self, manager):
        task = manager.create_task(title="T", description="D")
        manager.delete(task.id)
        assert manager.list_tasks() == []

    def test_set_status(self, manager):
        task = manager.create_task(title="T", description="D")
        manager.set_status(task, "queued", "ready")
        assert task.status == "queued"
        assert task.notes == "ready"
        assert manager.load(task.id).status == "queued"


def test_statuses_are_exhaustive():
    assert "suggestion" in TASK_STATUSES
    assert "queued" in TASK_STATUSES
    assert "in_progress" in TASK_STATUSES
    assert "needs_approval" in TASK_STATUSES
    assert "completed" in TASK_STATUSES
    assert "rolled_back" in TASK_STATUSES
    assert "failed" in TASK_STATUSES
    assert "cancelled" in TASK_STATUSES
    assert "rejected" in TASK_STATUSES


def test_from_dict_tolerates_missing_fields():
    task = DevTask.from_dict({"id": "devx", "title": "T"})
    assert task.id == "devx"
    assert task.status == "suggestion"
    assert task.files == []
