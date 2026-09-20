"""Tests for the gated voice layer that exposes the self-development tools.

DeveloperTools sits on top of Developer and ToolGate. Every tool must be
unreachable unless the user turn has the wake word and an explicit development
request, and each tool must defer to TaskManager/Developer, never performing
its own filesystem or git work.
"""

import asyncio

import pytest
from livekit.agents.llm import ToolError

from developer.maintenance import MaintenanceController
from developer.task_manager import TaskManager
from gates import ToolGate
from tools import DeveloperTools, _task_summary


class FakeDeveloper:
    def __init__(self, tasks):
        self.tasks = tasks
        self.start_calls = []
        self.apply_calls = []
        self.merge_calls = []
        self.reject_calls = []
        self.cancel_calls = []
        self.rollback_calls = []
        self.diff_calls = []
        self.tests_calls = 0

    async def start_task(self, task):
        self.start_calls.append(task.id)
        return self.tasks.set_status(task, "needs_approval", "started")

    async def apply_approved(self, task):
        self.apply_calls.append(task.id)
        task.next_action = "merge"
        return self.tasks.set_status(task, "needs_approval", "applied")

    def merge_task(self, task):
        self.merge_calls.append(task.id)
        return self.tasks.set_status(task, "completed", "merged to main")

    def reject_task(self, task):
        self.reject_calls.append(task.id)
        return self.tasks.set_status(task, "rejected", "rejected")

    def cancel_task(self, task):
        self.cancel_calls.append(task.id)
        return self.tasks.set_status(task, "cancelled", "cancelled")

    def rollback_task(self, task):
        self.rollback_calls.append(task.id)
        return self.tasks.set_status(task, "rolled_back", "rolled back everything")

    def task_diff(self, task):
        self.diff_calls.append(task.id)
        return '- src/greeting.py | 1 +\n+ GREETING = "hi"'

    def run_tests(self):
        self.tests_calls += 1
        return "PASS pytest (23 tests)"


@pytest.fixture
def kit(tmp_path):
    tasks = TaskManager(tmp_path / "improvement_queue")
    developer = FakeDeveloper(tasks)
    gate = ToolGate()
    maintenance = MaintenanceController(
        tmp_path / "improvement_queue" / "maintenance.json"
    )
    tools = DeveloperTools(
        developer=developer, gate=gate, tasks=tasks, maintenance=maintenance
    )
    return tasks, developer, gate, maintenance, tools


def _arm_dev_request(gate, request="Jarvis, developer mode"):
    gate.set_user_request(request)


def _task_with_action(tasks, action="merge"):
    task = tasks.create_task(
        title="Fix greeting", description="fix src/greeting.py", status="needs_approval"
    )
    task.next_action = action
    tasks.save(task)
    return task


class TestGateEnforcement:
    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("developer_mode", {"request": "Jarvis, improve your code"}),
            ("dev_list_tasks", {}),
            ("dev_show", {}),
            ("dev_approve", {}),
            ("dev_reject", {}),
            ("dev_cancel", {}),
            ("dev_diff", {}),
            ("dev_tests", {}),
            ("dev_rollback", {}),
            ("dev_maintenance", {"action": "status"}),
        ],
    )
    async def test_every_tool_blocked_without_wake_word(self, kit, name, args):
        tasks, developer, _, _, tools = kit
        tool = getattr(tools, name)
        with pytest.raises(ToolError, match="Jarvis"):
            await tool({}, **args)
        assert developer.start_calls == []
        assert len(tasks.list_tasks()) == 0

    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("developer_mode", {"request": "what is the weather"}),
            ("dev_list_tasks", {}),
            ("dev_show", {}),
            ("dev_approve", {}),
            ("dev_reject", {}),
            ("dev_cancel", {}),
            ("dev_diff", {}),
            ("dev_tests", {}),
            ("dev_rollback", {}),
            ("dev_maintenance", {"action": "status"}),
        ],
    )
    async def test_every_tool_blocked_without_dev_request(self, kit, name, args):
        _, developer, gate, _, tools = kit
        gate.set_user_request("Jarvis, what is the weather")
        tool = getattr(tools, name)
        with pytest.raises(ToolError, match="own code"):
            await tool({}, **args)
        assert developer.start_calls == []

    async def test_dev_approve_blocked_on_finished_task(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        task = _task_with_action(tasks)
        tasks.set_status(task, "completed", "already done")
        with pytest.raises(ToolError, match="not awaiting approval"):
            await tools.dev_approve({}, task_id=task.id)
        assert developer.merge_calls == []

    async def test_dev_cancel_blocked_on_finished_task(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        task = _task_with_action(tasks)
        tasks.set_status(task, "rolled_back", "done")
        with pytest.raises(ToolError, match="dev_rollback"):
            await tools.dev_cancel({}, task_id=task.id)
        assert developer.cancel_calls == []


class TestDeveloperMode:
    async def test_creates_queued_task_and_runs_in_background(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        message = await tools.developer_mode(
            {}, request="Jarvis, fix the slow browser timeout please"
        )
        assert "background" in message
        stored = tasks.list_tasks()
        assert len(stored) == 1
        assert stored[0].status == "queued"
        assert stored[0].category == "voice"
        assert stored[0].created_by == "user"
        assert developer.start_calls == []
        await asyncio.sleep(0.01)
        assert developer.start_calls == [stored[0].id]
        assert tasks.latest().status == "needs_approval"

    async def test_blocks_second_task_while_running(self, kit):
        tasks, _, gate, _, tools = kit
        _arm_dev_request(gate)
        tools._running_task = asyncio.create_task(asyncio.sleep(10))
        message = await tools.developer_mode({}, request="Jarvis, improve your code")
        assert "still running" in message
        assert len(tasks.list_tasks()) == 0
        tools._running_task.cancel()


class TestListAndShow:
    async def test_list_tasks_empty_message(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate)
        assert "no development tasks" in await tools.dev_list_tasks({})

    async def test_list_tasks_filter_by_status(self, kit):
        tasks, _, gate, _, tools = kit
        _arm_dev_request(gate)
        tasks.create_task(title="Done thing", description="x")
        queued = tasks.create_task(
            title="Pending thing", description="y", status="queued"
        )
        output = await tools.dev_list_tasks({}, status="queued")
        assert queued.id in output
        assert "Done thing" not in output

    async def test_show_defaults_to_latest(self, kit):
        tasks, _, gate, _, tools = kit
        _arm_dev_request(gate)
        tasks.create_task(title="First", description="a")
        latest = tasks.create_task(title="Second", description="b")
        output = await tools.dev_show({})
        assert latest.id in output

    async def test_show_by_id(self, kit):
        tasks, _, gate, _, tools = kit
        _arm_dev_request(gate)
        task = tasks.create_task(title="Only", description="c")
        output = await tools.dev_show({}, task_id=task.id)
        assert task.id in output

    async def test_show_unknown_id_raises(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate)
        with pytest.raises(ToolError, match="Task not found"):
            await tools.dev_show({}, task_id="dev-unknown")


class TestApproveRejectCancelRollback:
    async def test_approve_merges_task(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        task = _task_with_action(tasks, action="merge")
        output = await tools.dev_approve({}, task_id=task.id)
        assert developer.merge_calls == [task.id]
        assert "merged to main" in output

    async def test_approve_applies_then_awaits_merge(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        task = _task_with_action(tasks, action="apply")
        message = await tools.dev_approve({}, task_id=task.id)
        assert "background" in message
        await asyncio.sleep(0.01)
        assert developer.apply_calls == [task.id]
        assert tasks.latest().next_action == "merge"

    async def test_reject_discards(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        task = _task_with_action(tasks)
        output = await tools.dev_reject({}, task_id=task.id)
        assert developer.reject_calls == [task.id]
        assert "Rejected" in output

    async def test_cancel_task(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        task = tasks.create_task(title="Wip", description="x", status="in_progress")
        output = await tools.dev_cancel({}, task_id=task.id)
        assert developer.cancel_calls == [task.id]
        assert "Cancelled" in output

    async def test_rollback_uses_note(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        task = _task_with_action(tasks)
        output = await tools.dev_rollback({}, task_id=task.id)
        assert developer.rollback_calls == [task.id]
        assert "rolled back everything" in output


class TestDiffAndTests:
    async def test_dev_diff_includes_task_id(self, kit):
        tasks, _, gate, _, tools = kit
        _arm_dev_request(gate)
        task = _task_with_action(tasks)
        output = await tools.dev_diff({}, task_id=task.id)
        assert task.id in output
        assert "greeting.py" in output

    async def test_dev_tests_runs_full_suite(self, kit):
        _, developer, gate, _, tools = kit
        _arm_dev_request(gate)
        output = await tools.dev_tests({})
        assert developer.tests_calls == 1
        assert "PASS pytest" in output


class TestNewIntegration:
    async def test_creates_integration_task_and_runs_in_background(self, kit):
        tasks, developer, gate, _, tools = kit
        _arm_dev_request(gate, "Jarvis, create an integration for Philips Hue")
        message = await tools.dev_new_integration({}, service="Philips Hue")
        assert "background" in message
        stored = tasks.list_tasks()
        assert len(stored) == 1
        assert stored[0].category == "integration"
        assert stored[0].created_by == "user"
        assert stored[0].description == "create an integration for Philips Hue"
        assert developer.start_calls == []
        await asyncio.sleep(0.01)
        assert developer.start_calls == [stored[0].id]

    async def test_empty_service_rejected(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate)
        with pytest.raises(ToolError, match="name the service"):
            await tools.dev_new_integration({}, service="   ")

    async def test_blocks_second_task_while_running(self, kit):
        tasks, _, gate, _, tools = kit
        _arm_dev_request(gate)
        tools._running_task = asyncio.create_task(asyncio.sleep(10))
        message = await tools.dev_new_integration({}, service="Strava")
        assert "still running" in message
        assert len(tasks.list_tasks()) == 0
        tools._running_task.cancel()


class TestMaintenance:
    async def test_status_when_disabled(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate, "Jarvis, what is the status")
        output = await tools.dev_maintenance({}, action="status")
        assert "disabled" in output

    async def test_enable_and_disable_round_trip(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate, "Jarvis, enable maintenance")
        await tools.dev_maintenance({}, action="enable")
        assert tools.maintenance.is_enabled()
        gate.set_user_request("Jarvis, disable maintenance")
        await tools.dev_maintenance({}, action="disable")
        assert not tools.maintenance.is_enabled()

    async def test_invalid_action_rejected(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate)
        with pytest.raises(ToolError, match="status"):
            await tools.dev_maintenance({}, action="spin")

    async def test_interval_action_sets_and_enables(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate, "Jarvis, enable maintenance")
        output = await tools.dev_maintenance({}, action="interval", minutes=30)
        assert "30 minutes" in output
        assert tools.maintenance.is_enabled()
        assert tools.maintenance.interval_minutes == 30

    async def test_invalid_minutes_rejected(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate)
        with pytest.raises(ToolError, match="number of minutes"):
            await tools.dev_maintenance({}, action="interval", minutes="abc")

    async def test_status_includes_interval(self, kit):
        _, _, gate, _, tools = kit
        _arm_dev_request(gate, "Jarvis, what is the status")
        output = await tools.dev_maintenance({}, action="status")
        assert "60 minutes" in output


class TestTaskSummary:
    def test_renders_files_and_state(self, kit):
        tasks, _, _, _, _ = kit
        task = _task_with_action(tasks)
        task.files = ["src/greeting.py"]
        task.notes = "Working hard"
        task.summary = "added greeting"
        output = _task_summary(task)
        assert task.id in output
        assert "needs_approval" in output
        assert "src/greeting.py" in output
        assert "Working hard" in output
        assert "added greeting" in output
