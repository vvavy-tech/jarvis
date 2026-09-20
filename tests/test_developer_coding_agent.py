"""Tests for the development orchestrator: proposal parsing, apply, checks,
commit, approval, rollback, and the git-less fallback path.
"""

import subprocess

import pytest

from developer.coding_agent import Developer, Edit, ProposalError, parse_proposal
from developer.git_manager import GitManager
from developer.project_tools import ProjectAccess
from developer.task_manager import TaskActionError, TaskManager
from developer.test_runner import CheckResult

GREETING = '{"summary": "Add a greeting module.", "edits": [{"op": "create", "path": "src/greeting.py", "new": "GREETING = \\"hi\\"\\n"}]}\n'


def _ok(name="pytest"):
    return CheckResult(name=name, ok=True, detail="", duration_s=0.0)


def _fail(name="lint"):
    return CheckResult(name=name, ok=False, detail="broken", duration_s=0.1)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if not self.responses:
            return ""
        return self.responses.pop(0)


class FakeRunner:
    def __init__(self, results_per_run):
        self._results = list(results_per_run)
        self.runs = 0

    def run_all(self, *, include_tests=True):
        self.runs += 1
        return self._results.pop(0) if self._results else [_ok()]


class FakeGit:
    def available(self):
        return False

    def is_repo(self):
        return False

    @staticmethod
    def high_impact_report(paths):
        return False, []


def _init_git(path):
    path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "init", "-b", "main"], cwd=path, capture_output=True, text=True
    )
    if result.returncode != 0:
        subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
        subprocess.run(
            ["git", "checkout", "-b", "main"], cwd=path, check=True, capture_output=True
        )
    subprocess.run(
        ["git", "config", "user.name", "Jarvis Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "jarvis@test.local"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# demo\n", encoding="utf-8")
    (path / "src").mkdir()
    (path / "src" / "agent.py").write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True
    )
    return path


def _make_developer(tmp_path, *, llm, runner, use_git=True):
    root = tmp_path / "proj"
    root.mkdir()
    git = GitManager(root) if use_git else FakeGit()
    if use_git:
        _init_git(root)
        git = GitManager(root)
    project = ProjectAccess(root)
    tasks = TaskManager(root / "improvement_queue")
    developer = Developer(
        project=project, tasks=tasks, git=git, runner=runner, llm=llm, root=root
    )
    return developer, tasks, git, root


def _queued(tasks, title="Add greeting", description="please add src/greeting.py"):
    return tasks.create_task(title=title, description=description, status="queued")


@pytest.fixture
def developer_kit(tmp_path):
    llm = FakeLLM([GREETING])
    runner = FakeRunner([[_ok() for _ in range(5)]])
    return _make_developer(tmp_path, llm=llm, runner=runner)


class TestParseProposal:
    def test_valid_proposal(self):
        proposal = parse_proposal(
            '{"edits": [{"op": "replace", "path": "src/a.py", "old": "1", "new": "2"}]}'
        )
        assert proposal["edits"][0]["op"] == "replace"

    def test_fenced_json(self):
        proposal = parse_proposal(
            '```json\n{"edits": [{"op": "create", "path": "a.py", "new": "x"} ]}\n```'
        )
        assert proposal["edits"][0]["path"] == "a.py"

    def test_normalizes_leading_dots(self):
        proposal = parse_proposal(
            '{"edits": [{"op": "create", "path": "./src/a.py", "new": "x"}]}'
        )
        assert proposal["edits"][0]["path"] == "src/a.py"

    def test_rejects_traversal(self):
        with pytest.raises(ProposalError):
            parse_proposal(
                '{"edits": [{"op": "create", "path": "../escape.py", "new": "x"}]}'
            )

    def test_rejects_bad_op(self):
        with pytest.raises(ProposalError):
            parse_proposal('{"edits": [{"op": "delete_all", "path": "a.py"}]}')

    def test_replace_requires_old(self):
        with pytest.raises(ProposalError):
            parse_proposal('{"edits": [{"op": "replace", "path": "a.py", "new": "x"}]}')

    def test_create_requires_new(self):
        with pytest.raises(ProposalError):
            parse_proposal('{"edits": [{"op": "create", "path": "a.py"}]}')

    def test_delete_forbids_old(self):
        with pytest.raises(ProposalError):
            parse_proposal('{"edits": [{"op": "delete", "path": "a.py", "old": "x"}]}')

    def test_empty_edits_rejected(self):
        with pytest.raises(ProposalError):
            parse_proposal('{"edits": []}')

    def test_non_json_rejected(self):
        with pytest.raises(ProposalError):
            parse_proposal("I will edit the file now.")

    def test_duplicate_path_rejected(self):
        with pytest.raises(ProposalError):
            parse_proposal(
                '{"edits": [{"op": "create", "path": "a.py", "new": "x"}, '
                '{"op": "replace", "path": "a.py", "old": "x", "new": "y"}]}'
            )

    def test_edit_dataclass_round_trip(self):
        edit = Edit.from_dict(
            {"op": "create", "path": "src/x.py", "new": "y", "reason": "why"}
        )
        assert edit.path == "src/x.py"
        assert edit.to_dict()["op"] == "create"


class TestHappyPath:
    async def test_propose_apply_commit_and_merge(self, developer_kit):
        developer, tasks, git, root = developer_kit
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert task.status == "needs_approval"
        assert task.next_action == "merge"
        assert task.files == ["src/greeting.py"]
        assert task.branch and git.branch_exists(task.branch)
        assert (root / "src" / "greeting.py").exists()

        merged = developer.merge_task(task)
        assert merged.status == "completed"
        assert not git.branch_exists(f"jarvis-dev/{task.id}")
        assert (root / "src" / "greeting.py").read_text(
            encoding="utf-8"
        ) == 'GREETING = "hi"\n'


class TestHighImpactGating:
    async def test_high_impact_waits_for_approval(self, tmp_path):
        proposal = (
            '{"summary": "bump version", "edits": [{"op": "replace", '
            '"path": "src/agent.py", "old": "VERSION = 1\\n", "new": "VERSION = 3\\n"}]}'
        )
        llm = FakeLLM([proposal])
        runner = FakeRunner([[_ok() for _ in range(5)]])
        developer, tasks, _, root = _make_developer(tmp_path, llm=llm, runner=runner)
        task = tasks.create_task(
            title="bump", description="bump src/agent.py", status="queued"
        )
        task = await developer.start_task(task)
        assert task.status == "needs_approval"
        assert task.next_action == "apply"
        assert task.high_impact is True
        assert (root / "src" / "agent.py").read_text(
            encoding="utf-8"
        ) == "VERSION = 1\n"

        task = await developer.apply_approved(task)
        assert task.status == "needs_approval"
        assert task.next_action == "merge"
        assert (root / "src" / "agent.py").read_text(
            encoding="utf-8"
        ) == "VERSION = 3\n"

        merged = developer.merge_task(task)
        assert merged.status == "completed"


class TestRepairLoop:
    async def test_repairs_after_failing_checks(self, tmp_path):
        proposal = (
            '{"edits": [{"op": "create", "path": "src/greeting.py", '
            '"new": "GREETING = \\"hi\\"\\n"}]}'
        )
        repair = (
            '{"edits": [{"op": "replace", "path": "src/greeting.py", '
            '"old": "GREETING = \\"hi\\"\\n", '
            '"new": "GREETING = \\"hello\\"\\n"}]}'
        )
        llm = FakeLLM([proposal, repair])
        runner = FakeRunner(
            [[_fail("lint") for _ in range(5)], [_ok() for _ in range(5)]]
        )
        developer, tasks, _, root = _make_developer(tmp_path, llm=llm, runner=runner)
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert task.status == "needs_approval"
        assert task.repairs == 1
        assert (root / "src" / "greeting.py").exists()

    async def test_repairs_capped_then_fails(self, tmp_path):
        llm = FakeLLM([GREETING, GREETING, GREETING])
        runner = FakeRunner([[_fail() for _ in range(5)] for _ in range(3)])
        developer, tasks, git, root = _make_developer(tmp_path, llm=llm, runner=runner)
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert task.status == "failed"
        assert task.repairs == 2
        assert not git.branch_exists(f"jarvis-dev/{task.id}")
        assert not (root / "src" / "greeting.py").exists()


class TestApproveReject:
    async def test_reject_cleans_up_branch(self, developer_kit):
        developer, tasks, git, root = developer_kit
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert git.current_branch() == f"jarvis-dev/{task.id}"
        rejected = developer.reject_task(task)
        assert rejected.status == "rejected"
        assert git.current_branch() == "main"
        assert not git.branch_exists(f"jarvis-dev/{task.id}")
        assert not (root / "src" / "greeting.py").exists()

    async def test_cancel_cleans_up(self, developer_kit):
        developer, tasks, git, _ = developer_kit
        task = _queued(tasks)
        task = await developer.start_task(task)
        cancelled = developer.cancel_task(task)
        assert cancelled.status == "cancelled"
        assert git.current_branch() == "main"

    async def test_rollback_after_merge(self, developer_kit):
        developer, tasks, _, root = developer_kit
        task = _queued(tasks)
        task = await developer.start_task(task)
        merged = developer.merge_task(task)
        assert merged.status == "completed"
        rolled = developer.rollback_task(task)
        assert rolled.status == "rolled_back"
        assert not (root / "src" / "greeting.py").exists()

    async def test_bad_transitions_raise(self, developer_kit):
        developer, tasks, _, _ = developer_kit
        task = tasks.create_task(title="T", description="D")
        with pytest.raises(TaskActionError):
            await developer.start_task(task)  # suggestion status cannot start

    async def test_merge_task_requires_approval(self, developer_kit):
        developer, tasks, _, _ = developer_kit
        task = tasks.create_task(title="T", description="D")
        with pytest.raises(TaskActionError, match="not waiting"):
            developer.merge_task(task)

    async def test_apply_approved_requires_pending_proposal(self, developer_kit):
        developer, tasks, _, _ = developer_kit
        task = tasks.create_task(title="T", description="D")
        with pytest.raises(TaskActionError, match="No pending proposal"):
            await developer.apply_approved(task)

    async def test_merge_records_merge_commit(self, developer_kit):
        developer, tasks, git, _ = developer_kit
        task = _queued(tasks)
        task = await developer.start_task(task)
        merged = developer.merge_task(task)
        assert merged.merge_commit
        assert git.head_commit("main") == merged.merge_commit


class TestProposalFailurePaths:
    async def test_invalid_json_records_failure_and_cleans_up(self, tmp_path):
        llm = FakeLLM(["I will fix the greeting for you."])
        runner = FakeRunner([[_ok() for _ in range(5)]])
        developer, tasks, git, _ = _make_developer(tmp_path, llm=llm, runner=runner)
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert task.status == "failed"
        assert "no JSON" in task.notes
        assert git.current_branch() == "main"
        assert not git.branch_exists(f"jarvis-dev/{task.id}")

    async def test_llm_exception_records_failure_type(self, tmp_path):
        class RaisingLLM:
            def complete(self, system, user):
                raise RuntimeError("model overloaded")

        runner = FakeRunner([[_ok() for _ in range(5)]])
        developer, tasks, _, _ = _make_developer(
            tmp_path, llm=RaisingLLM(), runner=runner
        )
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert task.status == "failed"
        assert "Proposal failed" in task.notes
        assert "RuntimeError" in task.notes


class TestCautionFlag:
    async def test_warns_when_baseline_dirty_overlaps(self, tmp_path):
        proposal = (
            '{"edits": [{"op": "replace", "path": "src/agent.py", '
            '"old": "VERSION = 9\\n", "new": "VERSION = 3\\n"}]}'
        )
        root = tmp_path / "proj"
        root.mkdir()
        _init_git(root)
        (root / "src" / "agent.py").write_text("VERSION = 9\n", encoding="utf-8")
        tasks = TaskManager(root / "improvement_queue")
        runner = FakeRunner([[_ok() for _ in range(5)]])
        developer = Developer(
            project=ProjectAccess(root),
            tasks=tasks,
            git=GitManager(root),
            runner=runner,
            llm=FakeLLM([proposal]),
            root=root,
        )
        task = tasks.create_task(
            title="bump", description="bump src/agent.py", status="queued"
        )
        task = await developer.start_task(task)
        assert task.caution is True


class TestParseProposalExtras:
    def test_plain_fenced_json(self):
        proposal = parse_proposal(
            '```\n{"edits": [{"op": "create", "path": "a.py", "new": "x"}]}\n```'
        )
        assert proposal["edits"][0]["path"] == "a.py"

    def test_too_many_edits_rejected(self):
        edits = ", ".join(
            f'{{"op": "create", "path": "f{i}.py", "new": "x"}}' for i in range(9)
        )
        with pytest.raises(ProposalError, match="Too many"):
            parse_proposal('{"edits": [' + edits + "]}")

    def test_oversized_edit_rejected(self):
        huge = "x" * 7000
        with pytest.raises(ProposalError, match="too large"):
            parse_proposal(
                '{"edits": [{"op": "create", "path": "a.py", "new": "' + huge + '"}]}'
            )

    def test_missing_edits_key_rejected(self):
        with pytest.raises(ProposalError, match="no 'edits'"):
            parse_proposal('{"summary": "nothing"}')


class TestGitLessFallback:
    async def test_backup_applies_in_place(self, tmp_path):
        llm = FakeLLM([GREETING])
        runner = FakeRunner([[_ok() for _ in range(5)]])
        developer, tasks, _, root = _make_developer(
            tmp_path, llm=llm, runner=runner, use_git=False
        )
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert task.status == "needs_approval"
        assert task.next_action == "merge"
        assert (root / "src" / "greeting.py").exists()
        merged = developer.merge_task(task)
        assert merged.status == "completed"
        assert (root / "src" / "greeting.py").exists()

    async def test_failure_restores_from_backup(self, tmp_path):
        llm = FakeLLM([GREETING])
        runner = FakeRunner([[_fail() for _ in range(5)]])
        developer, tasks, _, root = _make_developer(
            tmp_path, llm=llm, runner=runner, use_git=False
        )
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert task.status == "failed"
        assert not (root / "src" / "greeting.py").exists()

    async def test_rollback_removes_created_file(self, tmp_path):
        llm = FakeLLM([GREETING])
        runner = FakeRunner([[_ok() for _ in range(5)]])
        developer, tasks, _, root = _make_developer(
            tmp_path, llm=llm, runner=runner, use_git=False
        )
        task = _queued(tasks)
        task = await developer.start_task(task)
        merged = developer.merge_task(task)
        assert merged.status == "completed"
        assert (root / "src" / "greeting.py").exists()
        rolled = developer.rollback_task(task)
        assert rolled.status == "rolled_back"
        assert not (root / "src" / "greeting.py").exists()


class TestTaskDiff:
    async def test_diff_reports_changes(self, developer_kit):
        developer, tasks, _, _ = developer_kit
        task = _queued(tasks)
        task = await developer.start_task(task)
        diff = developer.task_diff(task)
        assert "greeting.py" in diff

    async def test_diff_gitless_message(self, tmp_path):
        llm = FakeLLM([GREETING])
        runner = FakeRunner([[_ok() for _ in range(5)]])
        developer, tasks, _, _ = _make_developer(
            tmp_path, llm=llm, runner=runner, use_git=False
        )
        task = _queued(tasks)
        task = await developer.start_task(task)
        assert "git-less" in developer.task_diff(task)
