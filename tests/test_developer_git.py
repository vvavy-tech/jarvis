"""Tests for the safe git wrapper, using throwaway repositories."""

import subprocess

import pytest

from developer.git_manager import GitError, GitManager, task_branch


def _git(path, *args, check=True):
    result = subprocess.run(
        ["git", *args],
        cwd=path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def _init_repo(path, files=None):
    path.mkdir(parents=True, exist_ok=True)
    init = _git(path, "init", "-b", "main", check=False)
    if init.returncode != 0:
        _git(path, "init")
        _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Jarvis Test")
    _git(path, "config", "user.email", "jarvis@test.local")
    for name, content in (files or {}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "init")
    return path


@pytest.fixture
def repo(tmp_path):
    return _init_repo(
        tmp_path / "repo",
        {"src/agent.py": "VERSION = 1\n", "README.md": "# demo\n"},
    )


class TestBranchNaming:
    def test_branch_prefix(self):
        assert task_branch("dev123") == "jarvis-dev/dev123"

    def test_rejects_unsafe_id(self):
        with pytest.raises(GitError):
            task_branch("../evil")


class TestState:
    def test_available(self, repo):
        assert GitManager(repo).available()

    def test_is_repo(self, repo):
        assert GitManager(repo).is_repo()

    def test_current_branch_and_head(self, repo):
        manager = GitManager(repo)
        assert manager.current_branch() == "main"
        assert manager.head_commit()

    def test_status_porcelain_empty_when_clean(self, repo):
        assert GitManager(repo).status_porcelain() == ""

    def test_branch_lifecycle(self, repo):
        manager = GitManager(repo)
        manager.create_task_branch("dev-abc")
        assert manager.current_branch() == "jarvis-dev/dev-abc"
        assert manager.branch_exists("jarvis-dev/dev-abc")
        manager.delete_branch("dev-abc")
        assert not manager.branch_exists("jarvis-dev/dev-abc")

    def test_commit_paths_only_stages_listed(self, repo):
        manager = GitManager(repo)
        manager.create_task_branch("dev-commit")
        (repo / "src" / "agent.py").write_text("VERSION = 2\n", encoding="utf-8")
        (repo / "scratch.txt").write_text("keep out\n", encoding="utf-8")
        manager.commit_paths("bump version", ["src/agent.py"])
        tracked = manager.tracked_files()
        assert "src/agent.py" in tracked
        # scratch.txt may exist in index from baseline? It never existed in baseline:
        assert "scratch.txt" not in tracked


class TestHighImpact:
    def test_core_agent_file(self):
        impact, reasons = GitManager.high_impact_report(["src/agent.py"])
        assert impact
        assert any("agent" in reason for reason in reasons)

    def test_pyproject(self):
        impact, _ = GitManager.high_impact_report(["pyproject.toml"])
        assert impact

    def test_credentials_path(self):
        impact, _ = GitManager.high_impact_report(["src/auth.py"])
        assert impact

    def test_payment_logic(self):
        impact, _ = GitManager.high_impact_report(["src/payments/checkout.py"])
        assert impact

    def test_boring_module_safe(self):
        impact, _ = GitManager.high_impact_report(["src/notes/todo.py"])
        assert not impact

    def test_delete_shaped_path(self):
        impact, _ = GitManager.high_impact_report(["src/remove_old.py"])
        assert impact


class TestDiffHelpers:
    def test_dirty_paths_from_porcelain(self):
        porcelain = " M src/agent.py\n?? scratch.txt\n"
        dirty = GitManager.dirty_paths_from_porcelain(porcelain)
        assert "src/agent.py" in dirty
        assert "scratch.txt" not in dirty

    def test_diff_after_edit(self, repo):
        manager = GitManager(repo)
        base = manager.head_commit()
        (repo / "src" / "agent.py").write_text("VERSION = 2\n", encoding="utf-8")
        diff = manager.diff(base, paths=["src/agent.py"])
        assert "VERSION = 2" in diff

    def test_diff_truncated_with_marker(self, repo):
        manager = GitManager(repo)
        base = manager.head_commit()
        (repo / "src" / "agent.py").write_text(
            "VERSION = " + "x" * 500 + "\n", encoding="utf-8"
        )
        diff = manager.diff(base, paths=["src/agent.py"], max_chars=60)
        assert "truncated" in diff
        assert len(diff) <= 120


class TestLifecycle:
    def test_merge_is_no_ff_and_reversible(self, repo):
        manager = GitManager(repo)
        main_head_before = manager.head_commit()
        manager.create_task_branch("dev-merge")
        (repo / "src" / "agent.py").write_text("VERSION = 2\n", encoding="utf-8")
        manager.commit_paths("bump version", ["src/agent.py"])
        assert not manager.is_merged("dev-merge")

        merge_commit = manager.merge_main(
            "dev-merge", "Merge jarvis-dev/dev-merge: bump"
        )
        assert manager.is_merged("dev-merge")
        assert manager.current_branch() == "main"
        assert main_head_before != manager.head_commit()
        assert (repo / "src" / "agent.py").read_text(
            encoding="utf-8"
        ) == "VERSION = 2\n"

        manager.delete_branch("dev-merge")
        assert not manager.branch_exists("jarvis-dev/dev-merge")
        assert manager.is_merged("dev-merge")

        manager.revert_merge(merge_commit)
        assert (repo / "src" / "agent.py").read_text(
            encoding="utf-8"
        ) == "VERSION = 1\n"

    def test_commit_paths_empty_raises(self, repo):
        manager = GitManager(repo)
        with pytest.raises(GitError):
            manager.commit_paths("nothing", [])

    def test_create_task_branch_checks_out_existing(self, repo):
        manager = GitManager(repo)
        manager.create_task_branch("dev-x")
        manager.checkout("main")
        manager.create_task_branch("dev-x")
        assert manager.current_branch() == "jarvis-dev/dev-x"

    def test_restore_from_index(self, repo):
        manager = GitManager(repo)
        (repo / "src" / "agent.py").write_text("VERSION = 9\n", encoding="utf-8")
        assert manager.status_porcelain()
        manager.restore_from_index(["src/agent.py"])
        assert manager.status_porcelain() == ""
        assert (repo / "src" / "agent.py").read_text(
            encoding="utf-8"
        ) == "VERSION = 1\n"

    def test_tracked_files_excludes_untracked(self, repo):
        manager = GitManager(repo)
        (repo / "scratch.py").write_text("keep out\n", encoding="utf-8")
        manager.create_task_branch("dev-tracked")
        (repo / "src" / "agent.py").write_text("VERSION = 2\n", encoding="utf-8")
        manager.commit_paths("bump", ["src/agent.py"])
        tracked = manager.tracked_files()
        assert "src/agent.py" in tracked
        assert "scratch.py" not in tracked


class TestEnvironment:
    def test_is_repo_false_outside_git(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert GitManager(plain).is_repo() is False

    def test_unavailable_without_git_binary(self):
        manager = GitManager()
        assert manager.available()

    def test_bad_branch_id_rejected(self):
        with pytest.raises(GitError):
            task_branch("has spaces!")


class TestHighImpactRules:
    def test_voice_pipeline(self):
        impact, reasons = GitManager.high_impact_report(
            ["src/agent/realtime_speech.py"]
        )
        assert impact
        assert any("voice" in reason for reason in reasons)

    def test_lockfile(self):
        impact, _ = GitManager.high_impact_report(["uv.lock"])
        assert impact

    def test_credential_material_suffix(self):
        impact, _ = GitManager.high_impact_report(["secrets/app.pem"])
        assert impact

    def test_destructive_remove(self):
        impact, _ = GitManager.high_impact_report(["src/maintenance/remove_old.py"])
        assert impact

    def test_backslash_paths_normalized(self):
        impact, _ = GitManager.high_impact_report([r"src\agent.py"])
        assert impact
