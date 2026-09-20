"""Unit tests for the sandboxed project access helpers."""

import pytest

from developer.project_tools import (
    PROJECT_ROOT,
    ProjectAccess,
    ProjectAccessError,
    extract_path_tokens,
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "agent.py").write_text("# agent\nVERSION = 1\n", encoding="utf-8")
    (root / "notes").mkdir()
    (root / "notes" / "todo.txt").write_text("todo\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".venv").mkdir()
    (root / "node_modules").mkdir()
    (root / "improvement_queue").mkdir()
    (root / "logs").mkdir()
    return ProjectAccess(root)


class TestResolve:
    def test_inside_root_allowed(self, project):
        target = project.resolve_project_path("src/agent.py")
        assert target == project.root / "src" / "agent.py"

    def test_absolute_outside_blocked(self, project, tmp_path):
        with pytest.raises(ProjectAccessError):
            project.resolve_project_path(tmp_path / "secrets.txt")

    def test_parent_traversal_blocked(self, project):
        with pytest.raises(ProjectAccessError):
            project.resolve_project_path("../outside.txt")

    def test_has_path(self, project):
        assert project.has_path("src/agent.py")
        assert not project.has_path("src/missing.py")


class TestListing:
    def test_list_excludes_blocked_and_secret(self, project):
        files = project.list_files()
        assert "src/agent.py" in files
        assert "notes/todo.txt" in files
        assert ".env" not in files
        assert not any(f.startswith(".git/") for f in files)
        assert not any(f.startswith(".venv/") for f in files)
        assert not any(f.startswith("node_modules/") for f in files)


class TestRead:
    def test_read_file(self, project):
        result = project.read_file("src/agent.py")
        assert "VERSION = 1" in result["content"]

    def test_read_missing_raises(self, project):
        with pytest.raises(ProjectAccessError):
            project.read_file("src/missing.py")

    def test_read_dir_raises(self, project):
        with pytest.raises(ProjectAccessError):
            project.read_file("src")

    def test_read_blocked_dir_raises(self, project):
        with pytest.raises(ProjectAccessError):
            project.read_file(".git/config")

    def test_read_secret_file_raises(self, project):
        with pytest.raises(ProjectAccessError):
            project.read_file(".env")


class TestWrite:
    def test_create_and_read_back(self, project):
        written = project.write_file("src/new.txt", "hello\n")
        assert written["created"] is True
        assert project.read_file("src/new.txt")["content"] == "hello\n"

    def test_overwrite_existing(self, project):
        project.write_file("src/new.txt", "one\n")
        written = project.write_file("src/new.txt", "two\n")
        assert written["overwritten"] is True
        assert project.read_file("src/new.txt")["content"] == "two\n"

    def test_no_overwrite_raises(self, project):
        project.write_file("src/new.txt", "one\n")
        with pytest.raises(ProjectAccessError):
            project.write_file("src/new.txt", "two\n", overwrite=False)

    def test_write_blocked_dir_raises(self, project):
        with pytest.raises(ProjectAccessError):
            project.write_file(".venv/site.py", "x")
        with pytest.raises(ProjectAccessError):
            project.write_file("node_modules/pkg.js", "x")

    def test_write_secret_name_raises(self, project):
        with pytest.raises(ProjectAccessError):
            project.write_file("config/credentials.json", "{}")
        with pytest.raises(ProjectAccessError):
            project.write_file(".env.prod", "X=1")
        with pytest.raises(ProjectAccessError):
            project.write_file("keys/private.pem", "x")

    def test_delete(self, project):
        project.write_file("src/tmp.py", "x")
        project.delete_file("src/tmp.py")
        assert project.has_path("src/tmp.py") is False


class TestSearch:
    def test_finds_matching_lines(self, project):
        matches = project.search_files(r"VERSION")
        assert any(m["path"] == "src/agent.py" and m["line"] == 2 for m in matches)

    def test_skips_blocked_dirs(self, project):
        (project.root / ".git" / "hooks.txt").write_text("VERSION\n", encoding="utf-8")
        matches = project.search_files(r"VERSION")
        assert all(m["path"] != ".git/hooks.txt" for m in matches)

    def test_invalid_regex_raises(self, project):
        with pytest.raises(ProjectAccessError):
            project.search_files("([unclosed")


class TestExtractPathTokens:
    def test_extracts_likely_paths(self):
        tokens = extract_path_tokens("the bug is in src/agent.py and maybe gates.py")
        assert any("src/agent.py" in token for token in tokens)
        assert any("gates.py" in token for token in tokens)


def test_project_root_points_at_repo():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
