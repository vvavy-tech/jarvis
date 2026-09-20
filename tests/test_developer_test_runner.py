"""Tests for the project check runner."""

import sys

from developer.test_runner import CheckResult, format_results
from developer.test_runner import TestRunner as _Runner


class TestSyntaxCheck:
    def test_valid_module_passes(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "ok.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
        runner = _Runner(root=tmp_path, python=sys.executable)
        assert runner.syntax_check().ok

    def test_broken_module_fails(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text("def f(:\n", encoding="utf-8")
        runner = _Runner(root=tmp_path, python=sys.executable)
        result = runner.syntax_check()
        assert result.ok is False
        assert result.detail


class TestImportCheck:
    def test_project_modules_import(self):
        runner = _Runner(python=sys.executable)
        assert runner.import_check().ok


class TestFormatResults:
    def test_builds_pass_lines(self):
        results = [CheckResult("lint", True, "", 0.5)]
        output = format_results(results)
        assert "PASS lint" in output

    def test_builds_fail_detail(self):
        results = [CheckResult("pytest", False, "1 failed", 2.1)]
        output = format_results(results)
        assert "FAIL pytest" in output
        assert "1 failed" in output


class TestProjectPytest:
    def test_passes_clean_suite(self, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_mini.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        runner = _Runner(root=tmp_path, python=sys.executable)
        result = runner.test_check(paths=("tests",))
        assert result.ok, result.detail
        assert result.name == "pytest"

    def test_reports_failing_suite(self, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_mini.py").write_text(
            "def test_bad():\n    assert False\n", encoding="utf-8"
        )
        runner = _Runner(root=tmp_path, python=sys.executable)
        result = runner.test_check(paths=("tests",))
        assert result.ok is False
        assert "test_bad" in result.detail
