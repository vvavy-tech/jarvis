"""Project checks orchestrated before and after every development task."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from developer.project_tools import PROJECT_ROOT

DETAIL_MAX = 2000

IMPORT_STMT = (
    "import agent, gates, tools, browser_tools, developer; "
    "import developer.coding_agent, developer.failures, developer.git_manager, "
    "developer.integration_scaffold, developer.maintenance, developer.project_tools, "
    "developer.task_manager, developer.test_runner; "
    "import integrations; "
    "import integrations.base, integrations.registry, integrations.capability_tools; "
    "import integrations.spotify, integrations.gmail, "
    "integrations.google_calendar, integrations.meta_ads; "
    "import integrations.screen_vision; "
    "print('imports ok')"
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    duration_s: float


def _crop(text: str) -> str:
    text = text.strip()
    if len(text) > DETAIL_MAX:
        return text[:DETAIL_MAX] + "\n[... truncated ...]"
    return text


class TestRunner:
    """Runs syntax, import, lint, format, and unit-test checks."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        python: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.root = (root or PROJECT_ROOT).resolve()
        self.python = python or sys.executable
        self.timeout_s = timeout_s

    def _run(
        self, args: list[str], timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )

    def _result(
        self, name: str, process: subprocess.CompletedProcess[str], start: float
    ) -> CheckResult:
        output = process.stdout or ""
        if process.stderr:
            output += "\n" + process.stderr
        return CheckResult(
            name=name,
            ok=process.returncode == 0,
            detail=_crop(output) if output else f"exit {process.returncode}",
            duration_s=round(time.monotonic() - start, 2),
        )

    def syntax_check(self) -> CheckResult:
        start = time.monotonic()
        process = self._run(
            [self.python, "-m", "compileall", "-q", str(self.root / "src")], 180
        )
        return self._result("syntax", process, start)

    def import_check(self) -> CheckResult:
        start = time.monotonic()
        process = self._run([self.python, "-c", IMPORT_STMT], 180)
        return self._result("imports", process, start)

    def lint_check(self) -> CheckResult:
        start = time.monotonic()
        process = self._run([self.python, "-m", "ruff", "check", "src", "tests"], 300)
        return self._result("lint", process, start)

    def format_check(self) -> CheckResult:
        start = time.monotonic()
        process = self._run(
            [self.python, "-m", "ruff", "format", "--check", "src", "tests"], 300
        )
        return self._result("format", process, start)

    def test_check(self, paths: tuple[str, ...] = ()) -> CheckResult:
        start = time.monotonic()
        args = [self.python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        if paths:
            args.extend(paths)
        process = self._run(args, self.timeout_s)
        return self._result("pytest", process, start)

    def run_all(self, *, include_tests: bool = True) -> list[CheckResult]:
        results = [
            self.syntax_check(),
            self.import_check(),
            self.lint_check(),
            self.format_check(),
        ]
        if include_tests:
            results.append(self.test_check())
        return results


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        lines.append(f"{mark} {result.name} ({result.duration_s}s)")
        if not result.ok:
            lines.append("  " + _crop(result.detail).replace("\n", "\n  "))
    return "\n".join(lines) or "no checks ran"
