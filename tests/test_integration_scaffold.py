"""Tests for deterministic integration scaffolding in Developer Mode.

The generated package must be valid, standards-compliant, lint-clean, and
importable from the project layout, and the whole proposal must flow through the
normal Developer pipeline without any LLM involvement.
"""

import subprocess
import sys

import pytest

from developer.coding_agent import Developer
from developer.git_manager import GitManager
from developer.integration_scaffold import (
    IntegrationScaffold,
    classify_name,
    is_integration_scaffold_request,
    service_name_from_request,
    slugify,
)
from developer.project_tools import ProjectAccess
from developer.task_manager import TaskManager
from developer.test_runner import CheckResult


def _ok(name="pytest"):
    return CheckResult(name=name, ok=True, detail="", duration_s=0.0)


class RaisingLLM:
    """Fails the test if the orchestrator tries to call a language model."""

    def complete(self, system, user):
        raise AssertionError("scaffold paths must never call the LLM")


class _FakeRunner:
    def __init__(self, ok=True):
        self._ok = ok
        self.runs = 0

    def run_all(self, *, include_tests=True):
        self.runs += 1
        results = [_ok(n) for n in ("lint", "format", "import", "pytest")]
        if not self._ok:
            results[0] = CheckResult("lint", False, "broken", 0.1)
        return results


class TestScaffoldNaming:
    def test_slugify(self):
        assert slugify("Philips Hue") == "philips_hue"
        assert slugify("  Strava%") == "strava"
        assert slugify("3D Printers") == "svc_3d_printers"

    def test_slugify_rejects_empty(self):
        with pytest.raises(ValueError):
            slugify("   ")

    def test_classify_name(self):
        assert classify_name("Philips Hue") == "PhilipsHue"
        assert classify_name("open-weather map") == "OpenWeatherMap"


class TestScaffoldDetection:
    def test_detects_scaffold_request(self):
        assert is_integration_scaffold_request("create an integration for Philips Hue")
        assert (
            service_name_from_request("create an integration for Philips Hue")
            == "Philips Hue"
        )

    def test_detects_alternative_form(self):
        assert (
            service_name_from_request("developer mode: add a new Strava integration")
            == "Strava"
        )

    def test_rejects_non_integration_request(self):
        assert is_integration_scaffold_request("please add src/greeting.py") is False

    def test_missing_service_raises(self):
        with pytest.raises(ValueError, match="service"):
            service_name_from_request("improve the code")


class TestProposalFor:
    def test_proposal_shape(self):
        proposal = IntegrationScaffold.proposal_for("Philips Hue", "ignore me")
        assert "summary" in proposal
        edits = proposal["edits"]
        assert len(edits) == 5
        paths = {edit["path"] for edit in edits}
        assert paths == {
            "src/integrations/philips_hue/__init__.py",
            "src/integrations/philips_hue/client.py",
            "src/integrations/philips_hue/integration.py",
            "tests/test_integrations_philips_hue.py",
            "docs/integrations/philips_hue.md",
        }
        assert all(edit["op"] == "create" for edit in edits)

    def test_generated_python_is_valid(self):
        proposal = IntegrationScaffold.proposal_for("Philips Hue", "x")
        for edit in proposal["edits"]:
            if edit["path"].endswith(".py"):
                compile(edit["new"], edit["path"], "exec")

    def test_no_placeholders_left(self):
        proposal = IntegrationScaffold.proposal_for("Philips Hue", "x")
        for edit in proposal["edits"]:
            assert "__SLUG__" not in edit["new"]
            assert "__CLASS__" not in edit["new"]
            assert "__ENVVAR__" not in edit["new"]

    def test_generated_code_is_ruff_clean(self, tmp_path):
        proposal = IntegrationScaffold.proposal_for("Nest Thermostat", "x")
        src = tmp_path / "src"
        src.mkdir()
        for edit in proposal["edits"]:
            if edit["path"].endswith(".py"):
                target = tmp_path / edit["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit["new"], encoding="utf-8")
        check = subprocess.run(
            [sys.executable, "-m", "ruff", "check", str(src)],
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stdout
        fmt = subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--check", str(src)],
            capture_output=True,
            text=True,
        )
        assert fmt.returncode == 0, fmt.stdout


class TestScaffoldPipeline:
    """The scaffold proposal must run through Developer Mode end-to-end, offline."""

    def _make_developer(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        subprocess.run(
            ["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Jarvis Test"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "jarvis@test.local"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        (root / "README.md").write_text("# demo\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "agent.py").write_text("VERSION = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True
        )
        runner = _FakeRunner()
        developer = Developer(
            project=ProjectAccess(root),
            tasks=TaskManager(root / "improvement_queue"),
            git=GitManager(root),
            runner=runner,
            llm=RaisingLLM(),
            root=root,
        )
        return developer, root

    def test_scaffold_flows_through_pipeline_without_llm(self, tmp_path):
        developer, root = self._make_developer(tmp_path)
        task = developer.tasks.create_task(
            title="Create one integration",
            description="create an integration for Philips Hue",
            status="queued",
        )
        task = run_sync(developer.start_task(task))
        assert task.status == "needs_approval"
        assert task.next_action == "merge"
        assert (
            root / "src" / "integrations" / "philips_hue" / "integration.py"
        ).exists()
        assert (root / "src" / "integrations" / "philips_hue" / "client.py").exists()
        assert (root / "tests" / "test_integrations_philips_hue.py").exists()
        assert (root / "docs" / "integrations" / "philips_hue.md").exists()

        merged = developer.merge_task(task)
        assert merged.status == "completed"

    def test_scaffold_branch_cleans_up_on_reject(self, tmp_path):
        developer, root = self._make_developer(tmp_path)
        task = developer.tasks.create_task(
            title="Create one integration",
            description="create an integration for Strava",
            status="queued",
        )
        run_sync(developer.start_task(task))
        assert git_branch_exists(developer.git, f"jarvis-dev/{task.id}")
        rejected = developer.reject_task(task)
        assert rejected.status == "rejected"
        assert not (root / "src" / "integrations" / "strava").exists()


class TestFrameworkGuide:
    def test_guide_mentions_action_levels(self):
        from developer.integration_scaffold import framework_guide

        guide = framework_guide()
        assert "SAFE_READ" in guide
        assert "CONSEQUENTIAL" in guide
        assert "_TOOLS" in guide


def git_branch_exists(git, branch_name: str) -> bool:
    return git.branch_exists(branch_name)


def run_sync(coro):
    import asyncio

    return asyncio.run(coro)
