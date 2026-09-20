"""Tests for the Hermes capability: registry status, gate enforcement, the
voice tools' failure isolation, and the Developer Mode planning hook.

Hermes must never break JARVIS: when it is missing or fails, status reflects
that and no tool raises into the voice loop.
"""

import asyncio
import sys

import pytest
from livekit.agents.llm import ToolError

from agents import hermes_agent as ha
from agents.hermes_agent import HermesResult
from agents.router import HermesRouter
from developer.coding_agent import Developer
from developer.task_manager import TaskManager
from gates import ToolGate, is_hermes_request
from integrations import build_default_registry


class _ActiveGate:
    """A gate that is always in an active conversation with a turn."""

    def __init__(self, turn: str = "Jarvis, use Hermes for this"):
        self.turn = turn

    def ensure_active_conversation(self) -> None:
        return None

    def ensure_action_level(self, level: int, requested: bool = False) -> None:
        return None

    def ensure_hermes_requested(self) -> None:
        return None


class _FakeHermes:
    def __init__(self, available=True, result=None):
        self._available = available
        self._result = result
        self.calls: list[tuple[str, str]] = []
        self.history: list[str] = []

    def is_available(self):
        return self._available

    def health_status(self):
        if not self._available:
            return "unavailable"
        if self._result is not None and not self._result.ok:
            return "error"
        return "healthy"

    async def run(self, focus, request):
        self.calls.append((focus, request))
        if not self._available:
            return HermesResult(
                ok=False,
                outcome="unavailable",
                text="",
                duration_ms=0,
                note="",
            )
        if self._result is not None:
            return self._result
        return HermesResult(
            ok=True,
            outcome="success",
            text=f"{focus} plan: {request}",
            duration_ms=5,
            note="",
        )

    async def ask(self, request):
        return await self.run("ask", request)

    async def plan(self, request):
        return await self.run("plan", request)

    async def analyse(self, request):
        return await self.run("analyse", request)

    async def develop(self, request):
        return await self.run("develop", request)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(ha.shutil, "which", lambda _name: None)


class TestGateIntent:
    def test_explicit_and_deep_phrases_recognised(self):
        assert is_hermes_request("Jarvis, ask Hermes to plan this") is True
        assert is_hermes_request("Jarvis, use Hermes for this") is True
        assert is_hermes_request("Jarvis, analyse the root cause deeply") is True
        assert is_hermes_request("Jarvis, what time is it") is False
        assert is_hermes_request("Jarvis, play some jazz") is False

    def test_gate_requires_explicit_or_deep_hermes_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, tell me a joke")
        with pytest.raises(ToolError, match="Hermes"):
            gate.ensure_hermes_requested()

        gate.set_user_request("Jarvis, ask Hermes to optimise the code")
        gate.ensure_hermes_requested()

        gate.set_user_request("Jarvis")
        with pytest.raises(ToolError, match="Hermes"):
            gate.ensure_hermes_requested()

    def test_gate_requires_active_conversation(self):
        gate = ToolGate()
        with pytest.raises(ToolError, match="wake word"):
            gate.ensure_hermes_requested()


class TestRegistryStatus:
    def test_registry_registers_hermes_as_unavailable(self):
        registry = build_default_registry()
        assert "hermes" in registry.names()
        integration = registry.get("hermes")
        assert integration is not None
        status = integration.status()
        assert status.available is False
        assert status.authenticated is False

    def test_health_reports_unavailable(self):
        integration = build_default_registry().get("hermes")
        health = run_sync(integration.health_check())
        assert health["status"] == "unavailable"
        assert health["ok"] is False

    def test_available_agent_is_reflected_in_registry(self):
        fake = _FakeHermes(available=True)
        registry = build_default_registry(hermes_agent=fake)
        integration = registry.get("hermes")
        status = integration.status()
        assert status.available is True
        assert status.authenticated is True

    def test_injected_agent_health_is_healthy(self):
        fake = _FakeHermes(available=True)
        integration = build_default_registry(hermes_agent=fake).get("hermes")
        health = run_sync(integration.health_check())
        assert health["status"] == "healthy"
        assert health["ok"] is True

    def test_capabilities_include_hermes(self):
        registry = build_default_registry(hermes_agent=_FakeHermes(available=True))
        caps = registry.capabilities()
        hermes = next(c for c in caps if c["name"] == "hermes")
        assert hermes["available"] is True
        assert "reasoning" in hermes["note"]

    def test_capabilities_show_unavailable(self):
        registry = build_default_registry()
        caps = registry.capabilities()
        hermes = next(c for c in caps if c["name"] == "hermes")
        assert hermes["available"] is False
        assert "HERMES_COMMAND" in hermes["note"]
        assert "health" in hermes["note"]

    def test_status_tool_reports_role_and_health(self):
        fake = _FakeHermes(available=True)
        registry = build_default_registry(gate=_ActiveGate(), hermes_agent=fake)
        integration = registry.get("hermes")
        result = run_sync(integration.get_hermes_status(None))
        assert result["available"] is True
        assert result["role"] == "reasoning/development"
        assert result["status"] == "healthy"
        assert integration.required_env == ()

    def test_unavailable_status_tool_is_honest(self):
        registry = build_default_registry(gate=_ActiveGate())
        integration = registry.get("hermes")
        result = run_sync(integration.get_hermes_status(None))
        assert result["available"] is False
        assert result["status"] == "unavailable"

    def test_registry_reuses_the_single_hermes_backend(self):
        """B1 regression - there must be exactly ONE HermesAgent behind JARVIS.

        agent.py wires `hermes_agent` into the Developer (hermes=...). The voice
        registry must reuse that same instance, so voice tools and Developer mode
        share one backend, one lock, and one health history.
        """
        import agent as agent_module

        integration = agent_module.capability_registry.get("hermes")
        assert integration.agent is agent_module.hermes_agent

    def test_empty_request_never_spawns_a_subprocess(self, monkeypatch):
        """B2 regression - an empty/whitespace request must not spawn Hermes."""
        spawned = []

        async def _forbidden_spawn(*args, **kwargs):
            spawned.append(args)
            raise AssertionError("empty request must not spawn a subprocess")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _forbidden_spawn)

        from agents.hermes_agent import HermesAgent

        agent = HermesAgent(command=[sys.executable, "unused-placeholder.py"])
        result = run_sync(agent.run("ask", "   "))
        assert result.outcome == "malformed"
        assert result.ok is False
        assert spawned == []


class TestVoiceTools:
    def test_ask_hermes_delegates_planned_focus(self):
        fake = _FakeHermes(available=True)
        integration = _make_integration(fake)
        result = run_sync(
            integration.ask_hermes(None, focus="plan", request="refactor the router")
        )
        assert fake.calls == [("plan", "refactor the router")]
        assert result["status"] == "ok"
        assert "refactor the router" in result["summary"]

    def test_ask_hermes_unknown_focus_falls_back_to_ask(self):
        fake = _FakeHermes(available=True)
        integration = _make_integration(fake)
        result = run_sync(
            integration.ask_hermes(None, focus="dance", request="what shall we do")
        )
        assert fake.calls == [("ask", "what shall we do")]
        assert result["status"] == "ok"

    def test_ask_hermes_blocked_without_explicit_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, what time is it")
        integration = _make_integration(_FakeHermes(available=True), gate=gate)
        with pytest.raises(ToolError, match="Hermes"):
            run_sync(integration.ask_hermes(None, focus="plan", request="x"))

    def test_ask_hermes_blocked_without_wake_word(self):
        integration = _make_integration(_FakeHermes(available=True), gate=ToolGate())
        with pytest.raises(ToolError, match="wake word"):
            run_sync(integration.ask_hermes(None, focus="plan", request="x"))

    def test_unavailable_does_not_raise(self):
        integration = _make_integration(_FakeHermes(available=False))
        result = run_sync(integration.ask_hermes(None, focus="plan", request="x"))
        assert result["status"] == "unavailable"
        assert "not available" in result["message"].casefold()

    def test_error_result_does_not_raise(self):
        fake = _FakeHermes(
            available=True,
            result=HermesResult(
                ok=False, outcome="error", text="", duration_ms=5, note="exit code 3"
            ),
        )
        integration = _make_integration(fake)
        result = run_sync(integration.ask_hermes(None, focus="plan", request="x"))
        assert result["status"] == "error"
        assert "exit code 3" in result["message"]

    def test_timeout_does_not_raise(self):
        fake = _FakeHermes(
            available=True,
            result=HermesResult(
                ok=False, outcome="timeout", text="", duration_ms=900, note=""
            ),
        )
        integration = _make_integration(fake)
        result = run_sync(integration.ask_hermes(None, focus="develop", request="x"))
        assert result["status"] == "timeout"

    def test_malformed_does_not_raise(self):
        fake = _FakeHermes(
            available=True,
            result=HermesResult(
                ok=False, outcome="malformed", text="", duration_ms=3, note=""
            ),
        )
        integration = _make_integration(fake)
        result = run_sync(integration.ask_hermes(None, focus="analyse", request="x"))
        assert result["status"] == "malformed"

    def test_ask_hermes_is_reversible_level_with_status_read(self):
        integration = _make_integration(_FakeHermes(available=True))
        assert int(integration.tool_level("ask_hermes")) == 2
        assert int(integration.tool_level("get_hermes_status")) == 1
        assert "ask_hermes" in integration.tool_names
        assert "get_hermes_status" in integration.tool_names


class TestDeveloperModeIntegration:
    def _developer(self, fake):
        return Developer(project=None, hermes=fake)

    @pytest.mark.asyncio
    async def test_plan_is_included_when_available(self, tmp_path):
        tasks = TaskManager(tmp_path / "improvement_queue")
        task = tasks.create_task(
            title="Refactor router",
            description="refactor src/agents/router.py",
            status="queued",
        )
        fake = _FakeHermes(available=True)
        developer = self._developer(fake)
        context = await developer._maybe_hermes_context(task)
        assert "Refactor router" in context
        assert fake.calls == [
            ("plan", "Refactor router\n\nrefactor src/agents/router.py")
        ]

    @pytest.mark.asyncio
    async def test_failing_hermes_falls_back_to_empty(self, tmp_path):
        tasks = TaskManager(tmp_path / "improvement_queue")
        task = tasks.create_task(
            title="Refactor router",
            description="refactor src/agents/router.py",
            status="queued",
        )
        fake = _FakeHermes(
            available=True,
            result=HermesResult(
                ok=False, outcome="error", text="", duration_ms=2, note="exit code 9"
            ),
        )
        developer = self._developer(fake)
        context = await developer._maybe_hermes_context(task)
        assert context == ""

    @pytest.mark.asyncio
    async def test_no_hermes_skips_planning(self, tmp_path):
        tasks = TaskManager(tmp_path / "improvement_queue")
        task = tasks.create_task(
            title="Refactor router",
            description="refactor src/agents/router.py",
            status="queued",
        )
        developer = Developer(project=None)
        assert developer.hermes is None
        context = await developer._maybe_hermes_context(task)
        assert context == ""

    @pytest.mark.asyncio
    async def test_pathological_hermes_exception_never_escapes(self, tmp_path):
        tasks = TaskManager(tmp_path / "improvement_queue")
        task = tasks.create_task(
            title="Refactor router",
            description="refactor src/agents/router.py",
            status="queued",
        )

        class _Boom:
            def is_available(self):
                return True

            async def plan(self, request):
                raise RuntimeError("hermes exploded")

        developer = self._developer(_Boom())
        context = await developer._maybe_hermes_context(task)
        assert context == ""


def _make_integration(fake, gate=None):
    from agents.hermes_agent import HermesIntegration

    return HermesIntegration(gate=gate or _ActiveGate(), agent=fake)


def run_sync(coro):
    return asyncio.run(coro)


class _ProbeRouter:
    def route(self, text):
        return HermesRouter().route(text)
