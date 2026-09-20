"""Behaviour tests for the Hermes backend adapter.

The adapter must detect a real Hermes interface at runtime, run it as an
isolated subprocess, enforce a configurable timeout, and never throw into the
voice loop. It must log op / outcome / duration only, never request content.
"""

import asyncio
import logging
import sys

import pytest

from agents import hermes_agent as ha
from agents.hermes_agent import HermesAgent, HermesResult

DEFAULT_TIMEOUT = 120.0


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_TIMEOUT_SECONDS", raising=False)


def _script(tmp_path, body: str, name: str = "hermes.py") -> str:
    path = tmp_path / name
    path.write_text(
        "import sys, time\nop = sys.argv[-1]\nbody = sys.stdin.read()\n" + body,
        encoding="utf-8",
    )
    return str(path)


def _no_hermes_on_path(monkeypatch):
    monkeypatch.setattr(ha.shutil, "which", lambda _name: None)


ECHO = "sys.stdout.write(f'{op}:{body}')\nsys.stdout.flush()\n"
EMPTY = "pass\n"
FAIL = "sys.stderr.write('boom detail\\n')\nsys.exit(3)\n"
SLEEP_AND_MARK = "open(sys.argv[-2], 'a').write('x')\ntime.sleep(30)\n"


class TestAvailability:
    def test_unavailable_when_not_installed(self, monkeypatch):
        _no_hermes_on_path(monkeypatch)
        assert HermesAgent().is_available() is False

    def test_env_command_enables(self, monkeypatch):
        _no_hermes_on_path(monkeypatch)
        monkeypatch.setenv("HERMES_COMMAND", "hermes-agent")
        assert HermesAgent().is_available() is True

    def test_injected_command_enables(self, tmp_path):
        script = _script(tmp_path, ECHO)
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        assert agent.is_available() is True

    def test_which_based_detection(self, monkeypatch):
        monkeypatch.setattr(ha.shutil, "which", lambda name: "C:/hermes/hermes.exe")
        assert HermesAgent().is_available() is True

    def test_default_timeout_seconds(self):
        assert HermesAgent().timeout_s == DEFAULT_TIMEOUT

    def test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_TIMEOUT_SECONDS", "7")
        assert HermesAgent().timeout_s == 7.0


class TestExecution:
    @pytest.mark.asyncio
    async def test_ask_passes_op_and_request(self, tmp_path):
        script = _script(tmp_path, ECHO)
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        result = await agent.ask("refactor the router")
        assert result.ok is True
        assert result.outcome == "success"
        assert result.text == "ask:refactor the router"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("focus", ["ask", "plan", "analyse", "develop"])
    async def test_all_ops_dispatch_their_focus(self, tmp_path, focus):
        script = _script(tmp_path, ECHO)
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        result = await getattr(agent, focus)("design the module")
        assert result.ok is True
        assert result.text == f"{focus}:design the module"

    @pytest.mark.asyncio
    async def test_empty_output_is_malformed(self, tmp_path):
        script = _script(tmp_path, EMPTY)
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        result = await agent.plan("anything")
        assert result.ok is False
        assert result.outcome == "malformed"

    @pytest.mark.asyncio
    async def test_nonzero_exit_is_error_without_stderr_leak(self, tmp_path):
        script = _script(tmp_path, FAIL)
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        result = await agent.analyse("why is this failing")
        assert result.ok is False
        assert result.outcome == "error"
        assert result.text == ""
        assert result.note == "exit code 3"
        assert "boom detail" not in result.note

    @pytest.mark.asyncio
    async def test_timeout_is_handled(self, tmp_path):
        script = _script(tmp_path, SLEEP_AND_MARK, name="slow1.py")
        agent = HermesAgent(command=[sys.executable, script], timeout_s=0.4)
        result = await agent.develop("build long thing")
        assert result.ok is False
        assert result.outcome == "timeout"
        assert result.text == ""
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_no_infinite_retry(self, tmp_path):
        marker = tmp_path / "spawns.txt"
        script = _script(tmp_path, SLEEP_AND_MARK, name="slow2.py")
        agent = HermesAgent(
            command=[sys.executable, script, str(marker)], timeout_s=0.4
        )
        await agent.ask("slow task")
        assert marker.read_text() == "x"

    @pytest.mark.asyncio
    async def test_success_updates_status(self, tmp_path):
        script = _script(tmp_path, ECHO, name="ok.py")
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        await agent.ask("hello")
        assert agent.is_available() is True
        assert agent.health_status() == "healthy"
        assert agent.last_outcome == "success"
        assert agent.last_error == ""
        assert len(agent.history) == 1

    @pytest.mark.asyncio
    async def test_error_updates_health(self, tmp_path):
        script = _script(tmp_path, FAIL, name="bad.py")
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        await agent.ask("hello")
        assert agent.health_status() == "error"


class TestIsolation:
    @pytest.mark.asyncio
    async def test_unavailable_call_does_not_raise(self, monkeypatch):
        _no_hermes_on_path(monkeypatch)
        agent = HermesAgent()
        result = await agent.ask("anything")
        assert result.ok is False
        assert result.outcome == "unavailable"
        assert agent.health_status() == "unavailable"

    @pytest.mark.asyncio
    async def test_loop_stays_responsive_during_long_call(self, tmp_path):
        import time as _time

        script = tmp_path / "slow3.py"
        script.write_text("import time; time.sleep(30)\n", encoding="utf-8")
        agent = HermesAgent(command=[sys.executable, str(script)], timeout_s=0.6)
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.05)

        tick = asyncio.create_task(ticker())
        started = _time.monotonic()
        result = await agent.ask("slow")
        elapsed = _time.monotonic() - started
        done.set()
        await tick
        assert result.outcome == "timeout"
        assert ticks > 0
        assert elapsed >= 0.5

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_serialized(self, tmp_path):
        script = tmp_path / "slowish.py"
        script.write_text(
            "import sys, time; time.sleep(0.3); sys.stdout.write('done')\n",
            encoding="utf-8",
        )
        agent = HermesAgent(command=[sys.executable, str(script)], timeout_s=10)
        results = await asyncio.gather(agent.ask("a"), agent.ask("b"))
        assert all(result.ok for result in results)
        assert sum(result.duration_ms for result in results) >= 500


class TestLogging:
    @pytest.mark.asyncio
    async def test_logs_metadata_but_never_content(self, tmp_path, caplog):
        script = _script(tmp_path, ECHO, name="logger.py")
        agent = HermesAgent(command=[sys.executable, script], timeout_s=10)
        with caplog.at_level(logging.INFO, logger="agents.hermes"):
            await agent.ask("SUPER_SECRET_REQUEST_XYZ")
        assert "SUPER_SECRET_REQUEST_XYZ" not in caplog.text
        assert "ask" in caplog.text
        assert "outcome=success" in caplog.text
        assert "duration_ms" in caplog.text

    def test_result_is_a_named_type(self):
        result = HermesResult(ok=False, outcome="unavailable", text="", duration_ms=0)
        assert result.outcome == "unavailable"
