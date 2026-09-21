"""Hermes backend agent adapter.

Hermes is JARVIS's specialist backend agent for deep planning and development
work. It is always optional: JARVIS starts and runs normally whether or not
Hermes is installed. The adapter detects a *real* interface at runtime and never
invents one.

Interface detection order:
1. ``HERMES_COMMAND`` environment variable (any executable or command prefix).
2. An executable named ``hermes`` on ``PATH`` (``shutil.which``).

CLI contract for the detected executable: JARVIS invokes the official Hermes
one-shot form ``<command> -z "<prompt>"``. The prompt is built as
``Focus: <focus>\\nRequest: <user request>`` so the focus (``ask``, ``plan``,
``analyse``, ``develop``) is carried inside the prompt text, never as a
separate command (``hermes ask`` etc. are not supported commands). Hermes must
answer on stdout; a non-zero exit code is a failure and empty stdout is treated
as malformed output.

Process and failure isolation:
- every call runs in its own subprocess via ``asyncio.create_subprocess_exec``
  (never a blocking call on the voice event loop);
- each call is bounded by ``HERMES_TIMEOUT_SECONDS`` (default 120) and the child
  is killed on timeout;
- there is no retry loop;
- timeout, crash, unavailable, and malformed output are returned as a
  :class:`HermesResult` and never raised into the voice loop.

``HERMES_COMMAND`` is read as an executable path first: the whole value is used
verbatim when it resolves to a file (important on Windows, where the path
``C:\\...\\hermes.exe`` must not be shell-parsed, which would strip the
backslashes). Only when it does not resolve is it split as an
executable-plus-arguments prefix such as ``python -m hermes_cli``.

Logging records only the op, outcome, and duration - never request content or
output, so secrets and user content stay out of the logs.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import shutil
import subprocess
import typing
from dataclasses import dataclass
from typing import Any, Literal

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

from gates import ToolGate
from integrations.base import ActionLevel, Integration

logger = logging.getLogger("agents.hermes")

HERMES_TIMEOUT_SECONDS = 120.0
HERMES_MAX_CHARS = 2000
HERMES_HISTORY_LIMIT = 20

_NO_WINDOW_FLAG = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

Outcome = Literal["unavailable", "success", "timeout", "error", "malformed"]


@dataclass(frozen=True)
class HermesResult:
    """Outcome of one Hermes backend call. Never thrown; always returned."""

    ok: bool
    outcome: Outcome
    text: str
    duration_ms: int
    note: str = ""


def _timeout_value(value: float | None) -> float:
    if value is not None:
        try:
            return max(0.1, float(value))
        except (TypeError, ValueError):
            return HERMES_TIMEOUT_SECONDS
    raw = os.environ.get("HERMES_TIMEOUT_SECONDS", "")
    try:
        return max(0.1, float(raw)) if raw.strip() else HERMES_TIMEOUT_SECONDS
    except ValueError:
        return HERMES_TIMEOUT_SECONDS


class HermesAgent:
    """Detects and drives the real Hermes interface with full isolation."""

    FOCUSES = ("ask", "plan", "analyse", "develop")

    def __init__(
        self,
        *,
        command: list[str] | None = None,
        timeout_s: float | None = None,
        max_chars: int = HERMES_MAX_CHARS,
    ) -> None:
        self._command = list(command) if command else None
        self._timeout_s = _timeout_value(timeout_s)
        self._max_chars = max_chars if max_chars and max_chars > 0 else HERMES_MAX_CHARS
        self._lock = asyncio.Lock()
        self.last_outcome: str | None = None
        self.last_error: str = ""
        self.last_duration_ms: int = 0
        self.history: list[tuple[str, str, int]] = []

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    # ------------------------------------------------------------------ #
    # detection
    # ------------------------------------------------------------------ #

    def _parse_command(self, raw: str) -> list[str] | None:
        """Turn a ``HERMES_COMMAND`` value into an argv list.

        Order matters: the whole value is first treated as a single executable
        path and used verbatim when it resolves. This keeps a plain Windows path
        like ``C:\\Users\\Me\\AppData\\Local\\hermes\\bin\\hermes.exe`` intact
        (shell-parsing it would strip the backslashes). Only when the value does
        not resolve is it split as an executable-plus-arguments prefix (e.g.
        ``python -m hermes_cli``); on Windows a second, backslash-preserving
        split is tried when the first one does not resolve.
        """
        if not raw.strip():
            return None
        if os.path.exists(raw) or shutil.which(raw):
            return [raw]
        candidates = [shlex.split(raw)]
        if os.name == "nt":
            candidates.append(shlex.split(raw, posix=False))
        for parts in candidates:
            if parts and (os.path.exists(parts[0]) or shutil.which(parts[0])):
                return parts
        return None

    def _resolve_command(self) -> list[str] | None:
        if self._command is not None:
            return list(self._command)
        raw = os.environ.get("HERMES_COMMAND", "")
        command = self._parse_command(raw)
        if command is not None:
            return command
        found = shutil.which("hermes")
        if found:
            return [found]
        return None

    def is_available(self) -> bool:
        return self._resolve_command() is not None

    def health_status(self) -> str:
        if not self.is_available():
            return "unavailable"
        if self.last_outcome in {"error", "timeout", "malformed"}:
            return "error"
        return "healthy"

    # ------------------------------------------------------------------ #
    # driving
    # ------------------------------------------------------------------ #

    async def ask(self, request: str) -> HermesResult:
        return await self.run("ask", request)

    async def plan(self, request: str) -> HermesResult:
        return await self.run("plan", request)

    async def analyse(self, request: str) -> HermesResult:
        return await self.run("analyse", request)

    async def develop(self, request: str) -> HermesResult:
        return await self.run("develop", request)

    async def run(self, focus: str, request: str) -> HermesResult:
        focus = focus if focus in self.FOCUSES else "ask"
        async with self._lock:
            command = self._resolve_command()

            if command is None:
                return self._record(focus, "unavailable", 0)

            if not (request or "").strip():
                return self._record(focus, "malformed", 0, note="empty request")

            prompt = f"Focus: {focus}\nRequest: {(request or '').strip()}"
            started = asyncio.get_running_loop().time()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    "-z",
                    prompt,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=_NO_WINDOW_FLAG,
                )
            except (OSError, ValueError) as exc:
                elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)
                return self._record(
                    focus, "error", elapsed_ms, note=f"could not start: {exc}"
                )

            try:
                stdout, _stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout_s
                )
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(OSError):
                    await proc.wait()
                for _s in (proc.stdout, proc.stderr):
                    if _s is None:
                        continue
                    transport = getattr(_s, "_transport", None)
                    if transport is not None:
                        transport.close()
                elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)
                return self._record(focus, "timeout", elapsed_ms, note="timed out")

            elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            if proc.returncode != 0:
                return self._record(
                    focus, "error", elapsed_ms, note=f"exit code {proc.returncode}"
                )

            text = stdout.decode("utf-8", "replace").strip()
            if not text:
                return self._record(focus, "malformed", elapsed_ms, note="no output")

            truncated = len(text) > self._max_chars
            return self._record(
                focus,
                "success",
                elapsed_ms,
                text=text[: self._max_chars],
                note="truncated" if truncated else "",
            )

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #

    def _record(
        self,
        focus: str,
        outcome: str,
        duration_ms: int,
        *,
        text: str = "",
        note: str = "",
    ) -> HermesResult:
        self.last_outcome = outcome
        self.last_duration_ms = int(duration_ms)
        self.last_error = note
        self.history.append((focus, outcome, int(duration_ms)))
        if len(self.history) > HERMES_HISTORY_LIMIT:
            del self.history[:-HERMES_HISTORY_LIMIT]
        logger.info(
            "hermes op=%s outcome=%s duration_ms=%d", focus, outcome, duration_ms
        )
        return HermesResult(
            ok=outcome == "success",
            outcome=outcome,
            text=text,
            duration_ms=int(duration_ms),
            note=note,
        )


class HermesIntegration(Integration):
    """Registers Hermes in the capability registry and exposes voice tools."""

    name = "hermes"
    description = (
        "Hermes backend agent for deep planning and development analysis. "
        "JARVIS stays the main assistant; Hermes answers behind it."
    )
    read_only = False
    required_env = ()
    default_level = ActionLevel.SAFE_READ
    _TOOLS = ("get_hermes_status", "ask_hermes")
    _LEVELS: typing.ClassVar[dict[str, ActionLevel]] = {
        "get_hermes_status": ActionLevel.SAFE_READ,
        "ask_hermes": ActionLevel.REVERSIBLE,
    }
    _STATUS_MESSAGES: typing.ClassVar[dict[str, str]] = {
        "unavailable": "Hermes is not available right now. Continue without it.",
        "timeout": "Hermes did not answer in time. Continue without it.",
        "malformed": "Hermes returned an empty answer. Continue without it.",
        "error": "Hermes failed. Continue without it.",
    }

    def __init__(
        self,
        *,
        gate: ToolGate | None = None,
        failure_log: Any | None = None,
        agent: HermesAgent | None = None,
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self.agent = agent or HermesAgent()

    def is_available(self) -> bool:
        return self.agent.is_available()

    def is_authenticated(self) -> bool:
        return self.agent.is_available()

    def status_note(self) -> str:
        health = self.agent.health_status()
        base = f"role: reasoning/development; health: {health}"
        if health == "unavailable":
            base += (
                ". Set HERMES_COMMAND or install the 'hermes' executable on "
                "PATH to enable Hermes."
            )
        return base

    async def health_check(self) -> dict[str, Any]:
        health = self.agent.health_status()
        base = f"Hermes is {health}."
        if health == "unavailable":
            base += " Set HERMES_COMMAND or install the hermes executable to enable it."
        elif health == "error":
            base += " The last Hermes invocation failed."
        return {
            "name": self.name,
            "ok": health == "healthy",
            "status": health,
            "detail": base,
        }

    # ------------------------------------------------------------------ #
    # voice tools
    # ------------------------------------------------------------------ #

    @function_tool()
    async def get_hermes_status(self, context: RunContext) -> dict[str, Any]:
        """Report the Hermes backend: whether it is available, its role, and
        its current health (healthy, unavailable, or error)."""
        self.gate.ensure_active_conversation()
        return {
            "name": "hermes",
            "available": self.agent.is_available(),
            "role": "reasoning/development",
            "status": self.agent.health_status(),
            "note": self.status_note(),
            "history": list(self.agent.history[-5:]),
        }

    @function_tool()
    async def ask_hermes(
        self, context: RunContext, focus: str = "ask", request: str = ""
    ) -> dict[str, Any]:
        """Delegate a deep planning or development request to the Hermes backend.

        Use Hermes only for deep planning, analysis, or development work, or when
        the user explicitly asks you to consult Hermes. Summarise its output in
        your own words; never present Hermes output as your own analysis. If
        Hermes is unavailable or fails, say so briefly and continue helping.

        Args:
            focus: The kind of task: 'plan', 'analyse', 'develop', or 'ask'.
            request: The full user request to delegate, in the user's words.
        """
        self.gate.ensure_hermes_requested()
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        focus = (focus or "ask").strip()
        if focus not in HermesAgent.FOCUSES:
            focus = "ask"
        request = str(request or "").strip()
        result = await self.agent.run(focus, request)
        if result.ok:
            return {
                "status": "ok",
                "focus": focus,
                "summary": result.text,
                "message": "Hermes returned the following; summarise it in your own words.",
            }
        note = result.note or ""
        message = self._STATUS_MESSAGES.get(result.outcome, "Hermes failed.")
        message += f" ({note})" if note else ""
        raise ToolError(message)
