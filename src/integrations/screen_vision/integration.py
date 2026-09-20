"""Conservative, read-only screen vision for JARVIS.

The tools here are entirely on-demand: nothing captures continuously, nothing
runs in the background, and every blocking call (monitor enumeration, cursor
lookup, screen grab, image encoding) happens in a worker thread via
``asyncio.to_thread`` so the realtime voice pipeline is never blocked. Any
failure degrades to a normal :class:`ToolError` and never crashes the agent
session. None of the tools mutate anything - they read the screen and return
text.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, ClassVar

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

from integrations.base import ActionLevel, Integration
from integrations.screen_vision import capture
from integrations.screen_vision.analysis import VisionAnalyzer, analyse_captured
from integrations.screen_vision.backend import default_backend

_NOT_GIVEN = object()

_LEVELS = dict.fromkeys(
    (
        "screen_status",
        "list_monitors",
        "get_cursor_position",
        "capture_monitor",
        "capture_cursor_area",
        "analyze_screen_on_demand",
    ),
    ActionLevel.SAFE_READ,
)

_MESSAGES = {
    "unavailable": (
        "Screen vision is not available on this system, so I cannot see your "
        "screen. I will keep helping you normally."
    ),
    "capture": "I could not capture that part of your screen just now.",
    "cursor": "I could not read your cursor position just now.",
    "monitors": "I could not enumerate your monitors just now.",
    "analysis": "I could not analyse that image just now.",
}


class ScreenVisionIntegration(Integration):
    """Read-only, on-demand screen inspection exposed as an integration."""

    name = "screen_vision"
    description = (
        "Read the user's screen only when explicitly asked: enumerate monitors, "
        "report the cursor, capture a monitor or the area around the cursor, "
        "and answer text questions about what is on screen."
    )
    read_only = True
    default_level = ActionLevel.SAFE_READ
    _TOOLS = tuple(_LEVELS)  # type: ignore[assignment]
    _LEVELS: ClassVar[dict[str, ActionLevel]] = _LEVELS

    def __init__(
        self,
        *,
        gate: Any | None = None,
        failure_log: Any | None = None,
        backend: Any = _NOT_GIVEN,
        analyzer: VisionAnalyzer | None = None,
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self._backend: Any = _NOT_GIVEN if backend is _NOT_GIVEN else backend
        self._analyzer = analyzer
        if sys.platform != "win32":
            self._backend = None  # capture is only implemented on Windows

    # ------------------------------------------------------------------ #
    # capability --------------------------------------------------- #
    # ------------------------------------------------------------------ #

    def _resolve_backend(self) -> Any:
        if self._backend is _NOT_GIVEN:
            self._backend = default_backend()
        return self._backend

    def is_available(self) -> bool:
        backend = self._resolve_backend()
        if backend is None:
            return False
        try:
            backend.list_monitors()
            return True
        except Exception:
            return False

    def is_authenticated(self) -> bool:
        return self.is_available()

    def status_note(self) -> str:
        if not self.is_available():
            return "requires Windows with a working screen capture; disabled here"
        return "screen capture ready; analysis needs GOOGLE_API_KEY"

    def _require_backend(self) -> Any:
        backend = self._resolve_backend()
        if backend is None:
            raise ToolError(_MESSAGES["unavailable"])
        return backend

    # ------------------------------------------------------------------ #
    # helpers ------------------------------------------------------- #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _monitor_by_id(
        monitors: list[dict[str, Any]], monitor_id: int
    ) -> dict[str, Any]:
        for monitor in monitors:
            if monitor["id"] == monitor_id:
                return monitor
        raise ToolError(
            f"I could not find a monitor with id {monitor_id}. "
            "Call list_monitors to see what is available."
        )

    async def _run_threaded(self, tool_name: str, fn: Any, *args: Any) -> Any:
        try:
            return await asyncio.to_thread(fn, *args)
        except ToolError:
            raise
        except Exception as exc:
            self._log_failure(tool=tool_name, target=str(args[:2]), exc=exc)
            raise ToolError(_MESSAGES.get(tool_name, _MESSAGES["capture"])) from exc

    async def _save_image(self, image: Any) -> Path:
        try:
            return await asyncio.to_thread(capture.save_temp_image, image)
        except Exception as exc:
            self._log_failure(tool="capture", target="save screenshot", exc=exc)
            raise ToolError(_MESSAGES["capture"]) from exc

    async def _capture_monitor(self, monitor: dict[str, Any]) -> Path:
        bbox = capture.monitor_bbox(monitor)
        backend = self._require_backend()
        image = await self._run_threaded("capture", lambda: backend.capture(bbox))
        return await self._save_image(image)

    async def _capture_cursor_area(self, width: int, height: int) -> Path:
        backend = self._require_backend()
        cursor = await self._run_threaded(
            "get_cursor_position", backend.get_cursor_position
        )
        monitor_list = await self._run_threaded("list_monitors", backend.list_monitors)
        bbox = capture.cursor_region(monitor_list, cursor, width, height)
        if bbox is None:
            raise ToolError("I could not find a screen area around your cursor.")
        image = await self._run_threaded("capture", lambda: backend.capture(bbox))
        return await self._save_image(image)

    # ------------------------------------------------------------------ #
    # tools ---------------------------------------------------------- #
    # ------------------------------------------------------------------ #

    @function_tool()
    async def screen_status(self, context: RunContext) -> dict[str, Any]:
        """Report what screen-vision capability is available."""
        self.gate.ensure_active_conversation()
        backend = self._resolve_backend()
        status: dict[str, Any] = {
            "name": self.name,
            "available": backend is not None,
            "vision_configured": False,
        }
        if backend is not None:
            try:
                status["monitors"] = await self._run_threaded(
                    "list_monitors", backend.list_monitors
                )
                status["cursor"] = await self._run_threaded(
                    "get_cursor_position", backend.get_cursor_position
                )
            except ToolError:
                status["monitors"] = None
                status["cursor"] = None
            analyzer = self._analyzer or VisionAnalyzer()
            status["vision_configured"] = analyzer.configured()
        return status

    @function_tool()
    async def list_monitors(self, context: RunContext) -> list[dict[str, Any]]:
        """Enumerate the connected monitors and their sizes/positions. Use this
        before capturing a specific monitor. Read-only."""
        self.gate.ensure_active_conversation()
        backend = self._require_backend()
        return await self._run_threaded("list_monitors", backend.list_monitors)

    @function_tool()
    async def get_cursor_position(self, context: RunContext) -> dict[str, int]:
        """Report the current mouse cursor position in screen coordinates.
        Read-only."""
        self.gate.ensure_active_conversation()
        backend = self._require_backend()
        return await self._run_threaded(
            "get_cursor_position", backend.get_cursor_position
        )

    @function_tool()
    async def capture_monitor(
        self, context: RunContext, monitor_id: int = 0
    ) -> dict[str, Any]:
        """Capture the monitor with the given id (default: primary) and save a
        temporary screenshot. Returns the image path and monitor info. This is
        read-only; use it when the user says things like 'look at my screen'.
        Call list_monitors first to pick a monitor id."""
        self.gate.ensure_active_conversation()
        backend = self._require_backend()
        monitor_list = await self._run_threaded("list_monitors", backend.list_monitors)
        monitor = self._monitor_by_id(monitor_list, monitor_id)
        path = await self._capture_monitor(monitor)
        return {
            "monitor": {
                "id": monitor["id"],
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"],
                "primary": monitor["primary"],
            },
            "image_path": str(path),
        }

    @function_tool()
    async def capture_cursor_area(
        self, context: RunContext, width: int = 800, height: int = 600
    ) -> dict[str, Any]:
        """Capture an area centered on the cursor and save a temporary
        screenshot. Use this when the user points at something on screen
        ('what is this?', 'look where my cursor is'). Read-only."""
        self.gate.ensure_active_conversation()
        path = await self._capture_cursor_area(width, height)
        return {"image_path": str(path), "area": "cursor"}

    @function_tool()
    async def analyze_screen_on_demand(
        self,
        context: RunContext,
        question: str,
        image_path: str = "",
        monitor_id: int | None = None,
        cursor_area: bool = False,
    ) -> dict[str, Any]:
        """Answer a text question about what is on the screen. Captures the
        primary monitor by default, the area around the cursor when
        cursor_area=True, a specific monitor when monitor_id is given, or
        analyses an existing screenshot passed via image_path. Returns a short
        text answer. Only call this when the user asks you to look at their
        screen. Read-only."""
        self.gate.ensure_active_conversation()
        analyzer = self._analyzer or VisionAnalyzer()
        owned_path: Path | None = None
        try:
            if image_path:
                resolved = Path(image_path)
                if not resolved.exists():
                    raise ToolError("That image path does not exist.")
            elif cursor_area:
                owned_path = await self._capture_cursor_area(800, 600)
            else:
                backend = self._require_backend()
                monitor_list = await self._run_threaded(
                    "list_monitors", backend.list_monitors
                )
                monitor_id = 0 if monitor_id is None else monitor_id
                monitor = self._monitor_by_id(monitor_list, monitor_id)
                owned_path = await self._capture_monitor(monitor)
            target = owned_path if owned_path is not None else Path(image_path)
            analysis = await analyse_captured(analyzer, target, question)
            return {"analysis": analysis}
        except ToolError:
            raise
        except Exception as exc:
            self._log_failure(tool="analyze_screen_on_demand", target=question, exc=exc)
            raise ToolError(_MESSAGES["analysis"]) from exc
        finally:
            if owned_path is not None:
                await asyncio.to_thread(capture.remove_temp_image, owned_path)
