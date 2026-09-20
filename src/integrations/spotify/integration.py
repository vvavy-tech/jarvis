"""Spotify capability.

JARVIS controls Spotify through the best available mechanism in order:
(1) the Spotify desktop app + Windows media keys when installed, (2) a future
Spotify Web API connection. The API is deliberately NOT required for basic
local control to work. When the API is later connected, the conversational
interface stays identical.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

from integrations.base import ActionLevel, Integration
from integrations.media import (
    ON_WINDOWS,
    MediaControlError,
    launch_spotify,
    send_media_key,
    send_volume_delta,
    spotify_health,
    terminate_spotify,
)
from integrations.spotify.client import SpotifyApiClient

MAX_VOLUME_STEPS = 20


class SpotifyIntegration(Integration):
    name = "spotify"
    description = (
        "Control Spotify: open the app, play/pause, skip tracks, and adjust "
        "volume either through the local desktop app (media keys) or the "
        "Spotify Web API when connected."
    )
    read_only = False
    default_level = ActionLevel.SAFE_READ
    _TOOLS = (
        "spotify_status",
        "open_spotify",
        "restart_spotify",
        "media_play_pause",
        "media_next",
        "media_previous",
        "volume_up",
        "volume_down",
        "media_mute",
    )
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {
        "spotify_status": ActionLevel.SAFE_READ,
        "open_spotify": ActionLevel.REVERSIBLE,
        "restart_spotify": ActionLevel.REVERSIBLE,
        "media_play_pause": ActionLevel.REVERSIBLE,
        "media_next": ActionLevel.REVERSIBLE,
        "media_previous": ActionLevel.REVERSIBLE,
        "volume_up": ActionLevel.REVERSIBLE,
        "volume_down": ActionLevel.REVERSIBLE,
        "media_mute": ActionLevel.REVERSIBLE,
    }

    def __init__(
        self,
        *,
        gate=None,
        failure_log=None,
        client: SpotifyApiClient | None = None,
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self._api = client or SpotifyApiClient()

    # ------------------------------------------------------------------ #
    # capability state
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        return True

    def is_authenticated(self) -> bool:
        return self._api.is_authenticated()

    def status_note(self) -> str:
        if self.is_authenticated():
            return ""
        if ON_WINDOWS:
            return "local control available, API not connected"
        return "available but not connected"

    async def health_check(self) -> dict[str, Any]:
        base = await super().health_check()
        if self.is_authenticated():
            return base
        return {
            "name": self.name,
            "ok": ON_WINDOWS,
            "status": "local_control" if ON_WINDOWS else "not_connected",
            "detail": (
                "Local media control is available; Spotify Web API is not connected."
                if ON_WINDOWS
                else "Spotify is available but neither local nor API control is active."
            ),
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _require_local(self, tool: str) -> None:
        if not ON_WINDOWS:
            raise ToolError(
                "Local Spotify control is not available on this platform. "
                "Connect the Spotify Web API to control music here."
            )

    def _run_local(self, tool: str, action: str) -> dict[str, str]:
        try:
            send_media_key(action)
        except MediaControlError as exc:
            self._log_failure(tool, action, exc)
            raise ToolError(str(exc)) from exc
        return {"action": action, "mechanism": "local_media_keys"}

    @staticmethod
    def _focus_spotify_window() -> None:
        if not ON_WINDOWS:
            return
        try:
            from browser_tools import _focus_window_with_title

            _focus_window_with_title("Spotify", timeout_seconds=1.0)
        except Exception:
            return

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #

    @function_tool()
    async def spotify_status(self, context: RunContext) -> dict[str, Any]:
        """Report how Spotify is currently reachable (local app, API, or neither)
        and, locally, whether the app is running and responding."""
        self.gate.ensure_active_conversation()
        status = self.status()
        health = None
        if ON_WINDOWS:
            try:
                health = await asyncio.to_thread(spotify_health)
            except MediaControlError as exc:
                self._log_failure("spotify_status", "spotify", exc)
                health = {"error": str(exc), "running": None, "responding": None}
        return {
            "name": self.name,
            "available": status.available,
            "local_control": ON_WINDOWS,
            "api_connected": status.authenticated,
            "read_only": self.read_only,
            "note": status.note
            or ("connected" if status.authenticated else "not connected"),
            "process_count": health.get("process_count") if health else None,
            "running": health.get("running") if health else None,
            "responding": health.get("responding") if health else None,
        }

    @function_tool()
    async def restart_spotify(self, context: RunContext) -> dict[str, Any]:
        """Restart Spotify: stop a hung or running instance, then launch it
        again and bring its window to the front."""
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("restart_spotify")
        try:
            before = await asyncio.to_thread(spotify_health)
            killed = await asyncio.to_thread(terminate_spotify)
            await asyncio.sleep(1.0)
            result = launch_spotify()
        except MediaControlError as exc:
            self._log_failure("restart_spotify", "spotify", exc)
            raise ToolError(str(exc)) from exc
        self._focus_spotify_window()
        return {
            "was_running": bool(before.get("running")),
            "was_responding": before.get("responding"),
            "killed_processes": killed,
            "launched": bool(result.get("launched")),
        }

    @function_tool()
    async def open_spotify(self, context: RunContext) -> dict[str, str]:
        """Open the Spotify desktop application."""
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("open_spotify")
        try:
            result = launch_spotify()
        except MediaControlError as exc:
            self._log_failure("open_spotify", "spotify", exc, fallback="api")
            raise ToolError(str(exc)) from exc
        self._focus_spotify_window()
        return result

    @function_tool()
    async def media_play_pause(self, context: RunContext) -> dict[str, str]:
        """Play or pause the current Spotify track."""
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("media_play_pause")
        self._focus_spotify_window()
        return self._run_local("media_play_pause", "play_pause")

    @function_tool()
    async def media_next(self, context: RunContext) -> dict[str, str]:
        """Skip to the next Spotify track."""
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("media_next")
        self._focus_spotify_window()
        return self._run_local("media_next", "next")

    @function_tool()
    async def media_previous(self, context: RunContext) -> dict[str, str]:
        """Go back to the previous Spotify track."""
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("media_previous")
        self._focus_spotify_window()
        return self._run_local("media_previous", "previous")

    @function_tool()
    async def volume_up(self, context: RunContext, steps: int = 5) -> dict[str, str]:
        """Turn the system volume up by a number of steps (1-20).

        Args:
            steps: Number of volume steps (default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("volume_up")
        try:
            send_volume_delta("up", steps)
        except MediaControlError as exc:
            self._log_failure("volume_up", str(steps), exc)
            raise ToolError(str(exc)) from exc
        return {"direction": "up", "steps": min(max(int(steps), 1), MAX_VOLUME_STEPS)}

    @function_tool()
    async def volume_down(self, context: RunContext, steps: int = 5) -> dict[str, str]:
        """Turn the system volume down by a number of steps (1-20).

        Args:
            steps: Number of volume steps (default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("volume_down")
        try:
            send_volume_delta("down", steps)
        except MediaControlError as exc:
            self._log_failure("volume_down", str(steps), exc)
            raise ToolError(str(exc)) from exc
        return {"direction": "down", "steps": min(max(int(steps), 1), MAX_VOLUME_STEPS)}

    @function_tool()
    async def media_mute(self, context: RunContext) -> dict[str, str]:
        """Mute or unmute the system volume."""
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        self._require_local("media_mute")
        return self._run_local("media_mute", "mute")
