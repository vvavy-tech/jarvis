"""Capture backends behind a small protocol.

The real backend captures through PIL/GDI on Windows. Backends are plain
synchronous objects; the integration layer calls them from worker threads so
the agent event loop is never blocked.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from integrations.screen_vision import capture, monitors


class ScreenBackend(Protocol):
    """Minimal surface the integration depends on (fakes can implement it)."""

    def list_monitors(self) -> list[dict[str, Any]]: ...

    def get_cursor_position(self) -> dict[str, int]: ...

    def capture(self, bbox: tuple[int, int, int, int]) -> Image.Image: ...


class WindowsScreenBackend:
    """Captures the user's screen through PIL ImageGrab on Windows."""

    def __init__(self) -> None:
        monitors.best_effort_dpi_aware()

    def list_monitors(self) -> list[dict[str, Any]]:
        return monitors.monitors()

    def get_cursor_position(self) -> dict[str, int]:
        return monitors.cursor_position()

    def capture(self, bbox: tuple[int, int, int, int]) -> Image.Image:
        return capture.capture_bbox(*bbox)


def default_backend() -> ScreenBackend | None:
    """Build the Windows backend, or return ``None`` when capture is impossible.

    The backend is probed lazily on every JARVIS start; returning ``None`` here
    never prevents the agent from running, it simply removes the screen tools.
    """
    try:
        backend = WindowsScreenBackend()
        backend.list_monitors()
        return backend
    except Exception:
        return None
