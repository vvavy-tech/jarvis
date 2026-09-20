"""Screen capture and temporary screenshot handling.

Capture and clipping helpers are plain synchronous code intended to run inside
``asyncio.to_thread`` so the agent event loop is never blocked. Screenshots are
written to a dedicated, git-ignored temp directory and pruned after a short
age so nothing persistent is saved by default.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageGrab

from developer.project_tools import PROJECT_ROOT
from integrations.screen_vision.monitors import containing_monitor

SCREEN_CAPTURE_DIR = PROJECT_ROOT / "screen_captures"
DEFAULT_AREA_WIDTH = 800
DEFAULT_AREA_HEIGHT = 600
_TEMP_STEM = "screen_*.png"
_MAX_TEMP_AGE_S = 15 * 60


def capture_bbox(left: int, top: int, right: int, bottom: int) -> Image.Image:
    """Capture a rectangular region of the virtual desktop."""
    if right <= left or bottom <= top:
        raise ValueError(f"invalid capture region ({left},{top},{right},{bottom})")
    try:
        return ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
    except Exception as exc:
        raise RuntimeError(f"failed to capture screen region: {exc}") from exc


def monitor_bbox(monitor: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return ``(left, top, right, bottom)`` for a monitor metadata dict."""
    return (
        monitor["left"],
        monitor["top"],
        monitor["left"] + monitor["width"],
        monitor["top"] + monitor["height"],
    )


def cursor_region(
    monitor_list: list[dict[str, Any]],
    cursor: dict[str, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Compute a region centered on the cursor, clipped to a containing monitor.

    Falls back to the union of all monitors when the cursor is not inside any
    single monitor (e.g. a corner shared by physical screens). Returns ``None``
    when there is nothing to capture.
    """
    if not monitor_list:
        return None
    width = max(64, width)
    height = max(64, height)

    clipped_to = containing_monitor(cursor, monitor_list)
    if clipped_to is not None:
        left, top, right, bottom = monitor_bbox(clipped_to)
    else:
        left = min(monitor["left"] for monitor in monitor_list)
        top = min(monitor["top"] for monitor in monitor_list)
        right = max(monitor["left"] + monitor["width"] for monitor in monitor_list)
        bottom = max(monitor["top"] + monitor["height"] for monitor in monitor_list)

    half_w = width // 2
    half_h = height // 2
    region_left = max(left, cursor["x"] - half_w)
    region_top = max(top, cursor["y"] - half_h)
    region_right = min(right, region_left + width)
    region_bottom = min(bottom, region_top + height)
    if region_right <= region_left or region_bottom <= region_top:
        return None
    return region_left, region_top, region_right, region_bottom


def save_temp_image(image: Image.Image) -> Path:
    """Save a screenshot to the temp capture dir; returns its ``Path``."""
    SCREEN_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    _prune_stale_captures()
    path = SCREEN_CAPTURE_DIR / f"screen_{time.time_ns()}.png"
    try:
        image.save(path, format="PNG")
    except Exception as exc:
        raise RuntimeError(f"could not save screenshot: {exc}") from exc
    return path


def remove_temp_image(path: Path) -> None:
    """Delete a temporary screenshot, ignoring any failure."""
    with contextlib.suppress(OSError):
        Path(path).unlink(missing_ok=True)


def _prune_stale_captures(max_age_s: int = _MAX_TEMP_AGE_S) -> None:
    now = time.time()
    for child in SCREEN_CAPTURE_DIR.glob(_TEMP_STEM):
        try:
            if now - child.stat().st_mtime > max_age_s:
                child.unlink(missing_ok=True)
        except OSError:
            pass
