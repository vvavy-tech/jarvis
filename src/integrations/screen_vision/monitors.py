"""Windows monitor enumeration and cursor lookup.

Coordinate space is the virtual desktop in physical screen pixels, which is the
same space ``PIL.ImageGrab`` works in once the process is DPI aware. Monitors
can sit to the left of or above the primary monitor, so their origins may be
negative. Everything here is plain, synchronous code intended to run inside
``asyncio.to_thread`` from the integration layer.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Any

_ON_WINDOWS = sys.platform == "win32"


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", ctypes.c_wchar * 32),
    ]


_MONITOR_INFO_PRIMARY = 1

_MONITOR_ENUM_PROC = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HMONITOR,
    wintypes.HDC,
    ctypes.POINTER(wintypes.RECT),
    wintypes.LPARAM,
)


def _user32() -> ctypes.WinDLL | None:
    if not _ON_WINDOWS:
        return None
    try:
        return ctypes.WinDLL("user32", use_last_error=True)
    except OSError:
        return None


def best_effort_dpi_aware() -> None:
    """Mark the process per-monitor DPI aware when possible.

    Keeps :func:`monitors` and :func:`capture_bbox` on the same pixel grid.
    Best-effort: any failure is ignored and only degrades coordinate accuracy,
    never agent availability.
    """
    if not _ON_WINDOWS:
        return
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetProcessDpiAwarenessContext(-4)  # PER_MONITOR_AWARE_V2
    except Exception:
        try:
            shcore = ctypes.WinDLL("shcore", use_last_error=True)
            shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            return


def monitors() -> list[dict[str, Any]]:
    """Return metadata for every monitor::

    {"id": 0, "left": 0, "top": 0, "width": 1920, "height": 1080,
     "primary": True, "name": "\\\\.\\DISPLAY1"}
    """
    user32 = _user32()
    if user32 is None:
        raise RuntimeError("screen monitoring requires Windows with user32/gdi32")

    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.GetMonitorInfoW.argtypes = [
        wintypes.HMONITOR,
        ctypes.POINTER(_MonitorInfo),
    ]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.EnumDisplayMonitors.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        _MONITOR_ENUM_PROC,
        wintypes.LPARAM,
    ]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL

    collected: list[dict[str, Any]] = []

    @_MONITOR_ENUM_PROC
    def _callback(_hmonitor, _hdc, _lprc, _data) -> int:
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        if user32.GetMonitorInfoW(_hmonitor, ctypes.byref(info)):
            rect = info.rcMonitor
            collected.append(
                {
                    "left": rect.left,
                    "top": rect.top,
                    "width": rect.right - rect.left,
                    "height": rect.bottom - rect.top,
                    "primary": bool(info.dwFlags & _MONITOR_INFO_PRIMARY),
                    "name": info.szDevice,
                }
            )
        return 1

    hdc = user32.GetDC(None)
    try:
        user32.EnumDisplayMonitors(hdc, None, _callback, 0)
    finally:
        user32.ReleaseDC(None, hdc)

    if not collected:
        raise RuntimeError("no monitors enumerated on this system")
    collected.sort(key=lambda m: (not m["primary"], m["left"], m["top"]))
    return [{"id": index, **meta} for index, meta in enumerate(collected)]


def cursor_position() -> dict[str, int]:
    """Return the current cursor position in virtual-desktop coordinates."""
    user32 = _user32()
    if user32 is None:
        raise RuntimeError("cursor lookup requires Windows with user32")
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        raise RuntimeError("could not read the cursor position")
    return {"x": point.x, "y": point.y}


def containing_monitor(
    position: dict[str, int], monitor_list: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the monitor whose bounds contain ``position``, else ``None``."""
    for monitor in monitor_list:
        if (
            monitor["left"] <= position["x"] <= monitor["left"] + monitor["width"]
            and monitor["top"] <= position["y"] <= monitor["top"] + monitor["height"]
        ):
            return monitor
    return None
