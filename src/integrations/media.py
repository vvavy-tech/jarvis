"""Local Windows media controls and app launching.

These helpers drive the desktop "media keys" (play/pause, next/previous,
volume) and launch installed applications. They only work on Windows; on other
platforms callers receive a clear :class:`MediaControlError` instead of a
silent no-op.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Any

ON_WINDOWS = sys.platform == "win32"

_PROCESS_TERMINATE = 0x0001
_TH32CS_SNAPPROCESS = 0x00000002
_INVALID_HANDLE = ctypes.c_void_p(-1).value

_MEDIA_VK: dict[str, int] = {
    "play_pause": 0xB3,  # VK_MEDIA_PLAY_PAUSE
    "next": 0xB0,  # VK_MEDIA_NEXT_TRACK
    "previous": 0xB1,  # VK_MEDIA_PREV_TRACK
    "volume_up": 0xAF,  # VK_VOLUME_UP
    "volume_down": 0xAE,  # VK_VOLUME_DOWN
    "mute": 0xAD,  # VK_VOLUME_MUTE
}


class MediaControlError(RuntimeError):
    """Raised when a local media or app-control action cannot be performed."""


def _send_key(vk: int) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event.argtypes = [
        wintypes.BYTE,
        wintypes.BYTE,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    user32.keybd_event(wintypes.BYTE(vk), 0, 0, None)
    user32.keybd_event(wintypes.BYTE(vk), 0, 0x0002, None)  # KEYEVENTF_KEYUP


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _kernel32() -> ctypes.WinDLL:
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _user32() -> ctypes.WinDLL:
    return ctypes.WinDLL("user32", use_last_error=True)


def spotify_processes() -> list[int]:
    """Return the process IDs of every running ``Spotify.exe``."""
    if not ON_WINDOWS:
        raise MediaControlError("Spotify process control requires Windows.")
    kernel32 = _kernel32()
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot in (None, _INVALID_HANDLE):
        raise MediaControlError(
            f"Could not enumerate running processes (error {ctypes.get_last_error()})."
        )
    pids: list[int] = []
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(_ProcessEntry)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.casefold() == "spotify.exe":
                pids.append(entry.th32ProcessID)
            entry.dwSize = ctypes.sizeof(_ProcessEntry)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return pids


def _matches_spotify_window(
    process_pids: set[int], process_id: int, visible: bool
) -> bool:
    """Match a Spotify window by owning process, never by title.

    Spotify retitles its main window to the currently playing track, so titles
    must never be part of the match.
    """
    return process_id in process_pids and visible


def spotify_windows() -> list[dict[str, Any]]:
    """Return visible Spotify top-level windows with their responsiveness.

    Windows are matched by their owning process (``Spotify.exe``), not by
    title: Spotify retitles its main window to the currently playing track,
    so a title filter would miss it while music is playing.
    """
    if not ON_WINDOWS:
        raise MediaControlError("Spotify process control requires Windows.")
    user32 = _user32()
    pids = set(spotify_processes())
    if not pids:
        return []
    found: list[dict[str, Any]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _callback(hwnd: int, _lparam: int) -> int:
        process = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process))
        if not _matches_spotify_window(
            pids, process.value, user32.IsWindowVisible(hwnd)
        ):
            return 1
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        found.append(
            {
                "pid": process.value,
                "title": buffer.value,
                "responding": not bool(user32.IsHungAppWindow(hwnd)),
            }
        )
        return 1

    user32.EnumWindows(_callback, 0)
    return found


def spotify_health() -> dict[str, Any]:
    """Diagnose whether Spotify is running and responding.

    ``responding`` is ``None`` when Spotify runs but shows no visible window,
    ``True`` when every visible window responds, and ``False`` when any window
    is hung. Errors are reported in ``error`` instead of raising so the caller
    can speak honestly rather than crash.
    """
    error = ""
    procs: list[int] = []
    windows: list[dict[str, Any]] = []
    try:
        procs = spotify_processes()
        windows = spotify_windows()
    except Exception as exc:
        error = str(exc)
    responding: bool | None = None
    if windows:
        responding = all(bool(window["responding"]) for window in windows)
    return {
        "running": bool(procs),
        "process_count": len(procs),
        "window": bool(windows),
        "window_count": len(windows),
        "responding": responding,
        "error": error,
    }


def terminate_spotify() -> int:
    """Terminate every running Spotify process; returns how many were killed."""
    if not ON_WINDOWS:
        raise MediaControlError("Spotify process control requires Windows.")
    kernel32 = _kernel32()
    pids = spotify_processes()
    killed = 0
    for pid in pids:
        handle = kernel32.OpenProcess(_PROCESS_TERMINATE, False, pid)
        if handle:
            kernel32.TerminateProcess(handle, 0)
            kernel32.CloseHandle(handle)
            killed += 1
    return killed


def send_media_key(name: str) -> None:
    """Send one synthetic media-key press to the operating system."""
    if not ON_WINDOWS:
        raise MediaControlError(
            "Local media controls require Windows; use a connected API on this platform."
        )
    vk = _MEDIA_VK.get(name)
    if vk is None:
        raise MediaControlError(f"Unsupported media key: {name!r}")
    try:
        _send_key(vk)
    except OSError as exc:
        raise MediaControlError(f"Could not send media key {name!r}: {exc}") from exc


def send_volume_delta(direction: str, steps: int) -> None:
    if direction not in {"up", "down"}:
        raise MediaControlError("Volume direction must be 'up' or 'down'.")
    steps = min(max(int(steps), 1), 20)
    key = f"volume_{direction}"
    for _ in range(steps):
        send_media_key(key)


def _spotify_executable_candidates() -> list[Path]:
    bases = [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    candidates: list[Path] = []
    for base in bases:
        if not base:
            continue
        local = Path(base) / "Spotify" / "Spotify.exe"
        if local.is_file():
            candidates.append(local)
    return candidates


def launch_spotify() -> dict[str, str]:
    """Launch the Spotify desktop app if it is installed."""
    if not ON_WINDOWS:
        raise MediaControlError(
            "The Spotify desktop app is only launchable on Windows."
        )
    for exe in _spotify_executable_candidates():
        try:
            subprocess.Popen([str(exe)])
            return {"launched": True, "method": "desktop_app"}
        except OSError as exc:
            raise MediaControlError(f"Could not launch Spotify: {exc}") from exc
    try:
        os.startfile("spotify:")  # type: ignore[attr-defined]
        return {"launched": True, "method": "app_uri"}
    except OSError:
        raise MediaControlError(
            "Spotify does not appear to be installed. It can be reached via the "
            "browser once an account is connected, or the app installed."
        ) from None
