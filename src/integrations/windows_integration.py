"""Windows platform capability.

Windows control is a real capability today (browser window focus and media
keys already use it), and it is designed so that targeted, safe controls
(open/close/minimise/focus applications, volume) can be added over time. There
is deliberately no unrestricted shell access here.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from typing import Any, ClassVar

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

from integrations.base import ActionLevel, Integration

_SAFE_APP_LAUNCH: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "paint": ["mspaint.exe"],
    "cmd": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "file explorer": ["explorer.exe"],
}


def windows_focus_app(app_name: str) -> bool:
    """Bring a running application window to the foreground by title substring."""
    if sys.platform != "win32":
        return False
    try:
        from browser_tools import _focus_window_with_title

        return _focus_window_with_title(app_name)
    except Exception:
        return False


def list_media_controls() -> dict[str, Any]:
    if sys.platform != "win32":
        return {"available": False, "reason": "requires Windows"}
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    if not hasattr(user32, "keybd_event"):
        return {"available": False, "reason": "media keys unavailable"}
    return {"available": True, "mechanism": "desktop media keys"}


class WindowsIntegration(Integration):
    name = "windows"
    description = (
        "Local Windows desktop control: application launching and focus, plus "
        "system media keys. No unrestricted shell access."
    )
    read_only = False
    default_level = ActionLevel.SAFE_READ
    _TOOLS = ("windows_status", "open_application")
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {
        "windows_status": ActionLevel.SAFE_READ,
        "open_application": ActionLevel.REVERSIBLE,
    }

    def is_authenticated(self) -> bool:
        return sys.platform == "win32"

    def status_note(self) -> str:
        return "" if self.is_available() else "requires Windows"

    @function_tool()
    async def windows_status(self, context: RunContext) -> dict[str, Any]:
        """Report what local Windows controls are available to the assistant."""
        self.gate.ensure_active_conversation()
        media = list_media_controls()
        return {
            "name": self.name,
            "available": self.is_available(),
            "media_controls": media,
            "open_application_supported": sys.platform == "win32",
        }

    @function_tool()
    async def open_application(self, context: RunContext, app: str) -> dict[str, str]:
        """Open a known local application from a safe, fixed allow-list.

        Args:
            app: The application name, such as 'notepad', 'calculator', 'cmd',
                'powershell', or 'file explorer'.
        """
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))
        key = (app or "").strip().casefold()
        command = _SAFE_APP_LAUNCH.get(key)
        if command is None:
            raise ToolError(
                "I can only open applications from my known list: "
                + ", ".join(sorted(_SAFE_APP_LAUNCH))
                + "."
            )
        try:
            subprocess.Popen(command)
        except OSError as exc:
            raise ToolError(f"I could not open {app!r}: {exc}") from exc
        return {"app": app, "launched": True}
