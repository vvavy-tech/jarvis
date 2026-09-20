"""Google Calendar client scaffold (read-only).

Only reading is scaffolded today: listing calendars, events, and availability.
Creating or deleting events is intentionally not exposed and will be added
later as consequential actions (ActionLevel 3+) with explicit approval.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REQUIRED_ENV = ("GOOGLE_OAUTH_CREDENTIALS",)

DEFAULT_TOKEN_PATH = Path.home() / ".jarvis" / "calendar_token.json"


class CalendarNotAuthenticatedError(RuntimeError):
    """Raised when a calendar read is attempted before connection."""


def _credentials_configured() -> bool:
    path = os.environ.get("GOOGLE_OAUTH_CREDENTIALS", "").strip()
    if not path:
        return False
    return Path(path).is_file()


def token_path() -> Path:
    configured = os.environ.get("CALENDAR_TOKEN_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_TOKEN_PATH


class GoogleCalendarClient:
    """Read-only abstraction over the Google Calendar API (scaffold)."""

    def __init__(self, token_file: Path | None = None) -> None:
        self._token_file = token_file or token_path()

    def is_authenticated(self) -> bool:
        return _credentials_configured() and self._token_file.is_file()

    # ------------------------------------------------------------------ #
    # read-only API surface (implemented after credentials are connected)
    # ------------------------------------------------------------------ #

    async def list_calendars(self) -> list[dict[str, Any]]:
        self._require_authenticated()
        raise NotImplementedError("Calendar listing is implemented after connection.")

    async def get_events(
        self,
        time_min: str,
        time_max: str,
        *,
        calendar: str = "primary",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._require_authenticated()
        raise NotImplementedError("Calendar events are implemented after connection.")

    async def availability(
        self,
        time_min: str,
        time_max: str,
    ) -> dict[str, Any]:
        self._require_authenticated()
        raise NotImplementedError(
            "Calendar availability is implemented after connection."
        )

    def _require_authenticated(self) -> None:
        if not self.is_authenticated():
            raise CalendarNotAuthenticatedError(
                "Google Calendar is available but not connected. Connect it via "
                "GOOGLE_OAUTH_CREDENTIALS, then I can read your schedule."
            )
