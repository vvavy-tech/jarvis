"""Gmail client scaffold (read-only).

JARVIS only reads Gmail for now. Sending, deleting, or archiving email is
deliberately NOT implemented and will be added later behind consequential
action approval (ActionLevel 3+).

Authentication is intentionally not present yet: without credentials every
method raises :class:`GmailNotAuthenticatedError` so the agent reports "Gmail
is available but not connected" honestly instead of inventing inbox data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REQUIRED_ENV = ("GMAIL_OAUTH_CREDENTIALS",)

DEFAULT_TOKEN_PATH = Path.home() / ".jarvis" / "gmail_token.json"


class GmailNotAuthenticatedError(RuntimeError):
    """Raised when a Gmail read is attempted before connection."""


def _credentials_configured() -> bool:
    path = os.environ.get("GMAIL_OAUTH_CREDENTIALS", "").strip()
    if not path:
        return False
    return Path(path).is_file()


def token_path() -> Path:
    configured = os.environ.get("GMAIL_TOKEN_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_TOKEN_PATH


class GmailClient:
    """Read-only abstraction over the Gmail API (scaffold)."""

    def __init__(self, token_file: Path | None = None) -> None:
        self._token_file = token_file or token_path()

    def is_authenticated(self) -> bool:
        return _credentials_configured() and self._token_file.is_file()

    # ------------------------------------------------------------------ #
    # read-only API surface (implemented after credentials are connected)
    # ------------------------------------------------------------------ #

    async def list_messages(self, limit: int = 5) -> list[dict[str, Any]]:
        self._require_authenticated()
        raise NotImplementedError(
            "Gmail message listing is implemented after connection."
        )

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        self._require_authenticated()
        raise NotImplementedError("Gmail search is implemented after connection.")

    async def get_message(self, message_id: str) -> dict[str, Any]:
        self._require_authenticated()
        raise NotImplementedError(
            "Gmail message reading is implemented after connection."
        )

    async def list_unread(self, limit: int = 5) -> list[dict[str, Any]]:
        self._require_authenticated()
        raise NotImplementedError(
            "Gmail unread listing is implemented after connection."
        )

    def _require_authenticated(self) -> None:
        if not self.is_authenticated():
            raise GmailNotAuthenticatedError(
                "Gmail is available but not connected. Connect the Gmail account "
                "via GMAIL_OAUTH_CREDENTIALS, then I can read your email."
            )
