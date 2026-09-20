"""Gmail capability (read-only).

Provides safe-read tools for email while remaining honest about connection
state. Unauthenticated calls return a clear "not connected" answer instead of
invented inbox content, so JARVIS never fabricates email data.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import RunContext, function_tool

from integrations.base import ActionLevel, Integration
from integrations.gmail.client import (
    GmailClient,
    GmailNotAuthenticatedError,
)

MAX_LIMIT = 20


class GmailIntegration(Integration):
    name = "gmail"
    description = (
        "Read Gmail: list recent emails, search the inbox, read a message, and "
        "identify unread mail. Read-only for now; sending or deleting is not enabled."
    )
    read_only = True
    default_level = ActionLevel.SAFE_READ
    _TOOLS = (
        "read_recent_emails",
        "search_emails",
        "read_email",
        "list_unread_emails",
        "summarise_recent_emails",
    )
    _LEVELS = dict.fromkeys(_TOOLS, ActionLevel.SAFE_READ)

    def __init__(
        self, *, gate=None, failure_log=None, client: GmailClient | None = None
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self._client = client or GmailClient()

    def is_authenticated(self) -> bool:
        return self._client.is_authenticated()

    def status_note(self) -> str:
        return "" if self.is_authenticated() else "available but not connected"

    async def health_check(self) -> dict[str, Any]:
        base = await super().health_check()
        return base

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _clamp(self, limit: int) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError):
            return 5
        return min(max(value, 1), MAX_LIMIT)

    def _not_connected(self) -> dict[str, str]:
        return {
            "status": "not_connected",
            "message": (
                "Gmail is available but not connected. The user has not connected "
                "a Gmail account yet, so I cannot see the real inbox. Do not invent "
                "email content."
            ),
        }

    async def _with_client(self, tool: str, operation: str, **kwargs) -> dict[str, Any]:
        if not self._client.is_authenticated():
            return self._not_connected()
        try:
            method = getattr(self._client, operation)
            result = await method(**kwargs)
        except GmailNotAuthenticatedError:
            return self._not_connected()
        except NotImplementedError as exc:
            return {
                "status": "not_implemented",
                "message": str(exc),
            }
        except Exception as exc:
            self._log_failure(tool, operation, exc)
            return {"status": "error", "message": f"Email lookup failed: {exc}"}
        return {"status": "ok", "messages": result}

    # ------------------------------------------------------------------ #
    # tools (all safe reads)
    # ------------------------------------------------------------------ #

    @function_tool()
    async def read_recent_emails(
        self, context: RunContext, limit: int = 5
    ) -> dict[str, Any]:
        """Read the user's most recent emails.

        Args:
            limit: Number of emails to return (1-20, default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "read_recent_emails", "list_messages", limit=self._clamp(limit)
        )

    @function_tool()
    async def search_emails(
        self, context: RunContext, query: str, limit: int = 5
    ) -> dict[str, Any]:
        """Search the user's email for messages matching a query.

        Args:
            query: Search terms, such as 'invoice from Shopify'.
            limit: Number of results (1-20, default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "search_emails", "search", query=query, limit=self._clamp(limit)
        )

    @function_tool()
    async def read_email(self, context: RunContext, message_id: str) -> dict[str, Any]:
        """Read a single email message by its id.

        Args:
            message_id: The email message id returned by a search or list.
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        if not (message_id or "").strip():
            return {"status": "error", "message": "A message id is required."}
        return await self._with_client(
            "read_email", "get_message", message_id=message_id.strip()
        )

    @function_tool()
    async def list_unread_emails(
        self, context: RunContext, limit: int = 5
    ) -> dict[str, Any]:
        """List the user's unread emails.

        Args:
            limit: Number of unread emails (1-20, default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "list_unread_emails", "list_unread", limit=self._clamp(limit)
        )

    @function_tool()
    async def summarise_recent_emails(
        self, context: RunContext, limit: int = 5
    ) -> dict[str, Any]:
        """Return today's or the latest emails so they can be summarised aloud.

        Args:
            limit: Number of emails to summarise (1-20, default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        result = await self._with_client(
            "summarise_recent_emails", "list_messages", limit=self._clamp(limit)
        )
        return result
