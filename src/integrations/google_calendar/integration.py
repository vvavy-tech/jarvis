"""Google Calendar capability (read-only scaffold).

Exposes safe reads: "what do I have tomorrow?", "when is my next appointment?",
"am I free Friday afternoon?". Honest "not connected" answers until a Google
account is linked. Unavailable write operations are documented for later
consequential-approval work but are not registered as tools.
"""

from __future__ import annotations

from typing import Any, ClassVar

from livekit.agents import RunContext, function_tool

from integrations.base import ActionLevel, Integration
from integrations.google_calendar.client import (
    CalendarNotAuthenticatedError,
    GoogleCalendarClient,
)

MAX_EVENTS = 20


class GoogleCalendarIntegration(Integration):
    name = "google_calendar"
    description = (
        "Read the Google Calendar: today's or upcoming events, next appointment, "
        "and availability checks. Read-only for now; creating or deleting events "
        "is not enabled."
    )
    read_only = True
    default_level = ActionLevel.SAFE_READ
    _TOOLS = ("calendar_events", "next_appointments", "calendar_free_check")
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {
        "calendar_events": ActionLevel.SAFE_READ,
        "next_appointments": ActionLevel.SAFE_READ,
        "calendar_free_check": ActionLevel.SAFE_READ,
    }

    def __init__(
        self,
        *,
        gate=None,
        failure_log=None,
        client: GoogleCalendarClient | None = None,
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self._client = client or GoogleCalendarClient()

    def is_authenticated(self) -> bool:
        return self._client.is_authenticated()

    def status_note(self) -> str:
        return "" if self.is_authenticated() else "available but not connected"

    async def health_check(self) -> dict[str, Any]:
        return await super().health_check()

    def _not_connected(self) -> dict[str, str]:
        return {
            "status": "not_connected",
            "message": (
                "Google Calendar is available but not connected. The user has not "
                "connected a calendar account, so I cannot see the real schedule. "
                "Do not invent appointments."
            ),
        }

    async def _with_client(self, tool: str, operation: str, **kwargs) -> dict[str, Any]:
        if not self._client.is_authenticated():
            return self._not_connected()
        try:
            method = getattr(self._client, operation)
            result = await method(**kwargs)
        except CalendarNotAuthenticatedError:
            return self._not_connected()
        except NotImplementedError as exc:
            return {"status": "not_implemented", "message": str(exc)}
        except Exception as exc:
            self._log_failure(tool, operation, exc)
            return {"status": "error", "message": f"Calendar lookup failed: {exc}"}
        return {"status": "ok", "result": result}

    @function_tool()
    async def calendar_events(
        self,
        context: RunContext,
        time_min: str,
        time_max: str,
        calendar: str = "primary",
    ) -> dict[str, Any]:
        """List calendar events between two ISO timestamps.

        Args:
            time_min: Start of range, ISO format e.g. '2026-09-20T00:00:00'.
            time_max: End of range, ISO format e.g. '2026-09-21T00:00:00'.
            calendar: Calendar id (default 'primary').
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "calendar_events",
            "get_events",
            time_min=time_min,
            time_max=time_max,
            calendar=calendar,
            limit=MAX_EVENTS,
        )

    @function_tool()
    async def next_appointments(
        self, context: RunContext, count: int = 3
    ) -> dict[str, Any]:
        """List the next upcoming appointments.

        Args:
            count: How many appointments to return (1-10, default 3).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        try:
            value = max(1, min(int(count), 10))
        except (TypeError, ValueError):
            value = 3
        return await self._with_client(
            "next_appointments", "get_events", time_min="now", time_max="", limit=value
        )

    @function_tool()
    async def calendar_free_check(
        self,
        context: RunContext,
        time_min: str,
        time_max: str,
    ) -> dict[str, Any]:
        """Check whether the user is free between two ISO timestamps.

        Args:
            time_min: Start of the window, ISO format.
            time_max: End of the window, ISO format.
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "calendar_free_check", "availability", time_min=time_min, time_max=time_max
        )
