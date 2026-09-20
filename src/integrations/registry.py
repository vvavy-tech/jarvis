"""Central capability registry.

The registry knows which integrations exist, which are connected, which
permissions each one holds, whether it is read-only, whether user confirmation
is required for its actions, and whether it is currently available. It is the
single place the agent asks "what can I actually do?".
"""

from __future__ import annotations

from typing import Any

from integrations.base import Integration, IntegrationStatus


def _display_label(status: IntegrationStatus) -> str:
    if not status.available:
        return "unavailable"
    if status.authenticated:
        return "connected"
    if status.note:
        return status.note
    return "available but not connected"


class CapabilityRegistry:
    """Holds every registered integration and answers capability queries."""

    def __init__(self) -> None:
        self._items: dict[str, Integration] = {}

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #

    def register(self, integration: Integration) -> Integration:
        name = (integration.name or "").strip()
        if not name:
            raise ValueError("An integration must have a non-empty 'name'.")
        if name in self._items:
            raise ValueError(f"Integration already registered: {name!r}")
        self._items[name] = integration
        return integration

    def register_many(self, integrations: list[Integration]) -> None:
        for integration in integrations:
            self.register(integration)

    # ------------------------------------------------------------------ #
    # lookups
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> Integration | None:
        return self._items.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def integrations(self) -> list[Integration]:
        return [self._items[name] for name in self.names()]

    def tool_levels(self) -> dict[str, int]:
        """Map every registered tool name to its numeric action level."""
        levels: dict[str, int] = {}
        for integration in self._items.values():
            for tool_name in integration.tool_names:
                levels[tool_name] = int(integration.tool_level(tool_name))
        return levels

    # ------------------------------------------------------------------ #
    # tool aggregation
    # ------------------------------------------------------------------ #

    def tools(self) -> list[Any]:
        """All tools from currently available integrations, flatten to one list."""
        aggregated: list[Any] = []
        for integration in self._items.values():
            if not integration.is_available():
                continue
            aggregated.extend(integration.tools)
        return aggregated

    # ------------------------------------------------------------------ #
    # capability / status answers
    # ------------------------------------------------------------------ #

    def capabilities(self) -> list[dict[str, Any]]:
        return [self._items[name].status().to_dict() for name in self.names()]

    def connected_names(self) -> list[str]:
        return [
            name
            for name in self.names()
            if self._items[name].is_available() and self._items[name].is_authenticated()
        ]

    def report(self) -> str:
        lines = []
        for name in self.names():
            status = self._items[name].status()
            label = _display_label(status)
            markers = []
            if status.read_only:
                markers.append("read-only")
            if status.confirmation_required:
                markers.append("confirmation required")
            suffix = f" ({', '.join(markers)})" if markers else ""
            lines.append(f"{name.capitalize()}: {label}{suffix}")
        return "\n".join(lines) or "No capabilities registered."

    async def health_report(self) -> str:
        lines = []
        for name in self.names():
            health = await self._items[name].health_check()
            lines.append(f"{name.capitalize()}: {health['status']}")
        return "\n".join(lines) or "No capabilities registered."
