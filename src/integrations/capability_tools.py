"""Voice tools JARVIS uses to know and report its own capabilities."""

from __future__ import annotations

from typing import Any

from livekit.agents import RunContext, function_tool

from gates import ToolGate
from integrations.registry import CapabilityRegistry


class CapabilityTools:
    """Exposes capability state to the model and to the user.

    ``get_capabilities`` feeds precise, current capability metadata into the
    conversation so JARVIS never has to guess what it can do. ``system_status``
    is the human-facing "what systems are connected?" report.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        gate: ToolGate | None = None,
    ) -> None:
        self.registry = registry
        self.gate = gate or ToolGate()

    @property
    def tools(self) -> list[Any]:
        return [self.get_capabilities, self.system_status]

    @function_tool()
    async def get_capabilities(self, context: RunContext) -> dict[str, Any]:
        """Return JARVIS's current capabilities: which services exist, whether each
        is connected, whether it is read-only, and whether its actions need
        confirmation. Use this before promising or performing an action so you
        never claim a service works when it is not connected."""
        self.gate.ensure_active_conversation()
        return {
            "capabilities": self.registry.capabilities(),
            "action_levels": {
                "safe_read": "Read-only lookups, no confirmation.",
                "reversible_action": "Reversible actions, no unnecessary confirmation.",
                "consequential_action": "Explicit user approval required.",
                "security_change": "Explicit approval and never automatic.",
            },
            "rule": (
                "Only report data from connected services. If a capability is "
                "'available but not connected', say so and never invent data."
            ),
        }

    @function_tool()
    async def system_status(self, context: RunContext) -> dict[str, Any]:
        """Report which systems JARVIS can currently reach, with their health."""
        self.gate.ensure_active_conversation()
        health = []
        for integration in self.registry.integrations():
            health.append(await integration.health_check())
        return {
            "report": self.registry.report(),
            "health": health,
        }
