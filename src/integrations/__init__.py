"""JARVIS integrations framework.

Public entry points:
- :class:`Integration`, :class:`ToolGroup`, :class:`ActionLevel` - the base API.
- :class:`CapabilityRegistry` - central capability registry.
- :func:`build_default_registry` - wire every capability into one registry.
"""

from __future__ import annotations

from typing import Any

from gates import ToolGate
from integrations.base import (
    ActionLevel,
    Integration,
    IntegrationStatus,
    ToolGroup,
    level_label,
)
from integrations.capability_tools import CapabilityTools
from integrations.gmail.integration import GmailIntegration
from integrations.google_calendar.integration import GoogleCalendarIntegration
from integrations.meta_ads.integration import MetaAdsIntegration
from integrations.registry import CapabilityRegistry
from integrations.spotify.integration import SpotifyIntegration
from integrations.windows_integration import WindowsIntegration

try:
    from integrations.screen_vision import ScreenVisionIntegration
except Exception:  # pragma: no cover - dependency guard
    ScreenVisionIntegration = None

__all__ = [
    "ActionLevel",
    "CapabilityRegistry",
    "CapabilityTools",
    "GmailIntegration",
    "GoogleCalendarIntegration",
    "HermesIntegration",
    "Integration",
    "IntegrationStatus",
    "MemoryIntegration",
    "MetaAdsIntegration",
    "ScreenVisionIntegration",
    "SpotifyIntegration",
    "ToolGroup",
    "WindowsIntegration",
    "build_default_registry",
    "level_label",
]


def build_default_registry(
    *,
    gate: ToolGate | None = None,
    failure_log: Any | None = None,
    external_groups: list[ToolGroup] | None = None,
    hermes_agent: Any | None = None,
    memory_provider: Any | None = None,
) -> CapabilityRegistry:
    """Construct the registry with the standard capability set.

    ``external_groups`` carries pre-built tool groups (browser, developer mode)
    that are already gated and wired by the rest of the agent.

    ``hermes_agent`` injects the Hermes backend adapter; when omitted a default
    (runtime-detected) adapter is used. Hermes is always optional: with no
    installed interface the agent still starts and reports it unavailable.

    ``memory_provider`` injects the durable memory backend; when omitted a
    memory capability is still registered but reports itself unavailable.
    Memory is always optional: the agent starts and keeps the voice loop
    working even if the store is missing or corrupted.

    Optional integrations are only added if importing them does not fail. No
    shared state or credentials are required, so the agent always starts.
    """
    registry = CapabilityRegistry()
    for group in external_groups or []:
        registry.register(group)
    from agents.hermes_agent import HermesIntegration
    from memory.tools import MemoryIntegration

    registry.register(WindowsIntegration(gate=gate, failure_log=failure_log))
    registry.register(SpotifyIntegration(gate=gate, failure_log=failure_log))
    registry.register_many(
        [
            GmailIntegration(gate=gate, failure_log=failure_log),
            GoogleCalendarIntegration(gate=gate, failure_log=failure_log),
            MetaAdsIntegration(gate=gate, failure_log=failure_log),
            HermesIntegration(gate=gate, failure_log=failure_log, agent=hermes_agent),
            MemoryIntegration(
                gate=gate, failure_log=failure_log, provider=memory_provider
            ),
        ]
    )
    if ScreenVisionIntegration is not None:
        registry.register(ScreenVisionIntegration(gate=gate, failure_log=failure_log))
    return registry
