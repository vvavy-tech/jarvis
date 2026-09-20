"""Core integration framework: action levels, status, and the base class.

Every external service (Gmail, Calendar, Spotify, Meta Ads, and future
integrations) is wrapped in an :class:`Integration`. An integration declares
its name, description, required environment variables, read/write posture, and
the tools it exposes. Tools carry an explicit :class:`ActionLevel` so the voice
gate can enforce "no confirmation" vs "explicit confirmation" policies as
metadata rather than as prompt text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar

from gates import ToolGate


class ActionLevel(IntEnum):
    """Classification of how consequential an action is (metac-data policy).

    Level 1 - safe read          : read statistics, read email, inspect status.
    Level 2 - reversible action  : pause music, skip a track, open an app.
    Level 3 - consequential      : send email, create an event, merge code.
    Level 4 - security change    : change auth, grant permissions, expose secrets.
    """

    SAFE_READ = 1
    REVERSIBLE = 2
    CONSEQUENTIAL = 3
    SECURITY = 4


LEVEL_LABELS: dict[ActionLevel, str] = {
    ActionLevel.SAFE_READ: "safe_read",
    ActionLevel.REVERSIBLE: "reversible_action",
    ActionLevel.CONSEQUENTIAL: "consequential_action",
    ActionLevel.SECURITY: "security_change",
}


def level_label(level: int | ActionLevel) -> str:
    return LEVEL_LABELS.get(ActionLevel(level), "safe_read")


def _env_present(name: str) -> bool:
    value = os.environ.get(name, "")
    return bool(value and value.strip())


@dataclass(frozen=True)
class IntegrationStatus:
    name: str
    available: bool
    authenticated: bool
    read_only: bool
    confirmation_required: bool
    action_level: int = ActionLevel.SAFE_READ
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "authenticated": self.authenticated,
            "read_only": self.read_only,
            "confirmation_required": self.confirmation_required,
            "action_level": level_label(self.action_level),
            "note": self.note,
        }


class Integration:
    """Base interface every capability must implement.

    Subclasses declare tool methods (decorated with ``@function_tool()``) and
    list them in ``_TOOLS``; the base class binds and exposes them as
    ``self.tools``. Each tool name may be mapped to an :class:`ActionLevel` in
    ``_LEVELS`` for confirmation policy.
    """

    name: str = ""
    description: str = ""
    required_env: tuple[str, ...] = ()
    read_only: bool = True
    default_level: ActionLevel = ActionLevel.SAFE_READ
    _TOOLS: tuple[str, ...] = ()
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {}

    def __init__(
        self,
        *,
        gate: ToolGate | None = None,
        failure_log: Any | None = None,
    ) -> None:
        self.gate = gate or ToolGate()
        self._failure_log = failure_log
        self.tools: list[Any] = [
            getattr(self, tool_name) for tool_name in type(self)._TOOLS
        ]

    @property
    def tool_names(self) -> list[str]:
        return list(type(self)._TOOLS)

    def tool_level(self, tool_name: str) -> ActionLevel:
        return type(self)._LEVELS.get(tool_name, self.default_level)

    @property
    def confirmation_required(self) -> bool:
        return any(
            level >= ActionLevel.CONSEQUENTIAL for level in type(self)._LEVELS.values()
        )

    # ------------------------------------------------------------------ #
    # capability queries
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Whether this integration can operate at all on this system."""
        return True

    def is_authenticated(self) -> bool:
        """Whether all required credentials/configuration are present."""
        return all(_env_present(name) for name in self.required_env)

    def status_note(self) -> str:
        return ""

    def status(self) -> IntegrationStatus:
        return IntegrationStatus(
            name=self.name,
            available=self.is_available(),
            authenticated=self.is_authenticated(),
            read_only=self.read_only,
            confirmation_required=self.confirmation_required,
            action_level=int(self.default_level),
            note=self.status_note(),
        )

    async def health_check(self) -> dict[str, Any]:
        if not self.is_available():
            return {
                "name": self.name,
                "ok": False,
                "status": "unavailable",
                "detail": f"{self.name} is not available on this system.",
            }
        if not self.is_authenticated():
            return {
                "name": self.name,
                "ok": False,
                "status": "not_connected",
                "detail": f"{self.name} is available but not connected. "
                "Configure its environment variables to connect it.",
            }
        return {
            "name": self.name,
            "ok": True,
            "status": "connected",
            "detail": f"{self.name} is connected.",
        }

    def _log_failure(
        self,
        tool: str,
        target: str,
        exc: Exception,
        fallback: str = "",
    ) -> None:
        if self._failure_log is None:
            return
        self._failure_log.append(
            tool=tool,
            target=target,
            reason=str(exc),
            fallback=fallback,
            fallback_result="",
        )


class ToolGroup(Integration):
    """Wraps pre-existing, externally built tools as a registered capability.

    Used for inherited groups such as the browser and developer mode, which
    already own their gating and tool lists but should appear in the capability
    registry like any other integration.
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[Any],
        *,
        gate: ToolGate | None = None,
        read_only: bool = False,
        level: ActionLevel = ActionLevel.REVERSIBLE,
        tool_names: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.read_only = read_only
        self.__tool_level = level
        self.__external_tools = list(tools)
        self.__external_tool_names = list(tool_names or [])
        super().__init__(gate=gate)
        self.tools = self.__external_tools

    @property
    def tool_names(self) -> list[str]:
        return list(self.__external_tool_names)

    def tool_level(self, tool_name: str) -> ActionLevel:
        return self.__tool_level

    @property
    def confirmation_required(self) -> bool:
        return self.__tool_level >= ActionLevel.CONSEQUENTIAL

    def is_authenticated(self) -> bool:
        return True
