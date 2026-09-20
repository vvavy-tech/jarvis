"""Deterministic integration scaffolding for Developer Mode.

When the user says something like "developer mode: create an integration for
Philips Hue", :class:`IntegrationScaffold` generates a complete, standards
compliant integration package plus tests and documentation. The proposal is
deterministic (no LLM), so it is reliable, and it flows through the normal
Developer pipeline: task branch -> apply -> full check suite -> commit ->
await approval. The generated code is intentionally minimal but complete:
a client scaffold, an :class:`Integration` exposing a status tool, a test, and
a per-service doc.
"""

from __future__ import annotations

import re
from typing import Any

_INTEGRATION_RE = re.compile(
    r"\b(?:create|add|build|make)\b"
    r"(?:[\s\S]{0,80})"
    r"\bintegration\b"
    r"[\s\S]{0,80}?"
    r"\bfor\s+([\w .'-]+?)\s*\.?\s*$",
    re.IGNORECASE,
)

_SINGLE_INTEGRATION_RE = re.compile(
    r"\b(?:create|add|build|make)\b"
    r"\s+(?:an?|a)\s+(?:new\s+)?([\w .'-]+?)\s+integration\b",
    re.IGNORECASE,
)


def slugify(service: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", service.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("A service name is required to scaffold an integration.")
    if cleaned[0].isdigit():
        cleaned = f"svc_{cleaned}"
    return cleaned


def classify_name(service: str) -> str:
    """Turn a free-text service name into a Pythonic CamelCase class name."""
    slug = slugify(service)
    parts = [part for part in slug.split("_") if part]
    return "".join(part[0].upper() + part[1:] for part in parts)


def _service_from_request(description: str) -> str | None:
    match = _INTEGRATION_RE.search(description.strip())
    if match:
        return match.group(1).strip()
    match = _SINGLE_INTEGRATION_RE.search(description.strip())
    if match:
        return match.group(1).strip()
    return None


def is_integration_scaffold_request(description: str) -> bool:
    if "integration" not in (description or "").casefold():
        return False
    return _service_from_request(description) is not None


def service_name_from_request(description: str) -> str:
    service = _service_from_request(description)
    if not service:
        raise ValueError("No integration service could be found in the request.")
    return service


def framework_guide() -> str:
    """Reference text handed to the LLM when planning integration code."""
    return """\
Reference: the JARVIS integrations framework.

To add a service JARVIS can use, create a package under src/integrations/<slug>/
containing:
- integration.py: a class 'XIntegration(Integration)' from integrations.base
  with a 'name', 'description', 'read_only', '_TOOLS' (names of @function_tool()
  methods), and '_LEVELS' mapping each tool name to an ActionLevel
  (SAFE_READ=1, REVERSIBLE=2, CONSEQUENTIAL=3, SECURITY=4).
- client.py: a client scaffold that raises 'XNotAuthenticatedError' with a
  friendly message until its required_env variables are configured.
- __init__.py re-exporting the public classes.
- tests/test_integrations_<slug>.py asserting unconfigured status and tools.
Every tool starts with self.gate.ensure_action_level(int(ActionLevel.<LEVEL>)).
Integration must return an honest status: available but not authenticated means
the voice layer reports "available but not connected" and never fakes data.
Never put real secrets in code; require environment/configuration variables.
"""


class IntegrationScaffold:
    """Builds an edit-plan proposal that creates a new integration package."""

    @staticmethod
    def proposal_for(service: str, description: str) -> dict[str, Any]:
        slug = slugify(service)
        class_name = classify_name(service)
        env_var = f"{slug.upper()}_TOKEN"
        label = slug.replace("_", " ").capitalize()
        replacements = {
            "__SLUG__": slug,
            "__CLASS__": class_name,
            "__ENVVAR__": env_var,
            "__LABEL__": label,
        }
        edits = [
            {
                "op": "create",
                "path": f"src/integrations/{slug}/__init__.py",
                "new": _INSTALL_TEMPLATES["init"].replace_chars(replacements),
                "reason": "Package entry point re-exporting the integration.",
            },
            {
                "op": "create",
                "path": f"src/integrations/{slug}/client.py",
                "new": _INSTALL_TEMPLATES["client"].replace_chars(replacements),
                "reason": "Credentials-aware client scaffold with a not-connected error.",
            },
            {
                "op": "create",
                "path": f"src/integrations/{slug}/integration.py",
                "new": _INSTALL_TEMPLATES["integration"].replace_chars(replacements),
                "reason": "Integration subclass exposing a status tool.",
            },
            {
                "op": "create",
                "path": f"tests/test_integrations_{slug}.py",
                "new": _INSTALL_TEMPLATES["test"].replace_chars(replacements),
                "reason": "Tests for unconfigured status and honest responses.",
            },
            {
                "op": "create",
                "path": f"docs/integrations/{slug}.md",
                "new": _INSTALL_TEMPLATES["doc"].replace_chars(replacements),
                "reason": "Setup documentation for the new integration.",
            },
        ]
        summary = (
            f"Created a standards-based {slug} integration with a client scaffold, "
            f"an {class_name} integration exposing status tooling, tests, and docs. "
            "It reports available-but-not-connected until configured."
        )
        return {"summary": summary, "edits": edits}


class _Template:
    def __init__(self, content: str) -> None:
        self.content = content

    def replace_chars(self, replacements: dict[str, str]) -> str:
        out = self.content
        for token, value in replacements.items():
            out = out.replace(token, value)
        return out


_INIT_TEMPLATE = '''\
"""__LABEL__ integration."""

from integrations.__SLUG__.client import (
    __CLASS__Client,
    __CLASS__NotAuthenticatedError,
)
from integrations.__SLUG__.integration import __CLASS__Integration

__all__ = [
    "__CLASS__Client",
    "__CLASS__Integration",
    "__CLASS__NotAuthenticatedError",
]
'''

_CLIENT_TEMPLATE = '''\
"""__CLASS__ client scaffold.

JARVIS accesses the __SLUG__ service through this client. No credentials exist
yet, so every data method raises __CLASS__NotAuthenticatedError with a friendly
message until the service is connected.
"""

from __future__ import annotations

import os
from typing import Any

REQUIRED_ENV = ("__ENVVAR__",)


class __CLASS__NotAuthenticatedError(RuntimeError):
    """Raised when a __SLUG__ call is attempted before connection."""


def _configured() -> bool:
    return bool(os.environ.get(REQUIRED_ENV[0], "").strip())


class __CLASS__Client:
    """Read-only abstraction (scaffolded)."""

    def __init__(self) -> None:
        self._configured = _configured()

    def is_authenticated(self) -> bool:
        return self._configured

    async def status(self) -> dict[str, Any]:
        self._require_authenticated()
        raise NotImplementedError(
            "__CLASS__ data access is implemented after connection."
        )

    def _require_authenticated(self) -> None:
        if not self._configured:
            raise __CLASS__NotAuthenticatedError(
                "__LABEL__ is available but not connected. Configure "
                "__ENVVAR__ to enable it."
            )
'''

_INTEGRATION_TEMPLATE = '''\
"""__CLASS__ capability (scaffold).

Provides an honest status tool. Extend _TOOLS with more @function_tool() methods
(each with an ActionLevel) to add real capabilities.
"""

from __future__ import annotations

from typing import Any, ClassVar

from integrations.__SLUG__.client import __CLASS__Client
from livekit.agents import RunContext, function_tool

from integrations.base import ActionLevel, Integration


class __CLASS__Integration(Integration):
    name = "__SLUG__"
    description = "Access __SLUG__ services."
    read_only = True
    default_level = ActionLevel.SAFE_READ
    _TOOLS = ("__SLUG___status",)
    # Tools inherit default_level (SAFE_READ) unless listed in _LEVELS below.
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {}

    def __init__(self, *, gate=None, failure_log=None, client=None):
        super().__init__(gate=gate, failure_log=failure_log)
        self._client = client or __CLASS__Client()

    def is_authenticated(self) -> bool:
        return self._client.is_authenticated()

    def status_note(self) -> str:
        return "" if self.is_authenticated() else "available but not connected"

    @function_tool()
    async def __SLUG___status(self, context: RunContext) -> dict[str, Any]:
        """Report whether the __SLUG__ service is connected."""
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        status = self.status()
        if not status.authenticated:
            return {
                "status": "not_connected",
                "message": (
                    "__LABEL__ is available but not connected. Do not invent data."
                ),
            }
        return {"status": "ok", "service": "__SLUG__"}
'''

_TEST_TEMPLATE = '''\
"""Tests for the __SLUG__ integration scaffold."""

import asyncio

from integrations.__SLUG__.integration import __CLASS__Integration


def test_unconfigured_status():
    integration = __CLASS__Integration()
    status = integration.status()
    assert status.available
    assert not status.authenticated
    assert status.read_only
    assert status.action_level == 1


def test_unconfigured_health():
    integration = __CLASS__Integration()
    health = asyncio.run(integration.health_check())
    assert health["status"] == "not_connected"
'''

_DOC_TEMPLATE = """\
# __LABEL__ integration

__CLASS__ capsulises access to the __SLUG__ service.

## Status

- Available without configuration; reports available-but-not-connected until
  __ENVVAR__ is set.

## Environment variables

- `__ENVVAR__`: credential for the __SLUG__ service.

## Adding real capabilities

Extend the client with real read methods and register further
`@function_tool()` methods in the integration's `_TOOLS` / `_LEVELS`.
"""

_INSTALL_TEMPLATES: dict[str, _Template] = {
    "init": _Template(_INIT_TEMPLATE),
    "client": _Template(_CLIENT_TEMPLATE),
    "integration": _Template(_INTEGRATION_TEMPLATE),
    "test": _Template(_TEST_TEMPLATE),
    "doc": _Template(_DOC_TEMPLATE),
}
