"""Tests for the integration framework: action levels, base, registry, and groups."""

import pytest
from livekit.agents import function_tool

from integrations import (
    ActionLevel,
    ToolGroup,
    build_default_registry,
)
from integrations.base import Integration, level_label
from integrations.registry import CapabilityRegistry

_OPTIONAL_CREDENTIAL_ENV = (
    "GMAIL_OAUTH_CREDENTIALS",
    "GOOGLE_OAUTH_CREDENTIALS",
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "META_ACCESS_TOKEN",
    "META_AD_ACCOUNT_ID",
)


class TestActionLevel:
    def test_levels_and_labels(self):
        assert int(ActionLevel.SAFE_READ) == 1
        assert int(ActionLevel.REVERSIBLE) == 2
        assert int(ActionLevel.CONSEQUENTIAL) == 3
        assert int(ActionLevel.SECURITY) == 4
        assert level_label(ActionLevel.SAFE_READ) == "safe_read"
        assert level_label(ActionLevel.REVERSIBLE) == "reversible_action"
        assert level_label(ActionLevel.CONSEQUENTIAL) == "consequential_action"
        assert level_label(ActionLevel.SECURITY) == "security_change"


class _ProbeIntegration(Integration):
    name = "probe"
    description = "test integration"
    read_only = True

    def is_authenticated(self) -> bool:
        return False


class TestCapabilityRegistry:
    def test_default_registry_registers_standard_services(self, monkeypatch):
        for env in _OPTIONAL_CREDENTIAL_ENV:
            monkeypatch.delenv(env, raising=False)
        registry = build_default_registry()
        names = registry.names()
        assert "gmail" in names
        assert "google_calendar" in names
        assert "meta_ads" in names
        assert "spotify" in names
        assert "windows" in names

    def test_unconnected_services_are_honest(self, monkeypatch):
        for env in _OPTIONAL_CREDENTIAL_ENV:
            monkeypatch.delenv(env, raising=False)
        registry = build_default_registry()
        for capability in registry.capabilities():
            if capability["name"] in {"gmail", "google_calendar", "meta_ads"}:
                assert capability["authenticated"] is False
                assert capability["available"] is True
                assert "not connected" in capability["note"]
        spotify = registry.get("spotify").status().to_dict()
        assert spotify["authenticated"] is False
        assert "not connected" in spotify["note"]

    def test_tool_levels_are_explicit_metadata(self, monkeypatch):
        for env in _OPTIONAL_CREDENTIAL_ENV:
            monkeypatch.delenv(env, raising=False)
        registry = build_default_registry()
        levels = registry.tool_levels()
        assert levels["spotify_status"] == 1
        assert levels["open_spotify"] == 2
        assert levels["media_play_pause"] == 2
        assert levels["windows_status"] == 1
        assert levels["open_application"] == 2
        assert levels["read_recent_emails"] == 1
        assert levels["meta_ads_overview"] == 1
        assert levels["calendar_events"] == 1

    def test_register_requires_unique_name(self):
        registry = CapabilityRegistry()
        registry.register(_ProbeIntegration())
        with pytest.raises(ValueError):
            registry.register(_ProbeIntegration())

    def test_register_requires_nonempty_name(self):
        class Nameless(Integration):
            name = ""

        registry = CapabilityRegistry()
        with pytest.raises(ValueError):
            registry.register(Nameless())

    def test_report_lists_every_capability(self, monkeypatch):
        for env in _OPTIONAL_CREDENTIAL_ENV:
            monkeypatch.delenv(env, raising=False)
        registry = build_default_registry()
        report = registry.report().casefold()
        assert "gmail" in report
        assert "windows" in report
        assert "not connected" in report

    def test_connected_names_excludes_unconnected(self, monkeypatch):
        for env in _OPTIONAL_CREDENTIAL_ENV:
            monkeypatch.delenv(env, raising=False)
        registry = build_default_registry()
        assert "windows" in registry.connected_names()
        assert "gmail" not in registry.connected_names()


class TestToolGroup:
    def test_group_wraps_external_tools_with_levels(self):
        @function_tool(name="probe_tool")
        async def probe_tool(context):
            return 1

        group = ToolGroup(
            "module",
            "module tools",
            [probe_tool],
            tool_names=["probe_tool"],
            level=ActionLevel.REVERSIBLE,
        )
        assert group.tool_names == ["probe_tool"]
        assert int(group.tool_level("probe_tool")) == 2
        assert group.confirmation_required is False
        assert group.tools == [probe_tool]

    def test_high_level_group_requires_confirmation(self):
        group = ToolGroup(
            "risky",
            "high-impact tools",
            [],
            tool_names=[],
            level=ActionLevel.CONSEQUENTIAL,
        )
        assert group.confirmation_required is True
        assert int(group.tool_level("anything")) == 3
