"""Tests for the capability status tools and the voice-gate action levels."""

import pytest
from livekit.agents.llm import ToolError

from gates import ToolGate
from integrations import build_default_registry
from integrations.capability_tools import CapabilityTools


def _tools_for(turn: str):
    gate = ToolGate()
    gate.set_user_request(turn)
    registry = build_default_registry()
    tools = CapabilityTools(registry, gate=gate)
    return gate, tools


class TestGetCapabilities:
    def test_reports_capabilities_and_action_levels(self):
        _, tools = _tools_for("Jarvis, what can you do?")
        result = run_sync(tools.get_capabilities(None))
        assert "capabilities" in result
        assert isinstance(result["capabilities"], list)
        assert result["action_levels"]["consequential_action"]
        assert "never invent" in result["rule"].casefold()

    def test_blocked_without_wake_word_or_conversation(self):
        _, tools = _tools_for("turn the lights on")
        with pytest.raises(ToolError, match="wake word"):
            run_sync(tools.get_capabilities(None))

    def test_works_in_active_conversation(self):
        gate, tools = _tools_for("Jarvis, tell me about yourself")
        gate.conversation.activate()
        result = run_sync(tools.get_capabilities(None))
        assert result


class TestSystemStatus:
    def test_reports_health_per_service(self):
        _, tools = _tools_for("Jarvis, what is connected?")
        result = run_sync(tools.system_status(None))
        assert "report" in result
        assert "health" in result
        assert isinstance(result["health"], list)
        assert all("status" in item for item in result["health"])

    def test_unconnected_service_told_plainly(self, monkeypatch):
        for env in (
            "GMAIL_OAUTH_CREDENTIALS",
            "GOOGLE_OAUTH_CREDENTIALS",
            "META_ACCESS_TOKEN",
            "META_AD_ACCOUNT_ID",
        ):
            monkeypatch.delenv(env, raising=False)
        _, tools = _tools_for("Jarvis, what is connected?")
        health = run_sync(tools.system_status(None))["health"]
        unconnected = [
            h for h in health if h["name"] in {"gmail", "google_calendar", "meta_ads"}
        ]
        assert unconnected
        assert all(h["status"] == "not_connected" for h in unconnected)


class TestActionLevelGate:
    def test_level_one_needs_only_conversation(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, tell me my status")
        assert gate.ensure_action_level(1) is None

    def test_level_two_needs_only_conversation(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, pause the music")
        assert gate.ensure_action_level(2) is None

    def test_level_three_requires_consequential_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, play my music please")
        with pytest.raises(ToolError, match="explicit approval"):
            gate.ensure_action_level(3)

    def test_level_three_allows_consequential_verb(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, send a reply to my email")
        assert gate.ensure_action_level(3) is None

    def test_level_three_allows_confirmations(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, send the email")
        gate.ensure_action_level(3)
        gate.set_user_request("yes")
        assert gate.ensure_action_level(3) is None

    def test_level_three_skipped_when_requested_flag(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, continue the merge")
        assert gate.ensure_action_level(3, requested=True) is None

    def test_security_level_requires_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, what time is it?")
        with pytest.raises(ToolError, match="explicit approval"):
            gate.ensure_action_level(4)


def run_sync(coro):
    import asyncio

    return asyncio.run(coro)
