"""Tests for development-mode request detection and gating."""

import pytest
from livekit.agents.llm import ToolError

from gates import ToolGate, is_developer_request


class TestIsDeveloperRequest:
    def test_developer_mode(self):
        assert is_developer_request("Jarvis, enter developer mode")

    def test_improve_your_code(self):
        assert is_developer_request("Jarvis, please improve your code")

    def test_add_ability(self):
        assert is_developer_request("Jarvis, add the ability to control my windows")

    def test_create_integration(self):
        assert is_developer_request("Jarvis, create a calendar integration")

    def test_fix_your_bug(self):
        assert is_developer_request("Jarvis, fix your browser bug")

    def test_refactor(self):
        assert is_developer_request("Jarvis, refactor gates.py")

    def test_show_tasks(self):
        assert is_developer_request("Jarvis, show the development tasks")

    def test_list_tasks(self):
        assert is_developer_request("Jarvis, list my tasks")

    def test_what_did_you_change(self):
        assert is_developer_request("Jarvis, what did you change?")

    def test_approve_change(self):
        assert is_developer_request("Jarvis, approve the change")

    def test_reject_task(self):
        assert is_developer_request("Jarvis, reject the task")

    def test_cancel_task(self):
        assert is_developer_request("Jarvis, cancel the task")

    def test_rollback(self):
        assert is_developer_request("Jarvis, rollback")

    def test_enable_maintenance(self):
        assert is_developer_request("Jarvis, enable maintenance")

    def test_status(self):
        assert is_developer_request("Jarvis, what is the status of my task")

    def test_weather_is_not_developer(self):
        assert not is_developer_request("Jarvis, what is the weather today")

    def test_plan_is_not_developer(self):
        assert not is_developer_request("Jarvis, help me develop a training plan")

    def test_none_or_empty(self):
        assert not is_developer_request(None)
        assert not is_developer_request("")

    def test_case_insensitive(self):
        assert is_developer_request("JARVIS, IMPROVE YOUR CODE")


class TestEnsureDeveloperRequested:
    def test_blocked_without_wake_word(self):
        gate = ToolGate()
        gate.set_user_request("improve your code")
        with pytest.raises(ToolError, match="Jarvis"):
            gate.ensure_developer_requested()

    def test_blocked_for_plain_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, what is the weather")
        with pytest.raises(ToolError, match="developer"):
            gate.ensure_developer_requested()

    def test_allowed_with_explicit_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, enter developer mode and improve your code")
        gate.ensure_developer_requested()

    def test_allowed_with_control_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, show the development tasks")
        gate.ensure_developer_requested()

    def test_blocked_without_request(self):
        gate = ToolGate()
        with pytest.raises(ToolError, match="development"):
            gate.ensure_developer_requested()
