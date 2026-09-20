"""Tests for the BrowserTools wrappers and their gating behaviour.

The function_tool wrappers require a live RunContext, so these exercise the
async trear building blocks directly: the gate enforcement (no browser call is
made unless the user's turn has the wake word and an explicit browser request)
and the helper functions, using a stub browser so no real browser is needed.
"""

import pytest
from livekit.agents.llm import ToolError

from browser_tools import BrowserError
from gates import ToolGate
from tools import BrowserTools, duckduckgo_search_url


class StubBrowser:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def open_url(self, url: str) -> dict[str, str]:
        self.calls.append(f"open_url:{url}")
        return {"url": url}

    async def read_page(self) -> dict[str, str | bool]:
        self.calls.append("read_page")
        return {"text": ""}

    async def inspect_page(self) -> dict[str, object]:
        self.calls.append("inspect_page")
        return {"elements": []}

    async def go_back(self) -> dict[str, str]:
        self.calls.append("go_back")
        return {"url": ""}

    async def take_screenshot(self) -> dict[str, str | int | bool]:
        self.calls.append("take_screenshot")
        return {"path": ""}

    async def click(self, target: str) -> dict[str, str]:
        self.calls.append(f"click:{target}")
        return {"target": target}

    async def type_text(self, target: str, text: str) -> dict[str, str]:
        self.calls.append(f"type_text:{target}")
        return {"target": target}

    async def scroll(self, direction: str) -> dict[str, str]:
        self.calls.append(f"scroll:{direction}")
        return {"direction": direction}

    async def press_key(self, key: str) -> dict[str, str]:
        self.calls.append(f"press_key:{key}")
        return {"key": key}


class TestDuckDuckGoSearchUrl:
    def test_builds_query_string(self):
        url = duckduckgo_search_url("livekit agents")
        assert url == "https://duckduckgo.com/?q=livekit+agents"

    def test_urlencodes_special_chars(self):
        url = duckduckgo_search_url("café & bar")
        assert url == "https://duckduckgo.com/?q=caf%C3%A9+%26+bar"

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            duckduckgo_search_url("   ")


class TestGateWiring:
    @pytest.mark.asyncio
    async def test_open_url_blocked_without_wake_word(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("open google.com")
        with pytest.raises(ToolError, match="Jarvis"):
            await tools.open_url({}, url="https://google.com")
        assert browser.calls == []

    @pytest.mark.asyncio
    async def test_open_url_blocked_without_browser_intent(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("Jarvis, what is the weather")
        with pytest.raises(ToolError, match="browser"):
            await tools.open_url({}, url="https://google.com")
        assert browser.calls == []

    @pytest.mark.asyncio
    async def test_open_url_allowed_with_explicit_request(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("Jarvis, open google.com")
        result = await tools.open_url({}, url="https://google.com")
        assert result == {"url": "https://google.com"}
        assert browser.calls == ["open_url:https://google.com"]

    @pytest.mark.asyncio
    async def test_search_the_web_allowed_with_search_request(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("Jarvis, zoek naar livekit")
        await tools.search_the_web({}, query="livekit")
        assert browser.calls and browser.calls[0].startswith("open_url:")

    @pytest.mark.asyncio
    async def test_read_page_after_armed_browser(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("Jarvis, open nu.nl")
        gate.set_user_request("ja")
        await tools.read_page({})
        assert browser.calls == ["read_page"]

    @pytest.mark.asyncio
    async def test_browser_follow_up_allowed_in_active_conversation(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("Jarvis, open google.com")
        gate.ensure_wake_word()  # the wake word opens the conversation window
        gate.set_user_request("scroll down")
        await tools.scroll({}, direction="down")
        assert browser.calls == ["scroll:down"]

    @pytest.mark.asyncio
    async def test_browser_follow_up_blocked_without_active_conversation(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("open google.com")
        with pytest.raises(ToolError, match="Jarvis"):
            await tools.scroll({}, direction="down")
        assert browser.calls == []

    @pytest.mark.asyncio
    async def test_browser_error_wrapped_in_tool_error(self):
        class FailBrowser(StubBrowser):
            async def go_back(self) -> dict[str, str]:
                raise BrowserError("cannot go back")

        browser = FailBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("Jarvis, ga terug")
        with pytest.raises(ToolError, match="cannot go back"):
            await tools.go_back({})

    @pytest.mark.asyncio
    async def test_confirm_browser_action_requires_wake_word(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("just a comment, no wake word")
        with pytest.raises(ToolError, match="Jarvis"):
            await tools.confirm_browser_action({}, target="Buy")

    @pytest.mark.asyncio
    async def test_confirm_browser_action_allowed_after_request(self):
        browser = StubBrowser()
        gate = ToolGate()
        tools = BrowserTools(browser, gate=gate)
        gate.set_user_request("Jarvis, confirm clicking Buy")
        result = await tools.confirm_browser_action({}, target="Buy")
        assert "Buy" in result


class TestGateWiringAllow:
    def test_set_user_request_wires_to_gate(self):
        browser = StubBrowser()
        tools = BrowserTools(browser)
        tools.set_user_request("Jarvis, what is the weather")
        assert tools.gate.current_turn_text == "Jarvis, what is the weather"
        assert tools.gate.allows_reply() is True

    def test_set_user_request_empty_resets_gate(self):
        browser = StubBrowser()
        tools = BrowserTools(browser)
        tools.set_user_request("Jarvis, what is the weather")
        assert tools.gate.allows_reply() is True
        tools.set_user_request("")
        assert tools.gate.current_turn_text == ""
        assert tools.gate.allows_reply() is False

    def test_set_user_request_without_wake_word_keeps_gate_closed(self):
        browser = StubBrowser()
        tools = BrowserTools(browser)
        tools.set_user_request("the weather looks nice today")
        assert tools.gate.allows_reply() is False

    def test_set_user_request_none_resets_gate(self):
        browser = StubBrowser()
        tools = BrowserTools(browser)
        tools.set_user_request("Jarvis, open google.com")
        assert tools.gate.allows_reply() is True
        tools.set_user_request(None)
        assert tools.gate.allows_reply() is False


class TestRequiresConfirmation:
    def test_risky_target(self):
        assert BrowserTools._requires_confirmation("Buy now")

    def test_harmless_target(self):
        assert not BrowserTools._requires_confirmation("Read more")
