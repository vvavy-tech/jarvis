"""Stability and voice-safety tests for the on-demand screen-vision tools.

These tests never capture the real desktop, never call the network, and never
touch the realtime voice pipeline. Backends and vision analyzers are fakes, so
every behaviour is exercised hermetically. The core guarantee under test: a
screen-tool failure is always a normal ToolError, never a crash or a stall of
the agent session.
"""

import os
import sys
import time
from pathlib import Path

import pytest
from livekit.agents.llm import ToolError
from PIL import Image

from gates import ToolGate
from integrations import ScreenVisionIntegration, build_default_registry
from integrations.registry import CapabilityRegistry
from integrations.screen_vision import capture
from integrations.screen_vision.analysis import VisionAnalyzer


def run_sync(coro):
    import asyncio

    return asyncio.run(coro)


_PRIMARY = {
    "id": 0,
    "left": 0,
    "top": 0,
    "width": 1920,
    "height": 1080,
    "primary": True,
    "name": r"\\.\DISPLAY1",
}
_RIGHT_OF_PRIMARY = {
    "id": 1,
    "left": 1920,
    "top": -348,
    "width": 1080,
    "height": 1920,
    "primary": False,
    "name": r"\\.\DISPLAY2",
}


class _FakeBackend:
    def __init__(self, *, fail_capture=False, cursor=None, track=False):
        self.fail_capture = fail_capture
        self.cursor = dict(cursor or {"x": 2070, "y": 100})
        self.calls = [] if track else None

    def _record(self, name):
        if self.calls is not None:
            self.calls.append(name)

    def list_monitors(self):
        self._record("list_monitors")
        return [dict(_PRIMARY), dict(_RIGHT_OF_PRIMARY)]

    def get_cursor_position(self):
        self._record("get_cursor_position")
        return dict(self.cursor)

    def capture(self, bbox):
        self._record("capture")
        if self.fail_capture:
            raise RuntimeError("synthetic capture failure")
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        return Image.new("RGB", (width, height), "blue")


class _FakeStream:
    """Mirrors the real plugin: ``to_str_iterable()`` is a plain method that
    returns an async iterator (never await ``chat()``/``to_str_iterable()``)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def to_str_iterable(self):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()


class _FakeLLM:
    """Mirrors ``google.LLM``: ``chat()`` is synchronous and returns a stream."""

    def __init__(self, chunks=("The screen shows a calendar view.",), fail=False):
        self._chunks = list(chunks)
        self.fail = fail
        self.requests = []

    def chat(self, *, chat_ctx):
        if self.fail:
            raise RuntimeError("synthetic analysis failure")
        self.requests.append(chat_ctx)
        return _FakeStream(self._chunks)


def _fake_analyzer(fail=False, chunks=("The screen shows a calendar view.",)):
    return VisionAnalyzer(llm_factory=lambda: _FakeLLM(chunks=chunks, fail=fail))


def _vision(*, backend=None, analyzer=None, gate=None):
    if backend is None:
        backend = _FakeBackend()
    if analyzer is None:
        analyzer = _fake_analyzer()
    return ScreenVisionIntegration(gate=gate, backend=backend, analyzer=analyzer)


def _active_turn_vision():
    gate = ToolGate()
    gate.set_user_request("Jarvis, look at my screen")
    gate.conversation.activate()
    integration = _vision(gate=gate)
    return gate, integration


# --------------------------------------------------------------------------- #
# registration + availability
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "win32", reason="capture is Windows-only")
def test_default_registry_registers_screen_vision():
    registry = build_default_registry()
    assert "screen_vision" in registry
    assert "screen_vision" in registry.names()


def test_registered_when_backend_works():
    registry = CapabilityRegistry()
    registry.register(ScreenVisionIntegration(backend=_FakeBackend()))
    assert "screen_vision" in registry.names()
    screen = registry.get("screen_vision")
    assert screen is not None
    assert screen.is_available()
    tool_names = {tool.info.name for tool in registry.tools()}
    assert "capture_monitor" in tool_names
    assert "analyze_screen_on_demand" in tool_names


def test_unavailable_integration_is_excluded_from_tools():
    registry = CapabilityRegistry()
    integration = ScreenVisionIntegration(backend=None)
    registry.register(integration)
    status = integration.status()
    assert not status.available
    assert not status.authenticated
    assert "screen" in integration.status_note().casefold()
    assert not any("screen" in str(tool).casefold() for tool in registry.tools())


def test_unavailable_when_capture_platform_missing(monkeypatch):
    if sys.platform == "win32":
        monkeypatch.setattr(sys, "platform", "linux")
    integration = _vision(backend=_FakeBackend())
    assert not integration.is_available()
    assert integration.status().note


def test_screen_tools_are_safe_read_only_level_one():
    integration = _vision()
    assert integration.read_only
    assert not integration.confirmation_required
    for tool in integration.tool_names:
        assert int(integration.tool_level(tool)) == 1


# --------------------------------------------------------------------------- #
# monitor / cursor / capture behaviour
# --------------------------------------------------------------------------- #


def test_list_monitors_returns_clean_metadata():
    _, integration = _active_turn_vision()
    result = run_sync(integration.list_monitors(None))
    assert [monitor["id"] for monitor in result] == [0, 1]
    for monitor in result:
        assert set(monitor) >= {"id", "left", "top", "width", "height", "primary"}
    assert result[0]["primary"] is True


@pytest.mark.skipif(
    sys.platform != "win32", reason="monitor enumeration is Windows-only"
)
def test_real_monitor_metadata_matches_contract():
    from integrations.screen_vision import monitors

    results = monitors.monitors()
    assert results
    for monitor in results:
        assert set(monitor) >= {
            "id",
            "left",
            "top",
            "width",
            "height",
            "primary",
            "name",
        }
        assert monitor["width"] >= 0 and monitor["height"] >= 0
    assert sum(1 for monitor in results if monitor["primary"]) == 1


def test_analysis_handles_sync_chat_contract(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    fake = _FakeLLM()
    analyzer = VisionAnalyzer(llm_factory=lambda: fake)
    integration = _vision(analyzer=analyzer)
    gate = ToolGate()
    gate.set_user_request("Jarvis, what is on my screen?")
    gate.conversation.activate()
    integration.gate = gate
    result = run_sync(integration.analyze_screen_on_demand(None, question="what"))
    assert "calendar" in result["analysis"].casefold()
    assert len(fake.requests) == 1


def test_get_cursor_position():
    _, integration = _active_turn_vision()
    result = run_sync(integration.get_cursor_position(None))
    assert result == {"x": 2070, "y": 100}


def test_cursor_region_clips_into_negative_top_monitor():
    monitors = [dict(_PRIMARY), dict(_RIGHT_OF_PRIMARY)]
    region = capture.cursor_region(monitors, {"x": 2070, "y": 100}, 800, 600)
    assert region == (1920, -200, 2720, 400)


def test_cursor_region_falls_back_to_union_when_between_monitors():
    monitors = [dict(_PRIMARY), dict(_RIGHT_OF_PRIMARY)]
    region = capture.cursor_region(monitors, {"x": 1918, "y": 0}, 800, 600)
    assert region is not None


def test_cursor_region_empty_without_monitors():
    assert capture.cursor_region([], {"x": 0, "y": 0}, 800, 600) is None


def test_capture_monitor_saves_and_reports_temp_path():
    _, integration = _active_turn_vision()
    result = run_sync(integration.capture_monitor(None, monitor_id=0))
    path = Path(result["image_path"])
    assert path.exists()
    assert result["monitor"]["id"] == 0
    capture.remove_temp_image(path)


def test_capture_cursor_area():
    _, integration = _active_turn_vision()
    result = run_sync(integration.capture_cursor_area(None))
    path = Path(result["image_path"])
    assert path.exists()
    assert result["area"] == "cursor"
    capture.remove_temp_image(path)


# --------------------------------------------------------------------------- #
# failure isolation: capture + vision never crash the session
# --------------------------------------------------------------------------- #


def test_capture_failure_is_a_normal_tool_error_and_recovers():
    backend = _FakeBackend(fail_capture=True)
    integration = _vision(backend=backend)
    gate = ToolGate()
    gate.set_user_request("Jarvis, look at my screen")
    gate.conversation.activate()
    integration.gate = gate
    with pytest.raises(ToolError):
        run_sync(integration.capture_monitor(None, monitor_id=0))
    backend.fail_capture = False
    result = run_sync(integration.capture_monitor(None, monitor_id=0))
    assert Path(result["image_path"]).exists()
    capture.remove_temp_image(result["image_path"])


def test_vision_failure_is_a_normal_tool_error_and_recovers(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    integration = _vision(analyzer=_fake_analyzer(fail=True))
    gate = ToolGate()
    gate.set_user_request("Jarvis, what is on my screen?")
    gate.conversation.activate()
    integration.gate = gate
    with pytest.raises(ToolError):
        run_sync(integration.analyze_screen_on_demand(None, question="what is this?"))
    integration._analyzer = _fake_analyzer(fail=False)
    result = run_sync(
        integration.analyze_screen_on_demand(None, question="what is this?")
    )
    assert "calendar" in result["analysis"].casefold()


def test_screen_tools_fail_before_backend_access_without_wake_word():
    backend = _FakeBackend(track=True)
    gate = ToolGate()
    gate.set_user_request("the weather is nice today")
    integration = _vision(backend=backend, gate=gate)
    with pytest.raises(ToolError, match="wake word"):
        run_sync(integration.list_monitors(None))
    assert backend.calls == []


def test_chain_of_failures_then_success_leaves_session_fully_usable():
    monkeypatch_placeholder = pytest.MonkeyPatch()
    monkeypatch_placeholder.setenv("GOOGLE_API_KEY", "test-key")
    try:
        backend = _FakeBackend(fail_capture=True)
        analyzer = _fake_analyzer(fail=True)
        integration = _vision(backend=backend, analyzer=analyzer)
        gate = ToolGate()
        gate.set_user_request("Jarvis, look at my screen")
        gate.conversation.activate()
        integration.gate = gate

        with pytest.raises(ToolError):
            run_sync(integration.capture_monitor(None, monitor_id=0))
        with pytest.raises(ToolError):
            run_sync(integration.analyze_screen_on_demand(None, question="what"))

        backend.fail_capture = False
        integration._analyzer = _fake_analyzer(fail=False)
        status = run_sync(integration.screen_status(None))
        assert status["available"] is True
        result = run_sync(integration.analyze_screen_on_demand(None, question="what"))
        assert "calendar" in result["analysis"].casefold()
    finally:
        monkeypatch_placeholder.undo()


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _Model404ThenSuccessLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, *, chat_ctx):
        self.calls += 1
        if self.calls == 1:
            raise _StatusError(404)
        return _FakeStream(("recovered text",))


class _ServerErrorLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, *, chat_ctx):
        self.calls += 1
        raise _StatusError(500)


class _Always404LLM:
    def chat(self, *, chat_ctx):
        raise _StatusError(404)


def _analyze_direct(analyzer, question="what"):
    path = capture.save_temp_image(Image.new("RGB", (8, 8), "red"))
    try:
        return run_sync(analyzer.analyse(path, question))
    finally:
        capture.remove_temp_image(path)


def test_retired_model_falls_back_to_default_once(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JARVIS_VISION_MODEL", "gemini-2.0-ghost")
    fake = _Model404ThenSuccessLLM()
    analyzer = VisionAnalyzer(llm_factory=lambda: fake)
    result = _analyze_direct(analyzer)
    assert "recovered" in result.casefold()
    assert fake.calls == 2


def test_non_model_error_does_not_retry(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    fake = _ServerErrorLLM()
    analyzer = VisionAnalyzer(llm_factory=lambda: fake)
    with pytest.raises(ToolError, match="trouble reading"):
        _analyze_direct(analyzer)
    assert fake.calls == 1  # no fallback retry for non-404 errors


def test_all_models_retired_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JARVIS_VISION_MODEL", "gemini-2.0-ghost")
    analyzer = VisionAnalyzer(llm_factory=lambda: _Always404LLM())
    with pytest.raises(ToolError, match="trouble reading"):
        _analyze_direct(analyzer)


# --------------------------------------------------------------------------- #
# on-demand analysis only + privacy
# --------------------------------------------------------------------------- #


def test_analysis_requires_explicit_screen_request():
    backend = _FakeBackend(track=True)
    gate = ToolGate()
    gate.set_user_request("tell me a joke")
    integration = _vision(backend=backend, gate=gate)
    with pytest.raises(ToolError, match="wake word"):
        run_sync(integration.analyze_screen_on_demand(None, question="joke"))
    assert backend.calls == []


def test_analysis_removes_owned_temp_file(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def temp_files():
        return set(capture.SCREEN_CAPTURE_DIR.glob("screen_*.png"))

    capture.SCREEN_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    before = temp_files()
    _, integration = _active_turn_vision()
    result = run_sync(integration.analyze_screen_on_demand(None, question="what"))
    assert "calendar" in result["analysis"].casefold()
    assert temp_files() == before


def test_analysis_keeps_user_supplied_image(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from integrations.screen_vision.capture import save_temp_image

    path = save_temp_image(Image.new("RGB", (64, 64), "red"))
    try:
        _, integration = _active_turn_vision()
        result = run_sync(
            integration.analyze_screen_on_demand(
                None, question="what", image_path=str(path)
            )
        )
        assert "calendar" in result["analysis"].casefold()
        assert path.exists()  # caller-owned file is not deleted
    finally:
        capture.remove_temp_image(path)


def test_unconfigured_vision_raises_tool_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    _, integration = _active_turn_vision()
    with pytest.raises(ToolError, match="not configured"):
        run_sync(integration.analyze_screen_on_demand(None, question="what"))


def test_vision_configured_flag_reflects_api_key(monkeypatch):
    integration = _vision()
    gate = ToolGate()
    gate.set_user_request("Jarvis, screen status")
    gate.conversation.activate()
    integration.gate = gate
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    status = run_sync(integration.screen_status(None))
    assert status["vision_configured"] is False
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    status = run_sync(integration.screen_status(None))
    assert status["vision_configured"] is True


# --------------------------------------------------------------------------- #
# temp-file cleanup
# --------------------------------------------------------------------------- #


def test_save_temp_image_prunes_stale_captures():
    stale = capture.SCREEN_CAPTURE_DIR / "screen_999999999999999999.png"
    capture.SCREEN_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x")
    old = time.time() - 2000
    os.utime(stale, (old, old))
    try:
        fresh = capture.save_temp_image(Image.new("RGB", (8, 8), "red"))
        assert not stale.exists()
        assert fresh.exists()
    finally:
        capture.remove_temp_image(stale)
        capture.remove_temp_image(fresh)


# --------------------------------------------------------------------------- #
# the prompt teaches explicit-request-only usage
# --------------------------------------------------------------------------- #


def test_prompt_forbids_automatic_screen_capture():
    import inspect

    from agent import Assistant

    source = inspect.getsource(Assistant).casefold()
    assert source.count("# screen vision") >= 1
    assert "only when they ask" in source
    assert "never inspect or capture the screen automatically" in source
