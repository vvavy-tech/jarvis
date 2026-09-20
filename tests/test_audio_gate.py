"""Tests for the realtime audio-output wake-word gate.

The agent runs a realtime model with server-side turn detection, so the model
generates a reply before we could ever veto it. The deterministic enforcement
point is `realtime_audio_output_node`: model audio is buffered briefly and only
released once the current turn is authorized to speak.

These tests exercise `Assistant._gate_realtime_audio` directly without a live
LiveKit session or Playwright, against the module-global tool gate shared by
`agent.browser_tools`.
"""

import asyncio
from collections.abc import AsyncIterable

import pytest
from livekit import rtc

import agent

SAMPLE_RATE = 24000


def _frame(index: int) -> rtc.AudioFrame:
    data = (index & 0xFF).to_bytes(1, "little") * 1920
    return rtc.AudioFrame(
        data=data,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=960,
    )


async def _frames(count: int, delay_s: float = 0.0) -> AsyncIterable[rtc.AudioFrame]:
    for index in range(count):
        if delay_s:
            await asyncio.sleep(delay_s)
        yield _frame(index)


async def _collect(source) -> list[int]:
    result = []
    async for frame in source:
        result.append(frame.data[0] & 0xFF)
    return result


class TestAudioGate:
    @pytest.fixture(autouse=True)
    def _reset_global_gate(self):
        gate = agent.browser_tools.gate
        gate.deactivate()
        gate.set_user_request(None)
        yield

    async def test_allowed_turn_passes_straight_through(self):
        agent.browser_tools.gate.set_user_request("Jarvis, open google.com")
        out = await _collect(
            agent.Assistant()._gate_realtime_audio(_frames(3), settle_s=0.05)
        )
        assert out == [0, 1, 2]

    async def test_unauthorized_turn_is_muted(self):
        agent.browser_tools.gate.set_user_request("the weather looks nice")
        out = await _collect(
            agent.Assistant()._gate_realtime_audio(_frames(3), settle_s=0.05)
        )
        assert out == []

    async def test_wake_word_arriving_late_releases_buffered_audio(self):
        assistant = agent.Assistant()
        source = _frames(8, delay_s=0.02)

        async def _speak_late() -> None:
            await asyncio.sleep(0.06)
            agent.browser_tools.gate.set_user_request("Jarvis, open google.com")

        task = asyncio.create_task(_speak_late())
        out = await _collect(assistant._gate_realtime_audio(source, settle_s=0.5))
        assert out == [0, 1, 2, 3, 4, 5, 6, 7]
        await task

    async def test_confirmation_continuation_stays_audible(self):
        gate = agent.browser_tools.gate
        gate.set_user_request("Jarvis, open google.com")
        assert gate.allows_reply() is True  # wake word opens the conversation window
        gate.set_user_request("ja")
        out = await _collect(
            agent.Assistant()._gate_realtime_audio(_frames(2), settle_s=0.05)
        )
        assert out == [0, 1]

    async def test_conversation_follow_up_releases_audio_without_wake_word(self):
        gate = agent.browser_tools.gate
        gate.set_user_request("Jarvis, open google.com")
        assert gate.allows_reply() is True
        gate.set_user_request("scroll down")
        out = await _collect(
            agent.Assistant()._gate_realtime_audio(_frames(3), settle_s=0.05)
        )
        assert out == [0, 1, 2]

    async def test_sleep_command_mutes_following_ignored_turn(self):
        gate = agent.browser_tools.gate
        gate.set_user_request("Jarvis, what time is it")
        assert gate.allows_reply() is True  # opens the conversation window
        gate.set_user_request("Jarvis, go to sleep")
        assert gate.allows_reply() is True  # brief goodbye is still allowed
        assert gate.is_conversation_active() is False
        gate.set_user_request("what time is it")
        out = await _collect(
            agent.Assistant()._gate_realtime_audio(_frames(3), settle_s=0.05)
        )
        assert out == []

    async def test_wake_word_after_default_settle_is_still_audible(self):
        """A real user's transcript arrives long after the old 0.7s cap.

        ASR routinely takes ~1s, while the model starts preemptive generation
        immediately. The reply must still be audible when the wake word
        arrives after the historical default settle window.
        """
        assistant = agent.Assistant()
        source = _frames(50, delay_s=0.03)

        async def _wake_late() -> None:
            await asyncio.sleep(1.0)
            agent.browser_tools.gate.set_user_request("Jarvis, tell me about yourself")

        task = asyncio.create_task(_wake_late())
        out = await _collect(assistant._gate_realtime_audio(source))
        await task
        assert out == list(range(50))

    async def test_late_wake_word_after_settle_stays_muted(self):
        agent.browser_tools.gate.set_user_request("the weather looks nice")
        assistant = agent.Assistant()
        source = _frames(4, delay_s=0.03)

        async def _wake_too_late() -> None:
            await asyncio.sleep(0.2)
            agent.browser_tools.gate.set_user_request("Jarvis, open google.com")

        task = asyncio.create_task(_wake_too_late())
        out = await _collect(assistant._gate_realtime_audio(source, settle_s=0.08))
        assert out == []
        await task
