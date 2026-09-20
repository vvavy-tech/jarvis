"""Unit tests for the wake-word and browser-intent gating logic.

These exercise the pure helpers and the ToolGate used to enforce that the
agent stays silent and never touches the browser unless the user's current
utterance contains the wake word and an explicit browser request.
"""

import pytest
from livekit.agents.llm import ToolError

from gates import (
    CONVERSATION_TIMEOUT_SECONDS,
    ConversationGate,
    ToolGate,
    browser_intent,
    has_wake_word,
    is_confirmation,
    is_sleep_command,
)


class _FixedClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------ #
# wake word detection
# ------------------------------------------------------------------ #


class TestHasWakeWord:
    def test_plain_jarvis(self):
        assert has_wake_word("Jarvis")

    def test_lowercase(self):
        assert has_wake_word("jarvis")

    def test_uppercase(self):
        assert has_wake_word("JARVIS")

    def test_at_start(self):
        assert has_wake_word("Jarvis, what is the weather?")

    def test_in_middle(self):
        assert has_wake_word("Hey Jarvis open google")

    def test_at_end(self):
        assert has_wake_word("Thanks Jarvis")

    def test_followed_by_punctuation(self):
        assert has_wake_word("hello jarvis!")

    def test_diacritics_ignored(self):
        assert has_wake_word("Járvis")

    def test_no_wake_word(self):
        assert not has_wake_word("What time is it?")

    def test_background_speech(self):
        assert not has_wake_word("and then the movie ended quite abruptly")

    def test_empty(self):
        assert not has_wake_word("")
        assert not has_wake_word(None)

    def test_jarvis_as_subword_not_triggered_by_prefix_only(self):
        assert not has_wake_word("jar")


# ------------------------------------------------------------------ #
# browser intent detection
# ------------------------------------------------------------------ #


class TestBrowserIntent:
    def test_named_url(self):
        assert browser_intent("Jarvis open google.com")

    def test_https_url(self):
        assert browser_intent("Jarvis, go to https://example.com")

    def test_www_url(self):
        assert browser_intent("jarvis open www.espn.com")

    def test_dutch_ga_naar_domain(self):
        assert browser_intent("Jarvis ga naar nu.nl")

    def test_browser_word(self):
        assert browser_intent("Jarvis, open the browser")

    def test_internet_word(self):
        assert browser_intent("Jarvis, check that on the internet")

    def test_website_word(self):
        assert browser_intent("Jarvis, wat staat er op de website van de NOS?")

    def test_tab_word(self):
        assert browser_intent("Jarvis, open een tab")

    def test_search_english(self):
        assert browser_intent("Jarvis, search for livekit agents")

    def test_search_dutch(self):
        assert browser_intent("Jarvis, zoek de beste restaurants in Utrecht")

    def test_search_op(self):
        assert browser_intent("Jarvis, zoek op wat de hoofdstad is")

    def test_google_as_verb(self):
        assert browser_intent("Jarvis, google what time sunrise is today")

    def test_continuation_scroll(self):
        assert browser_intent("Jarvis, scroll down")

    def test_continuation_go_back(self):
        assert browser_intent("Jarvis, ga terug")

    def test_continuation_next(self):
        assert browser_intent("Jarvis, volgende pagina")

    def test_plain_question_not_browser(self):
        assert not browser_intent("what is the weather today")

    def test_smalltalk_not_browser(self):
        assert not browser_intent("tell me a joke")

    def test_local_action_not_browser(self):
        assert not browser_intent("open the door")

    def test_bed_not_browser(self):
        assert not browser_intent("I need to go to bed")

    def test_computation_not_browser(self):
        assert not browser_intent("what is two plus two")

    def test_none_or_empty(self):
        assert not browser_intent("")
        assert not browser_intent(None)


# ------------------------------------------------------------------ #
# confirmation detection
# ------------------------------------------------------------------ #


class TestIsConfirmation:
    def test_dutch_ja(self):
        assert is_confirmation("ja")

    def test_english_yes(self):
        assert is_confirmation("yes")

    def test_go_ahead(self):
        assert is_confirmation("go ahead")

    def test_dutch_ga_door(self):
        assert is_confirmation("ga door")

    def test_okay(self):
        assert is_confirmation("okay")

    def test_doe_maar(self):
        assert is_confirmation("doe maar")

    def test_not_confirmation(self):
        assert not is_confirmation("what time is it")


# ------------------------------------------------------------------ #
# ToolGate flow
# ------------------------------------------------------------------ #


class TestToolGate:
    def test_requires_wake_word(self):
        gate = ToolGate()
        gate.set_user_request("what is the weather")
        with pytest.raises(ToolError, match="Jarvis"):
            gate.ensure_wake_word()

    def test_allows_with_wake_word(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, what is the weather")
        gate.ensure_wake_word()

    def test_browser_requires_wake_word(self):
        gate = ToolGate()
        gate.set_user_request("open google.com")
        with pytest.raises(ToolError, match="Jarvis"):
            gate.ensure_browser_requested()

    def test_browser_blocked_without_intent(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, what is the weather")
        with pytest.raises(ToolError, match="browser"):
            gate.ensure_browser_requested()

    def test_browser_allowed_with_intent(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, open espn.com")
        gate.ensure_browser_requested()

    def test_browser_allowed_with_search(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, zoek naar livekit")
        gate.ensure_browser_requested()

    def test_confirmation_continues_armed_browser(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, open google.com")
        gate.ensure_browser_requested()
        gate.set_user_request("ja")
        gate.ensure_browser_requested()

    def test_confirmation_without_prior_browser_blocked(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, wat is nu?")
        gate.set_user_request("ja")
        with pytest.raises(ToolError, match="browser"):
            gate.ensure_browser_requested()

    def test_non_browser_turn_disarms_browser(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, open google.com")
        gate.set_user_request("Jarvis, what is two plus two")
        with pytest.raises(ToolError, match="browser"):
            gate.ensure_browser_requested()

    def test_no_request_blocked(self):
        gate = ToolGate()
        with pytest.raises(ToolError, match="Jarvis"):
            gate.ensure_browser_requested()

    def test_confirmation_allowed_with_wake_word(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, confirm clicking Buy")
        gate.ensure_confirmation_allowed()

    def test_confirmation_allowed_with_armed_browser(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, open google.com")
        gate.set_user_request("ja")
        gate.ensure_confirmation_allowed()

    def test_confirmation_without_auth_blocked(self):
        gate = ToolGate()
        gate.set_user_request("wat is nu")
        with pytest.raises(ToolError, match="Jarvis"):
            gate.ensure_confirmation_allowed()

    def test_allows_reply_with_wake_word(self):
        gate = ToolGate()
        gate.set_user_request("nothing special here")
        assert gate.allows_reply() is False
        gate.set_user_request("Jarvis, what is the weather")
        assert gate.allows_reply() is True

    def test_allows_reply_for_confirmation_continuation(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, open google.com")
        gate.ensure_wake_word()  # the wake word opens the conversation window
        gate.set_user_request("ja")
        assert gate.allows_reply() is True

    def test_blocks_reply_for_unrelated_talk(self):
        gate = ToolGate()
        gate.set_user_request("the weather looks nice today")
        assert gate.allows_reply() is False

    def test_allows_reply_for_confirmation_in_active_conversation(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, wat is nu?")
        gate.ensure_wake_word()  # the wake word opens the conversation window
        gate.set_user_request("ja")
        assert gate.allows_reply() is True

    def test_blocks_reply_for_empty_turn(self):
        gate = ToolGate()
        assert gate.allows_reply() is False


# ------------------------------------------------------------------ #
# conversation wake behaviour
# ------------------------------------------------------------------ #


class TestIsSleepCommand:
    def test_go_to_sleep(self):
        assert is_sleep_command("go to sleep")

    def test_go_to_sleep_with_wake_word(self):
        assert is_sleep_command("Jarvis, go to sleep")

    def test_stop_listening(self):
        assert is_sleep_command("stop listening")

    def test_thats_all(self):
        assert is_sleep_command("that's all")

    def test_thats_all_thank_you(self):
        assert is_sleep_command("thank you, that's all")

    def test_unrelated_sleep_talk_not_a_command(self):
        assert not is_sleep_command("help me fix my sleep schedule")

    def test_plain_sleep_word_not_a_command(self):
        assert not is_sleep_command("I need to sleep")

    def test_none(self):
        assert not is_sleep_command(None)


class TestConversationBehaviour:
    def test_default_timeout_is_configurable_constant(self):
        gate = ToolGate()
        assert gate.conversation.timeout_seconds == CONVERSATION_TIMEOUT_SECONDS

    def test_inactive_background_speech_ignored(self):
        gate = ToolGate()
        gate.set_user_request("the baby is crying in the other room")
        assert gate.is_conversation_active() is False
        assert gate.allows_reply() is False

    def test_wake_word_activates_conversation(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, what time is it?")
        assert gate.allows_reply() is True  # waking up opens the conversation
        assert gate.is_conversation_active() is True

    def test_wake_word_case_insensitive(self):
        for phrase in (
            "JARVIS, what time is it?",
            "jarvis tell me a joke",
            "Hey Jarvis, good morning",
        ):
            gate = ToolGate()
            gate.set_user_request(phrase)
            assert gate.allows_reply() is True

    def test_follow_up_without_wake_word_accepted_while_active(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, open google.com")
        assert gate.allows_reply() is True
        gate.set_user_request("And the weather in Utrecht?")
        assert gate.allows_reply() is True
        assert gate.is_conversation_active() is True

    def test_timeout_deactivates_conversation(self):
        clock = _FixedClock(0.0)
        gate = ToolGate(conversation=ConversationGate(timeout_s=5.0, now=clock))
        gate.set_user_request("Jarvis, tell me a joke")
        assert gate.allows_reply() is True
        clock.t = 6.0
        gate.set_user_request("Another one")
        assert gate.allows_reply() is False
        assert gate.is_conversation_active() is False

    def test_timeout_expired_requires_wake_word_again(self):
        clock = _FixedClock(0.0)
        gate = ToolGate(conversation=ConversationGate(timeout_s=5.0, now=clock))
        gate.set_user_request("Jarvis, tell me a joke")
        assert gate.allows_reply() is True
        clock.t = 6.0
        gate.set_user_request("tell me another one")
        assert gate.allows_reply() is False
        gate.set_user_request("Jarvis, another one then")
        assert gate.allows_reply() is True

    def test_each_accepted_utterance_refreshes_the_window(self):
        clock = _FixedClock(0.0)
        gate = ToolGate(conversation=ConversationGate(timeout_s=5.0, now=clock))
        gate.set_user_request("Jarvis, start the timer")
        assert gate.allows_reply() is True  # activated, last interaction at t=0
        clock.t = 4.0
        gate.set_user_request("tick")
        assert gate.allows_reply() is True  # accepted, refreshes to t=4
        clock.t = 8.0
        gate.set_user_request("tock")
        assert gate.allows_reply() is True  # accepted, refreshes to t=8
        clock.t = 13.5
        gate.set_user_request("tick tock")
        assert gate.allows_reply() is False  # 5.5s since last refresh: expired

    def test_sleep_command_closes_conversation(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, help me plan my day")
        assert gate.allows_reply() is True
        gate.set_user_request("Jarvis, go to sleep")
        assert gate.allows_reply() is True  # brief goodbye still allowed
        assert gate.is_conversation_active() is False

    def test_after_sleep_wake_word_required_again(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, play some music")
        assert gate.allows_reply() is True
        gate.set_user_request("Jarvis, stop listening")
        assert gate.allows_reply() is True  # brief goodbye, then asleep again
        assert gate.is_conversation_active() is False
        gate.set_user_request("play some music")
        assert gate.allows_reply() is False
        gate.set_user_request("Hey Jarvis, play some music")
        assert gate.allows_reply() is True

    def test_empty_or_whitespace_turn_always_ignored(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, open google.com")
        assert gate.allows_reply() is True
        gate.set_user_request("")
        assert gate.allows_reply() is False
        gate.set_user_request("   ")
        assert gate.allows_reply() is False
        gate.set_user_request(None)
        assert gate.allows_reply() is False

    def test_unrelated_sleep_word_does_not_end_conversation(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, help me plan my day")
        assert gate.allows_reply() is True
        gate.set_user_request("Can we do something about my sleep schedule?")
        assert gate.allows_reply() is True
        assert gate.is_conversation_active() is True
