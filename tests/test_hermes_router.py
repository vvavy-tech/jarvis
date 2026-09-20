"""Tests for the deterministic Hermes delegation router.

The router must be trivially inspectable: given the current user text it
answers with a fixed target (JARVIS or Hermes), the reason (explicit, deep, or
simple), and the Hermes focus it would use. Simple / conversational / browser /
Windows / Spotify requests always stay with JARVIS.
"""

import pytest

from agents.router import HermesRouter

router = HermesRouter()


class TestRoutingTarget:
    @pytest.mark.parametrize(
        "text",
        [
            "Jarvis, ask Hermes to design the parking sensor module",
            "Jarvis, use Hermes for this",
            "Can you ask the hermes agent to investigate the audio dropout?",
            "Jarvis, consult Hermes about the architecture first",
            "Jarvis, developer mode: use Hermes to build this feature",
            "Call Hermes to plan the rollout order",
        ],
    )
    def test_explicit_phrases_delegate_to_hermes(self, text):
        route = router.route(text)
        assert route.target == "hermes"
        assert route.reason == "explicit"

    @pytest.mark.parametrize(
        "text",
        [
            "Jarvis, why is my Meta ads spend so high? Analyse it properly.",
            "Jarvis, investigate the root cause of the camera lag",
            "Jarvis, can you design the architecture for the new endpoint?",
            "Jarvis, deep dive on why the browser times out",
        ],
    )
    def test_deep_intents_delegate_to_hermes(self, text):
        route = router.route(text)
        assert route.target == "hermes"
        assert route.reason == "deep"

    @pytest.mark.parametrize(
        "text",
        [
            "Jarvis, what's the weather in Amsterdam?",
            "Jarvis, play something by the Beatles",
            "Jarvis, open github.com",
            "Jarvis, tell me a joke",
            None,
            "",
        ],
    )
    def test_simple_requests_stay_with_jarvis(self, text):
        route = router.route(text)
        assert route.target == "jarvis"
        assert route.reason == "simple"

    def test_route_is_inspectable(self):
        route = router.route("Jarvis, ask Hermes to plan the migration")
        assert route.describe() == "Route: hermes (explicit -> plan)"
        route = router.route("Jarvis, good morning")
        assert route.describe() == "Route: jarvis (simple -> ask)"


class TestFocusDetection:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Jarvis, ask Hermes to build the new integration", "develop"),
            ("Jarvis, use Hermes to fix this bug", "develop"),
            ("Jarvis, ask Hermes to implement the feature properly", "develop"),
            ("Jarvis, ask Hermes to plan the migration", "plan"),
            ("Jarvis, use Hermes to design the architecture", "plan"),
            ("Jarvis, ask Hermes to draw up a roadmap", "plan"),
            ("Jarvis, ask Hermes for ideas about the design", "plan"),
            ("Jarvis, ask Hermes why the ads stopped delivering", "analyse"),
            ("Jarvis, use Hermes to debug the connection drop", "analyse"),
            ("Jarvis, investigate the root cause with Hermes", "analyse"),
            ("Jarvis, ask Hermes what you recommend", "ask"),
        ],
    )
    def test_focus_mapping(self, text, expected):
        route = router.route(text)
        assert route.target == "hermes"
        assert route.focus == expected
