"""Deterministic Hermes delegation.

The router answers one question: given the current user text, should the work
stay with JARVIS or be delegated to the Hermes backend? It is deliberately a
pure, regex-based decision with a reason, so it is trivial to inspect and
debug. Simple, conversational, browser, Windows, Spotify, and factual requests
always stay with JARVIS; only explicit "ask Hermes to ..." phrases and clear
deep planning / analysis / development intents delegate to Hermes.
"""

from __future__ import annotations

import re

from gates import is_hermes_deep, is_hermes_explicit, normalize_text

_DEVELOP_FOCUS_RE = re.compile(
    r"\b(?:build|implement|develop|fix|code|integration|refactor|create)\b",
    re.IGNORECASE,
)
_PLAN_FOCUS_RE = re.compile(
    r"\b(?:plan|design|architecture|roadmap|blueprint|strategy|approach)\b",
    re.IGNORECASE,
)
_ANALYSE_FOCUS_RE = re.compile(
    r"\b(?:analyse|analyze|investigate|diagnos|debug|root\s+cause|why)\b",
    re.IGNORECASE,
)


class Route:
    """One inspectable routing decision."""

    __slots__ = ("focus", "reason", "target")

    def __init__(self, target: str, reason: str, focus: str) -> None:
        self.target = target  # "jarvis" or "hermes"
        self.reason = reason  # "explicit", "deep", or "simple"
        self.focus = focus  # one of ask / plan / analyse / develop

    def describe(self) -> str:
        return f"Route: {self.target} ({self.reason} -> {self.focus})"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return self.describe()


class HermesRouter:
    """Decides whether an utterance delegates to Hermes, without an LLM."""

    def route(self, text: str | None) -> Route:
        norm = normalize_text(text)
        if not norm:
            return Route("jarvis", "simple", "ask")
        if is_hermes_explicit(norm):
            return Route("hermes", "explicit", self.focus(norm))
        if is_hermes_deep(norm):
            return Route("hermes", "deep", self.focus(norm))
        return Route("jarvis", "simple", "ask")

    @staticmethod
    def focus(text: str | None) -> str:
        norm = normalize_text(text)
        if _DEVELOP_FOCUS_RE.search(norm):
            return "develop"
        if _PLAN_FOCUS_RE.search(norm):
            return "plan"
        if _ANALYSE_FOCUS_RE.search(norm):
            return "analyse"
        return "ask"
