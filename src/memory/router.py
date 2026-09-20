"""Deterministic memory routing for explicit write/recall intents.

The router is the code-level guarantee that correctness-critical memory
commands never depend on the LLM choosing a tool at random. It:

* classifies a user utterance as WRITE, RECALL, or NONE (high-confidence only),
* extracts the *fact* to store, not the command boilerplate,
* extracts a concise keyword query for recall, stripping auxiliary words,
* supports English and Dutch write/recall phrases,
* still consults :class:`memory.policy.MemoryPolicy` before writing
  (secrets are refused) and dedupes via the provider,
* never raises into the voice loop: provider failures become a polite
  handled result.

The router mirrors the semantics of ``memory.tools.MemoryIntegration`` so the
deterministic path and the optional LLM-driven tools stay consistent.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field

from memory.base import MemoryEntry, MemoryProvider
from memory.manager import MemoryManager
from memory.policy import MAX_MEMORY_RESULTS, MemoryPolicy

_WAKE_WORD_RE = re.compile(r"\bjarvis\b", re.IGNORECASE)

# --- write trigger phrases ------------------------------------------------- #
# Order matters: the longest/richest prefixes are tried first so "remember
# that X" captures X without keeping "that".
_WRITE_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bremember\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"\bsave\s+(?:this|that|it)\s*[:,\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bkeep\s+(?:this|that)\s+in\s+mind\s*[:,\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bkeep\s+in\s+mind\s+that\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bdon[\u2019']t\s+forget\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"\bnote\s+(?:this|that|down)\s*[:,\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bstore\s+(?:this|that|it)\s*[:,\-]?\s*(.+)$", re.IGNORECASE),
    # Dutch
    re.compile(r"\bonthoud(?:en)?\s+(?:dat\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"\bsla\s+dit\s+op\s*[:,\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bbewaar\s+dit\s*[:,\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bvergeet\s+niet\s+(?:dat\s+)?(.+)$", re.IGNORECASE),
)

_WRITE_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"remember\b|"
    r"save\s+(?:this|that|it)\b|"
    r"keep\s+(?:this|that)\s+in\s+mind\b|"
    r"keep\s+in\s+mind\b|"
    r"don[\u2019']t\s+forget\b|"
    r"note\s+(?:this|that|down)\b|"
    r"store\s+(?:this|that|it)\b|"
    r"onthoud(?:en)?\b|"
    r"sla\s+dit\s+op\b|"
    r"bewaar\s+dit\b|"
    r"vergeet\s+niet\b"
    r")\b",
    re.IGNORECASE,
)

# --- recall phrases -------------------------------------------------------- #
_RECALL_PHRASE_RE = re.compile(
    r"\b(?:"
    r"what\s+.{0,40}\b(?:did|have|had)\b.{0,40}\b(?:ask(?:ed)?\s+you\s+to\s+remember|"
    r"remember)\b|"
    r"what\s+do\s+you\s+(?:remember|know\s+about)\b|"
    r"what\s+have\s+i\s+told\s+you\s+about\b|"
    r"do\s+you\s+remember\b|"
    r"did\s+you\s+remember\b|"
    r"search\s+(?:your\s+)?memories?\s+for\b|"
    r"look\s+in\s+(?:your\s+)?memory\s+for\b|"
    r"what\s+is\s+in\s+(?:your\s+)?memory\b|"
    r"tell\s+me\s+(?:what\s+)?(?:you\s+)?(?:remember|recall)\b|"
    r"remind\s+me\b|"
    r"recall\b|"
    r"check\s+(?:your\s+)?memory\b|"
    # Dutch recalls
    r"weet\s+je\s+nog\b|"
    r"wat\s+had\s+ik\s+je\s+gevraagd\s+te\s+onthouden\b|"
    r"wat\s+hebt?(?:\s+je)?\s+onthouden\s+over\b|"
    r"wat\s+heb\s+je\s+onthouden\s+over\b|"
    r"zoek(?:t|en)?\s+in\s+je\s+geheugen\s+naar\b|"
    r"wat\s+weet\s+je\s+nog\s+over\b|"
    r"welk(?:e)?\s+\w+\s+moest\s+(?:je|jij)\s+(?:van\s+)?mij\s+onthouden\b|"
    r"welk(?:e)?\s+\w+\s+moest\s+(?:je|jij)\s+onthouden\b|"
    r"wat\s+moest\s+(?:je|jij)\s+onthouden\b"
    r")\b",
    re.IGNORECASE,
)

# --- recall query extraction ---------------------------------------------- #
# Boilerplate tokens that carry no search value; the router keeps only the
# meaningful keywords (for "what colour did I ask you to remember?" -> colour).
_BOILERPLATE_EN = frozenset(
    {
        "a",
        "about",
        "again",
        "also",
        "and",
        "are",
        "ask",
        "asked",
        "at",
        "be",
        "before",
        "been",
        "but",
        "by",
        "can",
        "check",
        "could",
        "did",
        "do",
        "does",
        "earlier",
        "for",
        "from",
        "had",
        "has",
        "have",
        "her",
        "him",
        "his",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "know",
        "known",
        "me",
        "memories",
        "memory",
        "mention",
        "mentioned",
        "my",
        "need",
        "needed",
        "of",
        "on",
        "or",
        "our",
        "please",
        "recall",
        "remember",
        "remind",
        "said",
        "search",
        "shall",
        "she",
        "should",
        "sir",
        "so",
        "some",
        "stuff",
        "that",
        "the",
        "their",
        "things",
        "this",
        "to",
        "told",
        "us",
        "want",
        "wanted",
        "was",
        "we",
        "well",
        "were",
        "what",
        "which",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_NL_MAPPINGS: dict[str, str] = {
    "kleur": "colour",
    "naam": "name",
    "verjaardag": "birthday",
    "afspraak": "appointment",
    "project": "project",
    "voorkeur": "preference",
    "bedrijf": "company",
    "werk": "work",
    "wachtwoord": "password",
    "code": "code",
    "datum": "date",
    "tijd": "time",
    "vergadering": "meeting",
    "adres": "address",
    "nummer": "number",
    "taal": "language",
    "vriend": "friend",
    "vriendin": "friend",
    "huisdier": "pet",
}


class MemoryIntent(enum.IntEnum):
    """Deterministic routing outcome for one utterance."""

    NONE = 0
    WRITE = 1
    RECALL = 2


@dataclass(frozen=True)
class RouterResult:
    """What the router decided and did, without echoing secrets."""

    intent: MemoryIntent
    handled: bool = False
    accepted: bool = False
    language: str = ""
    content: str = ""
    count: int = 0
    entry_id: str = ""
    reason: str = ""
    message: str = ""
    results: tuple[MemoryEntry, ...] = field(default_factory=tuple)

    @property
    def is_write(self) -> bool:
        return self.intent is MemoryIntent.WRITE

    @property
    def is_recall(self) -> bool:
        return self.intent is MemoryIntent.RECALL


def _strip_wake_word(text: str) -> str:
    return _WAKE_WORD_RE.sub(" ", text or "")


def detect_intent(utterance: str | None) -> MemoryIntent:
    """High-confidence intent detection; normal conversation stays NONE."""
    if not utterance or not utterance.strip():
        return MemoryIntent.NONE
    cleaned = _strip_wake_word(utterance)
    if _RECALL_PHRASE_RE.search(cleaned):
        return MemoryIntent.RECALL
    if _WRITE_TRIGGER_RE.search(cleaned):
        return MemoryIntent.WRITE
    return MemoryIntent.NONE


def extract_write_content(utterance: str | None) -> str:
    """The fact to store, not the command (never includes "Jarvis" or verbs)."""
    if not utterance:
        return ""
    cleaned = _strip_wake_word(utterance).strip()
    for pattern in _WRITE_PREFIXES:
        match = pattern.search(cleaned)
        if not match:
            continue
        content = match.group(1).strip(" \t\r\n:,.-")
        content = _WAKE_WORD_RE.sub(" ", content).strip()
        content = re.sub(r"\s+", " ", content)
        if content:
            return content
    return ""


_BOILERPLATE_NL = frozenset(
    {
        "aan",
        "als",
        "bij",
        "dat",
        "de",
        "dit",
        "een",
        "en",
        "gaat",
        "gehad",
        "geheugen",
        "gevraagd",
        "had",
        "heb",
        "hebt",
        "heeft",
        "het",
        "hoe",
        "ik",
        "in",
        "is",
        "je",
        "jij",
        "jouw",
        "juist",
        "jullie",
        "kunt",
        "maar",
        "met",
        "mij",
        "mijn",
        "moest",
        "moet",
        "naar",
        "niet",
        "nog",
        "of",
        "onthoud",
        "onthouden",
        "op",
        "over",
        "sla",
        "te",
        "u",
        "van",
        "voor",
        "wat",
        "we",
        "weet",
        "wel",
        "welke",
        "welk",
        "zijn",
        "zoek",
        "zoeken",
        "zoekt",
    }
)


def extract_recall_query(utterance: str | None) -> str:
    """A concise keyword query, stripping boilerplate and "Jarvis"."""
    if not utterance:
        return ""
    cleaned = _strip_wake_word(utterance).strip()
    cleaned = re.sub(r"^[,;:\-.\s]+", "", cleaned)
    cleaned = cleaned.strip(" \t?.,!;:")
    keywords: list[str] = []
    for word in re.split(r"[\s,;]+", cleaned):
        token = re.sub(r"[^\w\u00c0-\uffff'-]", "", word)
        token = token.strip("'\"-_")
        if not token:
            continue
        low = token.casefold()
        if low in _BOILERPLATE_EN or low in _BOILERPLATE_NL:
            continue
        if low in _NL_MAPPINGS and _NL_MAPPINGS[low] not in keywords:
            keywords.append(_NL_MAPPINGS[low])
        if low not in keywords:
            keywords.append(low)
    if not keywords:
        return ""
    return " ".join(dict.fromkeys(keywords))


class MemoryRouter:
    """Deterministic write/recall routing against a single memory backend."""

    def __init__(
        self,
        *,
        provider: MemoryProvider | None,
        policy: MemoryPolicy,
        manager: MemoryManager | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.manager = manager or MemoryManager(provider, policy)
        self.last_error: str = ""

    # ------------------------------------------------------------------ #
    # synchronous classification helpers (unit-testable, no backend)
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_intent(utterance: str | None) -> MemoryIntent:
        return detect_intent(utterance)

    @staticmethod
    def extract_write_content(utterance: str | None) -> str:
        return extract_write_content(utterance)

    @staticmethod
    def extract_recall_query(utterance: str | None) -> str:
        return extract_recall_query(utterance)

    # ------------------------------------------------------------------ #
    # deterministic routing
    # ------------------------------------------------------------------ #

    async def route(self, utterance: str | None) -> RouterResult:
        """Classify and, if it is an explicit memory command, act on it.

        The write/read happens HERE in code -- no LLM tool choice involved.
        Provider failures never raise: they become a handled, polite result.
        """
        intent = self.detect_intent(utterance)
        if intent is MemoryIntent.NONE:
            return RouterResult(intent=MemoryIntent.NONE)
        if intent is MemoryIntent.WRITE:
            return await self._route_write(utterance or "")
        return await self._route_recall(utterance or "")

    async def _route_write(self, utterance: str) -> RouterResult:
        content = self.extract_write_content(utterance)
        if not content:
            return RouterResult(
                intent=MemoryIntent.WRITE,
                handled=True,
                reason="no storeable content found",
                message="I did not catch anything to remember.",
            )
        decision = self.policy.should_store(content)
        if not decision.accepted:
            return RouterResult(
                intent=MemoryIntent.WRITE,
                handled=True,
                accepted=False,
                reason=decision.reason,
                message="I can't store that.",
            )
        if self.provider is None or not self.provider.is_available():
            self.last_error = "memory store unavailable"
            return RouterResult(
                intent=MemoryIntent.WRITE,
                handled=True,
                accepted=False,
                reason=self.last_error,
                message="Memory is not available right now. Continue without it.",
            )
        try:
            entry = MemoryEntry(content=content, category="general")
            existing = await self.provider.find_exact(entry.content)
            if existing is not None:
                updated = await self.provider.update(
                    existing.id, content=entry.content, category="general"
                )
                entry_id = updated.id if updated is not None else existing.id
                return RouterResult(
                    intent=MemoryIntent.WRITE,
                    handled=True,
                    accepted=True,
                    language=_language_hint(utterance),
                    content=content,
                    entry_id=entry_id,
                    message="Saved to long-term memory.",
                )
            stored = await self.provider.store(entry)
            return RouterResult(
                intent=MemoryIntent.WRITE,
                handled=True,
                accepted=True,
                language=_language_hint(utterance),
                content=content,
                entry_id=stored.id,
                message="Saved to long-term memory.",
            )
        except Exception as exc:
            self.last_error = str(exc)
            return RouterResult(
                intent=MemoryIntent.WRITE,
                handled=True,
                accepted=False,
                reason="provider failure",
                message="Memory is not available right now. Continue without it.",
            )

    async def _route_recall(self, utterance: str) -> RouterResult:
        query = self.extract_recall_query(utterance)
        if self.provider is None or not self.provider.is_available():
            self.last_error = "memory store unavailable"
            return RouterResult(
                intent=MemoryIntent.RECALL,
                handled=True,
                reason=self.last_error,
                message="Memory is not available right now. Continue without it.",
            )
        try:
            terms = query.split() if query else []
            results = (
                await self.provider.search(query, limit=MAX_MEMORY_RESULTS)
                if terms
                else await self.provider.list_all(limit=MAX_MEMORY_RESULTS)
            )
        except Exception as exc:
            self.last_error = str(exc)
            return RouterResult(
                intent=MemoryIntent.RECALL,
                handled=True,
                reason="provider failure",
                message="Memory is not available right now. Continue without it.",
            )
        return RouterResult(
            intent=MemoryIntent.RECALL,
            handled=True,
            accepted=bool(results),
            language=_language_hint(utterance),
            content=query,
            count=len(results),
            results=tuple(results),
            message=(
                f"Recall found {len(results)} stored item(s)."
                if results
                else "No stored memory matched; answer truthfully that nothing is stored."
            ),
        )


def _language_hint(utterance: str) -> str:
    norm = utterance.casefold()
    if any(
        w in norm
        for w in (
            "onthoud",
            "onthouden",
            "sla dit op",
            "bewaar dit",
            "vergeet niet",
            "weet je nog",
            "geheugen",
            "kleur",
            "moest",
            "gevraagd",
        )
    ):
        return "nl"
    return "en"


__all__ = [
    "MemoryIntent",
    "MemoryRouter",
    "RouterResult",
    "detect_intent",
    "extract_recall_query",
    "extract_write_content",
]
