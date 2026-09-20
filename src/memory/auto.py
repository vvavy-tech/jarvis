"""Safe automatic memory for JARVIS (OPT-IN, isolated secondary layer).

As an alternative to running LLM memory, the default JARVIS voice path keeps
working if this layer is enabled: auto-memory only observes *final, accepted*
transcripts and runs as a tracked background task. The realtime reply path
never awaits it, never calls ``generate_reply`` because of it, and never sees
its exceptions (they are caught and logged as a concise diagnostic).

Architecture
------------
* Stage 1 -- cheap deterministic filter (:func:`stage1_skip_reason`): skips
  questions, commands, greetings, fillers, jokes, calculations, navigation,
  tool commands, short fragments and obvious ASR garbage.
* Recall -- natural recall detection (:func:`auto_recall_query`) recognises
  questions like ``what is my favourite colour?`` so bounded relevant
  retrieval still works without saying "search your memory".
* Stage 2 -- conservative classifier/extractor (:func:`classify_candidate`):
  a declarative statement only persists when it carries a durable signal
  (preferences, decisions, supplier/provider choices, technical choices,
  project constraints). UNCERTAIN defaults to SKIP; nothing is inferred.
* Policy -- the existing :class:`memory.policy.MemoryPolicy` secret filter is
  applied to both the raw utterance and the extracted fact. Secrets are never
  stored and never logged.
* Persistence -- duplicate facts update the existing row; a changed preference
  (e.g. "My favourite colour is blue." -> "green now.") supersedes the older
  row instead of piling up duplicates.

The explicit "remember"/"onthoud" path is untouched: explicit write intents
are skipped here so they keep flowing to the deterministic router / tools.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from memory.base import MemoryEntry, MemoryProvider
from memory.policy import MAX_MEMORY_RESULTS, MemoryPolicy
from memory.router import (
    MemoryIntent,
    _strip_wake_word,
    detect_intent,
    extract_recall_query,
)

logger = logging.getLogger("agent")

ACTION_STORE = "store"
ACTION_SKIP = "skip"
ACTION_RECALL = "recall"

MIN_CANDIDATE_WORDS = 4

# --------------------------------------------------------------------------- #
# deterministic filters (stage 1) and surface cleaning
# --------------------------------------------------------------------------- #

_WAKE_WORD_RE = re.compile(r"\bjarvis\b", re.IGNORECASE)

_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:yeah\s+no|yeah|yea|yep|yup|so|anyway|ok|okay|oke|alright|"
    r"alrighty|well|oh|hmm|uhm|um|right|sure|man|sir|hey|hoi|sorry|"
    r"by\s+the\s+way|btw)\b[\s,;:]*",
    re.IGNORECASE,
)

_TRAILING_FILLER_RE = re.compile(
    r"\b(?:thanks|thank\s+you|thank\s*u|please|dank\s*(?:je|jewel|u)|bedankt|"
    r"that'?s\s+(?:it|all)|alright|okay|oke|is\s+all)\s*[.,!]?$",
    re.IGNORECASE,
)

_BTW_ANYWHERE_RE = re.compile(r"\bby\s+the\s+way\b", re.IGNORECASE)

# Hedges/filler adverbs removed from candidate facts; they add no durable
# meaning ("actually", "definitely", "just" ...).
_HEDGE_WORDS = frozenset(
    {
        "absolutely",
        "actually",
        "basically",
        "definitely",
        "honestly",
        "just",
        "literally",
        "maybe",
        "obviously",
        "probably",
        "quite",
        "really",
        "simply",
    }
)

_QUESTION_TERMINAL_RE = re.compile(r"[?？]+$")  # noqa: RUF001
_QUESTION_LEAD_RE = re.compile(
    r"^(?:what|which|when|where|who|whom|whose|why|how|is|are|was|were|can|"
    r"could|do|does|did|should|would|will|shall|have|has|had|wat|welke|welk|"
    r"hoe|waarom|waar|wanneer|wie|kan|kun|moest)\b",
    re.IGNORECASE,
)

_LEADING_COMMAND_RE = re.compile(
    r"^(?:open|play|pause|resume|stop|close|quit|exit|turn|show|send|delete|"
    r"remove|call|search|google|look|go|scroll|click|type|press|read|set|"
    r"enable|disable|launch|run|book|order|cancel|mute|unmute|put|switch|"
    r"change|upload|download|email|text|message|navigate|browse|refresh|"
    r"reload|restart|wake|sleep|activate|deactivate|connect|disconnect|sign|"
    r"login|logout|follow|like|share|post|tweet|print|screenshot|create|"
    r"give|shut|drop)\b",
    re.IGNORECASE,
)

_TOOL_COMMAND_RE = re.compile(
    r"\b(?:open|play|pause|resume|stop|close|launch|call|ask)\s+(?:spotify|"
    r"netflix|youtube|music|the\s+browser|browser|hermes|chrome|firefox|edge|"
    r"playlist|album|track|song|video)\b",
    re.IGNORECASE,
)

_NAVIGATION_RE = re.compile(
    r"\b(?:go\s+back|go\s+forward|go\s+to|go\s+up|go\s+down|scroll\s+(?:up|down|"
    r"left|right)|next|previous|refresh|reload|home|menu|close\s+(?:tab|window|"
    r"page)|open\s+(?:tab|window|new\s+tab)|ga\s+terug|terug|back)\b",
    re.IGNORECASE,
)

_CALCULATION_RE = re.compile(
    r"\b(?:plus|minus|times|multiplied\s+by|divided\s+by|equals?|how\s+much\s+is|"
    r"calculate|sum\s+of|wat\s+is)\b|\d+\s*[+\-*/x×÷=%~]\s*\d+",  # noqa: RUF001
    re.IGNORECASE,
)

_GREETING_RE = re.compile(
    r"^(?:hi|hello|heya|hey|hiya|yo|howdy|good\s+(?:morning|afternoon|evening|"
    r"night)|morning|afternoon|evening|hola|hallo|goedemorgen|goedenavond|"
    r"goedendag|dag)\b",
    re.IGNORECASE,
)

_JOKE_RE = re.compile(
    r"\b(?:jok(?:e|es)|make\s+me\s+laugh|something\s+funny|something\s+humorous|"
    r"say\s+something\s+funny)\b",
    re.IGNORECASE,
)

_SLEEP_RE = re.compile(
    r"\b(?:go\s+to\s+sleep|stop\s+listening|that['\u2019']?s\s+all|"
    r"go\s+away|leave\s+it|forget\s+it|never\s+mind)\b",
    re.IGNORECASE,
)

_ASR_GARBAGE_RE = re.compile(
    r"^(?:h+a+|h+u+|u+h+|u+m+|a+h+|e+h+|h+m+|m+o+h+|lol+|bruh|wat\??|what\??|"
    r"hmm+|mmm+|dacht|ja|nee)\s*$",
    re.IGNORECASE,
)

_NO_VOWEL_RE = re.compile(r"^[^aeiouy]+$")

_ACK_FRAGMENTS: frozenset[str] = frozenset(
    {
        "ok",
        "okay",
        "oke",
        "yes",
        "yeah",
        "yep",
        "yup",
        "no",
        "nope",
        "thanks",
        "thank you",
        "thank u",
        "thx",
        "dank je",
        "dankjewel",
        "bedankt",
        "no problem",
        "no worries",
        "never mind",
        "whatever",
        "nothing",
        "good",
        "fine",
        "nice",
        "cool",
        "great",
        "awesome",
        "perfect",
        "alright",
        "sounds good",
        "bye",
        "goodbye",
        "see you",
        "see ya",
        "later",
        "haha",
        "ha ha",
        "hahaha",
        "hehe",
        "lol",
        "mm",
        "mhm",
        "ah ok",
        "ok thanks",
        "thanks ok",
        "very good",
        "that works",
        "sure thing",
        "you too",
        "love it",
        "love you",
        "gotta go",
    }
)


def _is_gibberish(word: str) -> bool:
    return len(word) >= 4 and bool(_NO_VOWEL_RE.match(word))


def stage1_skip_reason(text: str | None) -> str | None:
    """Cheap deterministic: an obvious non-memory utterance -> a reason.

    Returns ``None`` when the utterance plausibly is (or could be) a durable
    declarative statement.
    """
    raw = text or ""
    norm = " ".join(raw.casefold().split())
    if not norm:
        return "empty"
    key = norm.strip(" .,!?;:")
    if key in _ACK_FRAGMENTS or key.rstrip(".!") in _ACK_FRAGMENTS:
        return "ack"
    if _ASR_GARBAGE_RE.match(key):
        return "asr-garbage"
    words = norm.split()
    if len(norm) < 5 or len(words) < 2:
        return "too-short"
    if _GREETING_RE.match(norm):
        return "greeting"
    if _JOKE_RE.search(norm):
        return "joke"
    if _SLEEP_RE.search(norm):
        return "idle-command"
    if _NAVIGATION_RE.search(norm):
        return "navigation"
    if _CALCULATION_RE.search(norm):
        return "calculation"
    if _LEADING_COMMAND_RE.match(norm):
        return "command"
    if _TOOL_COMMAND_RE.search(norm):
        return "tool-command"
    if _QUESTION_TERMINAL_RE.search(norm) or _QUESTION_LEAD_RE.match(norm):
        return "question"
    if len(words) == 1 and _is_gibberish(words[0]):
        return "asr-garbage"
    return None


def clean_candidate(raw: str | None) -> str:
    """Strip chords/filler and normalise a candidate into a clean fact."""
    text = _WAKE_WORD_RE.sub(" ", raw or "")
    text = _BTW_ANYWHERE_RE.sub(" ", text)
    text = _LEADING_FILLER_RE.sub(" ", text)
    text = text.strip(" \t\r\n,.;:!?\u2014-")
    text = _TRAILING_FILLER_RE.sub("", text).strip(" \t\r\n,.;:!?\u2014-")
    words = [
        word
        for word in text.split()
        if word.strip(".,;:!?\u2014'").casefold() not in _HEDGE_WORDS
    ]
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:!?\u2014")
    if text and text[0].islower() and len(text) > 1:
        text = text[0].upper() + text[1:]
    return text


# --------------------------------------------------------------------------- #
# stage 2: conservative durable-memory classifier / extractor
# --------------------------------------------------------------------------- #

_PREFIXES_TO_STRIP = ("a ", "an ", "the ", "my ", "our ", "i ", "we ")

_MY_IS_TEMPLATE_RE = re.compile(
    r"\b(?:my|mijn)\s+[a-zA-Z][a-zA-Z' -]{1,40}?\s+(?:is|are|was|were|remains|"
    r"zijn|waren)\b",
    re.IGNORECASE,
)

_PREFERENCE_TEMPLATE_RE = re.compile(
    r"\b(?:my|mijn)\s+(?:favourite|favorite|favoriet|favoriete|preferred|"
    r"voorkeur|normal|usual|go[- ]to|ideal|standard|default)\b",
    re.IGNORECASE,
)

_PREFER_RE = re.compile(r"\b(?:prefer\w*|voorkeur\w*)\b", re.IGNORECASE)

_DECISION_RE = re.compile(
    r"\b(?:decid(?:e|ed)|decision|agree(?:d)?|cho(?:se|sen|ose)|settled\s+on|"
    r"went\s+with|landed\s+on|opted\s+for|rolled?\s+out|launch(?:es|ed)?|"
    r"roll\s*[- ]out|drop(?:s)?|roadmap|strategy)\b",
    re.IGNORECASE,
)

_SUPPLIER_RE = re.compile(r"\b(?:supplier|provider|vendor)\b", re.IGNORECASE)

_TECHNICAL_RE = re.compile(
    r"\b(?:sqlite|postgres|mysql|database|db\b|schema|backend|frontend|"
    r"architecture|api\b|library|framework|server|hosting|deployment|migration|"
    r"config|stack|repo(?:sitory)?|branch|integration|plugin|tool|"
    r"implementation|codebase)\b",
    re.IGNORECASE,
)

_CONSTRAINT_RE = re.compile(
    r"\b(?:only\s+when|only\s+if|from\s+now\s+on|every\s+time|"
    r"whenever\s+(?:i|we)|always|never)\b",
    re.IGNORECASE,
)

_PROJECT_NOUNS = re.compile(
    r"\b(?:issue|project|drop|launch|brand|store|collection|line|range|"
    r"deployment|build|release|rollout|campaign|client|segment|market)\b",
    re.IGNORECASE,
)

_PROJECT_TEMPLATE_RE = re.compile(
    r"\bfor\s+([a-zA-Z0-9][a-zA-Z0-9' ._-]{1,40}?)[\s\S]{0,80}?\b(?:want|prefer|"
    r"need|like|decided|choose|chose|dropping|make)\b",
    re.IGNORECASE,
)

_INCIDENTAL_ACTION_RE = re.compile(
    r"\b(?:i|we)\s+(?:'d\s+(?:like|love|prefer)|would\s+(?:like|love|prefer)|"
    r"(?:really|definitely|just)\s+)?(?:want|wanted|like|love|need)\s+to\s+"
    r"(?:open|play|check|order|buy|purchase|call|search|google|go|watch|read|"
    r"listen|view|take|see|visit|install|download|upload|switch|change|turn|"
    r"set|ask|email|text|book|happen|set\s+up)\b",
    re.IGNORECASE,
)

_SUBJECT_RE = re.compile(
    r"\b(?:i|i['\u2019]?m|i['\u2019]?ve|my|we|our|ours|let['\u2019]?s|us|ik|mijn|"
    r"onze|ons)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClassifierOutcome:
    """Stage-2 decision for a cleaned declarative candidate."""

    store: bool
    category: str = ""
    fact: str = ""
    reason: str = ""


def _project_signal(text: str) -> bool:
    match = _PROJECT_TEMPLATE_RE.search(text)
    if not match:
        return False
    project = match.group(1).strip().rstrip(".,;:!?")
    if not project:
        return False
    if _PROJECT_NOUNS.search(project):
        return True
    return project[0].isupper()


def classify_candidate(cleaned: str | None) -> ClassifierOutcome:
    """Conservative: STORE only clearly durable declarative statements."""
    text = (cleaned or "").strip()
    if not text:
        return ClassifierOutcome(False, reason="empty")
    norm = " ".join(text.casefold().split())
    words = norm.split()
    if len(words) < MIN_CANDIDATE_WORDS:
        return ClassifierOutcome(False, reason="too-short")
    if _INCIDENTAL_ACTION_RE.search(norm):
        return ClassifierOutcome(False, reason="incidental-action")
    subject_hit = bool(_SUBJECT_RE.search(text))
    project_hit = _project_signal(text)
    if project_hit:
        subject_hit = True
    if not subject_hit:
        return ClassifierOutcome(False, reason="no-user-context")

    signals: list[str] = []
    if _SUPPLIER_RE.search(norm):
        signals.append("business")
    if project_hit:
        signals.append("project")
    if _TECHNICAL_RE.search(norm):
        signals.append("technical")
    if _DECISION_RE.search(norm):
        signals.append("decision")
        if _PROJECT_NOUNS.search(norm):
            signals.append("project")
    if _CONSTRAINT_RE.search(norm):
        signals.append("workflow")
    if (
        _MY_IS_TEMPLATE_RE.search(norm)
        or _PREFERENCE_TEMPLATE_RE.search(norm)
        or _PREFER_RE.search(norm)
    ):
        signals.append("preference")
    if not signals:
        return ClassifierOutcome(False, reason="no-durable-signal")

    _category_order = (
        "business",
        "project",
        "technical",
        "decision",
        "workflow",
        "preference",
    )
    category = "general"
    for candidate in _category_order:
        if candidate in signals:
            category = candidate
            break
    return ClassifierOutcome(True, category=category, fact=text, reason="stored")


# --------------------------------------------------------------------------- #
# natural recall detection (bounded, only when relevant context is stored)
# --------------------------------------------------------------------------- #

_RECALL_NEEDLES = (
    "favourite",
    "favorite",
    "preference",
    "prefer",
    "preferred",
    "supplier",
    "provider",
    "vendor",
    "decide",
    "decided",
    "decision",
    "choose",
    "chose",
    "chosen",
    "choice",
    "colour",
    "color",
    "fit",
    "size",
    "drop",
    "icon",
    "project",
    "sqlite",
    "plan",
    "kleur",
    "voorkeur",
    "naam",
)

_STRONG_RECALL_TERMS = frozenset(
    {
        "name",
        "supplier",
        "favourite",
        "favorite",
        "colour",
        "color",
        "preference",
        "decide",
        "decided",
        "choice",
        "drop",
    }
)


def _is_interrogative(norm: str) -> bool:
    return bool(_QUESTION_LEAD_RE.match(norm) or _QUESTION_TERMINAL_RE.search(norm))


def auto_recall_query(text: str | None) -> str:
    """A keyword query when the utterance is a natural (not explicit) recall.

    Explicit memory phrases are owned by the deterministic router, so they are
    deliberately excluded here. Returns "" when the question does not appear to
    concern stored user/project context (no storage, no reply steering).
    """
    if not text or not text.strip():
        return ""
    if detect_intent(text) is not MemoryIntent.NONE:
        return ""
    norm = " ".join(_strip_wake_word(text).casefold().split())
    if not norm:
        return ""
    if not _is_interrogative(norm):
        return ""
    has_my = bool(re.search(r"\b(?:my|mijn)\b", norm))
    has_needle = any(
        re.search(rf"\b{re.escape(tok)}\b", norm) for tok in _RECALL_NEEDLES
    )
    if not has_my and not has_needle:
        return ""
    terms = extract_recall_query(text).split()
    if not terms:
        return ""
    if len(terms) < 2 and terms[0] not in _STRONG_RECALL_TERMS:
        return ""
    return " ".join(dict.fromkeys(terms))


# --------------------------------------------------------------------------- #
# deduplication / supersede (topic-key based, conservative)
# --------------------------------------------------------------------------- #


def _topic_key(content: str, category: str) -> str:
    norm = " ".join((content or "").casefold().split())
    match = re.search(
        r"\b(?:my|mijn)\s+(?:favourite|favorite|favoriete?|preferred|voorkeur)\s+"
        r"([a-z][a-z' -]{0,24}?)\s+(?:is|was|remains|should\s+be|zijn|waren)\b",
        norm,
    )
    if match:
        return f"{category}:favourite:{match.group(1).strip()}"
    match = re.search(
        r"\b(?:my|mijn)\s+([a-z][a-z' -]{0,20}?)\s+(?:is|are|was|were|remains|"
        r"zijn|waren)\b",
        norm,
    )
    if match:
        return f"{category}:my:{match.group(1).strip()}"
    match = re.search(
        r"\bi\s+(?:'d\s+|would\s+|really\s+|always\s+|definitely\s+)+?prefer\s+"
        r"(?:[a-z'-]+\s+)?([a-z'-]+)\b",
        norm,
    )
    if match:
        return f"{category}:prefer:{match.group(1)}"
    match = re.search(
        r"\bfor\s+([a-z0-9][a-z0-9' ._-]{1,40}?)[\s,].{0,80}?\bwant\s+"
        r"(?:the\s+|a\s+|an\s+)?(?:[a-z'-]+\s+){0,2}?([a-z'-]+)\b",
        norm,
    )
    if match:
        return f"{category}:project:{match.group(1).strip()}:{match.group(2)}"
    return ""


def _key_to_query(topic_key: str) -> str:
    return " ".join(part for part in topic_key.split(":")[1:] if part)


# --------------------------------------------------------------------------- #
# result + pipeline + controller
# --------------------------------------------------------------------------- #


@dataclass
class AutoMemoryResult:
    """Safe outcome of one auto-memory evaluation (never contains secrets)."""

    action: str = ACTION_SKIP
    accepted: bool = False
    stored: bool = False
    category: str = ""
    fact: str = ""
    reason: str = ""
    error: str = ""
    query: str = ""
    count: int = 0
    results: tuple[MemoryEntry, ...] = field(default_factory=tuple)

    @property
    def should_store(self) -> bool:
        return self.action == ACTION_STORE and self.accepted

    @property
    def is_recall(self) -> bool:
        return self.action == ACTION_RECALL


class AutoMemoryPipeline:
    """Two-stage candidate filter + extractor + dedup persistence.

    Every public entry point (:meth:`detect`) is failure-isolated: provider
    errors become a safe ``SKIP`` result with a concise diagnostic, never an
    exception and never a reply.
    """

    def __init__(
        self,
        *,
        provider: MemoryProvider | None,
        policy: MemoryPolicy,
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.last_error: str = ""

    # ------------------------------------------------------------------ #
    # sync classification helpers (unit-testable, no backend)
    # ------------------------------------------------------------------ #

    @staticmethod
    def skip_reason(text: str | None) -> str | None:
        return stage1_skip_reason(text)

    @staticmethod
    def classify(text: str | None) -> ClassifierOutcome:
        return classify_candidate(clean_candidate(text))

    @staticmethod
    def recall_query(text: str | None) -> str:
        return auto_recall_query(text)

    # ------------------------------------------------------------------ #
    # full pipeline
    # ------------------------------------------------------------------ #

    async def detect(
        self, text: str | None, *, accepted: bool = True
    ) -> AutoMemoryResult:
        try:
            if not accepted:
                return AutoMemoryResult(action=ACTION_SKIP, reason="gate-asleep")
            raw = (text or "").strip()
            if not raw:
                return AutoMemoryResult(action=ACTION_SKIP, reason="empty")
            # Explicit memory commands belong to the explicit path, never here.
            if detect_intent(raw) is MemoryIntent.WRITE:
                return AutoMemoryResult(action=ACTION_SKIP, reason="explicit-write")
            # Secret filter first: obvious credentials are never persisted and
            # never logged, regardless of where the classifier lands.
            if not self.policy.should_store(raw).accepted:
                return AutoMemoryResult(action=ACTION_SKIP, reason="secret-refused")
            # Natural recall -> bounded relevant retrieval only.
            query = auto_recall_query(raw)
            if query:
                results = await self._bounded_search(query)
                if not results:
                    return AutoMemoryResult(
                        action=ACTION_SKIP, query=query, reason="recall-no-match"
                    )
                return AutoMemoryResult(
                    action=ACTION_RECALL,
                    query=query,
                    count=len(results),
                    results=tuple(results),
                    reason="recall",
                )
            # Stage 1: cheap deterministic filter.
            reason = stage1_skip_reason(raw)
            if reason:
                return AutoMemoryResult(action=ACTION_SKIP, reason=reason)
            # Clean extraction.
            cleaned = clean_candidate(raw)
            if len(cleaned.split()) < MIN_CANDIDATE_WORDS:
                return AutoMemoryResult(action=ACTION_SKIP, reason="too-short")
            logger.info("AUTO MEMORY: candidate")
            # Stage 2: durable classifier / extractor.
            outcome = classify_candidate(cleaned)
            if not outcome.store:
                return AutoMemoryResult(action=ACTION_SKIP, reason=outcome.reason)
            # Secret filter on the extracted fact as well.
            if not self.policy.should_store(outcome.fact).accepted:
                return AutoMemoryResult(action=ACTION_SKIP, reason="secret-refused")
            # Persist with dedup / supersede.
            result = await self._upsert(outcome.fact, outcome.category)
            return AutoMemoryResult(
                action=ACTION_STORE,
                accepted=True,
                stored=True,
                category=outcome.category,
                fact=outcome.fact,
                reason=result,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = type(exc).__name__
            logger.warning(
                "auto-memory: evaluation failed safely: %s", type(exc).__name__
            )
            return AutoMemoryResult(
                action=ACTION_SKIP, reason="error", error=self.last_error
            )

    async def _bounded_search(self, query: str) -> list[MemoryEntry]:
        if self.provider is None or not self.provider.is_available():
            return []
        try:
            return await self.provider.search(query, limit=MAX_MEMORY_RESULTS)
        except Exception:  # pragma: no cover - defensive
            return []

    async def _upsert(self, fact: str, category: str) -> str:
        if self.provider is None or not self.provider.is_available():
            raise RuntimeError("memory store unavailable")
        existing = await self.provider.find_exact(fact)
        if existing is not None:
            await self.provider.update(existing.id, content=fact, category=category)
            logger.info("AUTO MEMORY: updated-duplicate")
            return "updated-duplicate"
        topic_key = _topic_key(fact, category)
        if topic_key:
            query = _key_to_query(topic_key)
            if query:
                candidates = await self.provider.search(
                    query, limit=MAX_MEMORY_RESULTS, category=category
                )
                for candidate in candidates:
                    if _topic_key(candidate.content, candidate.category) == topic_key:
                        await self.provider.update(
                            candidate.id, content=fact, category=category
                        )
                        logger.info("AUTO MEMORY: superseded")
                        return "superseded"
        entry = MemoryEntry(content=fact, category=category, source="voice")
        await self.provider.store(entry)
        logger.info("AUTO MEMORY: stored category=%s", category)
        return "stored"


class AutoMemoryController:
    """Tracks auto-memory background tasks so they never leak across sessions.

    Failure-isolated by construction: every scheduled evaluation catches its
    own errors, and :meth:`shutdown` cancels + drains all pending tasks so no
    "Task was destroyed but it is pending!" warning can appear.
    """

    def __init__(self, pipeline: AutoMemoryPipeline) -> None:
        self._pipeline = pipeline
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    @property
    def pipeline(self) -> AutoMemoryPipeline:
        return self._pipeline

    def submit_evaluate(
        self,
        text: str,
        *,
        accepted: bool = True,
        on_result: Callable[[AutoMemoryResult], Awaitable[None]] | None = None,
    ) -> bool:
        """Schedule a best-effort evaluation; never blocks the caller."""
        if self._closed:
            return False

        async def _dispatch() -> None:
            try:
                result = await self._pipeline.detect(text, accepted=accepted)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "auto-memory: background dispatch failed safely: %s",
                    type(exc).__name__,
                )
                return
            # Only bounded recall results may influence the reply path; a store
            # never does (no competing reply, no generate_reply).
            if result.is_recall and result.results and on_result is not None:
                try:
                    await on_result(result)
                except Exception:  # pragma: no cover - defensive
                    logger.warning("auto-memory: recall callback failed safely")

        task = asyncio.ensure_future(_dispatch())
        self._tasks.add(task)
        task.add_done_callback(self._discard)
        return True

    def _discard(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)

    def pending(self) -> list[asyncio.Task]:
        return list(self._tasks)

    async def shutdown(self) -> int:
        """Cancel and drain tracked tasks (idempotent). Returns cancelled count."""
        self._closed = True
        pending = list(self._tasks)
        if not pending:
            return 0
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        return len(pending)


__all__ = [
    "ACTION_RECALL",
    "ACTION_SKIP",
    "ACTION_STORE",
    "AutoMemoryController",
    "AutoMemoryPipeline",
    "AutoMemoryResult",
    "ClassifierOutcome",
    "auto_recall_query",
    "classify_candidate",
    "clean_candidate",
    "stage1_skip_reason",
]
