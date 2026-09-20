import re
import time
import unicodedata
from collections.abc import Callable

from livekit.agents.llm import ToolError

_WAKE_WORD = "jarvis"

CONVERSATION_TIMEOUT_SECONDS = 25.0

_SLEEP_COMMAND_RE = re.compile(
    r"\b(?:go\s+to\s+sleep|stop\s+listening|that[s\u2019']?s\s+all)\b",
    re.IGNORECASE,
)

_DOMAIN_RE = re.compile(
    r"\b(?:https?://\S+|www\.\S+|"
    r"[a-z0-9-]+\.(?:com|org|net|io|dev|co|uk|nl|de|fr|be|eu|info|biz|me|"
    r"tv|app|ai|xyz|cloud|shop|gg))\b"
)

_BROWSER_WORDS_RE = re.compile(
    r"\b(?:browsers?|web\s*browsers?|internet|websites?|webpaginas?|"
    r"web\s*pages?|sites?|tab|tabs|zoekmachine|web)\b"
)

_SEARCH_VERBS_RE = re.compile(
    r"\b(?:zoek(?:en|t|e)?|opzoeken|search(?:es|ing)?|look\s*up|google|"
    r"duckduckgo|bing)\b"
)

_CONTINUATION_RE = re.compile(
    r"\b(?:scroll(?:en)?|click(?:een)?|type|druk\s*op|drukken|close|sluit|"
    r"back|terug|ga\s*terug|volgende|next|vorige|previous|refresh|ververs|"
    r"verversen|reload|stop|print|afdrukken|bookmark|menu|home|read|lees|"
    r"lees\s*voor|screenshot|schermafbeelding)\b"
)

_DEVELOPER_PHRASES_RE = re.compile(
    r"\b(?:"
    r"developer\s*mode|"
    r"(?:enter|turn\s*on|enable|use)\s+developer\s*mode|"
    r"improve\s+(?:your|the)\s+(?:code|agent|assistant|skill|behaviour|behavior|integration|tools?)|"
    r"fix\s+(?:your|the)\s+(?:[\w-]+\s+){0,3}(?:code|bug|issue|problem|agent)|"
    r"add\s+(?:the\s+)?ability\s+to|"
    r"create\s+(?:an?|a)\s+(?:[\w-]+\s+){0,3}(?:integration|plugin|tool|service|command|app|website|feature)|"
    r"write\s+(?:your\s+own|some|new)\s+code|"
    r"refactor|"
    r"optimise\s+(?:your\s+)?code|optimize\s+(?:your\s+)?code|"
    r"extend\s+(?:your\s+)?(?:code|agent|assistant)"
    r")\b"
)

_DEV_CONTROL_RE = re.compile(
    r"\b(?:"
    r"(?:show|list)\s+(?:the\s+|my\s+)?(?:development\s+)?tasks|"
    r"(?:what\s+is|what'?s)\s+(?:the\s+)?(?:status|progress)|"
    r"approve(?:\s+the)?\s+(?:change|task|merge|work)|"
    r"reject(?:\s+the)?\s+(?:change|task)|"
    r"cancel\s+(?:the\s+)?(?:task|change|work)|"
    r"roll\s*back|"
    r"what\s+did\s+you\s+change|"
    r"(?:enable|disable)\s+maintenance"
    r")\b"
)

_HERMES_EXPLICIT_RE = re.compile(
    r"\b(?:"
    r"ask\s+(?:the\s+|your\s+|an?\s+)?hermes\b|"
    r"use\s+hermes\w*\b|"
    r"call\s+(?:the\s+|on\s+)?hermes\b|"
    r"consult\s+(?:the\s+)?hermes\b|"
    r"hermes\s+(?:agent|mode|to|cans?|should|will)\b"
    r")\b"
)

_HERMES_DEEP_RE = re.compile(
    r"\b(?:"
    r"analyse(?:s|d)?|analyze(?:s|d)?|analysis|"
    r"investigate(?:s|d|ion)?|diagnos(?:e|es|ing)?|debug(?:ging)?|"
    r"root\s+cause|deep\s+dive|deep\s+analysis|"
    r"design|architecture|architect|refactor|roadmap|blueprint|strategy|"
    r"feasibility|trade[\s-]*offs?|how\s+should\s+we|multi-step|"
    r"complex\s+(?:task|build|problem|project)|"
    r"plan\s+(?:this|that|it|out|the\s+rollout|the\s+migration|"
    r"the\s+improvements|the\s+implementation)"
    r")\b"
)

_CONSEQUENTIAL_REQUEST_RE = re.compile(
    r"\b(?:"
    r"send|create|add|post|buy|purchase|pay|approve|merge|delete|remove|"
    r"submit|cancel|enable|disable|change\s+(?:the\s+)?(?:ad|budget|price)"
    r")\b"
)

_CONFIRMATIONS = frozenset(
    {
        "ja",
        "jaa",
        "yes",
        "yeah",
        "yep",
        "yup",
        "ok",
        "okay",
        "oke",
        "oké",
        "go ahead",
        "go on",
        "do it",
        "doe maar",
        "doe het",
        "ga door",
        "ga maar door",
        "ga je gang",
        "please",
        "please do",
        "sure",
        "zeker",
        "prima",
        "perfect",
        "goed",
        "klinkt goed",
        "sounds good",
        "door",
        "verder",
    }
)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def has_wake_word(text: str | None) -> bool:
    return _WAKE_WORD in normalize_text(text)


def is_confirmation(text: str | None) -> bool:
    return normalize_text(text).strip(" .,!?") in _CONFIRMATIONS


def is_sleep_command(text: str | None) -> bool:
    return bool(_SLEEP_COMMAND_RE.search(normalize_text(text)))


def browser_intent(text: str | None) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    return bool(
        _DOMAIN_RE.search(norm)
        or _BROWSER_WORDS_RE.search(norm)
        or _SEARCH_VERBS_RE.search(norm)
        or _CONTINUATION_RE.search(norm)
    )


def is_developer_request(text: str | None) -> bool:
    """Whether the utterance asks to modify the agent's own code or manage tasks."""
    norm = normalize_text(text)
    if not norm:
        return False
    return bool(_DEVELOPER_PHRASES_RE.search(norm) or _DEV_CONTROL_RE.search(norm))


def is_hermes_explicit(text: str | None) -> bool:
    """Whether the utterance explicitly asks to delegate to Hermes."""
    norm = normalize_text(text)
    return bool(norm and _HERMES_EXPLICIT_RE.search(norm))


def is_hermes_deep(text: str | None) -> bool:
    """Whether the utterance shows a deep planning / analysis / dev intent."""
    norm = normalize_text(text)
    return bool(norm and _HERMES_DEEP_RE.search(norm))


def is_hermes_request(text: str | None) -> bool:
    """Whether the utterance should be delegated to the Hermes backend."""
    return is_hermes_explicit(text) or is_hermes_deep(text)


class ConversationGate:
    """Tracks the active conversation window opened by the wake word.

    The wake word opens a short conversation window. While it stays open,
    follow-up utterances are accepted without repeating "Jarvis" and each one
    refreshes the window. The window closes after a timeout or an explicit
    sleep command.
    """

    def __init__(
        self,
        timeout_s: float = CONVERSATION_TIMEOUT_SECONDS,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._now = now or time.monotonic
        self._active = False
        self._last_interaction = 0.0

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_s

    def activate(self) -> None:
        self._active = True
        self._last_interaction = self._now()

    def refresh(self) -> None:
        if self._active:
            self._last_interaction = self._now()

    def deactivate(self) -> None:
        self._active = False
        self._last_interaction = 0.0

    def is_active(self) -> bool:
        if not self._active:
            return False
        if self._now() - self._last_interaction > self._timeout_s:
            self._active = False
            return False
        return True

    def should_accept(self, text: str | None) -> bool:
        norm = normalize_text(text)
        if not norm.strip():
            return False
        if not self.is_active():
            if is_sleep_command(norm):
                return False
            if has_wake_word(norm):
                self.activate()
                return True
            return False
        if is_sleep_command(norm):
            self.deactivate()
            return True
        self.refresh()
        return True


class ToolGate:
    """Enforces wake-word, conversation, and explicit-browser-request rules.

    The wake word opens a short conversation window. While it is open,
    follow-up utterances (without repeating "Jarvis") are accepted and refresh
    the window. The window closes on timeout or an explicit sleep command. The
    browser remains strictly permission-only.
    """

    def __init__(self, conversation: ConversationGate | None = None) -> None:
        self._turn_text: str | None = None
        self._browser_armed = False
        self._conversation = conversation or ConversationGate()

    @property
    def conversation(self) -> ConversationGate:
        return self._conversation

    @property
    def current_turn_text(self) -> str | None:
        return self._turn_text

    def should_accept(self, text: str | None = None) -> bool:
        return self._conversation.should_accept(
            text if text is not None else self._turn_text
        )

    def is_conversation_active(self) -> bool:
        return self._conversation.is_active()

    def deactivate(self) -> None:
        self._conversation.deactivate()

    def set_user_request(self, text: str | None) -> None:
        self._turn_text = text
        if text:
            if browser_intent(text):
                self._browser_armed = True
            elif not is_confirmation(text):
                self._browser_armed = False
        else:
            self._browser_armed = False

    def ensure_wake_word(self) -> None:
        if not self.should_accept():
            raise ToolError(
                "The user did not say the wake word 'Jarvis' or start an "
                "active conversation. Do not call any tool or take any "
                "action. Wait for a request that includes 'Jarvis'."
            )

    def ensure_active_conversation(self) -> None:
        """Require the wake word or an already-active conversation window."""
        if not self.should_accept():
            raise ToolError(
                "The user did not say the wake word 'Jarvis' or start an "
                "active conversation. Do not call any tool or take any "
                "action. Wait for a request that includes 'Jarvis'."
            )

    def ensure_action_level(self, level: int, requested: bool = False) -> None:
        """Enforce an integration action level against the current turn.

        Level 1 (safe read) and level 2 (reversible action) only need an active
        conversation. Levels 3+ (consequential / security) additionally require
        the user to have explicitly requested the action in the current turn or
        to have just confirmed it; otherwise the agent must stop and ask.
        """
        self.ensure_active_conversation()
        if int(level) < 3:
            return
        norm = normalize_text(self._turn_text or "")
        if requested:
            return
        if is_confirmation(norm):
            return
        if _CONSEQUENTIAL_REQUEST_RE.search(norm):
            return
        raise ToolError(
            "This action requires the user's explicit approval first. "
            "Do not perform it yet: ask the user to confirm the exact action "
            "and wait for a clear confirmation before retrying."
        )

    def allows_reply(self) -> bool:
        """Whether the agent may produce audible output for the current turn.

        The wake word opens a short conversation window; while it stays open,
        follow-up utterances are authorized so the agent can reply naturally.
        """
        return self.should_accept()

    def ensure_confirmation_allowed(self) -> None:
        if self.allows_reply():
            return
        if is_confirmation(self._turn_text) and self._browser_armed:
            return
        raise ToolError(
            "The user did not say the wake word 'Jarvis' or authorize this "
            "action. Wait for an explicit request that includes 'Jarvis'."
        )

    def ensure_browser_requested(self) -> None:
        accepted = self.should_accept()
        if accepted:
            allowed = self._browser_armed or browser_intent(self._turn_text)
        elif is_confirmation(self._turn_text):
            allowed = self._browser_armed
        else:
            allowed = False

        if not allowed:
            if not accepted and not is_confirmation(self._turn_text):
                raise ToolError(
                    "The user did not say the wake word 'Jarvis' or start an "
                    "active conversation in this request. Do not call any "
                    "tool or take any action. Wait until the next user "
                    "request includes 'Jarvis'."
                )
            raise ToolError(
                "The user did not explicitly ask to use the browser in this "
                "request. Never open or use the browser unless the user asks "
                "for it explicitly (e.g. 'open <website>', 'go to <url>', "
                "'search the internet'). Ask the user for permission first."
            )

    def ensure_developer_requested(self) -> None:
        """Require an active conversation plus an explicit development request."""
        if not self.should_accept():
            raise ToolError(
                "The user did not say the wake word 'Jarvis' or start an "
                "active conversation. Wait for a request that includes "
                "'Jarvis' before using any development tool."
            )
        if not is_developer_request(self._turn_text):
            raise ToolError(
                "The user did not ask to change or manage your own code. "
                "Never modify source files, run checks, or touch git on your "
                "own initiative. Only act when the user explicitly asks, such "
                "as 'enter developer mode', 'improve your code', or 'show the "
                "development tasks'."
            )

    def ensure_hermes_requested(self) -> None:
        """Require an active conversation plus explicit or deep Hermes work.

        Hermes is a backend specialist. The user must either explicitly ask to
        delegate to Hermes ("ask Hermes to ...") or raise a deep
        planning / analysis / development task; everything else stays with
        JARVIS.
        """
        self.ensure_active_conversation()
        if not is_hermes_request(self._turn_text):
            raise ToolError(
                "The user did not ask to delegate this to the Hermes backend. "
                "Only use Hermes when the user explicitly asks (such as 'ask "
                "Hermes to analyse this') or for deep planning or development "
                "work; otherwise answer yourself."
            )
