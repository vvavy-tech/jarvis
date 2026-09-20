"""Memory policy: what may be stored, and how reads stay bounded.

The policy is the single place that decides:

* **Writes** — only explicit, non-empty, secret-free content is acceptable.
  Credentials (API keys, tokens, passwords, private keys, ``Bearer ...``,
  high-entropy strings) are refused and never echoed back in diagnostics.
* **Duplicates** — repeated saves of the same content update the existing
  record instead of piling up endless copies (no embeddings needed in Phase 1).
* **Bounded reads** — the number of results and the size of context injected
  into any conversation are hard-capped so memory can never flood the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_MEMORY_RESULTS = 5
MAX_MEMORY_CONTEXT_CHARS = 1200

# Patterns that mark content as a credential. Broad by design: if a stored
# fact even resembles a secret, refuse it. Never reuse a captured secret in a
# returned message.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),  # OpenAI-style API keys
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE),
    re.compile(r"\bpassword\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"\bpasswd\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"\bsecret\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|access[_-]?key)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"\btoken\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"\bclient[_-]?secret\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"\bpasswd\b", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\b", re.IGNORECASE),
    re.compile(r"\bsecret\b", re.IGNORECASE),
    re.compile(r"\bprivate\s+key\b", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b[\dA-Za-z_-]{32,}\b"),  # high-entropy blobs
)

# Phrases that indicate an explicit memory write request.
_EXPLICIT_REQUEST_RE = re.compile(
    r"\b(?:"
    r"remember\b|"
    r"save\s+(?:this|that|it|note\b)|"
    r"keep\s+(?:this|that)\s+in\s+mind|"
    r"store\s+(?:this|that|it)|"
    r"take\s+a\s+note|"
    r"don[\u2019']?t\s+forget\b"
    r")\b",
    re.IGNORECASE,
)

# Phrases that explicitly ask JARVIS to recall stored memory.
_EXPLICIT_RECALL_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:[\w\u2019'&-]+\s+){0,3}did\s+(?:i|we)\s+ask\s+you\s+to\s+remember\b|"
    r"what\s+did\s+(?:i|we)\s+ask\s+you\s+to\s+remember\b|"
    r"do\s+you\s+remember\b|"
    r"did\s+you\s+remember\b|"
    r"(?:tell\s+me|remind\s+me)\s+what(?:'s|\s+is)?\s+in\s+(?:your\s+)?memory|"
    r"check\s+(?:your\s+)?memory|"
    r"recall\b|"
    r"search\s+(?:your\s+)?memories?|"
    r"look\s+in\s+(?:your\s+)?memory|"
    r"what\s+do\s+you\s+know\s+about\b"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryDecision:
    accepted: bool
    reason: str = ""


class MemoryPolicy:
    """Decides what may be stored and how memory is read back."""

    def should_store(self, content: str) -> MemoryDecision:
        """Return whether ``content`` may be persisted.

        Refusals never echo the offending secret back to the caller.
        """
        text = (content or "").strip()
        if not text:
            return MemoryDecision(False, "memory content is empty")
        if _contains_secret(text):
            return MemoryDecision(
                False,
                "that looks like a credential or secret and cannot be stored",
            )
        return MemoryDecision(True)

    def is_duplicate(self, existing: str, candidate: str) -> bool:
        """Exact/normalised equality, used to update rather than pile up."""
        return self._normalize(existing) == self._normalize(candidate)

    def is_explicit_write(self, utterance: str | None) -> bool:
        if not utterance or not utterance.strip():
            return False
        return bool(_EXPLICIT_REQUEST_RE.search(utterance))

    def is_explicit_recall(self, utterance: str | None) -> bool:
        if not utterance or not utterance.strip():
            return False
        return bool(_EXPLICIT_RECALL_RE.search(utterance))

    def should_auto_context(self, utterance: str | None) -> bool:
        """Whether the current turn warrants consulting stored memory.

        Broad automatic reads are avoided: only explicit recalls get context.
        """
        return self.is_explicit_recall(utterance)

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = " ".join(text.lower().split())
        return re.sub(r"[\s,.;:!?'\"]+", " ", cleaned).strip()


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)
