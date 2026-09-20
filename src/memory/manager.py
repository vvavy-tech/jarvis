"""MemoryManager: policy-aware coordination between the provider and callers.

The manager owns the bounded read path: it never injects the whole database,
never blocks, and never raises into the voice loop. It is the seam used by
Hermes/Developer for read-only context and by the memory tools.
"""

from __future__ import annotations

from memory.base import MemoryProvider
from memory.policy import MAX_MEMORY_CONTEXT_CHARS, MAX_MEMORY_RESULTS, MemoryPolicy


class MemoryManager:
    """Coordinates provider + policy with failure isolation."""

    def __init__(self, provider: MemoryProvider | None, policy: MemoryPolicy) -> None:
        self.provider = provider
        self.policy = policy
        self.last_error: str = ""

    async def build_context(
        self, query: str, *, limit: int = MAX_MEMORY_RESULTS
    ) -> str:
        """Bounded, forgetful relevant context for a query; "" when nothing."""
        if self.provider is None or not self.provider.is_available():
            return ""
        try:
            results = await self.provider.search(query, limit=limit)
        except Exception as exc:
            self.last_error = str(exc)
            return ""
        if not results:
            return ""
        snippets = [f"- {entry.content}" for entry in results]
        context = "\n".join(snippets)
        if len(context) > MAX_MEMORY_CONTEXT_CHARS:
            context = context[:MAX_MEMORY_CONTEXT_CHARS].rstrip() + "…"
        return context
