"""MemoryIntegration: registers persistent memory as a capability and exposes
voice tools for explicit store/recall, gated like every other capability.

Policy and bounded retrieval live in :class:`memory.policy.MemoryPolicy` and
:class:`memory.manager.MemoryManager`; this class only wires them to the gate
and the LLM. Memory must never break JARVIS: every tool catches provider
failure and returns a polite, unblocking message instead of raising.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from livekit.agents import RunContext, function_tool

from gates import ToolGate
from integrations.base import ActionLevel, Integration
from memory.base import MemoryEntry, MemoryProvider
from memory.manager import MemoryManager
from memory.policy import MemoryPolicy

logger = logging.getLogger("agent")

_GATE_MSG = (
    "Long-term memory is only written or updated when the user explicitly "
    "asks (such as 'remember ...', 'save this ...', 'keep that in mind'). "
    "Never store memories on your own initiative or from background speech."
)
_UNSTRUCTURED_MSG = (
    "I can keep that in memory for you. What exactly would you like me to remember?"
)


class MemoryIntegration(Integration):
    """Registers persistent memory in the capability registry + voice tools."""

    name = "persistent_memory"
    description = (
        "Durable long-term memory kept locally across restarts. Reads are "
        "bounded; writes only happen on explicit user request."
    )
    read_only = False
    required_env = ()
    default_level = ActionLevel.SAFE_READ
    _TOOLS = (
        "remember_memory",
        "search_memory",
        "get_memory",
        "update_memory",
        "forget_memory",
        "list_memories",
        "memory_status",
    )
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {
        "remember_memory": ActionLevel.REVERSIBLE,
        "update_memory": ActionLevel.REVERSIBLE,
        "forget_memory": ActionLevel.REVERSIBLE,
        "search_memory": ActionLevel.SAFE_READ,
        "get_memory": ActionLevel.SAFE_READ,
        "list_memories": ActionLevel.SAFE_READ,
        "memory_status": ActionLevel.SAFE_READ,
    }

    def __init__(
        self,
        *,
        gate: ToolGate | None = None,
        failure_log: Any | None = None,
        provider: MemoryProvider | None = None,
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self.provider = provider
        self.policy = MemoryPolicy()
        self.manager = MemoryManager(self.provider, self.policy)

    # ------------------------------------------------------------------ #
    # capability status
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        return self.provider is not None and self.provider.is_available()

    def is_authenticated(self) -> bool:
        return self.provider is not None

    def status_note(self) -> str:
        if self.provider is None:
            return (
                "persistent memory is configured but not available. "
                "Health: unavailable."
            )
        if not self.provider.is_available():
            return "persistent memory store is unavailable. Health: error."
        return (
            "persistent storage: local SQLite; semantic_search: false; health: healthy"
        )

    async def health_check(self) -> dict[str, Any]:
        if self.provider is None or not self.provider.is_available():
            return {
                "name": self.name,
                "ok": False,
                "status": "unavailable",
                "detail": "persistent memory is not available right now.",
            }
        return await self.provider.health_check()

    # ------------------------------------------------------------------ #
    # write tools (explicit-request only)
    # ------------------------------------------------------------------ #

    @function_tool()
    async def remember_memory(
        self, context: RunContext, content: str = "", category: str = "general"
    ) -> dict[str, Any]:
        """Persist a personal fact to long-term memory.

        Use only when the user explicitly asks you to remember something
        (e.g. 'remember that ...', 'save this ...', 'keep this in mind').
        Never save passwords, API keys, tokens, or any other secret. If the
        same fact is already saved, update it instead of adding a duplicate.

        Args:
            content: The exact fact the user asked you to remember.
            category: The kind of fact (preference, project, business, person,
                decision, workflow, technical, integration, general).
        """
        self._require_write()
        logger.info("MEMORY write requested")
        try:
            decision = self.policy.should_store(content)
            if not decision.accepted:
                logger.info("MEMORY write refused")
                return {
                    "status": "refused",
                    "message": decision.reason,
                }
            entry = MemoryEntry(content=content.strip(), category=category or "general")
            existing = await self.provider.find_exact(entry.content)
            if existing is not None:
                await self.provider.update(
                    existing.id, content=entry.content, category=entry.category
                )
                logger.info("MEMORY write updated id=%s", existing.id)
                return {
                    "status": "stored",
                    "id": existing.id,
                    "message": "Already saved; I updated that memory.",
                }
            stored = await self.provider.store(entry)
            logger.info("MEMORY write succeeded id=%s", stored.id)
            return {
                "status": "stored",
                "id": stored.id,
                "message": "Saved to long-term memory.",
            }
        except Exception as exc:
            return self._unavailable_message(exc)

    @function_tool()
    async def update_memory(
        self, context: RunContext, memory_id: str, content: str = ""
    ) -> dict[str, Any]:
        """Update an existing memory by id with new content.

        Uses the same explicit-request rule as remember, and never stores
        secrets. Returns the refreshed record when successful.

        Args:
            memory_id: The id of the memory to update.
            content: The new content for that memory.
        """
        self._require_write()
        try:
            if not content.strip():
                return {"status": "refused", "message": "new content is empty"}
            updated = await self.provider.update(memory_id, content=content.strip())
            if updated is None:
                return {"status": "missing", "message": "no such memory"}
            return {
                "status": "updated",
                "id": updated.id,
                "message": "Memory updated.",
            }
        except Exception as exc:
            return self._unavailable_message(exc)

    @function_tool()
    async def forget_memory(
        self, context: RunContext, memory_id: str
    ) -> dict[str, Any]:
        """Permanently remove one memory by id.

        Reminder: only use this when the user explicitly asks to forget or
        delete something they previously asked you to remember.

        Args:
            memory_id: The id of the memory to forget.
        """
        self._require_write()
        try:
            removed = await self.provider.delete(memory_id)
            if not removed:
                return {"status": "missing", "message": "no such memory"}
            return {"status": "forgotten", "message": "That memory was forgotten."}
        except Exception as exc:
            return self._unavailable_message(exc)

    # ------------------------------------------------------------------ #
    # read tools
    # ------------------------------------------------------------------ #

    @function_tool()
    async def search_memory(
        self, context: RunContext, query: str = "", category: str | None = None
    ) -> dict[str, Any]:
        """Find relevant memories by keyword search.

        Only consult memory when the user asks about something they previously
        told you (their preferences, past decisions, prior facts) or when the
        task genuinely depends on stored context. Use a concise keyword query,
        not the full user sentence. Never claim a memory exists unless it was
        actually retrieved.

        Args:
            query: A short keyword or phrase to search for in memory.
            category: Optionally limit to one category.
        """
        self.gate.ensure_active_conversation()
        try:
            if not query.strip():
                return {"status": "ok", "summary": "", "results": []}
            logger.info("MEMORY recall requested query=%s", query.strip()[:120])
            results = await self.provider.search(
                query.strip(), limit=5, category=category
            )
            logger.info("MEMORY recall results=%d", len(results))
            return self._results_payload(results)
        except Exception as exc:
            return self._unavailable_message(exc)

    @function_tool()
    async def get_memory(self, context: RunContext, memory_id: str) -> dict[str, Any]:
        """Load one memory by id; returns its content verbatim.

        Args:
            memory_id: The id of the memory to load.
        """
        self.gate.ensure_active_conversation()
        try:
            entry = await self.provider.get(memory_id)
            if entry is None:
                return {"status": "missing", "message": "no such memory"}
            return {
                "status": "ok",
                "content": entry.content,
                "category": entry.category,
            }
        except Exception as exc:
            return self._unavailable_message(exc)

    @function_tool()
    async def list_memories(
        self,
        context: RunContext,
        category: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """List recent memories, optionally filtered by category or project.

        Args:
            category: Optional category (preference, project, person, etc.).
            project: Optional project label to narrow the list.
        """
        self.gate.ensure_active_conversation()
        try:
            entries = await self.provider.list_all(
                category=category, project=project, limit=20
            )
            return self._results_payload(entries, count_all=len(entries))
        except Exception as exc:
            return self._unavailable_message(exc)

    @function_tool()
    async def memory_status(self, context: RunContext) -> dict[str, Any]:
        """Report whether persistent memory is healthy and how it is stored.

        Answers 'is your memory working?'. Only report healthy when the store
        initialized and answers a health probe.
        """
        if self.provider is None:
            return {
                "status": "unavailable",
                "mode": "none",
                "message": "No memory store.",
            }
        health = await self.provider.health_check()
        mode = "persistence: local sqlite" if health.get("ok") else "unavailable"
        return {
            "status": health.get("status", "error"),
            "mode": mode,
            "message": "Long-term memory is working."
            if health.get("ok")
            else ("Long-term memory is not available right now."),
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _require_write(self) -> None:
        self.gate.ensure_memory_requested()
        self.gate.ensure_action_level(int(ActionLevel.REVERSIBLE))

    def _results_payload(
        self, entries: list[MemoryEntry], *, count_all: int | None = None
    ) -> dict[str, Any]:
        payload = {
            "status": "ok",
            "summary": "",
            "count": count_all if count_all is not None else len(entries),
            "results": [
                {"id": e.id, "content": e.content, "category": e.category}
                for e in entries
            ],
            "message": (
                f"Found {len(entries)} relevent memory"
                + ("ies." if len(entries) != 1 else ".")
                if entries
                else "No matching memories found."
            ),
        }
        if entries:
            payload["summary"] = "\n".join(f"- {e.content}" for e in entries)
        return payload

    def _unavailable_message(self, exc: Exception) -> dict[str, Any]:
        self._log_failure("memory", "local-sqlite", exc, fallback=_UNSTRUCTURED_MSG)
        return {
            "status": "error",
            "message": "Memory is not available right now. Continue without it.",
        }


__all__ = ["MemoryIntegration"]
