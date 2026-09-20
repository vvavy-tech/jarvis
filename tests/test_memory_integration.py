"""Tests for the Persistent Memory capability (Phase 1, LOCAL ONLY).

Covers: registry status, the voice tools' gating and failure isolation, the
LocalSQLiteMemoryProvider (schema creation, CRUD, persistence across restart,
bounded search), the MemoryPolicy (explicit-only writes, secret protection,
duplicate handling), the bounded Hermes read-context, and the real persistence
round-trip. Memory must never break JARVIS: when the store is missing or
corrupts, status reflects that and no tool raises into the voice loop.
"""

import asyncio
import os
import sqlite3

import pytest
from livekit.agents.llm import ToolError

from gates import ToolGate, is_memory_request
from integrations import build_default_registry
from memory import (
    LocalSQLiteMemoryProvider,
    MemoryEntry,
    MemoryIntegration,
    MemoryManager,
    MemoryPolicy,
)
from memory.policy import MAX_MEMORY_CONTEXT_CHARS, MAX_MEMORY_RESULTS


class _ActiveGate:
    """A gate that is always in an active conversation with a turn."""

    def __init__(self, turn: str = "Jarvis, remember my test colour"):
        self.turn = turn

    def ensure_active_conversation(self) -> None:
        return None

    def ensure_action_level(self, level: int, requested: bool = False) -> None:
        return None

    def ensure_memory_requested(self) -> None:
        return None


@pytest.fixture()
def policy() -> MemoryPolicy:
    return MemoryPolicy()


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / "jarvis_memory.db")


def run_sync(coro):
    return asyncio.run(coro)


class TestGateIntent:
    def test_memory_write_phrases_recognised(self):
        assert is_memory_request("Jarvis, remember that I like cobalt blue") is True
        assert is_memory_request("Jarvis, save this: coffee with milk") is True
        assert is_memory_request("Jarvis, keep in mind the project deadline") is True
        assert is_memory_request("Jarvis, what time is it") is False
        assert is_memory_request("Jarvis, play some jazz") is False

    def test_gate_requires_explicit_memory_request(self):
        gate = ToolGate()
        gate.set_user_request("Jarvis, tell me a joke")
        with pytest.raises(ToolError, match=r"save +memory|remember"):
            gate.ensure_memory_requested()

        gate.set_user_request("Jarvis, remember my test colour")
        gate.ensure_memory_requested()

    def test_gate_requires_active_conversation(self):
        gate = ToolGate()
        with pytest.raises(ToolError, match="wake word"):
            gate.ensure_memory_requested()


class TestRegistryStatus:
    def test_build_default_registry_registers_memory(self):
        registry = build_default_registry()
        assert "persistent_memory" in registry.names()
        integration = registry.get("persistent_memory")
        assert integration is not None
        status = integration.status()
        assert status.available is False
        assert status.authenticated is False

    def test_build_default_registry_takes_provider(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        registry = build_default_registry(memory_provider=provider)
        integration = registry.get("persistent_memory")
        assert integration is not None
        assert integration.is_available() is True

    def test_injected_provider_health_is_healthy(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        integration = build_default_registry(memory_provider=provider).get(
            "persistent_memory"
        )
        health = run_sync(integration.health_check())
        assert health["status"] == "healthy"
        assert health["ok"] is True

    def test_capabilities_include_memory(self, db_path):
        registry = build_default_registry(
            memory_provider=LocalSQLiteMemoryProvider(path=db_path)
        )
        caps = registry.capabilities()
        memory = next(c for c in caps if c["name"] == "persistent_memory")
        assert memory["available"] is True
        assert memory["authenticated"] is True
        assert "persistent" in memory["note"]

    def test_capabilities_show_unavailable_without_provider(self):
        registry = build_default_registry()
        caps = registry.capabilities()
        memory = next(c for c in caps if c["name"] == "persistent_memory")
        assert memory["available"] is False

    def test_report_lists_memory(self, db_path):
        registry = build_default_registry(
            memory_provider=LocalSQLiteMemoryProvider(path=db_path)
        )
        assert "persistent_memory: connected" in registry.report().casefold()


class TestLocalSqliteProvider:
    def test_provider_creates_database_and_schema(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        assert os.path.exists(db_path)
        connected = sqlite3.connect(db_path)
        try:
            assert connected.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
        finally:
            connected.close()
        run_sync(provider.close())

    def test_provider_initialize_is_idempotent(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        run_sync(provider.initialize())
        run_sync(provider.close())

    def test_store_and_get_roundtrip(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        entry = MemoryEntry(
            content="my test colour is cobalt blue", category="preference"
        )
        stored = run_sync(provider.store(entry))
        assert stored.id
        fetched = run_sync(provider.get(stored.id))
        assert fetched is not None
        assert fetched.content == "my test colour is cobalt blue"
        assert fetched.category == "preference"
        run_sync(provider.close())

    def test_persistence_survives_restart(self, db_path):
        first = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(first.initialize())
        stored = run_sync(
            first.store(
                MemoryEntry(
                    content="persistent fact survives restart",
                    category="decision",
                    project="jarvis",
                )
            )
        )
        run_sync(first.close())

        second = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(second.initialize())
        fetched = run_sync(second.get(stored.id))
        assert fetched is not None
        assert fetched.content == "persistent fact survives restart"
        assert fetched.project == "jarvis"
        run_sync(second.close())

    def test_keyword_search(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        run_sync(
            provider.store(
                MemoryEntry(
                    content="cobalt blue is my favourite", category="preference"
                )
            )
        )
        run_sync(
            provider.store(
                MemoryEntry(
                    content="the router runs on port 8080", category="technical"
                )
            )
        )
        results = run_sync(provider.search("cobalt blue"))
        assert len(results) == 1
        assert "cobalt blue" in results[0].content
        run_sync(provider.close())

    def test_search_is_bounded(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        for i in range(25):
            run_sync(
                provider.store(
                    MemoryEntry(content=f"common token {i}", category="general")
                )
            )
        results = run_sync(provider.search("common token"))
        assert len(results) <= MAX_MEMORY_RESULTS
        run_sync(provider.close())

    def test_update_memory(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        stored = run_sync(
            provider.store(MemoryEntry(content="old content", category="general"))
        )
        updated = run_sync(provider.update(stored.id, content="new content"))
        assert updated is not None
        assert updated.content == "new content"
        fetched = run_sync(provider.get(stored.id))
        assert fetched.content == "new content"
        run_sync(provider.close())

    def test_forget_removes_memory(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        stored = run_sync(
            provider.store(MemoryEntry(content="expendable", category="general"))
        )
        removed = run_sync(provider.delete(stored.id))
        assert removed is True
        assert run_sync(provider.get(stored.id)) is None
        run_sync(provider.close())

    def test_category_and_project_metadata(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        stored = run_sync(
            provider.store(
                MemoryEntry(
                    content="meeting at noon",
                    category="person",
                    project="acme",
                    tags=("client", "meeting"),
                )
            )
        )
        refetched = run_sync(provider.get(stored.id))
        assert refetched.category == "person"
        assert refetched.project == "acme"
        assert "client" in refetched.tags
        run_sync(provider.close())

    def test_close_does_not_raise(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        run_sync(provider.close())
        run_sync(provider.close())


class TestMemoryPolicy:
    def test_empty_content_is_rejected(self, policy):
        decision = policy.should_store("   ")
        assert decision.accepted is False
        assert decision.reason

    def test_password_is_rejected(self, policy):
        decision = policy.should_store("remember this password: hunter2")
        assert decision.accepted is False
        assert hunter2_absent(decision.reason)

    def test_api_key_is_rejected(self, policy):
        decision = policy.should_store("my api key is sk-ABCD1234xYzW")
        assert decision.accepted is False

    def test_secret_not_echoed_in_reason(self, policy):
        secret = "super-secret-hunter2-xyz"
        decision = policy.should_store(f"the password is {secret}")
        assert decision.accepted is False
        assert secret not in decision.reason

    def test_normally_explicit_memory_is_allowed(self, policy):
        decision = policy.should_store("cobalt blue is my favourite colour")
        assert decision.accepted is True
        assert not decision.reason

    def test_duplicate_is_updated_not_added(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        first = MemoryEntry(
            content="the test colour is cobalt blue", category="preference"
        )
        stored_first = run_sync(provider.store(first))
        existing = run_sync(provider.find_exact("the test colour is cobalt blue"))
        assert existing is not None
        if existing is None:
            return
        run_sync(provider.update(existing.id, content="the test colour is cobalt blue"))
        results = run_sync(provider.search("cobalt blue"))
        assert len(results) == 1
        assert results[0].id == stored_first.id
        run_sync(provider.close())

    def test_search_is_bounded_by_policy(self, policy):
        assert MAX_MEMORY_RESULTS > 0
        assert MAX_MEMORY_CONTEXT_CHARS > 0


class TestMemoryTools:
    def test_remember_memory_stores(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        integration = _make_memory_integration(provider)
        result = run_sync(
            integration.remember_memory(
                None, content="my test colour is cobalt blue", category="preference"
            )
        )
        assert result["status"] == "stored"
        assert result["id"]
        assert run_sync(provider.count()) == 1

    def test_search_memory_retrieves(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        run_sync(
            provider.store(
                MemoryEntry(content="the launch colour is coral", category="preference")
            )
        )
        integration = _make_memory_integration(provider)
        result = run_sync(integration.search_memory(None, query="coral"))
        assert result["status"] == "ok"
        assert "coral" in result["summary"]

    def test_get_memory_by_id(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        stored = run_sync(
            provider.store(MemoryEntry(content="referenced memory", category="general"))
        )
        integration = _make_memory_integration(provider)
        result = run_sync(integration.get_memory(None, memory_id=stored.id))
        assert result["status"] == "ok"
        assert result["content"] == "referenced memory"

    def test_update_memory_tool(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        stored = run_sync(
            provider.store(MemoryEntry(content="draft", category="general"))
        )
        integration = _make_memory_integration(provider)
        result = run_sync(
            integration.update_memory(None, memory_id=stored.id, content="final")
        )
        assert result["status"] == "updated"

    def test_forget_memory_tool(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        stored = run_sync(
            provider.store(MemoryEntry(content="trivia", category="general"))
        )
        integration = _make_memory_integration(provider)
        result = run_sync(integration.forget_memory(None, memory_id=stored.id))
        assert result["status"] == "forgotten"

    def test_list_memories(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        run_sync(provider.store(MemoryEntry(content="alpha", category="project")))
        run_sync(provider.store(MemoryEntry(content="beta", category="project")))
        integration = _make_memory_integration(provider)
        result = run_sync(integration.list_memories(None, category="project"))
        assert result["status"] == "ok"
        assert result["count"] == 2

    def test_memory_status_reports_health(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        integration = _make_memory_integration(provider)
        result = run_sync(integration.memory_status(None))
        assert result["status"] == "healthy"
        assert "persistence" in result["mode"]

    def test_secret_write_is_refused_and_not_echoed(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        integration = _make_memory_integration(provider)
        secret = "hunter2-super-secret"
        result = run_sync(
            integration.remember_memory(None, content=f"my password is {secret}")
        )
        assert result["status"] == "refused"
        assert secret not in result["message"]
        assert run_sync(provider.count()) == 0

    def test_duplicate_remember_does_not_grow_unbounded(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        integration = _make_memory_integration(provider)
        for _ in range(3):
            run_sync(
                integration.remember_memory(
                    None, content="the test colour is cobalt blue"
                )
            )
        assert run_sync(provider.count()) == 1

    def test_write_tools_gated_as_reversible_and_reads_as_safe(self, db_path):
        integration = _make_memory_integration(LocalSQLiteMemoryProvider(path=db_path))
        assert int(integration.tool_level("remember_memory")) == 2
        assert int(integration.tool_level("forget_memory")) == 2
        assert int(integration.tool_level("search_memory")) == 1
        assert int(integration.tool_level("memory_status")) == 1

    def test_remember_blocked_without_wake_word(self, db_path):
        integration = _make_memory_integration(
            LocalSQLiteMemoryProvider(path=db_path), gate=ToolGate()
        )
        with pytest.raises(ToolError, match="wake word"):
            run_sync(integration.remember_memory(None, content="anything"))

    def test_remember_blocked_without_explicit_request(self, db_path):
        gate = ToolGate()
        gate.set_user_request("Jarvis, what time is it")
        integration = _make_memory_integration(
            LocalSQLiteMemoryProvider(path=db_path), gate=gate
        )
        with pytest.raises(ToolError, match="Jarvis"):
            run_sync(integration.remember_memory(None, content="anything"))

    def test_explicit_remember_phrase_allowed(self, db_path):
        gate = ToolGate()
        gate.set_user_request("Jarvis, remember my test colour is cobalt blue")
        integration = _make_memory_integration(
            LocalSQLiteMemoryProvider(path=db_path), gate=gate
        )
        result = run_sync(
            integration.remember_memory(None, content="my test colour is cobalt blue")
        )
        assert result["status"] == "stored"


class TestFailureIsolation:
    def test_missing_database_still_registers_agent(self):
        registry = build_default_registry(memory_provider=None)
        integration = registry.get("persistent_memory")
        assert integration is not None
        assert integration.is_available() is False

    def test_corrupt_database_reports_error_not_crash(self, db_path):
        with open(db_path, "w") as handle:
            handle.write("not a sqlite database")
        provider = LocalSQLiteMemoryProvider(path=db_path)
        integration = _make_memory_integration(provider)
        result = run_sync(integration.memory_status(None))
        assert result["status"] in {"error", "unavailable"}

    def test_no_tool_raises_when_provider_missing(self):
        integration = _make_memory_integration(None)
        result = run_sync(integration.remember_memory(None, content="whatever"))
        assert result["status"] in {"unavailable", "error"}
        assert "memory" in result["message"].casefold()


class TestMemoryContext:
    def test_auto_context_requires_explicit_recall(self, policy):
        assert (
            policy.should_auto_context("Jarvis, what colour did I ask you to remember?")
            is True
        )
        assert policy.should_auto_context("Jarvis, play some jazz") is False

    def test_context_is_bounded(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        for i in range(20):
            run_sync(
                provider.store(
                    MemoryEntry(content=f"fact number {i}", category="general")
                )
            )
        manager = MemoryManager(provider, MemoryPolicy())
        context = run_sync(manager.build_context("facts"))
        assert len(context) <= MAX_MEMORY_CONTEXT_CHARS
        run_sync(provider.close())

    def test_context_returns_empty_when_no_memory(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        manager = MemoryManager(provider, MemoryPolicy())
        context = run_sync(manager.build_context("nothing relevant"))
        assert context == ""
        run_sync(provider.close())


class TestRealPersistence:
    def test_persistence_round_trip(self, db_path):
        token = "MEMORY_PERSISTENCE_TEST_2026"
        first = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(first.initialize())
        stored = run_sync(first.store(MemoryEntry(content=token, category="test")))
        run_sync(first.close())

        second = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(second.initialize())
        fetched = run_sync(second.get(stored.id))
        assert fetched is not None
        assert fetched.content == token
        run_sync(second.delete(stored.id))
        assert run_sync(second.get(stored.id)) is None
        run_sync(second.close())


class TestHermesMemoryIsolation:
    def test_hermes_integration_has_no_memory_write_tools(self):
        from agents.hermes_agent import HermesIntegration

        integration = HermesIntegration(gate=_ActiveGate())
        assert "remember_memory" not in integration.tool_names
        assert "forget_memory" not in integration.tool_names

    def test_memory_status_is_read_only_in_conversation(self, db_path):
        integration = _make_memory_integration(LocalSQLiteMemoryProvider(path=db_path))
        assert int(integration.tool_level("memory_status")) == 1

    @pytest.mark.asyncio
    async def test_hermes_reads_bounded_memory_context(self, db_path):
        from developer.coding_agent import Developer

        provider = LocalSQLiteMemoryProvider(path=db_path)
        await provider.initialize()
        await provider.store(
            MemoryEntry(
                content="the launch colour is coral",
                category="preference",
                source="voice",
            )
        )
        manager = MemoryManager(provider, MemoryPolicy())
        developer = Developer(project=None, hermes=None, memory=manager)
        context = await developer._maybe_memory_context("coral branding")
        assert "coral" in context
        assert len(context) <= MAX_MEMORY_CONTEXT_CHARS
        await provider.close()

    @pytest.mark.asyncio
    async def test_memory_context_empty_when_no_memory(self, db_path):
        from developer.coding_agent import Developer

        provider = LocalSQLiteMemoryProvider(path=db_path)
        await provider.initialize()
        manager = MemoryManager(provider, MemoryPolicy())
        developer = Developer(project=None, hermes=None, memory=manager)
        context = await developer._maybe_memory_context("nothing stored")
        assert context == ""
        await provider.close()


def _make_memory_integration(provider, gate=None):

    return MemoryIntegration(gate=gate or _ActiveGate(), provider=provider)


def hunter2_absent(text: str) -> bool:
    return "hunter2" not in text


class TestRuntimePath:
    """Regression: the runtime DB path must be absolute/project-anchored.

    The write and the recall must hit the exact same SQLite file, even after a
    full process restart, regardless of where the agent process is launched
    from. A CWD-relative path like ``data/jarvis_memory.db`` silently points to
    different files depending on the launch directory, which loses memory.
    """

    def test_default_memory_db_path_is_absolute(self):
        from pathlib import Path

        from memory.config import default_memory_db_path

        path = Path(default_memory_db_path())
        assert path.is_absolute()
        assert path.name == "jarvis_memory.db"
        assert "data" in path.parts

    def test_default_memory_db_path_is_project_anchored(self):
        from pathlib import Path

        from developer.project_tools import PROJECT_ROOT
        from memory.config import default_memory_db_path

        path = Path(default_memory_db_path())
        assert PROJECT_ROOT in path.parents or path.parent.parent == PROJECT_ROOT

    def test_default_memory_db_path_is_stable_across_launch_dir(self, monkeypatch):
        from pathlib import Path

        from memory.config import default_memory_db_path

        baseline = default_memory_db_path()
        monkeypatch.chdir(Path(__import__("tempfile").gettempdir()))
        assert default_memory_db_path() == baseline

    def test_registry_tools_expose_memory_tools(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        registry = build_default_registry(memory_provider=provider)

        def _tool_names():
            for tool in registry.tools():
                info = getattr(tool, "info", None)
                if info is not None:
                    yield info.name
                elif hasattr(tool, "name"):
                    yield tool.name

        names = set(_tool_names())
        assert {
            "remember_memory",
            "search_memory",
            "forget_memory",
            "memory_status",
        } <= names


class TestGeminiRoutingGuidance:
    """Regression: the live agent prompt must tell Gemini when to use memory.

    With no memory guidance in the instructions, Gemini has no reason to call
    the write tool on "remember ..." or the search tool on a recall question,
    so nothing is ever stored or retrieved at runtime.
    """

    def test_instructions_tell_gemini_to_use_write_tool(self):
        from memory.prompt import memory_instructions

        text = memory_instructions()
        assert "remember_memory" in text
        assert "search_memory" in text

    def test_instructions_only_write_on_explicit_request(self):
        from memory.prompt import memory_instructions

        text = memory_instructions().casefold()
        assert "explicitly ask" in text
        assert "never" in text

    def test_instructions_search_before_answering_recall(self):
        from memory.prompt import memory_instructions

        text = memory_instructions().casefold()
        assert "before answering" in text
        assert "what colour did i ask you" in text or "ask you to remember" in text

    def test_agent_prompt_actually_includes_memory_instructions(self):
        from pathlib import Path

        source = (Path(__file__).parents[1] / "src" / "agent.py").read_text(
            encoding="utf-8"
        )
        assert "memory_instructions()" in source

    def test_agent_wires_deterministic_router_into_final_transcripts(self):
        from pathlib import Path

        source = (Path(__file__).parents[1] / "src" / "agent.py").read_text(
            encoding="utf-8"
        )
        assert "memory_router = MemoryRouter(" in source
        assert "_route_memory" in source
        assert "ev.is_final" in source
        assert "MemoryIntent.NONE" in source

    def test_agent_logs_runtime_observability_lines(self):
        from pathlib import Path

        source = (Path(__file__).parents[1] / "src" / "agent.py").read_text(
            encoding="utf-8"
        )
        assert "MEMORY DB: " in source
        assert "MEMORY HEALTH: " in source
        assert "JARVIS ACTIVE TOOLS: " in source
        assert "TOOL CALL: " in source
        assert "TOOL RESULT STATUS: " in source
        assert "LIVEKIT-AGENTS VERSION: " in source
        assert "GOOGLE PLUGIN VERSION: " in source

    def test_runtime_agent_writes_then_recalls_same_file_after_restart(self, tmp_path):
        path = str(tmp_path / "jarvis_memory.db")
        first = LocalSQLiteMemoryProvider(path=path)
        gate1 = ToolGate()
        gate1.set_user_request("Jarvis, remember that my test colour is cobalt blue")
        integration1 = MemoryIntegration(gate=gate1, provider=first)
        stored = run_sync(
            integration1.remember_memory(None, content="my test colour is cobalt blue")
        )
        assert stored["status"] == "stored"
        run_sync(first.close())

        second = LocalSQLiteMemoryProvider(path=path)
        gate2 = ToolGate()
        gate2.set_user_request("Jarvis, what colour did I ask you to remember?")
        integration2 = MemoryIntegration(gate=gate2, provider=second)
        result = run_sync(integration2.search_memory(None, query="cobalt colour"))
        assert result["status"] == "ok"
        assert any("cobalt blue" in r["content"] for r in result["results"])
        run_sync(second.close())


class TestDeterministicRouter:
    """STEP-12 regressions: explicit write/recall must route in code.

    Correctness-critical memory commands must never depend on Gemini choosing a
    tool. The router classifies the utterance, extracts the fact or keyword,
    and persists/searches in code. A plain session plus router therefore shows
    the same write -> restart -> recall outcome as the LLM-driven tools.
    """

    def test_explicit_write_does_not_need_an_llm(self, db_path):
        from memory.router import MemoryRouter

        provider = LocalSQLiteMemoryProvider(path=db_path)
        router = MemoryRouter(provider=provider, policy=MemoryPolicy())

        result = run_sync(
            router.route("Jarvis, remember that my test colour is cobalt blue")
        )
        assert result.is_write
        assert result.accepted
        assert result.content == "my test colour is cobalt blue"
        assert result.entry_id
        assert result.count == 0
        run_sync(provider.close())

    def test_explicit_recall_searches_without_an_llm(self, db_path):
        from memory.router import MemoryRouter

        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        run_sync(
            provider.store(
                MemoryEntry(content="my test colour is cobalt blue", category="general")
            )
        )
        router = MemoryRouter(provider=provider, policy=MemoryPolicy())

        result = run_sync(
            router.route("Jarvis, what colour did I ask you to remember?")
        )
        assert result.is_recall
        assert result.accepted
        assert result.content == "colour"
        assert result.count == 1
        assert result.results[0].content == "my test colour is cobalt blue"
        run_sync(provider.close())

    def test_recall_query_strips_boilerplate(self):
        from memory.router import extract_recall_query

        assert (
            extract_recall_query("Jarvis, what colour did I ask you to remember?")
            == "colour"
        )
        assert (
            extract_recall_query("Jarvis, do you remember my test colour")
            == "test colour"
        )
        assert extract_recall_query("Jarvis, what did I ask you to remember") == ""
        assert extract_recall_query("Jarvis, which colour should I buy") == "colour buy"

    def test_dutch_write_extracts_the_fact(self, db_path):
        from memory.router import MemoryRouter

        provider = LocalSQLiteMemoryProvider(path=db_path)
        router = MemoryRouter(provider=provider, policy=MemoryPolicy())

        result = run_sync(
            router.route("Jarvis, onthoud dat mijn testkleur kobaltblauw is")
        )
        assert result.is_write
        assert result.accepted
        assert result.language == "nl"
        assert result.content == "mijn testkleur kobaltblauw is"
        run_sync(provider.close())

    def test_dutch_recall_retrieves_across_languages(self, db_path):
        from memory.router import MemoryRouter

        provider = LocalSQLiteMemoryProvider(path=db_path)
        run_sync(provider.initialize())
        run_sync(
            provider.store(
                MemoryEntry(content="my test colour is cobalt blue", category="general")
            )
        )
        router = MemoryRouter(provider=provider, policy=MemoryPolicy())

        result = run_sync(
            router.route("Jarvis, welke kleur moest je van mij onthouden?")
        )
        assert result.is_recall
        assert result.accepted
        assert "colour" in result.content
        assert "kleur" in result.content
        assert any("cobalt blue" in e.content for e in result.results)
        run_sync(provider.close())

    def test_wake_gate_still_blocks_write_while_asleep(self, db_path):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        gate = ToolGate()

        gate.set_user_request("place the delivery order")
        with pytest.raises(ToolError, match="wake word"):
            gate.ensure_memory_requested()

        assert run_sync(provider.list_all()) == []
        run_sync(provider.close())

    def test_secret_policy_blocks_deterministic_write(self, db_path):
        from memory.router import MemoryRouter

        provider = LocalSQLiteMemoryProvider(path=db_path)
        router = MemoryRouter(provider=provider, policy=MemoryPolicy())

        result = run_sync(router.route("Jarvis, remember my password is hunter2"))
        assert result.is_write
        assert not result.accepted
        assert "secret" in result.reason or "credential" in result.reason
        assert run_sync(provider.list_all()) == []
        run_sync(provider.close())

    def test_normal_conversation_bypasses_memory(self, db_path):
        from memory.router import MemoryIntent, MemoryRouter

        provider = LocalSQLiteMemoryProvider(path=db_path)
        router = MemoryRouter(provider=provider, policy=MemoryPolicy())

        for text in (
            "Jarvis, what is five plus five?",
            "Jarvis, open Spotify",
            "Jarvis, what's the weather like?",
            "Jarvis, tell me a joke",
        ):
            assert router.detect_intent(text) is MemoryIntent.NONE
        assert run_sync(provider.list_all()) == []
        run_sync(provider.close())

    def test_write_then_new_provider_process_then_recall(self, tmp_path):
        from memory.router import MemoryRouter

        path = str(tmp_path / "jarvis_memory.db")
        first = LocalSQLiteMemoryProvider(path=path)
        router1 = MemoryRouter(provider=first, policy=MemoryPolicy())
        write = run_sync(
            router1.route("Jarvis, remember that my test colour is cobalt blue")
        )
        assert write.accepted
        run_sync(first.close())

        second = LocalSQLiteMemoryProvider(path=path)
        router2 = MemoryRouter(provider=second, policy=MemoryPolicy())
        recall = run_sync(
            router2.route("Jarvis, what colour did I ask you to remember?")
        )
        assert recall.is_recall
        assert recall.accepted
        assert any("cobalt blue" in e.content for e in recall.results)
        run_sync(second.close())

    def test_final_active_tool_sets_contain_memory_tools(self, db_path):
        from integrations import build_default_registry

        provider = LocalSQLiteMemoryProvider(path=db_path)
        registry = build_default_registry(memory_provider=provider)

        tools = registry.tools()
        names = set()
        for tool in tools:
            info = getattr(tool, "info", None)
            if info is not None:
                names.add(info.name)
            elif hasattr(tool, "name"):
                names.add(tool.name)

        assert {
            "remember_memory",
            "search_memory",
            "forget_memory",
            "memory_status",
            "get_memory",
            "list_memories",
            "update_memory",
        } <= names
        run_sync(provider.close())


class TestAgentWiringDeterministicRouter:
    """The wired `_route_memory` path must persist/recall in code and steer a reply.

    `Assistant._route_memory` is the live hook (final transcripts). It must
    write/recall deterministically without any LLM, and push the outcome into
    the reply path through `generate_reply(instructions=...)`. These tests
    exercise the real method on a real Assistant instance with a stubbed
    session, so no live room or Google credentials are required.
    """

    def _make_assistant(self, monkeypatch, provider):
        import src.agent as agent_module
        from memory.router import MemoryRouter

        router = MemoryRouter(provider=provider, policy=MemoryPolicy())
        monkeypatch.setattr(agent_module, "memory_router", router)

        assistant = agent_module.Assistant.__new__(agent_module.Assistant)
        assistant._gate = ToolGate()
        assistant._memory_tasks = set()

        class _FakeSession:
            def __init__(self):
                self.instructions = []

            async def generate_reply(self, *, instructions):
                self.instructions.append(instructions)

        fake = _FakeSession()

        class _FakeActivity:
            session = fake

        monkeypatch.setattr(
            assistant, "_get_activity_or_raise", lambda: _FakeActivity()
        )
        return assistant, fake

    def test_write_persists_in_code_and_steers_confirmation(self, db_path, monkeypatch):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        assistant, fake = self._make_assistant(monkeypatch, provider)

        run_sync(
            assistant._route_memory(
                "Jarvis, remember that my test colour is cobalt blue"
            )
        )

        rows = run_sync(provider.list_all())
        assert len(rows) == 1
        assert "cobalt blue" in rows[0].content
        assert fake.instructions and "saved" in fake.instructions[0].casefold()
        run_sync(provider.close())

    def test_recall_replies_with_stored_fact(self, db_path, monkeypatch):
        from memory.router import MemoryRouter

        provider = LocalSQLiteMemoryProvider(path=db_path)
        seed = MemoryRouter(provider=provider, policy=MemoryPolicy())
        write = run_sync(
            seed.route("Jarvis, remember that my test colour is cobalt blue")
        )
        assert write.accepted
        assistant, fake = self._make_assistant(monkeypatch, provider)

        run_sync(
            assistant._route_memory("Jarvis, what colour did I ask you to remember?")
        )

        assert fake.instructions, "expected the reply to be steered with the fact"
        assert "cobalt blue" in fake.instructions[0]
        run_sync(provider.close())

    def test_asleep_gate_blocks_routing(self, db_path, monkeypatch):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        assistant, fake = self._make_assistant(monkeypatch, provider)

        run_sync(assistant._route_memory("play some music"))

        assert run_sync(provider.list_all()) == []
        assert fake.instructions == []
        run_sync(provider.close())

    def test_secret_write_is_refused_without_llm(self, db_path, monkeypatch):
        provider = LocalSQLiteMemoryProvider(path=db_path)
        assistant, fake = self._make_assistant(monkeypatch, provider)

        run_sync(
            assistant._route_memory("Jarvis, remember that my password is hunter2")
        )

        assert run_sync(provider.list_all()) == []
        assert fake.instructions and "cannot store" in fake.instructions[0].casefold()
        run_sync(provider.close())

    def test_final_transcript_event_persists_via_handler(self, db_path, monkeypatch):
        """Regression: the exact sync callback the SDK fires on final voice
        transcripts must schedule the deterministic route and persist a row.

        The production failure was upstream of this handler: the memory tools
        declared an untyped ``context`` parameter, so the Google realtime
        session died at startup while building tool schemas (KeyError
        'context'), no transcript ever reached this handler, and no spoken
        memory command ever landed in SQLite. Every registered tool must now
        build a realtime schema and the handler must persist a row end to end.
        """
        from livekit.agents import UserInputTranscribedEvent

        import src.agent as agent_module

        provider = LocalSQLiteMemoryProvider(path=db_path)
        assistant, fake = self._make_assistant(monkeypatch, provider)
        monkeypatch.setattr(agent_module, "MEMORY_ROUTING_LIVE", True)

        async def drive_final_transcript() -> None:
            assistant._on_user_input_transcribed(
                UserInputTranscribedEvent(
                    transcript="Jarvis, remember that my test colour is cobalt blue",
                    is_final=True,
                )
            )
            assert assistant._memory_tasks, "final transcript must schedule routing"
            await asyncio.gather(*assistant._memory_tasks)

        run_sync(drive_final_transcript())

        rows = run_sync(provider.list_all())
        assert len(rows) == 1
        assert "cobalt blue" in rows[0].content
        assert fake.instructions and "saved" in fake.instructions[0].casefold()

        from livekit.agents.llm import utils

        for tool in agent_module.capability_registry.tools():
            utils.build_legacy_openai_schema(tool, internally_tagged=True)
        run_sync(provider.close())

    def test_memory_routing_off_cannot_touch_the_reply_path(self, db_path, monkeypatch):
        """Voice-first regression: with live memory routing off (the shipped
        default), a final spoken transcript must never spawn background routing
        or steer the reply, so the memory integration cannot block, time out,
        or duplicate realtime voice generation."""
        from livekit.agents import UserInputTranscribedEvent

        import src.agent as agent_module

        assert agent_module.MEMORY_ROUTING_LIVE is False

        provider = LocalSQLiteMemoryProvider(path=db_path)
        assistant, fake = self._make_assistant(monkeypatch, provider)

        assistant._on_user_input_transcribed(
            UserInputTranscribedEvent(
                transcript="Jarvis, remember that my test colour is cobalt blue",
                is_final=True,
            )
        )

        assert not assistant._memory_tasks
        assert fake.instructions == []
        assert run_sync(provider.list_all()) == []
        run_sync(provider.close())
