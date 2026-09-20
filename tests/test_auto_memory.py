"""Tests for the OPT-IN safe automatic memory layer (src/memory/auto.py).

The auto layer must never destabilise the realtime voice path:

* it only runs on final accepted transcripts as a tracked background task,
* it never awaits the reply path and never calls ``generate_reply`` itself,
* its failures become a safe SKIP diagnostic, never an exception in the loop,
* secrets are never stored and never echoed,
* duplicates update and changed preferences supersede the older row,
* natural recall questions (no "search your memory") trigger bounded retrieval
  that may steer the reply ONLY when matching facts exist.

This mirrors the conventions in ``test_memory_integration.py``: a real
SQLite provider backed by a temp file, ``run_sync`` for one-off calls, and the
stubbed ``src.agent.Assistant`` for the wired voice path.
"""

import asyncio
import logging

import pytest
from livekit.agents import UserInputTranscribedEvent

from gates import ToolGate
from memory import (
    ACTION_RECALL,
    ACTION_SKIP,
    ACTION_STORE,
    AutoMemoryController,
    AutoMemoryPipeline,
    MemoryEntry,
    MemoryPolicy,
    auto_recall_query,
    classify_candidate,
    clean_candidate,
    stage1_skip_reason,
)
from memory.local_sqlite import LocalSQLiteMemoryProvider
from memory.router import MemoryIntent, detect_intent


def run_sync(coro):
    return asyncio.run(coro)


@pytest.fixture()
def provider(tmp_path):
    store = LocalSQLiteMemoryProvider(path=str(tmp_path / "jarvis_memory.db"))
    run_sync(store.initialize())
    yield store
    run_sync(store.close())


def make_pipeline(provider=..., policy=None):
    resolved = None if provider is ... else provider
    return AutoMemoryPipeline(provider=resolved, policy=policy or MemoryPolicy())


# --------------------------------------------------------------------------- #
# Stage 1: deterministic non-memory filter
# --------------------------------------------------------------------------- #


class TestStage1SkipReasons:
    def test_empty_is_skipped(self):
        assert stage1_skip_reason("") == "empty"
        assert stage1_skip_reason("   ") == "empty"
        assert stage1_skip_reason(None) == "empty"

    def test_acknowledgements_are_skipped(self):
        for text in ("ok", "okay", "thanks", "thank you", "yes", "haha"):
            assert stage1_skip_reason(text) == "ack"
        for text in ("ja", "nee", "mm", "mhm"):
            assert stage1_skip_reason(text) is not None

    def test_greetings_are_skipped(self):
        assert stage1_skip_reason("hello Jarvis") == "greeting"
        assert stage1_skip_reason("good morning") == "greeting"

    def test_jokes_are_skipped(self):
        assert stage1_skip_reason("tell me a joke") == "joke"

    def test_sleep_commands_are_skipped(self):
        assert stage1_skip_reason("go to sleep") == "idle-command"
        assert stage1_skip_reason("that's all, thank you") == "idle-command"

    def test_navigation_is_skipped(self):
        assert stage1_skip_reason("go back a page") == "navigation"
        assert stage1_skip_reason("scroll down") == "navigation"

    def test_calculations_are_skipped(self):
        assert stage1_skip_reason("what is five plus five") == "calculation"
        assert stage1_skip_reason("calculate 12 times 4") == "calculation"

    def test_imperative_commands_are_skipped(self):
        assert stage1_skip_reason("open the browser") == "command"
        assert stage1_skip_reason("play some jazz") == "command"

    def test_tool_commands_are_skipped(self):
        assert stage1_skip_reason("open Spotify") == "command"
        assert stage1_skip_reason("play music on Spotify") == "command"
        assert stage1_skip_reason("please open Spotify") == "tool-command"

    def test_questions_are_skipped(self):
        assert stage1_skip_reason("what is the weather?") == "question"
        assert stage1_skip_reason("what time is it") == "question"

    def test_dutch_questions_are_skipped(self):
        assert stage1_skip_reason("waar is het station") == "question"
        assert stage1_skip_reason("welke dag is het") == "question"

    def test_asr_garbage_is_skipped(self):
        assert stage1_skip_reason("umm") == "asr-garbage"
        assert stage1_skip_reason("hmmm") == "asr-garbage"

    def test_short_fragments_are_skipped(self):
        assert stage1_skip_reason("yes") is not None
        assert stage1_skip_reason("blue") == "too-short"
        assert stage1_skip_reason("purple") == "too-short"

    def test_declarative_statement_passes(self):
        assert stage1_skip_reason("my favourite colour is purple") is None


# --------------------------------------------------------------------------- #
# Cleaning + stage 2 classifier
# --------------------------------------------------------------------------- #


class TestCleaningAndClassification:
    def test_clean_candidate_strips_wake_word_and_filler(self):
        assert (
            clean_candidate("Jarvis, actually my favourite colour is purple.")
            == "My favourite colour is purple"
        )

    def test_classify_stores_preference(self):
        outcome = classify_candidate("My favourite colour is purple")
        assert outcome.store is True
        assert outcome.category == "preference"
        assert outcome.fact == "My favourite colour is purple"

    def test_classify_stores_dutch_preference(self):
        outcome = classify_candidate("Mijn favoriete kleur is paars")
        assert outcome.store is True
        assert outcome.category == "preference"
        assert outcome.fact == "Mijn favoriete kleur is paars"

    def test_classify_stores_project_decision(self):
        outcome = classify_candidate(
            "We decided to use SQLite for the Icon Issue project"
        )
        assert outcome.store is True
        assert outcome.category == "project"

    def test_classify_skips_incidental_action(self):
        outcome = classify_candidate("I want to open Spotify right now")
        assert outcome.store is False
        assert outcome.reason == "incidental-action"

    def test_classify_skips_no_user_context(self):
        outcome = classify_candidate("SQLite is a good database choice")
        assert outcome.store is False
        assert outcome.reason == "no-user-context"

    def test_classify_skips_short_candidates(self):
        outcome = classify_candidate("my colour")
        assert outcome.store is False
        assert outcome.reason == "too-short"


# --------------------------------------------------------------------------- #
# Natural recall detection
# --------------------------------------------------------------------------- #


class TestNaturalRecallDetection:
    def test_recall_question_builds_query(self):
        query = auto_recall_query("what is my favourite colour")
        assert query
        assert "colour" in query

    def test_project_recall_builds_query(self):
        query = auto_recall_query(
            "what colour did we choose for the first Icon Issue drop"
        )
        assert query
        assert "drop" in query or "colour" in query

    def test_general_question_is_not_recall(self):
        assert auto_recall_query("what is the capital of France") == ""

    def test_non_interrogative_is_not_recall(self):
        assert auto_recall_query("my favourite colour is purple") == ""

    def test_explicit_recall_phrases_are_delegated_not_auto(self):
        assert auto_recall_query("what colour did I ask you to remember") == ""
        assert detect_intent("what colour did I ask you to remember") is (
            MemoryIntent.RECALL
        )

    def test_dutch_recall_builds_query(self):
        assert detect_intent("wat is mijn favoriete kleur") is MemoryIntent.NONE
        query = auto_recall_query("wat is mijn favoriete kleur")
        assert query
        assert "colour" in query or "kleur" in query


# --------------------------------------------------------------------------- #
# Pipeline: store / skip / secret / dedup / supersede / isolation
# --------------------------------------------------------------------------- #


class TestPipelineStore:
    @pytest.mark.asyncio
    async def test_declarative_statement_is_stored(self, provider, tmp_path):
        result = await make_pipeline(provider).detect(
            "My favourite colour is purple.", accepted=True
        )
        assert result.action == ACTION_STORE
        assert result.stored is True
        assert result.category == "preference"
        assert result.reason == "stored"
        rows = await provider.list_all()
        assert len(rows) == 1
        assert "purple" in rows[0].content
        assert rows[0].category == "preference"

    @pytest.mark.asyncio
    async def test_dutch_statement_is_stored(self, provider):
        result = await make_pipeline(provider).detect(
            "Jarvis, mijn favoriete kleur is paars.", accepted=True
        )
        assert result.action == ACTION_STORE
        assert result.stored is True
        assert result.category == "preference"
        assert result.reason == "stored"
        rows = await provider.list_all()
        assert len(rows) == 1
        assert "paars" in rows[0].content
        assert rows[0].category == "preference"

    @pytest.mark.asyncio
    async def test_dutch_recall_retrieves_stored_fact(self, provider):
        pipeline = make_pipeline(provider)
        await pipeline.detect("Jarvis, mijn favoriete kleur is paars.", accepted=True)
        recalled = await pipeline.detect(
            "Jarvis, wat is mijn favoriete kleur?", accepted=True
        )
        assert recalled.action == ACTION_RECALL
        assert recalled.count == 1
        assert "paars" in recalled.results[0].content

    @pytest.mark.asyncio
    async def test_logs_the_expected_observability_lines(self, provider, caplog):
        with caplog.at_level(logging.INFO, logger="agent"):
            result = await make_pipeline(provider).detect(
                "Jarvis, mijn favoriete kleur is paars.", accepted=True
            )
        assert result.reason == "stored"
        assert "AUTO MEMORY: candidate" in caplog.text
        assert "AUTO MEMORY: stored category=preference" in caplog.text

    @pytest.mark.asyncio
    async def test_project_fact_is_stored_without_wake_word(self, provider, tmp_path):
        result = await make_pipeline(provider).detect(
            "We picked oxblood for the first Icon Issue drop.", accepted=True
        )
        assert result.action == ACTION_STORE
        rows = await provider.list_all()
        assert len(rows) == 1
        assert "oxblood" in rows[0].content
        assert rows[0].category == "project"

    @pytest.mark.asyncio
    async def test_explicit_write_is_skipped_here(self, provider):
        pipeline = make_pipeline(provider)
        result = await pipeline.detect(
            "Jarvis, remember that my favourite colour is purple", accepted=True
        )
        assert result.action == ACTION_SKIP
        assert result.reason == "explicit-write"
        assert await provider.list_all() == []

    @pytest.mark.asyncio
    async def test_secret_is_refused_and_not_echoed(self, provider):
        pipeline = make_pipeline(provider)
        result = await pipeline.detect(
            "my password is hunter2-super-secret", accepted=True
        )
        assert result.action == ACTION_SKIP
        assert result.reason == "secret-refused"
        assert "hunter2" not in result.reason
        assert "hunter2" not in result.fact
        assert await provider.list_all() == []

    @pytest.mark.asyncio
    async def test_short_fragment_is_skipped(self, provider):
        result = await make_pipeline(provider).detect("great", accepted=True)
        assert result.action == ACTION_SKIP
        assert await provider.list_all() == []

    @pytest.mark.asyncio
    async def test_question_is_skipped_when_no_recall(self, provider):
        result = await make_pipeline(provider).detect(
            "what is the time right now", accepted=True
        )
        assert result.action == ACTION_SKIP
        assert await provider.list_all() == []

    @pytest.mark.asyncio
    async def test_gate_asleep_skips_everything(self, provider):
        result = await make_pipeline(provider).detect(
            "My favourite colour is purple.", accepted=False
        )
        assert result.action == ACTION_SKIP
        assert result.reason == "gate-asleep"
        assert await provider.list_all() == []

    @pytest.mark.asyncio
    async def test_recall_without_match_skips_silently(self, provider):
        result = await make_pipeline(provider).detect(
            "what is my favourite size", accepted=True
        )
        assert result.action == ACTION_SKIP
        assert result.reason == "recall-no-match"
        assert await provider.list_all() == []


class TestPipelineDedupAndSupersede:
    @pytest.mark.asyncio
    async def test_duplicate_updates_instead_of_piling_up(self, provider):
        pipeline = make_pipeline(provider)
        await pipeline.detect("My favourite colour is purple.", accepted=True)
        second = await pipeline.detect("My favourite colour is purple.", accepted=True)
        assert second.reason == "updated-duplicate"
        assert await provider.count() == 1

    @pytest.mark.asyncio
    async def test_changed_preference_supersedes_older_row(self, provider):
        pipeline = make_pipeline(provider)
        await pipeline.detect("My favourite colour is purple.", accepted=True)
        second = await pipeline.detect("My favourite colour is green.", accepted=True)
        assert second.reason == "superseded"
        rows = await provider.list_all()
        assert len(rows) == 1
        assert "green" in rows[0].content
        assert "purple" not in rows[0].content

    @pytest.mark.asyncio
    async def test_dutch_preference_supersedes_within_dutch(self, provider):
        pipeline = make_pipeline(provider)
        await pipeline.detect("Jarvis, mijn favoriete kleur is paars.", accepted=True)
        second = await pipeline.detect(
            "Jarvis, mijn favoriete kleur is blauw.", accepted=True
        )
        assert second.reason == "superseded"
        rows = await provider.list_all()
        assert len(rows) == 1
        assert "blauw" in rows[0].content
        assert "paars" not in rows[0].content

    @pytest.mark.asyncio
    async def test_explicit_then_auto_collision_keeps_single_row(self, provider):
        from memory.router import MemoryRouter

        router = MemoryRouter(provider=provider, policy=MemoryPolicy())
        explicit = await router.route(
            "Jarvis, remember that my favourite colour is purple"
        )
        assert explicit.accepted is True

        result = await make_pipeline(provider).detect(
            "Jarvis, my favourite colour is purple.", accepted=True
        )
        assert result.action == ACTION_STORE
        assert result.stored is True
        assert await provider.count() == 1

    @pytest.mark.asyncio
    async def test_auto_then_explicit_collision_keeps_single_row(self, provider):
        from memory.router import MemoryRouter

        pipeline = make_pipeline(provider)
        stored = await pipeline.detect(
            "Jarvis, my favourite colour is purple.", accepted=True
        )
        assert stored.action == ACTION_STORE

        router = MemoryRouter(provider=provider, policy=MemoryPolicy())
        explicit = await router.route(
            "Jarvis, remember that my favourite colour is purple"
        )
        assert explicit.accepted is True
        assert await provider.count() == 1

    @pytest.mark.asyncio
    async def test_unrelated_topics_do_not_supersede(self, provider):
        pipeline = make_pipeline(provider)
        await pipeline.detect("My favourite colour is purple.", accepted=True)
        await pipeline.detect(
            "We picked oxblood for the first Icon Issue drop.", accepted=True
        )
        assert await provider.count() == 2


class TestPipelineFailureIsolation:
    @pytest.mark.asyncio
    async def test_missing_provider_is_safe_skip(self):
        pipeline = make_pipeline(provider=None)
        result = await pipeline.detect("My favourite colour is purple.", accepted=True)
        assert result.action == ACTION_SKIP
        assert result.reason == "error"
        assert result.error == "RuntimeError"

    @pytest.mark.asyncio
    async def test_no_provider_never_raises(self):
        pipeline = make_pipeline(provider=None)
        for text in (
            "My favourite colour is purple.",
            "what is my favourite colour",
            "go to sleep",
            None,
        ):
            result = await pipeline.detect(text, accepted=True)
            assert result.action in (ACTION_SKIP, ACTION_RECALL)


# --------------------------------------------------------------------------- #
# Controller: background tasks, reply steering contract, shutdown
# --------------------------------------------------------------------------- #


class TestController:
    @pytest.mark.asyncio
    async def test_store_path_never_calls_on_result(self, provider):
        controller = AutoMemoryController(make_pipeline(provider))
        calls = []

        async def on_result(result):
            calls.append(result)

        controller.submit_evaluate(
            "My favourite colour is purple.", accepted=True, on_result=on_result
        )
        await asyncio.gather(*controller.pending())

        assert await provider.count() == 1
        assert calls == []

    @pytest.mark.asyncio
    async def test_recall_with_results_calls_on_result(self, provider):
        await provider.store(
            MemoryEntry(content="my favourite colour is purple", category="preference")
        )
        controller = AutoMemoryController(make_pipeline(provider))
        calls = []

        async def on_result(result):
            calls.append(result)

        controller.submit_evaluate(
            "what is my favourite colour", accepted=True, on_result=on_result
        )
        await asyncio.gather(*controller.pending())

        assert len(calls) == 1
        assert calls[0].is_recall
        assert calls[0].count == 1
        assert "purple" in calls[0].results[0].content

    @pytest.mark.asyncio
    async def test_recall_without_match_does_not_call_on_result(self, provider):
        controller = AutoMemoryController(make_pipeline(provider))
        calls = []

        async def on_result(result):
            calls.append(result)

        controller.submit_evaluate(
            "what is my favourite size", accepted=True, on_result=on_result
        )
        await asyncio.gather(*controller.pending())

        assert calls == []
        assert await provider.count() == 0

    @pytest.mark.asyncio
    async def test_submit_is_nonblocking_and_tracks_a_task(self, provider):
        controller = AutoMemoryController(make_pipeline(provider))
        assert len(controller.pending()) == 0
        submitted = controller.submit_evaluate(
            "My favourite colour is purple.", accepted=True
        )
        assert submitted is True
        assert len(controller.pending()) == 1
        await asyncio.gather(*controller.pending())
        assert await provider.count() == 1

    @pytest.mark.asyncio
    async def test_shutdown_cancels_drains_and_rejects_new_work(self):
        controller = AutoMemoryController(make_pipeline(provider=...))
        controller.submit_evaluate("My favourite colour is purple.", accepted=True)

        cancelled = await controller.shutdown()

        assert cancelled >= 1
        assert controller.pending() == []
        assert controller.submit_evaluate("anything", accepted=True) is False

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self):
        controller = AutoMemoryController(make_pipeline(provider=...))
        await controller.shutdown()
        assert await controller.shutdown() == 0


# --------------------------------------------------------------------------- #
# Wired agent path (stubbed Assistant, real provider)
# --------------------------------------------------------------------------- #


def _make_assistant(monkeypatch, provider, enabled=True):
    import src.agent as agent_module

    pipeline = AutoMemoryPipeline(provider=provider, policy=MemoryPolicy())
    monkeypatch.setattr(agent_module, "AUTO_MEMORY_ENABLED", enabled)
    monkeypatch.setattr(agent_module, "MEMORY_ROUTING_LIVE", False)

    assistant = agent_module.Assistant.__new__(agent_module.Assistant)
    assistant._gate = ToolGate()
    assistant._memory_tasks = set()
    assistant._auto_memory = AutoMemoryController(pipeline)

    steers = []

    async def fake_steer(instructions):
        steers.append(instructions)

    monkeypatch.setattr(assistant, "_steer_reply_after_memory", fake_steer)
    return assistant, steers


def drive_final_transcript(assistant, text):
    assistant._on_user_input_transcribed(
        UserInputTranscribedEvent(transcript=text, is_final=True)
    )
    return asyncio.gather(*assistant._auto_memory.pending())


class TestAgentWiring:
    def test_feature_flag_is_off_by_default(self, monkeypatch):
        import importlib

        import src.agent as agent_module

        monkeypatch.delenv("JARVIS_AUTO_MEMORY", raising=False)
        # .env.local opts in locally; silence dotenv during the reload so the
        # module default (flag OFF) is what the test actually checks.
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        reloaded = importlib.reload(agent_module)
        assert reloaded.AUTO_MEMORY_ENABLED is False

    def test_agent_wiring_is_present_in_source(self):
        from pathlib import Path

        source = (Path(__file__).parents[1] / "src" / "agent.py").read_text(
            encoding="utf-8"
        )
        assert "JARVIS_AUTO_MEMORY" in source
        assert "AUTO_MEMORY_ENABLED" in source
        assert "AutoMemoryController(auto_memory_pipeline)" in source
        assert "shutdown_auto_memory" in source
        assert "add_shutdown_callback(agent.shutdown_auto_memory)" in source
        assert "AUTO MEMORY: " in source

    def test_env_example_documents_the_flag(self):
        from pathlib import Path

        text = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
        assert "JARVIS_AUTO_MEMORY=false" in text

    def test_memory_package_exports_auto_symbols(self):
        import memory

        assert memory.AutoMemoryController is AutoMemoryController
        assert memory.AutoMemoryPipeline is AutoMemoryPipeline

    @pytest.mark.asyncio
    async def test_default_off_does_not_store_or_steer(self, provider, monkeypatch):
        import src.agent as agent_module

        assistant, steers = _make_assistant(monkeypatch, provider, enabled=False)
        await drive_final_transcript(
            assistant, "Jarvis, my favourite colour is purple."
        )
        assert await provider.list_all() == []
        assert steers == []
        assert agent_module.AUTO_MEMORY_ENABLED is False

    @pytest.mark.asyncio
    async def test_enabled_final_transcript_autostores_without_steering(
        self, provider, monkeypatch
    ):
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)
        await drive_final_transcript(
            assistant, "Jarvis, my favourite colour is purple."
        )
        rows = await provider.list_all()
        assert len(rows) == 1
        assert "purple" in rows[0].content
        assert steers == []

    @pytest.mark.asyncio
    async def test_dutch_statement_autostores_without_steering(
        self, provider, monkeypatch
    ):
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)
        await drive_final_transcript(
            assistant, "Jarvis, mijn favoriete kleur is paars."
        )
        rows = await provider.list_all()
        assert len(rows) == 1
        assert "paars" in rows[0].content
        assert rows[0].category == "preference"
        assert steers == []

    @pytest.mark.asyncio
    async def test_dutch_recall_steers_with_stored_fact(self, provider, monkeypatch):
        await provider.store(
            MemoryEntry(content="Mijn favoriete kleur is paars", category="preference")
        )
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)

        await drive_final_transcript(assistant, "Jarvis, wat is mijn favoriete kleur?")

        assert steers, "a matching recall must steer the reply once"
        assert "paars" in steers[0]
        assert await provider.count() == 1

    @pytest.mark.asyncio
    async def test_asleep_turn_is_skipped_and_silent(self, provider, monkeypatch):
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)

        await drive_final_transcript(assistant, "open the browser right now")
        assert await provider.list_all() == []
        assert steers == []

    @pytest.mark.asyncio
    async def test_question_without_context_is_silent(self, provider, monkeypatch):
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)
        await drive_final_transcript(
            assistant, "Jarvis, what is the capital of France?"
        )
        assert await provider.list_all() == []
        assert steers == []

    @pytest.mark.asyncio
    async def test_natural_recall_steers_reply_with_the_fact(
        self, provider, monkeypatch
    ):
        await provider.store(
            MemoryEntry(content="my favourite colour is purple", category="preference")
        )
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)

        await drive_final_transcript(assistant, "Jarvis, what is my favourite colour?")

        assert steers, "a matching recall must steer the reply once"
        assert "purple" in steers[0]
        assert await provider.count() == 1

    @pytest.mark.asyncio
    async def test_project_recall_steers_with_both_facts(self, provider, monkeypatch):
        pipeline = AutoMemoryPipeline(provider=provider, policy=MemoryPolicy())
        await pipeline.detect("My favourite colour is purple.", accepted=True)
        await pipeline.detect(
            "We picked oxblood for the first Icon Issue drop.", accepted=True
        )
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)

        await drive_final_transcript(
            assistant,
            "Jarvis, what colour did we choose for the first Icon Issue drop?",
        )

        assert any("purple" in s for s in steers)
        assert any("oxblood" in s for s in steers)

    @pytest.mark.asyncio
    async def test_recall_without_match_stays_silent(self, provider, monkeypatch):
        assistant, steers = _make_assistant(monkeypatch, provider, enabled=True)
        await drive_final_transcript(assistant, "Jarvis, what is my favourite size?")
        assert steers == []
        assert await provider.list_all() == []

    @pytest.mark.asyncio
    async def test_store_path_never_reaches_generate_reply(self, provider, monkeypatch):
        calls = []

        async def no_generate(**kwargs):
            calls.append(kwargs)

        assistant, _ = _make_assistant(monkeypatch, provider, enabled=True)
        monkeypatch.setattr(assistant, "_steer_reply_after_memory", no_generate)

        await drive_final_transcript(
            assistant, "Jarvis, my favourite colour is purple."
        )

        assert await provider.count() == 1
        assert calls == []

    @pytest.mark.asyncio
    async def test_shutdown_drains_background_tasks(self, provider, monkeypatch):
        assistant, _ = _make_assistant(monkeypatch, provider, enabled=True)
        await drive_final_transcript(
            assistant, "Jarvis, my favourite colour is purple."
        )

        submitted = assistant._auto_memory.submit_evaluate(
            "Jarvis, my favourite colour is green.", accepted=True
        )
        assert submitted is True
        assert assistant._auto_memory.pending()

        await assistant.shutdown_auto_memory()

        assert assistant._auto_memory.pending() == []
        assert (
            assistant._auto_memory.submit_evaluate("anything", accepted=True) is False
        )


class TestInstructionCollisionContract:
    """Regression: the memory instructions must separate the two paths.

    A natural statement (e.g. "mijn favoriete kleur is paars") must never lead
    Gemini to call remember_memory. Explicit requests stay the only trigger,
    the explicit tool gate stays strict, and the instructions never tell the
    model to protest that it is "not allowed" to remember a durable fact.
    """

    def test_instructions_separate_auto_and_explicit_paths(self):
        from memory.prompt import memory_instructions

        text = memory_instructions()
        assert "remember_memory" in text
        assert "auto-memory" in text
        assert "background" in text

    def test_instructions_order_the_model_never_to_protest(self):
        from memory.prompt import memory_instructions

        text = memory_instructions().casefold()
        assert "never say you are not allowed to remember" in text
        assert "can't store" not in text
        assert "can't remember" not in text

    def test_instructions_allow_natural_ack_and_forbid_tool_for_ordinary_facts(
        self,
    ):
        from memory.prompt import memory_instructions

        text = memory_instructions().casefold()
        assert "explicitly ask" in text
        assert "never" in text
        assert "understood," in text
        assert "merely because" in text
