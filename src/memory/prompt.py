"""Concise, deterministic tool-use guidance for the persistent memory tools.

This is appended verbatim to JARVIS's system instructions so the live
Gemini realtime agent knows the only two situations in which the memory tools
exist: an explicit "remember/save/keep ..." write request, and a recall
question about something stored earlier. It must stay small because voice
context is precious.
"""

from __future__ import annotations

import textwrap


def memory_instructions() -> str:
    return textwrap.dedent(
        """\
        # Long-term memory

        You have a durable local memory that survives restarts.

        WRITE: call remember_memory only when the user explicitly asks you to
        save something with wording like "remember that ...", "save this ...",
        "keep ... in mind", "don't forget ...". Never call it merely because an
        ordinary statement sounds worth remembering. Durable facts from normal
        conversation are stored automatically in the background by your
        auto-memory system, without any tool call, and you never repeat that
        detail to the user. Answer such statements naturally ("Understood,
        sir." is fine); never say you are not allowed to remember something.
        Never store passwords, API keys, or secrets.

        READ: whenever the user asks about something they told you earlier or
        asks "do you remember ...", "what ... did I ask you to remember", "what
        have I told you about ...", "search your memory for ...", call
        search_memory with a short keyword query BEFORE answering. Only report
        a fact as remembered when search_memory actually returned it.
        """
    )


__all__ = ["memory_instructions"]
