"""Conservative, on-demand screen vision for JARVIS.

This package adds read-only screen capability: listing monitors, reading the
cursor, capturing a monitor or the area around the cursor, and answering text
questions about the screen through a separate, on-demand Gemini request. It is
deliberately decoupled from the realtime voice pipeline: nothing runs
continuously, all blocking work happens on worker threads, and any failure is a
normal tool error that can never crash or stall the agent.
"""

from __future__ import annotations

from integrations.screen_vision.integration import ScreenVisionIntegration

__all__ = ["ScreenVisionIntegration"]
