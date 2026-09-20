"""Hermes backend agent package.

Hermes is JARVIS's specialist backend agent for deep planning and development
work. It is always optional: JARVIS starts and runs normally whether or not
Hermes is installed. The adapter detects a real interface at runtime
(``HERMES_COMMAND`` or a ``hermes`` executable on ``PATH``) and every call runs
as an isolated, timed-out subprocess with failure isolation.
"""

from agents.hermes_agent import HermesAgent, HermesIntegration, HermesResult
from agents.router import HermesRouter, Route

__all__ = [
    "HermesAgent",
    "HermesIntegration",
    "HermesResult",
    "HermesRouter",
    "Route",
]
