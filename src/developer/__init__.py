"""Self-development layer for the JARVIS agent.

This package lets the agent plan and carry out safe changes to its own source
code when the user explicitly enters developer mode. Everything executes inside
a sandboxed project root, edits go through a deterministic applier, and all Git
work happens on throwaway ``jarvis-dev/<task>`` branches that are only merged
after explicit user approval.
"""

from developer.coding_agent import Developer, ProposalError
from developer.failures import FailureAnalyzer, FailureLog, redact
from developer.git_manager import GitError, GitManager
from developer.improvement_analyzer import ImprovementAnalyzer
from developer.maintenance import MaintenanceController
from developer.project_tools import (
    PROJECT_ROOT,
    ProjectAccess,
    ProjectAccessError,
)
from developer.task_manager import (
    DevTask,
    TaskActionError,
    TaskManager,
    TaskNotFoundError,
)
from developer.test_runner import CheckResult, TestRunner

__all__ = [
    "PROJECT_ROOT",
    "CheckResult",
    "DevTask",
    "Developer",
    "FailureAnalyzer",
    "FailureLog",
    "GitError",
    "GitManager",
    "ImprovementAnalyzer",
    "MaintenanceController",
    "ProjectAccess",
    "ProjectAccessError",
    "ProposalError",
    "TaskActionError",
    "TaskManager",
    "TaskNotFoundError",
    "TestRunner",
    "redact",
]
