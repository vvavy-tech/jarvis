import asyncio
from urllib.parse import urlencode

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

from browser_tools import BrowserError, BrowserManager
from developer.coding_agent import Developer
from developer.failures import FailureLog
from developer.improvement_analyzer import ImprovementAnalyzer
from developer.integration_scaffold import slugify
from developer.maintenance import MaintenanceController
from developer.task_manager import (
    DevTask,
    TaskActionError,
    TaskManager,
    TaskNotFoundError,
)
from gates import ToolGate


def duckduckgo_search_url(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("The search query cannot be empty.")
    return f"https://duckduckgo.com/?{urlencode({'q': query})}"


class BrowserTools:
    def __init__(
        self,
        browser: BrowserManager,
        gate: ToolGate | None = None,
        failure_log: FailureLog | None = None,
    ) -> None:
        self.browser = browser
        self.gate = gate or ToolGate()
        self._confirmed_target: str | None = None
        self._failure_log = failure_log

    def set_user_request(self, text: str | None) -> None:
        """Record the current user turn on the shared gate.

        Called by the agent when the user's speech starts and when a transcript
        is available, so the gate can authorize replies and tool calls.
        """
        self.gate.set_user_request(text)

    def _log_failure(
        self,
        tool: str,
        target: str,
        exc: Exception,
        fallback: str = "",
    ) -> None:
        if self._failure_log is None:
            return
        self._failure_log.append(
            tool=tool,
            target=target,
            reason=str(exc),
            fallback=fallback,
            fallback_result="",
        )

    @property
    def tools(self) -> list:
        return [
            self.open_url,
            self.search_the_web,
            self.read_page,
            self.inspect_page,
            self.go_back,
            self.take_screenshot,
            self.click,
            self.confirm_browser_action,
            self.type_text,
            self.scroll,
            self.press_key,
        ]

    @function_tool()
    async def search_the_web(
        self,
        context: RunContext,
        query: str,
    ) -> dict[str, str]:
        """Open fallback DuckDuckGo results in the agent-controlled browser.

        Use this only when the user needs a general internet search and did not name a
        website, service, or domain. If the user names a destination, open its official
        URL directly with open_url instead. Read or inspect the resulting page before
        answering the user.

        Args:
            query: A concise DuckDuckGo search query containing all relevant context.
        """
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.open_url(duckduckgo_search_url(query))
        except (BrowserError, ValueError) as exc:
            self._log_failure("search_the_web", query, exc, fallback="open_url")
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def open_url(self, context: RunContext, url: str) -> dict[str, str]:
        """Open a public webpage directly in the agent-controlled browser.

        Prefer this over DuckDuckGo whenever the user names a website, service, domain,
        or specific destination. Use the destination's official URL.

        Args:
            url: A complete http or https URL to open.
        """
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.open_url(url)
        except BrowserError as exc:
            self._log_failure("open_url", url, exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def read_page(self, context: RunContext) -> dict[str, str | bool]:
        """Read the visible text from the current browser page."""
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.read_page()
        except BrowserError as exc:
            self._log_failure("read_page", "", exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def inspect_page(self, context: RunContext) -> dict[str, object]:
        """Inspect the current page, including readable text and interactive element names.

        Use this before clicking or typing so you can choose a visible control by its
        returned name or role.
        """
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.inspect_page()
        except BrowserError as exc:
            self._log_failure("inspect_page", "", exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def go_back(self, context: RunContext) -> dict[str, str]:
        """Go back to the previous page in the agent-controlled browser."""
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.go_back()
        except BrowserError as exc:
            self._log_failure("go_back", "", exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def take_screenshot(self, context: RunContext) -> dict[str, str | int | bool]:
        """Capture the current browser page for diagnostics."""
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.take_screenshot()
        except BrowserError as exc:
            self._log_failure("take_screenshot", "", exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def click(self, context: RunContext, target: str) -> dict[str, str]:
        """Click a visible control by its accessible name.

        Args:
            target: The visible or accessible name of the control to click.
        """
        self.gate.ensure_browser_requested()
        if self._requires_confirmation(target):
            if self._confirmed_target != target.casefold():
                raise ToolError(
                    f"This action may be consequential. Ask the user to confirm clicking {target!r} before retrying."
                )
            self._confirmed_target = None

        try:
            return await self.browser.click(target)
        except BrowserError as exc:
            self._log_failure("click", target, exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def confirm_browser_action(self, context: RunContext, target: str) -> str:
        """Authorize one previously discussed consequential browser click.

        Call this only after the user explicitly confirms the exact action.

        Args:
            target: The exact accessible name of the control the user approved.
        """
        self.gate.ensure_confirmation_allowed()
        self._confirmed_target = target.casefold()
        return f"The user confirmed clicking {target!r}."

    @function_tool()
    async def type_text(
        self,
        context: RunContext,
        target: str,
        text: str,
    ) -> dict[str, str]:
        """Fill a visible text field by its label, placeholder, or accessible name.

        Args:
            target: The label, placeholder, or accessible name of the text field.
            text: The text to enter.
        """
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.type_text(target, text)
        except BrowserError as exc:
            self._log_failure("type_text", target, exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def scroll(self, context: RunContext, direction: str) -> dict[str, str]:
        """Scroll the current browser page up or down.

        Args:
            direction: Either 'up' or 'down'.
        """
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.scroll(direction)  # type: ignore[arg-type]
        except BrowserError as exc:
            self._log_failure("scroll", direction, exc)
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def press_key(self, context: RunContext, key: str) -> dict[str, str]:
        """Press a safe navigation key in the current browser page.

        Args:
            key: One of Enter, Escape, Tab, an arrow key, or Backspace.
        """
        self.gate.ensure_browser_requested()
        try:
            return await self.browser.press_key(key)
        except BrowserError as exc:
            self._log_failure("press_key", key, exc)
            raise ToolError(str(exc)) from exc

    @staticmethod
    def _requires_confirmation(target: str) -> bool:
        risky_words = {
            "buy",
            "confirm",
            "delete",
            "purchase",
            "remove",
            "send",
            "submit",
        }
        return bool(risky_words.intersection(target.casefold().split()))


def _task_summary(task: DevTask) -> str:
    files = ", ".join(task.files) if task.files else "-"
    return (
        f"Task {task.id}: {task.title} [{task.status}]"
        + (f" ({task.next_action})" if task.next_action else "")
        + f"\n  Files: {files}"
        + (f"\n  Notes: {task.notes}" if task.notes else "")
        + (f"\n  Summary: {task.summary}" if task.summary else "")
    )


class DeveloperTools:
    """Gated tools that expose the self-development layer over voice."""

    def __init__(
        self,
        developer: Developer,
        gate: ToolGate | None = None,
        tasks: TaskManager | None = None,
        analyzer: ImprovementAnalyzer | None = None,
        maintenance: MaintenanceController | None = None,
    ) -> None:
        self.developer = developer
        self.tasks = tasks or developer.tasks
        self.analyzer = analyzer or ImprovementAnalyzer(tasks=self.tasks)
        self.maintenance = maintenance or MaintenanceController()
        self.gate = gate or ToolGate()
        self._running_task: asyncio.Task | None = None

    @property
    def tools(self) -> list:
        return [
            self.developer_mode,
            self.dev_new_integration,
            self.dev_list_tasks,
            self.dev_show,
            self.dev_approve,
            self.dev_reject,
            self.dev_cancel,
            self.dev_diff,
            self.dev_tests,
            self.dev_rollback,
            self.dev_maintenance,
        ]

    def _load_task(self, task_id: str | None) -> DevTask:
        try:
            if task_id is None or task_id.strip() in {"", "latest", "latest task"}:
                task = self.tasks.latest()
                if task is None:
                    raise TaskNotFoundError("There are no development tasks yet")
                return task
            return self.tasks.load(task_id.strip())
        except TaskNotFoundError as exc:
            raise ToolError(str(exc)) from exc

    def _launch(self, coro) -> str:
        self._running_task = asyncio.create_task(coro)
        return "I am working on that in the background. Say Jarvis, show the development tasks for the latest status."

    @function_tool()
    async def developer_mode(self, context: RunContext, request: str) -> str:
        """Enter developer mode to change JARVIS's own code, then start the work.

        Use only when the user explicitly asks to modify, improve, extend, or fix the
        assistant's own code. The work runs in the background on a private git branch.

        Args:
            request: The full user request, in the user's own words.
        """
        self.gate.ensure_developer_requested()
        if self._running_task and not self._running_task.done():
            return "Another development task is still running. Say Jarvis, show the development tasks to check on it."
        title = " ".join(request.split())[:120]
        task = self.tasks.create_task(
            title=title,
            description=request,
            category="voice",
            created_by="user",
            status="queued",
        )
        return self._launch(self.developer.start_task(task))

    @function_tool()
    async def dev_new_integration(self, context: RunContext, service: str) -> str:
        """Create a new standards-based integration for a service.

        Developer mode scaffolds a complete integration package (client,
        integration tools, tests, and documentation), runs the full check
        suite, and leaves the work ready for the user's approval to merge.

        Args:
            service: The name of the service, e.g. 'Philips Hue'.
        """
        self.gate.ensure_developer_requested()
        if self._running_task and not self._running_task.done():
            return "Another development task is still running. Say Jarvis, show the development tasks to check on it."
        try:
            slugify(service)
        except ValueError:
            raise ToolError(
                "Please name the service to integrate, such as 'Philips Hue'."
            ) from None
        title = f"Create {service.strip().capitalize()} integration"
        task = self.tasks.create_task(
            title=title,
            description=f"create an integration for {service}",
            category="integration",
            created_by="user",
            status="queued",
        )
        return self._launch(self.developer.start_task(task))

    @function_tool()
    async def dev_list_tasks(
        self, context: RunContext, status: str | None = None
    ) -> str:
        """List development tasks, optionally filtered by status.

        Args:
            status: Optional status filter: suggestion, queued, in_progress,
                needs_approval, completed, rejected, cancelled, rolled_back, or failed.
        """
        self.gate.ensure_developer_requested()
        tasks = self.tasks.list_tasks(status=status)
        if not tasks:
            return "There are no development tasks yet."
        lines = ["Development tasks:"]
        for task in tasks[:10]:
            lines.append(f"- {task.id} [{task.status}] {task.title}")
        return "\n".join(lines)

    @function_tool()
    async def dev_show(self, context: RunContext, task_id: str | None = None) -> str:
        """Show the details and status of a development task.

        Args:
            task_id: The task id, or omit for the most recent task.
        """
        self.gate.ensure_developer_requested()
        task = self._load_task(task_id)
        return _task_summary(task)

    @function_tool()
    async def dev_approve(self, context: RunContext, task_id: str | None = None) -> str:
        """Approve a development task: apply a pending change or merge it to main.

        Args:
            task_id: The task id, or omit for the most recent task.
        """
        self.gate.ensure_developer_requested()
        task = self._load_task(task_id)
        if task.status != "needs_approval":
            raise ToolError(
                f"Task {task.id} is not awaiting approval (status: {task.status})."
            )
        try:
            if task.next_action == "apply":
                return self._launch(self.developer.apply_approved(task))
            return _task_summary(self.developer.merge_task(task))
        except TaskActionError as exc:
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def dev_reject(self, context: RunContext, task_id: str | None = None) -> str:
        """Reject and discard a development task's changes without merging.

        Args:
            task_id: The task id, or omit for the most recent task.
        """
        self.gate.ensure_developer_requested()
        task = self._load_task(task_id)
        try:
            self.developer.reject_task(task)
            return f"Rejected task {task.id} and cleaned up its branch."
        except TaskActionError as exc:
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def dev_cancel(self, context: RunContext, task_id: str | None = None) -> str:
        """Cancel a development task before it finishes.

        Args:
            task_id: The task id, or omit for the most recent task.
        """
        self.gate.ensure_developer_requested()
        task = self._load_task(task_id)
        if task.status in {"completed", "rolled_back"}:
            raise ToolError(
                f"Task {task.id} already finished; use dev_rollback instead."
            )
        try:
            self.developer.cancel_task(task)
            return f"Cancelled task {task.id}."
        except TaskActionError as exc:
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def dev_diff(self, context: RunContext, task_id: str | None = None) -> str:
        """Show the exact code changes a development task made, for review.

        Args:
            task_id: The task id, or omit for the most recent task.
        """
        self.gate.ensure_developer_requested()
        task = self._load_task(task_id)
        diff = self.developer.task_diff(task)
        return f"Changes in task {task.id}:\n{diff}"

    @function_tool()
    async def dev_tests(self, context: RunContext) -> str:
        """Run the full project checks: syntax, imports, lint, format, and the test suite."""
        self.gate.ensure_developer_requested()
        return self.developer.run_tests()

    @function_tool()
    async def dev_rollback(
        self, context: RunContext, task_id: str | None = None
    ) -> str:
        """Undo a development task's changes (reverse a merge, or restore files).

        Args:
            task_id: The task id, or omit for the most recent task.
        """
        self.gate.ensure_developer_requested()
        task = self._load_task(task_id)
        try:
            rolled = self.developer.rollback_task(task)
            return f"Rolled back task {task.id}: {rolled.notes}"
        except TaskActionError as exc:
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def dev_maintenance(
        self, context: RunContext, action: str, minutes: int | None = None
    ) -> str:
        """Enable or disable autonomous improvement, report its status, or set its interval.

        Maintenance only ever creates suggestion tasks from failure statistics; it
        never modifies source automatically.

        Args:
            action: 'enable', 'disable', 'status', or 'interval'.
            minutes: Interval in minutes when action is 'interval' or 'enable'.
        """
        self.gate.ensure_developer_requested()
        normalized = action.strip().casefold()
        if normalized in {"enable", "interval"} and minutes is not None:
            try:
                value = int(minutes)
            except (TypeError, ValueError):
                raise ToolError(
                    "The maintenance interval must be a number of minutes."
                ) from None
            clamped = self.maintenance.set_interval(value)
            self.maintenance.set_enabled(True)
            return (
                f"Autonomous improvement is enabled and will run at most every "
                f"{clamped} minutes, only creating suggestion tasks."
            )
        if normalized == "enable":
            self.maintenance.set_enabled(True)
            return "Autonomous improvement is enabled. It will only suggest tasks based on measured failures; it never edits code by itself."
        if normalized == "disable":
            self.maintenance.set_enabled(False)
            return "Autonomous improvement is disabled."
        if normalized == "status":
            return (
                f"Autonomous improvement is {'enabled' if self.maintenance.is_enabled() else 'disabled'} "
                f"in '{self.maintenance.mode}' mode, running at most every "
                f"{self.maintenance.interval_minutes} minutes."
            )
        raise ToolError("Action must be 'enable', 'disable', 'status', or 'interval'.")
