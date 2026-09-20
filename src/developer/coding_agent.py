"""The development orchestrator.

A :class:`Developer` turns a queued task into git-branch work: it asks a Gemini
model to propose a precise edit, applies the edits deterministically inside the
sandbox, validates with the real check suite, repairs up to a bounded number of
times, and leaves the result on a ``jarvis-dev/<task>`` branch awaiting the
user's approval to merge. It never runs git commands that are not explicitly
known to be safe, and it never writes outside the project root.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from developer.git_manager import GitManager
from developer.integration_scaffold import (
    IntegrationScaffold,
    framework_guide,
    is_integration_scaffold_request,
    service_name_from_request,
)
from developer.project_tools import (
    PROJECT_ROOT,
    ProjectAccess,
    ProjectAccessError,
)
from developer.task_manager import DevTask, TaskActionError, TaskManager
from developer.test_runner import TestRunner, format_results

MAX_EDITS = 8
MAX_EDIT_CHARS = 6000
MAX_GATHERED_FILES = 10
MAX_REPAIR_ROUNDS = 2

HERMES_DEV_MAX_CHARS = 4000

EDIT_OPS = frozenset({"replace", "create", "delete"})

_SYSTEM_PROMPT = (
    "You are the code editor inside JARVIS, a self-developing voice assistant. "
    "You propose precise, minimal, correct changes to the JARVIS codebase. "
    "You never apply changes yourself; you only return a JSON edit plan. "
    "Prefer the smallest change that satisfies the request. Match the existing "
    "code style. Do not invent dependencies; if a dependency is needed, say so "
    "in the summary. Never propose changes to .env files, credentials, secrets, "
    "logs, or the queue directory. Return ONLY valid JSON."
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ProposalError(RuntimeError):
    """Raised when an LLM proposal cannot be parsed or would be unsafe."""


@dataclass
class Edit:
    op: str
    path: str
    old: str = ""
    new: str = ""
    reason: str = ""
    existed: bool = True

    def to_dict(self) -> dict[str, str]:
        return {
            "op": self.op,
            "path": self.path,
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edit:
        op = str(data.get("op", "")).strip().lower()
        path = str(data.get("path", "")).strip()
        if op not in EDIT_OPS:
            raise ProposalError(f"Unsupported edit operation: {op or '<empty>'!r}")
        if not path:
            raise ProposalError("Edit is missing a file path")
        normalized = Path(path.replace("\\", "/")).as_posix().lstrip("/")
        if not normalized or normalized == "." or ".." in Path(normalized).parts:
            raise ProposalError(f"Unsafe edit path: {path!r}")
        old = str(data.get("old", ""))
        new = str(data.get("new", ""))
        if op == "replace" and (not old or not new):
            raise ProposalError(
                f"Replace edit for {normalized!r} needs both 'old' and 'new'"
            )
        if op == "create" and not new:
            raise ProposalError(f"Create edit for {normalized!r} needs 'new' content")
        if op == "delete" and old:
            raise ProposalError(
                f"Delete edit for {normalized!r} must not include 'old'"
            )
        for value in (old, new):
            if len(value) > MAX_EDIT_CHARS:
                raise ProposalError(
                    f"Edit content too large (max {MAX_EDIT_CHARS} chars)"
                )
        return cls(
            op=op,
            path=normalized,
            old=old,
            new=new,
            reason=str(data.get("reason", ""))[:300],
        )


def parse_proposal(raw: str) -> dict[str, Any]:
    """Parse the LLM's JSON edit plan, tolerating markdown fences."""
    text = ""
    block = _JSON_BLOCK_RE.search(raw)
    if block:
        text = block.group(1)
    else:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            text = brace.group(0)
    if not text:
        raise ProposalError("LLM returned no JSON edit plan")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProposalError(f"LLM edit plan is not valid JSON: {exc}") from exc
    edits = data.get("edits") if isinstance(data, dict) else None
    if not isinstance(edits, list) or not edits:
        raise ProposalError("Edit plan has no 'edits' list")
    if len(edits) > MAX_EDITS:
        raise ProposalError(f"Too many edits (max {MAX_EDITS})")
    parsed: list[Edit] = []
    seen: set[str] = set()
    for entry in edits:
        edit = Edit.from_dict(entry if isinstance(entry, dict) else {})
        if edit.path in seen:
            raise ProposalError(f"Duplicate target path: {edit.path!r}")
        seen.add(edit.path)
        parsed.append(edit)
    return {
        "edits": [edit.to_dict() for edit in parsed],
        "summary": str(data.get("summary", ""))[:500] if isinstance(data, dict) else "",
    }


class _Backups:
    """Snapshot copies of files touched by a task (used when git is absent)."""

    def __init__(self, backup_dir: Path) -> None:
        self.backup_dir = backup_dir

    def snapshot(self, project: ProjectAccess, paths: list[str]) -> None:
        shutil.rmtree(self.backup_dir, ignore_errors=True)
        for path in paths:
            try:
                resolved = project.resolve_project_path(path)
            except ProjectAccessError:
                continue
            if not resolved.is_file():
                continue
            target = self.backup_dir / resolved.relative_to(project.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, target)

    def restore(self, project: ProjectAccess, paths: list[str]) -> None:
        for path in paths:
            source = self.backup_dir / Path(path.replace("\\", "/"))
            if not source.is_file():
                continue
            target = project.resolve_project_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


class Developer:
    """Plans, applies, validates, and commits development tasks."""

    def __init__(
        self,
        *,
        project: ProjectAccess | None = None,
        tasks: TaskManager | None = None,
        git: GitManager | None = None,
        runner: TestRunner | None = None,
        llm: Any | None = None,
        root: Path | None = None,
        hermes: Any | None = None,
        memory: Any | None = None,
    ) -> None:
        resolved_root = (root or PROJECT_ROOT).resolve()
        self.project = project or ProjectAccess(resolved_root)
        self.tasks = tasks or TaskManager(resolved_root / "improvement_queue")
        self.git = git or GitManager(resolved_root)
        self.runner = runner or TestRunner(resolved_root)
        self._llm = llm
        self.hermes = hermes
        self.memory = memory
        self.root = resolved_root

    # ------------------------------------------------------------------ #
    # question and answer
    # ------------------------------------------------------------------ #

    def _get_llm(self) -> Any:
        if self._llm is None:
            from livekit.plugins import google

            self._llm = google.LLM()
        return self._llm

    async def _ask(self, system: str, user: str) -> str:
        llm = self._get_llm()
        if hasattr(llm, "complete"):
            result = llm.complete(system=system, user=user)
            if inspect.isawaitable(result):
                return await result
            return str(result)
        from livekit.agents.llm import ChatContext

        context = ChatContext.empty()
        context.add_message(role="system", content=system)
        context.add_message(role="user", content=user)
        stream = llm.chat(chat_ctx=context)
        chunks: list[str] = []
        async for chunk in stream.to_str_iterable():
            chunks.append(chunk)
        return "".join(chunks)

    # ------------------------------------------------------------------ #
    # context gathering
    # ------------------------------------------------------------------ #

    def _gather_files(self, description: str) -> dict[str, str]:
        """Return project-relative paths mapped to contents, hinted by text."""
        hinted: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"[\w./\\-]+\.[a-zA-Z0-9]{1,6}", description):
            candidate = Path(match.group(0).replace("\\", "/")).as_posix().lstrip("./")
            if not self.project.has_path(candidate):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            hinted.append(candidate)
            if len(hinted) >= MAX_GATHERED_FILES:
                break

        if not hinted:
            bare_tokens = re.findall(r"\b[a-zA-Z0-9_]+\.py\b", description)
            for token in bare_tokens:
                for candidate in (f"src/{token}", token):
                    if self.project.has_path(candidate) and candidate not in seen:
                        seen.add(candidate)
                        hinted.append(candidate)
                        break
        return {
            path: self.project.read_file(path)["content"]
            for path in hinted[:MAX_GATHERED_FILES]
        }

    def _index_files(self, max_files: int = 400) -> str:
        files = self.project.list_files()
        lines: list[str] = []
        for relative in files[:max_files]:
            try:
                size = self.project.resolve_project_path(relative).stat().st_size
            except (ProjectAccessError, OSError):
                size = 0
            lines.append(f"- {relative} ({size} bytes)")
        if len(files) > max_files:
            lines.append(f"- ... and {len(files) - max_files} more files")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # proposal
    # ------------------------------------------------------------------ #

    def _context_prompt(self, task: DevTask) -> str:
        contents = self._gather_files(task.description)
        sections: list[str] = [
            f"Task: {task.title}",
            f"\nDescription:\n{task.description}",
        ]
        index = self._index_files()
        sections.append(f"\nProject index (trimmed):\n{index}")
        if contents:
            body = "\n\n".join(
                f"### {path}\n\n{content[: MAX_EDIT_CHARS * 2]}"
                for path, content in contents.items()
            )
            sections.append(f"\nRelevant file contents:\n{body}")
        sections.append(
            '\nRespond with a JSON object: {"summary": "...", '
            '"edits": [{"op": "replace"|"create"|"delete", "path": "<repo-relative path>", '
            '"old": "<exact text to replace for replace ops>", '
            '"new": "<replacement or new file content>", "reason": "..."}]}.'
        )
        if "integration" in (task.description or "").casefold():
            sections.append(f"\nIntegrations framework reference:\n{framework_guide()}")
        return "\n".join(sections)

    async def _propose(self, task: DevTask) -> dict[str, Any]:
        if is_integration_scaffold_request(task.description):
            try:
                service = service_name_from_request(task.description)
            except ValueError as exc:
                raise ProposalError(str(exc)) from exc
            return IntegrationScaffold.proposal_for(service, task.description)
        hermes_plan = await self._maybe_hermes_context(task)
        memory_context = await self._maybe_memory_context(
            f"{task.title}\n\n{task.description}"
        )
        user = self._context_prompt(task)
        extra: list[str] = []
        if hermes_plan and hermes_plan.strip():
            extra.append(f"### Analysis from the Hermes backend\n{hermes_plan}")
        if memory_context and memory_context.strip():
            extra.append(memory_context)
        if extra:
            user = "\n\n".join(extra) + "\n\n" + user
        raw = await self._ask(_SYSTEM_PROMPT, user)
        return parse_proposal(raw)

    async def _maybe_hermes_context(self, task: DevTask) -> str:
        """Ask the Hermes backend to plan/analyse the task, best-effort.

        The Hermes backend is fully optional and strictly isolated: when it is
        missing, failing, timing out, or an exception escapes, this returns an
        empty string and the normal Gemini proposal flow proceeds untouched.
        The git branch / approval workflow is never affected.
        """
        if self.hermes is None:
            return ""
        try:
            if not self.hermes.is_available():
                return ""
            result = await self.hermes.plan(f"{task.title}\n\n{task.description}")
        except Exception:
            return ""
        if not result.ok or not (result.text or "").strip():
            return ""
        return result.text[:HERMES_DEV_MAX_CHARS]

    async def _maybe_memory_context(self, prompt: str) -> str:
        """Bounded, read-only memory context relevant to a Developer/Hermes task.

        Memory is consulted only to inform planning (architecture decisions,
        past lessons, integration choices). Hermes/Developer can never write to
        it here: writes are exclusively JARVIS's explicit-request path. Any
        failure returns an empty string.
        """
        if self.memory is None:
            return ""
        try:
            context = await self.memory.build_context(prompt)
        except Exception:
            return ""
        if not (context or "").strip():
            return ""
        return f"### Relevant long-term memory (read-only)\n{context}"

    async def _propose_repair(self, task: DevTask, problem: str) -> dict[str, Any]:
        user = (
            "Your previous edit plan failed validation. The problems:\n\n"
            f"{problem}\n\n"
            "Return a corrected JSON edit plan for the same task. Repropose the "
            "full set of edits needed (not just the failing one), relative to the "
            "original files. Do not duplicate paths."
        )
        raw = await self._ask(_SYSTEM_PROMPT, user)
        return parse_proposal(raw)

    # ------------------------------------------------------------------ #
    # apply
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_edits(project: ProjectAccess, proposal: dict[str, Any]) -> list[str]:
        applied: list[str] = []
        for entry in proposal["edits"]:
            edit = Edit.from_dict(entry)
            resolved = project._check_writable(edit.path)
            relative = resolved.relative_to(project.root).as_posix()
            if edit.op == "create":
                if resolved.exists():
                    raise ProjectAccessError(f"File already exists: {relative!r}")
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(edit.new, encoding="utf-8", newline="")
            elif edit.op == "delete":
                if not resolved.exists():
                    raise ProjectAccessError(f"File does not exist: {relative!r}")
                resolved.unlink()
            else:
                if not resolved.exists():
                    raise ProjectAccessError(f"File does not exist: {relative!r}")
                original = resolved.read_text(encoding="utf-8")
                if edit.old not in original:
                    raise ProposalError(f"Target text not found in {relative!r}")
                resolved.write_text(
                    original.replace(edit.old, edit.new, 1),
                    encoding="utf-8",
                    newline="",
                )
            applied.append(relative)
        return applied

    def _created_paths(self, proposal: dict[str, Any]) -> list[str]:
        return [
            entry["path"] for entry in proposal["edits"] if entry.get("op") == "create"
        ]

    # ------------------------------------------------------------------ #
    # backups
    # ------------------------------------------------------------------ #

    def _backup_dir(self, task_id: str) -> Path:
        return self.tasks.backups_dir / task_id

    def _pre_apply_backup(self, task: DevTask) -> None:
        if self.git.available() and self.git.is_repo():
            return
        _Backups(self._backup_dir(task.id)).snapshot(self.project, task.files)

    # ------------------------------------------------------------------ #
    # restore / cleanup
    # ------------------------------------------------------------------ #

    def _restore_uncommitted(self, task: DevTask) -> None:
        if self.git.available() and self.git.is_repo():
            tracked = self.git.tracked_files()
            restore_paths = [path for path in task.files if path in tracked]
            created_paths = [path for path in task.files if path not in tracked]
            self.git.restore_from_index(restore_paths)
            for path in created_paths:
                try:
                    self.project.delete_file(path)
                except ProjectAccessError:
                    continue
            return
        backups = _Backups(self._backup_dir(task.id))
        backups.restore(self.project, task.files)
        for path in self._created_paths(task.proposal or {}):
            try:
                self.project.delete_file(path)
            except ProjectAccessError:
                continue

    def _cleanup_branch(self, task: DevTask) -> None:
        if not (self.git.available() and self.git.is_repo()):
            return
        if task.branch and self.git.branch_exists(task.branch):
            self.git.checkout("main")
            self.git.delete_branch(task.id)
        elif (self.git.current_branch() or "").startswith("jarvis-dev/"):
            self.git.checkout("main")

    # ------------------------------------------------------------------ #
    # main flow
    # ------------------------------------------------------------------ #

    def _record_failure(self, task: DevTask, reason: str) -> DevTask:
        task = self.tasks.set_status(task, "failed", notes=f"Failed: {reason[:200]}")
        self._cleanup_branch(task)
        return task

    async def start_task(self, task: DevTask) -> DevTask:
        """Run a queued task through propose -> apply -> validate -> commit."""
        if task.status != "queued":
            raise TaskActionError(f"Task cannot start from status {task.status!r}")
        task = self.tasks.set_status(task, "in_progress", "Starting development work.")

        git_ok = self.git.available() and self.git.is_repo()
        if git_ok:
            if self.git.current_branch() != "main":
                self.git.checkout("main")
            task.base_commit = self.git.head_commit()
            task.baseline_status = self.git.status_porcelain()
            task.branch = self.git.create_task_branch(task.id)
            task.notes = f"Working on branch {task.branch}"
            self.tasks.save(task)

        try:
            task.proposal = await self._propose(task)
        except ProposalError as exc:
            return self._record_failure(task, str(exc))
        except Exception as exc:  # LLM / transport failures
            return self._record_failure(task, f"Proposal failed: {type(exc).__name__}")

        edits = task.proposal.get("edits", [])
        task.files = [entry["path"] for entry in edits]
        impact, _ = self.git.high_impact_report(task.files)
        task.high_impact = impact
        if git_ok:
            baseline_dirty = self.git.dirty_paths_from_porcelain(
                task.baseline_status or ""
            )
            task.caution = bool(baseline_dirty.intersection(task.files))
        else:
            task.caution = False
        self.tasks.save(task)

        if impact:
            task.next_action = "apply"
            return self.tasks.set_status(
                task,
                "needs_approval",
                "High-impact change proposed; awaiting approval before applying.",
            )

        self._pre_apply_backup(task)
        ok, problem = await self._apply_and_validate(task, rounds_max=MAX_REPAIR_ROUNDS)
        if not ok:
            self._restore_uncommitted(task)
            self._cleanup_branch(task)
            return self._record_failure(task, problem)

        if git_ok and task.branch:
            commit_message = f"{task.id}: {task.title}"
            self.git.commit_paths(commit_message, task.files)
            try:
                commit = self.git.head_commit()
            except Exception:
                commit = "committed"
            task.summary = f"{task.proposal.get('summary', '')} [commit {commit[:12]}]"
        else:
            task.summary = task.proposal.get("summary", "")
        task.next_action = "merge"
        return self.tasks.set_status(
            task,
            "needs_approval",
            "Changes validated and committed; awaiting approval to merge to main.",
        )

    async def _apply_and_validate(
        self, task: DevTask, *, rounds_max: int
    ) -> tuple[bool, str]:
        for attempt in range(rounds_max + 1):
            try:
                self._apply_edits(self.project, task.proposal or {})
            except (ProjectAccessError, ProposalError) as exc:
                task.repairs = attempt
                if attempt >= rounds_max:
                    return False, f"Apply failed: {exc}"
                task.repairs = attempt + 1
                task.proposal = await self._propose_repair(task, str(exc))
                continue
            results = self.runner.run_all(include_tests=True)
            if all(result.ok for result in results):
                return True, ""
            problem = format_results(results)
            task.repairs = attempt
            if attempt >= rounds_max:
                return False, problem
            task.repairs = attempt + 1
            try:
                task.proposal = await self._propose_repair(task, problem)
            except ProposalError as exc:
                return False, f"Repair proposal failed: {exc}"

        return False, "repair limit exceeded"

    async def apply_approved(self, task: DevTask) -> DevTask:
        """Apply a stashed high-impact proposal after the user approves it."""
        if task.next_action != "apply" or not task.proposal:
            raise TaskActionError("No pending proposal to apply for this task")
        if task.branch and not self.git.branch_exists(task.branch):
            self.git.create_task_branch(task.id)
        self._pre_apply_backup(task)
        ok, problem = await self._apply_and_validate(task, rounds_max=MAX_REPAIR_ROUNDS)
        if not ok:
            self._restore_uncommitted(task)
            self._cleanup_branch(task)
            return self._record_failure(task, problem)
        if task.branch:
            self.git.commit_paths(f"{task.id}: {task.title}", task.files)
        task.next_action = "merge"
        return self.tasks.set_status(
            task,
            "needs_approval",
            "Approved changes applied and committed; awaiting approval to merge.",
        )

    def merge_task(self, task: DevTask) -> DevTask:
        """Merge the task branch into main after user approval."""
        if task.next_action != "merge" or task.status != "needs_approval":
            raise TaskActionError("Task is not waiting for a merge")
        git_ok = self.git.available() and self.git.is_repo()
        if not git_ok or not task.branch or not self.git.branch_exists(task.branch):
            task.next_action = None
            task.merge_commit = None
            return self.tasks.set_status(
                task,
                "completed",
                "Applied in place (git-less environment).",
            )
        merge_commit = self.git.merge_main(
            task.id, f"Merge {task.branch}: {task.title}"
        )
        task.merge_commit = merge_commit
        self.git.delete_branch(task.id)
        task.next_action = None
        return self.tasks.set_status(
            task, "completed", f"Merged to main as {merge_commit[:12]}."
        )

    def reject_task(self, task: DevTask) -> DevTask:
        self._cleanup_branch(task)
        task.next_action = None
        return self.tasks.set_status(task, "rejected", "Task was rejected by the user.")

    def cancel_task(self, task: DevTask) -> DevTask:
        self._cleanup_branch(task)
        task.next_action = None
        return self.tasks.set_status(
            task, "cancelled", "Task was cancelled by the user."
        )

    def rollback_task(self, task: DevTask) -> DevTask:
        """Undo a merged or in-progress task by restoring its files."""
        git_ok = self.git.available() and self.git.is_repo()
        if git_ok and task.merge_commit:
            try:
                self.git.checkout("main")
                self.git.revert_merge(task.merge_commit)
                task.next_action = None
                return self.tasks.set_status(
                    task,
                    "rolled_back",
                    f"Merge {task.merge_commit[:12]} reverted on main.",
                )
            except Exception as exc:
                raise TaskActionError(f"Rollback failed: {exc}") from exc
        if git_ok and task.branch and self.git.branch_exists(task.branch):
            with contextlib.suppress(Exception):
                self.git.checkout(task.branch)
            self._restore_uncommitted(task)
            self._cleanup_branch(task)
            task.next_action = None
            return self.tasks.set_status(task, "rolled_back", "Branch changes undone.")
        backups = _Backups(self._backup_dir(task.id))
        backups.restore(
            self.project, task.files or self._created_paths(task.proposal or {})
        )
        for path in self._created_paths(task.proposal or {}):
            with contextlib.suppress(ProjectAccessError):
                self.project.delete_file(path)
        task.next_action = None
        return self.tasks.set_status(
            task, "rolled_back", "Applied changes undone from backup."
        )

    def task_diff(self, task: DevTask) -> str:
        if not task.files:
            return "No files were changed by this task."
        if self.git.available() and self.git.is_repo():
            return (
                self.git.diff(task.base_commit or None, paths=task.files) or "No diff."
            )
        return "Working in a git-less environment; no textual diff available."

    def run_tests(self) -> str:
        results = self.runner.run_all(include_tests=True)
        return format_results(results)
