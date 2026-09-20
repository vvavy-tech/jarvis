"""Structured failure logging and analysis for agent tool calls.

Every failed tool call is appended to ``logs/failures.jsonl`` as a redacted,
single-line JSON record. The :class:`FailureAnalyzer` reads that log back and,
once a category crosses a threshold, proposes improvement tasks.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from developer.project_tools import PROJECT_ROOT

DEFAULT_FAILURE_LOG = PROJECT_ROOT / "logs" / "failures.jsonl"

_KEY_VALUE_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|token|auth(?:orization)?|"
    r"credential|cookie)\s*[:=]\s*[\"']?[^\s,;\"']+"
)
_CREDENTIAL_URL_RE = re.compile(r"(https?://)[^/\s:@]+:[^/\s@]+@")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{24,}\b")
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)^(password|passwd|secret|token|api[_-]?key|auth(?:orization)?|"
    r"cookie|credential|access[_-]?key|private[_-]?key)$"
)


class FailureRecord(TypedDict, total=False):
    time: str
    tool: str
    target: str
    result: str
    reason: str
    fallback: str
    fallback_result: str


def redact(text: Any) -> str:
    """Mask obvious secrets so they never reach prompts, logs, or reports."""
    if text is None:
        return ""
    out = str(text)
    out = _BEARER_RE.sub("bearer <redacted>", out)
    out = _KEY_VALUE_RE.sub("<redacted>", out)
    out = _CREDENTIAL_URL_RE.sub(r"\1<redacted>@", out)
    out = _UUID_RE.sub("<redacted-id>", out)
    out = _LONG_HEX_RE.sub("<redacted>", out)
    return out


def _redact_record_keys(record: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in record.items():
        if _SENSITIVE_KEY_RE.match(key):
            out[key] = "<redacted>"
        else:
            out[key] = redact(value)
    return out


class FailureLog:
    """Appends and reads redacted failure records as JSON lines."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = (log_path or DEFAULT_FAILURE_LOG).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        *,
        tool: str,
        target: str = "",
        result: str = "error",
        reason: str = "",
        fallback: str = "",
        fallback_result: str = "",
    ) -> FailureRecord:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool": str(tool),
            "target": str(target),
            "result": str(result),
            "reason": str(reason),
            "fallback": str(fallback),
            "fallback_result": str(fallback_result),
        }
        redacted_record = _redact_record_keys(record)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redacted_record, ensure_ascii=False) + "\n")
        return redacted_record

    def read(self) -> list[FailureRecord]:
        if not self._path.exists():
            return []
        records: list[FailureRecord] = []
        try:
            lines = self._path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return records
        for line in lines:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "browser.click": ("click", "clicked"),
    "timeout": ("timeout", "timed out"),
    "type_text": ("type_text", "type", "fill"),
    "navigation": ("open_url", "go_back", "navigate"),
    "windows": ("windows", "desktop", "start menu", "taskbar"),
    "spotify": ("spotify", "media_play_pause", "open_spotify", "media key", "volume"),
    "email": ("gmail", "email", "read_recent_emails", "search_emails"),
    "screen_vision": (
        "screen",
        "screen_vision",
        "list_monitors",
        "capture_monitor",
        "capture_cursor_area",
        "get_cursor_position",
        "analyze_screen_on_demand",
    ),
    "import/startup": ("modulenotfound", "importerror", "startup", "import error"),
}

_DEFAULT_THRESHOLDS: dict[str, int] = {
    "browser.click": 3,
    "type_text": 2,
    "navigation": 3,
    "windows": 2,
    "spotify": 3,
    "email": 3,
    "screen_vision": 2,
    "timeout": 3,
    "import/startup": 2,
    "other": 5,
}


def failure_category(record: FailureRecord) -> str:
    blob = " ".join(
        str(record.get(key, "")) for key in ("tool", "target", "result", "reason")
    ).casefold()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in blob for keyword in keywords):
            return category
    return "other"


_CATEGORY_SUGGESTIONS: dict[str, str] = {
    "browser.click": "Improve browser click targeting and resilience",
    "type_text": "Harden the browser text-entry flow",
    "navigation": "Improve browser navigation error handling",
    "windows": "Improve local Windows control reliability",
    "spotify": "Improve Spotify control robustness and fallback handling",
    "email": "Improve Gmail read reliability",
    "screen_vision": "Harden screen capture or analysis resilience",
    "timeout": "Reduce repeated timeouts with smarter waits",
    "import/startup": "Fix the import or startup failure",
    "other": "Investigate repeated tool failures",
}


class FailureAnalyzer:
    """Turns failure-log statistics into improvement-task suggestions."""

    def __init__(self, thresholds: dict[str, int] | None = None) -> None:
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)

    def analyze(self, log_path: Path | None = None) -> list[dict[str, Any]]:
        log = FailureLog(log_path or DEFAULT_FAILURE_LOG)
        counts: dict[str, int] = {}
        for record in log.read():
            category = failure_category(record)
            counts[category] = counts.get(category, 0) + 1

        suggestions: list[dict[str, Any]] = []
        for category, count in sorted(counts.items()):
            threshold = self._thresholds.get(category, self._thresholds["other"])
            if count < threshold:
                continue
            title = _CATEGORY_SUGGESTIONS.get(
                category, f"Investigate repeated {category} failures"
            )
            suggestions.append(
                {
                    "category": category,
                    "count": count,
                    "kind": "suggestion",
                    "title": title,
                    "description": (
                        f"{count} failures in the '{category}' category have been "
                        "recorded. Review the entries in logs/failures.jsonl, find "
                        "the underlying cause, and propose a targeted fix."
                    ),
                    "status": "suggestion",
                }
            )
        return suggestions
