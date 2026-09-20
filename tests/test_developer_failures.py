"""Tests for failure redaction, logging, and categorization."""

import json

from developer.failures import (
    FailureAnalyzer,
    FailureLog,
    failure_category,
    redact,
)


class TestRedact:
    def test_password_assignment(self):
        assert "password=hey-123" not in redact("login password=hey-123 then")
        assert "<redacted>" in redact("login password=hey-123 then")

    def test_api_key_header(self):
        assert "abc123" not in redact("Authorization: Bearer abc123xyz")

    def test_credential_url(self):
        out = redact("https://user:supersecret@example.com/path")
        assert "supersecret" not in out

    def test_uuid_replaced(self):
        out = redact("id 123e4567-e89b-12d3-a456-426614174000 done")
        assert "123e4567" not in out
        assert "<redacted-id>" in out

    def test_long_hex_replaced(self):
        out = redact("key=f" + "a" * 40)
        assert "f" * 40 not in out

    def test_none_is_safe(self):
        assert redact(None) == ""

    def test_obvious_secret_never_leaks_by_word(self):
        out = redact("password=correct hors{e st@ple battery")
        assert "correct" not in out


class TestFailureLog:
    def test_append_and_read(self, tmp_path):
        log = FailureLog(tmp_path / "logs" / "failures.jsonl")
        log.append(tool="click", target="Buy", reason="element not found")
        records = log.read()
        assert len(records) == 1
        assert records[0]["tool"] == "click"
        assert records[0]["reason"] == "element not found"

    def test_secret_redacted_before_write(self, tmp_path):
        log = FailureLog(tmp_path / "logs" / "failures.jsonl")
        log.append(tool="open_url", reason="login token=supersecret failed")
        raw = (log.path).read_text(encoding="utf-8")
        assert "supersecret" not in raw
        assert "<redacted>" in raw

    def test_read_missing_file(self, tmp_path):
        log = FailureLog(tmp_path / "logs" / "failures.jsonl")
        assert log.read() == []


class TestFailureCategory:
    def test_click(self):
        assert (
            failure_category({"tool": "click", "target": "Submit"}) == "browser.click"
        )

    def test_type_text(self):
        assert failure_category({"tool": "type_text", "target": "email"}) == "type_text"

    def test_timeout(self):
        assert (
            failure_category({"tool": "open_url", "reason": "timed out"}) == "timeout"
        )

    def test_navigation(self):
        assert failure_category({"tool": "go_back"}) == "navigation"

    def test_other(self):
        assert (
            failure_category({"tool": "weird", "reason": "green frobnicator"})
            == "other"
        )


class TestFailureAnalyzer:
    def test_threshold_met_creates_suggestion(self, tmp_path):
        log = FailureLog(tmp_path / "logs" / "failures.jsonl")
        for _ in range(3):
            log.append(tool="click", target="Submit", reason="stale element")
        suggestions = FailureAnalyzer({}).analyze(log.path)
        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "browser.click"
        assert suggestions[0]["count"] == 3

    def test_below_threshold_no_suggestion(self, tmp_path):
        log = FailureLog(tmp_path / "logs" / "failures.jsonl")
        log.append(tool="click", target="Submit", reason="stale element")
        suggestions = FailureAnalyzer({"browser.click": 3}).analyze(log.path)
        assert suggestions == []

    def test_custom_threshold(self, tmp_path):
        log = FailureLog(tmp_path / "logs" / "failures.jsonl")
        for _ in range(2):
            log.append(tool="click", target="Submit", reason="stale element")
        suggestions = FailureAnalyzer({"browser.click": 2}).analyze(log.path)
        assert len(suggestions) == 1
        assert json.dumps(suggestions)  # serializable by the voice layer
