"""Behaviour tests for the concrete integrations (Spotify, Gmail, Calendar,
Meta Ads, Windows).

These tests never touch the real desktop, keyboard, or network: media keys and
launching are monkeypatched, and clients are fakes where connection state is
injected.
"""

import pytest
from livekit.agents.llm import ToolError

from gates import ToolGate
from integrations import build_default_registry
from integrations import media as media_module
from integrations.gmail.integration import GmailIntegration
from integrations.google_calendar.integration import GoogleCalendarIntegration
from integrations.meta_ads.client import factual_status
from integrations.meta_ads.integration import MetaAdsIntegration
from integrations.spotify import integration as spotify_module
from integrations.spotify.integration import SpotifyIntegration
from integrations.windows_integration import _SAFE_APP_LAUNCH, WindowsIntegration


class _ActiveGate:
    """A gate that is always in an active conversation with a turn."""

    def __init__(self, turn: str = "Jarvis, go ahead"):
        self.turn = turn

    def ensure_active_conversation(self) -> None:
        return None

    def ensure_action_level(self, level: int, requested: bool = False) -> None:
        return None


class _AuthClient:
    def __init__(self, result=None, is_authenticated=True):
        self._result = result
        self._is_authenticated = is_authenticated
        self.calls = []

    def is_authenticated(self) -> bool:
        return self._is_authenticated

    async def _run(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


class _GmailClient(_AuthClient):
    async def list_messages(self, **kwargs):
        return await self._run(**kwargs)

    async def search(self, **kwargs):
        return await self._run(**kwargs)

    async def get_message(self, message_id, **kwargs):
        return await self._run(message_id=message_id, **kwargs)

    async def list_unread(self, **kwargs):
        return await self._run(**kwargs)


class _CalendarClient(_AuthClient):
    async def get_events(self, **kwargs):
        return await self._run(**kwargs)

    async def availability(self, **kwargs):
        return await self._run(**kwargs)


class _MetaClient(_AuthClient):
    async def get_ad_accounts(self):
        return await self._run(operation="accounts")

    async def get_campaigns(self, **kwargs):
        return await self._run(**kwargs)

    async def get_insights(self, **kwargs):
        return await self._run(**kwargs)

    async def campaign_status(self, campaign_id, **kwargs):
        return self._result or {}


class TestSpotify:
    def test_status_reports_local_or_api(self, monkeypatch):
        for env in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
            monkeypatch.delenv(env, raising=False)
        integration = SpotifyIntegration(gate=_ActiveGate())
        status = integration.status()
        assert status.authenticated is False

    def test_status_tool_shows_api_not_connected(self, monkeypatch):
        for env in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setattr(spotify_module, "ON_WINDOWS", True)
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.spotify_status(None))
        assert result["api_connected"] is False
        assert result["local_control"] is True

    def test_media_play_pause_uses_local_keys(self, monkeypatch):
        sent = []
        monkeypatch.setattr(
            spotify_module, "send_media_key", lambda action: sent.append(action)
        )
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.media_play_pause(None))
        assert result["mechanism"] == "local_media_keys"
        assert sent == ["play_pause"]

    def test_volume_clamped_to_max(self, monkeypatch):
        sent = []
        monkeypatch.setattr(
            spotify_module,
            "send_volume_delta",
            lambda direction, steps: sent.append((direction, steps)),
        )
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.volume_up(None, steps=99))
        assert result["steps"] == 20

    def test_volume_negative_clamped_to_min(self, monkeypatch):
        sent = []
        monkeypatch.setattr(
            spotify_module,
            "send_volume_delta",
            lambda direction, steps: sent.append((direction, steps)),
        )
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.volume_down(None, steps=0))
        assert result["steps"] == 1

    def test_non_windows_blocks_local_control(self, monkeypatch):
        monkeypatch.setattr(spotify_module, "ON_WINDOWS", False)
        integration = SpotifyIntegration(gate=_ActiveGate())
        with pytest.raises(ToolError, match="not available on this platform"):
            run_sync(integration.media_play_pause(None))

    def test_open_spotify_launches(self, monkeypatch):
        monkeypatch.setattr(spotify_module, "ON_WINDOWS", True)
        monkeypatch.setattr(
            spotify_module,
            "launch_spotify",
            lambda: {"launched": True, "method": "desktop_app"},
        )
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.open_spotify(None))
        assert result["launched"] is True


class TestGmail:
    def test_unconnected_tool_is_honest(self, monkeypatch):
        client = _GmailClient(is_authenticated=False)
        integration = GmailIntegration(gate=_ActiveGate(), client=client)
        result = run_sync(integration.read_recent_emails(None, limit=5))
        assert result["status"] == "not_connected"
        assert "do not invent" in result["message"].casefold()

    def test_limit_clamped_to_max(self):
        client = _GmailClient(result=["mail"])
        integration = GmailIntegration(gate=_ActiveGate(), client=client)
        run_sync(integration.read_recent_emails(None, limit=99))
        assert client.calls[-1]["limit"] == 20

    def test_empty_query_is_clamped(self):
        client = _GmailClient(result=["mail"])
        integration = GmailIntegration(gate=_ActiveGate(), client=client)
        run_sync(integration.read_recent_emails(None, limit=-3))
        assert client.calls[-1]["limit"] == 1

    def test_search_passes_through(self):
        client = _GmailClient(result=["mail"])
        integration = GmailIntegration(gate=_ActiveGate(), client=client)
        run_sync(integration.search_emails(None, query="invoice from Shopify", limit=7))
        assert client.calls[-1]["query"] == "invoice from Shopify"


class TestCalendar:
    def test_unconnected_tool_is_honest(self):
        client = _CalendarClient(is_authenticated=False)
        integration = GoogleCalendarIntegration(gate=_ActiveGate(), client=client)
        result = run_sync(integration.calendar_events(None, time_min="a", time_max="b"))
        assert result["status"] == "not_connected"

    def test_next_appointments_clamped(self):
        client = _CalendarClient(result=[{"summary": "Meeting"}])
        integration = GoogleCalendarIntegration(gate=_ActiveGate(), client=client)
        run_sync(integration.next_appointments(None, count=99))
        assert client.calls[-1]["limit"] == 10
        run_sync(integration.next_appointments(None, count=1))
        assert client.calls[-1]["limit"] == 1

    def test_events_pass_max_event_limit(self):
        client = _CalendarClient(result=[])
        integration = GoogleCalendarIntegration(gate=_ActiveGate(), client=client)
        run_sync(integration.calendar_events(None, time_min="t0", time_max="t1"))
        assert client.calls[-1]["limit"] == 20


class TestSpotifyRecovery:
    def test_window_matcher_ignores_playing_track_title(self):
        pids = {11}
        assert media_module._matches_spotify_window(pids, 11, True) is True
        assert media_module._matches_spotify_window(pids, 11, False) is False
        assert media_module._matches_spotify_window(pids, 99, True) is False
        # a window titled like the currently playing track still counts
        assert media_module._matches_spotify_window(pids, 11, True)

    def test_health_reports_responding(self, monkeypatch):
        monkeypatch.setattr(
            media_module,
            "spotify_processes",
            lambda: [11],
        )
        monkeypatch.setattr(
            media_module,
            "spotify_windows",
            lambda: [{"pid": 11, "title": "Spotify", "responding": True}],
        )
        health = media_module.spotify_health()
        assert health["running"] is True
        assert health["responding"] is True
        assert health["window"] is True
        assert health["error"] == ""

    def test_health_reports_hung_app(self, monkeypatch):
        monkeypatch.setattr(media_module, "spotify_processes", lambda: [11])
        monkeypatch.setattr(
            media_module,
            "spotify_windows",
            lambda: [{"pid": 11, "title": "Spotify", "responding": False}],
        )
        health = media_module.spotify_health()
        assert health["responding"] is False

    def test_health_without_window_is_unknown(self, monkeypatch):
        monkeypatch.setattr(media_module, "spotify_processes", lambda: [77])
        monkeypatch.setattr(media_module, "spotify_windows", lambda: [])
        health = media_module.spotify_health()
        assert health["running"] is True
        assert health["responding"] is None

    def test_health_captures_errors(self, monkeypatch):
        def boom():
            raise media_module.MediaControlError("synthetic")

        monkeypatch.setattr(media_module, "spotify_processes", boom)
        health = media_module.spotify_health()
        assert health["error"]
        assert health["running"] is False

    def test_status_tool_includes_health(self, monkeypatch):
        monkeypatch.setattr(spotify_module, "ON_WINDOWS", True)
        monkeypatch.setattr(
            spotify_module,
            "spotify_health",
            lambda: {
                "running": True,
                "process_count": 4,
                "responding": False,
                "error": "",
            },
        )
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.spotify_status(None))
        assert result["api_connected"] is False
        assert result["running"] is True
        assert result["responding"] is False
        assert result["process_count"] == 4

    def test_status_tool_skips_health_off_windows(self, monkeypatch):
        monkeypatch.setattr(spotify_module, "ON_WINDOWS", False)
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.spotify_status(None))
        assert result["running"] is None

    def test_restart_kills_and_relaunches(self, monkeypatch):
        killed = []

        monkeypatch.setattr(spotify_module, "ON_WINDOWS", True)
        monkeypatch.setattr(
            spotify_module,
            "spotify_health",
            lambda: {"running": True, "responding": False},
        )

        def _kill():
            killed.append(1)
            return 4

        monkeypatch.setattr(spotify_module, "terminate_spotify", _kill)
        monkeypatch.setattr(
            spotify_module,
            "launch_spotify",
            lambda: {"launched": True, "method": "desktop_app"},
        )
        integration = SpotifyIntegration(gate=_ActiveGate())
        result = run_sync(integration.restart_spotify(None))
        assert result["killed_processes"] == 4
        assert result["launched"] is True
        assert result["was_responding"] is False

    def test_restart_spotify_requires_active_conversation(self):
        gate = ToolGate()
        gate.set_user_request("tell me a joke")
        integration = SpotifyIntegration(gate=gate)
        with pytest.raises(ToolError, match="wake word"):
            run_sync(integration.restart_spotify(None))

    def test_restart_spotify_off_windows(self, monkeypatch):
        monkeypatch.setattr(spotify_module, "ON_WINDOWS", False)
        integration = SpotifyIntegration(gate=_ActiveGate())
        with pytest.raises(ToolError, match="not available on this platform"):
            run_sync(integration.restart_spotify(None))

    def test_restart_failure_is_a_tool_error(self, monkeypatch):
        monkeypatch.setattr(spotify_module, "ON_WINDOWS", True)
        monkeypatch.setattr(spotify_module, "spotify_health", lambda: {})

        def boom():
            raise media_module.MediaControlError("synthetic relaunch failure")

        monkeypatch.setattr(spotify_module, "terminate_spotify", boom)
        integration = SpotifyIntegration(gate=_ActiveGate())
        with pytest.raises(ToolError, match="relaunch"):
            run_sync(integration.restart_spotify(None))

    def test_restart_spotify_is_reversible_level(self):
        integration = SpotifyIntegration(gate=_ActiveGate())
        assert int(integration.tool_level("restart_spotify")) == 2
        assert "restart_spotify" in integration.tool_names


class TestMetaAds:
    def test_factual_status_mapping(self):
        assert "active" in factual_status("ACTIVE")
        assert "paused" in factual_status("PAUSED")
        assert "in process" in factual_status("IN_PROCESS")
        assert factual_status(None) == "unknown"
        assert factual_status("WIBBLE") == "wibble"

    def test_unconnected_tool_is_honest(self):
        client = _MetaClient(is_authenticated=False)
        integration = MetaAdsIntegration(gate=_ActiveGate(), client=client)
        overview = run_sync(integration.meta_ads_overview(None))
        assert overview["accounts"]["status"] == "not_connected"
        assert overview["campaigns"]["status"] == "not_connected"

    def test_upholds_confirmed_delivery_reason(self):
        client = _MetaClient(
            result={
                "status": "PAUSED",
                "delivery_reason": "Ad set paused by advertiser.",
            }
        )
        integration = MetaAdsIntegration(gate=_ActiveGate(), client=client)
        detail = run_sync(integration.meta_ads_why_inactive(None, campaign_id="123"))
        assert detail["cause_confirmed"] is True
        assert "paused by advertiser" in detail["delivery_reason"]

    def test_never_invents_delivery_reason(self):
        client = _MetaClient(result={"status": "PAUSED"})
        integration = MetaAdsIntegration(gate=_ActiveGate(), client=client)
        detail = run_sync(integration.meta_ads_why_inactive(None, campaign_id="123"))
        assert detail["cause_confirmed"] is False
        assert "not confirmed" in detail["note"].casefold()
        assert "delivery_reason" not in detail

    def test_gated_status_is_not_fabricated(self):
        client = _MetaClient(result={"id": "52608752797914"})
        integration = MetaAdsIntegration(gate=_ActiveGate(), client=client)
        detail = run_sync(
            integration.meta_ads_why_inactive(None, campaign_id="52608752797914")
        )
        assert detail["factual_status"] == "unknown"
        assert detail["cause_confirmed"] is False
        assert "not readable" in detail["note"].casefold()

    def test_invalid_preset_falls_back(self):
        client = _MetaClient(result=[])
        integration = MetaAdsIntegration(gate=_ActiveGate(), client=client)
        run_sync(integration.meta_ads_insights(None, date_preset="since_always"))
        assert client.calls[-1]["date_preset"] == "last_7d"

    def test_registry_registers_meta(self, monkeypatch):
        monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("META_AD_ACCOUNT_ID", raising=False)
        assert "meta_ads" in build_default_registry().names()


class TestWindows:
    def test_safe_launch_allowlist(self, monkeypatch):
        launched = []
        monkeypatch.setattr(
            "integrations.windows_integration.subprocess.Popen",
            lambda cmd: launched.append(cmd),
        )
        integration = WindowsIntegration(gate=_ActiveGate())
        result = run_sync(integration.open_application(None, app="notepad"))
        assert result["launched"] is True
        assert launched == [["notepad.exe"]]

    def test_unknown_app_refused(self):
        integration = WindowsIntegration(gate=_ActiveGate())
        with pytest.raises(ToolError, match="known list"):
            run_sync(integration.open_application(None, app="format c:"))

    def test_known_apps_are_safe(self):
        assert set(_SAFE_APP_LAUNCH) == {
            "notepad",
            "calculator",
            "paint",
            "cmd",
            "powershell",
            "file explorer",
        }

    def test_status_reports_media_controls(self):
        integration = WindowsIntegration(gate=_ActiveGate())
        result = run_sync(integration.windows_status(None))
        assert "media_controls" in result
        assert result["open_application_supported"] is True


def run_sync(coro):
    import asyncio

    return asyncio.run(coro)
