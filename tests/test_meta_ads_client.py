"""Tests for the real Meta (Graph API) client.

No real network, no real token: the HTTP transport is a fake, and the env var
META_ACCESS_TOKEN is always a throwaway dummy. The guarantees under test mirror
the integration contract: report only fields the token actually returns, never
invent numbers, and never leak the access token through errors.
"""

import pytest

from integrations.meta_ads.client import (
    MetaAdsClient,
    MetaAdsError,
    MetaNotAuthenticatedError,
)

TOKEN = "fake-token-for-tests"


class _FakeResponse:
    def __init__(self, *, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self._text = text

    def json(self):
        return self._data

    @property
    def text(self):
        return self._text


class _FakeTransport:
    """Records calls and routes them to a handler (path, params) -> response."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    async def get(self, url, params):
        path = url.split("graph.facebook.com/", 1)[1]
        self.calls.append((path, params))
        return self.handler(path, params)


def _client(monkeypatch, *, account_env=None, transport, token=TOKEN):
    monkeypatch.setenv("META_ACCESS_TOKEN", token)
    if account_env:
        monkeypatch.setenv("META_AD_ACCOUNT_ID", account_env)
    else:
        monkeypatch.delenv("META_AD_ACCOUNT_ID", raising=False)
    return MetaAdsClient(transport=transport)


def _accounts_response(*accounts):
    data = [{"account_id": acc, "id": f"act_{acc}"} | extra for acc, extra in accounts]
    return _FakeResponse(data={"data": data})


def _empty_response():
    return _FakeResponse(data={"data": []})


# --------------------------------------------------------------------------- #
# authentication
# --------------------------------------------------------------------------- #


def test_authenticated_with_only_token(monkeypatch):
    client = _client(
        monkeypatch, transport=_FakeTransport(lambda *a: _empty_response())
    )
    assert client.is_authenticated() is True


def test_not_authenticated_without_token(monkeypatch):
    monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("META_AD_ACCOUNT_ID", raising=False)
    client = MetaAdsClient(transport=_FakeTransport(lambda *a: _empty_response()))
    assert client.is_authenticated() is False


def test_unauthenticated_call_raises(monkeypatch):
    monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("META_AD_ACCOUNT_ID", raising=False)
    client = MetaAdsClient(transport=_FakeTransport(lambda *a: _empty_response()))
    with pytest.raises(MetaNotAuthenticatedError):
        run = client.get_ad_accounts()
        import asyncio

        asyncio.run(run)


# --------------------------------------------------------------------------- #
# ad accounts + discovery
# --------------------------------------------------------------------------- #


def test_get_ad_accounts_parses_data(monkeypatch):
    transport = _FakeTransport(
        lambda path, params: _accounts_response(("2129167314378881", {}))
    )
    client = _client(monkeypatch, transport=transport)
    accounts = _await(client.get_ad_accounts())
    assert accounts[0]["account_id"] == "2129167314378881"
    assert transport.calls[0][0] == "v24.0/me/adaccounts"


def test_environment_account_id_used_without_discovery(monkeypatch):
    transport = _FakeTransport(lambda path, params: _empty_response())
    client = _client(monkeypatch, account_env="act_98888", transport=transport)
    _await(client.get_campaigns())
    assert transport.calls[0][0] == "v24.0/act_98888/campaigns"


def test_account_discovered_from_token_when_env_missing(monkeypatch):
    transport = _FakeTransport(
        lambda path, params: (
            _accounts_response(("2129167314378881", {}))
            if "adaccounts" in path
            else _empty_response()
        )
    )
    client = _client(monkeypatch, transport=transport)
    _await(client.get_campaigns())
    assert transport.calls[-1][0] == "v24.0/act_2129167314378881/campaigns"


def test_no_accounts_raises_honest_error(monkeypatch):
    transport = _FakeTransport(lambda path, params: _empty_response())
    client = _client(monkeypatch, transport=transport)
    with pytest.raises(MetaAdsError, match="no ad accounts"):
        _await(client.get_campaigns())


def test_insights_uses_discovered_account(monkeypatch):
    transport = _FakeTransport(
        lambda path, params: (
            _accounts_response(("2129167314378881", {}))
            if "adaccounts" in path
            else _FakeResponse(data={"data": []})
        )
    )
    client = _client(monkeypatch, transport=transport)
    _await(client.get_insights())
    assert transport.calls[-1][0] == "v24.0/act_2129167314378881/insights"


# --------------------------------------------------------------------------- #
# present-fields-only behaviour (permission-gated fields are simply absent)
# --------------------------------------------------------------------------- #


def test_campaigns_keep_only_present_fields(monkeypatch):
    transport = _FakeTransport(
        lambda path, params: _FakeResponse(
            data={
                "data": [
                    {
                        "id": "52608752797914",
                        "name": "Always On - Prospecting",
                        "status": "ACTIVE",
                    },
                    {"id": "52608752797714"},
                ]
            }
        )
    )
    client = _client(monkeypatch, account_env="act_1", transport=transport)
    rows = _await(client.get_campaigns())
    assert rows[0]["name"] == "Always On - Prospecting"
    assert rows[0]["status"] == "ACTIVE"
    assert rows[1] == {"id": "52608752797714"}


def test_campaigns_filter_by_status_when_visible_only(monkeypatch):
    transport = _FakeTransport(
        lambda path, params: _FakeResponse(
            data={
                "data": [
                    {"id": "a", "status": "ACTIVE"},
                    {"id": "b", "status": "PAUSED"},
                    {"id": "c"},
                ]
            }
        )
    )
    client = _client(monkeypatch, account_env="act_1", transport=transport)
    active = _await(client.get_campaigns(status="ACTIVE"))
    assert [row["id"] for row in active] == ["a"]


def test_insights_keep_only_visible_metrics(monkeypatch):
    transport = _FakeTransport(
        lambda path, params: _FakeResponse(
            data={
                "data": [
                    {
                        "campaign_id": "52608752797914",
                        "date_start": "2026-08-20",
                        "date_stop": "2026-09-18",
                        "impressions": "3855",
                        "spend": "89.34",
                    }
                ]
            }
        )
    )
    client = _client(monkeypatch, account_env="act_1", transport=transport)
    rows = _await(client.get_insights())
    assert rows[0]["spend"] == "89.34"
    assert rows[0]["impressions"] == "3855"
    assert "clicks" not in rows[0]


def test_insights_with_entity_id_uses_entity_path(monkeypatch):
    transport = _FakeTransport(lambda path, params: _FakeResponse(data={"data": []}))
    client = _client(monkeypatch, account_env="act_1", transport=transport)
    _await(client.get_insights(entity_id="52608752797914"))
    assert transport.calls[-1][0] == "v24.0/52608752797914/insights"


def test_get_ads_and_adsets_hit_account_paths(monkeypatch):
    transport = _FakeTransport(lambda path, params: _FakeResponse(data={"data": []}))
    client = _client(monkeypatch, account_env="act_1", transport=transport)
    _await(client.get_ad_sets("cid"))
    assert transport.calls[0][0] == "v24.0/act_1/adsets"
    assert transport.calls[0][1]["campaign_id"] == "cid"
    _await(client.get_ads("cid"))
    assert transport.calls[1][0] == "v24.0/act_1/ads"
    assert transport.calls[1][1]["campaign_id"] == "cid"


def test_campaign_status_returns_present_fields(monkeypatch):
    transport = _FakeTransport(
        lambda path, params: _FakeResponse(data={"id": "52608752797914"})
    )
    client = _client(monkeypatch, transport=transport)
    info = _await(client.campaign_status("52608752797914"))
    assert info["id"] == "52608752797914"
    assert "status" not in info  # gated: only the id is returned


# --------------------------------------------------------------------------- #
# error handling + secret safety
# --------------------------------------------------------------------------- #


def test_api_error_is_raised(monkeypatch):
    def handler(path, params):
        return _FakeResponse(status_code=400, data={"error": {"message": "boom"}})

    client = _client(monkeypatch, transport=_FakeTransport(handler))
    with pytest.raises(MetaAdsError, match="HTTP 400"):
        _await(client.get_ad_accounts())


def test_error_surfaces_never_leak_token(monkeypatch):
    def handler(path, params):
        body = {"error": {"message": f"bad token {TOKEN} please fix"}}
        return _FakeResponse(status_code=401, data=body, text=str(body))

    client = _client(monkeypatch, transport=_FakeTransport(handler))
    with pytest.raises(MetaAdsError) as excinfo:
        _await(client.get_ad_accounts())
    assert TOKEN not in str(excinfo.value)


def test_access_token_sent_in_every_request(monkeypatch):
    transport = _FakeTransport(lambda path, params: _empty_response())
    client = _client(monkeypatch, transport=transport)
    _await(client.get_ad_accounts())
    assert transport.calls[0][1]["access_token"] == TOKEN


def _await(coro):
    import asyncio

    return asyncio.run(coro)
