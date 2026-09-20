"""Meta Ads / Facebook Ads client (read-only analytics via the Marketing API).

JARVIS reports FACTS first and separates them from ANALYSIS:

- Factual data comes only from the fields the API returns with the connected
  token (status, spend, impressions, reach, clicks, CTR, CPC, CPM, conversions,
  and any delivery/error reason the API provides).
- The connected token's permissions decide what is visible: fields JARVIS is
  not entitled to read are simply absent, and JARVIS never invents them.
- JARVIS never invents a reason why an ad or campaign stopped. If the API
  provides a delivery/status/error reason it is reported; otherwise the cause
  is explicitly marked "not confirmed by the API".

The ad account can come from ``META_AD_ACCOUNT_ID`` or is discovered from the
token through ``/me/adaccounts``. The access token is only ever sent as the
``access_token`` query parameter and is redacted from any surfaced error.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

REQUIRED_ENV = ("META_ACCESS_TOKEN",)

# Neutral, factual labels derived only from API status values.
_FACTUAL_STATUS: dict[str, str] = {
    "ACTIVE": "active",
    "PAUSED": "paused",
    "DISABLED": "disabled",
    "DELETED": "deleted",
    "ARCHIVED": "archived",
    "IN_PROCESS": "in process",
    "WITH_ISSUES": "has delivery issues",
    "CAMPAIGN_PAUSED": "campaign is paused",
    "ADSET_PAUSED": "ad set is paused",
}

INSIGHT_METRICS = (
    "spend",
    "impressions",
    "reach",
    "clicks",
    "ctr",
    "cpc",
    "cpm",
    "inline_link_clicks",
    "actions",
)

DATE_PRESETS = (
    "today",
    "yesterday",
    "last_7d",
    "last_30d",
    "last_90d",
)

GRAPH_FIELDS = {
    "account": "id,name,account_status,currency,amount_spent",
    "campaign": "id,name,status,effective_status,objective",
    "adset": "id,name,status,effective_status",
    "ad": "id,name,status,effective_status",
}


class MetaNotAuthenticatedError(RuntimeError):
    """Raised when a Meta Ads call is attempted before the token is configured."""


class MetaAdsError(RuntimeError):
    """Raised when the Meta Graph API cannot fulfil a request."""


def _configured() -> bool:
    return all(bool(os.environ.get(name, "").strip()) for name in REQUIRED_ENV)


def factual_status(value: str | None) -> str:
    """Turn an API status value into neutral factual language."""
    if not value:
        return "unknown"
    return _FACTUAL_STATUS.get(value.upper(), value.lower())


class _AsyncHTTPXTransport:
    """Thin async wrapper so the client can be tested with a fake transport."""

    async def get(self, url: str, params: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=45.0) as client:
            return await client.get(url, params=params)


class MetaAdsClient:
    """Read-only client for the Meta Marketing API."""

    def __init__(
        self,
        *,
        transport: Any | None = None,
        graph_version: str | None = None,
    ) -> None:
        self._token = os.environ.get("META_ACCESS_TOKEN", "").strip()
        raw_account = os.environ.get("META_AD_ACCOUNT_ID", "").strip()
        self._account_id = raw_account.removeprefix("act_")
        self._graph_version = graph_version or os.environ.get(
            "META_GRAPH_VERSION", "v24.0"
        )
        self._transport = transport or _AsyncHTTPXTransport()
        self._discovered_account_id: str | None = None

    def is_authenticated(self) -> bool:
        return bool(self._token)

    def _require_authenticated(self) -> None:
        if not self._token:
            raise MetaNotAuthenticatedError(
                "Meta Ads is available but not connected. Configure "
                "META_ACCESS_TOKEN to read ad analytics."
            )

    async def _get(self, path: str, *, params: dict[str, Any]) -> Any:
        self._require_authenticated()
        url = f"https://graph.facebook.com/{self._graph_version}{path}"
        query = {"access_token": self._token, **params}
        try:
            response = await self._transport.get(url, params=query)
        except MetaAdsError:
            raise
        except Exception as exc:
            raise MetaAdsError(f"Meta API request failed: {exc}") from exc
        status = getattr(response, "status_code", 200)
        if status >= 400:
            raise MetaAdsError(
                f"Meta API error (HTTP {status}): {self._error_detail(response)}"
            )
        return response

    def _error_detail(self, response: Any) -> str:
        body = ""
        for extractor in (lambda r: json.dumps(r.json()), lambda r: r.text):
            try:
                body = extractor(response)
                break
            except Exception:
                continue
        if self._token and self._token in body:
            body = body.replace(self._token, "<redacted>")
        return (body[:300] or "unknown error").strip()

    async def _ad_account_id(self) -> str:
        if self._account_id:
            return self._account_id
        if self._discovered_account_id is None:
            accounts = await self.get_ad_accounts()
            if not accounts:
                raise MetaAdsError(
                    "no ad accounts are accessible with this token; set "
                    "META_AD_ACCOUNT_ID or grant account access."
                )
            first = accounts[0]
            self._discovered_account_id = first.get("account_id") or first.get(
                "id", ""
            ).removeprefix("act_")
        return self._discovered_account_id

    async def get_ad_accounts(self) -> list[dict[str, Any]]:
        response = await self._get(
            "/me/adaccounts",
            params={"fields": GRAPH_FIELDS["account"], "limit": 100},
        )
        return response.json().get("data") or []

    async def get_campaigns(self, status: str | None = None) -> list[dict[str, Any]]:
        account = await self._ad_account_id()
        response = await self._get(
            f"/act_{account}/campaigns",
            params={"fields": GRAPH_FIELDS["campaign"], "limit": 100},
        )
        rows = response.json().get("data") or []
        want = (status or "").strip().upper()
        if want:
            rows = [
                row
                for row in rows
                if (row.get("status") or row.get("effective_status") or "").upper()
                == want
            ]
        return rows

    async def get_ad_sets(
        self, campaign_id: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        account = await self._ad_account_id()
        params: dict[str, Any] = {
            "fields": GRAPH_FIELDS["adset"],
            "campaign_id": campaign_id,
            "limit": 100,
        }
        response = await self._get(f"/act_{account}/adsets", params=params)
        return response.json().get("data") or []

    async def get_ads(
        self,
        campaign_id: str,
        adset_id: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        account = await self._ad_account_id()
        params: dict[str, Any] = {
            "fields": GRAPH_FIELDS["ad"],
            "campaign_id": campaign_id,
            "limit": 100,
        }
        if adset_id:
            params["adset_id"] = adset_id
        response = await self._get(f"/act_{account}/ads", params=params)
        return response.json().get("data") or []

    async def get_insights(
        self,
        date_preset: str = "last_7d",
        *,
        level: str = "campaign",
        entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if entity_id is not None:
            prefix = f"/{entity_id}"
        else:
            account = await self._ad_account_id()
            prefix = f"/act_{account}"
        response = await self._get(
            f"{prefix}/insights",
            params={
                "date_preset": date_preset,
                "level": level,
                "fields": ",".join(INSIGHT_METRICS),
                "limit": 100,
            },
        )
        return response.json().get("data") or []

    async def campaign_status(self, campaign_id: str) -> dict[str, Any]:
        """Factual campaign status plus any API-provided delivery reason.

        Delivery reasons are only ever included when the API returns them.
        Fields (e.g. ``status``) may be absent when the token cannot read them.
        """
        response = await self._get(
            f"/{campaign_id}", params={"fields": GRAPH_FIELDS["campaign"]}
        )
        return response.json()
