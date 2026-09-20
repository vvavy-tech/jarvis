"""Meta Ads capability (read-only analytics scaffold).

All tools are safe reads. Output separates FACT (what the API reports: status,
spend, impressions, reach, clicks, CTR, CPC, CPM, conversions) from ANALYSIS.
When a campaign or ad is not delivering and the API provided no delivery reason,
the tool says so instead of guessing why.
"""

from __future__ import annotations

from typing import Any, ClassVar

from livekit.agents import RunContext, function_tool

from integrations.base import ActionLevel, Integration
from integrations.meta_ads.client import (
    DATE_PRESETS,
    MetaAdsClient,
    MetaNotAuthenticatedError,
    factual_status,
)

VALID_PRESETS = set(DATE_PRESETS)


class MetaAdsIntegration(Integration):
    name = "meta_ads"
    description = (
        "Read Facebook/Meta Ads analytics: ad accounts, campaigns, ad sets, ads, "
        "spend, impressions, reach, clicks, CTR, CPC, CPM and delivery status. "
        "Read-only; never modifies budgets or campaigns."
    )
    read_only = True
    default_level = ActionLevel.SAFE_READ
    _TOOLS = (
        "meta_ads_overview",
        "meta_ads_campaigns",
        "meta_ads_insights",
        "meta_ads_why_inactive",
    )
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {
        "meta_ads_overview": ActionLevel.SAFE_READ,
        "meta_ads_campaigns": ActionLevel.SAFE_READ,
        "meta_ads_insights": ActionLevel.SAFE_READ,
        "meta_ads_why_inactive": ActionLevel.SAFE_READ,
    }

    def __init__(
        self,
        *,
        gate=None,
        failure_log=None,
        client: MetaAdsClient | None = None,
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self._client = client or MetaAdsClient()

    def is_authenticated(self) -> bool:
        return self._client.is_authenticated()

    def status_note(self) -> str:
        return "" if self.is_authenticated() else "available but not connected"

    async def health_check(self) -> dict[str, Any]:
        return await super().health_check()

    def _not_connected(self) -> dict[str, str]:
        return {
            "status": "not_connected",
            "message": (
                "Meta Ads is available but not connected. The user has not "
                "connected a Meta Ads account, so I cannot see real ad data. "
                "Do not invent numbers."
            ),
        }

    async def _with_client(self, tool: str, operation: str, **kwargs) -> dict[str, Any]:
        if not self._client.is_authenticated():
            return self._not_connected()
        try:
            method = getattr(self._client, operation)
            result = await method(**kwargs)
        except MetaNotAuthenticatedError:
            return self._not_connected()
        except NotImplementedError as exc:
            return {"status": "not_implemented", "message": str(exc)}
        except Exception as exc:
            self._log_failure(tool, operation, exc)
            return {"status": "error", "message": f"Meta Ads lookup failed: {exc}"}
        return {"status": "ok", "result": result}

    def _preset(self, date_preset: str | None) -> str:
        value = (date_preset or "last_7d").strip()
        return value if value in VALID_PRESETS else "last_7d"

    # ------------------------------------------------------------------ #
    # tools (all safe reads)
    # ------------------------------------------------------------------ #

    @function_tool()
    async def meta_ads_overview(self, context: RunContext) -> dict[str, Any]:
        """Give an overview of the ad account: accounts, campaigns, and current status."""
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        accounts = await self._with_client("meta_ads_overview", "get_ad_accounts")
        campaigns = await self._with_client("meta_ads_overview", "get_campaigns")
        return {"accounts": accounts, "campaigns": campaigns}

    @function_tool()
    async def meta_ads_campaigns(
        self, context: RunContext, status: str | None = None
    ) -> dict[str, Any]:
        """List campaigns, optionally filtered by their API status.

        Args:
            status: Optional API status filter: ACTIVE, PAUSED, DELETED, etc.
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "meta_ads_campaigns", "get_campaigns", status=status
        )

    @function_tool()
    async def meta_ads_insights(
        self, context: RunContext, date_preset: str = "last_7d"
    ) -> dict[str, Any]:
        """Get aggregated ad performance for a date range.

        Args:
            date_preset: One of today, yesterday, last_7d, last_30d, last_90d.
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "meta_ads_insights", "get_insights", date_preset=self._preset(date_preset)
        )

    @function_tool()
    async def meta_ads_why_inactive(
        self, context: RunContext, campaign_id: str
    ) -> dict[str, Any]:
        """Explain why a campaign is not delivering, using only factual API data.

        If the API provides a delivery/error reason it is reported. Otherwise the
        cause is stated as not confirmed by the API; the assistant must never guess.

        Args:
            campaign_id: The campaign id to inspect.
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        if not (campaign_id or "").strip():
            return {"status": "error", "message": "A campaign id is required."}
        raw = await self._with_client(
            "meta_ads_why_inactive", "campaign_status", campaign_id=campaign_id.strip()
        )
        if raw.get("status") != "ok":
            return raw
        info = raw.get("result") or {}
        state = factual_status(info.get("status"))
        is_active = "active" in state and "in process" not in state
        detail = {
            "campaign_id": campaign_id,
            "factual_status": state,
            "api_status": info.get("status") or "unknown",
        }
        reason = info.get("delivery_reason") or info.get("error_reason")
        if reason:
            detail["delivery_reason"] = reason
            detail["cause_confirmed"] = True
        else:
            detail["cause_confirmed"] = False
            if not info.get("status"):
                detail["note"] = (
                    "This campaign's status is not readable with the connected "
                    "token; the API returned only an ID. I do not fabricate a "
                    "reason for it."
                )
            elif is_active:
                detail["note"] = "The campaign is delivering normally."
            else:
                detail["note"] = (
                    "The campaign is not delivering. The API did not provide a "
                    "delivery reason, so the cause is not confirmed."
                )
        return detail
