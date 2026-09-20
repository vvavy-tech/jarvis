"""Meta Ads / Facebook Ads integration (read-only analytics scaffold)."""

from integrations.meta_ads.client import (
    DATE_PRESETS,
    INSIGHT_METRICS,
    MetaAdsClient,
    MetaAdsError,
    MetaNotAuthenticatedError,
    factual_status,
)
from integrations.meta_ads.integration import MetaAdsIntegration

__all__ = [
    "DATE_PRESETS",
    "INSIGHT_METRICS",
    "MetaAdsClient",
    "MetaAdsError",
    "MetaAdsIntegration",
    "MetaNotAuthenticatedError",
    "factual_status",
]
