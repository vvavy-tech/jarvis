# Meta Ads

Slug: `meta_ads` — package `src/integrations/meta_ads/`

## Status

Read-only. Requires a Meta Marketing API access token and an Ad Account id;
until configured the agent reports "available but not connected".

## Setup

1. Create an app on Meta and get an access token with the
   `ads_management` / `ads_read` permission.
2. Find your ad account id (e.g. `act_123456789`).
3. Set in `.env.local`:

   ```
   META_ACCESS_TOKEN=...
   META_AD_ACCOUNT_ID=act_123456789
   ```

## Tools (available when connected)

- `meta_ads_overview` — high-level account summary.
- `meta_ads_campaigns` — campaign list, optional status filter.
- `meta_ads_insights` — performance numbers by preset (falls back to `last_7d`).
- `meta_ads_why_inactive` — why a campaign is not delivering, from the API's
  `result_reasons`; reports "cause is not confirmed" when the API says nothing,
  and never invents delivery reasons.

## Tests

`tests/test_integrations_status.py` (`TestMetaAds`) covers factual status
mapping, unconnected honesty, and the inactive-campaign fallback.