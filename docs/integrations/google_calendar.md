# Google Calendar

Slug: `google_calendar` — package `src/integrations/google_calendar/`

## Status

Read-only. Requires Google OAuth credentials; until configured the agent reports
"available but not connected".

## Setup

1. In the same (or a new) Google Cloud project enable the Google Calendar API.
2. Download the OAuth client JSON (Desktop app).
3. Set in `.env.local`:

   ```
   GOOGLE_OAUTH_CREDENTIALS=C:\path\to\client_secret.json
   CALENDAR_TOKEN_PATH=C:\path\to\calendar-token.json
   ```

4. First use triggers the OAuth consent flow and writes the token file.

## Tools (available when connected)

- `calendar_events` — upcoming events (limit clamped 1-20).
- `next_appointments` — next N appointments (limit 1-20).
- `calendar_free_check` — availability for a time window (requires start/end).

## Tests

`tests/test_integrations_status.py` (`TestCalendar`) covers unconnected honesty
and limit clamping.