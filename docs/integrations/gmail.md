# Gmail

Slug: `gmail` — package `src/integrations/gmail/`

## Status

Read-only. Requires OAuth credentials; until configured the agent reports
"available but not connected" and never fabricates email content.

## Setup

1. In Google Cloud Console create a project and enable the Gmail API.
2. Create an OAuth 2.0 client ID (Desktop app) and download the JSON.
3. Set in `.env.local`:

   ```
   GMAIL_OAUTH_CREDENTIALS=C:\path\to\client_secret.json
   GMAIL_TOKEN_PATH=C:\path\to\token.json
   ```

4. Run the agent once; the first call triggers the OAuth consent flow and
   writes `token.json`.

## Tools (available when connected)

- `gmail_search` — search messages (query + limit clamped 1-20).
- `gmail_unread` — list unread (limit clamped 1-20).
- `gmail_read` — read a message by id.

## Tests

`tests/test_integrations_status.py` (`TestGmail`) covers unconnected honesty and
limit clamping.