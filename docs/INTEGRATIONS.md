# JARVIS Integrations

JARVIS is built around a **capability registry**. Every service it can reach,
every permission it holds, and every tool it can call is declared as an
*integration*. Nothing is hardcoded in the instructions prompt; the model
discovers what it can do at runtime from the registry.

- JARVIS never fakes data. A service that is "available but not connected" is reported
  honestly (`system_status` / `get_capabilities`), and tools return a clear
  `not_connected` answer instead of invented content.
- Optional integrations can never break startup. JARVIS runs fine with no
  credentials configured; unconnected services simply report their state.
- The voice layer, wake word, browser, Windows control, and Developer Mode
  all behave exactly as before. This framework sits underneath them.

## Layout

```
src/integrations/
├── base.py                 # ActionLevel, IntegrationStatus, Integration, ToolGroup
├── registry.py             # CapabilityRegistry (single source of truth)
├── capability_tools.py     # get_capabilities + system_status voice tools
├── media.py                # Windows media-key control + Spotify launcher
├── windows_integration.py  # local Windows capability
├── __init__.py             # build_default_registry()
├── spotify/                # local media control today, Web API scaffold
├── gmail/                  # read-only scaffold (env-gated)
├── google_calendar/        # read-only scaffold (env-gated)
└── meta_ads/               # read-only analytics (env-gated)
```

Also relevant:

- `src/gates.py` — `ToolGate.ensure_active_conversation()` and
  `ensure_action_level()` enforce the voice rules below.
- `src/developer/integration_scaffold.py` — Developer Mode scaffolding.
- `src/developer/maintenance.py` — opt-in autonomous suggestion tasks.
- `.env.example` — all optional environment variables.
- `docs/integrations/<slug>.md` — per-service setup notes (scaffolded).

## Action levels

Every tool carries a numeric `ActionLevel`, enforced by the voice gate as
metadata rather than prompt text:

| Level | Name          | Meaning                              | Confirmation            |
|------:|---------------|--------------------------------------|-------------------------|
|     1 | `safe_read`   | Read statistics, email, status       | Active conversation only |
|     2 | `reversible`  | Pause music, skip track, open an app | Active conversation only |
|     3 | `consequential` | Send, create, merge, pay         | Explicit request/confirmation in the turn |
|     4 | `security`    | Change auth, grant permissions       | Explicit confirmation, never automatic |

Levels 3–4 are **not used by any current integration**. They are enforced by
`ToolGate.ensure_action_level(level, requested=...)` and reserved for future
write paths (e.g. sending email or creating calendar events).

## The registry

`CapabilityRegistry` answers `what can I actually do?`:

- `capabilities()` — list of `{name, available, authenticated, read_only,
  confirmation_required, action_level, note}` per service.
- `tools()` — every tool from currently available integrations (flattened).
- `report()` / `health_report()` — human-facing connected/unconnected text.
- `tool_levels()` — tool name → numeric action level.

In `src/agent.py`, one `ToolGate` is created and shared by the browser tools,
Developer Mode tools, and every integration. The agent exposes exactly
`registry.tools()`, so a new integration's tools appear in voice automatically
once registered.

`ToolGroup` wraps pre-existing tool lists (browser, developer_mode, core) so
they appear in the registry like any other integration, without losing their
own gating.

## The Integration contract

```python
class XIntegration(Integration):
    name = "myservice"                     # registry key
    description = "What JARVIS does with it."
    read_only = True                       # True means "never modifies"
    required_env = ("X_TOKEN",)            # => is_authenticated
    default_level = ActionLevel.SAFE_READ  # level of any tool without an explicit entry
    _TOOLS = ("myservice_status",)         # names of the @function_tool() methods
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {"myservice_status": ActionLevel.SAFE_READ}

    @function_tool()
    async def myservice_status(self, context: RunContext) -> dict[str, Any]:
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        ...
```

Rules:

- Every tool starts with `self.gate.ensure_action_level(int(ActionLevel.<LEVEL>))`.
- Override `is_authenticated()`, `status_note()`, `health_check()`, and
  register in `build_default_registry()`.
- An unauthenticated integration reports `available but not connected` and its
  tools return an honest `not_connected` dict. Never invent content.
- Secrets come only from environment variables / config files, never code.

## Current capabilities

| Capability      | Level | Read-only | Works now                                  | After connection            |
|-----------------|------:|:---------:|--------------------------------------------|-----------------------------|
| Windows         |   1–2 | No        | Media keys, open known apps, window focus  | —                           |
| Spotify         |   1–2 | No        | Local desktop control + media keys         | Web API: realtime queue etc.|
| Gmail           |     1 | Yes       | —                                          | Read/recent/search/unread   |
| Google Calendar |     1 | Yes       | —                                          | Events, next, free-busy     |
| Meta Ads        |     1 | Yes       | —                                          | Overview, campaigns, insights, delivery status |

### Spotify

Local media control works with no configuration:

- `open_spotify`, `media_play_pause`, `media_next`, `media_previous`,
  `volume_up/down` (steps clamped 1–20), `media_mute`, `spotify_status`.
- `read_only = False`; these are reversible L2 actions.

To enable the Web API path set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`.
Until then `spotify_status` says `local control available, API not connected`.

### Gmail / Google Calendar / Meta Ads (read-only, env-gated)

Until configured, every tool returns:

```
{"status": "not_connected", "message": "... Do not invent <data>."}
```

- **Gmail** — `read_recent_emails`, `search_emails`, `read_email`,
  `list_unread_emails`, `summarise_recent_emails`. Limit clamped to 1–20.
  Gate on `GMAIL_OAUTH_CREDENTIALS` (path to an OAuth client JSON) and the
  token file at `~/.jarvis/gmail_token.json` (override `GMAIL_TOKEN_PATH`).
- **Google Calendar** — `calendar_events`, `next_appointments`,
  `calendar_free_check`. Env: `GOOGLE_OAUTH_CREDENTIALS` and the token at
  `~/.jarvis/calendar_token.json` (override `CALENDAR_TOKEN_PATH`).
- **Meta Ads** — `meta_ads_overview`, `meta_ads_campaigns`, `meta_ads_insights`,
  `meta_ads_why_inactive`. Env: `META_ACCESS_TOKEN`, `META_AD_ACCOUNT_ID`.
  `meta_ads_why_inactive` reports a delivery reason only when the API provides
  one; otherwise it states the cause is **not confirmed**.

All three `client.py` files are scaffolds: data methods raise
`XNotAuthenticatedError` until configured and `NotImplementedError` until a
real client is implemented. Implement the client method and the voice tools
come alive without touching the registry.

## Connecting a service (Developer Mode)

Say **"Jarvis, developer mode: create an integration for Philips Hue"**. Developer
Mode:

1. Detects the request and deterministically scaffolds `src/integrations/philips_hue/`
   (client, integration, `__init__`, tests, docs) — no LLM involved in generation.
2. Runs the work through the normal pipeline: private branch → apply → full check
   suite (lint/format/import/pytest) → commit → your approval to merge.

Generated code is lint- and format-clean by construction and registers the new
service's status tool. Extend `_TOOLS`/`_LEVELS` and the client to add real
capabilities. Traded phrases: "create/add/build/make an integration for X".

For integrations planned outside Developer Mode, give the model the guide in
`integration_scaffold.framework_guide()`.

## Autonomous maintenance (opt-in)

Maintenance is **disabled by default** and state lives in
`improvement_queue/maintenance.json`. It never edits code itself: when enabled
it turns measured failure statistics into `suggestion` tasks that flow through
the normal approval workflow.

- Enable/disable/report: `Jarvis, enable maintenance` / `disable` / `status`.
- Interval (default 60 min, clamped 5–1440): `Jarvis, set maintenance to every 30 minutes`
  — or `JARVIS_MAINTENANCE_INTERVAL_MINUTES` (or `dev_maintenance(action="interval", minutes=30)`).
- The scheduler polls slowly in the background, and only acts when enabled *and*
  due. Failure categories now include `spotify` and `email` in addition to the
  existing browser/startup groups.

## Environment variables

All optional — nothing here is required for JARVIS to start.

| Variable                            | Service       | Purpose                          |
|-------------------------------------|---------------|----------------------------------|
| `SPOTIFY_CLIENT_ID` / `_SECRET`     | Spotify       | Web API control                  |
| `GMAIL_OAUTH_CREDENTIALS`           | Gmail         | Path to OAuth client JSON        |
| `GMAIL_TOKEN_PATH`                  | Gmail         | Stored token (default `~/.jarvis/gmail_token.json`) |
| `GOOGLE_OAUTH_CREDENTIALS`          | Calendar      | Path to OAuth client JSON        |
| `CALENDAR_TOKEN_PATH`               | Calendar      | Stored token (default `~/.jarvis/calendar_token.json`) |
| `META_ACCESS_TOKEN`                 | Meta Ads      | Marketing API token              |
| `META_AD_ACCOUNT_ID`                | Meta Ads      | Ad account id                    |
| `JARVIS_MAINTENANCE_INTERVAL_MINUTES` | Maintenance | Default interval (default 60)    |

Do not commit real credentials. Copy `.env.example` to `.env.local` and fill
the values you want to connect.

## Adding a future integration checklist

1. `src/integrations/<slug>/client.py` — auth check + honest error, real methods.
2. `src/integrations/<slug>/integration.py` — `Integration` subclass, tools with
   `ensure_action_level`, honest unauthenticated answers.
3. Register it in `build_default_registry()` in `src/integrations/__init__.py`.
4. Tests: unconfigured status honesty, tool levels, behaviours (see
   `tests/test_integrations_status.py` for patterns).
5. Run `uv run ruff check src tests`, `uv run ruff format --check src tests`,
   `uv run pytest -q`, then restart the agent.

New capabilities are only enabled after **your explicit approval** at the voice
gate (L3/L4), never on their own.