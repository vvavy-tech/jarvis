# Spotify

Slug: `spotify` — package `src/integrations/spotify/`

## What works with no configuration

Local desktop control of the installed Spotify app via Windows media keys:

- `spotify_status` — reports whether local control is available and API state.
- `media_play_pause`, `media_next`, `media_previous` — control playback.
- `volume_up` / `volume_down` (steps clamped 1-20), `media_mute` (toggles).
- `open_spotify` — launches the desktop app (finds the exe, falls back to the
  `spotify:` URI for the Microsoft Store install).

Local control uses `win32api.keybd_event` media keys, so it works while the
Spotify app has focus in the foreground in most cases; it is reported per status.

## Enabling the Web API (optional)

Gives search, play-by-name, and current-playback abilities over the network.

1. Create an app at https://developer.spotify.com/dashboard (type: Web API).
2. Add your machine's URL as a Redirect URI (the login flow is configured by
   `.env.local`).
3. Set in `.env.local`:

   ```
   SPOTIFY_CLIENT_ID=...
   SPOTIFY_CLIENT_SECRET=...
   ```

4. Implement `SpotifyClient._ensure_token` (currently a stub that returns
   unauthenticated) to perform the Authorization Code + PKCE flow.

## Tests

`tests/test_integrations_status.py` (`TestSpotify`) covers status honesty,
media-key mapping, volume clamping, and the non-Windows guard.