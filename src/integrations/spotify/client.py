"""Spotify Web API client scaffold.

The conversational interface never talks to the API directly; it goes through
:class:`SpotifyIntegration`, which currently drives the local Spotify desktop
app via media keys. When ``SPOTIFY_CLIENT_ID`` and ``SPOTIFY_CLIENT_SECRET`` are
configured later, the same integration methods will delegate here without the
user noticing any change in how they talk to JARVIS.

Nothing here requires credentials to exist. Every method raises
:class:`SpotifyNotAuthenticatedError` until the API is connected.
"""

from __future__ import annotations

import os


class SpotifyNotAuthenticatedError(RuntimeError):
    """Raised when a Spotify Web API call is attempted before connection."""


REQUIRED_ENV = ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET")


def _configured() -> bool:
    return all(bool(os.environ.get(name, "").strip()) for name in REQUIRED_ENV)


class SpotifyApiClient:
    """Ready-for-connection client abstraction (read + playback)."""

    def __init__(self) -> None:
        self._configured = _configured()

    def is_authenticated(self) -> bool:
        return self._configured

    async def search(self, query: str, limit: int = 5) -> dict:
        self._require_authenticated()
        raise NotImplementedError(
            "Spotify Web API search is implemented after credentials are connected."
        )

    async def current_playback(self) -> dict:
        self._require_authenticated()
        raise NotImplementedError(
            "Spotify Web API playback status is implemented after connection."
        )

    async def devices(self) -> list[dict]:
        self._require_authenticated()
        raise NotImplementedError(
            "Spotify Web API device listing is implemented after connection."
        )

    async def play(
        self, *, context_uri: str | None = None, uris: list[str] | None = None
    ) -> dict:
        self._require_authenticated()
        raise NotImplementedError(
            "Spotify Web API playback control is implemented after connection."
        )

    async def pause(self) -> dict:
        self._require_authenticated()
        raise NotImplementedError(
            "Spotify Web API pause is implemented after connection."
        )

    async def next_track(self) -> dict:
        self._require_authenticated()
        raise NotImplementedError(
            "Spotify Web API next-track is implemented after connection."
        )

    async def volume(self, percent: int) -> dict:
        self._require_authenticated()
        raise NotImplementedError(
            "Spotify Web API volume control is implemented after connection."
        )

    def _require_authenticated(self) -> None:
        if not self._configured:
            raise SpotifyNotAuthenticatedError(
                "Spotify Web API is available but not connected. Configure "
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to enable API-based control."
            )
