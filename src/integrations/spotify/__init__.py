"""Spotify integration (local media control + API-ready client)."""

from integrations.spotify.client import (
    REQUIRED_ENV,
    SpotifyApiClient,
    SpotifyNotAuthenticatedError,
)
from integrations.spotify.integration import SpotifyIntegration

__all__ = [
    "REQUIRED_ENV",
    "SpotifyApiClient",
    "SpotifyIntegration",
    "SpotifyNotAuthenticatedError",
]
