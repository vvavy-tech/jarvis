"""Gmail integration (read-only scaffold)."""

from integrations.gmail.client import (
    DEFAULT_TOKEN_PATH,
    GmailClient,
    GmailNotAuthenticatedError,
)
from integrations.gmail.integration import GmailIntegration

__all__ = [
    "DEFAULT_TOKEN_PATH",
    "GmailClient",
    "GmailIntegration",
    "GmailNotAuthenticatedError",
]
