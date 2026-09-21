"""Shopify client (read-only, GraphQL Admin API).

Supports two authentication modes:

**Custom App (recommended):** Set ``SHOPIFY_ADMIN_TOKEN`` with the permanent
Admin API access token from the Shopify Custom App dashboard. No OAuth
exchange is needed; the token is used directly in the
``X-Shopify-Access-Token`` header.

**Public App (OAuth):** Set ``SHOPIFY_CLIENT_ID`` and
``SHOPIFY_CLIENT_SECRET`` to perform a client-credentials token exchange
with Shopify's OAuth endpoint. Tokens are cached in memory and refreshed
automatically.

Any attempt to send a mutation is rejected locally before a network call.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger("agent")

# Custom app mode only needs the admin token + store.
_CUSTOM_APP_ENV = ("SHOPIFY_ADMIN_TOKEN", "SHOPIFY_STORE")
# Public OAuth mode needs client credentials + store.
_OAUTH_ENV = ("SHOPIFY_CLIENT_ID", "SHOPIFY_CLIENT_SECRET", "SHOPIFY_STORE")

# Prefixes for Shopify permanent / custom app tokens.
_SHOPIFY_TOKEN_PREFIXES = ("shpat_", "shpss_")

# Default API version if not set in the environment.
DEFAULT_API_VERSION = "2026-07"


class ShopifyNotAuthenticatedError(RuntimeError):
    """Raised when a Shopify call is attempted before authentication."""


class ShopifyMutationError(RuntimeError):
    """Raised when a mutation attempt is detected locally."""


class ShopifyAPIError(RuntimeError):
    """Raised when the Shopify API cannot fulfil a request."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_COMMON_TLDS = frozenset(
    (
        "com",
        "co",
        "net",
        "org",
        "io",
        "dev",
        "store",
        "shop",
        "app",
        "co.uk",
        "co.za",
    )
)


def _normalize_store(raw: str | None) -> str | None:
    """Return a bare shop subdomain (no .myshopify.com, no protocol, no TLD)."""
    if not raw:
        return None
    store = raw.strip().rstrip("/")
    for prefix in ("https://", "http://"):
        if store.startswith(prefix):
            store = store[len(prefix) :]
            break
    store = store.split("/")[0]
    store = store.removesuffix(".myshopify.com")
    # If the user entered a custom domain (e.g. "lunoacare.com"),
    # strip the TLD so we get the actual myshopify subdomain.
    if "." in store:
        parts = store.rsplit(".", 1)
        if len(parts) == 2 and parts[1].lower() in _COMMON_TLDS:
            store = parts[0]
    return store if store else None


def _configured() -> bool:
    return _is_custom_app_configured() or _is_oauth_configured()


def _is_custom_app_configured() -> bool:
    """Custom App mode: permanent Admin API token set directly."""
    return all(bool(os.environ.get(name, "").strip()) for name in _CUSTOM_APP_ENV)


def _is_oauth_configured() -> bool:
    """Public App mode: OAuth client credentials set."""
    return all(bool(os.environ.get(name, "").strip()) for name in _OAUTH_ENV)


def _shop_base() -> str | None:
    """Return the normalised shop hostname (without protocol).

    If ``SHOPIFY_STORE_URL`` is set, it is used directly (useful when the
    actual myshopify subdomain differs from the custom domain).
    Otherwise the hostname is derived from ``SHOPIFY_STORE``.
    """
    direct = os.environ.get("SHOPIFY_STORE_URL", "").strip()
    if direct:
        store = direct
        for prefix in ("https://", "http://"):
            if store.startswith(prefix):
                store = store[len(prefix) :]
                break
        store = store.split("/")[0]
        store = store.removesuffix(".myshopify.com")
        # Also strip TLDs from the override (e.g. "mystore.com" → "mystore")
        if "." in store:
            parts = store.rsplit(".", 1)
            if len(parts) == 2 and parts[1].lower() in _COMMON_TLDS:
                store = parts[0]
        return f"{store}.myshopify.com" if store else None
    store = _normalize_store(os.environ.get("SHOPIFY_STORE"))
    if not store:
        return None
    return f"{store}.myshopify.com"


# ---------------------------------------------------------------------------
# token management
# ---------------------------------------------------------------------------


class _TokenCache:
    """In-memory access-token cache with automatic expiry."""

    # Client-credentials tokens expire after ~24h; refresh 10 minutes early.
    _EXPIRY_BUFFER_S = 600

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0

    def get(self) -> str | None:
        if self._token and time.monotonic() < self._expires_at:
            return self._token
        return None

    def set(self, token: str, ttl_seconds: float = 82800) -> None:
        """Store the token with a TTL (default ~23h, leaving 10min buffer)."""
        self._token = token
        self._expires_at = time.monotonic() + ttl_seconds

    def clear(self) -> None:
        self._token = None
        self._expires_at = 0.0


# ---------------------------------------------------------------------------
# async transport
# ---------------------------------------------------------------------------


class _AsyncHTTPXTransport:
    """Thin async wrapper so the client can be tested with a fake transport.

    Uses explicit certifi CA bundle and disables HTTP/2 to avoid TLS
    negotiation issues with Shopify's servers.
    """

    def __init__(self) -> None:
        import certifi

        self._verify = certifi.where()

    async def post(
        self,
        url: str,
        *,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        async with httpx.AsyncClient(
            timeout=30.0, verify=self._verify, http2=False
        ) as client:
            return await client.post(url, data=data, json=json, headers=headers)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


class ShopifyClient:
    """Read-only GraphQL client for the Shopify Admin API.

    Supports two authentication modes:

    **Custom App** (via ``SHOPIFY_ADMIN_TOKEN``): permanent Admin API token
    from the Shopify Dev Dashboard. No OAuth exchange needed.

    **Public App** (via ``SHOPIFY_CLIENT_ID`` + ``SHOPIFY_CLIENT_SECRET``):
    Shopify Dev Dashboard client-credentials grant. The client exchanges
    credentials for a short-lived access token, which is cached and refreshed
    automatically.
    """

    def __init__(
        self,
        *,
        transport: Any | None = None,
        api_version: str | None = None,
        token_cache: _TokenCache | None = None,
    ) -> None:
        self._shop_base = _shop_base()
        self._api_version = api_version or os.environ.get(
            "SHOPIFY_API_VERSION", DEFAULT_API_VERSION
        )
        self._transport = transport or _AsyncHTTPXTransport()

        # --- Custom App mode (permanent Admin API token) --- #
        self._admin_token = os.environ.get("SHOPIFY_ADMIN_TOKEN", "").strip()
        if self._admin_token:
            logger.info("shopify auth mode: custom_app (SHOPIFY_ADMIN_TOKEN)")

        # --- Public App mode (Dev Dashboard client-credentials grant) --- #
        self._client_id = os.environ.get("SHOPIFY_CLIENT_ID", "").strip()
        self._client_secret = os.environ.get("SHOPIFY_CLIENT_SECRET", "").strip()
        if self._client_id and self._client_secret:
            logger.info("shopify auth mode: client_credentials (Dev Dashboard)")

        self._cache = token_cache or _TokenCache()

    # --- auth ---------------------------------------------------------- #

    def is_authenticated(self) -> bool:
        if self._shop_base is None:
            return False
        if self._admin_token:
            return True
        return bool(self._client_id and self._client_secret)

    def _require_authenticated(self) -> None:
        if not self.is_authenticated():
            raise ShopifyNotAuthenticatedError(
                "Shopify is available but not connected. Configure either "
                "SHOPIFY_ADMIN_TOKEN (Custom App) or "
                "SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET (Dev Dashboard), "
                "plus SHOPIFY_STORE or SHOPIFY_STORE_URL, to enable Shopify access."
            )

    async def _token(self) -> str:
        """Return the current access token, refreshing if necessary."""
        self._require_authenticated()
        # Custom App: permanent token, no refresh needed.
        if self._admin_token:
            return self._admin_token
        # Dev Dashboard client-credentials: cached or exchange.
        cached = self._cache.get()
        if cached:
            return cached
        return await self._get_access_token()

    async def _get_access_token(self) -> str:
        """Exchange Dev Dashboard client credentials for an access token.

        POST /admin/oauth/access_token with
        grant_type=client_credentials, client_id, client_secret
        (Content-Type: application/x-www-form-urlencoded)
        """
        url = f"https://{self._shop_base}/admin/oauth/access_token"
        payload = (
            f"grant_type=client_credentials"
            f"&client_id={self._client_id}"
            f"&client_secret={self._client_secret}"
        )
        logger.info(
            "SHOPIFY_TOKEN_EXCHANGE shop=%s url=%s",
            self._shop_base,
            url,
        )
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            resp = await self._transport.post(url, data=payload, headers=headers)
        except Exception as exc:
            raise ShopifyAPIError(f"Token exchange failed: {exc}") from exc
        logger.info("SHOPIFY_TOKEN_EXCHANGE_RESULT status=%s", resp.status_code)
        if resp.status_code >= 400:
            body = self._safe_body(resp)
            raise ShopifyAPIError(
                f"Token exchange error (HTTP {resp.status_code}): {body}"
            )
        data = resp.json()
        token = data.get("access_token", "")
        if not token:
            raise ShopifyAPIError("Token exchange returned no access_token")
        ttl = data.get("expires_in", 82800)
        self._cache.set(token, ttl_seconds=float(ttl))
        logger.info("shopify: token acquired, ttl=%ss", ttl)
        return token

    # --- graphql ------------------------------------------------------- #

    async def execute(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict:
        """Execute a read-only GraphQL query.

        Mutations are rejected locally before any network call.
        """
        self._require_authenticated()
        self._guard_no_mutation(query)
        token = await self._token()
        url = f"https://{self._shop_base}/admin/api/{self._api_version}/graphql.json"
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": token,
        }
        body: dict[str, Any] = {"query": query}
        if variables:
            body["variables"] = variables
        try:
            resp = await self._transport.post(url, json=body, headers=headers)
        except Exception as exc:
            raise ShopifyAPIError(f"GraphQL request failed: {exc}") from exc
        if resp.status_code >= 400:
            raw = self._safe_body(resp)
            raise ShopifyAPIError(f"Shopify API error (HTTP {resp.status_code}): {raw}")
        result = resp.json()
        if "errors" in result:
            raise ShopifyAPIError(
                f"GraphQL errors: {json.dumps(result['errors'], ensure_ascii=False)[:500]}"
            )
        return result.get("data", {})

    def _guard_no_mutation(self, query: str) -> None:
        """Reject any GraphQL operation containing a mutation keyword."""
        if "mutation" in query.lower():
            raise ShopifyMutationError(
                "Mutations are not allowed. Shopify integration is read-only."
            )

    # --- helpers ------------------------------------------------------- #

    def _safe_body(self, resp: Any) -> str:
        """Return a truncated response body with secrets redacted."""
        try:
            body = resp.text
        except Exception:
            return "unknown error"
        for secret in (self._client_secret, (self._cache.get() or "")):
            if secret and secret in body:
                body = body.replace(secret, "<redacted>")
        return (body[:300] or "unknown error").strip()
