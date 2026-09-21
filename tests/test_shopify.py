"""Tests for the read-only Shopify integration.

These tests mock the network layer; no real HTTP calls are made.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import pytest

from gates import has_wake_word, is_memory_request
from integrations.shopify.client import (
    ShopifyAPIError,
    ShopifyClient,
    ShopifyMutationError,
    _normalize_store,
    _TokenCache,
)
from integrations.shopify.integration import ShopifyIntegration

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: Any, status_code: int) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> Any:
        return self._body

    @property
    def text(self) -> str:
        import json

        return json.dumps(self._body)


class _FakeTransport:
    """Mimics httpx AsyncClient.post responses for testing."""

    def __init__(
        self,
        token_response: dict | None = None,
        graphql_response: dict | None = None,
        token_status: int = 200,
        graphql_status: int = 200,
        raise_on_post: Exception | None = None,
    ) -> None:
        self._token_response = token_response or {}
        self._graphql_response = graphql_response or {}
        self._token_status = token_status
        self._graphql_status = graphql_status
        self._raise_on_post = raise_on_post
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if self._raise_on_post:
            raise self._raise_on_post
        self.calls.append({"url": url, "data": data, "json": json, "headers": headers})
        if "oauth/access_token" in url:
            return _FakeResponse(self._token_response, self._token_status)
        return _FakeResponse(self._graphql_response, self._graphql_status)


class _ActiveGate:
    def ensure_active_conversation(self) -> None:
        return None

    def ensure_action_level(self, level: int, requested: bool = False) -> None:
        return None


class _ShopifyClientFake:
    def __init__(
        self,
        result: dict | None = None,
        is_authenticated: bool = True,
        raise_on_execute: Exception | None = None,
    ) -> None:
        self._result = result or {}
        self._auth = is_authenticated
        self._raise = raise_on_execute
        self.calls: list[dict[str, Any]] = []

    def is_authenticated(self) -> bool:
        return self._auth

    async def execute(self, query: str, variables: dict | None = None) -> dict:
        if self._raise:
            raise self._raise
        self.calls.append({"query": query, "variables": variables})
        return self._result


class _CustomerPagesFake:
    """Paginating fake for ``shopify_new_customers`` tests.

    Each call to ``execute`` returns the next page in order. The final page is
    returned for every subsequent call (rather than raising) so the test
    exercises the "next page not requested" logic cleanly.
    """

    def __init__(
        self, pages: list[dict[str, Any]], *, is_authenticated: bool = True
    ) -> None:
        self._pages = pages
        self._auth = is_authenticated
        self.calls: list[dict[str, Any]] = []

    def is_authenticated(self) -> bool:
        return self._auth

    async def execute(self, query: str, variables: dict | None = None) -> dict:
        self.calls.append({"query": query, "variables": variables or {}})
        idx = min(len(self.calls) - 1, len(self._pages) - 1)
        return self._pages[idx]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_client(transport=None, token_cache=None, **env_overrides):
    import os

    for key in (
        "SHOPIFY_STORE",
        "SHOPIFY_CLIENT_ID",
        "SHOPIFY_CLIENT_SECRET",
        "SHOPIFY_API_VERSION",
        "SHOPIFY_ADMIN_TOKEN",
    ):
        if key not in env_overrides:
            os.environ.pop(key, None)
    for k, v in env_overrides.items():
        if v is not None:
            os.environ[k] = v
    return ShopifyClient(transport=transport, token_cache=token_cache)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


class TestStoreNormalization:
    def test_bare_subdomain(self):
        assert _normalize_store("lunoacare") == "lunoacare"

    def test_custom_domain_strips_tld(self):
        assert _normalize_store("lunoacare.com") == "lunoacare"

    def test_custom_domain_co_uk(self):
        # .co.uk is a multi-part TLD; the code only strips single-part TLDs.
        # The user should enter the myshopify subdomain directly for such cases.
        assert _normalize_store("mystore.co.uk") == "mystore.co.uk"

    def test_full_hostname(self):
        assert _normalize_store("lunoacare.com") == "lunoacare"

    def test_full_myshopify(self):
        assert _normalize_store("lunoacare.myshopify.com") == "lunoacare"

    def test_with_protocol(self):
        assert _normalize_store("https://lunoacare.myshopify.com") == "lunoacare"

    def test_with_path(self):
        assert _normalize_store("https://lunoacare.com/admin") == "lunoacare"

    def test_with_protocol_custom_domain(self):
        assert _normalize_store("https://mystore.com") == "mystore"

    def test_with_protocol_custom_domain_path(self):
        assert _normalize_store("https://mystore.com/admin") == "mystore"

    def test_store_url_override(self, monkeypatch):
        """SHOPIFY_STORE_URL overrides derived hostname."""
        monkeypatch.setenv("SHOPIFY_STORE", "lunoacare.com")
        monkeypatch.setenv("SHOPIFY_STORE_URL", "zrrhmw-gq.myshopify.com")
        from integrations.shopify.client import _shop_base

        assert _shop_base() == "zrrhmw-gq.myshopify.com"

    def test_store_url_override_with_protocol(self, monkeypatch):
        monkeypatch.setenv("SHOPIFY_STORE_URL", "https://myshop.myshopify.com")
        from integrations.shopify.client import _shop_base

        assert _shop_base() == "myshop.myshopify.com"

    def test_store_url_override_custom_domain(self, monkeypatch):
        """Even a custom domain in STORE_URL is treated as the subdomain."""
        monkeypatch.setenv("SHOPIFY_STORE_URL", "mystore.com")
        from integrations.shopify.client import _shop_base

        # The override strips TLDs just like normal normalization
        assert _shop_base() == "mystore.myshopify.com"

    def test_none_returns_none(self):
        assert _normalize_store(None) is None

    def test_empty_returns_none(self):
        assert _normalize_store("") is None


# ---------------------------------------------------------------------------
# token cache
# ---------------------------------------------------------------------------


class TestTokenCache:
    def test_get_returns_none_when_empty(self):
        assert _TokenCache().get() is None

    def test_set_and_get(self):
        cache = _TokenCache()
        cache.set("tok123", ttl_seconds=3600)
        assert cache.get() == "tok123"

    def test_expired_token_returns_none(self, monkeypatch):
        base = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: base)
        cache = _TokenCache()
        cache.set("tok123", ttl_seconds=10)
        monkeypatch.setattr(time, "monotonic", lambda: base + 20)
        assert cache.get() is None

    def test_clear_removes_token(self):
        cache = _TokenCache()
        cache.set("tok123")
        cache.clear()
        assert cache.get() is None


# ---------------------------------------------------------------------------
# auth flow
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_unconfigured_is_not_authenticated(self):
        client = _make_client()
        assert client.is_authenticated() is False

    def test_custom_app_is_authenticated(self):
        client = _make_client(
            SHOPIFY_STORE="testshop",
            SHOPIFY_ADMIN_TOKEN="shpat_abc123",
        )
        assert client.is_authenticated() is True

    def test_oauth_is_authenticated(self):
        client = _make_client(
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        assert client.is_authenticated() is True

    def test_missing_secret_is_not_authenticated(self):
        client = _make_client(
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
        )
        assert client.is_authenticated() is False


class TestTokenExchange:
    async def test_exchange_returns_token(self):
        transport = _FakeTransport(
            token_response={"access_token": "shpat_xyz", "expires_in": 82800}
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        token = await client._get_access_token()
        assert token == "shpat_xyz"
        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert "oauth/access_token" in call["url"]
        assert call["data"]
        data_str = str(call["data"])
        assert "grant_type=client_credentials" in data_str
        assert "client_id=cid" in data_str

    async def test_exchange_error_raises(self):
        transport = _FakeTransport(
            token_response={"error": "invalid_client"}, token_status=401
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        with pytest.raises(ShopifyAPIError, match="Token exchange error"):
            await client._get_access_token()

    async def test_network_failure_raises(self):
        transport = _FakeTransport(raise_on_post=ConnectionError("no network"))
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        with pytest.raises(ShopifyAPIError, match="Token exchange failed"):
            await client._get_access_token()


class TestCustomAppAuth:
    async def test_custom_app_token_used_directly(self):
        """No OAuth exchange; the admin token is returned immediately."""
        transport = _FakeTransport(
            token_response={"access_token": "should_not_be_called"}
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_ADMIN_TOKEN="shpat_direct123",
        )
        token = await client._token()
        assert token == "shpat_direct123"
        # No OAuth exchange should have happened
        token_calls = [c for c in transport.calls if "oauth/access_token" in c["url"]]
        assert len(token_calls) == 0


class TestTokenCaching:
    async def test_token_cached_after_first_exchange(self):
        transport = _FakeTransport(
            token_response={"access_token": "shpat_abc", "expires_in": 3600}
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        t1 = await client._token()
        t2 = await client._token()
        assert t1 == t2 == "shpat_abc"
        token_calls = [c for c in transport.calls if "oauth/access_token" in c["url"]]
        assert len(token_calls) == 1

    async def test_token_refreshed_after_expiry(self, monkeypatch):
        base = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: base)
        transport = _FakeTransport(
            token_response={"access_token": "shpat_new", "expires_in": 10}
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        t1 = await client._token()
        monkeypatch.setattr(time, "monotonic", lambda: base + 20)
        t2 = await client._token()
        assert t1 == t2 == "shpat_new"
        token_calls = [c for c in transport.calls if "oauth/access_token" in c["url"]]
        assert len(token_calls) == 2


# ---------------------------------------------------------------------------
# secrets never logged
# ---------------------------------------------------------------------------


class TestSecretsNotLogged:
    def test_client_secret_not_in_safe_body(self):
        transport = _FakeTransport(
            token_response={"access_token": "tok", "expires_in": 3600}
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="supersecret",
        )
        fake_resp = _FakeResponse(
            {"error": "invalid", "details": "secret was supersecret"}, 401
        )
        body = client._safe_body(fake_resp)
        assert "supersecret" not in body
        assert "<redacted>" in body

    async def test_access_token_not_in_safe_body(self):
        transport = _FakeTransport(
            token_response={"access_token": "shpat_hidden", "expires_in": 3600}
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        await client._token()
        fake_resp = _FakeResponse(
            {"errors": [{"message": "token shpat_hidden invalid"}]}, 401
        )
        body = client._safe_body(fake_resp)
        assert "shpat_hidden" not in body
        assert "<redacted>" in body


# ---------------------------------------------------------------------------
# graphql
# ---------------------------------------------------------------------------


class TestGraphQL:
    async def test_correct_endpoint(self):
        transport = _FakeTransport(
            token_response={"access_token": "tok", "expires_in": 3600},
            graphql_response={"data": {"shop": {"name": "Test"}}},
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
            SHOPIFY_API_VERSION="2026-07",
        )
        result = await client.execute("{ shop { name } }")
        assert result == {"shop": {"name": "Test"}}
        graphql_calls = [c for c in transport.calls if "graphql.json" in c["url"]]
        assert len(graphql_calls) == 1
        assert "2026-07" in graphql_calls[0]["url"]
        assert graphql_calls[0]["headers"]["X-Shopify-Access-Token"] == "tok"

    async def test_mutation_rejected_locally(self):
        client = _make_client(
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        with pytest.raises(ShopifyMutationError, match="read-only"):
            await client.execute("mutation { productUpdate ... }")

    async def test_mutation_case_insensitive(self):
        client = _make_client(
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        with pytest.raises(ShopifyMutationError):
            await client.execute("MUTATION { ... }")

    async def test_graphql_api_error_raises(self):
        transport = _FakeTransport(
            token_response={"access_token": "tok", "expires_in": 3600},
            graphql_response={"errors": [{"message": "Field 'x' doesn't exist"}]},
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        with pytest.raises(ShopifyAPIError, match="GraphQL errors"):
            await client.execute("{ x }")

    async def test_graphql_http_error_raises(self):
        transport = _FakeTransport(
            token_response={"access_token": "tok", "expires_in": 3600},
            graphql_response={"errors": [{"message": "Rate limited"}]},
            graphql_status=429,
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        with pytest.raises(ShopifyAPIError, match="HTTP 429"):
            await client.execute("{ shop { name } }")


# ---------------------------------------------------------------------------
# integration tools
# ---------------------------------------------------------------------------


class TestIntegrationTools:
    async def test_not_connected_honest(self):
        fake = _ShopifyClientFake(is_authenticated=False)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_status(None)
        assert result["status"] == "not_connected"
        assert "not connected" in result["message"]

    async def test_status_query_works(self):
        fake = _ShopifyClientFake(
            result={"shop": {"name": "Lunoa Care", "currencyCode": "EUR"}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_status(None)
        assert result["status"] == "ok"
        assert result["data"]["shop"]["name"] == "Lunoa Care"

    async def test_recent_orders_query(self):
        fake = _ShopifyClientFake(
            result={"orders": {"edges": [], "pageInfo": {"hasNextPage": False}}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_recent_orders(None, count=3)
        assert result["status"] == "ok"
        assert len(fake.calls) == 1
        assert "orders" in fake.calls[0]["query"]
        assert fake.calls[0]["variables"]["first"] == 3

    async def test_products_query(self):
        fake = _ShopifyClientFake(
            result={"products": {"edges": [], "pageInfo": {"hasNextPage": False}}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_products(None, count=5)
        assert result["status"] == "ok"
        assert fake.calls[0]["variables"]["first"] == 5

    async def test_inventory_query(self):
        fake = _ShopifyClientFake(result={"products": {"edges": []}})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_inventory(None)
        assert result["status"] == "ok"

    async def test_low_stock_query(self):
        fake = _ShopifyClientFake(result={"products": {"edges": []}})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_low_stock(None, threshold=3)
        assert result["status"] == "ok"

    async def test_top_products_query(self):
        fake = _ShopifyClientFake(result={"products": {"edges": []}})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_top_products(None, count=3)
        assert result["status"] == "ok"
        assert "BEST_SELLING" in fake.calls[0]["query"]

    async def test_sales_summary_query(self):
        fake = _ShopifyClientFake(result={"orders": {"edges": []}})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_sales_summary(None, days=14)
        assert result["status"] == "ok"

    async def test_api_error_graceful(self):
        fake = _ShopifyClientFake(raise_on_execute=ShopifyAPIError("rate limited"))
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_status(None)
        assert result["status"] == "error"
        assert "failed" in result["message"]

    async def test_mutation_error_graceful(self):
        fake = _ShopifyClientFake(raise_on_execute=ShopifyMutationError("read-only"))
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_status(None)
        assert result["status"] == "error"
        assert "read-only" in result["message"]

    async def test_customers_query(self):
        fake = _ShopifyClientFake(
            result={"customers": {"edges": [], "pageInfo": {"hasNextPage": False}}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customers(None, count=5)
        assert result["status"] == "ok"
        assert "customers" in fake.calls[0]["query"]
        assert fake.calls[0]["variables"]["first"] == 5

    async def test_marketing_contacts_query(self):
        fake = _ShopifyClientFake(
            result={"customers": {"edges": [], "pageInfo": {"hasNextPage": False}}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_marketing_contacts(
            None, count=10, subscribed_only=True
        )
        assert result["status"] == "ok"
        assert "emailMarketingConsent" in fake.calls[0]["query"]
        assert fake.calls[0]["variables"]["first"] == 10
        assert fake.calls[0]["variables"]["filter"] == "accepts_marketing:true"

    async def test_marketing_contacts_includes_unsubscribed(self):
        fake = _ShopifyClientFake(
            result={"customers": {"edges": [], "pageInfo": {"hasNextPage": False}}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_marketing_contacts(
            None, count=10, subscribed_only=False
        )
        assert result["status"] == "ok"
        assert fake.calls[0]["variables"]["filter"] is None


# ---------------------------------------------------------------------------
# no write tools exist
# ---------------------------------------------------------------------------


class TestNoWriteTools:
    def test_all_tools_are_safe_read(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        for name in integration.tool_names:
            assert int(integration.tool_level(name)) == 1

    def test_read_only_flag(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        assert integration.read_only is True

    async def test_no_mutation_in_any_tool_query(self):
        """Every tool sends only queries, never mutations."""
        fake = _ShopifyClientFake(
            result={"products": {"edges": []}},
            is_authenticated=True,
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        tools = [
            integration.shopify_status,
            integration.shopify_sales_summary,
            integration.shopify_recent_orders,
            integration.shopify_products,
            integration.shopify_inventory,
            integration.shopify_low_stock,
            integration.shopify_top_products,
            integration.shopify_customers,
            integration.shopify_marketing_contacts,
            integration.shopify_customer_count,
            integration.shopify_new_customers,
        ]
        for tool in tools:
            await tool(None)
        for call in fake.calls:
            assert "mutation" not in call["query"].lower()


# ---------------------------------------------------------------------------
# customer result normalisation & privacy
# ---------------------------------------------------------------------------


class TestCustomerResultNormalisation:
    async def test_successful_customer_payload(self):
        fake = _ShopifyClientFake(
            result={
                "customers": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/Customer/1",
                                "displayName": "Alice Test",
                                "email": "alice@example.com",
                                "createdAt": "2024-01-15",
                                "numberOfOrders": 3,
                                "totalSpent": {
                                    "amount": "150.00",
                                    "currencyCode": "EUR",
                                },
                            }
                        }
                    ],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customers(None, count=5)
        assert result["status"] == "ok"
        assert result["count"] == 1
        assert "customers" in result
        c = result["customers"][0]
        assert c["name"] == "Alice Test"
        assert c["email"] == "a***@example.com"
        assert c["orders"] == 3
        assert c["total_spent"] == "150.00 EUR"
        assert c["created"] == "2024-01-15"

    async def test_graphql_validation_error_returns_error_status(self):
        fake = _ShopifyClientFake(
            raise_on_execute=ShopifyAPIError(
                'GraphQL errors: [{"message": "Field \'xyz\' doesn\'t exist"}]'
            )
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customers(None)
        assert result["status"] == "error"
        assert "failed" in result["message"]

    async def test_access_denied_returns_error_status(self):
        fake = _ShopifyClientFake(
            raise_on_execute=ShopifyAPIError(
                "Shopify API error (HTTP 403): ACCESS_DENIED"
            )
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customers(None)
        assert result["status"] == "error"
        assert "ACCESS_DENIED" in result["message"]

    async def test_null_email_shows_scope_hint(self):
        fake = _ShopifyClientFake(
            result={
                "customers": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/Customer/1",
                                "displayName": "Bob Test",
                                "email": None,
                                "createdAt": "2024-03-01",
                                "numberOfOrders": 1,
                                "totalSpent": {
                                    "amount": "50.00",
                                    "currencyCode": "USD",
                                },
                            }
                        }
                    ],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customers(None)
        assert result["status"] == "ok"
        assert result["count"] == 1
        c = result["customers"][0]
        assert "email unavailable" in c["email"]
        assert "read_customer_email" in c["email"]
        assert "note" in result
        assert "read_customer_email" in result["note"]

    async def test_tool_level_status_distinction(self):
        fake_err = _ShopifyClientFake(raise_on_execute=ShopifyAPIError("timeout"))
        fake_ok = _ShopifyClientFake(
            result={"customers": {"edges": [], "pageInfo": {"hasNextPage": False}}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake_err)
        err_result = await integration.shopify_customers(None)
        assert err_result["status"] == "error"

        integration2 = ShopifyIntegration(gate=_ActiveGate(), client=fake_ok)
        ok_result = await integration2.shopify_customers(None)
        assert ok_result["status"] == "ok"

    async def test_no_email_addresses_in_result(self):
        fake = _ShopifyClientFake(
            result={
                "customers": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/Customer/1",
                                "displayName": "Charlie",
                                "email": "charlie@testdomain.com",
                                "createdAt": "2024-06-01",
                                "numberOfOrders": 0,
                                "totalSpent": {"amount": "0.00", "currencyCode": "EUR"},
                            }
                        }
                    ],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customers(None)
        assert "charlie@testdomain.com" not in str(result)
        assert "c***@testdomain.com" in str(result)

    async def test_marketing_contacts_normalised(self):
        fake = _ShopifyClientFake(
            result={
                "customers": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/Customer/2",
                                "displayName": "Diana",
                                "email": "diana@shop.com",
                                "updatedAt": "2024-07-01",
                                "acceptsMarketing": True,
                                "emailMarketingConsent": {
                                    "marketingState": "SUBSCRIBED",
                                    "optInLevel": "SINGLE_OPT_IN",
                                },
                            }
                        }
                    ],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_marketing_contacts(None, count=5)
        assert result["status"] == "ok"
        assert result["count"] == 1
        c = result["customers"][0]
        assert c["name"] == "Diana"
        assert c["email"] == "d***@shop.com"
        assert c["marketing_state"] == "SUBSCRIBED"
        assert c["opt_in_level"] == "SINGLE_OPT_IN"
        assert c["accepts_marketing"] is True

    async def test_customer_count_exists(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        assert "shopify_customer_count" in integration.tool_names

    async def test_customer_count_uses_customers_count_query(self):
        fake = _ShopifyClientFake(
            result={"customersCount": {"count": 123, "precision": "EXACT"}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "ok"
        assert "customersCount" in fake.calls[0]["query"]

    async def test_exact_successful_customers_count_payload(self):
        fake = _ShopifyClientFake(
            result={"customersCount": {"count": 123, "precision": "EXACT"}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result == {
            "status": "ok",
            "count": 123,
            "precision": "EXACT",
            "message": "Total customer count: 123",
        }

    async def test_count_question_maps_to_count_tool_via_description(self):
        """The tool description must mention count-related phrases."""
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        tool = next(
            t for t in integration.tools if t.info.name == "shopify_customer_count"
        )
        desc = tool.info.description.casefold()
        assert "how many" in desc
        assert "count" in desc

    async def test_count_precision_preserved(self):
        fake = _ShopifyClientFake(
            result={"customersCount": {"count": 456, "precision": "APPROXIMATE"}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "ok"
        assert result["count"] == 456
        assert result["precision"] == "APPROXIMATE"
        assert "approximate" in result["message"].casefold()

    async def test_count_access_denied_returns_error_status(self):
        fake = _ShopifyClientFake(raise_on_execute=ShopifyAPIError("ACCESS_DENIED"))
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "error"
        assert "ACCESS_DENIED" in result["message"]

    async def test_count_graphql_validation_error_returns_error_status(self):
        fake = _ShopifyClientFake(
            raise_on_execute=ShopifyAPIError(
                'GraphQL errors: [{"message": "Field \'customersCount\' doesn\'t exist"}]'
            )
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "error"
        assert "GraphQL" in result["message"]

    async def test_null_customers_count_returns_error_with_code(self):
        fake = _ShopifyClientFake(result={"customersCount": None})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "error"
        assert result["code"] == "count_unavailable"
        assert "read_customers" in result["message"]
        assert result["factually_unavailable"] is True
        assert "DO NOT" in result["message"]

    async def test_missing_customers_count_returns_error_with_code(self):
        fake = _ShopifyClientFake(result={"shop": {"name": "Test"}})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "error"
        assert result["code"] == "count_unavailable"
        assert result["factually_unavailable"] is True

    async def test_count_precision_zero_is_valid(self):
        fake = _ShopifyClientFake(
            result={"customersCount": {"count": 0, "precision": "EXACT"}}
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "ok"
        assert result["count"] == 0
        assert result["precision"] == "EXACT"

    async def test_model_facing_status_distinction_count(self):
        """Success has status=ok, errors have status=error with code."""
        fake_ok = _ShopifyClientFake(
            result={"customersCount": {"count": 50, "precision": "EXACT"}}
        )
        integration_ok = ShopifyIntegration(gate=_ActiveGate(), client=fake_ok)
        ok_result = await integration_ok.shopify_customer_count(None)
        assert ok_result["status"] == "ok"
        assert "count" in ok_result
        assert ok_result.get("code") is None

        fake_err = _ShopifyClientFake(raise_on_execute=ShopifyAPIError("timeout"))
        integration_err = ShopifyIntegration(gate=_ActiveGate(), client=fake_err)
        err_result = await integration_err.shopify_customer_count(None)
        assert err_result["status"] == "error"
        assert "code" in err_result or "message" in err_result

    async def test_no_customer_pii_logged(self):
        """Verify customer emails are never present in the log summary."""
        fake = _ShopifyClientFake(
            result={
                "customers": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/Customer/1",
                                "displayName": "Eve",
                                "email": "eve@example.com",
                                "createdAt": "2024-08-01",
                                "numberOfOrders": 1,
                                "totalSpent": {
                                    "amount": "25.00",
                                    "currencyCode": "EUR",
                                },
                            }
                        }
                    ],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customers(None)
        assert "eve@example.com" not in str(result)
        assert "e***@example.com" in str(result)


# ---------------------------------------------------------------------------
# auth integration: custom app + customer count
# ---------------------------------------------------------------------------


class TestAuthIntegration:
    async def test_custom_app_customer_count_succeeds(self):
        """Custom app mode: no OAuth, direct token, count works."""
        fake = _ShopifyClientFake(
            result={"customersCount": {"count": 42, "precision": "EXACT"}},
            is_authenticated=True,
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_customer_count(None)
        assert result["status"] == "ok"
        assert result["count"] == 42
        assert result["precision"] == "EXACT"

    async def test_invalid_shop_domain_token_exchange_404(self):
        """When the shop hostname is wrong, token exchange returns 404."""
        transport = _FakeTransport(
            token_response={"errors": "Store unavailable"},
            token_status=404,
        )
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="nonexistent-store",
            SHOPIFY_CLIENT_ID="cid",
            SHOPIFY_CLIENT_SECRET="csec",
        )
        with pytest.raises(ShopifyAPIError, match="HTTP 404"):
            await client._get_access_token()

    async def test_oauth_exchange_not_called_in_custom_app_mode(self):
        """Verify OAuth is skipped when admin token is set."""
        transport = _FakeTransport(token_response={"access_token": "never_used"})
        client = _make_client(
            transport=transport,
            SHOPIFY_STORE="testshop",
            SHOPIFY_ADMIN_TOKEN="shpat_direct",
        )
        token = await client._token()
        assert token == "shpat_direct"
        assert len(transport.calls) == 0


# ---------------------------------------------------------------------------
# pagination bounded
# ---------------------------------------------------------------------------


class TestPaginationBounded:
    async def test_recent_orders_capped(self):
        fake = _ShopifyClientFake(result={"orders": {"edges": []}})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        await integration.shopify_recent_orders(None, count=999)
        assert fake.calls[0]["variables"]["first"] <= 50

    async def test_products_capped(self):
        fake = _ShopifyClientFake(result={"products": {"edges": []}})
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        await integration.shopify_products(None, count=999)
        assert fake.calls[0]["variables"]["first"] <= 50


# ---------------------------------------------------------------------------
# shopify_new_customers
# ---------------------------------------------------------------------------


def _created_at(days_back: int, hour: int, minute: int = 0) -> str:
    day = date.today() - timedelta(days=days_back)
    return f"{day.isoformat()}T{hour:02d}:{minute:02d}:00"


class TestNewCustomersTool:
    async def test_tool_registered_as_safe_read(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        assert "shopify_new_customers" in integration.tool_names
        assert int(integration.tool_level("shopify_new_customers")) == 1

    async def test_description_routes_requested_phrases(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        tool = next(
            t for t in integration.tools if t.info.name == "shopify_new_customers"
        )
        desc = tool.info.description.casefold()
        assert "how many new shopify customers did we get today" in desc
        assert "hoeveel nieuwe shopify klanten hebben we vandaag gekregen" in desc
        assert "new customers yesterday" in desc
        assert "signups in the last 7 days" in desc

    async def test_not_about_orders_or_sales(self):
        """The tool must not route sales/order questions to itself."""
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        tool = next(
            t for t in integration.tools if t.info.name == "shopify_new_customers"
        )
        desc = tool.info.description.casefold()
        assert "not about sales" in desc

    async def test_today_counts_only_todays_customers(self):
        fake = _CustomerPagesFake(
            [
                {
                    "customers": {
                        "edges": [
                            {"node": {"createdAt": _created_at(0, 10, 0)}},
                            {"node": {"createdAt": _created_at(0, 8, 0)}},
                            {"node": {"createdAt": _created_at(1, 23, 0)}},
                        ],
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            ]
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="today")
        assert result["status"] == "ok"
        assert result["count"] == 2
        assert result["period"] == "today"
        assert fake.calls[0]["variables"]["first"] == 50
        assert all("email" not in c["query"] for c in fake.calls)

    async def test_yesterday_counts_only_yesterday(self):
        fake = _CustomerPagesFake(
            [
                {
                    "customers": {
                        "edges": [
                            {"node": {"createdAt": _created_at(0, 10, 0)}},
                            {"node": {"createdAt": _created_at(1, 12, 0)}},
                            {"node": {"createdAt": _created_at(1, 0, 0)}},
                            {"node": {"createdAt": _created_at(2, 23, 0)}},
                        ],
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            ]
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="yesterday")
        assert result["status"] == "ok"
        assert result["count"] == 2

    async def test_not_connected_honest(self):
        fake = _ShopifyClientFake(is_authenticated=False)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None)
        assert result["status"] == "not_connected"
        assert "not connected" in result["message"]

    async def test_invalid_period_returns_error_without_network(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="monthly")
        assert result["status"] == "error"
        assert "period" in result["message"]
        assert fake.calls == []

    async def test_custom_requires_start_date(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="custom")
        assert result["status"] == "error"
        assert "start_date" in result["message"]

    async def test_custom_reversed_dates_returns_error(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(
            None, period="custom", start_date="2024-05-10", end_date="2024-05-01"
        )
        assert result["status"] == "error"
        assert "after end_date" in result["message"]

    async def test_custom_range_counts_inside(self):
        fake = _CustomerPagesFake(
            [
                {
                    "customers": {
                        "edges": [
                            {"node": {"createdAt": "2024-05-05T10:00:00"}},
                            {"node": {"createdAt": "2024-05-03T10:00:00"}},
                            {"node": {"createdAt": "2024-04-01T10:00:00"}},
                        ],
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            ]
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(
            None, period="custom", start_date="2024-05-01", end_date="2024-05-05"
        )
        assert result["status"] == "ok"
        assert result["count"] == 2
        assert result["start_date"] == "2024-05-01"
        assert result["end_date"] == "2024-05-05"

    async def test_paginates_all_pages_for_exact_count(self):
        pages = [
            {
                "customers": {
                    "edges": [
                        {"node": {"createdAt": _created_at(0, 10, 0)}},
                        {"node": {"createdAt": _created_at(2, 10, 0)}},
                    ],
                    "pageInfo": {"hasNextPage": True, "endCursor": "page-2"},
                }
            },
            {
                "customers": {
                    "edges": [
                        {"node": {"createdAt": _created_at(5, 10, 0)}},
                        {"node": {"createdAt": _created_at(6, 5, 0)}},
                    ],
                    "pageInfo": {"hasNextPage": True, "endCursor": "page-3"},
                }
            },
            {
                "customers": {
                    "edges": [
                        {"node": {"createdAt": _created_at(7, 10, 0)}},
                    ],
                    "pageInfo": {"hasNextPage": False},
                }
            },
        ]
        fake = _CustomerPagesFake(pages)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="last_7_days")
        assert result["status"] == "ok"
        assert result["count"] == 4
        assert len(fake.calls) == 3
        assert fake.calls[1]["variables"]["after"] == "page-2"
        assert fake.calls[2]["variables"]["after"] == "page-3"
        assert fake.calls[0]["variables"]["first"] == 50

    async def test_stops_paginating_when_out_of_range(self):
        fake = _CustomerPagesFake(
            [
                {
                    "customers": {
                        "edges": [
                            {"node": {"createdAt": _created_at(0, 10, 0)}},
                            {"node": {"createdAt": _created_at(40, 10, 0)}},
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": "page-2"},
                    }
                },
                {
                    "customers": {
                        "edges": [
                            {"node": {"createdAt": _created_at(50, 10, 0)}},
                        ],
                        "pageInfo": {"hasNextPage": False},
                    }
                },
            ]
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="today")
        assert result["status"] == "ok"
        assert result["count"] == 1
        assert len(fake.calls) == 1  # older customer ends the walk; page 2 not fetched

    async def test_zero_new_customers_has_message(self):
        fake = _CustomerPagesFake(
            [
                {
                    "customers": {
                        "edges": [
                            {"node": {"createdAt": _created_at(2, 1, 0)}},
                        ],
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            ]
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="today")
        assert result["status"] == "ok"
        assert result["count"] == 0
        assert "No new customers" in result["message"]

    async def test_graphql_error_graceful(self):
        fake = _ShopifyClientFake(raise_on_execute=ShopifyAPIError("rate limited"))
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None)
        assert result["status"] == "error"
        assert "failed" in result["message"]

    async def test_no_pii_in_query_or_result(self):
        fake = _CustomerPagesFake(
            [
                {
                    "customers": {
                        "edges": [
                            {"node": {"createdAt": _created_at(0, 10, 0)}},
                        ],
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            ]
        )
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_new_customers(None, period="today")
        query = fake.calls[0]["query"]
        assert "email" not in query
        assert "displayName" not in query
        assert "numberOfOrders" not in query
        assert "@" not in str(result)


# ---------------------------------------------------------------------------
# missing credentials
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    async def test_missing_env_graceful(self, monkeypatch):
        for key in (
            "SHOPIFY_STORE",
            "SHOPIFY_CLIENT_ID",
            "SHOPIFY_CLIENT_SECRET",
            "SHOPIFY_ADMIN_TOKEN",
        ):
            monkeypatch.delenv(key, raising=False)
        fake = _ShopifyClientFake(is_authenticated=False)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        result = await integration.shopify_status(None)
        assert result["status"] == "not_connected"


# ---------------------------------------------------------------------------
# voice/wake/memory unchanged
# ---------------------------------------------------------------------------


class TestVoiceWakeMemoryUnchanged:
    """Verify Shopify tools don't interfere with existing gate logic."""

    def test_shopify_tools_not_in_wake_word_list(self):
        assert not has_wake_word("shopify_status")

    def test_shopify_tools_not_memory_requests(self):
        assert not is_memory_request("shopify recent orders")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_shopify_in_default_registry(self, monkeypatch):
        from integrations import build_default_registry

        for key in (
            "SHOPIFY_STORE",
            "SHOPIFY_CLIENT_ID",
            "SHOPIFY_CLIENT_SECRET",
            "SHOPIFY_ADMIN_TOKEN",
        ):
            monkeypatch.delenv(key, raising=False)
        registry = build_default_registry()
        assert "shopify" in registry.names()

    def test_customer_tools_in_tool_names(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        assert "shopify_customers" in integration.tool_names
        assert "shopify_marketing_contacts" in integration.tool_names
        assert "shopify_customer_count" in integration.tool_names
        assert "shopify_new_customers" in integration.tool_names

    def test_customer_tools_are_safe_read(self):
        fake = _ShopifyClientFake(is_authenticated=True)
        integration = ShopifyIntegration(gate=_ActiveGate(), client=fake)
        assert int(integration.tool_level("shopify_customers")) == 1
        assert int(integration.tool_level("shopify_marketing_contacts")) == 1
        assert int(integration.tool_level("shopify_customer_count")) == 1

    def test_custom_app_auth_in_registry(self, monkeypatch):
        from integrations import build_default_registry

        monkeypatch.setenv("SHOPIFY_STORE", "testshop")
        monkeypatch.setenv("SHOPIFY_ADMIN_TOKEN", "shpat_abc123")
        monkeypatch.delenv("SHOPIFY_CLIENT_ID", raising=False)
        monkeypatch.delenv("SHOPIFY_CLIENT_SECRET", raising=False)
        registry = build_default_registry()
        assert "shopify" in registry.names()
