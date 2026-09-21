"""Shopify capability (read-only analytics and inventory via GraphQL).

All tools are safe reads. Output separates factual data from analysis.
Any GraphQL mutation is rejected locally before a network call.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, ClassVar

from livekit.agents import RunContext, function_tool

from integrations.base import ActionLevel, Integration
from integrations.shopify.client import (
    ShopifyAPIError,
    ShopifyClient,
    ShopifyMutationError,
    ShopifyNotAuthenticatedError,
)

logger = logging.getLogger("agent")

_MAX_PAGE = 50

_NEW_CUSTOMERS_QUERY = """
query ($first: Int, $after: String) {
  customers(
    first: $first
    after: $after
    sortKey: CREATED_AT
    reverse: true
  ) {
    edges {
      node {
        createdAt
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def _redact(value: str) -> str:
    """Mask a value to at most 50 chars for safe logging."""
    return value[:50] + ("..." if len(value) > 50 else "")


def _mask_email(email: str) -> str:
    """Return a privacy-safe email representation for the model.

    Shows only the first character of the local part and the full domain.
    Example: j***@example.com
    """
    if "@" not in email:
        return "***@***"
    local, domain = email.split("@", 1)
    first = local[0] if local else ""
    return f"{first}***@{domain}"


class ShopifyIntegration(Integration):
    name = "shopify"
    description = (
        "Read-only Shopify analytics: store status, sales summaries, recent "
        "orders, products, inventory, low-stock alerts, top products, customers, "
        "and marketing contacts. Never modifies orders, inventory, products, "
        "customers, or marketing settings."
    )
    read_only = True
    default_level = ActionLevel.SAFE_READ
    _TOOLS = (
        "shopify_status",
        "shopify_sales_summary",
        "shopify_recent_orders",
        "shopify_products",
        "shopify_inventory",
        "shopify_low_stock",
        "shopify_top_products",
        "shopify_customers",
        "shopify_marketing_contacts",
        "shopify_customer_count",
        "shopify_new_customers",
    )
    _LEVELS: ClassVar[dict[str, ActionLevel]] = {
        "shopify_status": ActionLevel.SAFE_READ,
        "shopify_sales_summary": ActionLevel.SAFE_READ,
        "shopify_recent_orders": ActionLevel.SAFE_READ,
        "shopify_products": ActionLevel.SAFE_READ,
        "shopify_inventory": ActionLevel.SAFE_READ,
        "shopify_low_stock": ActionLevel.SAFE_READ,
        "shopify_top_products": ActionLevel.SAFE_READ,
        "shopify_customers": ActionLevel.SAFE_READ,
        "shopify_marketing_contacts": ActionLevel.SAFE_READ,
        "shopify_customer_count": ActionLevel.SAFE_READ,
        "shopify_new_customers": ActionLevel.SAFE_READ,
    }

    def __init__(
        self,
        *,
        gate: Any | None = None,
        failure_log: Any | None = None,
        client: ShopifyClient | None = None,
    ) -> None:
        super().__init__(gate=gate, failure_log=failure_log)
        self._client = client or ShopifyClient()

    def is_authenticated(self) -> bool:
        return self._client.is_authenticated()

    def status_note(self) -> str:
        return "" if self.is_authenticated() else "available but not connected"

    def _not_connected(self) -> dict[str, str]:
        return {
            "status": "not_connected",
            "message": (
                "Shopify is available but not connected. Configure either "
                "SHOPIFY_ADMIN_TOKEN (Custom App) or "
                "SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET (Dev Dashboard), "
                "plus SHOPIFY_STORE or SHOPIFY_STORE_URL, to enable access."
            ),
        }

    async def _with_client(self, tool: str, query: str, **variables) -> dict[str, Any]:
        """Execute a GraphQL query with graceful error handling."""
        if not self._client.is_authenticated():
            return self._not_connected()
        try:
            result = await self._client.execute(query, variables or None)
        except ShopifyNotAuthenticatedError:
            return self._not_connected()
        except ShopifyMutationError as exc:
            if tool == "shopify_customer_count":
                logger.info(
                    "SHOPIFY_COUNT_EXCEPTION type=%s message=%s",
                    type(exc).__name__,
                    str(exc),
                )
            return {"status": "error", "message": str(exc)}
        except ShopifyAPIError as exc:
            self._log_failure(tool, "graphql", exc)
            if tool == "shopify_customer_count":
                logger.info(
                    "SHOPIFY_COUNT_EXCEPTION type=%s message=%s",
                    type(exc).__name__,
                    str(exc),
                )
            return {"status": "error", "message": f"Shopify lookup failed: {exc}"}
        except Exception as exc:
            self._log_failure(tool, "graphql", exc)
            if tool == "shopify_customer_count":
                logger.info(
                    "SHOPIFY_COUNT_EXCEPTION type=%s message=%s",
                    type(exc).__name__,
                    str(exc),
                )
            return {"status": "error", "message": f"Shopify lookup failed: {exc}"}
        # Safe debug log: reveal structure, never personal data.
        self._log_result_summary(tool, result)
        return {"status": "ok", "data": result}

    # ------------------------------------------------------------------ #
    # safe result-logging helpers (never emit personal data)
    # ------------------------------------------------------------------ #

    def _log_result_summary(self, tool: str, data: dict) -> None:
        """Log only structural metadata about a Shopify result.

        Never emits actual emails, names, or other PII.
        """
        summary: dict[str, Any] = {"tool": tool}
        for key, value in data.items():
            if isinstance(value, dict):
                summary[key] = self._summarise_node(key, value)
            else:
                summary[key] = _redact(str(value))
        logger.debug("shopify result summary: %s", summary)

    def _log_count_result_info(self, result: dict[str, Any]) -> None:
        """Log the exact model-facing payload for shopify_customer_count at INFO level.

        Safe fields only — no tokens, secrets, emails, or PII.
        """
        logger.info(
            "SHOPIFY_COUNT_MODEL_RESULT status=%s code=%s count=%s precision=%s message=%s",
            result.get("status"),
            result.get("code", "n/a"),
            result.get("count"),
            result.get("precision", "n/a"),
            str(result.get("message", ""))[:200],
        )

    @staticmethod
    def _summarise_node(name: str, node: dict) -> dict[str, Any]:
        """Return a redacted structural summary of a GraphQL node."""
        out: dict[str, Any] = {}
        if "edges" in node:
            edges = node["edges"]
            out["count"] = len(edges)
            if edges and "node" in edges[0]:
                sample = edges[0]["node"]
                has_email = "email" in sample
                email_null = has_email and sample.get("email") is None
                out["sample_fields"] = sorted(sample.keys())
                out["has_email_field"] = has_email
                out["email_is_null"] = email_null
                out["has_marketing_consent"] = "emailMarketingConsent" in sample
        elif "count" in node and "precision" in node:  # customersCount
            out["count_present"] = node.get("count") is not None
            out["precision"] = node.get("precision")
        elif "name" in node:  # shop / simple object
            out["name"] = node["name"]
        return out

    # ------------------------------------------------------------------ #
    # tools (all safe reads)
    # ------------------------------------------------------------------ #

    @function_tool()
    async def shopify_status(self, context: RunContext) -> dict[str, Any]:
        """Show the store name, currency, and whether the shop is active."""
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "shopify_status",
            """
            query {
              shop {
                name
                email
                currencyCode
                plan {
                  displayName
                }
              }
            }
            """,
        )

    @function_tool()
    async def shopify_customers(
        self, context: RunContext, count: int = 10
    ) -> dict[str, Any]:
        """List recent customers with basic details.

        Args:
            count: Number of customers to return (default 10).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        first = min(max(count, 1), _MAX_PAGE)
        raw = await self._with_client(
            "shopify_customers",
            """
            query ($first: Int) {
              customers(first: $first, sortKey: CREATED_AT, reverse: true) {
                edges {
                  node {
                    id
                    displayName
                    email
                    createdAt
                    numberOfOrders
                    totalSpent { amount currencyCode }
                  }
                }
                pageInfo { hasNextPage }
              }
            }
            """,
            first=first,
        )
        return self._normalise_customer_result(raw, "customers")

    @function_tool()
    async def shopify_marketing_contacts(
        self, context: RunContext, count: int = 20, subscribed_only: bool = True
    ) -> dict[str, Any]:
        """List customers with their email marketing consent state.

        Args:
            count: Number of contacts to return (default 20).
            subscribed_only: If true, only return customers who opted in
                (default true).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        first = min(max(count, 1), _MAX_PAGE)
        # Filter via the query string: accepts_marketing:true is a valid
        # customer search term.  None means "no filter" (return all).
        query_filter = "accepts_marketing:true" if subscribed_only else None
        raw = await self._with_client(
            "shopify_marketing_contacts",
            """
            query ($first: Int, $filter: String) {
              customers(
                first: $first
                sortKey: UPDATED_AT
                reverse: true
                query: $filter
              ) {
                edges {
                  node {
                    id
                    displayName
                    email
                    updatedAt
                    acceptsMarketing
                    emailMarketingConsent {
                      marketingState
                      optInLevel
                    }
                  }
                }
                pageInfo { hasNextPage }
              }
            }
            """,
            first=first,
            filter=query_filter,
        )
        return self._normalise_customer_result(raw, "customers", include_marketing=True)

    @function_tool()
    async def shopify_customer_count(self, context: RunContext) -> dict[str, Any]:
        """Return the total number of Shopify customers.

        Use this tool for questions such as:
        - "How many Shopify customers do I have?"
        - "Hoeveel Shopify klanten heb ik?"
        - "Total customer count"
        - "How many customers have we collected?"

        For listing individual customers, use shopify_customers instead.
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        raw = await self._with_client(
            "shopify_customer_count",
            """
            query {
              customersCount {
                count
                precision
              }
            }
            """,
        )
        result = self._normalise_count_result(
            raw, "customersCount", "shopify_customer_count"
        )
        self._log_count_result_info(result)
        return result

    # ------------------------------------------------------------------ #
    # new customer sign-ups within a period (exact count via pagination)
    # ------------------------------------------------------------------ #

    @function_tool()
    async def shopify_new_customers(
        self,
        context: RunContext,
        period: str = "today",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Count customers newly registered in a recent period.

        Use this for questions about NEW customer sign-ups/registrations one
        per customer. It is NOT about sales, orders, or revenue. Examples:
        - "How many new Shopify customers did we get today?"
        - "Hoeveel nieuwe Shopify klanten hebben we vandaag gekregen?"
        - "How many new customers yesterday?"
        - "Customer signups in the last 7 days"
        - "Hoeveel nieuwe klanten de afgelopen week?"

        The count is exact: every customer created inside the requested period
        is read (the customer list is walked page by page, newest first, until
        the period is fully covered - never just the first page). Period
        boundaries use the local calendar day. This is a read-only operation
        and reads only the customer creation time, never emails or other
        contact details.

        For the total number of customers, use shopify_customer_count. For
        listing individual customers, use shopify_customers.

        Args:
            period: "today", "yesterday", "last_7_days", or "custom".
            start_date: YYYY-MM-DD start (inclusive) when period="custom".
            end_date: YYYY-MM-DD end (inclusive) when period="custom"
                (defaults to start_date).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        if not self._client.is_authenticated():
            return self._not_connected()
        bounds = _resolve_period(period, start_date, end_date)
        if "error" in bounds:
            return {"status": "error", "message": bounds["error"]}
        try:
            count = await self._count_new_customers(bounds["start"], bounds["end"])
        except ShopifyNotAuthenticatedError:
            return self._not_connected()
        except Exception as exc:
            self._log_failure("shopify_new_customers", "graphql", exc)
            return {"status": "error", "message": f"Shopify lookup failed: {exc}"}
        logger.info(
            "SHOPIFY_NEW_CUSTOMERS period=%s start=%s end=%s count=%s",
            bounds["label"],
            bounds["start_iso"],
            bounds["end_iso"],
            count,
        )
        result: dict[str, Any] = {
            "status": "ok",
            "period": bounds["label"],
            "start_date": bounds["start_iso"],
            "end_date": bounds["end_iso"],
            "count": count,
        }
        if count == 0:
            result["message"] = "No new customers in that period."
        return result

    async def _count_new_customers(self, start: dt.datetime, end: dt.datetime) -> int:
        """Exact count of customers created within the inclusive [start, end].

        Walks the customers connection newest-first by creation time and stops
        as soon as a customer is older than the range start, so the count is
        exact without scanning the entire customer list. Only ``createdAt`` is
        read; emails and other PII are never fetched.
        """
        per_page = 50
        count = 0
        cursor: str | None = None
        while True:
            variables: dict[str, Any] = {"first": per_page}
            if cursor:
                variables["after"] = cursor
            raw = await self._client.execute(_NEW_CUSTOMERS_QUERY, variables)
            connection = raw.get("customers", {}) or {}
            edges = connection.get("edges", []) or []
            page_info = connection.get("pageInfo", {}) or {}
            reached_out_of_range = False
            for edge in edges:
                node = edge.get("node", {}) or {}
                created_raw = node.get("createdAt")
                if not created_raw:
                    continue
                created_at = _local_naive(created_raw)
                if created_at < start:
                    reached_out_of_range = True
                    break
                if created_at <= end:
                    count += 1
            if reached_out_of_range:
                break
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not cursor:
                break
        return count

    # ------------------------------------------------------------------ #
    # result normalisation for customer-facing payloads
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_customer_result(
        raw: dict[str, Any],
        connection_key: str,
        *,
        include_marketing: bool = False,
    ) -> dict[str, Any]:
        """Transform raw GraphQL edges into a model-friendly list.

        Never leaks actual email addresses in the model result.
        """
        if raw.get("status") != "ok":
            return raw
        data = raw.get("data", {})
        connection = data.get(connection_key, {})
        edges = connection.get("edges", [])
        has_more = connection.get("pageInfo", {}).get("hasNextPage", False)
        customers: list[dict[str, Any]] = []
        email_all_null = True
        for edge in edges:
            node = edge.get("node", {})
            email_raw = node.get("email")
            email_display: str | None
            if email_raw:
                email_display = _mask_email(email_raw)
                email_all_null = False
            else:
                email_display = "(email unavailable — check read_customer_email scope)"
            entry: dict[str, Any] = {
                "name": node.get("displayName"),
                "email": email_display,
                "created": node.get("createdAt"),
            }
            orders = node.get("numberOfOrders")
            if orders is not None:
                entry["orders"] = orders
            spent = node.get("totalSpent")
            if spent:
                entry["total_spent"] = (
                    f"{spent.get('amount', '?')} {spent.get('currencyCode', '')}".strip()
                )
            updated = node.get("updatedAt")
            if updated:
                entry["updated"] = updated
            accepts = node.get("acceptsMarketing")
            if accepts is not None:
                entry["accepts_marketing"] = accepts
            if include_marketing:
                mc = node.get("emailMarketingConsent") or {}
                entry["marketing_state"] = mc.get("marketingState")
                entry["opt_in_level"] = mc.get("optInLevel")
            customers.append(entry)
        result: dict[str, Any] = {
            "status": "ok",
            "count": len(customers),
        }
        if customers:
            result["customers"] = customers
        else:
            result["message"] = "No customers found."
        if email_all_null and edges:
            result["note"] = (
                "Email fields returned null. Ensure the app has the "
                "read_customer_email scope enabled in the Shopify admin."
            )
        if has_more:
            result["has_more"] = True
        return result

    @staticmethod
    def _normalise_count_result(
        raw: dict[str, Any], connection_key: str, tool_name: str
    ) -> dict[str, Any]:
        """Transform a raw GraphQL count result into a compact, model-friendly payload.

        Logs only structural metadata (never PII or secrets).
        """
        if raw.get("status") != "ok":
            logger.debug("shopify %s: non-ok status=%s", tool_name, raw.get("status"))
            return raw
        data = raw.get("data", {})
        # When the key exists but is null, data.get() returns None, not the default.
        count_info: dict | None = data.get(connection_key)  # type: ignore[assignment]
        if count_info is None:
            logger.debug(
                "shopify %s: %s is null/missing — check read_customers scope",
                tool_name,
                connection_key,
            )
            return {
                "status": "error",
                "code": "count_unavailable",
                "factually_unavailable": True,
                "message": (
                    "Shopify did not return a customer count. "
                    "The exact number of customers is NOT available. "
                    "DO NOT state or guess any number. "
                    "Tell the user that the customer count cannot be retrieved right now "
                    "because the required access scope (read_customers) is not enabled. "
                    "Do not invent a number."
                ),
            }
        total = count_info.get("count")
        precision = count_info.get("precision", "EXACT")
        logger.debug("shopify %s: count=%s precision=%s", tool_name, total, precision)
        if total is not None:
            return {
                "status": "ok",
                "count": total,
                "precision": precision,
                "message": (
                    f"Total customer count: {total}"
                    + (
                        f" (approximate, precision: {precision})"
                        if precision and precision != "EXACT"
                        else ""
                    )
                ),
            }
        logger.debug("shopify %s: count field is null in response", tool_name)
        return {
            "status": "error",
            "code": "count_null",
            "factually_unavailable": True,
            "message": (
                "Shopify returned the customersCount field but the count value is null. "
                "The exact number of customers is NOT available. "
                "DO NOT state or guess any number. "
                "Tell the user that the customer count cannot be retrieved. "
                "Do not invent a number."
            ),
        }

    @function_tool()
    async def shopify_sales_summary(
        self, context: RunContext, days: int = 7
    ) -> dict[str, Any]:
        """Summarise recent sales: total orders, gross sales, and net sales.

        Args:
            days: Number of past days to include (default 7).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "shopify_sales_summary",
            """
            query ($first: Int, $query: String) {
              orders(first: $first, query: $query) {
                edges {
                  node {
                    totalPriceSet { shopMoney { amount currencyCode } }
                    netPaymentSet { shopMoney { amount currencyCode } }
                    financialStatus
                    createdAt
                  }
                }
                pageInfo { hasNextPage }
              }
            }
            """,
            first=min(_MAX_PAGE, 50),
        )

    @function_tool()
    async def shopify_recent_orders(
        self, context: RunContext, count: int = 5
    ) -> dict[str, Any]:
        """List the most recent orders with key details.

        Args:
            count: Number of recent orders to return (default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        first = min(max(count, 1), _MAX_PAGE)
        return await self._with_client(
            "shopify_recent_orders",
            """
            query ($first: Int) {
              orders(first: $first, sortKey: CREATED_AT, reverse: true) {
                edges {
                  node {
                    id
                    name
                    createdAt
                    totalPriceSet { shopMoney { amount currencyCode } }
                    financialStatus
                    fulfillmentStatus
                    lineItems(first: 5) {
                      edges {
                        node {
                          title
                          quantity
                        }
                      }
                    }
                  }
                }
                pageInfo { hasNextPage }
              }
            }
            """,
            first=first,
        )

    @function_tool()
    async def shopify_products(
        self, context: RunContext, count: int = 10
    ) -> dict[str, Any]:
        """List products with their titles, prices, and inventory status.

        Args:
            count: Number of products to return (default 10).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        first = min(max(count, 1), _MAX_PAGE)
        return await self._with_client(
            "shopify_products",
            """
            query ($first: Int) {
              products(first: $first) {
                edges {
                  node {
                    id
                    title
                    productType
                    status
                    variants(first: 3) {
                      edges {
                        node {
                          id
                          title
                          price { amount currencyCode }
                          inventoryQuantity
                          sku
                        }
                      }
                    }
                  }
                }
                pageInfo { hasNextPage }
              }
            }
            """,
            first=first,
        )

    @function_tool()
    async def shopify_inventory(
        self, context: RunContext, product_title: str | None = None
    ) -> dict[str, Any]:
        """Show inventory levels for products or a specific product.

        Args:
            product_title: Optional product title to filter by.
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        query_str = """
            query ($first: Int, $query: String) {
              products(first: $first, query: $query) {
                edges {
                  node {
                    id
                    title
                    variants(first: 10) {
                      edges {
                        node {
                          id
                          title
                          inventoryQuantity
                          sku
                        }
                      }
                    }
                  }
                }
                pageInfo { hasNextPage }
              }
            }
            """
        if product_title:
            return await self._with_client(
                "shopify_inventory",
                query_str,
                first=min(_MAX_PAGE, 20),
                query=f"title:*{product_title}*",
            )
        return await self._with_client(
            "shopify_inventory",
            query_str,
            first=min(_MAX_PAGE, 20),
        )

    @function_tool()
    async def shopify_low_stock(
        self, context: RunContext, threshold: int = 5
    ) -> dict[str, Any]:
        """Find products with inventory below a threshold.

        Args:
            threshold: Maximum quantity to consider low stock (default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        return await self._with_client(
            "shopify_low_stock",
            """
            query ($first: Int) {
              products(first: $first) {
                edges {
                  node {
                    id
                    title
                    variants(first: 10) {
                      edges {
                        node {
                          id
                          title
                          inventoryQuantity
                          sku
                        }
                      }
                    }
                  }
                }
                pageInfo { hasNextPage }
              }
            }
            """,
            first=min(_MAX_PAGE, 50),
        )

    @function_tool()
    async def shopify_top_products(
        self, context: RunContext, count: int = 5
    ) -> dict[str, Any]:
        """List the best-selling products by total quantity sold.

        Args:
            count: Number of top products to return (default 5).
        """
        self.gate.ensure_action_level(int(ActionLevel.SAFE_READ))
        first = min(max(count, 1), _MAX_PAGE)
        return await self._with_client(
            "shopify_top_products",
            """
            query ($first: Int) {
              products(first: $first, sortKey: BEST_SELLING, reverse: false) {
                edges {
                  node {
                    id
                    title
                    totalInventory
                    variants(first: 1) {
                      edges {
                        node {
                          price { amount currencyCode }
                        }
                      }
                    }
                  }
                }
              }
            }
            """,
            first=first,
        )


# ---------------------------------------------------------------------- #
# helpers for shopify_new_customers
# ---------------------------------------------------------------------- #


def _resolve_period(
    period: str,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    """Resolve a requested period to inclusive local [start, end] datetimes."""
    period = (period or "").strip().lower()
    today = dt.date.today()
    if period == "today":
        start = end = today
    elif period == "yesterday":
        start = end = today - dt.timedelta(days=1)
    elif period in ("last_7_days", "last 7 days", "week", "last_week"):
        start = today - dt.timedelta(days=6)
        end = today
    elif period == "custom":
        parsed_start = _parse_date(start_date)
        if parsed_start is None:
            return {
                "error": (
                    "For a custom range, provide start_date (YYYY-MM-DD). "
                    "Use period='today', 'yesterday', or 'last_7_days' for "
                    "common ranges."
                )
            }
        parsed_end = _parse_date(end_date) if end_date else parsed_start
        if parsed_end is None:
            return {"error": "end_date must be a valid YYYY-MM-DD date."}
        if parsed_end < parsed_start:
            return {"error": "start_date must not be after end_date."}
        start, end = parsed_start, parsed_end
    else:
        return {
            "error": (
                "period must be 'today', 'yesterday', 'last_7_days', or "
                "'custom' with start_date."
            )
        }
    return {
        "label": period,
        "start": dt.datetime.combine(start, dt.time.min),
        "end": dt.datetime.combine(end, dt.time.max),
        "start_iso": start.isoformat(),
        "end_iso": end.isoformat(),
    }


def _parse_date(value: str | None) -> dt.date | None:
    if not value or not value.strip():
        return None
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        return None


def _local_naive(value: str) -> dt.datetime:
    """Parse a Shopify ISO timestamp into a naive, server-local datetime.

    ``createdAt`` is UTC; converting to the local zone keeps a comparison
    against local calendar-day boundaries consistent.
    """
    parsed = value.strip()
    if parsed.endswith(("Z", "z")):
        parsed = parsed[:-1] + "+00:00"
    try:
        result = dt.datetime.fromisoformat(parsed)
    except ValueError:
        return dt.datetime.max
    if result.tzinfo is not None:
        result = result.astimezone().replace(tzinfo=None)
    return result
