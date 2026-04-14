"""Client for a Polymarket subgraph on The Graph.

The public CLOB ``/prices-history`` endpoint only retains ~140 days.
Subgraph indexing goes back to contract deployment, which is what we
need to backtest the 50/50 invalidation thesis against the full
history of resolved FIFTY_FIFTY markets (~500 vs. ~20 in the
CLOB-retention window).

Requires a The Graph API key (free tier: 100k queries/mo). Set in
``.env`` as ``THEGRAPH_API_KEY``. If unset, every call raises a
clear error — no silent empty-response path.

Polymarket publishes several subgraphs; they don't share a schema:
  * CTF / positions  (``Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp``)
      — splits, merges, redemptions, positions. **No trade history.**
  * orderbook / orders-matched — trade fills with prices. This is
      what backtesting wants. Find the current ID on
      https://thegraph.com/explorer (search "polymarket" and look
      for one with ``orderFilledEvents`` / ``ordersMatched``).

Override via ``THEGRAPH_SUBGRAPH_ID`` env var so swapping subgraphs
doesn't need a rebuild.

Gateway endpoint format:
    https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}
"""
from __future__ import annotations

import os
from typing import Any

import httpx

# Defaults to the CTF/positions subgraph (what Polymarket originally
# documented under that ID). Override with THEGRAPH_SUBGRAPH_ID to point
# at the orderbook subgraph.
DEFAULT_SUBGRAPH_ID = "Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
DEFAULT_GATEWAY = "https://gateway.thegraph.com/api"


class SubgraphAuthError(RuntimeError):
    """Raised when the caller hasn't provided a The Graph API key."""


class SubgraphClient:
    def __init__(
        self,
        api_key: str | None = None,
        subgraph_id: str | None = None,
        gateway: str = DEFAULT_GATEWAY,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 30.0,
        page_size: int = 1000,
    ) -> None:
        self.api_key = api_key or os.environ.get("THEGRAPH_API_KEY")
        self.subgraph_id = (
            subgraph_id
            or os.environ.get("THEGRAPH_SUBGRAPH_ID")
            or DEFAULT_SUBGRAPH_ID
        )
        self.gateway = gateway.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout
        self.page_size = page_size

    @property
    def endpoint(self) -> str:
        if not self.api_key:
            raise SubgraphAuthError(
                "THEGRAPH_API_KEY is not set. Get a key at https://thegraph.com "
                "and export it (or put it in .env)."
            )
        return f"{self.gateway}/{self.api_key}/subgraphs/id/{self.subgraph_id}"

    async def __aenter__(self) -> "SubgraphClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def query(self, gql: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._client is not None, "use SubgraphClient as an async context manager"
        resp = await self._client.post(
            self.endpoint,
            json={"query": gql, "variables": variables or {}},
        )
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise RuntimeError(f"subgraph query errors: {body['errors']}")
        return body.get("data", {})

    async def trades_for_condition(
        self,
        condition_id: str,
        *,
        first: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return trades for a given ``conditionId``, ordered by timestamp.

        Returns a list of ``{timestamp, outcomeIndex, price, size, side}``
        rows (exact field names depend on the subgraph schema; query
        returns them as-indexed). Pages in batches of ``page_size`` and
        keeps fetching until fewer rows are returned or ``first`` is hit.
        """
        limit = first or self.page_size
        results: list[dict[str, Any]] = []
        skip = 0
        while True:
            page_size = min(self.page_size, limit - len(results)) if first else self.page_size
            if page_size <= 0:
                break
            gql = """
            query Trades($condition: String!, $first: Int!, $skip: Int!) {
              orderFilledEvents(
                where: { market_: { conditionId: $condition } }
                orderBy: timestamp
                orderDirection: asc
                first: $first
                skip: $skip
              ) {
                id
                timestamp
                makerAssetId
                takerAssetId
                makerAmountFilled
                takerAmountFilled
              }
            }
            """
            data = await self.query(
                gql,
                variables={"condition": condition_id.lower(), "first": page_size, "skip": skip},
            )
            batch = data.get("orderFilledEvents", []) or []
            if not batch:
                break
            results.extend(batch)
            if len(batch) < page_size:
                break
            skip += page_size
        return results
