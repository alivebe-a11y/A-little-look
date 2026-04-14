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

# Polymarket orderbook subgraph — has OrderFilled + NegRiskCtfExchangeOrderFilled
# entities (makerAssetId/takerAssetId/makerAmountFilled/takerAmountFilled/
# blockTimestamp). Override via THEGRAPH_SUBGRAPH_ID.
DEFAULT_SUBGRAPH_ID = "EZCTgSzLPuBSqQcuR3ifeiKHKBnpjHSNbYpty8Mnjm9D"
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

    async def trades_for_token_ids(
        self,
        token_ids: list[str],
        *,
        first: int | None = None,
        include_neg_risk: bool = True,
    ) -> list[dict[str, Any]]:
        """Return order-fill events involving any of the given CTF token IDs.

        The Polymarket orderbook subgraph (``EZCTgSz...``) indexes fills by
        ``makerAssetId`` / ``takerAssetId`` — the on-chain ERC1155 positionId,
        which is what CLOB calls ``token_id``. There's no ``conditionId``
        field, so callers must resolve conditionId → [yes_token, no_token]
        via CLOB before calling this.

        We run two queries (maker side + taker side) because subgraph ``or:``
        filter support is inconsistent across schema versions, and optionally
        include ``negRiskCtfExchangeOrderFilleds`` for multi-outcome markets.

        Each returned row carries a synthetic ``exchange`` key ("ctf" or
        "neg_risk") so downstream code can distinguish them.
        """
        if not token_ids:
            return []
        ids = [str(t) for t in token_ids]
        limit = first
        out: list[dict[str, Any]] = []

        entities: list[str] = ["orderFilleds"]
        if include_neg_risk:
            entities.append("negRiskCtfExchangeOrderFilleds")

        for entity in entities:
            exchange = "neg_risk" if entity.startswith("negRisk") else "ctf"
            for side_field in ("makerAssetId_in", "takerAssetId_in"):
                skip = 0
                while True:
                    remaining = (limit - len(out)) if limit is not None else self.page_size
                    if remaining <= 0:
                        return out
                    page = min(self.page_size, remaining)
                    gql = (
                        "query($ids: [BigInt!]!, $first: Int!, $skip: Int!) {"
                        f"  {entity}("
                        f"    where: {{ {side_field}: $ids }}"
                        "    orderBy: blockTimestamp"
                        "    orderDirection: asc"
                        "    first: $first"
                        "    skip: $skip"
                        "  ) {"
                        "    id orderHash maker taker"
                        "    makerAssetId takerAssetId"
                        "    makerAmountFilled takerAmountFilled"
                        "    fee blockNumber blockTimestamp transactionHash"
                        "  }"
                        "}"
                    )
                    data = await self.query(
                        gql,
                        variables={"ids": ids, "first": page, "skip": skip},
                    )
                    batch = data.get(entity, []) or []
                    if not batch:
                        break
                    for row in batch:
                        row["exchange"] = exchange
                        row["_match_side"] = side_field.replace("_in", "")
                    out.extend(batch)
                    if len(batch) < page:
                        break
                    skip += page
        return out


def derive_price_and_side(
    row: dict[str, Any],
    token_ids: list[str],
) -> dict[str, Any]:
    """Add ``timestamp_ts``, ``token_id``, ``outcome_index``, ``price``, ``size``.

    On Polymarket's CTF exchange, USDC has assetId ``0`` and outcome tokens
    have the positionId as assetId. A fill is either:

      * **taker buys token** — maker pays USDC, taker pays token asset ID.
        price = makerAmountFilled / takerAmountFilled (USDC per share).
      * **maker sells token** — maker pays token, taker pays USDC.
        price = takerAmountFilled / makerAmountFilled.

    Both legs are 6-decimal so the ratio is already the price per share.
    """
    maker_id = str(row.get("makerAssetId"))
    taker_id = str(row.get("takerAssetId"))
    try:
        maker_amt = int(row.get("makerAmountFilled", 0))
        taker_amt = int(row.get("takerAmountFilled", 0))
    except (TypeError, ValueError):
        maker_amt = taker_amt = 0

    if taker_id in token_ids:
        token_id = taker_id
        size = taker_amt / 1e6
        price = (maker_amt / taker_amt) if taker_amt else None
    elif maker_id in token_ids:
        token_id = maker_id
        size = maker_amt / 1e6
        price = (taker_amt / maker_amt) if maker_amt else None
    else:
        token_id = None
        size = 0.0
        price = None

    outcome_index = token_ids.index(token_id) if token_id in token_ids else None
    try:
        ts = int(row.get("blockTimestamp", 0))
    except (TypeError, ValueError):
        ts = 0

    row = dict(row)
    row["timestamp_ts"] = ts
    row["token_id"] = token_id
    row["outcome_index"] = outcome_index
    row["price"] = price
    row["size"] = size
    return row
