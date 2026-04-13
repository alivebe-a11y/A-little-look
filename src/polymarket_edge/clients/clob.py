"""Thin read-only client for the Polymarket CLOB REST API.

We only need historical price data for the backtest — specifically, the
last-traded price of a YES/NO token at (or just before) a given timestamp.

Docs: https://docs.polymarket.com/developers/CLOB/clients/methods-overview
"""
from __future__ import annotations

from typing import Any

import httpx

CLOB_BASE_URL = "https://clob.polymarket.com"


class ClobClient:
    def __init__(
        self,
        base_url: str = CLOB_BASE_URL,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout

    async def __aenter__(self) -> "ClobClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def price_history(
        self,
        token_id: str,
        *,
        interval: str = "max",
        fidelity: int | None = 1440,
    ) -> list[dict[str, Any]]:
        """Return list of {t: unix_seconds, p: price} points for a token.

        The CLOB `/prices-history` endpoint returns bucketed trade history.
        With ``interval="max"`` and no ``fidelity`` the server caps the
        window at ~30 days. Passing ``fidelity=1440`` (1 bucket per day)
        stretches that to ~140 days, which is what the backtest needs.
        """
        assert self._client is not None, "use ClobClient as an async context manager"
        params: dict[str, Any] = {"market": token_id, "interval": interval}
        if fidelity is not None:
            params["fidelity"] = fidelity
        resp = await self._client.get(f"{self.base_url}/prices-history", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("history", []) if isinstance(data, dict) else []

    async def get_market(self, condition_id: str) -> dict[str, Any] | None:
        """Fetch CLOB market metadata by condition_id.

        Returns the decoded JSON (including ``tokens: [{token_id, outcome}]``),
        or ``None`` if the market isn't known to CLOB. Gamma's own
        ``clobTokenIds`` field is unreliable — look up the real token IDs
        here before querying price history.
        """
        assert self._client is not None, "use ClobClient as an async context manager"
        resp = await self._client.get(f"{self.base_url}/markets/{condition_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, dict) else None

    @staticmethod
    def price_at(
        history: list[dict[str, Any]], target_ts: float
    ) -> float | None:
        """Return the last price recorded at or before target_ts, else None."""
        last: float | None = None
        for point in history:
            ts = point.get("t")
            price = point.get("p")
            if ts is None or price is None:
                continue
            if ts <= target_ts:
                last = float(price)
            else:
                break
        return last
