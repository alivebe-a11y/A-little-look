"""Async paginator for the public Polymarket Gamma markets API.

Docs: https://docs.polymarket.com/developers/gamma-markets-api/overview
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_PAGE_SIZE = 500


class GammaClient:
    def __init__(
        self,
        base_url: str = GAMMA_BASE_URL,
        client: httpx.AsyncClient | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        request_timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout

    async def __aenter__(self) -> "GammaClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def iter_markets(
        self,
        *,
        closed: bool | None = None,
        archived: bool | None = None,
        start_date_min: str | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw market dicts, paginating until the API returns fewer than
        `page_size` rows (or max_pages is hit)."""
        assert self._client is not None, "use GammaClient as an async context manager"
        offset = 0
        pages = 0
        while True:
            params: dict[str, Any] = {"limit": self.page_size, "offset": offset}
            if closed is not None:
                params["closed"] = str(closed).lower()
            if archived is not None:
                params["archived"] = str(archived).lower()
            if start_date_min:
                params["start_date_min"] = start_date_min

            resp = await self._client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                return
            for item in batch:
                yield item
            if len(batch) < self.page_size:
                return
            offset += self.page_size
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return
            # Gentle on the public endpoint.
            await asyncio.sleep(0.1)
