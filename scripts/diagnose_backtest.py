"""Trace why backtest produces 0 trades.

Run inside the container:
  docker compose exec web python scripts/diagnose_backtest.py
"""
from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pandas as pd

from polymarket_edge.clients.clob import ClobClient
from polymarket_edge.ingest import load_raw_markets
from polymarket_edge.models import Resolution

DATA = Path("/app/data")
DELTA = timedelta(hours=24)
MAX_ENTRY = 0.10


async def main() -> None:
    # 1. Verify the running image has the fix.
    print("=== fix check ===")
    print("has get_market:", "get_market" in dir(ClobClient))
    print("price_history sig:", inspect.signature(ClobClient.price_history))
    print()

    markets = load_raw_markets(DATA / "raw" / "markets.parquet")
    labels = pd.read_csv(DATA / "reports" / "labels.csv")
    labels["id"] = labels["id"].astype(str)
    fifty = labels[labels["resolution"] == Resolution.FIFTY_FIFTY.value]
    print(f"FIFTY_FIFTY markets: {len(fifty)}")

    markets["id"] = markets["id"].astype(str)
    m_by_id = markets.set_index("id")

    skip = Counter()
    ok_tokens = 0
    ok_prices = 0
    cheap_enough = 0
    now = pd.Timestamp.utcnow().timestamp()
    horizon_s = 140 * 86400

    samples: list[dict] = []

    async with ClobClient() as clob:
        for mid in fifty["id"].head(50):  # just probe first 50
            if mid not in m_by_id.index:
                skip["no_raw_row"] += 1
                continue
            raw = m_by_id.loc[mid]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[0]

            closed = raw.get("closedTime")
            end = fifty.set_index("id").loc[mid, "end_date"] if "end_date" in fifty.columns else None
            resolved_at = closed or end
            if resolved_at is None or (isinstance(resolved_at, float) and pd.isna(resolved_at)):
                skip["no_resolved_at"] += 1
                continue
            try:
                end_ts = pd.Timestamp(resolved_at).timestamp()
            except Exception:
                skip["bad_resolved_ts"] += 1
                continue

            if now - end_ts > horizon_s:
                skip["older_than_140d"] += 1
                continue

            cond = raw.get("conditionId") or raw.get("condition_id")
            if not cond:
                skip["no_condition_id"] += 1
                continue

            try:
                meta = await clob.get_market(str(cond))
            except Exception as e:
                skip[f"clob_get_market_err:{type(e).__name__}"] += 1
                continue
            if not meta:
                skip["clob_404"] += 1
                continue
            toks = [str(t.get("token_id")) for t in (meta.get("tokens") or []) if t.get("token_id")]
            if len(toks) < 2:
                skip["no_tokens"] += 1
                continue
            ok_tokens += 1

            target_ts = end_ts - DELTA.total_seconds()
            prices = []
            for tid in toks[:2]:
                try:
                    h = await clob.price_history(tid, interval="max", fidelity=1440)
                except Exception:
                    prices.append(None)
                    continue
                prices.append(ClobClient.price_at(h, target_ts))
            valid = [p for p in prices if p is not None and p > 0]
            if not valid:
                skip["no_price_at_target"] += 1
                samples.append({
                    "id": mid, "cond": cond, "resolved_at": str(resolved_at),
                    "age_days": round((now - end_ts) / 86400, 1),
                    "prices": prices,
                })
                continue
            ok_prices += 1
            cheap = min(valid)
            if cheap > MAX_ENTRY:
                skip[f"too_expensive_{cheap:.2f}"] += 1
                continue
            cheap_enough += 1

            await asyncio.sleep(0.05)

    print()
    print("=== skip reasons (first 50 FIFTY_FIFTY) ===")
    for reason, n in skip.most_common():
        print(f"  {n:4d}  {reason}")
    print()
    print(f"got tokens: {ok_tokens}")
    print(f"got prices: {ok_prices}")
    print(f"cheap enough (<= {MAX_ENTRY}): {cheap_enough}")
    print()
    if samples:
        print("=== sample 'no price at target' markets ===")
        for s in samples[:5]:
            print(f"  {s}")


if __name__ == "__main__":
    asyncio.run(main())
