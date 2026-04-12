"""Backtest: would buying the cheap side at T-Δ have been profitable?

Strategy for each resolved market:
  1. At `end_date - delta`, look up the price of each YES/NO token.
  2. If the cheaper side trades at `<= max_entry_price`, simulate buying
     `capital` dollars of it.
  3. On resolution, each token pays:
       YES:        $1 if market resolved YES else $0
       NO:         $1 if market resolved NO  else $0
       FIFTY_FIFTY: $0.50 either way
  4. pnl = payout - capital.

We aggregate over all markets in the input and group by feature bucket so the
user can see which heuristics produced positive EV.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from .clients.clob import ClobClient
from .models import Resolution

DELTA_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE)


def parse_delta(spec: str) -> timedelta:
    m = DELTA_RE.match(spec)
    if not m:
        raise ValueError(f"bad delta spec: {spec!r} (expected e.g. '24h', '30m')")
    n = int(m.group(1))
    unit = m.group(2).lower()
    return {
        "s": timedelta(seconds=n),
        "m": timedelta(minutes=n),
        "h": timedelta(hours=n),
        "d": timedelta(days=n),
    }[unit]


@dataclass
class Trade:
    market_id: str
    slug: str
    resolution: str
    entry_price: float
    payout_per_share: float
    capital: float
    pnl: float
    roi: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def payout_for(resolution: str, side_index: int) -> float:
    """Payout per $1-face token given a resolution and which side we bought.

    side_index: 0 for YES token, 1 for NO token.
    """
    if resolution == Resolution.FIFTY_FIFTY.value:
        return 0.5
    if resolution == Resolution.YES.value:
        return 1.0 if side_index == 0 else 0.0
    if resolution == Resolution.NO.value:
        return 1.0 if side_index == 1 else 0.0
    return 0.0


def simulate_trade(
    *,
    market_id: str,
    slug: str,
    resolution: str,
    entry_prices: tuple[float | None, float | None],
    capital: float,
    max_entry_price: float,
) -> Trade | None:
    """Pick the cheaper side; skip if both are None or too expensive."""
    valid = [(i, p) for i, p in enumerate(entry_prices) if p is not None and p > 0]
    if not valid:
        return None
    side_index, entry_price = min(valid, key=lambda ip: ip[1])
    if entry_price > max_entry_price:
        return None

    shares = capital / entry_price
    payout_per_share = payout_for(resolution, side_index)
    payout = shares * payout_per_share
    pnl = payout - capital
    roi = pnl / capital if capital else 0.0
    return Trade(
        market_id=market_id,
        slug=slug,
        resolution=resolution,
        entry_price=entry_price,
        payout_per_share=payout_per_share,
        capital=capital,
        pnl=pnl,
        roi=roi,
    )


async def _fetch_entry_prices(
    clob: ClobClient,
    token_ids: list[str],
    target_ts: float,
) -> list[float | None]:
    out: list[float | None] = []
    for tid in token_ids[:2]:  # only binary markets
        try:
            history = await clob.price_history(tid, interval="max")
        except Exception:
            out.append(None)
            continue
        out.append(ClobClient.price_at(history, target_ts))
    while len(out) < 2:
        out.append(None)
    return out


async def run_backtest(
    markets_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    delta: timedelta,
    capital: float = 1000.0,
    max_entry_price: float = 0.10,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the simulation over every labelled resolved market.

    Returns a DataFrame of Trade rows joined with the feature vector.
    """
    labels_by_id = labels_df.set_index("id")
    feats_by_id = features_df.set_index("id") if not features_df.empty else None

    trades: list[dict[str, Any]] = []
    async with ClobClient() as clob:
        count = 0
        for _, raw in markets_df.iterrows():
            mid = str(raw.get("id"))
            if mid not in labels_by_id.index:
                continue
            label_row = labels_by_id.loc[mid]
            resolution = str(label_row["resolution"])
            if resolution in (Resolution.UNRESOLVED.value, Resolution.OTHER.value):
                continue
            end_date = label_row["end_date"]
            if pd.isna(end_date):
                continue
            end_ts = pd.Timestamp(end_date).timestamp()
            target_ts = end_ts - delta.total_seconds()

            token_ids = raw.get("clobTokenIds") or raw.get("clob_token_ids") or []
            if isinstance(token_ids, str):
                try:
                    import json
                    token_ids = json.loads(token_ids)
                except Exception:
                    token_ids = []
            if not isinstance(token_ids, list) or len(token_ids) < 2:
                continue

            entry_prices = await _fetch_entry_prices(clob, token_ids, target_ts)
            trade = simulate_trade(
                market_id=mid,
                slug=str(raw.get("slug", "")),
                resolution=resolution,
                entry_prices=(entry_prices[0], entry_prices[1]),
                capital=capital,
                max_entry_price=max_entry_price,
            )
            if trade is None:
                continue
            row = trade.as_dict()
            if feats_by_id is not None and mid in feats_by_id.index:
                for col, val in feats_by_id.loc[mid].items():
                    row[f"feat_{col}"] = val
            trades.append(row)
            count += 1
            if limit is not None and count >= limit:
                break
            await asyncio.sleep(0.05)  # polite to the public CLOB endpoint
    return pd.DataFrame(trades)


def summarize_backtest(trades: pd.DataFrame) -> pd.DataFrame:
    """Produce per-feature-bucket summary: n, hit_rate, mean_roi, total_pnl."""
    if trades.empty:
        return pd.DataFrame(columns=["bucket", "n", "hit_rate", "mean_roi", "total_pnl"])
    rows = []
    rows.append(_bucket_stats("ALL", trades))
    for col in [c for c in trades.columns if c.startswith("feat_")]:
        positive = trades[trades[col] == True]  # noqa: E712
        negative = trades[trades[col] == False]  # noqa: E712
        if not positive.empty:
            rows.append(_bucket_stats(f"{col}=True", positive))
        if not negative.empty:
            rows.append(_bucket_stats(f"{col}=False", negative))
    return pd.DataFrame(rows)


def _bucket_stats(name: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "bucket": name,
        "n": int(len(df)),
        "hit_rate": float((df["pnl"] > 0).mean()) if len(df) else 0.0,
        "mean_roi": float(df["roi"].mean()) if len(df) else 0.0,
        "total_pnl": float(df["pnl"].sum()) if len(df) else 0.0,
    }
