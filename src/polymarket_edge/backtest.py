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


async def _resolve_token_ids(
    clob: ClobClient,
    condition_id: str,
    cache: dict[str, list[str] | None],
) -> list[str] | None:
    """Look up real CLOB token IDs for a market. Cached per condition_id.

    Gamma's own ``clobTokenIds`` often disagree with the actual ERC1155
    ids CLOB serves price history under, so we hit CLOB's
    ``/markets/<condition_id>`` to get the authoritative pair.
    """
    if condition_id in cache:
        return cache[condition_id]
    try:
        meta = await clob.get_market(condition_id)
    except Exception:
        cache[condition_id] = None
        return None
    toks: list[str] = []
    if meta:
        for t in meta.get("tokens", []) or []:
            tid = t.get("token_id")
            if tid:
                toks.append(str(tid))
    result = toks if len(toks) >= 2 else None
    cache[condition_id] = result
    return result


async def _fetch_entry_prices(
    clob: ClobClient,
    token_ids: list[str],
    target_ts: float,
) -> list[float | None]:
    out: list[float | None] = []
    for tid in token_ids[:2]:  # only binary markets
        try:
            history = await clob.price_history(tid, interval="max", fidelity=1440)
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
    max_entry_price: float = 0.30,
    limit: int | None = None,
    max_age_days: int = 140,
) -> pd.DataFrame:
    """Run the simulation over every labelled resolved market.

    ``max_age_days`` caps how far back we look — the public CLOB
    ``/prices-history`` endpoint only retains roughly the last 140 days
    of data, so older markets always return empty history and would be
    skipped anyway. Pre-filtering saves tens of thousands of HTTP calls.

    Returns a DataFrame of Trade rows joined with the feature vector.
    Also prints a per-skip-reason counter so a 0-trade result is
    diagnosable without rerunning the probe.
    """
    labels_by_id = labels_df.set_index("id")
    feats_by_id = features_df.set_index("id") if not features_df.empty else None

    # Narrow markets_df to labelled, resolved, recent rows up front.
    markets_df = markets_df.copy()
    markets_df["id"] = markets_df["id"].astype(str)
    labelled_ids = {str(i) for i in labels_by_id.index}
    resolved_ids = {
        str(i)
        for i, row in labels_by_id.iterrows()
        if str(row["resolution"])
        not in (Resolution.UNRESOLVED.value, Resolution.OTHER.value)
    }
    keep = labelled_ids & resolved_ids
    candidates = markets_df[markets_df["id"].isin(keep)].copy()
    candidates["_closed_parsed"] = pd.to_datetime(
        candidates.get("closedTime"), errors="coerce", utc=True
    )
    now = pd.Timestamp.now(tz="UTC")
    candidates["_age_days"] = (now - candidates["_closed_parsed"]).dt.days
    total_candidates = len(candidates)
    candidates = candidates[
        candidates["_age_days"].notna() & (candidates["_age_days"] <= max_age_days)
    ].sort_values("_age_days")
    print(
        f"backtest: {total_candidates} resolved + labelled markets, "
        f"{len(candidates)} within {max_age_days} days"
    )

    skip: dict[str, int] = {}

    def _skip(reason: str) -> None:
        skip[reason] = skip.get(reason, 0) + 1

    trades: list[dict[str, Any]] = []
    token_cache: dict[str, list[str] | None] = {}
    async with ClobClient() as clob:
        count = 0
        for _, raw in candidates.iterrows():
            mid = str(raw["id"])
            label_row = labels_by_id.loc[mid]
            resolution = str(label_row["resolution"])
            closed = raw["_closed_parsed"]
            if pd.isna(closed):
                _skip("no_resolved_at")
                continue
            end_ts = closed.timestamp()
            target_ts = end_ts - delta.total_seconds()

            condition_id = raw.get("conditionId") or raw.get("condition_id")
            if not condition_id:
                _skip("no_condition_id")
                continue
            token_ids = await _resolve_token_ids(clob, str(condition_id), token_cache)
            if not token_ids:
                _skip("no_clob_tokens")
                continue

            entry_prices = await _fetch_entry_prices(clob, token_ids, target_ts)
            if not any(p is not None and p > 0 for p in entry_prices):
                _skip("no_price_at_target")
                continue
            cheap = min(p for p in entry_prices if p is not None and p > 0)
            if cheap > max_entry_price:
                _skip(f"too_expensive_gt_{max_entry_price:.2f}")
                continue

            trade = simulate_trade(
                market_id=mid,
                slug=str(raw.get("slug", "")),
                resolution=resolution,
                entry_prices=(entry_prices[0], entry_prices[1]),
                capital=capital,
                max_entry_price=max_entry_price,
            )
            if trade is None:
                _skip("simulate_returned_none")
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

    if skip:
        print("backtest skip reasons:")
        for reason, n in sorted(skip.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {reason}")
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
