"""Orderbook snapshot ingest + analysis for market-making research.

Phase A of the MM effort: figure out *which* Polymarket markets are worth
quoting before we commit to building signing + execution infrastructure.

Flow:
  1. ``ingest_books`` snapshots the L2 orderbook of every open market's
     YES & NO tokens on a repeating schedule and writes one parquet per
     snapshot under ``data/raw/books/``.
  2. ``analyze_books`` reads every snapshot, collapses per-token time
     series into summary stats (median spread, depth inside 1%, mid
     volatility, number of snapshots), ranks markets by a simple
     "quotable score", and writes a CSV for inspection.

The "quotable score" is deliberately naive — we're filtering the universe
before any live trading, not optimizing execution. Refinement belongs in
the simulation phase (Phase B).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .clients.clob import ClobClient
from .ingest import load_raw_markets

BOOKS_DIR = Path("data/raw/books")
DEFAULT_BOOK_ANALYSIS_PATH = Path("data/reports/book_analysis.csv")

# Polymarket CLOB fee model as of writing: maker 0% / taker varies. We assume
# a minimum ~2¢ edge to beat gas + slippage + adverse selection. Markets with
# median spread below this are not quotable regardless of volume.
MIN_QUOTABLE_SPREAD = 0.02

# How deep "at top of book" is defined for depth stats. 1% of mid is a
# reasonable "would my order fill quickly" threshold for prediction markets.
DEPTH_BAND_PCT = 0.01


def _book_top(levels: list[dict[str, Any]], side: str) -> tuple[float, float] | None:
    """Return (price, size) of the best level, or None if empty.

    ``side`` is "bid" (want highest price) or "ask" (want lowest).
    """
    if not levels:
        return None
    parsed: list[tuple[float, float]] = []
    for lvl in levels:
        try:
            parsed.append((float(lvl["price"]), float(lvl["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not parsed:
        return None
    if side == "bid":
        return max(parsed, key=lambda pr: pr[0])
    return min(parsed, key=lambda pr: pr[0])


def _depth_within(levels: list[dict[str, Any]], mid: float, band: float) -> float:
    """Sum size across levels priced within ``band`` of ``mid``.

    Size of orders you'd need to clear to move the book 1% away from mid.
    """
    total = 0.0
    for lvl in levels:
        try:
            p = float(lvl["price"])
            s = float(lvl["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(p - mid) <= band:
            total += s
    return total


async def _snapshot_one(
    clob: ClobClient,
    token_id: str,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Fetch one token's book and flatten to a summary row for parquet."""
    try:
        book = await clob.get_book(token_id)
    except Exception:
        return None
    if not book:
        return None
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = _book_top(bids, "bid")
    best_ask = _book_top(asks, "ask")
    if not best_bid or not best_ask:
        return None
    bid_price, bid_size = best_bid
    ask_price, ask_size = best_ask
    mid = (bid_price + ask_price) / 2
    band = mid * DEPTH_BAND_PCT
    return {
        "ts": time.time(),
        "market_id": meta["market_id"],
        "slug": meta["slug"],
        "condition_id": meta["condition_id"],
        "token_id": token_id,
        "outcome_index": meta["outcome_index"],
        "best_bid": bid_price,
        "best_bid_size": bid_size,
        "best_ask": ask_price,
        "best_ask_size": ask_size,
        "spread": ask_price - bid_price,
        "mid": mid,
        "bid_depth_1pct": _depth_within(bids, mid, band),
        "ask_depth_1pct": _depth_within(asks, mid, band),
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
    }


async def _collect_targets(
    markets_df: pd.DataFrame,
    clob: ClobClient,
    limit: int | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve (token_id, meta) pairs for every open, known-to-CLOB market."""
    if "closed" in markets_df.columns:
        open_markets = markets_df[markets_df["closed"] != True]  # noqa: E712
    else:
        open_markets = markets_df
    if limit is not None:
        open_markets = open_markets.head(limit)

    targets: list[tuple[str, dict[str, Any]]] = []
    for _, raw in open_markets.iterrows():
        cid = raw.get("conditionId") or raw.get("condition_id")
        if not cid:
            continue
        try:
            mkt = await clob.get_market(str(cid))
        except Exception:
            continue
        if not mkt:
            continue
        for idx, tok in enumerate((mkt.get("tokens") or [])[:2]):
            tid = tok.get("token_id")
            if not tid:
                continue
            targets.append(
                (
                    str(tid),
                    {
                        "market_id": str(raw.get("id", "")),
                        "slug": str(raw.get("slug", "")),
                        "condition_id": str(cid),
                        "outcome_index": idx,
                    },
                )
            )
    return targets


async def snapshot_once(
    markets_df: pd.DataFrame,
    out_dir: Path = BOOKS_DIR,
    *,
    market_limit: int | None = None,
    concurrency: int = 10,
) -> Path | None:
    """Grab one orderbook snapshot across all open markets and write a parquet.

    Returns the file path, or None if nothing was snapshotted.
    """
    from tqdm.auto import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    async with ClobClient() as clob:
        targets = await _collect_targets(markets_df, clob, market_limit)
        sem = asyncio.Semaphore(concurrency)
        progress = tqdm(total=len(targets), desc="books", unit="tok", dynamic_ncols=True)

        async def _run(tid: str, meta: dict[str, Any]) -> None:
            async with sem:
                row = await _snapshot_one(clob, tid, meta)
                if row is not None:
                    rows.append(row)
                progress.update(1)

        await asyncio.gather(*(_run(tid, meta) for tid, meta in targets))
        progress.close()

    if not rows:
        return None
    df = pd.DataFrame(rows)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"books-{ts}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


async def ingest_books_loop(
    markets_df: pd.DataFrame,
    out_dir: Path = BOOKS_DIR,
    *,
    interval_seconds: int = 300,
    duration_seconds: int,
    market_limit: int | None = None,
    concurrency: int = 10,
) -> list[Path]:
    """Repeatedly snapshot books until ``duration_seconds`` elapses."""
    start = time.time()
    written: list[Path] = []
    while True:
        loop_started = time.time()
        path = await snapshot_once(
            markets_df, out_dir, market_limit=market_limit, concurrency=concurrency
        )
        if path is not None:
            written.append(path)
            print(f"  snapshot -> {path} ({path.stat().st_size} bytes)")
        if time.time() - start + interval_seconds >= duration_seconds:
            break
        elapsed = time.time() - loop_started
        await asyncio.sleep(max(1, interval_seconds - int(elapsed)))
    return written


def analyze_books(
    books_dir: Path = BOOKS_DIR,
    out_path: Path = DEFAULT_BOOK_ANALYSIS_PATH,
    *,
    min_snapshots: int = 3,
    min_spread: float = MIN_QUOTABLE_SPREAD,
) -> pd.DataFrame:
    """Collapse snapshots into per-market MM viability stats.

    Drops markets with fewer than ``min_snapshots`` observations (can't
    measure volatility with one point) and writes the ranked result to
    ``out_path``.

    Score = median spread × geometric-mean depth ÷ (1 + mid volatility).
    Bigger = better quoting candidate. This is a coarse filter, not an
    alpha model — it favors "wide spread, deep book, stable mid" which
    is what you want for basic two-sided market making.
    """
    files = sorted(books_dir.glob("books-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no snapshots in {books_dir}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    grouped = df.groupby(["market_id", "slug", "condition_id", "token_id", "outcome_index"])
    per_token = grouped.agg(
        n_snapshots=("ts", "count"),
        median_spread=("spread", "median"),
        mean_mid=("mid", "mean"),
        mid_vol=("mid", "std"),
        median_bid_depth=("bid_depth_1pct", "median"),
        median_ask_depth=("ask_depth_1pct", "median"),
    ).reset_index()

    per_token = per_token[per_token["n_snapshots"] >= min_snapshots]
    per_token["mid_vol"] = per_token["mid_vol"].fillna(0.0)
    # Geometric mean of the two sides' depth — punishes one-sided books.
    per_token["geo_depth"] = (per_token["median_bid_depth"].clip(lower=1) * per_token["median_ask_depth"].clip(lower=1)) ** 0.5
    per_token["quotable"] = per_token["median_spread"] >= min_spread
    per_token["score"] = (
        per_token["median_spread"] * per_token["geo_depth"] / (1.0 + per_token["mid_vol"])
    ).where(per_token["quotable"], 0.0)

    per_token = per_token.sort_values("score", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_token.to_csv(out_path, index=False)
    return per_token


def load_markets_or_die(path: Path) -> pd.DataFrame:
    """Helper: load markets.parquet and ensure it's not empty."""
    df = load_raw_markets(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty — run `ingest` first")
    return df
