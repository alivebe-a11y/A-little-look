from datetime import timedelta

import pandas as pd

from polymarket_edge.backtest import (
    parse_delta,
    payout_for,
    simulate_trade,
    summarize_backtest,
)
from polymarket_edge.models import Resolution


def test_parse_delta_variants():
    assert parse_delta("24h") == timedelta(hours=24)
    assert parse_delta("30m") == timedelta(minutes=30)
    assert parse_delta("2d") == timedelta(days=2)
    assert parse_delta("45s") == timedelta(seconds=45)


def test_payout_tables():
    assert payout_for(Resolution.FIFTY_FIFTY.value, 0) == 0.5
    assert payout_for(Resolution.FIFTY_FIFTY.value, 1) == 0.5
    assert payout_for(Resolution.YES.value, 0) == 1.0
    assert payout_for(Resolution.YES.value, 1) == 0.0
    assert payout_for(Resolution.NO.value, 0) == 0.0
    assert payout_for(Resolution.NO.value, 1) == 1.0


def test_simulate_50_50_yields_the_edge():
    # Buy 1000 shares of the YES token at $0.05. Market resolves 50/50:
    # payout = 20000 * 0.5 = 10_000. pnl = +9_000.
    trade = simulate_trade(
        market_id="m1",
        slug="m1",
        resolution=Resolution.FIFTY_FIFTY.value,
        entry_prices=(0.05, 0.95),
        capital=1000.0,
        max_entry_price=0.10,
    )
    assert trade is not None
    assert trade.entry_price == 0.05
    assert trade.payout_per_share == 0.5
    # shares = 1000 / 0.05 = 20_000; payout = 10_000; pnl = 9_000
    assert round(trade.pnl, 2) == 9000.0
    assert round(trade.roi, 2) == 9.0


def test_simulate_skips_expensive_sides():
    # Both sides above the max_entry_price threshold.
    trade = simulate_trade(
        market_id="m1",
        slug="m1",
        resolution=Resolution.FIFTY_FIFTY.value,
        entry_prices=(0.45, 0.55),
        capital=1000.0,
        max_entry_price=0.10,
    )
    assert trade is None


def test_simulate_yes_resolution_on_wrong_side():
    # Bought NO at $0.05 but market resolved YES → total loss.
    trade = simulate_trade(
        market_id="m1",
        slug="m1",
        resolution=Resolution.YES.value,
        entry_prices=(0.95, 0.05),
        capital=1000.0,
        max_entry_price=0.10,
    )
    assert trade is not None
    assert trade.payout_per_share == 0.0
    assert trade.pnl == -1000.0


def test_simulate_yes_resolution_on_right_side():
    # Bought YES at $0.05, market resolved YES → 20x payout.
    trade = simulate_trade(
        market_id="m1",
        slug="m1",
        resolution=Resolution.YES.value,
        entry_prices=(0.05, 0.95),
        capital=1000.0,
        max_entry_price=0.10,
    )
    assert trade is not None
    assert trade.payout_per_share == 1.0
    assert round(trade.pnl, 2) == 19000.0


def test_summarize_backtest_buckets_by_feature():
    trades = pd.DataFrame(
        [
            {"market_id": "a", "pnl": 500.0, "roi": 0.5, "feat_ambig_wording": True},
            {"market_id": "b", "pnl": -200.0, "roi": -0.2, "feat_ambig_wording": True},
            {"market_id": "c", "pnl": 100.0, "roi": 0.1, "feat_ambig_wording": False},
        ]
    )
    summary = summarize_backtest(trades)
    buckets = summary.set_index("bucket")["n"].to_dict()
    assert buckets["ALL"] == 3
    assert buckets["feat_ambig_wording=True"] == 2
    assert buckets["feat_ambig_wording=False"] == 1
