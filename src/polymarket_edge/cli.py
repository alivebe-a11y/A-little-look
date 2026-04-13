"""argparse CLI: ingest | label | features | backtest | report."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

from . import backtest as bt
from .features import compute_features
from .ingest import DEFAULT_RAW_PATH, ingest_markets, load_raw_markets
from .label import DEFAULT_LABELS_PATH, label_dataframe, summarize
from .models import Market

DEFAULT_FEATURES_PATH = Path("data/reports/features.csv")


def cmd_ingest(args: argparse.Namespace) -> int:
    count = asyncio.run(
        ingest_markets(
            out_path=args.out,
            start_date_min=args.since,
            closed=None if args.all else True,
            max_pages=args.max_pages,
        )
    )
    print(f"wrote {count} markets to {args.out}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    df = load_raw_markets(args.in_path)
    labels = label_dataframe(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)
    print(f"wrote {len(labels)} label rows to {args.out}")
    print("summary:", summarize(labels))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    df = load_raw_markets(args.in_path)
    rows = []
    for _, raw in df.iterrows():
        try:
            m = Market.model_validate(raw.to_dict())
        except Exception:
            continue
        rows.append(compute_features(m).as_dict())
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote features for {len(out)} markets to {args.out}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    markets = load_raw_markets(args.in_path)
    labels = pd.read_csv(args.labels)
    features = (
        pd.read_csv(args.features) if args.features.exists() else pd.DataFrame()
    )
    delta = bt.parse_delta(args.delta)
    trades = asyncio.run(
        bt.run_backtest(
            markets,
            labels,
            features,
            delta=delta,
            capital=args.capital,
            max_entry_price=args.max_entry_price,
            limit=args.limit,
            max_age_days=args.max_age_days,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.out, index=False)
    print(f"wrote {len(trades)} simulated trades to {args.out}")
    if not trades.empty:
        print("summary:")
        print(bt.summarize_backtest(trades).to_string(index=False))
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "polymarket_edge.web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if not args.trades.exists():
        print(f"no trades file at {args.trades} — run backtest first", file=sys.stderr)
        return 1
    try:
        trades = pd.read_csv(args.trades)
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame()
    if trades.empty:
        print(f"{args.trades} has no trades — backtest produced 0 rows")
        return 0
    summary = bt.summarize_backtest(trades)
    print(summary.to_string(index=False))

    # Concentration check: how much of total PnL is driven by the top N wins?
    # If the "edge" is really 3 lucky markets, we want to see it clearly.
    sorted_by_pnl = trades.sort_values("pnl", ascending=False)
    total_pnl = float(trades["pnl"].sum())
    print("\ntop 20 winners (most pnl):")
    top_cols = [c for c in ["slug", "resolution", "entry_price", "pnl", "roi"] if c in trades.columns]
    print(sorted_by_pnl.head(20)[top_cols].to_string(index=False))
    top20_pnl = float(sorted_by_pnl.head(20)["pnl"].sum())
    top5_pnl = float(sorted_by_pnl.head(5)["pnl"].sum())
    print(
        f"\nconcentration: top-5 winners = ${top5_pnl:,.0f}, "
        f"top-20 winners = ${top20_pnl:,.0f}, total = ${total_pnl:,.0f}"
    )
    if total_pnl != 0:
        print(
            f"  top-5 share: {top5_pnl / total_pnl:+.1%}  "
            f"top-20 share: {top20_pnl / total_pnl:+.1%}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polymarket-edge")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="fetch markets from Polymarket Gamma")
    s.add_argument("--out", type=Path, default=DEFAULT_RAW_PATH)
    s.add_argument("--since", default=None, help="e.g. 2024-01-01")
    s.add_argument("--all", action="store_true", help="include still-open markets")
    s.add_argument("--max-pages", type=int, default=None)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("label", help="classify closed markets YES/NO/50-50/OTHER")
    s.add_argument("--in-path", type=Path, default=DEFAULT_RAW_PATH)
    s.add_argument("--out", type=Path, default=DEFAULT_LABELS_PATH)
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("features", help="compute heuristic features per market")
    s.add_argument("--in-path", type=Path, default=DEFAULT_RAW_PATH)
    s.add_argument("--out", type=Path, default=DEFAULT_FEATURES_PATH)
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("backtest", help="simulate the cheap-side strategy")
    s.add_argument("--in-path", type=Path, default=DEFAULT_RAW_PATH)
    s.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    s.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    s.add_argument("--out", type=Path, default=Path("data/reports/trades.csv"))
    s.add_argument("--delta", default="24h")
    s.add_argument("--capital", type=float, default=1000.0)
    s.add_argument("--max-entry-price", type=float, default=0.10)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument(
        "--max-age-days",
        type=int,
        default=140,
        help="skip markets resolved longer ago than this (CLOB retention ~140d)",
    )
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("web", help="serve the FastAPI web UI")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_web)

    s = sub.add_parser("report", help="print summary from a prior backtest run")
    s.add_argument("--trades", type=Path, default=Path("data/reports/trades.csv"))
    s.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
