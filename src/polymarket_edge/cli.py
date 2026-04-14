"""argparse CLI: ingest | label | features | backtest | report."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

from . import backtest as bt
from . import books as bk
from .backtest import _resolve_token_ids
from .clients.clob import ClobClient
from .clients.subgraph import SubgraphAuthError, SubgraphClient, derive_price_and_side
from .features import compute_features
from .ingest import DEFAULT_RAW_PATH, ingest_markets, load_raw_markets
from .label import DEFAULT_LABELS_PATH, label_dataframe, summarize
from .models import Market, Resolution

DEFAULT_FEATURES_PATH = Path("data/reports/features.csv")
DEFAULT_TRADES_SUBGRAPH_PATH = Path("data/raw/trades_subgraph.parquet")


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


async def _ingest_trades(
    markets: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    out_path: Path,
    resolutions: set[str],
    limit: int | None,
    per_market_limit: int | None,
) -> int:
    """Pull trades for target markets from the subgraph and cache to parquet.

    ``resolutions`` filters which labelled rows to include (default: just
    FIFTY_FIFTY since that's the thesis we're testing). ``per_market_limit``
    caps trade count per market to keep API usage predictable on the free
    tier (100k queries/mo).
    """
    labels = labels.copy()
    labels["id"] = labels["id"].astype(str)
    target_ids = {
        str(row["id"])
        for _, row in labels.iterrows()
        if str(row["resolution"]) in resolutions
    }

    markets = markets.copy()
    markets["id"] = markets["id"].astype(str)
    target = markets[markets["id"].isin(target_ids)]
    target = target[target.get("conditionId").notna()] if "conditionId" in target.columns else target
    if limit is not None:
        target = target.head(limit)

    out_rows: list[dict[str, object]] = []
    from tqdm.auto import tqdm

    progress = tqdm(total=len(target), desc="subgraph", unit="mkt", dynamic_ncols=True)
    token_cache: dict[str, list[str] | None] = {}
    skipped_no_tokens = 0
    async with ClobClient() as clob, SubgraphClient() as sg:
        for _, raw in target.iterrows():
            progress.update(1)
            mid = str(raw["id"])
            cid = raw.get("conditionId") or raw.get("condition_id")
            if not cid:
                continue
            token_ids = await _resolve_token_ids(clob, str(cid), token_cache)
            if not token_ids:
                skipped_no_tokens += 1
                continue
            try:
                trades = await sg.trades_for_token_ids(
                    token_ids[:2], first=per_market_limit
                )
            except Exception as e:
                print(f"  {mid} {cid}: {e}", file=sys.stderr)
                continue
            for t in trades:
                enriched = derive_price_and_side(t, token_ids[:2])
                enriched["market_id"] = mid
                enriched["conditionId"] = str(cid)
                out_rows.append(enriched)
            progress.set_postfix(trades=len(out_rows), refresh=False)
    progress.close()
    if skipped_no_tokens:
        print(f"  skipped {skipped_no_tokens} markets with no CLOB tokens", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        pd.DataFrame().to_parquet(out_path, index=False)
        return 0
    df = pd.DataFrame(out_rows)
    df.to_parquet(out_path, index=False)
    return len(df)


async def _inspect_subgraph(subgraph_id: str | None) -> list[dict[str, object]]:
    """Introspect the top-level Query fields of the configured subgraph.

    Used to debug schema drift — the field names in our GraphQL queries
    must match what the subgraph actually exposes. When the orderbook
    subgraph migrates or we've grabbed the wrong ID, this is how we find
    out without guessing.
    """
    gql = "{ __schema { queryType { fields { name args { name } } } } }"
    async with SubgraphClient(subgraph_id=subgraph_id) as sg:
        data = await sg.query(gql)
    return data["__schema"]["queryType"]["fields"]


async def _inspect_subgraph_type(subgraph_id: str | None, type_name: str) -> list[dict[str, object]]:
    """Introspect the fields of a specific GraphQL type on the subgraph."""
    gql = """
    query($n: String!) {
      __type(name: $n) {
        name
        fields {
          name
          type {
            name
            kind
            ofType { name kind ofType { name kind } }
          }
        }
      }
    }
    """
    async with SubgraphClient(subgraph_id=subgraph_id) as sg:
        data = await sg.query(gql, variables={"n": type_name})
    t = data.get("__type")
    return t["fields"] if t else []


def _format_type(t: dict[str, object]) -> str:
    """Flatten a nested GraphQL type ref into a readable string."""
    if not t:
        return "?"
    kind = t.get("kind")
    name = t.get("name")
    of = t.get("ofType")
    if kind == "NON_NULL":
        return _format_type(of) + "!"
    if kind == "LIST":
        return "[" + _format_type(of) + "]"
    return str(name or "?")


def cmd_inspect_subgraph(args: argparse.Namespace) -> int:
    try:
        if args.type:
            fields = asyncio.run(_inspect_subgraph_type(args.id, args.type))
            if not fields:
                print(f"type {args.type!r} not found", file=sys.stderr)
                return 1
            for f in fields:
                print(f"  {f['name']}: {_format_type(f['type'])}")
            print(f"\n{len(fields)} fields on {args.type}")
        else:
            fields = asyncio.run(_inspect_subgraph(args.id))
            for f in sorted(fields, key=lambda f: f["name"]):
                arg_names = ", ".join(a["name"] for a in f.get("args", []))
                print(f"  {f['name']}({arg_names})")
            print(f"\n{len(fields)} top-level query fields")
    except SubgraphAuthError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


def cmd_ingest_trades(args: argparse.Namespace) -> int:
    markets = load_raw_markets(args.in_path)
    labels = pd.read_csv(args.labels)
    resolutions = set(args.resolutions.split(",")) if args.resolutions else {Resolution.FIFTY_FIFTY.value}
    try:
        n = asyncio.run(
            _ingest_trades(
                markets,
                labels,
                out_path=args.out,
                resolutions=resolutions,
                limit=args.limit,
                per_market_limit=args.per_market_limit,
            )
        )
    except SubgraphAuthError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"wrote {n} trade rows to {args.out}")
    return 0


def cmd_backtest_subgraph(args: argparse.Namespace) -> int:
    """Replay cheap-side strategy using the subgraph trade cache.

    Unlike the CLOB-backed backtest, this has full history (not just 140
    days), so we can actually measure the cheap-side strategy on the
    ~519 FIFTY_FIFTY markets instead of the ~20 that fit the CLOB window.
    """
    if not args.trades.exists():
        print(f"no trades file at {args.trades} — run ingest-trades first", file=sys.stderr)
        return 1
    trades = pd.read_parquet(args.trades)
    if trades.empty:
        print("trades file is empty")
        return 0

    markets = load_raw_markets(args.in_path)
    markets["id"] = markets["id"].astype(str)
    labels = pd.read_csv(args.labels)
    labels["id"] = labels["id"].astype(str)
    labels_by_id = labels.set_index("id")

    markets["_closed"] = pd.to_datetime(markets.get("closedTime"), errors="coerce", utc=True)

    delta = bt.parse_delta(args.delta)
    capital = args.capital
    max_entry = args.max_entry_price

    # For each (market, outcome) find the last trade at or before target_ts.
    trades = trades.sort_values("timestamp_ts")
    simulated: list[dict[str, object]] = []
    skip: dict[str, int] = {}

    def _skip(r: str) -> None:
        skip[r] = skip.get(r, 0) + 1

    for mid, market_trades in trades.groupby("market_id"):
        mid = str(mid)
        if mid not in labels_by_id.index:
            _skip("no_label")
            continue
        resolution = str(labels_by_id.loc[mid]["resolution"])
        mrow = markets[markets["id"] == mid]
        if mrow.empty or pd.isna(mrow["_closed"].iloc[0]):
            _skip("no_closed_time")
            continue
        target_ts = mrow["_closed"].iloc[0].timestamp() - delta.total_seconds()

        prior = market_trades[market_trades["timestamp_ts"] <= target_ts]
        if prior.empty:
            _skip("no_trade_before_target")
            continue
        entry_by_outcome: dict[int, float] = {}
        for oi in (0, 1):
            side_trades = prior[prior["outcome_index"] == oi]
            if side_trades.empty:
                continue
            entry_by_outcome[oi] = float(side_trades["price"].iloc[-1])
        if not entry_by_outcome:
            _skip("no_price_either_side")
            continue

        side_index = min(entry_by_outcome, key=lambda k: entry_by_outcome[k])
        entry_price = entry_by_outcome[side_index]
        if entry_price <= 0:
            _skip("zero_price")
            continue
        if entry_price > max_entry:
            _skip(f"too_expensive_gt_{max_entry:.2f}")
            continue

        shares = capital / entry_price
        payout_per_share = bt.payout_for(resolution, side_index)
        pnl = shares * payout_per_share - capital
        simulated.append(
            {
                "market_id": mid,
                "slug": str(mrow["slug"].iloc[0]) if "slug" in mrow.columns else "",
                "resolution": resolution,
                "entry_price": entry_price,
                "side_index": side_index,
                "payout_per_share": payout_per_share,
                "capital": capital,
                "pnl": pnl,
                "roi": pnl / capital,
            }
        )

    out_df = pd.DataFrame(simulated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} simulated trades to {args.out}")
    if skip:
        print("skip reasons:")
        for reason, n in sorted(skip.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {reason}")
    if not out_df.empty:
        print(bt.summarize_backtest(out_df).to_string(index=False))
    return 0


def cmd_ingest_books(args: argparse.Namespace) -> int:
    paths = asyncio.run(
        bk.ingest_books_loop(
            out_dir=args.out_dir,
            interval_seconds=args.interval,
            duration_seconds=args.duration,
            market_limit=args.limit,
            concurrency=args.concurrency,
        )
    )
    print(f"wrote {len(paths)} snapshot file(s) to {args.out_dir}")
    return 0


def cmd_analyze_books(args: argparse.Namespace) -> int:
    try:
        ranked = bk.analyze_books(
            books_dir=args.books_dir,
            out_path=args.out,
            min_snapshots=args.min_snapshots,
            min_spread=args.min_spread,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"wrote {len(ranked)} ranked tokens to {args.out}")
    cols = [
        "slug",
        "outcome_index",
        "n_snapshots",
        "median_spread",
        "mean_mid",
        "mid_vol",
        "geo_depth",
        "quotable",
        "score",
    ]
    cols = [c for c in cols if c in ranked.columns]
    head = ranked[cols].head(args.top)
    print(head.to_string(index=False))
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

    s = sub.add_parser(
        "inspect-subgraph",
        help="list the top-level query fields exposed by the subgraph (schema debug)",
    )
    s.add_argument(
        "--id",
        default=None,
        help="override subgraph ID (else THEGRAPH_SUBGRAPH_ID env or built-in default)",
    )
    s.add_argument(
        "--type",
        default=None,
        help="print fields of a specific GraphQL type (e.g. OrderFilled)",
    )
    s.set_defaults(func=cmd_inspect_subgraph)

    s = sub.add_parser(
        "ingest-trades",
        help="pull historical trades from the Polymarket subgraph (full history, needs THEGRAPH_API_KEY)",
    )
    s.add_argument("--in-path", type=Path, default=DEFAULT_RAW_PATH)
    s.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    s.add_argument("--out", type=Path, default=DEFAULT_TRADES_SUBGRAPH_PATH)
    s.add_argument(
        "--resolutions",
        default=Resolution.FIFTY_FIFTY.value,
        help="comma-separated resolutions to fetch (default: FIFTY_FIFTY only)",
    )
    s.add_argument("--limit", type=int, default=None, help="max markets to fetch")
    s.add_argument(
        "--per-market-limit",
        type=int,
        default=None,
        help="cap trades per market (default: unlimited — paginates fully)",
    )
    s.set_defaults(func=cmd_ingest_trades)

    s = sub.add_parser(
        "backtest-subgraph",
        help="cheap-side backtest against trades_subgraph.parquet (full history, not CLOB-limited)",
    )
    s.add_argument("--in-path", type=Path, default=DEFAULT_RAW_PATH)
    s.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    s.add_argument("--trades", type=Path, default=DEFAULT_TRADES_SUBGRAPH_PATH)
    s.add_argument("--out", type=Path, default=Path("data/reports/trades_subgraph_bt.csv"))
    s.add_argument("--delta", default="24h")
    s.add_argument("--capital", type=float, default=1000.0)
    s.add_argument("--max-entry-price", type=float, default=0.10)
    s.set_defaults(func=cmd_backtest_subgraph)

    s = sub.add_parser(
        "ingest-books",
        help="snapshot orderbooks for open markets on a schedule (MM research, Phase A)",
    )
    s.add_argument("--out-dir", type=Path, default=bk.BOOKS_DIR)
    s.add_argument("--interval", type=int, default=300, help="seconds between snapshots")
    s.add_argument("--duration", type=int, default=3600, help="total seconds to run for")
    s.add_argument("--limit", type=int, default=None, help="cap # markets per snapshot")
    s.add_argument("--concurrency", type=int, default=10, help="parallel /book fetches")
    s.set_defaults(func=cmd_ingest_books)

    s = sub.add_parser(
        "analyze-books",
        help="rank markets by MM quotability using snapshots from ingest-books",
    )
    s.add_argument("--books-dir", type=Path, default=bk.BOOKS_DIR)
    s.add_argument("--out", type=Path, default=bk.DEFAULT_BOOK_ANALYSIS_PATH)
    s.add_argument("--min-snapshots", type=int, default=3)
    s.add_argument(
        "--min-spread",
        type=float,
        default=bk.MIN_QUOTABLE_SPREAD,
        help="spread floor below which markets are flagged unquotable",
    )
    s.add_argument("--top", type=int, default=30, help="rows to print to stdout")
    s.set_defaults(func=cmd_analyze_books)

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
