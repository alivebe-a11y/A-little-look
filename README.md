# polymarket-edge

Research + backtest harness for Polymarket's **50/50 invalidation edge**.

When a Polymarket market can't be resolved cleanly (event cancelled, ambiguous
criteria, UMA oracle returns `UNKNOWN`/`p3`), **both YES and NO tokens pay
$0.50**. If you can flag a market as likely-to-invalidate *before* it
resolves, buying the side trading at e.g. $0.05 returns ~10x on cancellation.

This repo is **research-only for now** — it pulls historical markets, labels
the ones that resolved 50/50, runs heuristic features over them, and
backtests whether those heuristics would have been profitable. Auto-trading
is deferred until the edge is validated offline.

## Quickstart (local Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

polymarket-edge ingest --since 2024-01-01
polymarket-edge label
polymarket-edge features
polymarket-edge backtest --delta 24h --capital 1000
polymarket-edge report

# Browse the UI at http://localhost:8000
polymarket-edge web

pytest
```

## Quickstart (Docker)

```bash
docker compose build
docker compose up -d web                       # → http://localhost:8000

# One-shot data pipeline jobs (run whenever you want fresh data):
docker compose run --rm ingest
docker compose run --rm ingest polymarket-edge label
docker compose run --rm ingest polymarket-edge features
docker compose run --rm ingest polymarket-edge backtest --delta 24h --capital 1000
```

Data is persisted via the `./data` volume mount so results survive container
restarts. Drop your API token (if/when you get one) into `.env` and compose
will load it automatically.

## Web UI

The FastAPI app at `/` shows:
- pipeline status (what's been ingested / labelled / backtested),
- resolution breakdown (YES / NO / **FIFTY_FIFTY** / OTHER),
- per-feature-bucket backtest summary (hit rate, mean ROI, total PnL).

Drill-down pages:
- `/labels?resolution=FIFTY_FIFTY` — every market that resolved 50/50
- `/trades` — simulated trades + per-bucket breakdown
- `/api/summary` — JSON for programmatic consumers
- `/healthz` — liveness probe used by the Docker healthcheck

## Layout

```
src/polymarket_edge/
├── clients/gamma.py     # Polymarket Gamma API (market metadata)
├── clients/clob.py      # Polymarket CLOB REST (prices, trades)
├── clients/subgraph.py  # The Graph subgraph client (optional)
├── models.py            # pydantic Market
├── ingest.py            # paginate Gamma → parquet
├── label.py             # YES / NO / FIFTY_FIFTY / OTHER
├── features.py          # ambiguity & event-type heuristics
├── backtest.py          # buy-cheap-side simulation
└── cli.py               # argparse entry point
```

## Scope & ethics

This tool only consumes publicly-available market data. It does **not**
manipulate outcomes, file bad-faith UMA disputes, or trade on insider
information. Use at your own risk — Polymarket may be restricted or illegal
in your jurisdiction.
