"""Minimal FastAPI UI for viewing ingest / label / feature / backtest outputs.

All data is read from the parquet/CSV files under ``data/``. The web tier
is read-only — kicking off an ingest run still goes through the CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .backtest import summarize_backtest
from .ingest import DEFAULT_RAW_PATH, load_raw_markets
from .label import DEFAULT_LABELS_PATH, summarize

TEMPLATES_DIR = Path(__file__).parent / "templates"
DEFAULT_FEATURES_PATH = Path("data/reports/features.csv")
DEFAULT_TRADES_PATH = Path("data/reports/trades.csv")


def _data_root() -> Path:
    # Allow overriding where we read data from (useful inside Docker).
    import os
    return Path(os.environ.get("POLYMARKET_EDGE_DATA", "data"))


def _path(default: Path, *, name: str) -> Path:
    root = _data_root()
    rel = default
    if default.is_absolute():
        return default
    # Rewrite default "data/..." paths to use the configured data root.
    parts = rel.parts
    if parts and parts[0] == "data":
        return root / Path(*parts[1:])
    return root / name


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def create_app() -> FastAPI:
    app = FastAPI(title="polymarket-edge")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        labels = _load_csv(_path(DEFAULT_LABELS_PATH, name="reports/labels.csv"))
        trades = _load_csv(_path(DEFAULT_TRADES_PATH, name="reports/trades.csv"))
        markets_path = _path(DEFAULT_RAW_PATH, name="raw/markets.parquet")

        label_summary = summarize(labels) if not labels.empty else {}
        trade_summary = (
            summarize_backtest(trades).to_dict(orient="records")
            if not trades.empty
            else []
        )

        status = {
            "markets_ingested": markets_path.exists(),
            "labels_present": not labels.empty,
            "trades_present": not trades.empty,
            "n_markets_labelled": int(len(labels)),
            "n_trades": int(len(trades)),
        }

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "status": status,
                "label_summary": label_summary,
                "trade_summary": trade_summary,
            },
        )

    @app.get("/labels", response_class=HTMLResponse)
    def labels_page(request: Request, resolution: str | None = None, limit: int = 200) -> Any:
        labels = _load_csv(_path(DEFAULT_LABELS_PATH, name="reports/labels.csv"))
        if labels.empty:
            raise HTTPException(404, "no labels — run `polymarket-edge label` first")
        filtered = labels
        if resolution:
            filtered = filtered[filtered["resolution"] == resolution]
        rows = filtered.head(limit).to_dict(orient="records")
        return templates.TemplateResponse(
            request,
            "labels.html",
            {
                "rows": rows,
                "total": int(len(filtered)),
                "resolution": resolution,
                "available_resolutions": sorted(labels["resolution"].unique().tolist()),
            },
        )

    @app.get("/trades", response_class=HTMLResponse)
    def trades_page(request: Request, limit: int = 200) -> Any:
        trades = _load_csv(_path(DEFAULT_TRADES_PATH, name="reports/trades.csv"))
        if trades.empty:
            raise HTTPException(404, "no trades — run `polymarket-edge backtest` first")
        rows = trades.head(limit).to_dict(orient="records")
        summary = summarize_backtest(trades).to_dict(orient="records")
        return templates.TemplateResponse(
            request,
            "trades.html",
            {
                "rows": rows,
                "summary": summary,
                "total": int(len(trades)),
            },
        )

    @app.get("/api/summary")
    def api_summary() -> JSONResponse:
        labels = _load_csv(_path(DEFAULT_LABELS_PATH, name="reports/labels.csv"))
        trades = _load_csv(_path(DEFAULT_TRADES_PATH, name="reports/trades.csv"))
        return JSONResponse(
            {
                "labels": summarize(labels) if not labels.empty else {},
                "trades": (
                    summarize_backtest(trades).to_dict(orient="records")
                    if not trades.empty
                    else []
                ),
            }
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
