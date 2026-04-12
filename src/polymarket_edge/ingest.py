"""Paginate Gamma markets and dump to a parquet cache."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .clients.gamma import GammaClient

DEFAULT_RAW_PATH = Path("data/raw/markets.parquet")


async def ingest_markets(
    out_path: Path = DEFAULT_RAW_PATH,
    *,
    start_date_min: str | None = None,
    closed: bool | None = None,
    max_pages: int | None = None,
) -> int:
    """Fetch markets from Gamma and write to parquet. Returns row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    async with GammaClient() as gc:
        async for m in gc.iter_markets(
            closed=closed, start_date_min=start_date_min, max_pages=max_pages
        ):
            rows.append(m)
    if not rows:
        return 0

    df = pd.DataFrame(_flatten_for_parquet(rows))
    df.to_parquet(out_path, index=False)
    return len(df)


def _flatten_for_parquet(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parquet can't store arbitrarily-nested dicts reliably. JSON-encode the
    messy fields so they round-trip losslessly."""
    flat: list[dict[str, Any]] = []
    for r in rows:
        out: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (dict, list)):
                out[k] = json.dumps(v, default=str)
            else:
                out[k] = v
        flat.append(out)
    return flat


def load_raw_markets(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Decode the JSON-encoded columns back for downstream consumers.
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], str):
                s = sample.iloc[0].lstrip()
                if s.startswith("[") or s.startswith("{"):
                    df[col] = df[col].map(_safe_json)
    return df


def _safe_json(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v
