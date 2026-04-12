"""Classify closed Polymarket markets by how they resolved.

A market resolves:
  - YES         if outcomePrices ~= [1, 0]
  - NO          if outcomePrices ~= [0, 1]
  - FIFTY_FIFTY if both prices ~= 0.5  (the invalidation / UMA p3 case)
  - OTHER       if neither (e.g. still-live, multi-outcome, or oddly priced)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .models import Market, Resolution

DEFAULT_LABELS_PATH = Path("data/reports/labels.csv")
# Tolerance when checking the 50/50 split — in practice Polymarket writes
# exactly 0.5 / 0.5 but being permissive is harmless.
EPS = 0.05


def classify_outcome(outcome_prices: list[float] | None, *, closed: bool) -> Resolution:
    if not closed:
        return Resolution.UNRESOLVED
    if not outcome_prices or len(outcome_prices) < 2:
        return Resolution.OTHER
    a, b = float(outcome_prices[0]), float(outcome_prices[1])
    if abs(a - 0.5) <= EPS and abs(b - 0.5) <= EPS:
        return Resolution.FIFTY_FIFTY
    if a >= 1 - EPS and b <= EPS:
        return Resolution.YES
    if a <= EPS and b >= 1 - EPS:
        return Resolution.NO
    return Resolution.OTHER


def classify_row(row: dict[str, Any]) -> Resolution:
    try:
        m = Market.model_validate(row)
    except Exception:
        return Resolution.OTHER
    return classify_outcome(m.outcome_prices, closed=m.closed)


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of (id, question, slug, end_date, resolution)."""
    out_rows = []
    for _, row in df.iterrows():
        try:
            m = Market.model_validate(row.to_dict())
        except Exception:
            continue
        res = classify_outcome(m.outcome_prices, closed=m.closed)
        out_rows.append(
            {
                "id": m.id,
                "slug": m.slug,
                "question": m.question,
                "end_date": m.end_date,
                "resolution": res.value,
                "outcome_prices": m.outcome_prices,
            }
        )
    return pd.DataFrame(out_rows)


def summarize(labels: pd.DataFrame) -> dict[str, int]:
    counts = labels["resolution"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}
