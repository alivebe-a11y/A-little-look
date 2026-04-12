import os
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient


def _build_client(tmp_path: Path) -> TestClient:
    (tmp_path / "raw").mkdir()
    (tmp_path / "reports").mkdir()
    # Minimal labels.csv
    pd.DataFrame(
        [
            {
                "id": "1",
                "slug": "faze-match",
                "question": "Will FaZe win?",
                "end_date": "2024-07-31T00:00:00Z",
                "resolution": "FIFTY_FIFTY",
                "outcome_prices": "[0.5, 0.5]",
            },
            {
                "id": "2",
                "slug": "btc-70k",
                "question": "BTC > 70k?",
                "end_date": "2024-12-31T00:00:00Z",
                "resolution": "YES",
                "outcome_prices": "[1.0, 0.0]",
            },
        ]
    ).to_csv(tmp_path / "reports" / "labels.csv", index=False)
    # Minimal trades.csv
    pd.DataFrame(
        [
            {
                "market_id": "1",
                "slug": "faze-match",
                "resolution": "FIFTY_FIFTY",
                "entry_price": 0.05,
                "payout_per_share": 0.5,
                "capital": 1000.0,
                "pnl": 9000.0,
                "roi": 9.0,
                "feat_esports_or_match": True,
            }
        ]
    ).to_csv(tmp_path / "reports" / "trades.csv", index=False)

    os.environ["POLYMARKET_EDGE_DATA"] = str(tmp_path)
    from polymarket_edge.web import create_app

    return TestClient(create_app())


def test_index_renders(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert "Polymarket" in r.text
    assert "FIFTY_FIFTY" in r.text


def test_labels_page_filters(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    r = client.get("/labels?resolution=FIFTY_FIFTY")
    assert r.status_code == 200
    assert "faze-match" in r.text
    assert "btc-70k" not in r.text


def test_trades_page_renders(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    r = client.get("/trades")
    assert r.status_code == 200
    assert "faze-match" in r.text
    assert "9000" in r.text


def test_api_summary(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    r = client.get("/api/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["labels"].get("FIFTY_FIFTY") == 1
    assert isinstance(data["trades"], list)


def test_healthz(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    r = client.get("/healthz")
    assert r.json() == {"status": "ok"}
