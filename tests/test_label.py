import pandas as pd

from polymarket_edge.label import classify_outcome, label_dataframe, summarize
from polymarket_edge.models import Resolution


def test_classify_fifty_fifty():
    assert classify_outcome([0.5, 0.5], closed=True) == Resolution.FIFTY_FIFTY


def test_classify_yes_and_no():
    assert classify_outcome([1.0, 0.0], closed=True) == Resolution.YES
    assert classify_outcome([0.0, 1.0], closed=True) == Resolution.NO


def test_classify_other_when_odd_prices():
    assert classify_outcome([0.8, 0.2], closed=True) == Resolution.OTHER


def test_classify_unresolved_when_open():
    assert classify_outcome([0.3, 0.7], closed=False) == Resolution.UNRESOLVED


def test_label_dataframe_handles_mixed_rows():
    rows = [
        {
            "id": "a",
            "question": "A",
            "slug": "a",
            "endDate": "2024-01-01T00:00:00Z",
            "closed": True,
            "outcomePrices": '["0.5", "0.5"]',
        },
        {
            "id": "b",
            "question": "B",
            "slug": "b",
            "endDate": "2024-01-02T00:00:00Z",
            "closed": True,
            "outcomePrices": '["1", "0"]',
        },
        {
            "id": "c",
            "question": "C",
            "slug": "c",
            "endDate": "2024-01-03T00:00:00Z",
            "closed": False,
            "outcomePrices": '["0.4", "0.6"]',
        },
    ]
    df = pd.DataFrame(rows)
    labels = label_dataframe(df)
    resolutions = dict(zip(labels["id"], labels["resolution"]))
    assert resolutions == {
        "a": Resolution.FIFTY_FIFTY.value,
        "b": Resolution.YES.value,
        "c": Resolution.UNRESOLVED.value,
    }
    s = summarize(labels)
    assert s[Resolution.FIFTY_FIFTY.value] == 1
    assert s[Resolution.YES.value] == 1
