from polymarket_edge.models import Market


def test_market_parses_json_encoded_list_fields():
    raw = {
        "id": "123",
        "question": "Will X happen?",
        "slug": "will-x-happen",
        "endDate": "2024-07-31T23:59:00Z",
        "closed": True,
        "outcomePrices": '["0.5", "0.5"]',
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["tok_a", "tok_b"]',
        "tags": [{"id": "1", "label": "Sports"}, "esports"],
        "description": "some text",
    }
    m = Market.model_validate(raw)
    assert m.outcome_prices == [0.5, 0.5]
    assert m.outcomes == ["Yes", "No"]
    assert m.clob_token_ids == ["tok_a", "tok_b"]
    assert set(m.tags) == {"Sports", "esports"}
    assert m.closed is True


def test_market_tolerates_missing_fields():
    m = Market.model_validate({"id": "xyz"})
    assert m.id == "xyz"
    assert m.outcome_prices is None
    assert m.tags == []
