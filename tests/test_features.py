from polymarket_edge.features import (
    compute_features,
    has_explicit_date,
    is_ambiguous,
    is_esports_or_match,
    is_third_party_dependent,
)
from polymarket_edge.models import Market


def test_ambiguity_detection():
    assert is_ambiguous("Will this become popular by year end?")
    assert is_ambiguous("If notable news breaks...")
    assert not is_ambiguous("Will BTC close above $70,000 on 2024-12-31?")


def test_third_party_detection():
    assert is_third_party_dependent("Resolves according to official Riot announcement")
    assert not is_third_party_dependent("Resolves if the price > 100.")


def test_explicit_date_detection():
    assert has_explicit_date("Ends 2024-07-31.")
    assert has_explicit_date("Resolves on July 31, 2024.")
    assert has_explicit_date("Before Dec 2024 the event must...")
    assert not has_explicit_date("Some day in the future.")


def test_esports_from_tags():
    m = Market.model_validate(
        {"id": "x", "question": "Does team win?", "tags": ["esports"]}
    )
    assert is_esports_or_match(m)


def test_esports_from_keywords():
    m = Market.model_validate(
        {"id": "x", "question": "Will FaZe win their match tonight?"}
    )
    assert is_esports_or_match(m)


def test_compute_features_shape():
    m = Market.model_validate(
        {
            "id": "abc",
            "question": "Will FaZe win their popular match?",
            "description": "According to official Riot announcement.",
            "tags": ["esports"],
        }
    )
    fv = compute_features(m)
    assert fv.id == "abc"
    assert fv.ambig_wording is True
    assert fv.esports_or_match is True
    assert fv.third_party_dep is True
    assert fv.missing_explicit_date is True
