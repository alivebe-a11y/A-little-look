"""Pydantic models for Polymarket API responses.

We're intentionally permissive: Polymarket's Gamma API changes shape over
time and we only care about a handful of fields. Unknown fields are kept so
they're available later via the `.extra` attribute on the backing dict.
"""
from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Resolution(str, Enum):
    YES = "YES"
    NO = "NO"
    FIFTY_FIFTY = "FIFTY_FIFTY"
    OTHER = "OTHER"
    UNRESOLVED = "UNRESOLVED"


class Market(BaseModel):
    """Subset of Polymarket Gamma /markets fields we rely on."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    question: str = ""
    slug: str = ""
    description: str = ""
    end_date: datetime | None = Field(default=None, alias="endDate")
    closed: bool = False
    archived: bool = False
    # Gamma returns outcomePrices as a JSON-encoded string list, e.g. '["0.5","0.5"]'
    outcome_prices: list[float] | None = Field(default=None, alias="outcomePrices")
    outcomes: list[str] | None = None
    uma_resolution_status: str | None = Field(default=None, alias="umaResolutionStatus")
    resolution_source: str | None = Field(default=None, alias="resolutionSource")
    tags: list[str] = Field(default_factory=list)
    clob_token_ids: list[str] | None = Field(default=None, alias="clobTokenIds")

    @field_validator("outcome_prices", "outcomes", "tags", "clob_token_ids", mode="before")
    @classmethod
    def _parse_json_strings(cls, v: Any) -> Any:
        """Gamma frequently returns list-shaped fields as JSON-encoded strings."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except (ValueError, TypeError):
                return v
            return parsed
        return v

    @field_validator("outcome_prices", mode="after")
    @classmethod
    def _coerce_floats(cls, v: list[Any] | None) -> list[float] | None:
        if v is None:
            return v
        return [float(x) for x in v]

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    # Gamma sometimes returns [{"id": "...", "label": "Sports"}]
                    label = item.get("label") or item.get("slug") or item.get("name")
                    if isinstance(label, str):
                        out.append(label)
            return out
        return []
