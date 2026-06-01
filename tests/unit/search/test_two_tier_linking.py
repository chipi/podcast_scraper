"""Unit tests for gi.json grounding-quote extraction (RFC-090 Phase 2 / C1).

No LanceDB needed — _insight_grounding_quotes is pure parsing.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.search.two_tier_indexer import _insight_grounding_quotes

pytestmark = pytest.mark.unit


def test_missing_file_returns_empty(tmp_path):
    assert _insight_grounding_quotes(tmp_path / "none.gi.json") == {}


def test_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "bad.gi.json"
    p.write_text("{ not json", encoding="utf-8")
    assert _insight_grounding_quotes(p) == {}


def test_extracts_first_supporting_quote_timestamps(tmp_path):
    p = tmp_path / "g.gi.json"
    p.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "quote:q1",
                        "type": "Quote",
                        "properties": {"timestamp_start_ms": 1000, "timestamp_end_ms": 4000},
                    },
                    {
                        "id": "quote:q2",
                        "type": "Quote",
                        "properties": {"timestamp_start_ms": 9000},
                    },  # no end → None
                ],
                "edges": [
                    {"type": "SUPPORTED_BY", "from": "insight:n1", "to": "quote:q1"},
                    {"type": "SUPPORTED_BY", "from": "insight:n2", "to": "quote:q2"},
                    {"type": "ABOUT", "from": "insight:n1", "to": "topic:x"},  # ignored
                ],
            }
        ),
        encoding="utf-8",
    )
    out = _insight_grounding_quotes(p)
    assert out["insight:n1"] == (1.0, 4.0)
    assert out["insight:n2"] == (9.0, None)  # missing end tolerated
