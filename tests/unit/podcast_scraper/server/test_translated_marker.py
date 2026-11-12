"""D-36 slice (b), operator 2026-10-13: a translated quote says so wherever the app shows it.

The pipeline writes a ``translation`` block on every Quote node of a translated episode
(``translation/provenance.py``). The app's quote, insight and moment payloads carry its source
language as ``translated_from`` so the player can label the line "Translated from Spanish"; a
native-English quote carries ``None``.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from podcast_scraper.translation.provenance import translated_from

pytestmark = pytest.mark.unit


def _provenance(lang: str) -> dict[str, Any]:
    return {
        "translated": True,
        "source_language": lang,
        "model": "google/translategemma-12b-it",
        "unit_ids": ["u0"],
        "en_sha256": "x",
        "prompt_sha256": "y",
    }


def _artifact(source_language: Optional[str]) -> dict[str, Any]:
    quote_props: dict[str, Any] = {
        "text": "what she said",
        "timestamp_start_ms": 60_000,
        "timestamp_end_ms": 75_000,
    }
    if source_language:
        quote_props["translation"] = _provenance(source_language)
    return {
        "nodes": [
            {"id": "person:ann", "type": "Person", "properties": {"name": "Ann"}},
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {
                    "text": "the point",
                    "grounded": True,
                    "insight_type": "claim",
                    "routing_tag": "surface",
                },
            },
            {"id": "quote:1", "type": "Quote", "properties": quote_props},
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"},
            {"type": "SPOKEN_BY", "from": "quote:1", "to": "person:ann"},
        ],
    }


class TestTranslatedFrom:
    def test_reads_the_source_language_of_a_translated_node(self) -> None:
        assert translated_from({"translation": _provenance("es")}) == "es"

    @pytest.mark.parametrize(
        "props",
        [
            {},
            {"translation": None},
            {"translation": {"translated": False, "source_language": "es"}},
            {"translation": {"translated": True, "source_language": ""}},
            {"translation": {"translated": True, "source_language": "en"}},
        ],
    )
    def test_anything_else_is_not_translated(self, props: dict[str, Any]) -> None:
        assert translated_from(props) is None
