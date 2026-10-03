"""A KG reply that is not valid JSON must say WHERE it broke, at WARNING."""

from __future__ import annotations

import json
import logging

import pytest

from podcast_scraper.kg.llm_extract import parse_kg_graph_response

pytestmark = [pytest.mark.unit]

_LOGGER = "podcast_scraper.kg.llm_extract"


def _reply_with_unescaped_quote() -> str:
    good = {"label": "focus", "description": "Trainable attention. " * 6}
    bad = '{"label": "dopamine", "description": "The "spotlight" of focus."}'
    return (
        '{\n  "topics": [\n    '
        + json.dumps(good)
        + ",\n    "
        + bad
        + '\n  ],\n  "entities": []\n}'
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_decode_failure_logs_position_and_context(caplog: pytest.LogCaptureFixture) -> None:
    raw = _reply_with_unescaped_quote()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert parse_kg_graph_response(raw) is None
    msgs = [m for m in _warnings(caplog) if "not valid JSON" in m]
    assert len(msgs) == 1
    msg = msgs[0]
    err = None
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        err = e
    assert err is not None
    assert f"pos={err.pos}" in msg
    assert f"line={err.lineno}" in msg
    assert f"col={err.colno}" in msg
    assert f"reply {len(raw)} chars" in msg
    assert err.msg in msg
    assert "spotlight" in msg
    assert len(msg) < len(raw) + 400


def test_context_is_a_snippet_not_the_whole_reply(caplog: pytest.LogCaptureFixture) -> None:
    raw = _reply_with_unescaped_quote().replace("Trainable", "x" * 3000)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        parse_kg_graph_response(raw)
    msg = [m for m in _warnings(caplog) if "not valid JSON" in m][0]
    assert len(msg) < 800


def test_non_dict_top_level_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert parse_kg_graph_response('[{"label": "x"}]') is None
    assert any("not a JSON object" in m for m in _warnings(caplog))
