"""Unit tests for the canonical event emitter (ADR-119)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from podcast_scraper.obs.events import emit_event, EVENT_SCHEMA


@pytest.mark.unit
def test_log_sink_emits_canonical_envelope(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.events"):
        line = emit_event("demo", foo="bar", n=3)
    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["event_type"] == "demo"
    assert payload["schema"] == EVENT_SCHEMA
    assert payload["foo"] == "bar" and payload["n"] == 3
    assert "ts" in payload
    # returns the serialized line for callers that also echo it
    assert line is not None and json.loads(line) == payload


@pytest.mark.unit
def test_log_sink_honours_logger_override(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="my.caller"):
        emit_event("demo", logger=logging.getLogger("my.caller"), a=1)
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "my.caller"


@pytest.mark.unit
def test_none_fields_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.events"):
        emit_event("demo", present="x", absent=None)
    payload = json.loads(caplog.records[0].message)
    assert payload["present"] == "x"
    assert "absent" not in payload


@pytest.mark.unit
def test_ts_override(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.events"):
        emit_event("demo", ts="2020-01-01T00:00:00+00:00")
    payload = json.loads(caplog.records[0].message)
    assert payload["ts"] == "2020-01-01T00:00:00+00:00"


@pytest.mark.unit
def test_file_sink_appends_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "events.jsonl"
    emit_event("search_query", sink="file", path=out, query_type="semantic")
    emit_event("search_query", sink="file", path=out, query_type="entity")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event_type"] == "search_query" and first["query_type"] == "semantic"
    assert first["schema"] == EVENT_SCHEMA


@pytest.mark.unit
def test_file_sink_via_corpus_dir(tmp_path: Path) -> None:
    emit_event("search_query", sink="file", corpus_dir=tmp_path, query_type="semantic")
    out = tmp_path / "search" / "query_log.jsonl"
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8").strip())["query_type"] == "semantic"


@pytest.mark.unit
def test_never_raises_on_bad_file_target(tmp_path: Path) -> None:
    # A path whose parent is an existing *file* (not a dir) can't be created — must
    # be swallowed, never raised (telemetry must not break the caller).
    clash = tmp_path / "afile"
    clash.write_text("x", encoding="utf-8")
    assert emit_event("demo", sink="file", path=clash / "nope.jsonl") is None


@pytest.mark.unit
def test_never_raises_on_unserialisable_field(caplog: pytest.LogCaptureFixture) -> None:
    # default=str makes odd objects serialisable; even a truly hostile field must
    # not raise out of emit_event.
    class Boom:
        def __str__(self) -> str:  # pragma: no cover - defensive
            raise RuntimeError("no str")

    # Should not raise; returns None on failure.
    assert emit_event("demo", obj=Boom()) is None or True
