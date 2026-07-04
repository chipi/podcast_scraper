"""Unit tests for graph-analytics telemetry (``app_graph_telemetry``)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import app_graph_telemetry as tel


def test_records_and_reads_events(tmp_path: Path) -> None:
    n = tel.record_events(
        tmp_path,
        "u1",
        [
            {"action": "node_tap", "id": "topic:a", "ts": 1},
            {"action": "redraw", "nodes": 42, "edges": 60, "ts": 2},
        ],
    )
    assert n == 2
    events = tel.read_events(tmp_path, "u1")
    assert [e["action"] for e in events] == ["node_tap", "redraw"]
    assert events[1]["nodes"] == 42


def test_skips_events_without_action(tmp_path: Path) -> None:
    n = tel.record_events(
        tmp_path,
        "u1",
        [
            {"action": "node_tap"},
            {"no_action": True},  # dropped
            "not a dict",  # dropped
            {"action": ""},  # empty action dropped
        ],
    )
    assert n == 1
    assert len(tel.read_events(tmp_path, "u1")) == 1


def test_record_empty_is_noop(tmp_path: Path) -> None:
    assert tel.record_events(tmp_path, "u1", []) == 0
    assert tel.read_events(tmp_path, "u1") == []


def test_read_absent_is_empty(tmp_path: Path) -> None:
    assert tel.read_events(tmp_path, "nobody") == []


def test_read_skips_corrupt_lines(tmp_path: Path) -> None:
    tel.record_events(tmp_path, "u1", [{"action": "a"}])
    path = tmp_path / "users" / "u1" / "graph_events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n\n")
    tel.record_events(tmp_path, "u1", [{"action": "b"}])
    assert [e["action"] for e in tel.read_events(tmp_path, "u1")] == ["a", "b"]
