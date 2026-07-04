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


def test_aggregate_empty() -> None:
    a = tel.aggregate([])
    assert a["total_events"] == 0
    assert a["by_action"] == {}
    assert a["size"]["samples"] == 0
    assert a["breakage"]["count"] == 0


def test_aggregate_usage_size_breakage() -> None:
    events = [
        {"action": "graph_node_tap", "kind": "topic"},
        {"action": "graph_node_tap", "kind": "topic"},
        {"action": "graph_node_tap", "kind": "person"},
        {"action": "graph_rail_nav", "trail_size": 3},
        {"action": "graph_redraw", "nodes": 10, "edges": 12, "trail_size": 2},
        {"action": "graph_redraw", "nodes": 30, "edges": 40, "trail_size": 5},
        {"action": "graph_broke", "reason": "stuck-timeout"},
        {"action": "graph_broke", "reason": "stuck-timeout"},
        {"no_action": 1},  # ignored (no action)
    ]
    a = tel.aggregate(events)
    assert a["total_events"] == 8
    assert a["by_action"]["graph_node_tap"] == 3
    assert a["node_taps_by_kind"] == {"topic": 2, "person": 1}
    assert a["size"]["samples"] == 2
    assert a["size"]["nodes"]["min"] == 10 and a["size"]["nodes"]["max"] == 30
    assert a["size"]["nodes"]["avg"] == 20.0
    assert a["size"]["edges"]["max"] == 40
    assert a["breakage"]["count"] == 2
    assert a["breakage"]["by_reason"] == {"stuck-timeout": 2}


def test_read_all_events_across_users(tmp_path: Path) -> None:
    tel.record_events(tmp_path, "u1", [{"action": "graph_node_tap"}])
    tel.record_events(tmp_path, "u2", [{"action": "graph_redraw", "nodes": 5}])
    tel.record_events(tmp_path, "anon", [{"action": "graph_broke", "reason": "x"}])
    allev = tel.read_all_events(tmp_path)
    assert len(allev) == 3
    assert set(tel.aggregate(allev)["by_action"]) == {
        "graph_node_tap",
        "graph_redraw",
        "graph_broke",
    }
