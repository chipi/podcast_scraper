"""Unit tests for ranking-experiment telemetry (``app_ranking_telemetry``)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import app_ranking_telemetry as tel


def test_assign_variant_is_stable_per_user() -> None:
    variants = ["a", "b", "c"]
    v1 = tel.assign_variant("user-1", variants)
    assert v1 in variants
    assert tel.assign_variant("user-1", variants) == v1  # stable across calls


def test_assign_variant_empty_is_default() -> None:
    assert tel.assign_variant("user-1", []) == "default"
    assert tel.assign_variant("user-1", ["", " "]) == "default"


def test_assign_variant_distributes_across_users() -> None:
    variants = ["a", "b"]
    seen = {tel.assign_variant(f"user-{i}", variants) for i in range(50)}
    assert seen == {"a", "b"}  # a stable hash still hits both buckets over many users


def test_records_and_reads_impressions_and_clicks(tmp_path: Path) -> None:
    tel.record_impressions(tmp_path, "u1", shown=["s1", "s2", "s3"], variant="personalized", ts=100)
    tel.record_click(tmp_path, "u1", slug="s2", position=1, variant="personalized", ts=101)
    events = tel.read_events(tmp_path, "u1")
    assert len(events) == 2
    imp, click = events
    assert imp["kind"] == "impression"
    assert imp["shown"] == ["s1", "s2", "s3"]
    assert imp["variant"] == "personalized"
    assert click["kind"] == "click"
    assert click["slug"] == "s2"
    assert click["position"] == 1


def test_read_events_absent_is_empty(tmp_path: Path) -> None:
    assert tel.read_events(tmp_path, "nobody") == []


def test_read_events_skips_corrupt_lines(tmp_path: Path) -> None:
    tel.record_impressions(tmp_path, "u1", shown=["s1"], variant="recency", ts=1)
    path = tmp_path / "users" / "u1" / "ranking_events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write("\n")
    tel.record_click(tmp_path, "u1", slug="s1", position=0, variant="recency", ts=2)
    events = tel.read_events(tmp_path, "u1")
    assert [e["kind"] for e in events] == ["impression", "click"]  # corrupt/blank skipped
