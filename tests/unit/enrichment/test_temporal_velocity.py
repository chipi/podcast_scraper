"""Unit tests for temporal_velocity content-series tallying."""

from __future__ import annotations

from collections import defaultdict

from podcast_scraper.enrichment.enrichers.temporal_velocity import _tally_content_week


def _weekly() -> defaultdict:
    return defaultdict(lambda: defaultdict(int))


def test_tally_skips_unresolved_speaker_persons() -> None:
    """Anonymous diarization voices must not enter the trending person series (#1167)."""
    kg = {
        "nodes": [
            {"type": "Person", "id": "person:alice", "properties": {"name": "Alice"}},
            {"type": "Person", "id": "person:speaker-01", "properties": {"name": "SPEAKER_01"}},
            {"type": "Person", "id": "person:speaker-ep1-02", "properties": {"name": "Speaker 2"}},
        ]
    }
    weekly = _weekly()
    labels: dict[str, str] = {}
    _tally_content_week(kg, "Person", "2024-W01", weekly, labels)
    assert set(weekly) == {"person:alice"}
    assert "person:speaker-01" not in labels and "person:speaker-ep1-02" not in labels


def test_tally_keeps_topics_untouched() -> None:
    """The speaker guard is Person-only — Topic ids are never treated as placeholders."""
    kg = {
        "nodes": [
            {"type": "Topic", "id": "topic:speaker-training", "properties": {"label": "x"}},
        ]
    }
    weekly = _weekly()
    labels: dict[str, str] = {}
    _tally_content_week(kg, "Topic", "2024-W01", weekly, labels)
    assert set(weekly) == {"topic:speaker-training"}
