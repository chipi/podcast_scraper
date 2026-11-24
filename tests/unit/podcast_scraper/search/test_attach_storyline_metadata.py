"""``_attach_storyline_metadata`` builds the storyline summaries only when a hit needs them."""

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.search import corpus_search

pytestmark = [pytest.mark.unit]

SUMMARIES = [
    {"id": "thc:a", "label": "Alpha", "size": 3, "anchor_topic_id": "topic:alpha"},
    {"id": "thc:b", "label": "Beta", "size": 2, "anchor_topic_id": "topic:beta"},
]
EPISODES = {"thc:a": {"e2", "e1"}, "thc:b": {"e3"}}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def summaries(root: Path, limit: int, min_members: int = 1) -> list[dict[str, Any]]:
        seen.append("summaries")
        return [dict(s) for s in SUMMARIES]

    def episodes(root: Path) -> dict[str, set[str]]:
        seen.append("episodes")
        return EPISODES

    monkeypatch.setattr(corpus_search, "top_storylines_by_member_count", summaries)
    monkeypatch.setattr(corpus_search, "storyline_episode_ids", episodes)
    return seen


def _row(doc_type: str, source_id: str) -> dict[str, Any]:
    return {"id": source_id, "metadata": {"doc_type": doc_type, "source_id": source_id}}


def test_no_storyline_hit_builds_no_summaries(calls: list[str]) -> None:
    rows = [_row("insight", "i1"), _row("kg_topic", "t1")]
    out = corpus_search._attach_storyline_metadata(rows, Path("/c"))
    assert out == rows
    assert calls == []


def test_storyline_hits_are_joined_and_orphans_dropped(calls: list[str]) -> None:
    rows = [
        _row("insight", "i1"),
        _row("storyline", "thc:b"),
        _row("storyline", "thc:gone"),
        _row("storyline", "thc:a"),
    ]
    out = corpus_search._attach_storyline_metadata(rows, Path("/c"))
    assert [r["id"] for r in out] == ["i1", "thc:b", "thc:a"]
    assert out[1]["metadata"] == {
        "doc_type": "storyline",
        "source_id": "thc:b",
        "storyline_label": "Beta",
        "storyline_size": 2,
        "anchor_topic_id": "topic:beta",
        "storyline_episode_ids": ["e3"],
    }
    assert out[2]["metadata"]["storyline_episode_ids"] == ["e1", "e2"]
    assert out[2]["metadata"]["storyline_label"] == "Alpha"
    assert calls == ["summaries", "episodes"]
