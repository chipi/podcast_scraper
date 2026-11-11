"""Search reads an insight's visibility from its own episode only, not the whole corpus.

Building the visible set for every episode before the first result cost a cold search every GI
file in the corpus (most of the cold digest's time on the 2026-11 prod copy). A membership test
now reads one episode's GI; iterating still yields the whole set, which Common's operators read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from podcast_scraper.search import corpus_search as cs

pytestmark = pytest.mark.unit


def _gi(path: Path, insight_ids: list[str]) -> Path:
    nodes = [{"id": i, "type": "Insight", "properties": {"text": i}} for i in insight_ids]
    path.write_text(json.dumps({"nodes": nodes, "edges": []}), encoding="utf-8")
    return path


@pytest.fixture
def two_episodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    # The episode view shows every insight whose id does not end in "-hidden".
    class _Shown:
        def __init__(self, i: str) -> None:
            self.id = i

    monkeypatch.setattr(
        "podcast_scraper.read_hooks.insights_from_gi",
        lambda artifact: [
            _Shown(n["id"]) for n in artifact["nodes"] if not n["id"].endswith("-hidden")
        ],
    )
    return {
        "e1": _gi(tmp_path / "e1.gi.json", ["i1", "i2-hidden"]),
        "e2": _gi(tmp_path / "e2.gi.json", ["i3"]),
    }


def test_a_membership_test_reads_only_that_episode(
    two_episodes: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[Path | None] = []
    real: Callable[[Path | None], frozenset] = cs._visible_in_episode

    def counting(path: Path | None) -> frozenset:
        read.append(path)
        return real(path)

    monkeypatch.setattr(cs, "_visible_in_episode", counting)
    visible = cs._VisibleInsights(two_episodes)
    assert ("e1", "i1") in visible
    assert ("e1", "i2-hidden") not in visible
    assert read == [two_episodes["e1"]], "only e1's GI is read, and only once"


def test_iterating_yields_every_visible_insight(two_episodes: dict[str, Path]) -> None:
    assert sorted(cs._VisibleInsights(two_episodes)) == [("e1", "i1"), ("e2", "i3")]


def test_an_unknown_or_unreadable_episode_shows_nothing(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gi.json"
    bad.write_text("{not json", encoding="utf-8")
    visible = cs._VisibleInsights({"bad": bad})
    assert ("bad", "x") not in visible
    assert ("missing", "x") not in visible
    assert "not-a-pair" not in visible
