"""Listening-scope (``scope=mine``) membership for search hits.

A passage is "mine" when its episode is. A STORYLINE has no episode, so filtering it by
``episode_slug`` dropped every storyline out of recall mode — silently, as a consequence of
filtering an episode-less row by episode rather than a decision anyone made. That hid the result
type best suited to recall: a storyline is the shape of a recurring thread across episodes the
listener has actually heard (operator 2026-09-17).

Membership is ANY overlap: one episode of a thirty-six-episode cluster makes it yours.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.routes.app_search import _in_listening_scope, _storyline_slugs

pytestmark = [pytest.mark.unit]


def _write_episode(root: Path, stem: str, episode_id: str) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f1", "title": "Show", "url": "https://pod.example/f.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": "2026-01-01T00:00:00",
            "duration_seconds": 100,
        },
        "summary": {"title": "S", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")


def _corpus(root: Path) -> dict[str, str]:
    """Two episodes; returns ``{episode_id: slug}``."""
    _write_episode(root, "0001-a", "ep-one")
    _write_episode(root, "0002-b", "ep-two")
    return {row.episode_id: slug_for_row(row) for row in cached_catalog(root) if row.episode_id}


def _storyline_hit(episode_ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "doc_type": "storyline",
            "source_id": "thc:risk",
            "storyline_label": "Managing risk across domains",
            "storyline_episode_ids": episode_ids,
        }
    )


def test_storyline_is_mine_when_one_of_its_episodes_is_heard(tmp_path: Path) -> None:
    """ANY overlap counts — the operator's call, over a threshold.

    A threshold would make a storyline near-unreachable in the very scope where it is most useful:
    requiring most of a 36-episode cluster means almost no listener ever qualifies.
    """
    slugs = _corpus(tmp_path)
    hit = _storyline_hit(["ep-one"])
    assert _in_listening_scope(tmp_path, hit, {slugs["ep-one"]}) is True
    # Heard the OTHER episode only → this storyline draws on nothing the listener has heard.
    assert _in_listening_scope(tmp_path, hit, {slugs["ep-two"]}) is False


def test_storyline_spanning_many_episodes_needs_only_one_heard(tmp_path: Path) -> None:
    slugs = _corpus(tmp_path)
    hit = _storyline_hit(["ep-one", "ep-two", "ep-absent"])
    assert _in_listening_scope(tmp_path, hit, {slugs["ep-two"]}) is True


def test_storyline_with_no_recorded_episodes_is_not_mine(tmp_path: Path) -> None:
    """Empty membership never matches, rather than matching everything."""
    slugs = _corpus(tmp_path)
    assert _in_listening_scope(tmp_path, _storyline_hit([]), {slugs["ep-one"]}) is False
    hit = SimpleNamespace(metadata={"doc_type": "storyline", "source_id": "thc:risk"})
    assert _in_listening_scope(tmp_path, hit, {slugs["ep-one"]}) is False


def test_a_passage_still_filters_on_its_own_episode(tmp_path: Path) -> None:
    """The existing rule is untouched: a passage is in scope when ITS episode is."""
    slugs = _corpus(tmp_path)
    hit = SimpleNamespace(metadata={"doc_type": "transcript", "episode_slug": slugs["ep-one"]})
    assert _in_listening_scope(tmp_path, hit, {slugs["ep-one"]}) is True
    assert _in_listening_scope(tmp_path, hit, {slugs["ep-two"]}) is False


def test_storyline_slugs_resolves_episode_ids_through_the_catalogue(tmp_path: Path) -> None:
    """``mine`` is in SLUGS; the artifact records episode IDS. The catalogue bridges them."""
    slugs = _corpus(tmp_path)
    meta = {"storyline_episode_ids": ["ep-one", "ep-absent"]}
    assert _storyline_slugs(tmp_path, meta) == {slugs["ep-one"]}
