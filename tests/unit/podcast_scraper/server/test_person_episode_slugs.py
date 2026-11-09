"""``person_episode_slugs`` is the person card's episode list without projecting the cards.

The people rail's quotes built a whole person card per person, and each episode card read that
episode's GI for its moments flag: 3.2 s of a listener's first Mine rails on the prod-shaped corpus
(2026-11-08). These tests pin that the slugs come out exactly as the card lists them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from podcast_scraper.server import relational_view
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.kg_index import get_kg_index
from podcast_scraper.server.schemas import AppEntity
from podcast_scraper.server.slugs import slug_for_row

FIXTURE = Path("tests/fixtures/app-validation-corpus/v3")


def _row(episode_id: str, publish_date: str | None) -> CatalogEpisodeRow:
    return CatalogEpisodeRow(
        metadata_relative_path=f"feeds/f/{episode_id}.metadata.json",
        feed_id="f",
        feed_title="F",
        episode_id=episode_id,
        episode_title=episode_id,
        publish_date=publish_date,
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="",
        kg_relative_path="",
        bridge_relative_path="",
        has_gi=False,
        has_kg=False,
        has_bridge=False,
    )


def test_newest_first_undated_last_ties_in_catalog_order(monkeypatch: pytest.MonkeyPatch) -> None:
    me = AppEntity(id="person:a", name="A", kind="person")
    other = AppEntity(id="person:b", name="B", kind="person")
    rows = [
        _row("old", "2024-01-01"),
        _row("undated", None),
        _row("new", "2026-05-01"),
        _row("tie-1", "2025-03-03"),
        _row("not-them", "2026-09-09"),
        _row("tie-2", "2025-03-03"),
    ]
    matched: list[tuple[CatalogEpisodeRow, list[AppEntity], list[object]]] = [
        (r, [other] if r.episode_id == "not-them" else [me, other], []) for r in rows
    ]
    monkeypatch.setattr(relational_view, "_person_episodes", lambda *_a: matched)

    got = relational_view.person_episode_slugs(Path("/corpus"), "person:a")

    by_id = {r.episode_id: slug_for_row(r) for r in rows}
    assert got == [by_id[i] for i in ("new", "tie-1", "tie-2", "old", "undated")]


def test_same_slugs_in_the_same_order_as_the_person_card() -> None:
    if not FIXTURE.is_dir():
        pytest.fail(f"fixture corpus missing: {FIXTURE}")
    root = Path(os.path.abspath(FIXTURE))
    people = sorted(get_kg_index(root).person_to_eps)
    assert people, "the fixture has no people, so the comparison proves nothing"
    several = 0
    for pid in people:
        card = relational_view.build_person_card(root, pid)
        want = [e.slug for e in card.episodes] if card else []
        several += len(want) >= 2
        assert relational_view.person_episode_slugs(root, pid) == want, pid
    assert several, "no person has two episodes, so order is not compared"


def test_unknown_person_has_no_episodes() -> None:
    root = Path(os.path.abspath(FIXTURE))
    assert relational_view.person_episode_slugs(root, "person:no-such-person") == []
