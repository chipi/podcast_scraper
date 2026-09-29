"""Language on the app API: present, absent, and pre-migration (#2176 / slice S0.5).

Both pre-migration states are real and already in the tree, which is why they are tested rather
than imagined:

* ``app-validation-corpus/v3`` — ``feed.language: "en-us"`` on all 40 episodes, no episode pair
* ``viewer-validation-corpus/v3`` — no ``feed.language`` key at all, on all 40

The first is a VALUE CHANGE for existing clients (``en-us`` -> ``en``); the second is the
absent case. Neither is hypothetical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from podcast_scraper.server.corpus_catalog import (
    _episode_language,
    _feed_language,
    aggregate_feeds,
    build_catalog_rows,
)

pytestmark = pytest.mark.unit


def _doc(
    *,
    feed_language: Optional[str] = None,
    episode_language: Optional[str] = None,
) -> Dict[str, Any]:
    feed: Dict[str, Any] = {"feed_id": "p01", "title": "Show", "url": "https://e.com/f.xml"}
    episode: Dict[str, Any] = {"episode_id": "p01_e01", "title": "Ep"}
    if feed_language is not None:
        feed["language"] = feed_language
    if episode_language is not None:
        episode["language"] = episode_language
    return {"feed": feed, "episode": episode}


class TestNormalization:
    def test_the_stored_regional_tag_is_served_normalized(self) -> None:
        """The value change. ``app-validation-corpus/v3`` stores ``en-us`` on all 40 episodes,
        and that is what the API served before this slice."""
        assert _feed_language(_doc(feed_language="en-us")) == "en"
        assert _feed_language(_doc(feed_language="en-US")) == "en"
        assert _feed_language(_doc(feed_language="pt_BR")) == "pt"

    def test_absent_is_none_not_a_default(self) -> None:
        """The viewer corpus's state: no ``feed.language`` key at all on any episode.

        Absent must stay absent — inventing ``"en"`` here is the assumption Phase 0 removes.
        """
        assert _feed_language(_doc()) is None
        assert _episode_language(_doc()) is None

    def test_an_unusable_tag_is_none(self) -> None:
        assert _feed_language(_doc(feed_language="123")) is None
        assert _feed_language(_doc(feed_language="   ")) is None


class TestEpisodeOverFeed:
    def test_the_episode_language_wins(self) -> None:
        doc = _doc(feed_language="en-US", episode_language="es-ES")
        assert _episode_language(doc) == "es"

    def test_it_falls_back_to_the_feed_for_pre_2172_artifacts(self) -> None:
        """No episode pair exists on any artifact written before #2172 — the majority of the
        corpus — so the fallback is the normal path, not an edge case."""
        doc = _doc(feed_language="en-US")
        assert "language" not in doc["episode"]
        assert _episode_language(doc) == "en"


class TestOverTheRealFixtureCorpora:
    """The two states, read off disk rather than constructed."""

    REPO = Path(__file__).resolve().parents[3]

    def test_app_validation_carries_en_us_and_is_served_as_en(self) -> None:
        root = self.REPO / "tests/fixtures/app-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas, "fixture corpus missing"
        stored = {
            (json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("language")
            for m in metas
        }
        assert stored == {"en-us"}, f"the fixture's premise changed: {stored}"

        rows = build_catalog_rows(root)
        assert rows, "catalog scan found nothing"
        assert {r.feed_language for r in rows} == {"en"}
        assert {r.episode_language for r in rows} == {"en"}

        feeds = aggregate_feeds(rows)
        assert {f["language"] for f in feeds} == {"en"}

    def test_viewer_validation_has_no_language_and_stays_none(self) -> None:
        root = self.REPO / "tests/fixtures/viewer-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas, "fixture corpus missing"
        assert all(
            "language" not in (json.loads(m.read_text(encoding="utf-8")).get("feed") or {})
            for m in metas
        ), "the fixture's premise changed: it now carries a feed language"

        rows = build_catalog_rows(root)
        assert rows, "catalog scan found nothing"
        assert {r.feed_language for r in rows} == {None}
        assert {r.episode_language for r in rows} == {None}, "absent must not become 'en'"


class TestTheResponseModelsCarryIt:
    def test_every_surface_declares_a_language_field(self) -> None:
        """Three surfaces, because the show page, the episode card and the operator viewer's
        shows library each read a different one."""
        from podcast_scraper.server.schemas import (
            AppEpisodeDetail,
            AppEpisodeSummary,
            AppPodcastItem,
            CorpusFeedItem,
        )

        for model in (AppEpisodeDetail, AppEpisodeSummary, AppPodcastItem, CorpusFeedItem):
            assert "language" in model.model_fields, model.__name__
            assert model.model_fields["language"].default is None, (
                f"{model.__name__}.language must default to None — additive for every existing "
                "client, and absent is a real state"
            )
