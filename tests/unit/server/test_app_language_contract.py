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

    def test_app_validation_stores_normalized_languages_and_serves_them(self) -> None:
        """The corpus was regenerated 2026-09-30: it stores `en`/`es` now, not the raw `en-us`,
        and it carries its first non-English episode (`p10`, Spanish).

        This test used to assert `{"en-us"}` stored and `{"en"}` served — the interesting part
        being that the two differed. They no longer do, because the generator normalises. So
        what is worth asserting moved: that the SERVED value is normalised whatever is stored,
        and that a non-English episode is served as its own language rather than flattened to
        English.
        """
        root = self.REPO / "tests/fixtures/app-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas, "fixture corpus missing"
        stored = {
            (json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("language")
            for m in metas
        }
        # es, it, fr, de, pt — five non-English counterparts of p01, each in its own language.
        assert stored >= {
            "en",
            "es",
            "it",
            "fr",
            "de",
            "pt",
        }, f"the fixture's premise changed: {stored}"

        rows = build_catalog_rows(root)
        assert rows, "catalog scan found nothing"
        # What is SERVED must equal what is STORED — that is the whole contract. Comparing to
        # `stored` rather than to a written-down set means adding a language cannot make these
        # pass for the wrong reason, and cannot fail for a reason that is only bookkeeping.
        assert {r.feed_language for r in rows} == stored
        assert {r.episode_language for r in rows} == stored

        feeds = aggregate_feeds(rows)
        assert {f["language"] for f in feeds} == stored

    def test_the_spanish_feed_is_served_as_SPANISH_not_flattened(self) -> None:
        """One episode in one language, so the aggregate must not average it away. Serving a
        Spanish show as English is how a listener gets a feed they cannot understand."""
        root = self.REPO / "tests/fixtures/app-validation-corpus/v3"
        rows = [r for r in build_catalog_rows(root) if r.feed_id == "p10"]
        assert rows, "p10 is not in the catalog"
        assert {r.feed_language for r in rows} == {"es"}
        assert {r.episode_language for r in rows} == {"es"}

    def test_viewer_validation_serves_the_language_it_now_carries(self) -> None:
        """Backfilled 2026-10-01 (#2185), so both fixture corpora now exercise the PRESENT case.

        This asserted the opposite — no `feed.language`, served as `None` — because the viewer
        corpus predated the generator writing it, and the S0.4 audit consequently reported all 40
        episodes as "not enabled in the registry". It was an honest record of a gap, not a contract.

        THE CONTRACT IT ALSO CARRIED IS NOT LOST. "Absent must not become `en`" is the valuable
        half, and it is covered directly by `TestNormalization.test_absent_is_none_not_a_default`
        over a constructed artifact — which is the better home for it anyway, because it does not
        depend on a fixture happening to lack a field. With both corpora backfilled there is no
        fixture left in the absent state, so asserting it here would mean keeping a corpus wrong on
        purpose to test one branch.
        """
        root = self.REPO / "tests/fixtures/viewer-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas, "fixture corpus missing"
        stored = {
            (json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("language")
            for m in metas
        }
        assert stored == {"en"}, f"the viewer corpus's stored languages are {stored}"

        rows = build_catalog_rows(root)
        assert rows, "catalog scan found nothing"
        # Served equals stored, the same contract the app-corpus test asserts.
        assert {r.feed_language for r in rows} == {"en"}
        assert {r.episode_language for r in rows} == {"en"}


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
