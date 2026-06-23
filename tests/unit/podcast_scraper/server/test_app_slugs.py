"""Unit tests for the consumer episode slug contract (#1067, RFC-098 §4).

Pure functions over plain strings — no HTTP, no disk, no `[dev]` extras.
"""

from __future__ import annotations

import re

from podcast_scraper.server.app_slugs import episode_slug

_SLUG_RE = re.compile(r"^[a-z0-9-]+$")


class TestEpisodeSlug:
    def test_is_url_safe_and_deterministic(self) -> None:
        s1 = episode_slug("My Feed", "ep-1", "metadata/x.metadata.json")
        s2 = episode_slug("My Feed", "ep-1", "metadata/x.metadata.json")
        assert s1 == s2
        assert _SLUG_RE.match(s1), s1
        assert s1.startswith("my-feed-")

    def test_stable_across_metadata_path_when_episode_id_present(self) -> None:
        # Title/run/path must not affect the slug — only (feed_id, episode_id).
        a = episode_slug("feed", "ep1", "metadata/0001-old.metadata.json")
        b = episode_slug("feed", "ep1", "feeds/x/metadata/0009-new_run2.metadata.json")
        assert a == b

    def test_distinguishes_episodes_and_feeds(self) -> None:
        base = episode_slug("feed", "ep1", "p")
        assert episode_slug("feed", "ep2", "p") != base
        assert episode_slug("other", "ep1", "p") != base

    def test_fallback_without_episode_id_uses_metadata_path(self) -> None:
        a = episode_slug("feed", None, "metadata/one.metadata.json")
        b = episode_slug("feed", None, "metadata/two.metadata.json")
        assert a != b
        # deterministic
        assert a == episode_slug("feed", None, "metadata/one.metadata.json")
        assert _SLUG_RE.match(a)

    def test_empty_feed_id_still_url_safe(self) -> None:
        s = episode_slug("", "ep1", "p")
        assert s.startswith("feed-")
        assert _SLUG_RE.match(s)

    def test_unicode_feed_id_normalised(self) -> None:
        s = episode_slug("Café Crème", "ep1", "p")
        assert _SLUG_RE.match(s), s
