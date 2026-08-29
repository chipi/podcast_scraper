"""Unit tests for structured feeds spec (#626)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from podcast_scraper import config as cfg_mod
from podcast_scraper.rss.feeds_spec import (
    FeedsSpecDocument,
    load_feeds_spec_file,
    merge_feed_entry_into_config,
    RSS_FEED_ENTRY_OVERRIDE_KEYS,
    RssFeedEntry,
)


def test_load_feeds_spec_yaml_and_json_equivalent(tmp_path: Path) -> None:
    y = tmp_path / "f.yaml"
    j = tmp_path / "f.json"
    data = {
        "feeds": [
            "https://a.example/feed.xml",
            {"url": "https://b.example/feed.xml", "timeout": 99},
        ]
    }
    y.write_text(yaml.safe_dump(data), encoding="utf-8")
    j.write_text(json.dumps(data), encoding="utf-8")
    dy = load_feeds_spec_file(y)
    dj = load_feeds_spec_file(j)
    assert dy.model_dump() == dj.model_dump()
    assert dy.feeds[1].timeout == 99


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        yaml.safe_dump({"feeds": ["https://a.example/x"], "extra_root": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level"):
        load_feeds_spec_file(p)


def test_feed_entry_unknown_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        RssFeedEntry.model_validate({"url": "https://a.example/x", "not_allowed": 1})


def test_merge_feed_entry_overrides_global_timeout() -> None:
    base = cfg_mod.Config(
        rss="https://ignored.example/x",
        timeout=30,
        user_agent="global-ua",
    )
    ent = RssFeedEntry(url="https://feed.example/rss", timeout=77, user_agent=None)
    merged = merge_feed_entry_into_config(base, ent)
    assert merged.rss_url == "https://feed.example/rss"
    assert merged.timeout == 77
    assert merged.user_agent == "global-ua"
    assert merged.rss_urls is None


def test_merge_feed_entry_overrides_known_hosts_per_feed() -> None:
    # Step B: a network feed can name its recurring hosts that never self-introduce.
    base = cfg_mod.Config(rss="https://ignored.example/x", known_hosts=["Global Host"])
    ent = RssFeedEntry(
        url="https://feed.example/rss", known_hosts=["Erika Barris", "Nick Fountain"]
    )
    merged = merge_feed_entry_into_config(base, ent)
    assert merged.known_hosts == ["Erika Barris", "Nick Fountain"]  # feed overrides global
    # A feed that omits known_hosts inherits the global list.
    inherit = merge_feed_entry_into_config(base, RssFeedEntry(url="https://feed.example/rss"))
    assert inherit.known_hosts == ["Global Host"]


def test_merge_feed_entry_show_centric_per_feed() -> None:
    # A news-desk feed marks itself show-centric so an unnamed "Host" is expected, not a failure.
    base = cfg_mod.Config(rss="https://ignored.example/x")
    assert base.show_centric is False
    merged = merge_feed_entry_into_config(
        base, RssFeedEntry(url="https://feed.example/rss", show_centric=True)
    )
    assert merged.show_centric is True
    inherit = merge_feed_entry_into_config(base, RssFeedEntry(url="https://feed.example/rss"))
    assert inherit.show_centric is False


def test_merge_feed_entry_diarization_min_segment_ms_per_feed() -> None:
    # A news-desk feed with no real cameos can squelch phantom micro-speakers harder (#1170).
    base = cfg_mod.Config(rss="https://ignored.example/x")
    merged = merge_feed_entry_into_config(
        base, RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=1500)
    )
    assert merged.diarization_min_segment_ms == 1500
    # omitted -> inherits the base (global) value
    inherit = merge_feed_entry_into_config(base, RssFeedEntry(url="https://feed.example/rss"))
    assert inherit.diarization_min_segment_ms == base.diarization_min_segment_ms
    # in the override-key allowlist so the merge propagates it
    assert "diarization_min_segment_ms" in RSS_FEED_ENTRY_OVERRIDE_KEYS


def test_feed_entry_diarization_min_segment_ms_range_validated() -> None:
    # Field validator: 0 <= ms <= 60000.
    RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=0)
    RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=60000)
    with pytest.raises(ValidationError):
        RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=60001)
    with pytest.raises(ValidationError):
        RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=-1)


def test_feeds_spec_document_accepts_comment_keys() -> None:
    doc = FeedsSpecDocument.model_validate(
        {"_comment": "x", "_comment_resilience": "y", "feeds": ["https://a.example/z"]}
    )
    assert len(doc.feeds) == 1


@pytest.mark.unit
def test_repo_example_feeds_specs_load() -> None:
    """Tracked ``config/examples/feeds.spec.example.*`` stay valid feeds-spec documents."""
    root = Path(__file__).resolve().parents[4]
    examples_dir = root / "config" / "examples"
    for name in ("feeds.spec.example.yaml", "feeds.spec.example.json"):
        p = examples_dir / name
        if not p.is_file():
            pytest.skip(f"{p} not present")
        doc = load_feeds_spec_file(p)
        assert doc.feeds, f"{p.name}: expected at least one feed"


class TestPerFeedProfileOverride:
    """A feed may name its own deployment profile (2026-08-28, DGX onboarding).

    A batch run applies ONE profile to every feed, so onboarding feeds onto the DGX while the
    proven feeds stay on Deepgram needs per-feed routing. The override must RESOLVE the
    profile: ``model_copy(update=...)`` skips validators, so assigning the name alone would
    relabel the config and route nothing — the failure mode these tests exist to forbid.
    """

    @pytest.fixture()
    def base(self, monkeypatch):
        """cloud_balanced batch config. The key is monkeypatched, NOT os.environ.setdefault —
        that leaks into the process and breaks the tests asserting a MISSING deepgram key."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
        from podcast_scraper import config as config_mod

        return config_mod.Config(
            rss_url="https://placeholder.example/f", profile="cloud_balanced", max_episodes=10
        )

    def test_named_profile_actually_changes_routing(self, base):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        assert base.transcription_provider == "deepgram"  # batch default
        sub = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://new.example/f", profile="cloud_with_dgx_primary")
        )
        assert sub.profile == "cloud_with_dgx_primary"
        assert sub.transcription_provider == "tailnet_dgx_whisper", (
            "the per-feed profile relabelled the config without re-routing — the exact "
            "model_copy-skips-validators no-op this feature must not have"
        )

    def test_feed_without_profile_is_untouched(self, base):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        sub = merge_feed_entry_into_config(base, RssFeedEntry(url="https://old.example/f"))
        assert sub.profile == "cloud_balanced"
        assert sub.transcription_provider == "deepgram"

    def test_entry_own_overrides_win_over_the_named_profile(self, base):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        sub = merge_feed_entry_into_config(
            base,
            RssFeedEntry(
                url="https://new.example/f", profile="cloud_with_dgx_primary", max_episodes=3
            ),
        )
        assert sub.transcription_provider == "tailnet_dgx_whisper"
        assert sub.max_episodes == 3

    def test_unknown_profile_warns_and_keeps_the_batch_route(self, base, caplog):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        with caplog.at_level("WARNING"):
            sub = merge_feed_entry_into_config(
                base, RssFeedEntry(url="https://x.example/f", profile="no-such-profile")
            )
        assert sub.transcription_provider == "deepgram"
        assert "matched no registry preset" in caplog.text
