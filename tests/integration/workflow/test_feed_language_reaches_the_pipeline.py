"""The RSS `<language>` tag has to reach the PIPELINE, not just the parser (#2172 / S0.1a).

WHY THIS FILE EXISTS. Phase 0's headline claim is "language is an asserted, checked fact, per
episode". It was not, on the live path, and nothing caught it for a whole arc:

- `fetch_and_parse_rss` parses the tag, but the pipeline does not call that function — it builds
  its own `RssFeed` in `stages/scraping.py` and the field was never passed. So `feed.language`
  was always `None` and every new episode recorded `language_source: profile_default`, which is
  the "measured the configuration, not the corpus" signature §3.1 warns about.
- `transcription_language(cfg)` — THE one reader (S0.6) — took no feed language, on the strength
  of a comment claiming the tag "arrives via the per-feed Config copy". Nothing put it there.

Measured before the fix: a feed declaring `es-ES` gave `transcription_language(cfg) == "en"`
while the metadata writer gave `es / rss` — two answers to one question — and a `de`-tagged feed
produced NO skip reason, so it would have been transcribed as English by a chain that can fall
back to `base`.

The existing S0.8 tests could not see any of this, because every one of them sets the language
via `Config(language=...)` and the one threading test uses a `MagicMock` with `.language` set by
hand. So the pipeline never had to carry the tag for anything to go green. These tests make it
carry it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.languages import resolve_config_language, transcription_language
from podcast_scraper.models import RssFeed
from podcast_scraper.rss import feed_cache
from podcast_scraper.rss.parser import _channel_language
from podcast_scraper.workflow.episode_processor import _unsupported_language_skip_reason
from podcast_scraper.workflow.metadata_generation import _build_feed_metadata
from podcast_scraper.workflow.stages.scraping import fetch_and_parse_feed

pytestmark = pytest.mark.integration

_SPANISH_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "rss" / "p10_spanish.xml"


def _cfg(**kw: object) -> config.Config:
    return config.Config(rss="https://example.com/feed.xml", **kw)  # type: ignore[arg-type]


class TestTheTagIsParsedFromTheRealFixture:
    def test_the_spanish_fixture_declares_es_ES(self) -> None:
        """The fixture exists precisely to make this case reachable (S0.9)."""
        assert _SPANISH_FIXTURE.is_file()
        assert _channel_language(_SPANISH_FIXTURE.read_bytes()) == "es-ES"

    def test_THE_PIPELINE_FUNCTION_carries_the_tag_onto_its_feed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The function the pipeline actually calls — not `fetch_and_parse_rss`.

        This is the whole bug in one assertion. `fetch_and_parse_rss` has parsed the channel tag
        since S0.1, and its tests were green the entire time; the pipeline calls
        `stages.scraping.fetch_and_parse_feed`, which builds a SECOND `RssFeed` and did not pass
        the field. Driven through the on-disk RSS cache so it reaches the real parse path with no
        network and no mock feed — a `MagicMock` with `.language` set by hand is how every
        existing S0.8 test avoided needing the pipeline to carry anything.
        """
        monkeypatch.setenv(feed_cache.ENV_RSS_CACHE_DIR, str(tmp_path))
        url = "https://example.com/p10_spanish.xml"
        feed_cache.write_cached_rss(url, _SPANISH_FIXTURE.read_bytes())

        feed, _rss_bytes = fetch_and_parse_feed(_cfg().model_copy(update={"rss_url": url}))

        assert feed.language == "es-ES", "the publisher's tag, raw, on the pipeline's own feed"

    def test_and_the_description_beside_it_is_still_carried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pinned in the same place because it is the same smell-audit finding (F6) one field
        up, and it regressed once already by the same mechanism."""
        monkeypatch.setenv(feed_cache.ENV_RSS_CACHE_DIR, str(tmp_path))
        url = "https://example.com/p10_spanish.xml"
        feed_cache.write_cached_rss(url, _SPANISH_FIXTURE.read_bytes())

        feed, _rss_bytes = fetch_and_parse_feed(_cfg().model_copy(update={"rss_url": url}))

        assert feed.description


class TestTheTranscriberAndTheMetadataWriterAGREE:
    def test_both_answer_the_feeds_declared_language(self) -> None:
        """The split this closes: same cfg, same feed, one answer."""
        feed = RssFeed(
            title="Sesiones de Sendero",
            authors=[],
            items=[],
            base_url="https://example.com/",
            language="es-ES",
        )
        cfg = _cfg().model_copy(update={"feed_declared_language": feed.language})

        assert transcription_language(cfg) == "es"
        meta = _build_feed_metadata(
            feed, "https://example.com/feed.xml", "fid", cfg, None, None, None, None
        )
        assert meta.language == "es"
        assert meta.language_source == "rss", "measured from the feed, not defaulted"
        assert transcription_language(cfg) == meta.language

    def test_language_source_is_rss_and_not_profile_default(self) -> None:
        """`profile_default` on a tagged feed is the signature of measuring the configuration
        instead of the corpus — the thing the audit exists to detect."""
        cfg = _cfg().model_copy(update={"feed_declared_language": "es-ES"})
        _raw, language, source = resolve_config_language(cfg)
        assert (language, source) == ("es", "rss")


class TestTheSkipGateFiresOnADeclaredLanguage:
    def test_a_de_tagged_feed_is_REFUSED(self) -> None:
        """The arc's §3.1 acceptance, which was unmeetable before: "a `de`-tagged feed is
        skipped, with a reason". `de` is declared in the registry but not enabled, and the
        alternative to refusing is transcribing German with an English-only chain."""
        cfg = _cfg().model_copy(update={"feed_declared_language": "de"})
        reason = _unsupported_language_skip_reason(cfg)
        assert reason is not None
        assert "'de'" in reason
        assert "not enabled" in reason

    def test_the_reason_names_the_FEED_as_the_source(self) -> None:
        """It used to say "the profile default" for a feed-declared language, pointing an
        operator at the wrong thing to change."""
        cfg = _cfg().model_copy(update={"feed_declared_language": "de"})
        reason = _unsupported_language_skip_reason(cfg) or ""
        assert "feed's declared <language> tag" in reason

    def test_an_english_tagged_feed_is_not_refused(self) -> None:
        cfg = _cfg().model_copy(update={"feed_declared_language": "en-US"})
        assert _unsupported_language_skip_reason(cfg) is None

    def test_an_untagged_feed_still_proceeds(self) -> None:
        """Most of the corpus declares nothing; refusing those would stop ingesting the English
        corpus that works today."""
        assert _unsupported_language_skip_reason(_cfg()) is None


class TestPrecedence:
    def test_the_override_beats_the_feed_tag(self) -> None:
        """Publisher tags are routinely wrong, so the override is the remedy — and it has to win
        at the transcriber too, not only in the metadata."""
        cfg = _cfg().model_copy(
            update={"feed_declared_language": "es-ES", "language_override": "en"}
        )
        assert transcription_language(cfg) == "en"
        assert resolve_config_language(cfg)[2] == "override"

    def test_the_feed_tag_beats_the_profile_default(self) -> None:
        cfg = _cfg(language="en").model_copy(update={"feed_declared_language": "es-ES"})
        assert transcription_language(cfg) == "es"

    def test_an_explicit_feed_language_argument_still_wins(self) -> None:
        """The seam passes the feed object's language directly; that must not be overridden by
        the cfg fallback."""
        cfg = _cfg().model_copy(update={"feed_declared_language": "de"})
        assert resolve_config_language(cfg, feed_language="es")[1] == "es"


class TestItSurvivesTheSubConfigRoundTrip:
    """`transcription/factory.py` and `providers/ml/diarization/factory.py` build each chain tier
    as `Config.model_validate(cfg.model_dump())`.

    That round trip runs AFTER `run_pipeline` has set the feed language, so a field that did not
    survive it would give the FALLBACK tier a config answering from the profile default — i.e.
    the primary transcribes Spanish and the fallback transcribes the same audio as English, on a
    chain whose whole job is to be invisible. Nothing downstream would report a disagreement.

    It also pins why no validator forbids an operator from setting the field: this very round trip
    would trip it, on exactly the non-English episodes the field exists to serve.
    """

    def test_a_chain_tier_sees_the_same_language(self) -> None:
        cfg = _cfg().model_copy(update={"feed_declared_language": "es-ES"})
        tier = config.Config.model_validate(cfg.model_dump())

        assert tier.feed_declared_language == "es-ES"
        assert transcription_language(tier) == "es"
        assert resolve_config_language(tier) == ("es-ES", "es", "rss")

    def test_and_an_override_survives_it_too(self) -> None:
        cfg = _cfg().model_copy(
            update={"feed_declared_language": "es-ES", "language_override": "pt"}
        )
        tier = config.Config.model_validate(cfg.model_dump())

        assert transcription_language(tier) == "pt"
        assert resolve_config_language(tier)[2] == "override"
