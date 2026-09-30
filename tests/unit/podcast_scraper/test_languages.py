"""Language tag normalization and feed-level resolution (#2172 / slice S0.1a)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper.languages import (
    normalize_language_tag,
    resolve_language,
    SOURCE_PROFILE_DEFAULT,
    SOURCE_RSS,
)

pytestmark = pytest.mark.unit


class TestNormalizeLanguageTag:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("en", "en"),
            ("en-US", "en"),
            ("en_US", "en"),
            ("EN-us", "en"),
            ("es-ES", "es"),
            ("pt_BR", "pt"),
            ("  de-DE  ", "de"),
            ("zh-Hans-CN", "zh"),
        ],
    )
    def test_primary_subtag_lowercased(self, raw: str, expected: str) -> None:
        assert normalize_language_tag(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", None, "-", "_", "123", "1-en", "?"])
    def test_unusable_input_is_none_not_a_guess(self, raw: object) -> None:
        """``None`` means "the feed said nothing useful" — distinguishable from ``"en"``."""
        assert normalize_language_tag(raw) is None  # type: ignore[arg-type]

    def test_the_live_bug_this_fixes(self) -> None:
        """``en-US`` must reach ``en``, because ``en-us`` is not English to the code.

        ``whisper_utils.py:50`` is ``language.lower() in ("en", "english")``, so the merely
        lowercased ``"en-us"`` reads as NOT English and a non-``.en`` Whisper model gets picked
        for an English episode. All 40 episodes in ``app-validation-corpus/v3`` carry that value.
        """
        from podcast_scraper.providers.ml.whisper_utils import normalize_whisper_model_name

        assert normalize_language_tag("en-US") == "en"
        # The merely-lowercased form loses the English path; the normalized one keeps it.
        assert normalize_whisper_model_name("base.en", "en-us") != normalize_whisper_model_name(
            "base.en", normalize_language_tag("en-US")
        )


class TestResolveLanguage:
    def test_the_feed_wins_and_says_so(self) -> None:
        assert resolve_language("es-ES", "en") == ("es-ES", "es", SOURCE_RSS)

    def test_raw_is_kept_verbatim(self) -> None:
        """The operator judging an odd tag needs the publisher's original, not our reading."""
        raw, norm, _src = resolve_language("pt_BR", "en")
        assert (raw, norm) == ("pt_BR", "pt")

    def test_no_tag_falls_back_to_the_profile_and_says_so(self) -> None:
        assert resolve_language(None, "en-US") == (None, "en", SOURCE_PROFILE_DEFAULT)

    def test_und_is_passed_through_not_special_cased(self) -> None:
        """D-21: NO ``und`` / ``zxx`` / ``mul`` policy. This test originally asserted one.

        ``und`` is alphabetic, so it normalizes to ``"und"`` and the source is the feed. That
        is the decision, not an oversight: a feed declaring ``und`` is an onboarding
        conversation, and the registry will not have ``und`` enabled, so the episode gets
        skipped with a reason (S0.8) rather than silently reinterpreted as English here.
        """
        assert resolve_language("und", "en") == ("und", "und", SOURCE_RSS)

    def test_a_non_alphabetic_tag_keeps_the_raw_value_visible(self) -> None:
        """Declaring junk is not the same as declaring nothing, and the audit needs to tell."""
        assert resolve_language("123", "en") == ("123", "en", SOURCE_PROFILE_DEFAULT)

    def test_a_mock_feed_attribute_is_treated_as_absent(self) -> None:
        """A large number of tests pass MagicMock feeds.

        ``Mock().language`` is another Mock, whose ``.strip()`` is a Mock too — normalizing it
        would either raise or invent a language out of a test double.
        """
        assert resolve_language(MagicMock().language, "en") == (
            None,
            "en",
            SOURCE_PROFILE_DEFAULT,
        )

    def test_no_feed_tag_and_no_profile_default_resolves_to_nothing(self) -> None:
        assert resolve_language(None, None) == (None, None, SOURCE_PROFILE_DEFAULT)

    def test_source_is_always_recorded(self) -> None:
        """An audit that cannot say WHERE a language came from cannot distinguish a measured
        corpus from one that silently defaulted every episode."""
        for raw, default in (("es", "en"), (None, "en"), ("und", "en"), (None, None)):
            _raw, _norm, src = resolve_language(raw, default)
            assert src in (SOURCE_RSS, SOURCE_PROFILE_DEFAULT)


class TestRegistry:
    """``config/languages.yaml`` — the gate on which languages we ingest (#2174)."""

    def test_exactly_the_six_western_european_languages_are_enabled_today(self) -> None:
        """`es` was enabled 2026-09-30; `it`, `fr`, `de` and `pt` followed on 2026-10-01.

        Pinned as an EXACT set rather than "at least these", because `enabled` is the only gate
        there is (D-41: there is no feature flag) — so a language appearing here is a language
        the pipeline will ingest, translate and publish. That must be a deliberate edit with the
        arc's safety conditions met, never a side effect of touching the registry.

        This test did its job: enabling the four broke it, and that is the tripwire firing rather
        than a number going stale. What was met for each of the four before the edit —

          * a hand-authored transcript, parallel in content to `p01_e01` so a translation can be
            compared against one known meaning;
          * synthesized audio whose speakers a REAL pyannote run separates into three, which is
            what the transcript declares — including Italian and German, where macOS ships one
            voice and host and guest are the same synthesis at two pitches;
          * an authored ground-truth sidecar, so the corpus builder reads real topics and a real
            summary instead of falling into the stand-in path;
          * an RSS fixture declaring the regional tag, so normalisation stays load-bearing.

        Quality of the translations themselves is NOT among those conditions and is not claimed
        here — that is the V2 quality gate's job.
        """
        from podcast_scraper.languages import is_language_enabled, language_registry

        reg = language_registry()
        assert reg, "registry failed to load — every language would read as not-enabled"
        assert sorted(c for c, e in reg.items() if e.enabled) == [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "pt",
        ]
        for code in ("en", "es", "it", "fr", "de", "pt"):
            assert is_language_enabled(code) is True, code
        # Still described, still not ingested — the distinction the registry exists to carry.
        for code in ("ru", "ja", "ar"):
            assert is_language_enabled(code) is False, code

    def test_all_three_tiers_are_described(self) -> None:
        """Present-but-disabled is the point: a language we have described, not one we ingest."""
        from podcast_scraper.languages import language_registry

        tiers = {e.tier for e in language_registry().values()}
        assert {0, 1, 2, 3} <= tiers

    def test_an_unknown_language_is_not_enabled(self) -> None:
        """Absent → not enabled. Never a lenient default; that is the hazard being removed."""
        from podcast_scraper.languages import is_language_enabled

        for code in ("xx", "klingon", "", None):
            assert is_language_enabled(code) is False  # type: ignore[arg-type]

    def test_the_two_copies_do_not_drift(self) -> None:
        """The bundled fallback is the container's only copy; the known_models pair had already
        drifted silently once, which is why this is asserted rather than assumed."""
        import pathlib

        repo = pathlib.Path(__file__).resolve().parents[3]
        configured = repo / "config" / "languages.yaml"
        bundled = repo / "src" / "podcast_scraper" / "data" / "languages.yaml"
        assert configured.read_bytes() == bundled.read_bytes(), (
            "config/languages.yaml and src/podcast_scraper/data/languages.yaml have drifted; "
            "copy one over the other"
        )


class TestResolveEpisodeLanguage:
    def test_override_beats_the_feed(self) -> None:
        from podcast_scraper.languages import resolve_episode_language, SOURCE_OVERRIDE

        assert resolve_episode_language(
            override="es", feed_declared="de-DE", profile_default="en"
        ) == ("es", "es", SOURCE_OVERRIDE)

    def test_the_feed_beats_the_profile(self) -> None:
        from podcast_scraper.languages import resolve_episode_language

        assert resolve_episode_language(feed_declared="de-DE", profile_default="en") == (
            "de-DE",
            "de",
            SOURCE_RSS,
        )

    def test_the_profile_is_the_last_resort(self) -> None:
        from podcast_scraper.languages import resolve_episode_language

        assert resolve_episode_language(profile_default="en-US") == (
            None,
            "en",
            SOURCE_PROFILE_DEFAULT,
        )

    def test_the_mis_tagged_english_feed_can_be_rescued(self) -> None:
        """The case S0.8's ordering depends on: an English show tagged `de` would otherwise be
        skipped on every run — including relabels — with no remedy until the override exists."""
        from podcast_scraper.languages import resolve_episode_language, SOURCE_OVERRIDE

        _raw, lang, src = resolve_episode_language(
            override="en", feed_declared="de", profile_default="en"
        )
        assert (lang, src) == ("en", SOURCE_OVERRIDE)

    def test_resolution_does_not_consult_the_registry(self) -> None:
        """ "What language is this" and "do we ingest it" are separate questions.

        Folding them together would mean an unsupported language silently resolving to
        something else instead of being skipped with a reason.
        """
        from podcast_scraper.languages import is_language_enabled, resolve_episode_language

        _raw, lang, src = resolve_episode_language(feed_declared="ja", profile_default="en")
        assert (lang, src) == ("ja", SOURCE_RSS), "resolution reports what it found"
        assert is_language_enabled(lang) is False, "enablement is the separate answer"


class TestTheTwoLiveBugsThisSliceFixes:
    def test_config_language_en_US_now_passes_is_english(self) -> None:
        """``Config._normalize_language`` only lowercased, so ``en-US`` became ``en-us`` — which
        ``whisper_utils.py:50`` reads as NOT English, picking a non-``.en`` model."""
        from podcast_scraper import config as config_mod

        cfg = config_mod.Config(rss="https://example.com/f.xml", language="en-US")
        assert cfg.language == "en"

        from podcast_scraper.providers.ml.whisper_utils import normalize_whisper_model_name

        name, chain = normalize_whisper_model_name("base.en", cfg.language)
        assert name == "base.en", "an English episode must keep the .en model"
        # The chain carries a smaller fallback too (`tiny.en`); what matters is that EVERY
        # entry stays an English variant. Under the old `"en-us"` the `.en` suffix was stripped
        # and the chain became the multilingual ["base", "tiny"].
        assert chain and all(m.endswith(".en") for m in chain), chain

        stripped_name, stripped_chain = normalize_whisper_model_name("base.en", "en-us")
        assert not all(m.endswith(".en") for m in stripped_chain), (
            f"the merely-lowercased tag should lose the English path: "
            f"{stripped_name} {stripped_chain}"
        )

    def test_the_per_feed_override_needs_both_the_field_and_the_allowlist(self) -> None:
        """``extra="forbid"`` rejects the key without the field; Config's coercion silently
        DROPS it without the allowlist entry. Both, or the override does nothing."""
        from podcast_scraper.rss.feeds_spec import (
            RSS_FEED_ENTRY_OVERRIDE_KEYS,
            RssFeedEntry,
        )

        assert "language" in RSS_FEED_ENTRY_OVERRIDE_KEYS, "the YAML key must survive coercion"
        entry = RssFeedEntry.model_validate({"url": "https://example.com/f.xml", "language": "es"})
        assert entry.language_override == "es"

    def test_the_override_is_normalized_at_the_model_boundary(self) -> None:
        """``model_copy(update=...)`` skips Config's validators, so ``language: es-ES`` would
        otherwise reach the Config raw — the ``en-US`` bug arriving by a different door."""
        from podcast_scraper.rss.feeds_spec import RssFeedEntry

        assert (
            RssFeedEntry.model_validate(
                {"url": "https://example.com/f.xml", "language": "es-ES"}
            ).language_override
            == "es"
        )

    def test_a_typo_in_the_override_stops_the_run(self) -> None:
        """A hand-edited override is the one place a typo should be loud: silently falling back
        would leave the operator believing they had corrected a feed they had not."""
        import pydantic

        from podcast_scraper.rss.feeds_spec import RssFeedEntry

        with pytest.raises(pydantic.ValidationError):
            RssFeedEntry.model_validate({"url": "https://example.com/f.xml", "language": "123"})

    def test_the_override_lands_somewhere_it_can_outrank_the_feed(self) -> None:
        """It must NOT land on ``cfg.language``.

        There it would be indistinguishable from the profile default, so the artifact could not
        record which source won — and ``_build_feed_metadata`` prefers the feed's own tag over
        the profile default, so the override would lose to the wrong tag it exists to correct.
        """
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        cfg = config_mod.Config(rss="https://example.com/a.xml", language="en")
        merged = merge_feed_entry_into_config(
            cfg,
            RssFeedEntry.model_validate({"url": "https://example.com/f.xml", "language": "es"}),
        )
        assert merged.language_override == "es"
        assert merged.language == "en", "the profile default must stay distinguishable"
