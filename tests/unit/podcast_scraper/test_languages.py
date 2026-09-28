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
