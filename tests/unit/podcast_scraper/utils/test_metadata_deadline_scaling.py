"""#1920: the metadata-generation deadline scales with transcript length.

It used to be a flat ``summarization_timeout`` (1200s) wrapping summary+GI+KG, which is linear
in the transcript. Measured over the 2026-09-01 Batch A pass, Pearson(word_count, metadata_sec)
= 0.868 and the single overrun was the single longest episode — it completed successfully and
still raised an ERROR-level DEADLINE EXCEEDED. §5h of the onboarding plan allows two-hour
episodes, so a flat budget was guaranteed to keep firing on legitimate work.
"""

from __future__ import annotations

from podcast_scraper import config as cfgmod
from podcast_scraper.utils.timeout_config import (
    get_metadata_generation_timeout,
    METADATA_SEC_PER_1K_TRANSCRIPT_WORDS,
)


def _cfg(flat: int = 1200) -> cfgmod.Config:
    return cfgmod.Config(
        rss="https://example.com/feed.xml",
        summarization_timeout=flat,
    )


def test_short_episode_keeps_the_flat_budget() -> None:
    """No regression: below the crossover the configured value still wins."""
    assert get_metadata_generation_timeout(_cfg(), 7_892) == 1200.0


def test_long_episode_scales_above_the_flat_budget() -> None:
    """The 16,345-word episode that overran 1200s in production."""
    got = get_metadata_generation_timeout(_cfg(), 16_345)
    assert got is not None
    assert got > 1200.0
    assert got == 16.345 * METADATA_SEC_PER_1K_TRANSCRIPT_WORDS


def test_two_hour_episode_gets_room() -> None:
    """§5h ceiling (~20k words) must not be structurally guaranteed to overrun.

    Observed worst case was 74.5 s per 1k words, so a 20k-word episode needs ~1490s — more
    than the flat budget ever allowed.
    """
    got = get_metadata_generation_timeout(_cfg(), 20_000)
    assert got is not None
    assert got >= 20 * 74.5


def test_missing_word_count_falls_back_to_flat() -> None:
    """An unreadable transcript must not produce a tiny deadline."""
    for bad in (0, -1, None):
        assert get_metadata_generation_timeout(_cfg(), bad) == 1200.0  # type: ignore[arg-type]


def test_respects_a_configured_non_default_deadline() -> None:
    assert get_metadata_generation_timeout(_cfg(3600), 7_892) == 3600.0
    scaled = 10 * METADATA_SEC_PER_1K_TRANSCRIPT_WORDS
    assert get_metadata_generation_timeout(_cfg(100), 10_000) == scaled


def test_word_count_helper_never_raises(tmp_path) -> None:
    from podcast_scraper.workflow.stages.processing import _transcript_word_count

    assert _transcript_word_count(None) == 0
    assert _transcript_word_count(str(tmp_path / "does-not-exist.txt")) == 0
    f = tmp_path / "t.txt"
    f.write_text("one two three\nfour five\n", encoding="utf-8")
    assert _transcript_word_count(str(f)) == 5


def test_none_deadline_is_preserved_not_coerced() -> None:
    """``summarization_timeout: None`` is documented as "disable timeout" (config.py:799).

    The first cut of this function did ``float(getattr(cfg, ...))`` and raised TypeError on that
    setting; the caller's broad ``except Exception`` would have turned it into every-episode-
    failed. Coercing to 1200 would be equally wrong — it silently enables a deadline the
    operator disabled. ``timeout_context`` takes None to mean "do not observe".
    """
    cfg = cfgmod.Config(rss="https://example.com/feed.xml", summarization_timeout=None)
    assert get_metadata_generation_timeout(cfg, 10_000) is None
    assert get_metadata_generation_timeout(cfg, 0) is None


def test_zero_deadline_is_not_scaled_into_an_enabled_one() -> None:
    """<= 0 also disables observation; scaling it up would re-enable it behind the operator."""
    cfg = cfgmod.Config(rss="https://example.com/feed.xml", summarization_timeout=0)
    assert get_metadata_generation_timeout(cfg, 20_000) == 0


def test_rate_is_profile_overridable() -> None:
    """The built-in rate is the MEASURED prod_dgx_full number, not a universal constant.

    A cloud profile runs on different hardware with different latency and retry behaviour; its
    seconds-per-word is a different number and nobody has measured it. Inheriting the local rate
    would apply one environment's hardware to another, so profiles can override.
    """
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summarization_timeout=1200,
        metadata_sec_per_1k_words=400.0,
    )
    assert get_metadata_generation_timeout(cfg, 10_000) == 4000.0


def test_rate_default_is_the_dgx_measured_value() -> None:
    cfg = _cfg()
    assert get_metadata_generation_timeout(cfg, 10_000) == 10 * METADATA_SEC_PER_1K_TRANSCRIPT_WORDS


def test_override_still_respects_the_flat_floor() -> None:
    """A tiny override must not shrink the budget below the configured deadline."""
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summarization_timeout=1200,
        metadata_sec_per_1k_words=1.0,
    )
    assert get_metadata_generation_timeout(cfg, 10_000) == 1200.0
