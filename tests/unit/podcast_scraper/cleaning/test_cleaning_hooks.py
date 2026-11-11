"""Transcript cleaning through the stage hooks (E04 IP extraction plan, item 2).

With today's code as the built-in owner, every caller gets the cleaner and the cleaning it got
before. Without an owner (public alone, once ad detection moves), the pattern cleaner does generic
cleaning only: no commercial detection, no cross-promo excision, no diarization context.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import preprocessing, stage_hooks
from podcast_scraper.cleaning import HybridCleaner, LLMBasedCleaner, PatternBasedCleaner
from podcast_scraper.cleaning.commercial import CommercialDetector
from podcast_scraper.cleaning.commercial.patterns import DEFAULT_CONFIDENCE_THRESHOLD

pytestmark = pytest.mark.unit

SPONSORED = (
    "Welcome back to the show. Today we talk about interest rates and what they mean.\n\n"
    "This episode is brought to you by Acme. Visit acme.com and use promo code SHOW "
    "for 20% off.\n\n"
    "Back to rates: the central bank held them steady this month, as most expected."
)


@pytest.fixture
def no_owner() -> Iterator[None]:
    """Only an owner-less hook set: the plain defaults answer."""
    with patch.object(stage_hooks, "_all", lambda: iter([stage_hooks.PipelineHooks(name="none")])):
        yield


@pytest.mark.parametrize(
    ("strategy", "cls"),
    [("pattern", PatternBasedCleaner), ("llm", LLMBasedCleaner), ("hybrid", HybridCleaner)],
)
def test_the_built_in_owner_picks_todays_cleaner(strategy: str, cls: type) -> None:
    assert type(stage_hooks.transcript_cleaner(strategy)) is cls


def test_an_unknown_strategy_is_hybrid_as_before() -> None:
    assert type(stage_hooks.transcript_cleaner("something-else")) is HybridCleaner


def test_the_confidence_threshold_reaches_the_pattern_cleaner() -> None:
    cleaner = stage_hooks.transcript_cleaner("pattern", confidence_threshold=0.42)
    assert cleaner._confidence_threshold == 0.42


def test_sponsor_removal_is_the_commercial_detectors() -> None:
    expected = CommercialDetector(confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD).remove(
        SPONSORED
    )
    assert preprocessing.remove_sponsor_blocks(SPONSORED) == expected
    assert expected != SPONSORED  # the sample really carries a sponsor read


def test_clean_for_summary_passes_each_cleaner_what_it_takes() -> None:
    provider, metrics, kwargs = object(), object(), {"host_speaker_id": "SPEAKER_00"}

    hybrid = MagicMock(spec=HybridCleaner)
    stage_hooks.clean_for_summary(
        hybrid, "t", provider=provider, pipeline_metrics=metrics, pattern_kwargs=kwargs
    )
    hybrid.clean.assert_called_once_with(
        "t", provider=provider, pipeline_metrics=metrics, host_speaker_id="SPEAKER_00"
    )

    llm = MagicMock(spec=LLMBasedCleaner)
    stage_hooks.clean_for_summary(
        llm, "t", provider=provider, pipeline_metrics=metrics, pattern_kwargs=kwargs
    )
    llm.clean.assert_called_once_with("t", provider, pipeline_metrics=metrics)

    pattern = MagicMock(spec=PatternBasedCleaner)
    stage_hooks.clean_for_summary(
        pattern, "t", provider=provider, pipeline_metrics=metrics, pattern_kwargs=kwargs
    )
    pattern.clean.assert_called_once_with("t", host_speaker_id="SPEAKER_00")


def test_without_an_owner_the_cleaner_is_the_pattern_cleaner(no_owner: None) -> None:
    for strategy in ("pattern", "llm", "hybrid"):
        assert type(stage_hooks.transcript_cleaner(strategy)) is PatternBasedCleaner


def test_without_an_owner_no_ad_step_runs(no_owner: None) -> None:
    assert stage_hooks.remove_commercials(SPONSORED) == SPONSORED
    assert stage_hooks.excise_crosspromo(SPONSORED) == SPONSORED
    assert stage_hooks.cleaning_context("/nowhere/ep.txt") == (None, None)
    assert preprocessing.remove_sponsor_blocks(SPONSORED) == SPONSORED


def test_without_an_owner_clean_for_summary_calls_clean(no_owner: None) -> None:
    processor = MagicMock()
    processor.clean.return_value = "cleaned"
    out = stage_hooks.clean_for_summary(
        processor, "t", provider=object(), pattern_kwargs={"host_speaker_id": "S1"}
    )
    assert out == "cleaned"
    processor.clean.assert_called_once_with("t", host_speaker_id="S1")
