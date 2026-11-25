"""A news article's text is not put through the podcast ad cleaners before its summary.

Prod, the first news nightly (2026-11-24, GlitchTip podcast-pipeline 1191): a 704-character STAT
item was cut to 195 by ``HybridCleaner`` before summarisation, and the guard fell back to the raw
text. The news type's stage profile has no ``ad_cleaning``, and the ad-cleaning hooks already
honoured it; the summary's own cleaning (``stage_hooks.clean_for_summary`` on the staged path,
the pattern pre-clean on the bundled paths) did not.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper import content_types as ct, stage_hooks
from podcast_scraper.news import NEWS
from podcast_scraper.workflow import metadata_generation as mg

pytestmark = pytest.mark.unit

_TEXT = "Sponsored by Example Bank. Researchers report a new trial result today."


class _CuttingCleaner:
    """A cleaner that keeps a fragment, as the LLM and hybrid cleaners did on prod."""

    def clean(self, text: str, **_: Any) -> str:
        return text[:10]


def test_the_staged_summary_cleaning_leaves_a_news_article_unchanged() -> None:
    with ct.processing_item(ct.ItemContext(content_type=NEWS)):
        assert stage_hooks.clean_for_summary(_CuttingCleaner(), _TEXT) == _TEXT


def test_the_staged_summary_cleaning_still_cleans_a_podcast_episode() -> None:
    assert stage_hooks.clean_for_summary(_CuttingCleaner(), _TEXT) == _TEXT[:10]


def test_the_bundled_pre_clean_leaves_a_news_article_unchanged(monkeypatch) -> None:
    calls: Dict[str, int] = {"n": 0}

    class _Pattern:
        def clean(self, text: str, **_: Any) -> str:
            calls["n"] += 1
            return text[:10]

    monkeypatch.setattr("podcast_scraper.cleaning.PatternBasedCleaner", _Pattern)
    with ct.processing_item(ct.ItemContext(content_type=NEWS)):
        assert mg._pattern_pre_clean(_TEXT, {}) == _TEXT
    assert calls["n"] == 0
    assert mg._pattern_pre_clean(_TEXT, {}) == _TEXT[:10]
