"""The LLM cleaning stage must not accept a fragment (#1982).

The PATTERN stage got a destruction guard in #1822. The LLM stage never had one, so a truncated
generation was accepted verbatim. On 2026-09-05 three prod episodes across two feeds kept
23.9 / 24.2 / 26.6% of their transcript — a near-identical fraction, which is an output-budget
cutoff rather than editing.

The outer guard (``metadata_generation._MIN_CLEANED_RATIO`` = 0.30) then caught the fragment and
fell back to the RAW transcript — discarding the good pattern pass and sending ad-bearing text to
summarisation. That is the ad-contamination route #1976 was filed about. Keeping the
pattern-cleaned text closes it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.cleaning.hybrid import HybridCleaner
from podcast_scraper.workflow import metrics

pytestmark = pytest.mark.unit

_RAW = (
    ("Welcome back to the show. " * 100)
    + ("This episode is brought to you by Acme. " * 20)
    + ("Now to the real discussion about monetary policy and the Duma. " * 100)
)


class _Provider:
    """Minimal provider exposing the attribute the hybrid cleaner probes for."""

    def __init__(self, returns: str) -> None:
        self._returns = returns

    def clean_transcript(self, text: str, **_kw: object) -> str:  # pragma: no cover - shape only
        return self._returns


def _clean(monkeypatch, llm_output: str, m=None) -> str:
    cleaner = HybridCleaner()
    monkeypatch.setattr(cleaner.llm_cleaner, "clean", lambda text, provider, **kw: llm_output)
    monkeypatch.setattr(cleaner, "_needs_llm_cleaning", lambda *a, **k: True)
    return cleaner.clean(_RAW, provider=_Provider(llm_output), pipeline_metrics=m)


def test_a_truncated_llm_result_is_rejected(monkeypatch) -> None:
    """The regression: ~25% back from the model must not become the transcript."""
    fragment = _RAW[: int(len(_RAW) * 0.25)]
    out = _clean(monkeypatch, fragment)
    assert out != fragment, "a 25% fragment was accepted as the cleaned transcript"
    assert len(out) > len(_RAW) * 0.5


def test_the_fallback_is_pattern_cleaned_not_raw(monkeypatch) -> None:
    """Falling back to RAW would re-admit the ads the pattern stage already stripped (#1976)."""
    cleaner = HybridCleaner()
    pattern_only = cleaner.pattern_cleaner.clean(_RAW)
    monkeypatch.setattr(cleaner.llm_cleaner, "clean", lambda text, provider, **kw: "tiny")
    monkeypatch.setattr(cleaner, "_needs_llm_cleaning", lambda *a, **k: True)
    out = cleaner.clean(_RAW, provider=_Provider("tiny"))
    assert out == pattern_only


def test_a_plausible_llm_result_is_still_used(monkeypatch) -> None:
    """The guard must be narrow — normal ad removal is well above the floor."""
    plausible = _RAW[: int(len(_RAW) * 0.85)]
    assert _clean(monkeypatch, plausible) == plausible


def test_rejection_is_counted(monkeypatch) -> None:
    m = metrics.Metrics()
    _clean(monkeypatch, _RAW[: int(len(_RAW) * 0.2)], m)
    out = m.finish()
    assert out["llm_cleaning_rejected_events"] == 1
    assert out["llm_cleaning_rejected_chars_lost"] > 0


def test_short_texts_are_exempt(monkeypatch) -> None:
    """Below the 2000-char floor a large proportional drop is not evidence of truncation."""
    cleaner = HybridCleaner()
    short = "Hello there. " * 20
    monkeypatch.setattr(cleaner.llm_cleaner, "clean", lambda text, provider, **kw: "Hello.")
    monkeypatch.setattr(cleaner, "_needs_llm_cleaning", lambda *a, **k: True)
    assert cleaner.clean(short, provider=_Provider("Hello.")) == "Hello."
