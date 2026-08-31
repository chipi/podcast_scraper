"""An output-budget overflow is the caller's to fix, not another vendor's (#1888 / #1886).

Production, 2026-08-30, prod_dgx_full: ``extract_quotes_bundled`` ended at 2560/2560 tokens
with ``finish_reason == "length"`` on ten-insight batches. The bisect in
``_maybe_prefetch_bundled_candidates`` is built to halve exactly this and retry on the same
model — but it never ran, because the RFC-106 fallback wrapper caught the exception first and
answered from ``llama3.1:8b``. Seven times, across three of eight episodes, and every run
still reported success.

The load-bearing test here is the LAST one: a real outage must still fail over. A fix that
stops the bleeding by disabling failover for this method would pass every other test in this
file and silently remove the protection RFC-106 exists to provide.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config
from podcast_scraper.providers.common.bundle_extract_parser import (
    BundleExtractParseError,
    BundleOutputBudgetExceeded,
)
from podcast_scraper.providers.common.bundled_prompts import extract_quotes_bundled_max_tokens
from podcast_scraper.summarization.fallback import FallbackAwareSummarizationProvider


class _Primary:
    """Primary whose bundled call raises whatever it was handed."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def extract_quotes_bundled(self, **kwargs):
        self.calls += 1
        raise self._exc


class _Fallback:
    def __init__(self):
        self.calls = 0

    def extract_quotes_bundled(self, **kwargs):
        self.calls += 1
        return {0: ["from the weaker model"]}


def _wrap(primary, fallback, monkeypatch):
    cfg = config.Config.model_validate({"rss_url": "https://example.com/f.rss"})
    wrapper = FallbackAwareSummarizationProvider(primary, ["ollama"], cfg)
    monkeypatch.setattr(wrapper, "_get_or_build_fallback", lambda name: fallback)
    return wrapper


def test_budget_overflow_propagates_so_the_caller_can_bisect(monkeypatch):
    fb = _Fallback()
    primary = _Primary(BundleOutputBudgetExceeded("2560/2560 for 10 insights"))
    wrapper = _wrap(primary, fb, monkeypatch)

    with pytest.raises(BundleOutputBudgetExceeded):
        wrapper.extract_quotes_bundled(transcript="t", insight_texts=["a"])

    assert fb.calls == 0, "budget overflow must not be answered by the fallback tier"


def test_the_marker_is_what_the_wrapper_keys_on():
    """Not the method name — the same method must still fail over when the endpoint is down."""
    assert BundleOutputBudgetExceeded.caller_can_retry_smaller is True
    assert getattr(BundleExtractParseError, "caller_can_retry_smaller", False) is False


def test_plain_parse_error_still_fails_over(monkeypatch):
    """A malformed-JSON failure is NOT budget-shaped; the next vendor is a reasonable answer."""
    fb = _Fallback()
    primary = _Primary(BundleExtractParseError("bad json", truncation_suspected=True))
    wrapper = _wrap(primary, fb, monkeypatch)

    assert wrapper.extract_quotes_bundled(transcript="t", insight_texts=["a"]) == {
        0: ["from the weaker model"]
    }
    assert fb.calls == 1


def test_a_real_outage_still_fails_over(monkeypatch):
    """The regression guard: do not let this fix become 'disable failover for this method'."""
    fb = _Fallback()
    primary = _Primary(ConnectionError("vLLM endpoint unreachable"))
    wrapper = _wrap(primary, fb, monkeypatch)

    assert wrapper.extract_quotes_bundled(transcript="t", insight_texts=["a"]) == {
        0: ["from the weaker model"]
    }
    assert fb.calls == 1, "a genuine outage must still reach the fallback chain"


def test_budget_is_above_the_distribution_production_measured():
    """Ten insights must now exceed the 2560 that production hit exactly."""
    assert extract_quotes_bundled_max_tokens(10) > 2560
    # Still bounded, and the floor still applies to tiny batches.
    assert extract_quotes_bundled_max_tokens(1) == 1024
    assert extract_quotes_bundled_max_tokens(1000) == 8192
