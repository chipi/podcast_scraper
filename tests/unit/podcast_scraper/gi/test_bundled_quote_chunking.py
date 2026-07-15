"""Bundled quote extraction must CHUNK, and BISECT a chunk that fails — not dump the episode.

THE BUG (found in the bake-off): `extract_quotes_bundled` sent EVERY insight plus the transcript in
one call. The response is ~256 tokens/insight, so a 50+-insight episode overran the 8192-token cap
and came back as truncated JSON on deepseek, or timed out server-side (gemini 504) / client-side
(mistral) on the oversized request. The single call then RAISED, and the whole episode fell back to
one `extract_quotes` call PER insight — ~80 calls/episode instead of ~10, the 8x blow-up that made
DeepSeek take "5 hours".

The fix chunks the call (default 10 insights) and BISECTS any chunk that still fails, so a
reasoning model whose 10-insight payload truncates gets retried at 5, then 2, then 1 — shrinking
only where a provider needs it. Only a size-1 chunk that still fails drops to the per-insight path.

These tests use a fake provider whose bundled call fails above a size threshold — the exact shape of
the truncation bug — and assert the dispatcher recovers full coverage instead of collapsing.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from podcast_scraper.gi import pipeline as gipipe

pytestmark = pytest.mark.unit


class _FakeCandidate:
    def __init__(self, text: str) -> None:
        self.text = text
        self.char_start = 0
        self.char_end = len(text)
        self.qa_score = 1.0


class _SizeLimitedProvider:
    """Bundled extraction that FAILS when asked for more than ``limit`` insights at once — exactly
    how a truncation / timeout presents. Records every batch size it was called with."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.batch_sizes: List[int] = []

    def extract_quotes_bundled(
        self, transcript: str, insight_texts: List[str], **kwargs: Any
    ) -> Dict[int, List[Any]]:
        self.batch_sizes.append(len(insight_texts))
        if len(insight_texts) > self.limit:
            raise ValueError("invalid JSON: Unterminated string (simulated truncation)")
        return {i: [_FakeCandidate(f"quote for {t[:10]}")] for i, t in enumerate(insight_texts)}


def _cfg(chunk: int = 10) -> Any:
    cfg = MagicMock()
    cfg.gil_evidence_quote_mode = "bundled"
    cfg.gil_evidence_quote_bundle_chunk = chunk
    return cfg


def _insights(n: int) -> List[str]:
    return [f"insight number {i} about some topic" for i in range(n)]


def test_a_provider_that_handles_the_chunk_size_covers_everything_in_ceil_calls() -> None:
    prov = _SizeLimitedProvider(limit=10)
    out = gipipe._maybe_prefetch_bundled_candidates(
        cfg=_cfg(10),
        quote_extraction_provider=prov,
        transcript="t",
        insight_texts=_insights(25),
        pipeline_metrics=None,
    )
    assert out is not None
    assert set(out) == set(range(25)), "every insight index must be covered, remapped to global"
    # 25 insights / chunk 10 -> 3 calls (10, 10, 5); no bisection needed.
    assert prov.batch_sizes == [10, 10, 5] or sorted(prov.batch_sizes) == [5, 10, 10]


def test_a_truncating_chunk_is_BISECTED_not_dumped_to_per_insight() -> None:
    """THE REGRESSION. A provider that truncates above 3 insights must still reach FULL coverage by
    bisection — not return None (which would send all 12 insights to per-insight extraction)."""
    prov = _SizeLimitedProvider(limit=3)
    out = gipipe._maybe_prefetch_bundled_candidates(
        cfg=_cfg(10),
        quote_extraction_provider=prov,
        transcript="t",
        insight_texts=_insights(12),
        pipeline_metrics=None,
    )
    assert out is not None, "bisection must recover, not collapse to the staged path"
    assert set(out) == set(range(12)), "all 12 insights covered after bisecting the failed chunks"
    # The initial 10-chunk failed (>3), bisected to 5+5 (still >3), each to 2/3 — all <=3 succeed.
    assert max(prov.batch_sizes) == 10, "started at the chunk size"
    assert any(s <= 3 for s in prov.batch_sizes), "bisected down to a size the provider accepts"


def test_global_index_remap_is_correct_after_bisection() -> None:
    """Bisected sub-batches carry a start offset; a mis-remap would attach quotes to the wrong
    insight — silently corrupting evidence. Give each insight a unique quote and check alignment."""
    prov = _SizeLimitedProvider(limit=2)
    insights = _insights(8)
    out = gipipe._maybe_prefetch_bundled_candidates(
        cfg=_cfg(10),
        quote_extraction_provider=prov,
        transcript="t",
        insight_texts=insights,
        pipeline_metrics=None,
    )
    assert out is not None and set(out) == set(range(8))
    # Each returned candidate's text was derived from its insight's first 10 chars — verify the
    # global index maps to the right insight.
    for idx, cands in out.items():
        assert insights[idx][:10] in cands[0].text


def test_only_a_size_one_failure_falls_back_and_metrics_count_it() -> None:
    """A provider that fails even on ONE insight is a genuine per-insight fallback — recorded, and
    the OTHER insights still come back bundled."""

    class _AlwaysFailsOnFour:
        def extract_quotes_bundled(self, transcript, insight_texts, **kw):
            # Fails only when insight index 4's text is in the batch, at any size.
            if any("number 4 " in t for t in insight_texts):
                raise ValueError("simulated hard failure on insight 4")
            return {i: [_FakeCandidate(t)] for i, t in enumerate(insight_texts)}

    metrics = MagicMock()
    metrics.gi_evidence_extract_quotes_bundled_fallbacks = 0
    metrics.gi_evidence_extract_quotes_bundled_calls = 0
    prov = _AlwaysFailsOnFour()
    out = gipipe._maybe_prefetch_bundled_candidates(
        cfg=_cfg(10),
        quote_extraction_provider=prov,
        transcript="t",
        insight_texts=_insights(8),
        pipeline_metrics=metrics,
    )
    assert out is not None
    assert 4 not in out, "insight 4 could not be bundled at any size -> staged fallback"
    assert set(out) == set(range(8)) - {4}, "every other insight still bundled"
    assert metrics.gi_evidence_extract_quotes_bundled_fallbacks == 1


def test_disabled_or_missing_bundled_method_returns_none() -> None:
    cfg = _cfg(10)
    cfg.gil_evidence_quote_mode = "staged"
    assert (
        gipipe._maybe_prefetch_bundled_candidates(
            cfg=cfg,
            quote_extraction_provider=_SizeLimitedProvider(10),
            transcript="t",
            insight_texts=_insights(5),
            pipeline_metrics=None,
        )
        is None
    )
