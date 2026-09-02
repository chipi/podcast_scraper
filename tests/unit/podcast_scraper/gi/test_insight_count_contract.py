"""End-to-end contract on how many insights survive extraction, and WHICH ones (#1919).

Why this file exists: the #1919 fix removed a positional head-slice in the providers, and a
follow-up attempt to also raise the provider bound (so over-generated insights reach the value
gate) was **silently nullified** by ``_resolve_insight_specs``'s own bound one layer down —

    resolved_specs = resolved_specs[: max_insights * passes]      # gi/pipeline.py

— which, because ``plan_chunks`` returns 1 for transcripts under ``MIN_CHARS_TO_CHUNK``, is
exactly ``max_insights`` on a sub-45-minute episode, applied as a **head slice**. That would have
reintroduced the very defect #1919 removed, on the shortest episodes, while the provider-level
log line claimed the extras were kept.

Nothing caught it: the provider tests exercise ``insight_salvage`` in isolation and never run the
pipeline's own bound. These tests close the two layers together, so the next change to either one
has to face the coupling.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper import config_constants
from podcast_scraper.gi import pipeline as gi_pipeline
from podcast_scraper.gi.chunked_extraction import MIN_CHARS_TO_CHUNK, plan_chunks

pytestmark = pytest.mark.unit


class _OverGeneratingProvider:
    """A provider that ignores the requested count, like both models measured in production.

    Emits in TRANSCRIPT order — measured over 36 corpus episodes across two models,
    Pearson(emission order, position_hint) = 0.845, chronological on 34 of 36. Neither the
    local model nor the cloud model honours the prompts' "ORDER: most important first".
    """

    def __init__(self, produce: int) -> None:
        self.produce = produce
        self.requested: list[int] = []

    def generate_insights(self, *, text="", max_insights=25, **kwargs):  # noqa: ANN001
        """Signature matches ``generate_chunked``'s call: text=, max_insights=, keyword-only."""
        self.requested.append(max_insights)
        from podcast_scraper.providers import insight_salvage

        raw = [f"insight-{i:04d}" for i in range(self.produce)]
        # Mirror the committed #1919 provider behaviour: bound at the request, coverage-preserving.
        return insight_salvage.take_within_ceiling(raw, max_insights)


def _cfg(max_insights: int = 25, chunk_chars: int = 30_000) -> cfgmod.Config:
    return cfgmod.Config(
        rss="https://example.com/feed.xml",
        gi_max_insights=max_insights,
        gi_insight_chunk_chars=chunk_chars,
    )


def _short_transcript() -> str:
    """Under MIN_CHARS_TO_CHUNK, so plan_chunks == 1 and the pipeline bound == max_insights."""
    text = "word " * (MIN_CHARS_TO_CHUNK // 10)
    assert len(text) < MIN_CHARS_TO_CHUNK
    return text


def test_short_transcript_really_is_single_pass() -> None:
    """Guards the premise of the tests below rather than assuming it."""
    assert plan_chunks(_short_transcript(), 30_000) == 1


def test_single_pass_bound_equals_max_insights() -> None:
    """The coupling, stated as a test: on a short episode the pipeline bound IS max_insights.

    Any future change that raises the provider bound must relax this line too, or the extra
    insights are cut here instead — and cut positionally.
    """
    cfg = _cfg(max_insights=25)
    transcript = _short_transcript()
    passes = plan_chunks(transcript, int(cfg.gi_insight_chunk_chars or 0))
    scaled = config_constants.duration_scaled_max_insights(len(transcript), base=25)
    assert scaled * passes == scaled


def test_overgeneration_does_not_exceed_the_pipeline_bound() -> None:
    """A provider returning 4x the request must not inflate the resolved set."""
    cfg = _cfg(max_insights=25)
    transcript = _short_transcript()
    provider = _OverGeneratingProvider(produce=100)
    specs = gi_pipeline._resolve_insight_specs(transcript, cfg, insight_provider=provider)
    scaled = config_constants.duration_scaled_max_insights(len(transcript), base=25)
    assert len(specs) <= scaled * plan_chunks(transcript, 30_000)


def test_kept_insights_span_the_whole_episode_not_just_the_opening() -> None:
    """THE #1919 REGRESSION GUARD.

    Emission order tracks the transcript, so if any layer truncates positionally the surviving
    set collapses into the opening. Assert the last emitted insight survives — the single
    property a head slice can never satisfy.
    """
    cfg = _cfg(max_insights=25)
    transcript = _short_transcript()
    provider = _OverGeneratingProvider(produce=100)
    specs = gi_pipeline._resolve_insight_specs(transcript, cfg, insight_provider=provider)
    texts = [t for t, _ in specs]
    assert texts, "no insights resolved"
    indices = [int(t.split("-")[1]) for t in texts if t.startswith("insight-")]
    assert indices, f"unexpected insight texts: {texts[:3]}"
    # A head slice would end at index len-1 of the KEPT set; coverage means reaching the tail.
    assert max(indices) >= 75, (
        f"kept insights stop at index {max(indices)} of 100 — the set collapsed toward the "
        "opening of the episode, which is the #1919 defect"
    )
    assert min(indices) == 0, "the opening should still be represented"


def test_provider_receives_the_scaled_request_not_the_raw_config() -> None:
    """#1191 duration scaling must reach the provider, or long episodes are under-requested."""
    cfg = _cfg(max_insights=25)
    transcript = _short_transcript()
    provider = _OverGeneratingProvider(produce=10)
    gi_pipeline._resolve_insight_specs(transcript, cfg, insight_provider=provider)
    assert provider.requested, "provider was never asked"
    expected = config_constants.duration_scaled_max_insights(len(transcript), base=25)
    assert provider.requested[0] == expected


def test_under_generation_passes_through_untouched() -> None:
    """No bound should fire when the model returns fewer than asked."""
    cfg = _cfg(max_insights=25)
    provider = _OverGeneratingProvider(produce=7)
    specs = gi_pipeline._resolve_insight_specs(_short_transcript(), cfg, insight_provider=provider)
    assert len(specs) == 7
