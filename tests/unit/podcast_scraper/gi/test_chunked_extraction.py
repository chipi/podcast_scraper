"""Chunked extraction: more passes over the transcript, because models saturate per CALL.

qwen3.5:35b returns ~18 insights however long the episode is; gemini scales with the material.
Context is not the limit (a 90k transcript fits) — the ceiling is per-call. These tests pin the
behaviour that makes chunking safe: it never returns less than a single pass would, and a broken
chunk never costs the episode.
"""

from __future__ import annotations

from typing import Any, List

from podcast_scraper.gi.chunked_extraction import (
    generate_chunked,
    MIN_CHARS_TO_CHUNK,
    plan_chunks,
    split,
)


def _gen_factory(per_call: int, fail_on: int = -1):
    calls = {"n": 0}

    def gen(
        *, text: str, episode_title: Any, max_insights: int, params: Any, pipeline_metrics: Any
    ) -> List[str]:
        calls["n"] += 1
        if calls["n"] == fail_on:
            raise RuntimeError("chunk exploded")
        # each call returns distinct insights, tagged by call number
        return [
            f"call{calls['n']} insight {i} about a totally separate subject"
            for i in range(per_call)
        ]

    gen.calls = calls  # type: ignore[attr-defined]
    return gen


def test_short_episodes_are_never_chunked() -> None:
    """Below the floor the model is nowhere near its per-call ceiling; chunking cannot help."""
    assert plan_chunks("x" * (MIN_CHARS_TO_CHUNK - 1), 30_000) == 1
    assert plan_chunks("x" * 10_000, 30_000) == 1


def test_chunk_count_scales_with_length() -> None:
    assert plan_chunks("x" * 60_000, 30_000) == 2
    assert plan_chunks("x" * 90_000, 30_000) == 3


def test_chunking_disabled_by_default() -> None:
    assert plan_chunks("x" * 200_000, 0) == 1


def test_split_never_starts_mid_sentence() -> None:
    text = "\n".join(f"line {i}" for i in range(90))
    parts = split(text, 3)
    assert len(parts) == 3
    for p in parts:
        assert p.startswith("line ")


def test_split_handles_a_transcript_with_no_line_breaks() -> None:
    """Some transcripts are one unbroken string. Silently declining to chunk those would be a
    quiet no-op — the exact failure mode this codebase keeps producing."""
    text = ("The Fed held rates steady. Inflation cooled to two percent. " * 400).strip()
    parts = split(text, 3)
    assert len(parts) == 3, "must fall back to sentence boundaries"
    assert sum(len(p) for p in parts) >= len(text) - 10
    for p in parts:
        assert p.strip()


def test_chunked_yields_more_than_a_single_pass() -> None:
    gen = _gen_factory(per_call=18)
    out = generate_chunked(
        gen,
        "x" * 90_000,
        episode_title=None,
        max_insights=50,
        chunk_chars=30_000,
        dedupe_threshold=1.0,
    )
    assert gen.calls["n"] == 3  # type: ignore[attr-defined]
    assert len(out) > 18, "three passes must beat the per-call ceiling"


def test_a_failing_chunk_does_not_cost_the_episode() -> None:
    """One bad chunk must not take the whole episode down — that is the failure this repo keeps
    producing."""
    gen = _gen_factory(per_call=10, fail_on=2)
    out = generate_chunked(
        gen,
        "x" * 90_000,
        episode_title=None,
        max_insights=50,
        chunk_chars=30_000,
        dedupe_threshold=1.0,
    )
    assert len(out) >= 10, "surviving chunks must still contribute"


def test_falls_back_to_a_single_pass_when_chunking_yields_nothing() -> None:
    calls = {"n": 0}

    def gen(
        *, text: str, episode_title: Any, max_insights: int, params: Any, pipeline_metrics: Any
    ):
        calls["n"] += 1
        # every chunked call returns nothing; the whole-transcript retry returns real insights
        return [] if len(text) < 90_000 else ["a real insight from the whole transcript"]

    out = generate_chunked(
        gen,
        "x" * 90_000,
        episode_title=None,
        max_insights=50,
        chunk_chars=30_000,
        dedupe_threshold=1.0,
    )
    assert out == ["a real insight from the whole transcript"]


def test_short_episode_makes_exactly_one_call() -> None:
    gen = _gen_factory(per_call=12)
    out = generate_chunked(
        gen,
        "x" * 20_000,
        episode_title=None,
        max_insights=50,
        chunk_chars=30_000,
        dedupe_threshold=1.0,
    )
    assert gen.calls["n"] == 1  # type: ignore[attr-defined]
    assert len(out) == 12
