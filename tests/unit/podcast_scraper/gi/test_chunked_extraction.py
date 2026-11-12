"""Chunked extraction: more passes over the transcript, because models saturate per CALL.

qwen3.5:35b returns ~18 insights however long the episode is; gemini scales with the material.
Context is not the limit (a 90k transcript fits) — the ceiling is per-call. These tests pin the
behaviour that makes chunking safe: it never returns less than a single pass would, and a broken
chunk never costs the episode.
"""

from __future__ import annotations

from typing import Any, List


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


def test_no_profile_pins_chunking_to_zero() -> None:
    """#20: a good default is defeated if a profile pins gi_insight_chunk_chars: 0 — that was the
    original prod state that single-passed long episodes and truncated the back half. A non-zero
    chunk SIZE (e.g. 20000 on the bakeoff profiles) is fine; only 0 disables chunking. Guard every
    profile so an edit can't silently reopen it."""
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    profiles = sorted((repo_root / "config" / "profiles").glob("*.yaml"))
    assert profiles, "expected config/profiles/*.yaml"
    offenders = [
        p.name
        for p in profiles
        for line in p.read_text(encoding="utf-8").splitlines()
        if (m := re.match(r"\s*gi_insight_chunk_chars:\s*(\d+)\s*$", line)) and int(m.group(1)) == 0
    ]
    assert (
        not offenders
    ), f"profiles must not disable chunking (gi_insight_chunk_chars: 0): {offenders}"
