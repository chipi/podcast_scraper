"""Losing audio preprocessing must be visible in the ledger (#1647, #558).

When ffmpeg fails or times out, the pipeline falls back to the ORIGINAL audio and carries on.
That is the right behaviour — a transcript from unprocessed audio beats no transcript — but it
is a real degradation: no mono/16 kHz/loudness normalisation, and a file that may genuinely
exceed the 25 MB upload cap (which is when that cap finally means something).

Before this, the only trace was a WARNING line and a row in ``corpus_incidents.jsonl``. The
stage ledger — the artifact built specifically to answer "what actually happened to this
episode" — said nothing at all, so a degraded episode looked exactly like a clean one. That is
the #1646 shape: the failure is not that something broke, it is that nothing said so.

Found on the acceptance run, where 4 of 5 episodes silently lost preprocessing and the ledger
recorded none of it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

pytestmark = [pytest.mark.unit]


class _Metrics:
    """Records ledger writes; everything else is a no-op the caller may freely invoke."""

    def __init__(self) -> None:
        self.outcomes: List[Dict[str, Any]] = []

    def record_stage_outcome(
        self,
        stage: str,
        episode_idx: int,
        outcome: str,
        reason: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        self.outcomes.append(
            {
                "stage": stage,
                "outcome": outcome,
                "reason": reason,
                "detail": detail or {},
                "duration_seconds": duration_seconds,
            }
        )

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: None


def _preprocessing_rows(m: _Metrics) -> List[Dict[str, Any]]:
    return [o for o in m.outcomes if o["stage"] == "audio_preprocessing"]


class TestTheDegradationIsRecorded:
    """These assert the CONTRACT the episode_processor fallback must satisfy."""

    def test_a_fallback_records_degraded_not_failed(self) -> None:
        """`degraded`, not `failed`: transcription still happens, just from worse input.
        Calling it failed would over-report; saying nothing under-reports."""
        m = _Metrics()
        m.record_stage_outcome(
            "audio_preprocessing",
            1,
            "degraded",
            reason="preprocessing_failed_using_original_audio",
            detail={"fallback": "original_audio", "media_bytes": 91 * 1024 * 1024},
            duration_seconds=300.0,
        )
        row = _preprocessing_rows(m)[-1]
        assert row["outcome"] == "degraded"
        assert row["reason"] == "preprocessing_failed_using_original_audio"

    def test_the_detail_says_what_was_actually_transcribed(self) -> None:
        """An operator asking "why is this episode's quality poor / cost high" needs the size
        of the file that really went to the provider."""
        m = _Metrics()
        m.record_stage_outcome(
            "audio_preprocessing",
            1,
            "degraded",
            reason="preprocessing_failed_using_original_audio",
            detail={"fallback": "original_audio", "media_bytes": 95_000_000},
        )
        d = _preprocessing_rows(m)[-1]["detail"]
        assert d["fallback"] == "original_audio"
        assert d["media_bytes"] == 95_000_000

    def test_the_reason_slug_is_groupable(self) -> None:
        """Stable slug, not prose — a corpus report has to GROUP BY it to answer "how many
        episodes lost preprocessing", which is the question the acceptance run raised."""
        m = _Metrics()
        for idx in (1, 2, 3):
            m.record_stage_outcome(
                "audio_preprocessing",
                idx,
                "degraded",
                reason="preprocessing_failed_using_original_audio",
            )
        reasons = {r["reason"] for r in _preprocessing_rows(m)}
        assert reasons == {"preprocessing_failed_using_original_audio"}


class TestSuccessIsRecordedToo:
    """A ledger that only speaks on failure cannot distinguish "ran fine" from "never ran"."""

    def test_success_records_ran_with_both_sizes(self) -> None:
        m = _Metrics()
        m.record_stage_outcome(
            "audio_preprocessing",
            1,
            "ran",
            detail={"original_bytes": 91_500_000, "preprocessed_bytes": 9_100_000},
            duration_seconds=241.5,
        )
        row = _preprocessing_rows(m)[-1]
        assert row["outcome"] == "ran"
        assert row["detail"]["original_bytes"] > row["detail"]["preprocessed_bytes"]

    def test_ran_and_degraded_are_distinguishable(self) -> None:
        """The whole point: two episodes, two different fates, two different ledger rows."""
        m = _Metrics()
        m.record_stage_outcome("audio_preprocessing", 1, "ran", detail={"original_bytes": 1})
        m.record_stage_outcome(
            "audio_preprocessing", 2, "degraded", reason="preprocessing_failed_using_original_audio"
        )
        assert [r["outcome"] for r in _preprocessing_rows(m)] == ["ran", "degraded"]


class TestTheCallSitesExist:
    """The contract above is worthless if episode_processor does not actually call it — which
    is precisely how the composite-host fix shipped green and broken."""

    def _source(self) -> str:
        from pathlib import Path

        from podcast_scraper.workflow import episode_processor

        return Path(episode_processor.__file__).read_text(encoding="utf-8")

    def test_the_fallback_branch_records_degraded(self) -> None:
        src = self._source()
        assert "preprocessing_failed_using_original_audio" in src
        assert '"audio_preprocessing",' in src

    def test_EVERY_success_path_records_ran(self) -> None:
        """There are TWO success paths — a fresh preprocess and a CACHE HIT — and both mean the
        episode was transcribed from preprocessed audio. An earlier version of this test only
        checked the first ``record_preprocessing_size_reduction`` and caught the cache-hit path
        having no ledger entry at all, which would have made every cached episode read as
        "preprocessing never happened"."""
        src = self._source()
        starts = [
            i for i in range(len(src)) if src.startswith("record_preprocessing_size_reduction", i)
        ]
        assert len(starts) == 2, f"expected 2 success paths, found {len(starts)}"
        for i in starts:
            assert (
                '"ran"' in src[i : i + 1200]
            ), "a preprocessing success path records no ledger outcome"

    def test_the_cache_hit_is_distinguishable(self) -> None:
        src = self._source()
        assert '"cache_hit": True' in src

    def test_every_path_is_guarded_for_older_metrics_objects(self) -> None:
        """``pipeline_metrics`` is duck-typed across callers; an unguarded call would turn a
        preprocessing hiccup into an AttributeError that kills the episode."""
        src = self._source()
        # The guard moved into _record_preprocessing_outcome, which every site funnels through —
        # one guarded choke point is stronger than three inline copies.
        assert 'hasattr(pipeline_metrics, "record_stage_outcome")' in src
        assert src.count("_record_preprocessing_outcome(") >= 3
