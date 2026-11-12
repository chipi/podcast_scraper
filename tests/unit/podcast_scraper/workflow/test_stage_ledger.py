"""Unit tests for the per-episode stage ledger (#1647).

The ledger exists because the pipeline recorded only *durations*, so a stage that was
skipped, one that failed and was swallowed, and one that was never configured all reached
the metadata sidecar as the same ``null``. That ambiguity hid #1646 — speaker detection
silently skipped for every episode over 25 MB — across 72 % of the corpus.

These tests pin the property that makes the ledger worth having: **every exit path records an
outcome**, including the ones that record no duration. A regression here is not a cosmetic
reporting bug; it restores the blindness.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from podcast_scraper.workflow import metrics as metrics_mod
from podcast_scraper.workflow.metadata_generation import _extract_episode_stage_ledger, StageOutcome
from podcast_scraper.workflow.stages import processing

pytestmark = [pytest.mark.unit]


class _Episode:
    """Minimal stand-in for models.Episode: only what the detection path touches."""

    def __init__(self, idx: int = 1, title: str = "An Episode", media_url: str = "") -> None:
        self.idx = idx
        self.title = title
        self.media_url = media_url
        self.transcript_urls: List[str] = []
        self.item = MagicMock()


class _Cfg:
    """Config stub; attributes match what ``_detect_speakers_for_episode`` reads."""

    def __init__(self, **kwargs: Any) -> None:
        self.auto_speakers = True
        self.dry_run = False
        self.screenplay_speaker_names: List[str] = []
        self.cache_detected_hosts = False
        self.known_hosts: List[str] = []
        self.speaker_detector_provider = "litellm"
        self.transcribe_missing = True
        self.transcription_provider = "deepgram"
        self.user_agent = "test-agent"
        self.timeout = 5
        for key, value in kwargs.items():
            setattr(self, key, value)


def _ledger_for(m: metrics_mod.Metrics, idx: int = 1) -> Dict[str, Dict[str, Any]]:
    return m.stage_outcomes_by_episode.get(idx, {})


def _size_skip(cfg: Any, episode: Any) -> Any:
    """Typed seam for the duck-typed stubs below.

    The production signatures take the real ``Config``/``Episode``; constructing those here
    would drag provider validation and credentials into a unit test. One seam beats an
    ``arg-type`` ignore on every call site.
    """
    return processing._check_episode_size_skip(cfg, episode)


def _detect(episode: Any, cfg: Any, host_result: Any, m: Any, **kwargs: Any) -> Any:
    from podcast_scraper.speaker_stage import detection

    return detection._detect_speakers_for_episode(episode, cfg, host_result, m, **kwargs)


class TestRecordStageOutcome:
    def test_records_outcome_reason_detail_and_duration(self) -> None:
        m = metrics_mod.Metrics()
        m.record_stage_outcome(
            "speaker_detection",
            7,
            "skipped",
            reason="media_over_size_limit",
            detail={"media_bytes": 42871040},
            duration_seconds=1.5,
        )
        record = m.stage_outcomes_by_episode[7]["speaker_detection"]
        assert record["outcome"] == "skipped"
        assert record["reason"] == "media_over_size_limit"
        assert record["detail"] == {"media_bytes": 42871040}
        assert record["duration_seconds"] == 1.5

    def test_omits_absent_optional_fields_rather_than_writing_nulls(self) -> None:
        """A ledger full of explicit nulls re-creates the ambiguity it exists to remove."""
        m = metrics_mod.Metrics()
        m.record_stage_outcome("speaker_detection", 1, "ran")
        assert m.stage_outcomes_by_episode[1]["speaker_detection"] == {"outcome": "ran"}

    def test_separate_episodes_do_not_collide(self) -> None:
        m = metrics_mod.Metrics()
        m.record_stage_outcome("speaker_detection", 1, "ran")
        m.record_stage_outcome("speaker_detection", 2, "skipped", reason="dry_run")
        assert m.stage_outcomes_by_episode[1]["speaker_detection"]["outcome"] == "ran"
        assert m.stage_outcomes_by_episode[2]["speaker_detection"]["outcome"] == "skipped"

    def test_multiple_stages_coexist_on_one_episode(self) -> None:
        m = metrics_mod.Metrics()
        m.record_stage_outcome("speaker_detection", 1, "skipped", reason="dry_run")
        m.record_stage_outcome("transcription", 1, "ran", duration_seconds=23.7)
        assert set(_ledger_for(m)) == {"speaker_detection", "transcription"}

    def test_concurrent_writes_do_not_lose_records(self) -> None:
        """Episodes run on a pool; a dict mutated from several workers can drop entries."""
        import threading

        m = metrics_mod.Metrics()
        barrier = threading.Barrier(8)

        def _write(i: int) -> None:
            barrier.wait()
            m.record_stage_outcome("speaker_detection", i, "ran")

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(m.stage_outcomes_by_episode) == 8


class TestExtractEpisodeStageLedger:
    def test_builds_typed_outcomes(self) -> None:
        m = metrics_mod.Metrics()
        m.record_stage_outcome("speaker_detection", 1, "skipped", reason="dry_run")
        ledger = _extract_episode_stage_ledger(m, 1)
        assert ledger is not None
        assert isinstance(ledger["speaker_detection"], StageOutcome)
        assert ledger["speaker_detection"].reason == "dry_run"

    def test_returns_none_when_nothing_recorded(self) -> None:
        assert _extract_episode_stage_ledger(metrics_mod.Metrics(), 1) is None

    def test_returns_none_for_missing_metrics_or_index(self) -> None:
        assert _extract_episode_stage_ledger(None, 1) is None
        assert _extract_episode_stage_ledger(metrics_mod.Metrics(), None) is None

    def test_a_malformed_record_does_not_discard_the_others(self) -> None:
        """One bad entry must not cost the whole ledger — the rest is still the signal."""
        m = metrics_mod.Metrics()
        m.record_stage_outcome("speaker_detection", 1, "ran")
        m.stage_outcomes_by_episode[1]["broken"] = {"outcome": "not-a-valid-outcome"}
        m.stage_outcomes_by_episode[1]["no_outcome_key"] = {"reason": "x"}
        ledger = _extract_episode_stage_ledger(m, 1)
        assert ledger is not None
        assert set(ledger) == {"speaker_detection"}


class TestSizeGateCarriesItsReason:
    """The gate's decision used to survive while its reason did not (#1647)."""

    def test_no_skip_when_provider_is_not_size_limited(self) -> None:
        cfg = _Cfg(transcription_provider="whisper")
        result = _size_skip(cfg, _Episode(media_url="https://x/a.mp3"))
        assert result.skip_speaker_detection is False
        assert result.reason is None

    def test_oversize_media_is_advisory_and_does_not_disable_speaker_detection(
        self, monkeypatch
    ) -> None:
        """#1646: the size gate reports, it no longer gates.

        Speaker detection reads the episode title and description, so the media file's size
        is irrelevant to it. The probe still runs and still records what it saw — an operator
        wants to know chunking is coming (#557) — but ``skip_speaker_detection`` stays False.
        """
        episode = _Episode(media_url="https://example.com/big.mp3")
        response = MagicMock()
        response.headers = {"Content-Length": str(40 * 1024 * 1024)}
        monkeypatch.setattr(processing, "http_head", lambda *a, **k: response)

        result = _size_skip(_Cfg(), episode)

        assert result.skip_speaker_detection is False
        assert result.skip_episode is False
        assert result.reason is None
        # The observation is still recorded, just decoupled from the skip decision.
        assert result.media_oversize is True
        assert result.detail is not None
        assert result.detail["published_media_bytes"] == 40 * 1024 * 1024
        assert result.detail["limit_bytes"] == 25 * 1024 * 1024

    def test_under_the_limit_is_not_a_skip(self, monkeypatch) -> None:
        episode = _Episode(media_url="https://example.com/small.mp3")
        response = MagicMock()
        # 24.88 MB — the real margin observed on a The Journal. episode that correctly ran.
        response.headers = {"Content-Length": str(26_089_000)}
        monkeypatch.setattr(processing, "http_head", lambda *a, **k: response)

        result = _size_skip(_Cfg(), episode)

        assert result.skip_speaker_detection is False
        assert result.reason is None
