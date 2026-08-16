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

from typing import Any, Dict, List, Optional
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
    return processing._detect_speakers_for_episode(episode, cfg, host_result, m, **kwargs)


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


class TestDetectSpeakersRecordsEveryExit:
    """Each early return used to be silent. Silence is the bug."""

    @staticmethod
    def _run(cfg: _Cfg, m: metrics_mod.Metrics, **kwargs: Any) -> Optional[Any]:
        host_result = MagicMock()
        host_result.speaker_detector = None
        host_result.cached_hosts = set()
        return _detect(_Episode(), cfg, host_result, m, **kwargs)

    def test_skip_requested_by_caller_records_the_reason_it_was_given(self) -> None:
        m = metrics_mod.Metrics()
        self._run(
            _Cfg(),
            m,
            skip_speaker_detection=True,
            skip_reason="media_over_size_limit_no_transcript_urls",
            skip_detail={"media_bytes": 1},
        )
        record = _ledger_for(m)["speaker_detection"]
        assert record["outcome"] == "skipped"
        assert record["reason"] == "media_over_size_limit_no_transcript_urls"
        assert record["detail"] == {"media_bytes": 1}

    def test_skip_without_a_supplied_reason_still_records_one(self) -> None:
        """An unattributed skip is still a skip — it must never be recorded as nothing."""
        m = metrics_mod.Metrics()
        self._run(_Cfg(), m, skip_speaker_detection=True)
        record = _ledger_for(m)["speaker_detection"]
        assert record["outcome"] == "skipped"
        assert record["reason"] == "skip_requested_by_caller"

    def test_auto_speakers_disabled_is_recorded(self) -> None:
        m = metrics_mod.Metrics()
        self._run(_Cfg(auto_speakers=False), m)
        assert _ledger_for(m)["speaker_detection"]["reason"] == "auto_speakers_disabled"

    def test_auto_speakers_disabled_with_configured_names_is_distinguished(self) -> None:
        m = metrics_mod.Metrics()
        self._run(_Cfg(auto_speakers=False, screenplay_speaker_names=["Host", "Guest"]), m)
        record = _ledger_for(m)["speaker_detection"]
        assert record["reason"] == "auto_speakers_disabled_using_configured_names"

    def test_dry_run_is_recorded(self) -> None:
        m = metrics_mod.Metrics()
        self._run(_Cfg(dry_run=True), m)
        assert _ledger_for(m)["speaker_detection"]["reason"] == "dry_run"

    def test_missing_detector_is_recorded_with_the_configured_provider(self, monkeypatch) -> None:
        monkeypatch.setattr(processing, "_get_speaker_detector", lambda *a, **k: None)
        m = metrics_mod.Metrics()
        self._run(_Cfg(speaker_detector_provider="litellm"), m)
        record = _ledger_for(m)["speaker_detection"]
        assert record["outcome"] == "skipped"
        assert record["reason"] == "no_speaker_detector_available"
        assert record["detail"] == {"speaker_detector_provider": "litellm"}

    def test_successful_detection_records_ran_with_proposal_counts(self, monkeypatch) -> None:
        detector = MagicMock()
        detector.detect_speakers.return_value = (["Sarah Sachs", "Simon Last"], set(), True, False)
        monkeypatch.setattr(processing, "_get_speaker_detector", lambda *a, **k: detector)
        monkeypatch.setattr(processing, "extract_episode_description", lambda _item: "desc")
        monkeypatch.setattr(processing, "corroborate_guests", lambda proposed, **k: proposed[:1])

        m = metrics_mod.Metrics()
        self._run(_Cfg(), m)

        record = _ledger_for(m)["speaker_detection"]
        assert record["outcome"] == "ran"
        assert record["detail"]["proposed_count"] == 2
        assert record["detail"]["corroborated_count"] == 1
        assert record["duration_seconds"] is not None

    def test_detector_returning_nothing_is_ran_not_skipped(self, monkeypatch) -> None:
        """'Ran and found nothing' and 'never ran' are different facts about the episode.

        That intent is unchanged; the OUTCOME that expresses it was wrong. This asserted
        ``failed``, but ``detection_succeeded`` is ``bool(hosts or guests)`` — emptiness, not
        error — so nothing had failed. The real corpus showed the cost: Planet Money, whose feed
        states no hosts, recorded ``failed`` on a stage that had completed correctly, and
        ``stage_did_run`` then told the roster it never ran at all (#1657 acceptance).

        ``failed`` now means the detector raised; see
        ``test_speaker_detection_outcome_truthfulness.py``.
        """
        detector = MagicMock()
        detector.detect_speakers.return_value = ([], set(), False, False)
        monkeypatch.setattr(processing, "_get_speaker_detector", lambda *a, **k: detector)
        monkeypatch.setattr(processing, "extract_episode_description", lambda _item: "desc")

        m = metrics_mod.Metrics()
        self._run(_Cfg(), m)

        record = _ledger_for(m)["speaker_detection"]
        assert record["outcome"] == "ran"
        assert record["reason"] == "no_names_found_in_metadata"
        # The distinction the original test existed to protect, stated directly.
        assert record["outcome"] != "skipped"
        assert record["duration_seconds"] is not None

    def test_fallback_to_configured_names_is_degraded_not_ran(self, monkeypatch) -> None:
        detector = MagicMock()
        detector.detect_speakers.return_value = ([], set(), False, False)
        monkeypatch.setattr(processing, "_get_speaker_detector", lambda *a, **k: detector)
        monkeypatch.setattr(processing, "extract_episode_description", lambda _item: "desc")

        m = metrics_mod.Metrics()
        self._run(_Cfg(screenplay_speaker_names=["Host", "Guest"]), m)

        record = _ledger_for(m)["speaker_detection"]
        assert record["outcome"] == "degraded"
        assert record["reason"] == "detection_failed_using_configured_names"
