"""The ledger must say what actually happened to speaker detection (#1647, #1657 acceptance).

Found on the real corpus, not in a test. Planet Money's episode recorded::

    stage_ledger.speaker_detection.outcome = "failed"
    diagnostics.detection_stage_ran        = false

Both were untrue. Nothing raised — the detector ran for 1.6s, read the episode metadata, and
correctly found no names, because that feed states no hosts and NER over a description is
deliberately not used (#876: it returns the people an episode is ABOUT, which is how an
advertiser's name once became a host).

The mechanism was a misread flag. In ``speaker_detectors/detection.py``::

    detection_succeeded = bool(hosts or guests)

That is an EMPTINESS flag, not an error flag. Treating it as failure cost two things:

* a corpus report grouping by outcome showed every host-less show as a permanent failure with
  nothing to fix;
* ``stage_did_run`` returns ``outcome in ("ran", "degraded")``, so ``failed`` told the roster
  the stage never ran — collapsing "measured, and this voice cannot be named" into "never
  measured", which is precisely the distinction #1647 exists to preserve.

These tests pin the four outcomes to what they mean, and pin the two artifacts to each other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from podcast_scraper.workflow.stages import processing

pytestmark = [pytest.mark.unit]


class _Metrics:
    """Captures ledger writes and answers ``stage_did_run`` the way the real Metrics does."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.times: List[float] = []

    def record_stage_outcome(
        self,
        stage: str,
        episode_idx: int,
        outcome: str,
        reason: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        self.records.append(
            {
                "stage": stage,
                "outcome": outcome,
                "reason": reason,
                "detail": detail,
                "duration_seconds": duration_seconds,
            }
        )

    def record_extract_names_time(self, duration: float, episode_idx: int) -> None:
        self.times.append(duration)

    def stage_did_run(self, stage: str, episode_idx: int) -> Optional[bool]:
        """Mirrors metrics.stage_did_run — the roster's view of the same ledger."""
        for r in reversed(self.records):
            if r["stage"] == stage:
                return r["outcome"] in ("ran", "degraded")
        return None


class _Detector:
    def __init__(self, hosts: Set[str], guests: List[str], raises: Optional[Exception] = None):
        self._hosts, self._guests, self._raises = hosts, guests, raises

    def detect_speakers(
        self, *, episode_title: str, episode_description: str, known_hosts: Set[str]
    ) -> Tuple[List[str], Set[str], bool, bool]:
        if self._raises:
            raise self._raises
        # The real contract: succeeded is bool(hosts or guests) — emptiness, not error.
        return (list(self._guests), set(self._hosts), bool(self._hosts or self._guests), False)


class _Cfg:
    """Only the attributes the function actually reads (verified against the source)."""

    def __init__(self, **kw: Any) -> None:
        self.auto_speakers = True
        self.known_hosts: List[str] = []
        self.cache_detected_hosts = False
        self.screenplay_speaker_names: List[str] = []
        self.speaker_detector_provider = "litellm"
        self.dry_run = False
        for k, v in kw.items():
            setattr(self, k, v)


class _Episode:
    idx = 1
    title = "You bet your life insurance"
    item = object()  # a real RSS element; description extraction is stubbed below


@pytest.fixture(autouse=True)
def _stub_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extract_episode_description`` parses an XML element (``item.find(...)``).

    These tests are about which OUTCOME gets recorded, not about description parsing, so the
    extractor is stubbed rather than faked with a synthetic XML tree — a fake tree would test
    my mock instead of the code under test.
    """
    monkeypatch.setattr(
        processing, "extract_episode_description", lambda _item: "A story about life insurance."
    )


def _run(detector: _Detector, cfg: Optional[_Cfg] = None) -> _Metrics:
    """Call with the REAL signature: (episode, cfg, host_detection_result, pipeline_metrics)."""
    m = _Metrics()
    hd = processing.HostDetectionResult(set(), {}, detector)
    processing._detect_speakers_for_episode(
        _Episode(), cfg or _Cfg(), hd, m  # type: ignore[arg-type]
    )
    return m


class TestAnEmptyResultIsRanNotFailed:
    """Planet Money: the detector completed and correctly found nobody."""

    def test_outcome_is_ran(self) -> None:
        m = _run(_Detector(hosts=set(), guests=[]))
        rec = [r for r in m.records if r["stage"] == "speaker_detection"][-1]
        assert rec["outcome"] == "ran", f"an empty result is not a failure: {rec}"

    def test_the_reason_still_says_it_found_nothing(self) -> None:
        """The outcome changes; the diagnostic detail must not be lost."""
        m = _run(_Detector(hosts=set(), guests=[]))
        rec = [r for r in m.records if r["stage"] == "speaker_detection"][-1]
        assert rec["reason"] == "no_names_found_in_metadata"

    def test_duration_is_recorded(self) -> None:
        """It ran, so it took time — a stage that "never ran" would have none."""
        m = _run(_Detector(hosts=set(), guests=[]))
        rec = [r for r in m.records if r["stage"] == "speaker_detection"][-1]
        assert rec["duration_seconds"] is not None and rec["duration_seconds"] >= 0

    def test_the_roster_is_told_it_ran(self) -> None:
        """The bug that mattered: stage_did_run drives diagnostics.detection_stage_ran, so a
        wrong outcome told the roster an unnameable voice was merely unmeasured."""
        m = _run(_Detector(hosts=set(), guests=[]))
        assert m.stage_did_run("speaker_detection", 1) is True


class TestARealFailureIsStillFailed:
    """``failed`` keeps its meaning — otherwise the fix would just move the lie."""

    def test_a_raising_detector_records_failed(self) -> None:
        det = _Detector(set(), [], raises=RuntimeError("provider exploded"))
        m = _Metrics()
        hd = processing.HostDetectionResult(set(), {}, det)
        with pytest.raises(RuntimeError):
            processing._detect_speakers_for_episode(
                _Episode(), _Cfg(), hd, m  # type: ignore[arg-type]
            )
        rec = [r for r in m.records if r["stage"] == "speaker_detection"][-1]
        assert rec["outcome"] == "failed"
        assert rec["reason"] == "detector_raised"
        assert rec["detail"]["exception"] == "RuntimeError"

    def test_the_exception_still_propagates(self) -> None:
        """Recording must not swallow the error — control flow is unchanged on purpose."""
        det = _Detector(set(), [], raises=ValueError("boom"))
        hd = processing.HostDetectionResult(set(), {}, det)
        with pytest.raises(ValueError):
            processing._detect_speakers_for_episode(
                _Episode(), _Cfg(), hd, _Metrics()  # type: ignore[arg-type]
            )

    def test_a_raise_is_not_reported_as_ran(self) -> None:
        det = _Detector(set(), [], raises=RuntimeError("x"))
        m = _Metrics()
        hd = processing.HostDetectionResult(set(), {}, det)
        with pytest.raises(RuntimeError):
            processing._detect_speakers_for_episode(
                _Episode(), _Cfg(), hd, m  # type: ignore[arg-type]
            )
        assert m.stage_did_run("speaker_detection", 1) is False


class TestTheOtherOutcomesAreUnchanged:
    def test_a_successful_detection_is_ran(self) -> None:
        m = _run(_Detector(hosts={"Kevin Roose"}, guests=["Casey Newton"]))
        rec = [r for r in m.records if r["stage"] == "speaker_detection"][-1]
        assert rec["outcome"] == "ran"
        assert m.stage_did_run("speaker_detection", 1) is True

    def test_configured_names_standing_in_is_degraded(self) -> None:
        """Still 'degraded', not 'ran': real names were not found, substitutes were used."""
        cfg = _Cfg(screenplay_speaker_names=["Host", "Guest"])
        m = _run(_Detector(hosts=set(), guests=[]), cfg)
        rec = [r for r in m.records if r["stage"] == "speaker_detection"][-1]
        assert rec["outcome"] == "degraded"
        assert rec["reason"] == "detection_failed_using_configured_names"
        # degraded counts as having run — it did work, just with a fallback.
        assert m.stage_did_run("speaker_detection", 1) is True


class TestTheVocabularyStaysClosed:
    """ADR-151: outcome is one of ran / skipped / failed / degraded. Anything else breaks
    every GROUP BY built on it."""

    @pytest.mark.parametrize(
        "hosts,guests,names",
        [(set(), [], []), ({"A B"}, [], []), (set(), ["C D"], []), (set(), [], ["H", "G"])],
    )
    def test_every_path_emits_a_known_outcome(
        self, hosts: Set[str], guests: List[str], names: List[str]
    ) -> None:
        m = _run(_Detector(hosts, guests), _Cfg(screenplay_speaker_names=names))
        for r in m.records:
            assert r["outcome"] in ("ran", "skipped", "failed", "degraded"), r
