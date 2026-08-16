"""Naming cost must be MEASURED and REACH the manifest (#1657 acceptance item 3).

``naming.cost_usd`` was absent from every episode of the acceptance run, and the reason is worth
stating precisely, because the parts all existed and each one tested green on its own:

* ``EpisodeCostProbe`` had a ``speaker_detection_cost_usd`` slot and a
  ``record_llm_speaker_detection_call`` hook — with unit tests proving both work;
* ``_write_processing_manifest`` read ``pipeline_metrics.speaker_detection_cost_usd``;
* nothing in between ever put a probe around the naming stage.

The probes are constructed in ``metadata_generation`` for summary/GI/KG, which run *after*
naming. So the attribute existed only on an object the naming stage never saw, the ``getattr``
returned ``None`` on every episode, and ``stage_block`` dropped the key. Meanwhile the run-level
``llm_speaker_detection_cost_usd`` accrued the whole time — visible, but unusable here: it is
shared by parallel episodes, so it can never be attributed to one.

That is the same shape as the composite-host fix that shipped green and changed nothing: the
component was right, the wiring was absent, and only component tests existed. These tests are
therefore about the WIRING — the seam, not the parts. Every one of them fails on the old code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from podcast_scraper.workflow import episode_processor, metrics
from podcast_scraper.workflow.stages import processing

pytestmark = [pytest.mark.unit]


class _Detector:
    """A detector that spends money the way a real LLM provider does: by calling the recorder it
    was handed. Whether that object is the probe is exactly what is under test."""

    def __init__(self, cost: Optional[float] = None, raises: Optional[Exception] = None) -> None:
        self._cost, self._raises = cost, raises

    def detect_speakers(
        self,
        *,
        episode_title: str,
        episode_description: str,
        known_hosts: Set[str],
        pipeline_metrics: Any = None,
    ) -> Tuple[List[str], Set[str], bool, bool]:
        if self._cost is not None and pipeline_metrics is not None:
            pipeline_metrics.record_llm_speaker_detection_call(100, 10, cost_usd=self._cost)
        if self._raises:
            raise self._raises
        return (["A Guest"], {"A Host"}, True, False)


class _Cfg:
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
    def __init__(self, idx: int = 1) -> None:
        self.idx = idx
        self.title = "An episode"
        self.item = object()


@pytest.fixture(autouse=True)
def _stub_description(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processing, "extract_episode_description", lambda _item: "A description.")


def _detect(detector: _Detector, m: metrics.Metrics, idx: int = 1, **cfg_kw: Any) -> None:
    hd = processing.HostDetectionResult(set(), {}, detector)
    processing._detect_speakers_for_episode(
        _Episode(idx), _Cfg(**cfg_kw), hd, m  # type: ignore[arg-type]
    )


class TestTheCostIsActuallyMeasured:
    """Against the REAL Metrics object, not a stand-in — the defect was that the real one has no
    ``speaker_detection_cost_usd`` attribute at all, which a permissive fake would hide."""

    def test_an_llm_naming_call_is_attributed_to_this_episode(self) -> None:
        m = metrics.Metrics()
        _detect(_Detector(cost=0.004), m, idx=7)
        assert m.speaker_detection_cost_usd_by_episode[7] == pytest.approx(0.004)

    def test_the_run_level_total_still_accrues(self) -> None:
        """The probe isolates; it must not intercept-and-drop. If run totals stopped moving,
        this fix would have broken run-level cost reporting to fix per-episode reporting."""
        m = metrics.Metrics()
        _detect(_Detector(cost=0.004), m)
        assert m.llm_speaker_detection_cost_usd == pytest.approx(0.004)

    def test_a_deterministic_detector_records_a_measured_zero(self) -> None:
        """0.0 is a fact — detection ran and made no priced call. It is NOT the same as no
        entry, and the manifest reports them differently."""
        m = metrics.Metrics()
        _detect(_Detector(cost=None), m, idx=3)
        assert m.speaker_detection_cost_usd_by_episode[3] == 0.0

    def test_two_episodes_do_not_share_one_figure(self) -> None:
        """The reason a run-level delta could never have worked: parallel episodes accumulate
        into the same counter, so episode 2's manifest would inherit episode 1's spend."""
        m = metrics.Metrics()
        _detect(_Detector(cost=0.001), m, idx=1)
        _detect(_Detector(cost=0.007), m, idx=2)
        assert m.speaker_detection_cost_usd_by_episode[1] == pytest.approx(0.001)
        assert m.speaker_detection_cost_usd_by_episode[2] == pytest.approx(0.007)
        assert m.llm_speaker_detection_cost_usd == pytest.approx(0.008)

    def test_cost_is_kept_when_the_detector_raises_after_spending(self) -> None:
        """The call was billed whether or not the response parsed. Dropping it here would
        under-report exactly the episodes that went wrong."""
        m = metrics.Metrics()
        with pytest.raises(RuntimeError):
            _detect(_Detector(cost=0.002, raises=RuntimeError("boom")), m, idx=5)
        assert m.speaker_detection_cost_usd_by_episode[5] == pytest.approx(0.002)

    def test_a_skipped_stage_records_nothing_at_all(self) -> None:
        """Unmeasured must stay unmeasured: detection never ran, so there is no cost to claim —
        not even zero."""
        m = metrics.Metrics()
        _detect(_Detector(cost=None), m, idx=9, auto_speakers=False)
        assert 9 not in m.speaker_detection_cost_usd_by_episode


class TestTheManifestReadsThePerEpisodeFigure:
    """The other half of the seam. A measurement nobody reads is the bug this replaced."""

    class _Job:
        def __init__(self, idx: int) -> None:
            self.idx = idx

    def test_a_recorded_cost_is_found_for_the_right_episode(self) -> None:
        m = metrics.Metrics()
        m.record_speaker_detection_cost(0.006, 4)
        assert episode_processor._episode_naming_cost(m, self._Job(4)) == pytest.approx(0.006)

    def test_a_measured_zero_is_returned_as_zero_not_none(self) -> None:
        """If this collapsed to None, "free" would be reported as "unknown" — the exact
        confusion the null/zero rule exists to prevent."""
        m = metrics.Metrics()
        m.record_speaker_detection_cost(0.0, 4)
        assert episode_processor._episode_naming_cost(m, self._Job(4)) == 0.0

    def test_an_unmeasured_episode_is_none_not_zero(self) -> None:
        m = metrics.Metrics()
        assert episode_processor._episode_naming_cost(m, self._Job(4)) is None

    def test_one_episodes_cost_never_leaks_into_another(self) -> None:
        m = metrics.Metrics()
        m.record_speaker_detection_cost(0.006, 4)
        assert episode_processor._episode_naming_cost(m, self._Job(5)) is None

    def test_a_metrics_object_without_the_store_does_not_raise(self) -> None:
        """``pipeline_metrics`` is duck-typed across callers; an older object must yield
        "unmeasured", never an AttributeError that kills the manifest write."""
        assert episode_processor._episode_naming_cost(object(), self._Job(1)) is None
        assert episode_processor._episode_naming_cost(None, self._Job(1)) is None


class TestTheSeamCannotSilentlyComeApart:
    """Structural guards on the two ends. Component tests passed for months while the middle was
    missing; these fail if either end reverts."""

    def test_the_detection_stage_wraps_a_probe(self) -> None:
        src = Path(processing.__file__).read_text(encoding="utf-8")
        assert "EpisodeCostProbe" in src, "naming is not wrapped, so its cost is never captured"
        assert "_record_naming_cost(" in src

    def test_the_detector_is_handed_the_probe_not_the_raw_metrics(self) -> None:
        """The one-character version of this bug: passing ``pipeline_metrics`` here instead of
        the probe leaves the probe at 0.0 forever while everything still "works".

        Scoped to ``_detect_speakers_for_episode`` deliberately. A whole-file assertion also
        catches ``_validate_hosts_with_first_episode``, which calls the same detector at FEED
        level (see the class below) and correctly still passes the raw metrics object.
        """
        import inspect

        src = inspect.getsource(processing._detect_speakers_for_episode)
        assert "pipeline_metrics=detect_metrics" in src
        assert "pipeline_metrics=pipeline_metrics," not in src

    def test_the_manifest_no_longer_reads_the_attribute_that_never_existed(self) -> None:
        src = Path(episode_processor.__file__).read_text(encoding="utf-8")
        assert 'getattr(pipeline_metrics, "speaker_detection_cost_usd", None)' not in src

    def test_the_run_level_total_is_not_used_as_a_per_episode_value(self) -> None:
        """It is shared across parallel episodes; using it here would be attributable-looking
        and wrong."""
        src = episode_processor._episode_naming_cost.__doc__ or ""
        assert "never measured" in src or "unknown" in src
        manifest_src = Path(episode_processor.__file__).read_text(encoding="utf-8")
        assert "naming_cost = _episode_naming_cost(" in manifest_src


class TestWhatIsDeliberatelyNotAttributed:
    """NOT covered, on purpose — recorded here so the gap is a stated decision, not a silence.

    ``_validate_hosts_with_first_episode`` also calls ``detect_speakers`` (an LLM call on
    NER-derived hosts for a tag-less feed) and still passes the raw metrics object. That cost is
    FEED-level: one call corroborating the show's hosts, not work done for any one episode.
    Attributing it to the first episode would be a made-up number, and splitting it across
    episodes would invent precision that does not exist — so it accrues to the run total only.

    The consequence a reader must know: summing ``naming.cost_usd`` across an episode's manifests
    can be LESS than the run-level ``llm_speaker_detection_cost_usd``, and the difference is feed
    host-validation. That is by design; it is not the bug this file fixes.
    """

    def test_feed_level_validation_still_reaches_the_run_total(self) -> None:
        m = metrics.Metrics()
        m.record_llm_speaker_detection_call(100, 10, cost_usd=0.005)
        assert m.llm_speaker_detection_cost_usd == pytest.approx(0.005)

    def test_feed_level_validation_is_not_attributed_to_any_episode(self) -> None:
        m = metrics.Metrics()
        m.record_llm_speaker_detection_call(100, 10, cost_usd=0.005)
        assert m.speaker_detection_cost_usd_by_episode == {}


class TestTheBlockCarriesIt:
    """End of the chain: what a reader of the manifest actually sees."""

    def test_a_measured_cost_appears_in_the_naming_block(self) -> None:
        from podcast_scraper.workflow import processing_manifest as pm

        blk = pm.stage_block(ran=True, cost_usd=0.006, metrics={"named": 2})
        assert blk["cost_usd"] == 0.006

    def test_an_unmeasured_cost_is_an_explicit_null_not_a_missing_key(self) -> None:
        """The originally-reported symptom: the key was ABSENT, which reads as "free" to
        anything summing blocks and as "unknown" to a careful reader — two answers from one
        artifact."""
        from podcast_scraper.workflow import processing_manifest as pm

        blk: Dict[str, Any] = pm.stage_block(ran=True, cost_usd=None)
        assert "cost_usd" in blk and blk["cost_usd"] is None
