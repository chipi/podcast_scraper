"""``cost_usd``: 0.0 means measured and free; null means nobody measured (#1657 acceptance).

The acceptance run produced a manifest where two locally-run, genuinely-free stages disagreed::

    "diarization": {"cost_usd": 0.0}
    "naming":      {"cost_usd": null}

Both ran locally. Both cost nothing. ADR-132 had specified ``None`` for local diarization
("a truthful 'no billed cost', not a fabricated zero") while the code emitted ``0.0`` — so the
document and the implementation disagreed, and the implementation disagreed with itself.

The corrected rule, applied identically everywhere:

* **0.0** — the stage ran and its cost is known to be zero (local engine, no invoice).
* **null** — nobody measured it. That is a real and different fact, and keeping it meaningful
  is the whole point: a fabricated zero on an uninstrumented stage is how a roll-up silently
  under-reports.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from podcast_scraper.workflow import processing_manifest as pm

pytestmark = [pytest.mark.unit]


class TestMeasuredOrUnmeasured:
    def test_a_real_measurement_always_wins(self) -> None:
        assert (
            pm.measured_or_unmeasured(0.1716, "deepgram", pm.LOCAL_TRANSCRIPTION_PROVIDERS)
            == 0.1716
        )

    def test_a_local_engine_is_a_measured_zero(self) -> None:
        assert pm.measured_or_unmeasured(None, "whisper", pm.LOCAL_TRANSCRIPTION_PROVIDERS) == 0.0
        assert (
            pm.measured_or_unmeasured(None, "tailnet_dgx_whisper", pm.LOCAL_TRANSCRIPTION_PROVIDERS)
            == 0.0
        )
        assert pm.measured_or_unmeasured(None, "pyannote", pm.LOCAL_DIARIZATION_PROVIDERS) == 0.0

    def test_an_unmeasured_cloud_stage_stays_null(self) -> None:
        """The distinction that has to survive: a paid provider with no recorded cost means we
        failed to measure, NOT that it was free."""
        assert pm.measured_or_unmeasured(None, "deepgram", pm.LOCAL_TRANSCRIPTION_PROVIDERS) is None
        assert pm.measured_or_unmeasured(None, None, pm.LOCAL_TRANSCRIPTION_PROVIDERS) is None

    def test_a_measured_zero_from_a_cloud_provider_is_kept(self) -> None:
        """Deepgram's precomputed diarization genuinely adds no charge — 0.0, not null."""
        assert pm.measured_or_unmeasured(0.0, "deepgram", pm.LOCAL_DIARIZATION_PROVIDERS) == 0.0

    def test_provider_matching_is_case_insensitive(self) -> None:
        assert pm.measured_or_unmeasured(None, "PyAnnote", pm.LOCAL_DIARIZATION_PROVIDERS) == 0.0


class TestTheZeroSurvivesSerialisation:
    """Cost has exactly THREE states in a block, and all three must be readable.

    ``stage_block`` drops keys that are ``None`` — deliberate for most fields, where absence
    honestly means "no stage owns this". Cost is different: 0.0 (measured, free) and null (not
    measured) are already distinct claims, so dropping the key invented a fourth state that a
    reader can only guess at, and which reads as "free" to anything summing the blocks. That is
    how naming's spend stayed invisible on every episode of the acceptance run.

    This class previously asserted the opposite (``test_none_is_omitted``). The behaviour changed
    on purpose, to the rule the operator set: null = we did not measure, zero = we measured and it
    was zero.
    """

    def test_zero_is_written_not_omitted(self) -> None:
        blk = pm.stage_block(ran=True, cost_usd=0.0)
        assert "cost_usd" in blk and blk["cost_usd"] == 0.0

    def test_none_is_written_as_an_explicit_null(self) -> None:
        blk = pm.stage_block(ran=True, cost_usd=None)
        assert "cost_usd" in blk, "an absent key is a fourth state that means nothing"
        assert blk["cost_usd"] is None

    def test_the_three_states_are_distinguishable_after_a_json_round_trip(self) -> None:
        """The manifest is read back as JSON; the distinction has to survive that, not just
        live in the dict."""
        import json

        unmeasured = json.loads(json.dumps(pm.stage_block(ran=True, cost_usd=None)))
        free = json.loads(json.dumps(pm.stage_block(ran=True, cost_usd=0.0)))
        paid = json.loads(json.dumps(pm.stage_block(ran=True, cost_usd=0.0123)))
        assert unmeasured["cost_usd"] is None
        assert free["cost_usd"] == 0.0
        assert paid["cost_usd"] == 0.0123
        assert all("cost_usd" in b for b in (unmeasured, free, paid))

    def test_a_summing_reader_can_tell_free_from_unknown(self) -> None:
        """The consequence, stated as the reader sees it: totalling costs must be able to
        exclude unmeasured stages instead of silently counting them as zero."""
        blocks = [
            pm.stage_block(ran=True, cost_usd=0.02),
            pm.stage_block(ran=True, cost_usd=0.0),
            pm.stage_block(ran=True, cost_usd=None),
        ]
        assert sum(b["cost_usd"] for b in blocks if b["cost_usd"] is not None) == 0.02
        assert any(b["cost_usd"] is None for b in blocks)


class TestSpeakerDetectionCostIsCaptured:
    """Naming is not free by definition — ``cloud_balanced`` resolves voices with an LLM.

    The probe intercepted summary/gi/kg but not speaker detection, so every episode's manifest
    under-reported by that stage's spend and ``cost_usd_total`` inherited the gap.
    """

    class _Inner:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def record_llm_speaker_detection_call(
            self, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
        ) -> str:
            self.calls.append((input_tokens, output_tokens, cost_usd))
            return "forwarded"

    def test_it_accumulates_this_episodes_cost(self) -> None:
        probe = pm.EpisodeCostProbe(self._Inner())
        probe.record_llm_speaker_detection_call(100, 10, cost_usd=0.002)
        probe.record_llm_speaker_detection_call(50, 5, cost_usd=0.001)
        assert probe.speaker_detection_cost_usd == pytest.approx(0.003)

    def test_it_still_forwards_to_the_run_level_recorder(self) -> None:
        """Run totals must stay correct — the probe isolates, it does not intercept-and-drop."""
        inner = self._Inner()
        probe = pm.EpisodeCostProbe(inner)
        assert probe.record_llm_speaker_detection_call(100, 10, cost_usd=0.002) == "forwarded"
        assert inner.calls == [(100, 10, 0.002)]

    def test_no_llm_call_leaves_a_measured_zero(self) -> None:
        """The deterministic naming path costs nothing, and 0.0 says so honestly."""
        probe = pm.EpisodeCostProbe(self._Inner())
        assert probe.speaker_detection_cost_usd == 0.0
