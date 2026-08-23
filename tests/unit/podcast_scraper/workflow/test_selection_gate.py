"""The gate that would have stopped the 2026-08-18 incident before it cost anything.

That run selected 678 episodes for a 32-episode job and spent ~$48 under an active $5 cap. Every
downstream check was too late by construction: ASR completes in a background thread before spend
is ever inspected. Selection is the last moment at which the run has cost nothing, so it is the
only place a refusal is free.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.cost_monitoring import CostCapExceeded
from podcast_scraper.workflow.run_budget import get_run_budget, reset_run_budget
from podcast_scraper.workflow.selection_gate import (
    affordable_episode_count,
    enforce_selection_budget,
    estimate_selection,
    RETRY_ALLOWANCE,
)

ITUNES_DURATION = "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration"
DEEPGRAM_PER_MIN = 0.0043  # the nova-3 pricing row the incident's bill matched


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_run_budget()
    yield
    reset_run_budget()


def _episode(duration_seconds: int | None):
    """An Episode-shaped object: the gate only ever reads ``.item``."""
    item = ET.Element("item")
    ET.SubElement(item, "guid").text = "g"
    ET.SubElement(item, "title").text = "t"
    if duration_seconds is not None:
        ET.SubElement(item, ITUNES_DURATION).text = str(duration_seconds)
    return SimpleNamespace(item=item, idx=1, title="t")


def _cfg(cap=5.0, action="abort"):
    return SimpleNamespace(
        transcription_provider="deepgram",
        deepgram_model="nova-3",
        cost_soft_cap_usd_per_run=cap,
        cost_soft_cap_action=action,
        output_dir="/tmp/corpus",
        # The real Config default. An empty/None value short-circuits pricing entirely
        # (helpers._get_provider_pricing returns {} before any fallback), so a test that
        # passed None would only ever exercise the unpriceable path.
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )


# -- estimation -------------------------------------------------------------------------------


def test_it_prices_a_selection_from_audio_duration() -> None:
    eps = [_episode(3600) for _ in range(3)]  # 3 hours
    est = estimate_selection(eps, _cfg(), available=678)
    assert est.selected == 3
    assert est.available == 678
    assert est.priced == 3
    assert est.unpriced == 0
    assert est.audio_hours == pytest.approx(3.0)
    assert est.asr_usd == pytest.approx(3 * 60 * DEEPGRAM_PER_MIN * RETRY_ALLOWANCE, rel=1e-3)


def test_the_estimate_carries_a_retry_allowance_because_actual_exceeded_it() -> None:
    """The incident billed 187.6 hours for ~155 corpus-hours; a bare duration sum under-predicts."""
    est = estimate_selection([_episode(3600)], _cfg())
    bare = 60 * DEEPGRAM_PER_MIN
    assert est.asr_usd is not None
    assert (
        est.asr_usd > bare
    ), "an estimate that under-predicts is the failure mode that costs money"
    assert est.asr_usd == pytest.approx(bare * RETRY_ALLOWANCE, rel=1e-3)


def test_unknown_duration_episodes_are_COUNTED_never_treated_as_free() -> None:
    eps = [_episode(3600), _episode(None), _episode(None)]
    est = estimate_selection(eps, _cfg())
    assert est.selected == 3
    assert est.priced == 1
    assert est.unpriced == 2
    assert est.fully_priced is False
    assert "NO known duration" in est.describe()
    assert "the real cost is higher" in est.describe()


def test_a_zero_or_negative_duration_is_unknown_not_free() -> None:
    for bad in (0, -60):
        est = estimate_selection([_episode(bad)], _cfg())
        assert est.unpriced == 1, f"duration {bad} must not price as $0"


def test_the_manifest_line_shows_the_denominator() -> None:
    """32-of-678 and 678-of-678 differ by operator attention and by $48."""
    line = estimate_selection([_episode(600) for _ in range(32)], _cfg(), available=678).describe()
    assert "32 of 678 episodes" in line
    assert "audio-hours" in line
    assert "est. $" in line


def test_an_unpriceable_provider_reports_unknown_rather_than_zero() -> None:
    cfg = _cfg()
    cfg.transcription_provider = "no-such-provider"
    est = estimate_selection([_episode(3600)], cfg)
    assert est.asr_usd is None
    assert "UNKNOWN" in est.describe()


def test_an_empty_selection_prices_at_nothing_without_erroring() -> None:
    est = estimate_selection([], _cfg(), available=678)
    assert est.selected == 0
    assert est.audio_hours == 0.0


# -- the refusal ------------------------------------------------------------------------------


def test_a_selection_within_the_cap_is_allowed() -> None:
    reset_run_budget(cap_usd=5.0, action="abort")
    est = enforce_selection_budget([_episode(600) for _ in range(10)], _cfg())
    assert est.selected == 10


def test_THE_INCIDENT_a_678_episode_selection_is_REFUSED_before_spending() -> None:
    """678 episodes averaging 51 minutes is ~$186 of ASR. It must never start."""
    reset_run_budget(cap_usd=5.0, action="abort")
    eps = [_episode(51 * 60) for _ in range(678)]
    with pytest.raises(CostCapExceeded):
        enforce_selection_budget(eps, _cfg(), available=678)
    assert get_run_budget().spent_usd == 0.0, "the refusal must cost nothing"


def test_the_refusal_says_how_many_WOULD_fit(caplog) -> None:
    """A stop the operator cannot act on is half a fix."""
    reset_run_budget(cap_usd=5.0, action="abort")
    eps = [_episode(60 * 60) for _ in range(100)]  # 1h each: ~$0.32 apiece with the allowance
    with caplog.at_level("ERROR"):
        with pytest.raises(CostCapExceeded):
            enforce_selection_budget(eps, _cfg(), available=100)
    text = caplog.text
    assert "REFUSING TO START" in text
    assert "would fit" in text
    assert "split the work-list" in text


def test_the_gate_is_CUMULATIVE_across_feeds_not_per_feed() -> None:
    """The scoping bug, at the gate. Fourteen affordable feeds are not an affordable batch."""
    reset_run_budget(cap_usd=5.0, action="abort")
    cfg = _cfg()
    feed = [_episode(30 * 60) for _ in range(4)]  # 2 audio-hours -> ~$0.65 per feed

    for _ in range(7):
        enforce_selection_budget(feed, cfg)
        get_run_budget().record(0.65)  # the feed then actually spends it

    # ~$4.55 spent. The eighth feed is individually trivial and must still be refused.
    with pytest.raises(CostCapExceeded):
        enforce_selection_budget(feed, cfg)


@pytest.mark.parametrize("action", ["warn", "observe"])
def test_warn_and_observe_report_but_never_refuse(action) -> None:
    reset_run_budget(cap_usd=5.0, action=action)
    eps = [_episode(51 * 60) for _ in range(678)]
    est = enforce_selection_budget(eps, _cfg(cap=5.0, action=action))
    assert est.selected == 678, "only abort stops work"


def test_no_cap_configured_means_no_refusal() -> None:
    reset_run_budget(cap_usd=None)
    est = enforce_selection_budget([_episode(51 * 60) for _ in range(678)], _cfg(cap=None))
    assert est.selected == 678


def test_an_unpriceable_selection_warns_LOUDLY_and_proceeds(caplog) -> None:
    """Refusing every run whose provider lacks a pricing row would ground the pipeline on a
    config gap rather than a cost problem — but silence would read as "it's free"."""
    reset_run_budget(cap_usd=5.0, action="abort")
    cfg = _cfg()
    cfg.transcription_provider = "no-such-provider"
    with caplog.at_level("WARNING"):
        est = enforce_selection_budget([_episode(3600) for _ in range(500)], cfg)
    assert est.selected == 500
    assert "could NOT be priced" in caplog.text


def test_an_empty_selection_never_refuses() -> None:
    """A feed holding none of the work-list's episodes is normal (the multi-feed case)."""
    reset_run_budget(cap_usd=0.01, action="abort")
    assert enforce_selection_budget([], _cfg(cap=0.01)).selected == 0


# -- "how many fit" ---------------------------------------------------------------------------


def test_affordable_count_is_the_number_that_actually_fits() -> None:
    cfg = _cfg()
    eps = [_episode(60 * 60) for _ in range(50)]  # ~$0.3225 each with the allowance
    n = affordable_episode_count(eps, cfg, remaining_usd=1.0)
    assert n == 3, f"3 x 0.3225 = 0.97 fits, 4 x = 1.29 does not; got {n}"
    est = estimate_selection(eps[:n], cfg)
    assert est.asr_usd is not None and est.asr_usd <= 1.0


def test_affordable_count_is_zero_when_even_one_episode_is_too_expensive() -> None:
    assert affordable_episode_count([_episode(3 * 3600)], _cfg(), remaining_usd=0.01) == 0


def test_affordable_count_is_everything_when_uncapped() -> None:
    eps = [_episode(3600) for _ in range(9)]
    assert affordable_episode_count(eps, _cfg(), remaining_usd=float("inf")) == 9


def test_affordable_count_stops_at_an_unpriceable_episode() -> None:
    """An episode whose duration is unknown cannot be shown to fit, so it bounds the answer."""
    eps = [_episode(600), _episode(600), _episode(None), _episode(600)]
    assert affordable_episode_count(eps, _cfg(), remaining_usd=100.0) == 2


# -- self-hosted ASR: priced at zero, not "unpriceable" ----------------------------------------


def _local_cfg(provider="whisper", cap=5.0):
    return SimpleNamespace(
        transcription_provider=provider,
        whisper_model="base",
        cost_soft_cap_usd_per_run=cap,
        cost_soft_cap_action="abort",
        output_dir="/tmp/corpus",
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )


@pytest.mark.parametrize("provider", ["whisper", "tailnet_dgx_whisper", "moss"])
def test_self_hosted_ASR_is_priced_at_zero_not_reported_as_unpriceable(provider) -> None:
    """Found by running a real local corpus through the gate, not by reading the code.

    whisper has no pricing row because it runs on our own hardware, and the gate treated that
    identically to a CLOUD provider whose price could not be resolved — warning that "the cap
    cannot be applied". Alarming and correct for the latter; alarming and WRONG for the former,
    where there is no bill to cap.
    """
    est = estimate_selection([_episode(3600) for _ in range(40)], _local_cfg(provider))
    assert est.self_hosted is True
    assert est.asr_usd == 0.0
    assert "no per-call charge (self-hosted ASR)" in est.describe()
    assert "UNKNOWN" not in est.describe()


def test_a_self_hosted_run_is_never_refused_however_large(caplog) -> None:
    reset_run_budget(cap_usd=0.01, action="abort")
    with caplog.at_level("INFO"):
        est = enforce_selection_budget(
            [_episode(3 * 3600) for _ in range(500)], _local_cfg(cap=0.01)
        )
    assert est.selected == 500
    assert "self-hosted" in caplog.text
    assert "could NOT be priced" not in caplog.text, "the misleading warning must be gone"


def test_a_CLOUD_provider_with_no_price_still_warns() -> None:
    """The self-hosted carve-out must not silence the case it was carved out of."""
    est = estimate_selection([_episode(3600)], _local_cfg(provider="some-unpriced-cloud-vendor"))
    assert est.self_hosted is False
    assert est.asr_usd is None
    assert "UNKNOWN" in est.describe()
