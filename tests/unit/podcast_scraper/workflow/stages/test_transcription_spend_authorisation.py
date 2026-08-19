"""ASR must be refused BEFORE the call, because after it there is nothing left to stop.

This is the root cause of the 2026-08-18 incident. ``stages/transcription.py`` contained no cost
check of any kind; all ASR runs in a background thread, and the first enforcement point that sees
transcription spend is orchestration's ``check_cost_soft_cap_at_stage``, which runs only after
``transcription_thread.join()``. So every episode in a feed was transcribed and billed before the
cap could look at anything — one feed reached $9.63 under a $5 cap.

ASR is the one paid stage whose cost is knowable in advance (price x audio duration), which is
what makes a per-call authorisation possible here and not for the LLM stages.

THE OTHER HALF is how a refusal travels. It must NOT raise: an exception in this worker kills only
this thread while the main thread sits in ``join()`` and never learns — the wedge that
``test_processing_supervision.py`` exists to prevent (processing.py:2022-2029). Refusal returns a
normal "did not transcribe" result and latches the ledger; the main thread converts that latch
into ``CostCapExceeded``. Both halves are tested.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.cost_monitoring import CostCapExceeded, enforce_cost_soft_cap
from podcast_scraper.workflow.run_budget import get_run_budget, reset_run_budget
from podcast_scraper.workflow.stages.transcription import _authorise_transcription_spend

pytestmark = [pytest.mark.unit]

DEEPGRAM_PER_MIN = 0.0043


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_run_budget()
    yield
    reset_run_budget()


def _cfg(cap=5.0, action="abort", provider="deepgram"):
    return SimpleNamespace(
        transcription_provider=provider,
        deepgram_model="nova-3",
        whisper_model="base",
        cost_soft_cap_usd_per_run=cap,
        cost_soft_cap_action=action,
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )


def _job(idx: int = 1, duration_seconds: float | None = 3600.0):
    """A TranscriptionJob-shaped object. The guard reads idx and the duration accessor."""
    return SimpleNamespace(idx=idx, episode=None, episode_duration_seconds=duration_seconds)


# -- the authorisation decision ----------------------------------------------------------------


def test_an_affordable_episode_is_authorised() -> None:
    reset_run_budget(cap_usd=5.0, action="abort")
    assert _authorise_transcription_spend(_job(), _cfg()) is True


def test_an_episode_that_would_BREACH_the_cap_is_refused() -> None:
    reset_run_budget(cap_usd=5.0, action="abort")
    get_run_budget().record(4.9)
    assert _authorise_transcription_spend(_job(duration_seconds=3600.0), _cfg()) is False


def test_THE_INCIDENT_a_long_feed_stops_partway_instead_of_transcribing_all_of_it() -> None:
    """30 one-hour episodes under a $5 cap: the run must stop, not reach $9.63."""
    reset_run_budget(cap_usd=5.0, action="abort")
    cfg = _cfg()
    transcribed = 0
    for i in range(30):
        if not _authorise_transcription_spend(_job(idx=i), cfg):
            break
        # the episode is then transcribed and the real cost recorded
        get_run_budget().record(60 * DEEPGRAM_PER_MIN)
        transcribed += 1

    assert transcribed < 30, "the whole feed was transcribed — the cap did nothing"
    assert get_run_budget().spent_usd <= 5.0
    assert get_run_budget().tripped is True


def test_once_refused_it_stops_scheduling_WITHOUT_re_pricing_every_remaining_episode() -> None:
    reset_run_budget(cap_usd=5.0, action="abort")
    get_run_budget().record(99.0)
    cfg = _cfg()
    assert _authorise_transcription_spend(_job(1), cfg) is False
    # every subsequent episode is refused on the latch alone
    for i in range(2, 10):
        assert _authorise_transcription_spend(_job(i), cfg) is False


def test_the_refusal_tells_the_operator_the_numbers(caplog) -> None:
    reset_run_budget(cap_usd=5.0, action="abort")
    get_run_budget().record(4.99)
    with caplog.at_level("ERROR"):
        _authorise_transcription_spend(_job(idx=7), _cfg())
    assert "REFUSING to transcribe episode idx=7" in caplog.text
    assert "budget left" in caplog.text


# -- deliberate fail-open cases ----------------------------------------------------------------


def test_no_cap_configured_authorises_everything() -> None:
    reset_run_budget(cap_usd=None)
    assert _authorise_transcription_spend(_job(duration_seconds=99999.0), _cfg(cap=None)) is True


def test_an_episode_with_UNKNOWN_duration_is_allowed_through() -> None:
    """Cannot price it, so cannot refuse it. The ledger still records what it really cost."""
    reset_run_budget(cap_usd=5.0, action="abort")
    assert _authorise_transcription_spend(_job(duration_seconds=None), _cfg()) is True


def test_an_unpriceable_provider_is_allowed_through() -> None:
    """A missing pricing row is a config gap, not a cost problem; grounding the pipeline on it
    would be a worse failure than the one being prevented."""
    reset_run_budget(cap_usd=5.0, action="abort")
    assert _authorise_transcription_spend(_job(), _cfg(provider="no-such-provider")) is True


@pytest.mark.parametrize("action", ["warn", "observe"])
def test_warn_and_observe_never_refuse(action) -> None:
    reset_run_budget(cap_usd=5.0, action=action)
    get_run_budget().record(500.0)
    assert _authorise_transcription_spend(_job(), _cfg(action=action)) is True


def test_a_broken_guard_does_not_block_transcription(monkeypatch) -> None:
    """The guard sits in front of the pipeline's core work; it must fail open, never closed."""
    reset_run_budget(cap_usd=5.0, action="abort")
    import podcast_scraper.workflow.run_budget as rb

    monkeypatch.setattr(rb, "get_run_budget", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _authorise_transcription_spend(_job(), _cfg()) is True


# -- how the refusal crosses the thread boundary -----------------------------------------------


def test_a_refusal_does_NOT_raise_in_the_worker() -> None:
    """Raising here would kill only the worker while main blocks on join() — the 2026-08-12 wedge.

    The guard must return False, not raise, for the supervision design to hold.
    """
    reset_run_budget(cap_usd=5.0, action="abort")
    get_run_budget().record(99.0)
    assert _authorise_transcription_spend(_job(), _cfg()) is False  # no exception


def test_the_MAIN_thread_turns_the_latch_into_CostCapExceeded() -> None:
    """The other half: state set in the worker becomes the run's outcome after the join.

    Deliberately set up so SPEND IS STILL UNDER THE CAP. That is the case the latch exists for,
    and the only one that proves it: authorisation refuses on PROJECTED spend (already-spent plus
    this episode), so a run can legitimately stop with $4.90 recorded against a $5.00 cap. A main
    thread that only compared spend to cap would see $4.90 <= $5.00, raise nothing, and let the
    run report success while silently having skipped work. An earlier version of this test spent
    $99 in the worker, so the spend comparison raised on its own and the test passed with the
    latch check deleted — it proved nothing.
    """
    reset_run_budget(cap_usd=5.0, action="abort")
    cfg = _cfg()

    def worker():
        get_run_budget().record(4.9)
        # a 1-hour episode is ~$0.258: 4.9 + 0.258 > 5.0, so this is refused and latches
        assert _authorise_transcription_spend(_job(duration_seconds=3600.0), cfg) is False

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert get_run_budget().tripped is True
    assert get_run_budget().spent_usd <= 5.0, "precondition: spend alone must NOT breach the cap"

    with pytest.raises(CostCapExceeded):
        enforce_cost_soft_cap(cfg, None)  # what orchestration calls after the join


def test_a_latched_trip_does_not_abort_under_warn() -> None:
    reset_run_budget(cap_usd=5.0, action="abort")
    get_run_budget().trip("something")
    enforce_cost_soft_cap(_cfg(action="warn"), None)  # must not raise


def test_concurrent_workers_cannot_jointly_overspend() -> None:
    """Four transcription workers contend for the last dollar; at most one may proceed."""
    reset_run_budget(cap_usd=5.0, action="abort")
    get_run_budget().record(4.7)  # room for exactly one 1-hour episode (~$0.258)
    cfg = _cfg()
    allowed: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def contend(i: int):
        barrier.wait()
        ok = _authorise_transcription_spend(_job(i), cfg)
        if ok:
            get_run_budget().record(60 * DEEPGRAM_PER_MIN)
        with lock:
            allowed.append(ok)

    threads = [threading.Thread(target=contend, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert get_run_budget().spent_usd <= 5.0, f"overspent: ${get_run_budget().spent_usd}"
    assert sum(allowed) <= 1
