"""The run budget is the number that had to hold and didn't (2026-08-18, ~$48 under a $5 cap).

Every test here names the property of the OLD design it exists to prevent recurring, because
each one of them was individually sufficient to cause that incident.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.run_budget import (
    configure_run_budget,
    get_run_budget,
    reset_run_budget,
    RunBudget,
)


@pytest.fixture(autouse=True)
def _fresh_ledger():
    """A module singleton must not leak between tests."""
    reset_run_budget()
    yield
    reset_run_budget()


# -- accounting -------------------------------------------------------------------------------


def test_it_counts_what_it_is_told() -> None:
    b = RunBudget(cap_usd=5.0, action="abort")
    b.record(1.25)
    b.record(0.75)
    assert b.spent_usd == 2.0
    assert b.remaining_usd == 3.0


def test_junk_and_zero_and_none_are_ignored_not_fatal() -> None:
    """Recording is on the hot path of every provider call; it must never raise."""
    b = RunBudget(cap_usd=5.0, action="abort")
    for junk in (None, "not a number", float("nan") and None, 0.0, -3.0):
        b.record(junk)  # type: ignore[arg-type]
    assert b.spent_usd == 0.0


def test_an_uncapped_ledger_still_counts_so_a_run_can_report_what_it_spent() -> None:
    b = RunBudget(cap_usd=None)
    b.record(9.99)
    assert b.spent_usd == 9.99
    assert b.remaining_usd == float("inf")
    assert b.would_exceed(1_000_000.0) is False
    assert b.check_and_reserve(1_000_000.0) is True


@pytest.mark.parametrize("cap", [None, 0.0, -1.0])
def test_a_nonpositive_cap_means_unbounded(cap) -> None:
    """0 and negative must not read as "cap of zero, refuse everything"."""
    b = RunBudget(cap_usd=cap, action="abort")
    assert b.cap_usd is None
    assert b.check_and_reserve(500.0) is True
    assert b.enforced is False


def test_remaining_never_goes_negative() -> None:
    b = RunBudget(cap_usd=5.0, action="abort")
    b.record(8.0)
    assert b.remaining_usd == 0.0


# -- authorisation ----------------------------------------------------------------------------


def test_spend_that_fits_is_authorised() -> None:
    b = RunBudget(cap_usd=5.0, action="abort")
    b.record(3.0)
    assert b.check_and_reserve(1.5) is True
    assert b.tripped is False


def test_spend_that_breaches_is_REFUSED_and_latches_tripped() -> None:
    b = RunBudget(cap_usd=5.0, action="abort")
    b.record(4.5)
    assert b.check_and_reserve(1.0) is False
    assert b.tripped is True
    assert "would exceed" in (b.trip_reason or "")


def test_exactly_at_the_cap_is_allowed_and_a_cent_over_is_not() -> None:
    """The boundary is where an off-by-one silently doubles or halves a budget."""
    b = RunBudget(cap_usd=5.0, action="abort")
    b.record(4.0)
    assert b.check_and_reserve(1.0) is True, "spending exactly to the cap must be allowed"
    b2 = RunBudget(cap_usd=5.0, action="abort")
    b2.record(4.0)
    assert b2.check_and_reserve(1.01) is False


def test_authorising_does_NOT_itself_add_to_spend() -> None:
    """Counting the estimate AND the actual would drift the ledger up on every call."""
    b = RunBudget(cap_usd=5.0, action="abort")
    b.check_and_reserve(2.0)
    assert b.spent_usd == 0.0
    b.record(2.0)
    assert b.spent_usd == 2.0


@pytest.mark.parametrize("action", ["warn", "observe"])
def test_warn_and_observe_report_the_breach_but_never_refuse(action) -> None:
    b = RunBudget(cap_usd=5.0, action=action)
    b.record(4.9)
    assert b.check_and_reserve(10.0) is True
    assert b.tripped is False
    assert b.enforced is False


def test_only_abort_counts_as_enforced() -> None:
    assert RunBudget(cap_usd=5.0, action="abort").enforced is True
    assert RunBudget(cap_usd=5.0, action="warn").enforced is False
    assert RunBudget(cap_usd=None, action="abort").enforced is False


# -- the thread-safety that the transcription/processing overlap requires ----------------------


def test_concurrent_records_do_not_lose_money() -> None:
    """Transcription and processing record from different threads at the same time."""
    b = RunBudget(cap_usd=None)

    def spend_repeatedly() -> None:
        for _ in range(200):
            b.record(0.01)

    threads = [threading.Thread(target=spend_repeatedly) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert b.spent_usd == pytest.approx(16.0, abs=1e-6)  # 8 threads x 200 x $0.01


def test_two_threads_cannot_both_be_authorised_for_the_same_last_dollar() -> None:
    """The check and the trip latch are one atomic operation for exactly this reason.

    Without atomicity both threads read "spent 4.0, one dollar left", both are told yes, and the
    run spends 6.0 under a 5.0 cap while each decision was individually correct.
    """
    b = RunBudget(cap_usd=5.0, action="abort")
    b.record(4.0)
    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def contend() -> None:
        barrier.wait()
        allowed = b.check_and_reserve(1.0)
        if allowed:
            b.record(1.0)  # a real caller spends what it was authorised for
        with results_lock:
            results.append(allowed)

    threads = [threading.Thread(target=contend) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [False, True], f"expected exactly one winner, got {results}"
    assert b.spent_usd == 5.0


# -- the singleton, and the per-feed reset that caused the incident ----------------------------


def test_configure_carries_spend_ACROSS_feeds() -> None:
    """THE regression test for the incident's scoping bug.

    cli calls run_pipeline once per feed and each call reconfigures the ledger. If configuring
    reset the total, every feed would start again from $0 — which is precisely how a $5 cap
    permitted $48 across 14 feeds.
    """
    cfg = SimpleNamespace(cost_soft_cap_usd_per_run=5.0, cost_soft_cap_action="abort")

    configure_run_budget(cfg)  # feed 1
    get_run_budget().record(3.0)

    configure_run_budget(cfg)  # feed 2 — same batch, same process
    assert get_run_budget().spent_usd == 3.0, "spend must survive the next feed's configure"

    get_run_budget().record(1.5)
    configure_run_budget(cfg)  # feed 3
    assert get_run_budget().spent_usd == 4.5

    # The batch as a whole is now $0.50 from the cap, so a $1 feed must be refused — even
    # though no single feed came anywhere near $5.
    assert get_run_budget().check_and_reserve(1.0) is False


def test_a_trip_survives_reconfiguration_so_the_next_feed_cannot_clear_it() -> None:
    cfg = SimpleNamespace(cost_soft_cap_usd_per_run=5.0, cost_soft_cap_action="abort")
    configure_run_budget(cfg)
    get_run_budget().record(6.0)
    assert get_run_budget().check_and_reserve(1.0) is False
    assert get_run_budget().tripped is True

    configure_run_budget(cfg)
    assert get_run_budget().tripped is True, "a new feed must not clear the batch's trip"


def test_reset_is_what_starts_a_genuinely_new_batch() -> None:
    cfg = SimpleNamespace(cost_soft_cap_usd_per_run=5.0, cost_soft_cap_action="abort")
    configure_run_budget(cfg)
    get_run_budget().record(4.0)
    reset_run_budget(cap_usd=5.0, action="abort")
    assert get_run_budget().spent_usd == 0.0
    assert get_run_budget().tripped is False


def test_configure_reads_cap_and_action_off_the_config() -> None:
    configure_run_budget(
        SimpleNamespace(cost_soft_cap_usd_per_run=12.5, cost_soft_cap_action="warn")
    )
    assert get_run_budget().cap_usd == 12.5
    assert get_run_budget().action == "warn"


def test_configure_tolerates_a_config_missing_the_fields_entirely() -> None:
    configure_run_budget(SimpleNamespace())
    assert get_run_budget().cap_usd is None
    assert get_run_budget().action == "observe"


def test_an_unrecognised_action_falls_back_to_observe_not_abort() -> None:
    """Fail OPEN on a typo: silently aborting every run because of a bad string is worse."""
    configure_run_budget(
        SimpleNamespace(cost_soft_cap_usd_per_run=5.0, cost_soft_cap_action="halt-everything")
    )
    assert get_run_budget().action == "observe"


def test_summary_names_the_numbers_an_operator_needs() -> None:
    b = RunBudget(cap_usd=5.0, action="abort")
    b.record(2.5)
    s = b.summary()
    assert "2.5" in s and "5.0" in s and "abort" in s
    assert "TRIPPED" not in s
    b.trip("because")
    assert "TRIPPED" in b.summary()
    assert "no cap configured" in RunBudget(cap_usd=None).summary()
