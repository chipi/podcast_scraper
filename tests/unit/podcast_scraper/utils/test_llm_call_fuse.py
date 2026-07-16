"""The LLM call fuse must blow at its budget — the money guardrail, tested at the chokepoint.

gpt-5.5 once made ~3,500 successful LLM calls on ONE episode before anyone noticed. The failure
circuit breaker never fired (the calls SUCCEEDED, they were just wasteful). This fuse is the count
ceiling that catches that: past the budget it raises ``LLMCallBudgetExceeded`` and the run stops.
These tests pin both the mechanism and its integration with ``retry_with_metrics``, through which
every provider call (cloud and ollama/vllm) flows.
"""

from __future__ import annotations

import threading

import pytest

from podcast_scraper.utils import llm_call_fuse as fuse
from podcast_scraper.utils.provider_metrics import retry_with_metrics

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_fuses():
    """Every test starts with no fuse installed and leaves none behind."""
    fuse.install_run(0)
    fuse.install_episode(0, "")
    yield
    fuse.install_run(0)
    fuse.install_episode(0, "")


def test_no_fuse_installed_is_a_noop() -> None:
    for _ in range(1000):
        fuse.tick()  # must never raise when nothing is installed


def test_episode_budget_blows_at_the_limit() -> None:
    fuse.install_episode(5, "ep-x")
    for _ in range(5):
        fuse.tick()  # 5 allowed
    with pytest.raises(fuse.LLMCallBudgetExceeded, match="ep-x"):
        fuse.tick()  # the 6th blows


def test_run_budget_accumulates_across_episodes() -> None:
    """The run counter must NOT reset when a new episode starts — that is how cumulative overspend
    is caught even when no single episode looks pathological."""
    fuse.install_run(10)
    for ep in range(3):
        fuse.install_episode(100, f"ep{ep}")  # generous per-episode; run is binding
        for _ in range(3):
            fuse.tick()
    # 9 calls so far across 3 episodes, run budget 10 -> 1 more allowed, then blow
    fuse.install_episode(100, "ep3")
    fuse.tick()  # 10th
    with pytest.raises(fuse.LLMCallBudgetExceeded, match="this run made"):
        fuse.tick()  # 11th blows on the RUN budget


def test_install_episode_resets_the_per_episode_count() -> None:
    fuse.install_episode(3, "ep-a")
    fuse.tick()
    fuse.tick()
    fuse.install_episode(3, "ep-b")  # new episode -> counter resets
    for _ in range(3):
        fuse.tick()  # fresh 3 allowed
    with pytest.raises(fuse.LLMCallBudgetExceeded, match="ep-b"):
        fuse.tick()


def test_fuse_blows_THROUGH_retry_with_metrics() -> None:
    """THE INTEGRATION. Every provider call runs func() via retry_with_metrics; the fuse ticks
    there, so a runaway is stopped no matter which provider or call site produced it."""
    fuse.install_episode(5, "ep-int")
    n = 0
    with pytest.raises(fuse.LLMCallBudgetExceeded):
        for _ in range(100):
            retry_with_metrics(lambda: "ok", max_retries=0)
            n += 1
    assert n == 5, f"stopped after {n} calls, expected the 6th to blow"


def test_a_blown_fuse_is_NOT_retried_by_retry_with_metrics() -> None:
    """The budget error must abort, not trigger the retry loop (which would keep calling and defeat
    the point). It is raised outside the try, so it propagates straight through."""
    fuse.install_episode(2, "ep-noretry")
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        return "ok"

    with pytest.raises(fuse.LLMCallBudgetExceeded):
        for _ in range(10):
            retry_with_metrics(_fn, max_retries=3)
    # 2 real calls made; the 3rd tick blew before func() ran again — no retry storm on the budget.
    assert calls["n"] == 2


def test_run_fuse_is_thread_safe() -> None:
    """The run fuse is shared across threads; the counter must not lose increments under load."""
    fuse.install_run(100000)
    run = fuse._run_fuse.get()
    assert run is not None  # install_run(100000) set it

    def worker():
        for _ in range(1000):
            run.tick()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert run.calls == 8000, f"lost increments under contention: {run.calls}"


def test_run_fuse_fires_inside_a_pool_worker() -> None:
    """The production guard: install_run runs in the main thread, but the LLM calls happen inside
    ThreadPoolExecutor workers (summarization/processing). A ContextVar does NOT propagate into pool
    workers, so the run ceiling must be enforced via the process-global reference — otherwise the
    fuse silently never fires in production (the exact gap this wiring fixes)."""
    import concurrent.futures

    fuse.install_run(3)

    def do_ticks_in_worker() -> str:
        # a fresh pool thread does not inherit the main thread's ContextVar binding
        assert fuse._run_fuse.get() is None, "worker should NOT see the ContextVar-installed fuse"
        try:
            for _ in range(10):
                fuse.tick()  # must still blow via the global run fuse
        except fuse.LLMCallBudgetExceeded as exc:
            return str(exc)
        return "did not blow"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(do_ticks_in_worker).result()
    assert "BLEW" in result, f"run fuse did not fire inside the worker: {result}"


def test_run_pipeline_installs_the_run_fuse() -> None:
    """orchestration.run_pipeline must install the run fuse in the main thread before any stage, so
    the production ceiling is actually live (not only in the eval harness)."""
    import inspect

    from podcast_scraper.workflow import orchestration

    src = inspect.getsource(orchestration.run_pipeline)
    assert "install_run" in src, "run_pipeline must call llm_call_fuse.install_run"
    assert "llm_max_calls_per_run" in src


def test_generate_episode_metadata_installs_the_episode_fuse() -> None:
    """The per-episode ceiling must be installed where a single-episode LLM storm actually runs
    (generate_episode_metadata: cleaning + GI + KG). Otherwise a ~3,500-call bundled-evidence storm
    slips under the coarser run ceiling (default 8000) and the fuse never fires for the very
    incident it exists to stop."""
    import inspect

    from podcast_scraper.workflow import metadata_generation

    src = inspect.getsource(metadata_generation.generate_episode_metadata)
    assert "install_episode" in src, "generate_episode_metadata must install the per-episode fuse"
    assert "llm_max_calls_per_episode" in src
