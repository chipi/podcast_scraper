"""A fuse on LLM calls: bound how many any single run — or any single episode — may make.

WHY THIS EXISTS. gpt-5.5 once made ~3,500 LLM calls on ONE episode over an hour, on an expensive
model, for a result that came back empty — and nothing stopped it. The pipeline had a *failure*
circuit breaker, but every one of those calls SUCCEEDED (200 OK, empty content), so a failure
breaker never fires. Only a hard count catches a storm of successful-but-wasteful calls.

This is that count — a fuse, not a throttle. Past the budget it does not slow down; it BLOWS: raises
``LLMCallBudgetExceeded`` on the next call, which aborts the run loudly. A bounded, obvious failure
beats a silent bill.

Two scopes, both live at once:

* **per-run** — a shared, thread-safe ceiling on the whole run/session. Catches cumulative
  overspend even when no single episode looks pathological.
* **per-episode** — a scope-local ceiling. Catches one runaway episode before it eats the budget.

Every provider call routes through ``retry_with_metrics``, which calls :func:`tick` once per attempt
(retries included — they cost money too). Scopes are entered with the context managers below at the
run and episode boundaries; nothing in the provider signatures changes.

Counting calls, not dollars, on purpose: call count is 100% reliable today, whereas our
per-provider cost logging is not (OpenAI/grok log nothing, Mistral undercounts ~12x). A cost fuse
can layer on once that instrumentation is fixed; until then a call fuse is the honest safety.
"""

from __future__ import annotations

import contextlib
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, List, Optional


class LLMCallBudgetExceeded(RuntimeError):
    """The LLM call fuse blew — the current run or episode exceeded its call budget.

    Deliberately a hard error: the run stops rather than keep spending. The message names the scope
    and the budget so the operator sees immediately what tripped and where.
    """


@dataclass
class _RunFuse:
    """Shared across threads for one run; a plain lock guards the counter."""

    max_calls: int
    calls: int = 0
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self.calls += 1
            current = self.calls
        if current > self.max_calls:
            raise LLMCallBudgetExceeded(
                f"LLM call fuse BLEW: this run made {current} LLM calls, over its budget of "
                f"{self.max_calls}. Something is looping on the LLM (a bundled call failing to a "
                f"per-pair storm, a retry loop). The run is stopped to bound spend. Raise "
                f"llm_max_calls_per_run only after confirming the call volume is legitimate."
            )


@dataclass
class _EpisodeFuse:
    """Scope-local to one episode; contextvar isolation means no cross-episode bleed."""

    max_calls: int
    episode_id: str
    calls: int = 0

    def tick(self) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            raise LLMCallBudgetExceeded(
                f"LLM call fuse BLEW on episode {self.episode_id!r}: {self.calls} LLM calls, over "
                f"the per-episode budget of {self.max_calls}. One episode should never need this "
                f"many; a bundled evidence call is almost certainly failing into a per-pair storm. "
                f"The run is stopped to bound spend."
            )


_run_fuse: ContextVar[Optional[_RunFuse]] = ContextVar("llm_run_fuse", default=None)
_episode_fuse: ContextVar[Optional[_EpisodeFuse]] = ContextVar("llm_episode_fuse", default=None)

# The run fuse must be visible from worker threads: the production pipeline makes its LLM calls
# inside ThreadPoolExecutor workers (summarization/processing), and a ContextVar set in the main
# thread does NOT propagate into pool workers — so a run fuse delivered only via ContextVar would
# silently never fire in production. _RunFuse is already thread-safe (lock-guarded counter), so a
# process-global reference lets tick() find the run ceiling from any thread. The ContextVar is kept
# for the context-manager API and precise scoping in the sequential eval loop.
_run_fuse_global_lock = threading.Lock()
_run_fuse_global: Optional[_RunFuse] = None


def _active_run_fuse() -> Optional[_RunFuse]:
    """The run fuse in scope: the thread's ContextVar binding if set, else the process-global one
    (so pool workers, which don't inherit the ContextVar, still enforce the run ceiling)."""
    return _run_fuse.get() or _run_fuse_global


def tick(count: int = 1) -> None:
    """Register ``count`` LLM call attempts against every active fuse; raises if any budget blows.

    Called once per attempt from ``retry_with_metrics``. A no-op when no fuse is installed (unit
    tests, ad-hoc scripts), so it is always safe to call.
    """
    run = _active_run_fuse()
    ep = _episode_fuse.get()
    for _ in range(max(1, count)):
        if run is not None:
            run.tick()
        if ep is not None:
            ep.tick()


def active_scopes() -> List[str]:
    """Which fuses are currently installed — for diagnostics/tests."""
    out: List[str] = []
    if _active_run_fuse() is not None:
        out.append("run")
    if _episode_fuse.get() is not None:
        out.append("episode")
    return out


def current_episode_id() -> Optional[str]:
    """The episode id of the installed per-episode fuse, or None outside an episode scope.

    Lets cost/usage telemetry stamp the episode dimension for free wherever ``install_episode`` /
    ``episode_budget`` is active (the eval loop), without threading the id through every call site.
    """
    ep = _episode_fuse.get()
    return ep.episode_id if ep is not None else None


def install_run(max_calls: Optional[int]) -> None:
    """Imperative per-run install, for a top-level loop that cannot wrap itself in a ``with``.
    Call ONCE before the episode loop; the counter then accumulates across every episode. ``None``
    /<=0 clears the run fuse. Sets both the ContextVar and the process-global reference so worker
    threads (which don't inherit the ContextVar) also see the ceiling."""
    global _run_fuse_global
    fuse = _RunFuse(max_calls=int(max_calls)) if max_calls and max_calls > 0 else None
    _run_fuse.set(fuse)
    with _run_fuse_global_lock:
        _run_fuse_global = fuse


def install_episode(max_calls: Optional[int], episode_id: str) -> None:
    """Imperative per-episode install — call at the TOP of each loop iteration. Each call replaces
    the prior episode's fuse, so the per-episode count resets naturally. ``None``/<=0 clears it."""
    _episode_fuse.set(
        _EpisodeFuse(max_calls=int(max_calls), episode_id=episode_id)
        if max_calls and max_calls > 0
        else None
    )


@contextlib.contextmanager
def run_budget(max_calls: Optional[int]) -> Iterator[Optional[_RunFuse]]:
    """Install a per-run fuse for the duration of the block. ``None`` disables it (no ceiling)."""
    global _run_fuse_global
    if max_calls is None or max_calls <= 0:
        yield None
        return
    fuse = _RunFuse(max_calls=int(max_calls))
    token = _run_fuse.set(fuse)
    with _run_fuse_global_lock:
        prev_global = _run_fuse_global
        _run_fuse_global = fuse
    try:
        yield fuse
    finally:
        _run_fuse.reset(token)
        with _run_fuse_global_lock:
            _run_fuse_global = prev_global


@contextlib.contextmanager
def episode_budget(max_calls: Optional[int], episode_id: str) -> Iterator[Optional[_EpisodeFuse]]:
    """Install a per-episode fuse for the duration of the block. ``None`` disables it."""
    if max_calls is None or max_calls <= 0:
        yield None
        return
    fuse = _EpisodeFuse(max_calls=int(max_calls), episode_id=episode_id)
    token = _episode_fuse.set(fuse)
    try:
        yield fuse
    finally:
        _episode_fuse.reset(token)
