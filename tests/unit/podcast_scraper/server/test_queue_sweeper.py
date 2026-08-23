"""Unit tests for the job-queue sweeper (#1653).

The queue could stop permanently and stay stopped:

1. a job row is ``running`` with a pid;
2. its process dies without finalising (container killed / OOM / restart);
3. the row stays ``running`` and still counts toward ``max_concurrent_jobs``;
4. queued jobs then wait on "another job finishing" — which can no longer happen;
5. recovery needed a human to POST ``/api/jobs/reconcile``.

Observed, not theorised: enrichment job ``ef9f8f9c`` sat queued for 7.75 hours.

The property under test is the ORDER — reconcile before drain — because in the wedged case
freeing the ghost slot is what makes promotion possible at all, and a sweeper that drained
first would compute against a slot count it is about to invalidate.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from podcast_scraper.server import queue_sweeper

pytestmark = [pytest.mark.unit]


def _app(*, jobs_api: bool = True, output_dir: Any = None) -> Any:
    return SimpleNamespace(state=SimpleNamespace(jobs_api_enabled=jobs_api, output_dir=output_dir))


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Record the order of reconcile/drain calls."""
    order: List[str] = []
    import podcast_scraper.server.jobs as jobs_mod

    def _reconcile(_root: Path, **_kw: Any) -> tuple[int, list[str]]:
        order.append("reconcile")
        return 0, []

    async def _drain(_app: Any, _root: Path) -> None:
        order.append("drain")

    monkeypatch.setattr(jobs_mod, "apply_reconcile", _reconcile)
    monkeypatch.setattr(jobs_mod, "drain_queue_async", _drain)
    return order


class TestSweepOnce:
    def test_reconciles_before_draining(self, tmp_path: Path, spy: List[str]) -> None:
        """Order is the whole point — a freed slot is what makes the promotion possible."""
        asyncio.run(queue_sweeper.sweep_once(_app(), tmp_path))
        assert spy == ["reconcile", "drain"]

    def test_reports_how_many_rows_were_stranded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import podcast_scraper.server.jobs as jobs_mod

        monkeypatch.setattr(
            jobs_mod,
            "apply_reconcile",
            lambda _r, **_kw: (2, ["a: failed (dead pid)", "b: stale"]),
        )

        async def _drain(_app: Any, _root: Path) -> None:
            return None

        monkeypatch.setattr(jobs_mod, "drain_queue_async", _drain)
        assert asyncio.run(queue_sweeper.sweep_once(_app(), tmp_path)) == 2

    def test_a_failing_reconcile_still_drains(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One broken half must not disable the other — the point is that something watches."""
        import podcast_scraper.server.jobs as jobs_mod

        drained: List[str] = []

        def _boom(_root: Path, **_kw: Any) -> tuple[int, list[str]]:
            raise OSError("registry locked")

        async def _drain(_app: Any, _root: Path) -> None:
            drained.append("drain")

        monkeypatch.setattr(jobs_mod, "apply_reconcile", _boom)
        monkeypatch.setattr(jobs_mod, "drain_queue_async", _drain)

        assert asyncio.run(queue_sweeper.sweep_once(_app(), tmp_path)) == 0
        assert drained == ["drain"]

    def test_a_failing_drain_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import podcast_scraper.server.jobs as jobs_mod

        async def _boom(_app: Any, _root: Path) -> None:
            raise RuntimeError("no event loop for spawn")

        monkeypatch.setattr(jobs_mod, "apply_reconcile", lambda _r, **_kw: (0, []))
        monkeypatch.setattr(jobs_mod, "drain_queue_async", _boom)
        asyncio.run(queue_sweeper.sweep_once(_app(), tmp_path))


class TestThePauseSwitch:
    """Starting the API server must not mean starting whatever is queued.

    The startup sweep promotes within seconds of every boot. That is right for normal
    operation and wrong for a corpus repair, where the operator needs to bring the stack up,
    look at it, and decide what runs. It also covers a case the enqueue rework leaves open: a
    repair driven as a plain CLI run holds no registry slot, so nothing else would stop the
    sweeper promoting a queued enrichment pass over the files being rewritten.
    """

    def test_a_paused_corpus_reconciles_but_does_not_promote(
        self, tmp_path: Path, spy: List[str]
    ) -> None:
        (tmp_path / ".viewer").mkdir()
        (tmp_path / queue_sweeper.PAUSE_FLAG_RELPATH).write_text("", encoding="utf-8")
        asyncio.run(queue_sweeper.sweep_once(_app(), tmp_path))
        # Reconcile still runs: the registry must stay truthful while promotion is held,
        # otherwise the pause would also hide dead rows from the operator watching them.
        assert spy == ["reconcile"]

    def test_removing_the_flag_resumes_promotion(self, tmp_path: Path, spy: List[str]) -> None:
        flag = tmp_path / queue_sweeper.PAUSE_FLAG_RELPATH
        flag.parent.mkdir()
        flag.write_text("", encoding="utf-8")
        asyncio.run(queue_sweeper.sweep_once(_app(), tmp_path))
        flag.unlink()
        asyncio.run(queue_sweeper.sweep_once(_app(), tmp_path))
        assert spy == ["reconcile", "reconcile", "drain"]

    def test_an_unreadable_root_is_not_treated_as_paused(self, tmp_path: Path) -> None:
        """Failing closed here would stop the queue for the wrong reason, and look identical
        to an operator pause."""
        assert queue_sweeper.drain_is_paused(tmp_path / "does-not-exist") is False


class TestStartAndStop:
    def test_sweeps_immediately_at_startup(self, tmp_path: Path, spy: List[str]) -> None:
        """A restart is exactly when ghost rows exist — waiting a full interval to notice
        would leave the queue wedged through the window an operator is most likely watching."""

        async def _run() -> None:
            task = await queue_sweeper.start_queue_sweeper(
                _app(output_dir=tmp_path), interval_seconds=3600
            )
            await queue_sweeper.stop_queue_sweeper(task)

        asyncio.run(_run())
        assert spy == ["reconcile", "drain"]

    def test_keeps_sweeping_on_the_interval(self, tmp_path: Path, spy: List[str]) -> None:
        async def _run() -> None:
            task = await queue_sweeper.start_queue_sweeper(
                _app(output_dir=tmp_path), interval_seconds=0.01
            )
            await asyncio.sleep(0.06)
            await queue_sweeper.stop_queue_sweeper(task)

        asyncio.run(_run())
        # Startup sweep plus several loop sweeps; exact count is timing-dependent.
        assert spy.count("reconcile") >= 2

    def test_disabled_when_the_jobs_api_is_off(self, tmp_path: Path, spy: List[str]) -> None:
        async def _run() -> Any:
            return await queue_sweeper.start_queue_sweeper(
                _app(jobs_api=False, output_dir=tmp_path)
            )

        assert asyncio.run(_run()) is None
        assert spy == []

    def test_disabled_without_a_corpus_root(self, tmp_path: Path, spy: List[str]) -> None:
        async def _run() -> Any:
            return await queue_sweeper.start_queue_sweeper(_app(output_dir=None))

        assert asyncio.run(_run()) is None
        assert spy == []

    def test_stopping_a_none_task_is_safe(self) -> None:
        asyncio.run(queue_sweeper.stop_queue_sweeper(None))

    def test_the_loop_survives_a_bad_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A housekeeping loop that dies on one transient error re-creates the original bug:
        nothing watching."""
        calls: List[int] = []

        async def _flaky(_app: Any, _root: Path) -> int:
            calls.append(1)
            raise ValueError("transient")

        monkeypatch.setattr(queue_sweeper, "sweep_once", _flaky)

        async def _run() -> None:
            task = asyncio.create_task(queue_sweeper._sweep_loop(_app(), tmp_path, 0.01))
            await asyncio.sleep(0.05)
            await queue_sweeper.stop_queue_sweeper(task)

        asyncio.run(_run())
        assert len(calls) >= 2
