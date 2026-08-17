"""The post-pipeline enrichment chain ENQUEUES a job; it does not spawn a process (#1653).

This file previously asserted the opposite — that the chain fired a detached
``subprocess.Popen`` with ``stdin=DEVNULL`` and ``start_new_session=True``, logging to
``.viewer/enrichment_pipeline_spawn.log``. That behaviour was real, and it was the problem:

* no row in ``GET /api/jobs`` — it could not be listed, cancelled, or reconciled;
* not serialised against ``max_concurrent_jobs`` — free to run alongside an ingest;
* argv omitted ``--config`` — the child silently used a different operator YAML.

The jobs registry holds exactly zero auto-spawned ``corpus_enrichment`` rows across the whole
life of the corpus (H7): every run it ever had was operator-queued. The old tests passed
throughout, because they asserted the mechanism rather than the outcome.

It became urgent rather than merely untidy once #1648 landed: with the profile finally
reaching the child, that same path stopped being a 3 ms no-op and started doing a full corpus
pass — unqueued, uncancellable, and potentially concurrent with the repair rewriting its
inputs.

So these tests now assert the *outcome* — a queued job with the right argv — and deliberately
not how the API server later runs it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_scraper.workflow.orchestration import _maybe_spawn_enrichment_after_pipeline


class _EnqueueSpy:
    """Records every enqueue call in place of the real registry write."""

    calls: list[dict[str, Any]] = []

    @classmethod
    def install(cls, monkeypatch: pytest.MonkeyPatch) -> None:
        cls.calls = []

        def _fake(corpus_root: Path, **kwargs: Any) -> dict[str, Any]:
            cls.calls.append({"corpus_root": Path(corpus_root), **kwargs})
            return {"job_id": "job-under-test", "status": "queued"}

        import podcast_scraper.server.jobs as jobs_mod

        monkeypatch.setattr(jobs_mod, "enqueue_enrichment_job", _fake)


def _cfg(**overrides: Any) -> Any:
    """Minimal cfg-shaped object the helper reads from. Typed as Any so
    callers can pass it into ``_maybe_spawn_enrichment_after_pipeline``
    which accepts the duck-typed cfg."""
    base: dict[str, Any] = {"enrichment": {}, "profile": None}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestTheGate:
    def test_nothing_is_enqueued_when_the_block_is_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment=None), str(tmp_path))
        assert _EnqueueSpy.calls == []

    def test_nothing_is_enqueued_when_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": False}), str(tmp_path))
        assert _EnqueueSpy.calls == []


class TestItEnqueuesRatherThanSpawning:
    def test_enabled_enqueues_exactly_one_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))
        assert len(_EnqueueSpy.calls) == 1
        assert _EnqueueSpy.calls[0]["corpus_root"] == tmp_path

    def test_it_lands_as_queued_never_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RUNNING is a promise that a process was started, and this process cannot keep it.

        Only the API server can spawn (``start_job_if_running_record`` needs the app), so a
        pipeline-side enqueue that marked itself running would leave a phantom row for a
        process nobody ever started — worse than the detached spawn it replaced.
        """
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))
        assert _EnqueueSpy.calls[0]["force_queued"] is True

    def test_no_detached_process_is_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression that matters: any Popen here is the old defect returning."""
        import subprocess

        started: list[Any] = []

        def _spy_popen(*args: Any, **kwargs: Any) -> SimpleNamespace:
            """Record the call instead of spawning; a real Popen here IS the defect."""
            started.append(args)
            return SimpleNamespace(pid=1)

        monkeypatch.setattr(subprocess, "Popen", _spy_popen)
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))
        assert started == []

    def test_a_registry_failure_never_breaks_the_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The follow-up pass is best-effort; the ingest that just succeeded must still return."""
        import podcast_scraper.server.jobs as jobs_mod

        def _boom(*_a: Any, **_k: Any) -> dict[str, Any]:
            raise OSError("registry locked")

        monkeypatch.setattr(jobs_mod, "enqueue_enrichment_job", _boom)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))


class TestArgvInputs:
    def test_top_level_profile_is_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(
            _cfg(enrichment={"enabled": True}, profile="cloud_balanced"), str(tmp_path)
        )
        assert _EnqueueSpy.calls[0]["profile"] == "cloud_balanced"

    def test_profile_falls_back_to_the_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(
            _cfg(enrichment={"enabled": True, "profile": "cloud_thin"}, profile=None),
            str(tmp_path),
        )
        assert _EnqueueSpy.calls[0]["profile"] == "cloud_thin"

    def test_the_operator_yaml_is_passed_when_it_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closes the ``--config`` gap: the old spawn omitted it, so the child silently used a
        different operator YAML than the parent was running with."""
        (tmp_path / "viewer_operator.yaml").write_text("enrichment: {}\n", encoding="utf-8")
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))
        assert _EnqueueSpy.calls[0]["operator_yaml"] == tmp_path / "viewer_operator.yaml"

    def test_no_operator_yaml_is_passed_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Better None than a path that does not exist — the child has its own default."""
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))
        assert _EnqueueSpy.calls[0]["operator_yaml"] is None

    def test_with_ml_is_off_for_a_deterministic_only_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(
            _cfg(enrichment={"enabled": True, "enrichers": {"insight_density": {}}}),
            str(tmp_path),
        )
        assert _EnqueueSpy.calls[0]["with_ml"] is False

    def test_with_ml_is_on_when_an_enricher_declares_a_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without it those enrichers are warned-and-skipped, which reads as "ran, no output"."""
        _EnqueueSpy.install(monkeypatch)
        _maybe_spawn_enrichment_after_pipeline(
            _cfg(
                enrichment={
                    "enabled": True,
                    "enrichers": {"topic_similarity": {"provider": {"type": "embedding"}}},
                }
            ),
            str(tmp_path),
        )
        assert _EnqueueSpy.calls[0]["with_ml"] is True
