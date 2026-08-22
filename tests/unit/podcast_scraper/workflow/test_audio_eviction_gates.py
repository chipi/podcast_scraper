"""Audio eviction: what the RUN may do, and what only an OPERATOR may do (#1787, #1757).

The 2026-08-21 incident. A start-of-run, corpus-wide orphan sweep sat in ``run_pipeline`` three
lines ABOVE ``_fetch_and_prepare_episodes`` — the function that applies ``--reprocess-episode-ids``.
So the sweep ran before the run knew which episodes it had been asked about, and it could not have
used the answer anyway: ``sweep_corpus`` takes no episode argument. A one-episode repair walked all
678 episodes, one rclone round trip each, and sat silent for ~16 minutes twice before being killed.

The stack that proved it, taken from the live hung container via SIGABRT:

    orchestration.py:1852  _maybe_sweep_orphaned_audio
    archive/offload.py:233 sweep_corpus
    archive/offload.py:152 evict_run_dir
    archive/backfill.py:320 already_archived
    storage_backend.py:238 _rclone
    subprocess.py:1209     communicate          <- blocked here

So the split this module pins:

* END-of-run eviction stays on the run path. It is scoped to the run's OWN episodes — one
  backend call for a one-episode repair — and it is what stops local audio accumulating.
* The CORPUS-WIDE sweep is maintenance, not a precondition of processing an episode. It moved
  to ``archive sweep`` / sweep-prod-audio.yml, and ``run_pipeline`` must never call it again.
"""

# mypy: disable-error-code="arg-type"
# The gate helpers read cfg via getattr, so a lightweight SimpleNamespace double is sufficient and
# avoids constructing a full Config (which requires provider keys) just to test flag gating.
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow import orchestration as orch

pytestmark = [pytest.mark.unit]


class TestTheRunPathNeverSweepsTheCorpus:
    """Structural, via the AST — a string search would be satisfied by a comment.

    This is the regression that matters: the defect was not that the sweep was wrong, it was
    that it was CALLED FROM HERE. Re-adding the call is the only way to bring the incident back,
    so that is what is asserted, rather than any property of the sweep itself.
    """

    def _orchestration_tree(self) -> ast.Module:
        src = Path(inspect.getsourcefile(orch) or "").read_text(encoding="utf-8")
        return ast.parse(src)

    def test_orchestration_never_calls_sweep_corpus(self) -> None:
        called = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(self._orchestration_tree())
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert "sweep_corpus" not in called, (
            "run_pipeline is calling the corpus-wide sweep again. A repair asked for ONE episode "
            "must not pay a whole-corpus pass of backend round trips before it starts — see this "
            "module's docstring for the 16-minute stall this caused in prod."
        )

    def test_orchestration_does_not_import_sweep_corpus(self) -> None:
        imported = {
            alias.name
            for node in ast.walk(self._orchestration_tree())
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "sweep_corpus" not in imported

    def test_the_per_run_eviction_is_still_there(self) -> None:
        """The other half. Deleting the sweep must not also disable #1787's actual purpose."""
        assert hasattr(orch, "_maybe_evict_local_audio_after_offload")


class TestEvictionGates:
    def test_evict_noop_when_flag_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {"n": 0}
        monkeypatch.setattr(
            "podcast_scraper.archive.offload.evict_run_dir",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        cfg = SimpleNamespace(audio_evict_local_after_offload=False, audio_storage_backend="remote")
        orch._maybe_evict_local_audio_after_offload(cfg, str(tmp_path))
        assert called["n"] == 0

    def test_evict_refused_when_backend_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Evicting with a LOCAL backend would delete the archive's only copy — must be refused
        # BEFORE resolving a backend.
        called = {"n": 0}
        monkeypatch.setattr(
            "podcast_scraper.archive.offload.evict_run_dir",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        cfg = SimpleNamespace(audio_evict_local_after_offload=True, audio_storage_backend="local")
        orch._maybe_evict_local_audio_after_offload(cfg, str(tmp_path))
        assert called["n"] == 0

    def test_evict_calls_through_to_evict_run_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The enabled + remote-backend path resolves a backend and calls evict_run_dir.
        seen: dict = {}
        monkeypatch.setattr(
            "podcast_scraper.utils.audio_cache.resolve_backend", lambda *a, **k: "BACKEND"
        )
        monkeypatch.setattr(
            "podcast_scraper.archive.offload.evict_run_dir",
            lambda run_dir, backend: seen.update(run_dir=run_dir, backend=backend),
        )
        cfg = SimpleNamespace(audio_evict_local_after_offload=True, audio_storage_backend="remote")
        orch._maybe_evict_local_audio_after_offload(cfg, str(tmp_path))
        assert seen == {"run_dir": str(tmp_path), "backend": "BACKEND"}

    def test_evict_none_backend_is_a_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {"n": 0}
        monkeypatch.setattr(
            "podcast_scraper.utils.audio_cache.resolve_backend", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "podcast_scraper.archive.offload.evict_run_dir",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        cfg = SimpleNamespace(audio_evict_local_after_offload=True, audio_storage_backend="remote")
        orch._maybe_evict_local_audio_after_offload(cfg, str(tmp_path))
        assert called["n"] == 0

    def test_evict_swallows_exceptions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A reclaim step must never break a finished run.
        def boom(*a, **k):
            raise RuntimeError("resolve blew up")

        monkeypatch.setattr("podcast_scraper.utils.audio_cache.resolve_backend", boom)
        cfg = SimpleNamespace(audio_evict_local_after_offload=True, audio_storage_backend="remote")
        orch._maybe_evict_local_audio_after_offload(cfg, str(tmp_path))  # no raise
