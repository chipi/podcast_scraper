"""Orchestration gates + corpus-root resolution for audio eviction (#1787, advisor M1/gates).

The bug M1 caught: the start-of-run sweep resolved its root via ``_corpus_finalize_dir_for``,
which returns the corpus root only under ``single_feed_uses_corpus_layout`` — so in a multi-feed
prod batch (each feed a separate ``run_pipeline`` with a fresh empty run dir) the sweep scanned
nothing. ``_resolve_sweep_corpus_root`` fixes that by deriving the root from the run dir's path.
"""

# mypy: disable-error-code="arg-type"
# The gate helpers read cfg via getattr, so a lightweight SimpleNamespace double is sufficient and
# avoids constructing a full Config (which requires provider keys) just to test flag gating.
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow import orchestration as orch

pytestmark = [pytest.mark.unit]


class TestResolveSweepCorpusRoot:
    def test_multi_feed_run_dir_resolves_to_corpus_root(self, tmp_path: Path) -> None:
        # <corpus>/feeds/<slug>/run_<id> -> <corpus>
        run_dir = tmp_path / "feeds" / "acast_x" / "run_20260101-000000_abc"
        run_dir.mkdir(parents=True)
        assert orch._resolve_sweep_corpus_root(str(run_dir)) == str(tmp_path.resolve())

    def test_feed_dir_itself_resolves_to_corpus_root(self, tmp_path: Path) -> None:
        feed_dir = tmp_path / "feeds" / "acast_x"
        feed_dir.mkdir(parents=True)
        assert orch._resolve_sweep_corpus_root(str(feed_dir)) == str(tmp_path.resolve())

    def test_plain_run_without_feeds_ancestor_sweeps_itself(self, tmp_path: Path) -> None:
        plain = tmp_path / "transcripts"
        plain.mkdir()
        assert orch._resolve_sweep_corpus_root(str(plain)) == str(plain)


class TestEvictionGates:
    def _reset_swept(self) -> None:
        orch._SWEPT_CORPUS_ROOTS.clear()

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

    def test_sweep_runs_once_per_corpus_root_per_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._reset_swept()
        calls = {"n": 0}
        monkeypatch.setattr(
            "podcast_scraper.archive.offload.sweep_corpus",
            lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
        )
        monkeypatch.setattr(
            "podcast_scraper.utils.audio_cache.resolve_backend", lambda *a, **k: object()
        )
        run_dir = tmp_path / "feeds" / "a" / "run_1"
        run_dir.mkdir(parents=True)
        run_dir2 = tmp_path / "feeds" / "b" / "run_2"
        run_dir2.mkdir(parents=True)
        cfg = SimpleNamespace(audio_evict_local_after_offload=True, audio_storage_backend="remote")

        # Two sub-runs of the SAME corpus (multi-feed batch) -> sweep fires once.
        orch._maybe_sweep_orphaned_audio(cfg, str(run_dir))
        orch._maybe_sweep_orphaned_audio(cfg, str(run_dir2))
        assert calls["n"] == 1
        self._reset_swept()
