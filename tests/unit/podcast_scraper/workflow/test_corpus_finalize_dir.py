"""Unit tests for ``_corpus_finalize_dir_for`` — the scope of the corpus-integration finalize.

Regression guard for the incremental-add bug: a ``--single-feed-uses-corpus-layout`` run writes
into ``<corpus>/feeds/<slug>/run_<id>/``, and the finalize (edge-derivation / index / clusters /
enrichment) used to target that run subdir — building a throwaway run-local index so the new
episode never wove into the shared corpus (stale corpus index + clusters; search never surfaced
it). The finalize must target the CORPUS ROOT for corpus-layout runs, and keep the run dir for
everything else.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.config import Config
from podcast_scraper.workflow.orchestration import _corpus_finalize_dir_for


def _cfg(output_dir: str, corpus_layout: bool) -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": output_dir,
            "single_feed_uses_corpus_layout": corpus_layout,
        }
    )


def test_corpus_layout_finalize_targets_corpus_root(tmp_path) -> None:
    corpus = str(tmp_path / "corpus")
    cfg = _cfg(corpus, corpus_layout=True)
    assert cfg.output_dir  # validator wraps it to <corpus>/feeds/<slug>/; run dir sits under it
    run_dir = str(Path(cfg.output_dir) / "run_abc")
    result = _corpus_finalize_dir_for(cfg, run_dir)
    assert Path(result).name == "corpus", f"want corpus root, got {result!r}"
    assert result != run_dir, "must not target the throwaway run subdir"


def test_non_corpus_layout_finalize_keeps_run_dir(tmp_path) -> None:
    cfg = _cfg(str(tmp_path / "out"), corpus_layout=False)
    assert cfg.output_dir
    run_dir = str(Path(cfg.output_dir) / "run_abc")
    # Legacy single-feed + multi-feed per-feed steps keep their own scope (multi-feed defers
    # corpus integration to finalize_multi_feed_batch).
    assert _corpus_finalize_dir_for(cfg, run_dir) == run_dir
