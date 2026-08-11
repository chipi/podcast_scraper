"""Guardrail: /api/index/stats must not go STALE after an incremental index write (D2).

Prod defect (2026-08-11): ``read_lance_index_stats`` memoizes in ``perf_cache`` keyed on
``perf_cache.lance_mtime`` = the top-level ``lance_index/`` dir mtime. LanceDB upserts write into
per-TABLE subdirectories and do NOT bump the parent dir mtime, so after an incremental add the cache
key never changed and the endpoint kept reporting the pre-add counts (94 while the table held 106),
with a frozen ``last_updated``. The fix bumps the index dir mtime after a build that changed the
index (works across the pipeline subprocess boundary, unlike an in-process cache clear).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

pytest.importorskip("lancedb")

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.lance_index_stats import (  # noqa: E402
    clear_index_stats_cache,
    read_lance_index_stats,
)

_DIM = 8


def _fake_embed(text: str, model_id: str, *, allow_download: bool):
    h = abs(hash(text))
    return [float((h >> (i * 3)) & 0x7) / 7.0 for i in range(_DIM)]


def _rows(ep_id: str):
    return [
        (
            f"insight:{ep_id}",
            f"insight {ep_id}",
            {"doc_type": "insight", "episode_id": ep_id, "feed_id": "s"},
        ),
        (
            f"chunk:{ep_id}",
            f"chunk {ep_id}",
            {
                "doc_type": "transcript",
                "episode_id": ep_id,
                "feed_id": "s",
                "timestamp_start_ms": 0,
                "timestamp_end_ms": 1000,
            },
        ),
    ]


def _install(monkeypatch, tmp_path, episode_ids):
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True, exist_ok=True)
    metas = {ep: corpus / "metadata" / f"{ep}.metadata.json" for ep in episode_ids}
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: list(metas.values()))
    monkeypatch.setattr(
        tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": p.name.split(".")[0]}}
    )
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(
        tti, "_collect_docs_for_episode", lambda er, mp, *a, **k: _rows(mp.name.split(".")[0])
    )
    monkeypatch.setattr(tti, "_embed", _fake_embed)
    return corpus


def test_index_stats_reflects_incremental_add(tmp_path, monkeypatch):
    clear_index_stats_cache()
    episode_ids = ["ep1"]
    corpus = _install(monkeypatch, tmp_path, episode_ids)
    lance = corpus / "search" / "lance_index"

    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    first = read_lance_index_stats(lance)
    assert first is not None and first.total_vectors == 2  # 1 insight + 1 chunk

    # Incremental add of a NEW episode: writes into existing table subdirs (does NOT bump the
    # top-level lance_index/ dir mtime by itself). The cached stats must still refresh.
    episode_ids.append("ep2")
    _install(monkeypatch, tmp_path, episode_ids)
    tti.build_two_tier_index(corpus, lance, drop_existing=False)

    second = read_lance_index_stats(lance)
    assert second is not None
    assert second.total_vectors == 4, "index/stats went stale after an incremental add (D2)"
