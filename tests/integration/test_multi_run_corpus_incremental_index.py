"""Composition guardrail: discovery + two-tier indexer over a realistic MULTI-RUN corpus (D7/D8).

Codifies the prod Step-0/Step-1 scenario as an automated test — the exact path that had ZERO
coverage and let D7 (skip re-transcribe) and D8 (re-embed whole corpus) ship. Uses the shared
``build_multi_run_fixture`` generator (3 runs per feed with GUID overlap) so the assertions run over
the real ``feeds/<slug>/run_*/metadata`` layout, not a hand-built flat dir:

* discovery union + newest-run-wins → the index episode set == cumulative-UNIQUE episodes (not the
  sum across runs, and not the last-run-only count);
* a second reindex of the unchanged corpus re-embeds NOTHING (D8 incremental skip).

``_embed`` is stubbed as a counter (compute assertion, model-free); ``_collect_docs_for_episode`` /
``discover_metadata_files`` run for real against the fixture so the discovery path is exercised.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

pytest.importorskip("lancedb")

_GENERATOR = (
    Path(__file__).resolve().parents[2] / "scripts" / "tools" / "build_multi_run_fixture.py"
)
_spec = importlib.util.spec_from_file_location("_build_multi_run_fixture_index", _GENERATOR)
fixture = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(fixture)  # type: ignore[union-attr]

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402

_DIM = 8


class _EmbedCounter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, model_id: str, *, allow_download: bool):
        self.calls += 1
        h = abs(hash(text))
        return [float((h >> (i * 3)) & 0x7) / 7.0 for i in range(_DIM)]


def _distinct_indexed_episodes(lance) -> int:
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

    be = LanceDBBackend(str(lance))
    eps: set[str] = set()
    for tier in ("segment", "insight", "aux"):
        tbl = be._open_if_exists(tier)
        if tbl is None:
            continue
        n = tbl.count_rows()
        if "episode_id" in tbl.schema.names and n:
            for r in tbl.search().limit(n).select(["episode_id"]).to_list():
                if r.get("episode_id"):
                    eps.add(str(r["episode_id"]))
    return len(eps)


def test_index_over_multi_run_corpus_unions_newest_run(tmp_path, monkeypatch):
    counter = _EmbedCounter()
    monkeypatch.setattr(tti, "_embed", counter)

    corpus = tmp_path / "corpus"
    summary = fixture.build_fixture(
        corpus,
        n_feeds=1,
        probe_episodes=1,
        middle_episodes=3,
        latest_episodes=3,
        overlap=2,
    )
    # cumulative unique = middle + (latest - overlap) = 3 + (3 - 2) = 4 (newest run wins on overlap).
    expected_unique = summary["cumulative_unique_total"]
    assert expected_unique == 4

    lance = corpus / "search" / "lance_index"
    stats = tti.build_two_tier_index(corpus, lance, drop_existing=True)

    # The indexer must see the UNION across runs with newest-run-wins — 4 distinct episodes, not the
    # 7 written across the three run dirs, and not the 3 of the latest run alone.
    assert stats.episodes == expected_unique
    assert _distinct_indexed_episodes(lance) == expected_unique
    assert counter.calls > 0

    # D8 composition: reindex the unchanged multi-run corpus → zero re-embeds.
    before = counter.calls
    stats2 = tti.build_two_tier_index(corpus, lance, drop_existing=False)
    assert counter.calls - before == 0
    assert stats2.episodes_skipped_unchanged == expected_unique
