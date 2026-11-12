"""Re-indexing a reprocessed episode must not leave the previous run's rows behind (#1969).

The defect this pins: incremental indexing upserts on ``id``. Per-episode rows (episode_title,
episode_description, summary_short) carry episode-keyed ids and are overwritten. DERIVED rows —
insight / quote / kg_topic / kg_entity — are content-keyed, so a reprocessed episode's new rows are
INSERTED beside the old run's instead of replacing them, and nothing ever removed the old ones.

Measured on prod 2026-09-04: a full rebuild shed 3,149 derived vectors while episode coverage stayed
identical at 1,067 — roughly 30 episodes' worth of rows from runs that were no longer canonical.
Search served insights and quotes from those runs, silently, because stale rows make counts LARGER
rather than smaller so no gate fires.

Distinct from the 2026-08-31 § C2 fix, which corrected which artifact is *canonical* at DISCOVERY
(``corpus_metadata_index`` now uses the shared newest-run dedupe). That made a FULL rebuild clean;
it did nothing for the incremental path, which is what this covers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

pytest.importorskip("lancedb")

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402

_DIM = 8
EPISODE_ID = "ep-reprocessed"


def _embed_stub(text: str, model_id: str, *, allow_download: bool):
    h = abs(hash(text))
    return [float((h >> (i * 3)) & 0x7) / 7.0 for i in range(_DIM)]


def _write_run(corpus: Path, run: str, insight_texts: list[str]) -> None:
    """Write one run dir for the SAME episode, with caller-chosen insight text.

    Different text -> different content-keyed ids, which is exactly the condition under which the
    old rows used to survive.
    """
    base = corpus / "feeds" / "showx" / f"run_{run}"
    (base / "metadata").mkdir(parents=True, exist_ok=True)
    gi_rel = "gi.json"
    doc = {
        "feed": {"feed_id": "showx", "title": "Show X"},
        "episode": {
            "episode_id": EPISODE_ID,
            "guid": "guid-reprocessed",
            "title": "An episode that gets reprocessed",
            "description": "Description text.",
            "published_date": "2026-06-15T00:00:00Z",
        },
        "summary": {"bullets": [f"bullet from {run}"], "short_summary": f"short {run}"},
        "grounded_insights": {"artifact_path": gi_rel},
    }
    (base / "metadata" / "ep.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    gi = {
        "episode_id": EPISODE_ID,
        "schema_version": "1.0",
        "nodes": [
            {"id": f"insight:{run}:{i}", "type": "Insight", "properties": {"text": txt}}
            for i, txt in enumerate(insight_texts)
        ],
        "edges": [],
    }
    (base / gi_rel).write_text(json.dumps(gi), encoding="utf-8")


def _rows(lance: Path) -> dict[str, list[dict]]:
    be = LanceDBBackend(str(lance))
    out: dict[str, list[dict]] = {}
    for tier in ("segment", "insight", "aux"):
        tbl = be._open_if_exists(tier)
        if tbl is None:
            out[tier] = []
            continue
        n = tbl.count_rows()
        out[tier] = (
            tbl.search().limit(n).select(["id", "episode_id", "text"]).to_list() if n else []
        )
    return out


def test_full_reindex_is_untouched_by_the_prune(tmp_path, monkeypatch):
    """A full reindex clears tables wholesale; the prune must stay out of its way."""
    monkeypatch.setattr(tti, "_embed", _embed_stub)
    corpus = tmp_path / "corpus"
    lance = corpus / "search" / "lance_index"
    _write_run(corpus, "20260101-000000", ["alpha insight one"])
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    _write_run(corpus, "20260202-000000", ["beta insight one"])
    stats = tti.build_two_tier_index(corpus, lance, drop_existing=True)
    texts = {str(r.get("text") or "") for tier in _rows(lance) for r in _rows(lance)[tier]}
    assert not any("alpha" in t for t in texts)
    assert stats.stale_rows_pruned == 0, "a full reindex must not need the incremental prune"


def test_unchanged_episode_is_not_pruned(tmp_path, monkeypatch):
    """The fingerprint skip must keep an untouched episode's rows intact.

    The dangerous failure mode for this fix is pruning an episode the build did not re-emit rows
    for — that would delete a healthy episode on the strength of an empty buffer.
    """
    monkeypatch.setattr(tti, "_embed", _embed_stub)
    corpus = tmp_path / "corpus"
    lance = corpus / "search" / "lance_index"
    _write_run(corpus, "20260101-000000", ["alpha insight one", "alpha insight two"])
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    before = _rows(lance)
    before_ids = {r["id"] for tier in before for r in before[tier]}

    stats = tti.build_two_tier_index(corpus, lance, drop_existing=False)
    after = _rows(lance)
    after_ids = {r["id"] for tier in after for r in after[tier]}

    assert stats.episodes_skipped_unchanged >= 1
    assert after_ids == before_ids, "an unchanged episode lost rows"
    assert stats.stale_rows_pruned == 0


class TestCountEpisodeRowsFailsSafe:
    """``count_episode_rows`` feeds the prune guard, so its failure mode decides data safety.

    The guard refuses to prune when the emitted set is much smaller than what is indexed. If the
    count blew up or raised, the guard would either crash the build or — worse, if it returned
    something truthy-but-wrong — let a destructive prune through. It returns 0, which makes the
    guard's ``if existing and …`` fall through to the normal prune path: the behaviour before this
    guard existed, which is the right default for "I could not measure".
    """

    def test_a_tier_that_does_not_exist_counts_zero(self, tmp_path) -> None:
        be = LanceDBBackend(str(tmp_path / "empty_index"))

        assert be.count_episode_rows("insight", "ep-anything") == 0

    def test_a_raising_table_counts_zero_rather_than_propagating(
        self, tmp_path, monkeypatch
    ) -> None:
        be = LanceDBBackend(str(tmp_path / "empty_index"))

        class _Exploding:
            def count_rows(self, *_a, **_k):
                raise RuntimeError("lance blew up mid-count")

        monkeypatch.setattr(be, "_open_if_exists", lambda _tier: _Exploding())

        assert be.count_episode_rows("insight", "ep-anything") == 0


def _write_kg_run(corpus: Path, run: str, topics: list[str]) -> None:
    """One run dir for the SAME episode with no GI: its content-keyed rows are kg_topic rows."""
    base = corpus / "feeds" / "showx" / f"run_{run}"
    (base / "metadata").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "showx", "title": "Show X"},
        "episode": {
            "episode_id": EPISODE_ID,
            "guid": "guid-reprocessed",
            "title": "An episode that gets reprocessed",
            "description": "Description text.",
            "published_date": "2026-06-15T00:00:00Z",
        },
        "knowledge_graph": {"artifact_path": "kg.json"},
    }
    (base / "metadata" / "ep.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    kg = {
        "episode_id": EPISODE_ID,
        "nodes": [
            {"id": "topic:" + t.replace(" ", "-"), "type": "Topic", "properties": {"label": t}}
            for t in topics
        ],
        "edges": [],
    }
    (base / "kg.json").write_text(json.dumps(kg), encoding="utf-8")


def _texts(lance: Path) -> set[str]:
    rows = _rows(lance)
    return {str(r.get("text") or "") for tier in rows for r in rows[tier]}


def test_reindex_removes_kg_rows_from_the_superseded_run(tmp_path, monkeypatch):
    monkeypatch.setattr(tti, "_embed", _embed_stub)
    corpus = tmp_path / "corpus"
    lance = corpus / "search" / "lance_index"
    _write_kg_run(corpus, "20260101-000000", ["alpha topic one", "alpha topic two"])
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    assert any("alpha topic" in t for t in _texts(lance)), "run 1 indexed no topics"

    _write_kg_run(corpus, "20260202-000000", ["beta topic one", "beta topic two"])
    stats = tti.build_two_tier_index(corpus, lance, drop_existing=False)

    texts = _texts(lance)
    assert not any("alpha topic" in t for t in texts), sorted(texts)
    assert any("beta topic" in t for t in texts), "the current run's rows are missing"
    assert stats.stale_rows_pruned > 0


def test_partial_emission_does_not_prune_the_healthy_kg_rows(tmp_path, monkeypatch, caplog):
    """#2158 on public's rows: a build keeping far fewer rows than it replaces must not prune.

    20 topics, not 6: the guard counts per tier, and the aux tier's per-episode rows (title,
    description, short summary) count as kept, so with a handful of topics a one-topic emission
    still keeps "half" and is pruned.
    """
    import logging

    monkeypatch.setattr(tti, "_embed", _embed_stub)
    corpus = tmp_path / "corpus"
    lance = corpus / "search" / "lance_index"
    _write_kg_run(corpus, "20260101-000000", [f"alpha topic {i}" for i in range(20)])
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    assert sum("alpha topic" in t for t in _texts(lance)) == 20

    _write_kg_run(corpus, "20260202-000000", ["beta topic only one"])
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.search.two_tier_indexer"):
        tti.build_two_tier_index(corpus, lance, drop_existing=False)

    assert sum("alpha topic" in t for t in _texts(lance)) == 20, "healthy rows were pruned"
    assert "REFUSING to prune" in caplog.text


def test_a_slightly_smaller_kg_reprocess_still_prunes(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setattr(tti, "_embed", _embed_stub)
    corpus = tmp_path / "corpus"
    lance = corpus / "search" / "lance_index"
    _write_kg_run(corpus, "20260101-000000", [f"alpha topic {i}" for i in range(6)])
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    _write_kg_run(corpus, "20260202-000000", [f"beta topic {i}" for i in range(5)])
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.search.two_tier_indexer"):
        tti.build_two_tier_index(corpus, lance, drop_existing=False)
    texts = _texts(lance)
    assert not any("alpha topic" in t for t in texts), sorted(texts)
    assert sum("beta topic" in t for t in texts) == 5
    assert "REFUSING to prune" not in caplog.text
