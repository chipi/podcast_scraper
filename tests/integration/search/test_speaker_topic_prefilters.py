"""Speaker- and topic-scoped searches narrow the QUERY, not the results (operator 2026-10-10).

The other scopes (show, date, episode, a set of episodes) went into the query as LanceDB
prefilters in ``test_scoped_search_prefilters``. Speaker and topic stayed result filters, because
they resolve through the GI files rather than an index column: the search ranked the whole corpus,
kept a few hundred rows, and only then asked who said each insight — so one speaker's insights
were lost whenever other people's ranked above them. They are now resolved to the matching
insight ids up front and go into the query per table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.backend import AuxDocument, InsightDocument  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402

DIM = 4
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]
CROWD = 300  # more than the candidate pool a top_k=1 search fetches (k x 6, then a few hundred)


def _gi(episode: str, insights: Sequence[tuple[str, str, str | None]]) -> dict:
    """insights: (insight id, speaker person id, topic id or None)."""
    nodes: list[dict] = [
        {
            "id": "topic:risk-management",
            "type": "Topic",
            "properties": {"label": "Risk management"},
        },
        {"id": "topic:diet", "type": "Topic", "properties": {"label": "Diet"}},
    ]
    edges: list[dict] = []
    for iid, person, topic in insights:
        qid = f"q-{iid}"
        nodes.append({"id": iid, "type": "Insight", "properties": {"text": iid}})
        nodes.append(
            {"id": qid, "type": "Quote", "properties": {"text": "q", "speaker_id": person}}
        )
        edges.append({"type": "SUPPORTED_BY", "from": iid, "to": qid})
        if topic:
            edges.append({"type": "ABOUT", "from": iid, "to": topic})
    return {"episode_id": episode, "nodes": nodes, "edges": edges}


def _corpus(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``CROWD`` insights by ``person:crowd`` about diet sit right on the query; one far-off
    insight by ``person:quiet-voice`` about risk management is the only one in either scope."""
    from podcast_scraper.search import corpus_search, hybrid_search

    gi_dir = root / "gi"
    gi_dir.mkdir()
    crowd = [(f"insight:c{i}", "person:crowd", "topic:diet") for i in range(CROWD)]
    quiet = [("insight:q0", "person:quiet-voice", "topic:risk-management")]
    paths = {}
    for episode, insights in (("ep-crowd", crowd), ("ep-quiet", quiet)):
        p = gi_dir / f"{episode}.gi.json"
        p.write_text(json.dumps(_gi(episode, insights)), encoding="utf-8")
        paths[episode] = p
    monkeypatch.setattr(corpus_search, "cached_episode_gi_paths", lambda _root: paths)

    b = LanceDBBackend(str(root / "search" / "lance_index"), embed_dim=DIM)
    b.upsert_insights(
        [
            *(
                InsightDocument(
                    id=f"insight:ep-crowd:c{i}",
                    text="risk drawdown",
                    show_id="s",
                    episode_id="ep-crowd",
                    entity_type="insight",
                    confidence=1.0,
                    derived=True,
                    embedding=[1.0, 0.001 * i, 0.0, 0.0],
                    source_id=f"insight:c{i}",
                )
                for i in range(CROWD)
            ),
            InsightDocument(
                id="insight:ep-quiet:q0",
                text="one passing mention of risk",
                show_id="s",
                episode_id="ep-quiet",
                entity_type="insight",
                confidence=1.0,
                derived=True,
                embedding=[0.0, 0.0, 1.0, 0.0],
                source_id="insight:q0",
            ),
        ]
    )
    b.upsert_auxes(
        [
            AuxDocument(
                id="kg_topic:ep-quiet:risk",
                text="Risk management",
                show_id="s",
                episode_id="ep-quiet",
                doc_type="kg_topic",
                embedding=[0.0, 1.0, 0.0, 0.0],
                source_id="topic:risk-management",
            )
        ]
    )
    b.create_indices()
    b.write_index_meta("test-model")
    monkeypatch.setattr(hybrid_search.embedding_loader, "encode", lambda *a, **k: list(QUERY_VEC))


def _ids(out: dict) -> list:
    assert not out.get("error"), out
    return [r["metadata"].get("source_id") for r in out["results"]]


def test_a_speaker_scope_finds_their_insight_on_a_crowded_corpus(tmp_path, monkeypatch):
    from podcast_scraper.search.capability import structured_corpus_search

    _corpus(tmp_path, monkeypatch)
    out = structured_corpus_search(tmp_path, "risk", speaker="quiet-voice", top_k=1)
    assert _ids(out) == ["insight:q0"]


def test_a_topic_scope_finds_its_insight_on_a_crowded_corpus(tmp_path, monkeypatch):
    from podcast_scraper.search.capability import structured_corpus_search

    _corpus(tmp_path, monkeypatch)
    out = structured_corpus_search(
        tmp_path, "risk", topic="risk-management", doc_types=["insight"], top_k=1
    )
    assert _ids(out) == ["insight:q0"]


def test_the_scopes_never_let_anything_else_through(tmp_path, monkeypatch):
    from podcast_scraper.search.capability import structured_corpus_search

    _corpus(tmp_path, monkeypatch)
    by_speaker = _ids(structured_corpus_search(tmp_path, "risk", speaker="quiet-voice", top_k=20))
    # Only their insight among the insights; rows with no speaker (a topic here) pass a speaker
    # scope, as they always have in the result filter.
    assert [i for i in by_speaker if i.startswith("insight:")] == ["insight:q0"]
    by_topic = set(_ids(structured_corpus_search(tmp_path, "risk", topic="risk", top_k=20)))
    assert by_topic == {"insight:q0", "topic:risk-management"}
    both = _ids(
        structured_corpus_search(tmp_path, "risk", speaker="crowd", topic="risk-man", top_k=20)
    )
    # No insight by crowd is about risk management; the topic row has no speaker, so it passes.
    assert both == ["topic:risk-management"]


def test_the_resolution_matches_the_result_filters_exactly(tmp_path):
    """The prefilter is resolved from the GI files the same way the result filters read them, so it
    is never stricter: same insights for every key, speaker and topic."""
    from podcast_scraper.search.cli_handlers import (
        _insight_passes_speaker_filter,
        _insight_passes_topic_filter,
    )
    from podcast_scraper.search.corpus_search import _insight_scopes_from_gi

    art = _gi(
        "ep",
        [
            ("insight:a", "person:ann-smith", "topic:risk-management"),
            ("insight:b", "person:bo", None),
            ("insight:c", "person:ann-smith", "topic:diet"),
        ],
    )
    art["nodes"].append({"id": "insight:bare", "type": "Insight", "properties": {"text": "x"}})
    scopes = {s.insight_id: s for s in _insight_scopes_from_gi("ep", art)}
    for key in ("ann", "ANN-smith", "bo", "person:", "nobody", "risk", "Risk management", "diet"):
        for iid, s in scopes.items():
            k = key.lower()
            assert any(k in sp for sp in s.speakers) == _insight_passes_speaker_filter(
                art, iid, key
            )
            assert any(k in t for t in s.topics) == _insight_passes_topic_filter(art, iid, key)


def test_a_table_the_scope_cannot_match_is_not_read(tmp_path):
    from podcast_scraper.search.backend import SearchQuery

    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=DIM)
    q = SearchQuery(
        text="x",
        embedding=QUERY_VEC,
        filters={"show_id__contains": "s"},
        tier_filters={"insight": {"source_id__in": ["a"]}, "segment": None},
    )
    assert b._where_for(q, "insight") == "(lower(show_id) LIKE '%s%') AND (source_id IN ('a'))"
    assert b._where_for(q, "aux") == "(lower(show_id) LIKE '%s%')"


def test_the_scope_index_is_built_once_per_corpus_generation(tmp_path, monkeypatch):
    """Memoized on the GI path map, which cached_episode_gi_paths keeps until the corpus changes."""
    from podcast_scraper.search import corpus_search

    calls = []
    real = corpus_search._insight_scopes_from_gi
    monkeypatch.setattr(
        corpus_search,
        "_insight_scopes_from_gi",
        lambda ep, art: calls.append(ep) or real(ep, art),
    )
    _corpus(tmp_path, monkeypatch)
    first = corpus_search._cached_insight_scopes(tmp_path)
    assert corpus_search._cached_insight_scopes(tmp_path) is first
    assert sorted(calls) == ["ep-crowd", "ep-quiet"]
    # A new generation is a new map: rebuilt.
    paths = dict(corpus_search.cached_episode_gi_paths(tmp_path))
    monkeypatch.setattr(corpus_search, "cached_episode_gi_paths", lambda _root: paths)
    assert corpus_search._cached_insight_scopes(tmp_path) is not first
    assert len(calls) == 4
