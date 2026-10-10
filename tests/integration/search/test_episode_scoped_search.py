"""Search inside ONE episode filters before ranking, not after (operator 2026-10-10).

The Brief's search must find anything in its episode, and nothing from any other. It used to rank
the whole corpus, keep the top few hundred, and only then drop what was not this episode — so on a
large corpus an episode's own matches never made the cut and the Brief found nothing. The episode
is now part of the query (a LanceDB prefilter), so other episodes are never considered at all.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.backend import SearchQuery, SegmentDocument  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402

DIM = 4
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]


def _seg(n: int, episode: str, vec: list[float], text: str) -> SegmentDocument:
    return SegmentDocument(
        id=f"{episode}_chunk_{n}",
        text=text,
        show_id="show",
        episode_id=episode,
        start_time=float(n),
        end_time=float(n) + 5.0,
        embedding=vec,
    )


@pytest.fixture()
def crowded(tmp_path) -> LanceDBBackend:
    """Fifty rows from other episodes sit right on the query; the target's one row is far off."""
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=DIM)
    others = [
        _seg(i, f"ep-other-{i}", [1.0, 0.01 * i, 0.0, 0.0], "risk risk risk drawdown")
        for i in range(50)
    ]
    target = _seg(0, "ep-target", [0.0, 0.0, 1.0, 0.0], "one passing mention of risk")
    b.upsert_segments([*others, target])
    b.create_indices()
    return b


def test_vector_search_scoped_to_an_episode_finds_its_row_even_when_others_rank_higher(crowded):
    rows = crowded.search_vector(
        SearchQuery(
            text="risk",
            embedding=QUERY_VEC,
            filters={"episode_id": "ep-target"},
            k=1,
            tier="segment",
        )
    )
    assert [r.payload["episode_id"] for r in rows] == ["ep-target"]


def test_keyword_search_scoped_to_an_episode_finds_its_row_even_when_others_rank_higher(crowded):
    rows = crowded.search_bm25(
        SearchQuery(
            text="risk",
            embedding=[],
            filters={"episode_id": "ep-target"},
            k=1,
            tier="segment",
        )
    )
    assert [r.payload["episode_id"] for r in rows] == ["ep-target"]


def test_nothing_from_another_episode_ever_comes_back(crowded):
    rows = crowded.search_vector(
        SearchQuery(
            text="risk",
            embedding=QUERY_VEC,
            filters={"episode_id": "ep-target"},
            k=20,
            tier="segment",
        )
    )
    assert rows and {r.payload["episode_id"] for r in rows} == {"ep-target"}


def test_corpus_search_scoped_to_an_episode_finds_it_on_a_crowded_corpus(tmp_path, monkeypatch):
    """The route-level path: ``structured_corpus_search(..., episode_id=)``.

    The repro (operator 2026-10-10, prod): fifty other episodes outrank the target for the query, so
    a search that ranks the corpus first and filters after returns NOTHING from the episode. Scoped
    in the query, the target's own row comes back.
    """
    from podcast_scraper.search import hybrid_search
    from podcast_scraper.search.capability import structured_corpus_search

    index_dir = tmp_path / "search" / "lance_index"
    b = LanceDBBackend(str(index_dir), embed_dim=DIM)
    b.upsert_segments(
        [
            *(
                _seg(i, f"ep-other-{i}", [1.0, 0.01 * i, 0.0, 0.0], "risk risk risk drawdown")
                for i in range(50)
            ),
            _seg(0, "ep-target", [0.0, 0.0, 1.0, 0.0], "one passing mention of risk"),
        ]
    )
    b.create_indices()
    b.write_index_meta("test-model")
    # The query embedding is the only ML in the path; a fixed vector keeps this test model-free.
    monkeypatch.setattr(hybrid_search.embedding_loader, "encode", lambda *a, **k: list(QUERY_VEC))

    out = structured_corpus_search(tmp_path, "risk", episode_id="ep-target", top_k=1)

    assert not out.get("error"), out
    episodes = [r["metadata"].get("episode_id") for r in out["results"]]
    assert episodes == ["ep-target"]
