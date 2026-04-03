"""Unit tests for RFC-061 semantic ranking in ``collect_insights``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, List
from unittest.mock import patch

import pytest

from podcast_scraper.gi import build_artifact, write_artifact
from podcast_scraper.gi.explore import (
    collect_insights,
    default_vector_index_dir,
    explore_resolve_insights_and_loaded,
    load_artifacts,
    scan_artifact_paths,
)


def _unit(*xs: float) -> list[float]:
    import numpy as np

    v = np.array(xs, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero vector")
    return cast(List[float], (v / n).tolist())


@pytest.mark.unit
def test_default_vector_index_dir_none_without_index(tmp_path: Path) -> None:
    assert default_vector_index_dir(tmp_path) is None
    (tmp_path / "search").mkdir()
    assert default_vector_index_dir(tmp_path) is None


@pytest.mark.unit
def test_default_vector_index_dir_when_faiss_present(tmp_path: Path) -> None:
    from podcast_scraper.search.faiss_store import FaissVectorStore

    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    store = FaissVectorStore(3, index_dir=idx)
    store.upsert("x", _unit(1, 0, 0), {"doc_type": "insight"})
    store.persist(idx)
    got = default_vector_index_dir(tmp_path)
    assert got == idx.resolve()


@pytest.mark.unit
def test_collect_insights_semantic_ranked_when_index_hits(tmp_path: Path) -> None:
    """With mocked encode, vector path returns insights not matching substring topic."""
    from podcast_scraper.search.faiss_store import FaissVectorStore

    artifact = build_artifact(
        "ep:vec",
        "Transcript body for quotes.",
        prompt_version="v1",
        insight_texts=["Zebra alpha bravo semantic-only match body."],
    )
    insight_id = next(str(n["id"]) for n in artifact["nodes"] if n.get("type") == "Insight")
    (tmp_path / "metadata").mkdir()
    gi_path = tmp_path / "metadata" / "epvec.gi.json"
    write_artifact(gi_path, artifact, validate=True)
    meta_doc = {
        "episode": {"episode_id": "ep:vec"},
        "grounded_insights": {"artifact_path": "metadata/epvec.gi.json"},
    }
    (tmp_path / "metadata" / "epvec.metadata.json").write_text(
        json.dumps(meta_doc),
        encoding="utf-8",
    )

    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    store = FaissVectorStore(4, index_dir=idx)
    emb = _unit(1, 0, 0, 0)
    store.upsert(
        f"insight:ep:vec:{insight_id}",
        emb,
        {
            "doc_type": "insight",
            "episode_id": "ep:vec",
            "source_id": insight_id,
            "feed_id": "f",
            "publish_date": "2024-01-01",
            "text": "Zebra alpha bravo semantic-only match body.",
        },
    )
    store.persist(idx)

    with patch("podcast_scraper.providers.ml.embedding_loader.encode") as enc:
        enc.return_value = emb
        insights, ranked = collect_insights(
            [],
            topic="no-substring-in-insight-text-xyz",
            semantic_index_dir=idx,
            output_dir=tmp_path,
        )
    assert ranked is True
    assert len(insights) == 1
    assert insights[0].insight_id == insight_id
    assert "Zebra" in insights[0].text


@pytest.mark.unit
def test_collect_insights_falls_back_when_semantic_empty(tmp_path: Path) -> None:
    """Empty FAISS index → substring path (topic matches text)."""
    from podcast_scraper.search.faiss_store import FaissVectorStore

    artifact = build_artifact(
        "ep:fb",
        "Transcript.",
        prompt_version="v1",
        insight_texts=["Fallback substring alpha."],
    )
    (tmp_path / "metadata").mkdir()
    gi_path = tmp_path / "metadata" / "epfb.gi.json"
    write_artifact(gi_path, artifact, validate=True)
    loaded = load_artifacts([gi_path], validate=False)

    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    store = FaissVectorStore(2, index_dir=idx)
    store.persist(idx)

    insights, ranked = collect_insights(
        loaded,
        topic="alpha",
        semantic_index_dir=idx,
    )
    assert ranked is False
    assert len(insights) == 1


@pytest.mark.unit
def test_explore_resolve_semantic_loads_only_indexed_episodes(tmp_path: Path) -> None:
    """Semantic explore loads gi for hit episodes only; scan still sees extra orphan gi files."""
    from podcast_scraper.search.faiss_store import FaissVectorStore

    artifact_a = build_artifact(
        "ep:res-a",
        "Transcript for indexed episode.",
        prompt_version="v1",
        insight_texts=["Zebra semantic unique qqq token for vector hit."],
    )
    insight_id_a = next(str(n["id"]) for n in artifact_a["nodes"] if n.get("type") == "Insight")
    (tmp_path / "metadata").mkdir()
    gi_a = tmp_path / "metadata" / "indexed_ep.gi.json"
    write_artifact(gi_a, artifact_a, validate=True)

    artifact_b = build_artifact(
        "ep:res-b",
        "Other transcript.",
        prompt_version="v1",
        insight_texts=["Orphan episode not in metadata map."],
    )
    gi_b = tmp_path / "metadata" / "orphan.gi.json"
    write_artifact(gi_b, artifact_b, validate=True)

    meta_doc = {
        "episode": {"episode_id": "ep:res-a"},
        "grounded_insights": {"artifact_path": "metadata/indexed_ep.gi.json"},
    }
    (tmp_path / "metadata" / "resa.metadata.json").write_text(
        json.dumps(meta_doc),
        encoding="utf-8",
    )

    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    store = FaissVectorStore(4, index_dir=idx)
    emb = _unit(1, 0, 0, 0)
    store.upsert(
        f"insight:ep:res-a:{insight_id_a}",
        emb,
        {
            "doc_type": "insight",
            "episode_id": "ep:res-a",
            "source_id": insight_id_a,
            "feed_id": "f",
            "publish_date": "2024-01-01",
            "text": "Zebra semantic unique qqq token for vector hit.",
        },
    )
    store.persist(idx)

    paths = scan_artifact_paths(tmp_path)
    assert len(paths) >= 2

    with patch("podcast_scraper.providers.ml.embedding_loader.encode") as enc:
        enc.return_value = emb
        insights, sem_ok, loaded, ep_n = explore_resolve_insights_and_loaded(
            tmp_path,
            paths,
            topic="no-substring-match-semantic-only-xyz",
            speaker=None,
            grounded_only=False,
            min_confidence=None,
            sort_by="confidence",
            strict=False,
        )

    assert sem_ok is True
    assert len(insights) == 1
    assert insights[0].episode_id == "ep:res-a"
    assert len(loaded) == 1
    assert loaded[0][0].resolve() == gi_a.resolve()
    assert ep_n == 1
