"""Unit tests for search/index CLI handlers (#484 Step 4)."""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import cast, List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.search.cli_handlers import (
    EXIT_INVALID_ARGS,
    EXIT_NO_ARTIFACTS,
    EXIT_SUCCESS,
    run_index_cli,
    run_search_cli,
)

_LOG = logging.getLogger("test_search_cli_handlers")


def _search_args_ns(
    tmp_run: Path,
    *,
    query: list[str],
    doc_type: str | None = None,
    feed: str | None = None,
    speaker: str | None = None,
    grounded_only: bool = False,
    since: str | None = None,
    top_k: int = 10,
    fmt: str = "json",
) -> Namespace:
    return Namespace(
        query=query,
        output_dir=str(tmp_run),
        index_path=None,
        doc_type=doc_type,
        feed=feed,
        speaker=speaker,
        grounded_only=grounded_only,
        since=since,
        top_k=top_k,
        format=fmt,
        embedding_model=None,
        command="search",
    )


def _unit(*xs: float) -> list[float]:
    import numpy as np

    v = np.array(xs, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero vector")
    return cast(List[float], (v / n).tolist())


@pytest.mark.unit
def test_run_search_cli_requires_query() -> None:
    args = Namespace(query=[], output_dir="/tmp", command="search")
    assert run_search_cli(args, _LOG) == EXIT_INVALID_ARGS


@pytest.mark.unit
def test_run_search_cli_missing_index(tmp_path: Path) -> None:
    args = Namespace(
        query=["hello"],
        output_dir=str(tmp_path),
        index_path=None,
        doc_type=None,
        feed=None,
        speaker=None,
        grounded_only=False,
        since=None,
        top_k=10,
        format="json",
        embedding_model=None,
        command="search",
    )
    assert run_search_cli(args, _LOG) == EXIT_NO_ARTIFACTS


@pytest.mark.unit
def test_run_search_cli_json(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from podcast_scraper.search.faiss_store import FaissVectorStore

    out = tmp_path / "run"
    out.mkdir()
    idx = out / "search"
    idx.mkdir()
    store = FaissVectorStore(4, index_dir=idx)
    e0 = _unit(1, 0, 0, 0)
    store.upsert(
        "d1",
        e0,
        {
            "doc_type": "insight",
            "episode_id": "ep1",
            "feed_id": "feed_a",
            "publish_date": "2024-06-01",
            "source_id": "n1",
            "text": "unique snippet xyz",
        },
    )
    store.persist(idx)

    args = Namespace(
        query=["q"],
        output_dir=str(out),
        index_path=None,
        doc_type=None,
        feed=None,
        speaker=None,
        grounded_only=False,
        since=None,
        top_k=5,
        format="json",
        embedding_model=None,
        command="search",
    )

    with patch("podcast_scraper.search.corpus_search.embedding_loader.encode") as enc:
        enc.return_value = e0
        assert run_search_cli(args, _LOG) == EXIT_SUCCESS

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["query"] == "q"
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "unique snippet xyz"
    assert "text" not in data["results"][0]["metadata"]


@pytest.mark.unit
def test_run_search_cli_doc_type_kg_topic(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """``--type kg_topic`` restricts FAISS pre-filter (mixed corpus)."""
    from podcast_scraper.search.faiss_store import FaissVectorStore

    out = tmp_path / "run"
    out.mkdir()
    idx = out / "search"
    idx.mkdir()
    e_kg = _unit(1, 0, 0, 0)
    e_in = _unit(0, 1, 0, 0)
    store = FaissVectorStore(4, index_dir=idx)
    store.upsert(
        "kg1",
        e_kg,
        {
            "doc_type": "kg_topic",
            "episode_id": "ep1",
            "feed_id": "f1",
            "publish_date": "2024-01-01",
            "source_id": "t1",
            "text": "Quantum widgets topic",
        },
    )
    store.upsert(
        "in1",
        e_in,
        {
            "doc_type": "insight",
            "episode_id": "ep1",
            "feed_id": "f1",
            "publish_date": "2024-01-01",
            "source_id": "i1",
            "grounded": True,
            "text": "Unrelated insight text",
        },
    )
    store.persist(idx)

    args = _search_args_ns(out, query=["q"], doc_type="kg_topic", top_k=5)
    with patch("podcast_scraper.search.corpus_search.embedding_loader.encode") as enc:
        enc.return_value = e_kg
        assert run_search_cli(args, _LOG) == EXIT_SUCCESS

    data = json.loads(capsys.readouterr().out)
    assert len(data["results"]) == 1
    assert data["results"][0]["metadata"]["doc_type"] == "kg_topic"
    assert "Quantum" in data["results"][0]["text"]


@pytest.mark.unit
def test_run_search_cli_grounded_only_drops_ungrounded_insight(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from podcast_scraper.search.faiss_store import FaissVectorStore

    out = tmp_path / "run"
    out.mkdir()
    idx = out / "search"
    idx.mkdir()
    emb = _unit(1, 0, 0, 0)
    store = FaissVectorStore(4, index_dir=idx)
    store.upsert(
        "g1",
        emb,
        {
            "doc_type": "insight",
            "episode_id": "ep1",
            "feed_id": "f1",
            "publish_date": "2024-01-01",
            "source_id": "a",
            "grounded": False,
            "text": "Ungrounded",
        },
    )
    store.upsert(
        "g2",
        emb,
        {
            "doc_type": "insight",
            "episode_id": "ep1",
            "feed_id": "f1",
            "publish_date": "2024-01-01",
            "source_id": "b",
            "grounded": True,
            "text": "Grounded line",
        },
    )
    store.persist(idx)

    args = _search_args_ns(out, query=["q"], grounded_only=True, top_k=10)
    with patch("podcast_scraper.search.corpus_search.embedding_loader.encode") as enc:
        enc.return_value = emb
        assert run_search_cli(args, _LOG) == EXIT_SUCCESS

    data = json.loads(capsys.readouterr().out)
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "Grounded line"


@pytest.mark.unit
def test_run_search_cli_feed_and_since_filters(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from podcast_scraper.search.faiss_store import FaissVectorStore

    out = tmp_path / "run"
    out.mkdir()
    idx = out / "search"
    idx.mkdir()
    emb = _unit(1, 0, 0, 0)
    store = FaissVectorStore(4, index_dir=idx)
    store.upsert(
        "x1",
        emb,
        {
            "doc_type": "insight",
            "episode_id": "ep1",
            "feed_id": "podcast_alpha_feed",
            "publish_date": "2023-01-15",
            "source_id": "s1",
            "grounded": True,
            "text": "Old alpha",
        },
    )
    store.upsert(
        "x2",
        emb,
        {
            "doc_type": "insight",
            "episode_id": "ep2",
            "feed_id": "podcast_alpha_feed",
            "publish_date": "2024-06-20",
            "source_id": "s2",
            "grounded": True,
            "text": "New alpha",
        },
    )
    store.persist(idx)

    args = _search_args_ns(
        out,
        query=["q"],
        feed="alpha",
        since="2024-01-01",
        top_k=10,
    )
    with patch("podcast_scraper.search.corpus_search.embedding_loader.encode") as enc:
        enc.return_value = emb
        assert run_search_cli(args, _LOG) == EXIT_SUCCESS

    data = json.loads(capsys.readouterr().out)
    assert len(data["results"]) == 1
    assert data["results"][0]["metadata"]["episode_id"] == "ep2"


@pytest.mark.unit
def test_run_search_cli_speaker_filter_on_quote(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from podcast_scraper.search.faiss_store import FaissVectorStore

    out = tmp_path / "run"
    out.mkdir()
    meta = out / "metadata"
    meta.mkdir()
    gi = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": "ep1",
        "nodes": [
            {
                "id": "ins1",
                "type": "Insight",
                "properties": {"text": "I1", "episode_id": "ep1", "grounded": True},
            },
            {
                "id": "q1",
                "type": "Quote",
                "properties": {
                    "text": "host speaks",
                    "episode_id": "ep1",
                    "speaker_id": "HOST_A",
                },
            },
        ],
        "edges": [],
    }
    (meta / "e.gi.json").write_text(json.dumps(gi), encoding="utf-8")

    idx = out / "search"
    idx.mkdir()
    emb = _unit(1, 0, 0, 0)
    store = FaissVectorStore(4, index_dir=idx)
    store.upsert(
        "q1",
        emb,
        {
            "doc_type": "quote",
            "episode_id": "ep1",
            "feed_id": "f1",
            "publish_date": "2024-01-01",
            "source_id": "q1",
            "speaker_id": "GUEST_X",
            "text": "guest quote",
        },
    )
    store.upsert(
        "q2",
        emb,
        {
            "doc_type": "quote",
            "episode_id": "ep1",
            "feed_id": "f1",
            "publish_date": "2024-01-01",
            "source_id": "q2",
            "speaker_id": "HOST_A",
            "text": "host quote",
        },
    )
    store.persist(idx)

    args = _search_args_ns(out, query=["q"], speaker="HOST", top_k=10)
    with patch("podcast_scraper.search.corpus_search.embedding_loader.encode") as enc:
        enc.return_value = emb
        assert run_search_cli(args, _LOG) == EXIT_SUCCESS

    data = json.loads(capsys.readouterr().out)
    assert len(data["results"]) == 1
    assert "host" in data["results"][0]["text"].lower()


@pytest.mark.unit
def test_run_index_cli_stats_missing_index(tmp_path: Path) -> None:
    args = Namespace(
        output_dir=str(tmp_path),
        stats=True,
        rebuild=False,
        vector_index_path=None,
        embedding_model=None,
        command="index",
    )
    assert run_index_cli(args, _LOG) == EXIT_NO_ARTIFACTS


@pytest.mark.unit
def test_run_index_cli_stats_ok(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from podcast_scraper.search.faiss_store import FaissVectorStore

    out = tmp_path / "o"
    out.mkdir()
    idx = out / "search"
    idx.mkdir()
    store = FaissVectorStore(2, index_dir=idx)
    store.upsert("a", _unit(1, 0), {"doc_type": "insight"})
    store.persist(idx)

    args = Namespace(
        output_dir=str(out),
        stats=True,
        rebuild=False,
        vector_index_path=None,
        embedding_model=None,
        command="index",
    )
    assert run_index_cli(args, _LOG) == EXIT_SUCCESS
    blob = json.loads(capsys.readouterr().out)
    assert blob["total_vectors"] == 1
    assert blob["doc_type_counts"].get("insight") == 1


@pytest.mark.unit
@patch("podcast_scraper.search.cli_handlers.index_corpus")
def test_run_index_cli_calls_index_corpus_when_not_stats(
    mock_ic: MagicMock, tmp_path: Path
) -> None:
    """Non-``--stats`` path must call ``index_corpus`` (regression guard)."""
    from podcast_scraper.search.indexer import IndexRunStats

    mock_ic.return_value = IndexRunStats()
    args = Namespace(
        output_dir=str(tmp_path),
        stats=False,
        rebuild=True,
        vector_index_path=None,
        embedding_model=None,
        vector_faiss_index_mode=None,
        vector_index_types=None,
        command="index",
    )
    assert run_index_cli(args, _LOG) == EXIT_SUCCESS
    mock_ic.assert_called_once()
    cargs, ckwargs = mock_ic.call_args
    assert cargs[0] == str(tmp_path)
    assert ckwargs["rebuild"] is True
