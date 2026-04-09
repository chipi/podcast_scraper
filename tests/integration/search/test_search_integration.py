"""Integration checks: corpus indexer builds a loadable FAISS index (RFC-061 Step 3)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper import cli
from podcast_scraper.config import Config
from podcast_scraper.search.faiss_store import FaissVectorStore
from podcast_scraper.search.indexer import EPISODE_FINGERPRINTS_FILE, index_corpus


def _fake_embeddings(texts: list[str], dim: int = 384) -> list[list[float]]:
    out: list[list[float]] = []
    for i, _ in enumerate(texts):
        row = [0.0] * dim
        row[0] = float(i + 1) / max(len(texts), 1)
        out.append(row)
    return out


@pytest.mark.integration
@patch("podcast_scraper.search.indexer.embedding_loader.encode")
def test_index_corpus_produces_searchable_faiss_index(mock_encode, tmp_path: Path) -> None:
    """Round-trip: metadata + gi → index on disk → FaissVectorStore.load + non-empty search."""
    mock_encode.side_effect = lambda texts, *a, **k: _fake_embeddings(list(texts))

    out = tmp_path / "run"
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True)
    trx_dir = out / "transcripts"
    trx_dir.mkdir(parents=True)
    (trx_dir / "ep.txt").write_text(
        "Hello world transcript sample. Second sentence here. Third sentence too.",
        encoding="utf-8",
    )

    gi = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": "ep:int-1",
        "nodes": [
            {
                "id": "insight:int1",
                "type": "Insight",
                "properties": {
                    "text": "Integration insight alpha.",
                    "episode_id": "ep:int-1",
                    "grounded": True,
                },
            },
            {
                "id": "quote:int1",
                "type": "Quote",
                "properties": {
                    "text": "Hello world",
                    "episode_id": "ep:int-1",
                    "char_start": 0,
                    "char_end": 11,
                },
            },
        ],
        "edges": [],
    }
    (meta_dir / "ep1.gi.json").write_text(json.dumps(gi), encoding="utf-8")

    doc = {
        "feed": {"title": "F", "url": "https://e.com/f", "feed_id": "feed:f1"},
        "episode": {"title": "T", "episode_id": "ep:int-1"},
        "content": {"transcript_file_path": "transcripts/ep.txt"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(out),
            "config_snapshot": {},
        },
        "summary": {
            "generated_at": "2020-01-01T00:00:00",
            "bullets": ["Summary bullet one."],
        },
        "grounded_insights": {
            "artifact_path": "metadata/ep1.gi.json",
            "insight_count": 1,
            "generated_at": "2020-01-01T00:00:00",
        },
    }
    (meta_dir / "ep1.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_index_path=str(out / "search"),
        vector_chunk_size_tokens=20,
        vector_chunk_overlap_tokens=8,
        vector_embedding_model="minilm-l6",
    )
    stats = index_corpus(str(out), cfg, rebuild=True)
    assert stats.episodes_scanned == 1
    assert stats.vectors_upserted >= 1
    assert not stats.errors

    idx_dir = out / "search"
    assert (idx_dir / "vectors.faiss").is_file()
    assert (idx_dir / EPISODE_FINGERPRINTS_FILE).is_file()

    store = FaissVectorStore.load(idx_dir)
    assert store.ntotal >= 1
    st = store.stats()
    assert "insight" in st.doc_type_counts or store.ntotal > 0

    q = _fake_embeddings(["Integration insight alpha."], dim=384)[0]
    hits = store.search(q, top_k=5)
    assert hits, "indexed corpus should return at least one hit for matching query"
    assert hits[0].metadata.get("episode_id") == "ep:int-1"
    assert hits[0].metadata.get("doc_type") in (
        "insight",
        "quote",
        "summary",
        "transcript",
    )


@pytest.mark.integration
@patch("podcast_scraper.search.indexer.embedding_loader.encode")
def test_index_corpus_emits_vector_jsonl(mock_encode, tmp_path: Path) -> None:
    mock_encode.side_effect = lambda texts, *a, **k: _fake_embeddings(list(texts))

    out = tmp_path / "run"
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True)
    jsonl_path = tmp_path / "metrics.jsonl"

    doc = {
        "feed": {"title": "F", "url": "https://e.com/f", "feed_id": "feed:f1"},
        "episode": {"title": "T", "episode_id": "ep:j1"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(out),
            "config_snapshot": {},
        },
    }
    (meta_dir / "solo.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_index_path=str(out / "search"),
        vector_embedding_model="minilm-l6",
        jsonl_metrics_enabled=True,
        jsonl_metrics_path=str(jsonl_path),
    )
    index_corpus(str(out), cfg, rebuild=True)
    text = jsonl_path.read_text(encoding="utf-8").strip()
    assert text
    events = [json.loads(line) for line in text.splitlines()]
    done = [e for e in events if e.get("event_type") == "vector_index_completed"]
    assert len(done) == 1
    assert "vector_index_sec" in done[0]
    assert done[0].get("vector_index_kind") == "faiss_flat_ip_idmap"


@pytest.mark.integration
@patch("podcast_scraper.providers.ml.embedding_loader.encode")
def test_cli_main_index_then_search_json(
    mock_encode, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dispatch wiring: ``cli.main`` index + search with shared mocked embedder."""

    def _encode_index_or_query(texts_or_query, *a, **k):
        if isinstance(texts_or_query, list):
            return _fake_embeddings(texts_or_query)
        return _fake_embeddings([texts_or_query])[0]

    mock_encode.side_effect = _encode_index_or_query

    out = tmp_path / "run"
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True)
    trx_dir = out / "transcripts"
    trx_dir.mkdir(parents=True)
    (trx_dir / "ep.txt").write_text("Body.", encoding="utf-8")

    gi = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": "ep:cli-1",
        "nodes": [
            {
                "id": "ins:cli",
                "type": "Insight",
                "properties": {
                    "text": "CLI integration insight gamma.",
                    "episode_id": "ep:cli-1",
                    "grounded": True,
                },
            },
        ],
        "edges": [],
    }
    (meta_dir / "ep1.gi.json").write_text(json.dumps(gi), encoding="utf-8")

    doc = {
        "feed": {"title": "F", "url": "https://e.com/f", "feed_id": "feed:f1"},
        "episode": {"title": "T", "episode_id": "ep:cli-1"},
        "content": {"transcript_file_path": "transcripts/ep.txt"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(out),
            "config_snapshot": {},
        },
        "summary": {"generated_at": "2020-01-01T00:00:00", "bullets": ["B1."]},
        "grounded_insights": {
            "artifact_path": "metadata/ep1.gi.json",
            "insight_count": 1,
            "generated_at": "2020-01-01T00:00:00",
        },
    }
    (meta_dir / "ep1.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    assert cli.main(["index", "--output-dir", str(out), "--rebuild"]) == 0
    capsys.readouterr()
    assert (
        cli.main(
            [
                "search",
                "CLI",
                "integration",
                "gamma",
                "--output-dir",
                str(out),
                "--format",
                "json",
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    raw = capsys.readouterr().out.strip()
    payload = json.loads(raw)
    assert payload.get("query") == "CLI integration gamma"
    assert len(payload.get("results", [])) >= 1
    first = payload["results"][0]
    assert first.get("metadata", {}).get("episode_id") == "ep:cli-1"


@pytest.mark.integration
@patch("podcast_scraper.providers.ml.embedding_loader.encode")
def test_index_corpus_kg_topic_filterable_in_search(mock_encode, tmp_path: Path) -> None:
    """KG rows from indexer appear in ``store.search`` with ``doc_type`` filter."""
    mock_encode.side_effect = lambda texts, *a, **k: _fake_embeddings(list(texts))

    out = tmp_path / "run"
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True)

    gi = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": "ep:kgf-1",
        "nodes": [],
        "edges": [],
    }
    (meta_dir / "e1.gi.json").write_text(json.dumps(gi), encoding="utf-8")

    kg = {
        "schema_version": "1.1",
        "episode_id": "ep:kgf-1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2020-01-01T00:00:00Z",
            "transcript_ref": "t",
        },
        "nodes": [
            {
                "id": "tn",
                "type": "Topic",
                "properties": {"label": "Retrieval filter topic", "slug": "rft"},
            },
        ],
        "edges": [],
    }
    (meta_dir / "e1.kg.json").write_text(json.dumps(kg), encoding="utf-8")

    doc = {
        "feed": {"title": "F", "url": "https://e.com/f", "feed_id": "feed:f1"},
        "episode": {"title": "T", "episode_id": "ep:kgf-1"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(out),
            "config_snapshot": {},
        },
        "grounded_insights": {
            "artifact_path": "metadata/e1.gi.json",
            "insight_count": 0,
            "generated_at": "2020-01-01T00:00:00",
        },
        "knowledge_graph": {
            "artifact_path": "metadata/e1.kg.json",
            "node_count": 1,
            "edge_count": 0,
            "generated_at": "2020-01-01T00:00:00",
        },
    }
    (meta_dir / "e1.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_index_path=str(out / "search"),
        vector_embedding_model="minilm-l6",
    )
    index_corpus(str(out), cfg, rebuild=True)

    idx_dir = out / "search"
    store = FaissVectorStore.load(idx_dir)
    q = _fake_embeddings(["Retrieval filter topic"], dim=384)[0]
    hits = store.search(q, top_k=5, filters={"doc_type": "kg_topic"})
    assert hits, "expected at least one kg_topic hit"
    assert hits[0].metadata.get("doc_type") == "kg_topic"
    assert hits[0].metadata.get("episode_id") == "ep:kgf-1"
