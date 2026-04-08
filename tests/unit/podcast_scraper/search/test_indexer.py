"""Unit tests for corpus indexer (#484 Step 3) with mocked embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.config import Config
from podcast_scraper.search.corpus_scope import index_fingerprint_scope_key
from podcast_scraper.search.indexer import (
    EPISODE_FINGERPRINTS_FILE,
    index_corpus,
    IndexRunStats,
    maybe_index_corpus,
)


def _fake_embeddings(texts: list[str], dim: int = 384) -> list[list[float]]:
    out: list[list[float]] = []
    for i, _ in enumerate(texts):
        row = [0.0] * dim
        row[0] = float(i + 1) / max(len(texts), 1)
        out.append(row)
    return out


@pytest.mark.unit
@patch("podcast_scraper.search.indexer.embedding_loader.encode")
def test_index_corpus_indexes_gi_summary_transcript(mock_encode, tmp_path: Path) -> None:
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
        "episode_id": "ep:index-1",
        "nodes": [
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {
                    "text": "A key insight.",
                    "episode_id": "ep:index-1",
                    "grounded": True,
                },
            },
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {
                    "text": "Hello world",
                    "episode_id": "ep:index-1",
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
        "episode": {"title": "T", "episode_id": "ep:index-1"},
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
    assert stats.episodes_reindexed == 1
    assert stats.vectors_upserted >= 4
    assert not stats.errors

    idx_dir = out / "search"
    assert (idx_dir / "vectors.faiss").is_file()
    fp_path = idx_dir / EPISODE_FINGERPRINTS_FILE
    assert fp_path.is_file()
    fps = json.loads(fp_path.read_text(encoding="utf-8"))
    assert index_fingerprint_scope_key("feed:f1", "ep:index-1") in fps

    stats2 = index_corpus(str(out), cfg, rebuild=False)
    assert stats2.episodes_skipped_unchanged == 1
    assert stats2.vectors_upserted == 0


@pytest.mark.unit
def test_index_corpus_empty_output_dir(tmp_path: Path) -> None:
    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_index_path=str(tmp_path / "search"),
        vector_embedding_model="minilm-l6",
    )
    stats = index_corpus(str(tmp_path), cfg, rebuild=True)
    assert stats.episodes_scanned == 0
    assert (tmp_path / "search" / EPISODE_FINGERPRINTS_FILE).is_file()


@pytest.mark.unit
def test_maybe_index_corpus_skips_qdrant_backend(tmp_path: Path) -> None:
    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_backend="qdrant",
        vector_embedding_model="minilm-l6",
    )
    maybe_index_corpus(str(tmp_path), cfg)
    assert not (tmp_path / "search" / "vectors.faiss").is_file()


@pytest.mark.unit
@patch("podcast_scraper.search.indexer.index_corpus")
def test_maybe_index_corpus_invokes_index_when_faiss_and_enabled(
    mock_ic: MagicMock, tmp_path: Path
) -> None:
    mock_ic.return_value = IndexRunStats()
    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_backend="faiss",
        vector_embedding_model="minilm-l6",
    )
    maybe_index_corpus(str(tmp_path), cfg)
    mock_ic.assert_called_once_with(str(tmp_path), cfg)


@pytest.mark.unit
@patch("podcast_scraper.search.indexer.index_corpus")
def test_maybe_index_corpus_skips_when_skip_auto_vector_index(
    mock_ic: MagicMock, tmp_path: Path
) -> None:
    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_backend="faiss",
        vector_embedding_model="minilm-l6",
        skip_auto_vector_index=True,
    )
    maybe_index_corpus(str(tmp_path), cfg)
    mock_ic.assert_not_called()


@pytest.mark.unit
@patch("podcast_scraper.search.indexer.embedding_loader.encode")
def test_index_corpus_vector_index_types_insight_only(mock_encode, tmp_path: Path) -> None:
    mock_encode.side_effect = lambda texts, *a, **k: _fake_embeddings(list(texts))

    out = tmp_path / "run"
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True)
    trx_dir = out / "transcripts"
    trx_dir.mkdir(parents=True)
    (trx_dir / "ep.txt").write_text("Hello world transcript sample.", encoding="utf-8")

    gi = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": "ep:vt-1",
        "nodes": [
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {
                    "text": "Only insight.",
                    "episode_id": "ep:vt-1",
                    "grounded": True,
                },
            },
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {
                    "text": "Hello",
                    "episode_id": "ep:vt-1",
                    "char_start": 0,
                    "char_end": 5,
                },
            },
        ],
        "edges": [],
    }
    (meta_dir / "ep1.gi.json").write_text(json.dumps(gi), encoding="utf-8")

    doc = {
        "feed": {"title": "F", "url": "https://e.com/f", "feed_id": "feed:f1"},
        "episode": {"title": "T", "episode_id": "ep:vt-1"},
        "content": {"transcript_file_path": "transcripts/ep.txt"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(out),
            "config_snapshot": {},
        },
        "summary": {"generated_at": "2020-01-01T00:00:00", "bullets": ["Bullet."]},
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
        vector_index_types=["insight"],
    )
    stats = index_corpus(str(out), cfg, rebuild=True)
    assert stats.vectors_upserted == 1
    from podcast_scraper.search.faiss_store import METADATA_FILE

    md = json.loads((out / "search" / METADATA_FILE).read_text(encoding="utf-8"))
    for meta in md.values():
        assert meta.get("doc_type") == "insight"


@pytest.mark.unit
@patch("podcast_scraper.search.indexer.embedding_loader.encode")
def test_index_corpus_indexes_kg_topic_and_entity(mock_encode, tmp_path: Path) -> None:
    mock_encode.side_effect = lambda texts, *a, **k: _fake_embeddings(list(texts))

    out = tmp_path / "run"
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True)

    gi = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": "ep:kgx-1",
        "nodes": [],
        "edges": [],
    }
    (meta_dir / "ep1.gi.json").write_text(json.dumps(gi), encoding="utf-8")

    kg = {
        "schema_version": "1.1",
        "episode_id": "ep:kgx-1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2020-01-01T00:00:00Z",
            "transcript_ref": "transcripts/x.txt",
        },
        "nodes": [
            {
                "id": "epn",
                "type": "Episode",
                "properties": {
                    "podcast_id": "pod:1",
                    "title": "Ep",
                    "publish_date": "2020-01-01",
                },
            },
            {
                "id": "top1",
                "type": "Topic",
                "properties": {
                    "label": "Quantum computing",
                    "slug": "quantum-computing",
                    "description": "QC context line.",
                },
            },
            {
                "id": "ent1",
                "type": "Entity",
                "properties": {
                    "name": "Alice Labs",
                    "entity_kind": "organization",
                    "description": "A startup.",
                },
            },
        ],
        "edges": [],
    }
    (meta_dir / "ep1.kg.json").write_text(json.dumps(kg), encoding="utf-8")

    doc = {
        "feed": {"title": "F", "url": "https://e.com/f", "feed_id": "feed:f1"},
        "episode": {"title": "T", "episode_id": "ep:kgx-1"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(out),
            "config_snapshot": {},
        },
        "grounded_insights": {
            "artifact_path": "metadata/ep1.gi.json",
            "insight_count": 0,
            "generated_at": "2020-01-01T00:00:00",
        },
        "knowledge_graph": {
            "artifact_path": "metadata/ep1.kg.json",
            "node_count": 3,
            "edge_count": 0,
            "generated_at": "2020-01-01T00:00:00",
        },
    }
    (meta_dir / "ep1.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_index_path=str(out / "search"),
        vector_embedding_model="minilm-l6",
        vector_index_types=["kg_topic", "kg_entity"],
    )
    stats = index_corpus(str(out), cfg, rebuild=True)
    assert stats.vectors_upserted == 2
    from podcast_scraper.search.faiss_store import METADATA_FILE

    md = json.loads((out / "search" / METADATA_FILE).read_text(encoding="utf-8"))
    types = {m.get("doc_type") for m in md.values()}
    assert types == {"kg_topic", "kg_entity"}


@pytest.mark.unit
@patch("podcast_scraper.search.indexer.embedding_loader.encode")
def test_index_corpus_discovers_nested_feeds_metadata(mock_encode, tmp_path: Path) -> None:
    """Corpus parent with feeds/<id>/metadata indexes both (#505)."""
    mock_encode.side_effect = lambda texts, *a, **k: _fake_embeddings(list(texts))

    corpus = tmp_path / "corpus"
    for fid, eid in (("feedA", "ep:a1"), ("feedB", "ep:b1")):
        out = corpus / "feeds" / fid
        meta_dir = out / "metadata"
        meta_dir.mkdir(parents=True)
        trx_dir = out / "transcripts"
        trx_dir.mkdir(parents=True)
        (trx_dir / f"{eid}.txt").write_text(
            "Hello world transcript sample. Second sentence. Third sentence.",
            encoding="utf-8",
        )
        gi = {
            "schema_version": "1.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": eid,
            "nodes": [
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {"text": "Insight text.", "episode_id": eid, "grounded": True},
                },
            ],
            "edges": [],
        }
        (meta_dir / f"{eid}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
        doc = {
            "feed": {"title": "F", "url": "https://e.com/f", "feed_id": fid},
            "episode": {"title": "T", "episode_id": eid},
            "content": {"transcript_file_path": f"transcripts/{eid}.txt"},
            "processing": {
                "processing_timestamp": "2020-01-01T00:00:00",
                "output_directory": str(out),
                "config_snapshot": {},
            },
            "summary": {"generated_at": "2020-01-01T00:00:00", "bullets": ["B."]},
            "grounded_insights": {
                "artifact_path": f"metadata/{eid}.gi.json",
                "insight_count": 1,
                "generated_at": "2020-01-01T00:00:00",
            },
        }
        (meta_dir / f"{eid}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_index_path=str(corpus / "search"),
        vector_chunk_size_tokens=20,
        vector_chunk_overlap_tokens=8,
        vector_embedding_model="minilm-l6",
    )
    stats = index_corpus(str(corpus), cfg, rebuild=True)
    assert stats.episodes_scanned == 2
    assert stats.episodes_reindexed == 2
    assert stats.vectors_upserted >= 2


@pytest.mark.unit
@patch("podcast_scraper.search.indexer.embedding_loader.encode")
def test_index_corpus_discovers_hybrid_parent_and_feeds_metadata(
    mock_encode, tmp_path: Path
) -> None:
    """Corpus with ``feeds/<id>/metadata`` plus parent ``metadata/`` indexes both (#505 hybrid)."""
    mock_encode.side_effect = lambda texts, *a, **k: _fake_embeddings(list(texts))

    corpus = tmp_path / "corpus"
    # Per-feed subtree (one episode)
    out = corpus / "feeds" / "f1"
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True)
    trx_dir = out / "transcripts"
    trx_dir.mkdir(parents=True)
    eid_feed = "ep:index-hybrid-feed"
    (trx_dir / f"{eid_feed}.txt").write_text(
        "Alpha transcript sample. Second sentence. Third sentence here.",
        encoding="utf-8",
    )
    gi_feed = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": eid_feed,
        "nodes": [
            {
                "id": "insight:f",
                "type": "Insight",
                "properties": {"text": "Feed insight.", "episode_id": eid_feed, "grounded": True},
            },
        ],
        "edges": [],
    }
    (meta_dir / f"{eid_feed}.gi.json").write_text(json.dumps(gi_feed), encoding="utf-8")
    doc_feed = {
        "feed": {"title": "F1", "url": "https://e.com/f1", "feed_id": "feed:f1"},
        "episode": {"title": "T1", "episode_id": eid_feed},
        "content": {"transcript_file_path": f"transcripts/{eid_feed}.txt"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(out),
            "config_snapshot": {},
        },
        "summary": {"generated_at": "2020-01-01T00:00:00", "bullets": ["B1."]},
        "grounded_insights": {
            "artifact_path": f"metadata/{eid_feed}.gi.json",
            "insight_count": 1,
            "generated_at": "2020-01-01T00:00:00",
        },
    }
    (meta_dir / f"{eid_feed}.metadata.json").write_text(json.dumps(doc_feed), encoding="utf-8")

    # Parent-level metadata (legacy / auxiliary)
    root_meta = corpus / "metadata"
    root_trx = corpus / "transcripts"
    root_meta.mkdir(parents=True)
    root_trx.mkdir(parents=True)
    eid_root = "ep:index-hybrid-root"
    (root_trx / f"{eid_root}.txt").write_text(
        "Root transcript sample. Second sentence. Third sentence here.",
        encoding="utf-8",
    )
    gi_root = {
        "schema_version": "1.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": eid_root,
        "nodes": [
            {
                "id": "insight:r",
                "type": "Insight",
                "properties": {"text": "Root insight.", "episode_id": eid_root, "grounded": True},
            },
        ],
        "edges": [],
    }
    (root_meta / f"{eid_root}.gi.json").write_text(json.dumps(gi_root), encoding="utf-8")
    doc_root = {
        "feed": {"title": "Root", "url": "https://e.com/root", "feed_id": "feed:root"},
        "episode": {"title": "Root ep", "episode_id": eid_root},
        "content": {"transcript_file_path": f"transcripts/{eid_root}.txt"},
        "processing": {
            "processing_timestamp": "2020-01-01T00:00:00",
            "output_directory": str(corpus),
            "config_snapshot": {},
        },
        "summary": {"generated_at": "2020-01-01T00:00:00", "bullets": ["B0."]},
        "grounded_insights": {
            "artifact_path": f"metadata/{eid_root}.gi.json",
            "insight_count": 1,
            "generated_at": "2020-01-01T00:00:00",
        },
    }
    (root_meta / f"{eid_root}.metadata.json").write_text(json.dumps(doc_root), encoding="utf-8")

    cfg = Config(
        rss="https://example.com/feed.xml",
        vector_search=True,
        vector_index_path=str(corpus / "search"),
        vector_chunk_size_tokens=20,
        vector_chunk_overlap_tokens=8,
        vector_embedding_model="minilm-l6",
    )
    stats = index_corpus(str(corpus), cfg, rebuild=True)
    assert stats.episodes_scanned == 2
    assert stats.episodes_reindexed == 2
    assert stats.vectors_upserted >= 2
    assert not stats.errors

    fp_path = corpus / "search" / EPISODE_FINGERPRINTS_FILE
    fps = json.loads(fp_path.read_text(encoding="utf-8"))
    assert index_fingerprint_scope_key("feed:f1", eid_feed) in fps
    assert index_fingerprint_scope_key("feed:root", eid_root) in fps
