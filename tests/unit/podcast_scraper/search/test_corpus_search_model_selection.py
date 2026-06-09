"""run_corpus_search must embed the query with the SAME model that built the index.

Querying a FAISS index with a different embedding model produces a vector in a
foreign space -> silently wrong nearest neighbours (the #897 family of failures).
run_corpus_search defaults the query model to the persisted index model, and only
warns (does not hard-fail) when the caller forces a mismatching --embedding-model.
These drive the real load -> stats().embedding_model -> selection -> encode path
against a real persisted store and pin both behaviours.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from podcast_scraper.providers.ml import embedding_loader
from podcast_scraper.search import corpus_search
from podcast_scraper.search.faiss_store import FaissVectorStore

pytestmark = pytest.mark.unit

_INDEX_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_index(output_dir: Path) -> None:
    index_dir = output_dir / "search"
    store = FaissVectorStore(4, embedding_model=_INDEX_MODEL, index_dir=index_dir)
    store.upsert("quote:a", [1.0, 0.0, 0.0, 0.0], {"doc_type": "quote", "text": "alpha"})
    store.upsert("quote:b", [0.0, 1.0, 0.0, 0.0], {"doc_type": "quote", "text": "beta"})
    store.persist()


def _capture_encode(monkeypatch) -> dict:
    captured: dict = {}

    def fake_encode(text, model_id, **kwargs):
        captured["model_id"] = model_id
        return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(embedding_loader, "encode", fake_encode)
    return captured


def test_query_defaults_to_index_model(tmp_path, monkeypatch):
    _build_index(tmp_path)
    captured = _capture_encode(monkeypatch)

    outcome = corpus_search.run_corpus_search(tmp_path, "alpha")

    assert outcome.error is None
    # No --embedding-model supplied -> query embeds with the persisted index model.
    assert captured["model_id"] == _INDEX_MODEL


def test_mismatched_embedding_model_warns_but_proceeds(tmp_path, monkeypatch, caplog):
    _build_index(tmp_path)
    captured = _capture_encode(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=corpus_search.logger.name):
        outcome = corpus_search.run_corpus_search(
            tmp_path, "alpha", embedding_model="BAAI/bge-small-en-v1.5"
        )

    assert outcome.error is None
    # Current contract: a forced mismatching model is honoured (used for the query)
    # but a divergence warning is emitted. This pins the behaviour so any future
    # change (e.g. hard-fail, or coerce to the index model) is a deliberate edit.
    assert captured["model_id"] == "BAAI/bge-small-en-v1.5"
    assert any(
        "differs from index model" in r.getMessage() for r in caplog.records
    ), "expected an embedding-model divergence warning"


def test_unknown_embedding_model_does_not_crash(tmp_path, monkeypatch, caplog):
    # embedding_model arrives unvalidated from the CLI. An unknown alias (no "/" and
    # not registered) makes resolve_evidence_model_id raise -- which used to escape
    # run_corpus_search as an uncaught traceback. It must degrade to a divergence
    # warning + a clean outcome instead. Here encode is stubbed so we isolate the
    # resolve guard; in production the bogus model would fail encode -> embed_failed.
    _build_index(tmp_path)
    _capture_encode(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=corpus_search.logger.name):
        outcome = corpus_search.run_corpus_search(tmp_path, "alpha", embedding_model="minilm-typo")

    assert outcome.error is None  # no exception escaped
    assert any("differs from index model" in r.getMessage() for r in caplog.records)
