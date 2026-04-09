"""Integration tests for embedding loader (Issue #435).

Exercises the full load → encode flow with a fake SentenceTransformer backend.
Requires ``sentence-transformers`` (``pip install -e '.[ml]'``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from podcast_scraper.providers.ml import embedding_loader

pytestmark = [pytest.mark.integration]


class TestLoadEmbeddingModelIntegration:
    """Full wiring: Config alias → registry resolve → SentenceTransformer init → encode."""

    @staticmethod
    def _make_fake_st(captured):
        """Return a fake SentenceTransformer class that records constructor kwargs."""

        class FakeST:
            def __init__(self, model_id, device=None, cache_folder=None, **kwargs):
                captured.append(
                    {"model_id": model_id, "device": device, "cache_folder": cache_folder, **kwargs}
                )

            def encode(self, texts, normalize_embeddings=True, batch_size=64):
                return [[0.1, 0.2, 0.3] for _ in texts]

        return FakeST

    def test_load_and_encode_single_text(self, monkeypatch, tmp_path):
        """load_embedding_model → encode() returns embedding vectors."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            self._make_fake_st(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        # Clear process-level cache so our fake is used
        embedding_loader._embedding_models.clear()

        result = embedding_loader.encode(
            "Hello world",
            model_id="minilm-l6",
            device="cpu",
        )

        assert isinstance(result, list)
        assert len(result) == 3
        assert captured[0]["model_id"] == "all-MiniLM-L6-v2"
        assert captured[0]["device"] == "cpu"
        assert captured[0]["local_files_only"] is True

    def test_load_and_encode_batch(self, monkeypatch, tmp_path):
        """encode() with a list of texts returns one vector per text."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            self._make_fake_st(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        embedding_loader._embedding_models.clear()

        result = embedding_loader.encode(
            ["text one", "text two", "text three"],
            model_id="minilm-l6",
            device="cpu",
        )

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(v) == 3 for v in result)

    def test_allow_download_omits_local_files_only(self, monkeypatch, tmp_path):
        """allow_download=True does not pass local_files_only to SentenceTransformer."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            self._make_fake_st(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        embedding_loader._embedding_models.clear()

        embedding_loader.load_embedding_model("minilm-l6", device="cpu", allow_download=True)

        assert captured[0].get("local_files_only") is not True

    def test_get_embedding_model_caches_instance(self, monkeypatch, tmp_path):
        """get_embedding_model returns the same instance on repeated calls."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            self._make_fake_st(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        embedding_loader._embedding_models.clear()

        a = embedding_loader.get_embedding_model("minilm-l6", device="cpu")
        b = embedding_loader.get_embedding_model("minilm-l6", device="cpu")

        assert a is b
        assert len(captured) == 1

    def test_strips_sentence_transformers_prefix(self, monkeypatch, tmp_path):
        """Full HF id sentence-transformers/X is passed to ST as X."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            self._make_fake_st(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        embedding_loader._embedding_models.clear()

        embedding_loader.load_embedding_model(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            allow_download=True,
        )

        assert captured[0]["model_id"] == "all-MiniLM-L6-v2"
