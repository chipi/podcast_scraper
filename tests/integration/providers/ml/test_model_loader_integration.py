"""Integration tests for model_loader preload and cache-check functions (Issue #435).

Exercises the preload_evidence_models and is_evidence_model_cached wiring with
fake backends.  Requires ``transformers`` and ``sentence_transformers``
(``pip install -e '.[ml]'``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

from podcast_scraper.providers.ml import model_loader
from podcast_scraper.providers.ml.model_registry import ModelRegistry

pytestmark = [pytest.mark.integration]


class TestIsEvidenceModelCached:
    """is_evidence_model_cached: checks cache dir for model artifacts."""

    def test_returns_false_for_missing_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        assert model_loader.is_evidence_model_cached("minilm-l6") is False

    def test_returns_true_when_cache_dir_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        resolved = ModelRegistry.resolve_evidence_model_id("minilm-l6")
        cache_name = resolved.replace("/", "--")
        (tmp_path / f"models--{cache_name}").mkdir()

        assert model_loader.is_evidence_model_cached("minilm-l6") is True

    def test_unknown_alias_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        assert model_loader.is_evidence_model_cached("nonexistent-model") is False


class TestPreloadEvidenceModels:
    """preload_evidence_models: orchestrates download of embedding, QA, NLI models."""

    def test_skips_already_cached_models(self, tmp_path, monkeypatch):
        """Models whose cache dir already exists are not re-downloaded."""
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        resolved_emb = ModelRegistry.resolve_evidence_model_id("minilm-l6")
        (tmp_path / f"models--{resolved_emb.replace('/', '--')}").mkdir()

        download_calls: list = []
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader._download_sentence_transformer_for_cache",
            lambda mid: download_calls.append(mid),
        )
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader._download_qa_pipeline_for_cache",
            lambda mid: download_calls.append(mid),
        )
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader._download_nli_cross_encoder_for_cache",
            lambda mid: download_calls.append(mid),
        )

        model_loader.preload_evidence_models(
            embedding_models=["minilm-l6"],
            qa_models=[],
            nli_models=[],
        )

        assert resolved_emb not in download_calls

    def test_downloads_uncached_models(self, tmp_path, monkeypatch):
        """Uncached models trigger download functions."""
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        download_calls: list = []
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader._download_sentence_transformer_for_cache",
            lambda mid: download_calls.append(("st", mid)),
        )
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader._download_qa_pipeline_for_cache",
            lambda mid: download_calls.append(("qa", mid)),
        )
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader._download_nli_cross_encoder_for_cache",
            lambda mid: download_calls.append(("nli", mid)),
        )

        model_loader.preload_evidence_models(
            embedding_models=["minilm-l6"],
            qa_models=["roberta-squad2"],
            nli_models=["nli-deberta-base"],
        )

        types = [t for t, _ in download_calls]
        assert "st" in types
        assert "qa" in types
        assert "nli" in types


class TestBuildHuggingfaceQaPipeline:
    """build_huggingface_qa_pipeline: wiring to transformers.pipeline."""

    def test_offline_mode_passes_local_files_only(self, monkeypatch, tmp_path):
        """local_files_only=True is forwarded to model_kwargs."""
        captured: list = []

        def fake_pipeline(task, model, device, model_kwargs=None, **kw):
            captured.append({"task": task, "model": model, "model_kwargs": model_kwargs, **kw})
            return MagicMock()

        monkeypatch.setattr("transformers.pipeline", fake_pipeline)
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )

        model_loader.build_huggingface_qa_pipeline(
            "deepset/roberta-base-squad2",
            device=-1,
            local_files_only=True,
        )

        assert len(captured) == 1
        assert captured[0]["task"] == "question-answering"
        assert captured[0]["model_kwargs"]["local_files_only"] is True
