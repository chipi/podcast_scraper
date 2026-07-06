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
            "podcast_scraper.providers.ml.model_loader._download_hf_evidence_model",
            lambda kind, mid: download_calls.append((kind, mid)),
        )

        model_loader.preload_evidence_models(
            embedding_models=["minilm-l6"],
            qa_models=[],
            nli_models=[],
        )

        # Embedding already cached — no calls at all.
        assert resolved_emb not in [m for _, m in download_calls]
        assert download_calls == []

    def test_downloads_uncached_models(self, tmp_path, monkeypatch):
        """Uncached models trigger download functions."""
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        download_calls: list = []
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader._download_hf_evidence_model",
            lambda kind, mid: download_calls.append((kind, mid)),
        )

        model_loader.preload_evidence_models(
            embedding_models=["minilm-l6"],
            qa_models=["roberta-squad2"],
            nli_models=["nli-deberta-base"],
        )

        kinds = [k for k, _ in download_calls]
        assert "embedding" in kinds
        assert "qa" in kinds
        assert "nli" in kinds


class TestDownloadHFEvidenceModel:
    """_download_hf_evidence_model: kind-dispatched cache warmer.

    Post-#382 Phase G: three parallel _download_*_for_cache helpers
    collapsed into one function with a Literal["qa", "nli", "embedding"]
    kind discriminator. build_huggingface_qa_pipeline (v4-era test target)
    is gone — transformers v5 removed pipeline("question-answering").
    """

    def test_qa_kind_calls_auto_model_for_question_answering(self, monkeypatch, tmp_path):
        """QA path instantiates AutoModelForQuestionAnswering + AutoTokenizer."""
        captured: list = []

        def fake_auto_tok(model_id, **kw):
            captured.append(("tokenizer", model_id, kw))
            return MagicMock()

        def fake_auto_qa(model_id, **kw):
            captured.append(("model", model_id, kw))
            return MagicMock()

        import sys

        fake_transformers = MagicMock()
        fake_transformers.AutoTokenizer.from_pretrained.side_effect = fake_auto_tok
        fake_transformers.AutoModelForQuestionAnswering.from_pretrained.side_effect = fake_auto_qa
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )

        model_loader._download_hf_evidence_model("qa", "deepset/roberta-base-squad2")

        # Both tokenizer + model loaded with standard preload kwargs
        # (local_files_only=False during preload — this is the only path
        # allowed to hit the hub).
        kinds = [k for k, *_ in captured]
        assert kinds == ["tokenizer", "model"]
        for _, model_id, kw in captured:
            assert model_id == "deepset/roberta-base-squad2"
            assert kw["local_files_only"] is False
            assert kw["trust_remote_code"] is False
            assert kw["low_cpu_mem_usage"] is False

    def test_unknown_kind_raises(self, tmp_path, monkeypatch):
        """kind='wat' — explicit ValueError, not silent fallthrough."""
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        import pytest

        with pytest.raises(ValueError, match="Unknown evidence kind"):
            model_loader._download_hf_evidence_model("wat", "x/y")  # type: ignore[arg-type]
