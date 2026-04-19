"""Unit tests for embedding loader (Issue #435)."""

import builtins
import sys
from unittest.mock import MagicMock

import pytest

from podcast_scraper.providers.ml import embedding_loader

pytestmark = [pytest.mark.unit]


class TestSentenceTransformerLoadName:
    """``_sentence_transformer_load_name`` strips org prefix."""

    def test_strips_sentence_transformers_prefix(self):
        assert (
            embedding_loader._sentence_transformer_load_name(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            == "all-MiniLM-L6-v2"
        )

    def test_passes_through_other_ids(self):
        assert embedding_loader._sentence_transformer_load_name("all-MiniLM-L6-v2") == (
            "all-MiniLM-L6-v2"
        )


class TestGetDevice:
    """``_get_device`` respects explicit value and torch when present."""

    def test_explicit_device(self):
        assert embedding_loader._get_device("CPU") == "cpu"

    def test_auto_prefers_cuda_when_available(self, monkeypatch):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        assert embedding_loader._get_device(None) == "cuda"

    def test_auto_cpu_when_torch_import_fails(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, globals_arg=None, locals_arg=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("unavailable")
            return real_import(name, globals_arg, locals_arg, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert embedding_loader._get_device(None) == "cpu"


class TestCosineSimilarity:
    """Tests for cosine_similarity helper."""

    def test_identical_vectors(self):
        """Identical normalized vectors have similarity 1.0."""
        v = [0.6, 0.8]
        assert embedding_loader.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert embedding_loader.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_mismatched_length_raises(self):
        """Different-length vectors raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            embedding_loader.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestEncodeMocked:
    """Tests for encode() with mocked model (no real model load)."""

    def test_encode_single_returns_list_of_floats(self, monkeypatch):
        """encode(single_str) returns a single list of floats."""
        fake_vec = [0.1] * 384

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, batch_size=64):
                return [fake_vec] * len(texts)

        monkeypatch.setattr(
            embedding_loader,
            "get_embedding_model",
            lambda *args, **kwargs: FakeModel(),
        )
        out = embedding_loader.encode("hello", model_id="minilm-l6")
        assert isinstance(out, list)
        assert len(out) == 384
        assert all(isinstance(x, float) for x in out)

    def test_encode_multi_returns_list_of_lists(self, monkeypatch):
        """encode(list of str) returns list of lists."""
        fake_vec = [0.2] * 384

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, batch_size=64):
                return [fake_vec] * len(texts)

        monkeypatch.setattr(
            embedding_loader,
            "get_embedding_model",
            lambda *args, **kwargs: FakeModel(),
        )
        out = embedding_loader.encode(["a", "b"], model_id="minilm-l6")
        assert isinstance(out, list)
        assert len(out) == 2
        assert len(out[0]) == 384
        assert len(out[1]) == 384

    def test_encode_passes_batch_size_to_model(self, monkeypatch):
        """encode forwards batch_size to model.encode."""
        seen = {}

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, batch_size=64):
                seen["batch_size"] = batch_size
                return [[0.1, 0.2]] * len(texts)

        monkeypatch.setattr(
            embedding_loader,
            "get_embedding_model",
            lambda *args, **kwargs: FakeModel(),
        )
        embedding_loader.encode(["x", "y"], model_id="minilm-l6", batch_size=128)
        assert seen["batch_size"] == 128

    def test_encode_return_numpy_single(self, monkeypatch):
        """return_numpy=True returns ndarray row for one text."""
        import numpy as np

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, batch_size=64):
                return np.array([[0.5, 0.5]], dtype=np.float32)

        monkeypatch.setattr(
            embedding_loader,
            "get_embedding_model",
            lambda *args, **kwargs: FakeModel(),
        )
        out = embedding_loader.encode("hi", model_id="minilm-l6", return_numpy=True)
        assert hasattr(out, "shape")
        assert out.shape == (2,)

    def test_encode_return_numpy_multi(self, monkeypatch):
        """return_numpy=True with multiple texts returns full embedding matrix."""
        import numpy as np

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, batch_size=64):
                return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        monkeypatch.setattr(
            embedding_loader,
            "get_embedding_model",
            lambda *args, **kwargs: FakeModel(),
        )
        out = embedding_loader.encode(["a", "b"], model_id="minilm-l6", return_numpy=True)
        assert hasattr(out, "shape")
        assert out.shape == (2, 2)

    def test_encode_single_coerces_sequence_without_tolist(self, monkeypatch):
        """``to_list`` falls back to ``list()`` when embedding row has no ``tolist``."""

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, batch_size=64):
                return [(0.25, 0.75)]

        monkeypatch.setattr(
            embedding_loader,
            "get_embedding_model",
            lambda *args, **kwargs: FakeModel(),
        )
        out = embedding_loader.encode("only", model_id="minilm-l6")
        assert out == [0.25, 0.75]

    def test_get_embedding_model_keyed_by_model_id(self, monkeypatch):
        """Different model_id values load separate cached instances."""
        loads = []

        class FakeModel:
            pass

        def fake_load(model_id, device=None, cache_dir=None, *, allow_download=False):
            loads.append(model_id)
            return FakeModel()

        monkeypatch.setattr(embedding_loader, "load_embedding_model", fake_load)
        monkeypatch.setattr(embedding_loader, "_embedding_models", {})
        m1 = embedding_loader.get_embedding_model("minilm-l6")
        m2 = embedding_loader.get_embedding_model("minilm-l6")
        m3 = embedding_loader.get_embedding_model("mpnet-base")
        assert m1 is m2
        assert m1 is not m3
        assert loads == ["minilm-l6", "mpnet-base"]

    def test_get_embedding_model_separate_cache_per_device_and_allow_download(self, monkeypatch):
        """Cache key includes device and allow_download so variants do not collide."""
        loads = []

        def fake_load(model_id, device=None, cache_dir=None, *, allow_download=False):
            loads.append((device, cache_dir, allow_download))
            return object()

        monkeypatch.setattr(embedding_loader, "load_embedding_model", fake_load)
        monkeypatch.setattr(embedding_loader, "_embedding_models", {})
        a = embedding_loader.get_embedding_model("minilm-l6", device="cpu", allow_download=False)
        b = embedding_loader.get_embedding_model("minilm-l6", device="cuda", allow_download=False)
        c = embedding_loader.get_embedding_model("minilm-l6", device="cpu", allow_download=True)
        assert a is not b
        assert a is not c
        assert len(loads) == 3


class TestEffectiveCacheFolder:
    """Tests for default HF hub folder alignment with preload."""

    def test_uses_get_transformers_cache_dir_when_missing(self, monkeypatch):
        """Omitted cache_dir resolves via get_transformers_cache_dir (project .cache, etc.)."""
        from pathlib import Path

        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: Path("/expected/huggingface/hub"),
        )
        assert embedding_loader._effective_cache_folder(None) == "/expected/huggingface/hub"
        assert embedding_loader._effective_cache_folder("") == "/expected/huggingface/hub"
        assert embedding_loader._effective_cache_folder("  /custom/cache  ") == "/custom/cache"


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model (resolve + load wiring).

    Injects a fake ``sentence_transformers`` module into ``sys.modules`` so the
    runtime ``from sentence_transformers import SentenceTransformer`` resolves
    without the real package installed.
    """

    @staticmethod
    def _inject_fake_st(monkeypatch, captured):
        """Install a fake ``sentence_transformers`` module and return it."""
        from types import ModuleType

        fake_mod = ModuleType("sentence_transformers")

        def fake_st(model_id, device=None, cache_folder=None, **kwargs):
            captured.append({"model_id": model_id, "cache_folder": cache_folder, **kwargs})
            return type("Fake", (), {"encode": lambda self, t, normalize_embeddings: []})()

        fake_mod.SentenceTransformer = fake_st  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
        return fake_mod

    def test_load_embedding_model_resolves_alias(self, monkeypatch):
        """load_embedding_model resolves alias via registry before loading."""
        from pathlib import Path

        from podcast_scraper.providers.ml.model_registry import ModelRegistry

        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: Path("/expected/huggingface/hub"),
        )
        captured: list = []
        self._inject_fake_st(monkeypatch, captured)

        embedding_loader.load_embedding_model("minilm-l6", device="cpu")
        assert len(captured) == 1
        assert captured[0]["model_id"] == "all-MiniLM-L6-v2"
        assert (
            ModelRegistry.resolve_evidence_model_id("minilm-l6")
            == "sentence-transformers/all-MiniLM-L6-v2"
        )
        # local_files_only is only passed when the SentenceTransformer constructor
        # explicitly accepts it (sentence-transformers >= 3.x). The fake constructor
        # uses **kwargs so introspection won't find the param — same as 2.x.
        # On real 3.x installs this would be True; on 2.x / fake it's absent.
        import inspect

        from sentence_transformers import SentenceTransformer as _ST

        _st_params = set(inspect.signature(_ST).parameters)
        if "local_files_only" in _st_params:
            assert captured[0].get("local_files_only") is True
        else:
            assert "local_files_only" not in captured[0]
        assert captured[0]["cache_folder"] == "/expected/huggingface/hub"

    def test_load_embedding_model_allow_download_omits_local_files_only(self, monkeypatch):
        """allow_download=True does not force local_files_only."""
        from pathlib import Path

        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: Path("/hub/default"),
        )
        captured: list = []
        self._inject_fake_st(monkeypatch, captured)

        embedding_loader.load_embedding_model("minilm-l6", device="cpu", allow_download=True)
        assert captured[0].get("local_files_only") is not True

    def test_load_embedding_model_strips_sentence_transformers_prefix(self, monkeypatch):
        """Full HF id sentence-transformers/X is passed to ST as X."""
        from pathlib import Path

        monkeypatch.setattr(
            "podcast_scraper.cache.directories.get_transformers_cache_dir",
            lambda: Path("/hub/default"),
        )
        captured: list = []
        self._inject_fake_st(monkeypatch, captured)

        embedding_loader.load_embedding_model(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            allow_download=True,
        )
        assert captured[0]["model_id"] == "all-MiniLM-L6-v2"
        assert captured[0]["cache_folder"] == "/hub/default"
