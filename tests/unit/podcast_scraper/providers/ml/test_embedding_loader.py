"""Unit tests for embedding loader (Issue #435)."""

import pytest

from podcast_scraper.providers.ml import embedding_loader

pytestmark = [pytest.mark.unit]


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


try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers required for load path test",
)
class TestLoadEmbeddingModel:
    """Tests for load_embedding_model (resolve + load wiring)."""

    def test_load_embedding_model_resolves_alias(self, monkeypatch):
        """load_embedding_model resolves alias via registry before loading."""
        from podcast_scraper.providers.ml.model_registry import ModelRegistry

        captured = []

        def fake_st(model_id, device=None, cache_folder=None, **kwargs):
            captured.append((model_id, kwargs.get("local_files_only")))
            return type("Fake", (), {"encode": lambda self, texts, normalize_embeddings: []})()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            fake_st,
            raising=False,
        )
        embedding_loader.load_embedding_model("minilm-l6", device="cpu")
        assert len(captured) == 1
        assert captured[0][0] == ModelRegistry.resolve_evidence_model_id("minilm-l6")
        assert captured[0][1] is True

    def test_load_embedding_model_allow_download_omits_local_files_only(self, monkeypatch):
        """allow_download=True does not force local_files_only."""
        captured = []

        def fake_st(model_id, device=None, cache_folder=None, **kwargs):
            captured.append(kwargs)
            return type("Fake", (), {})()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            fake_st,
            raising=False,
        )
        embedding_loader.load_embedding_model("minilm-l6", device="cpu", allow_download=True)
        assert captured[0].get("local_files_only") is not True
