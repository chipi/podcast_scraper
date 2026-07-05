"""Unit tests closing the last patch-coverage gaps introduced by #382.

Covers small, mock-testable branches that the fast-tier tests don't reach:

- ``resolve_evidence_device`` — CUDA and MPS positive branches
  (base tests only cover the CPU fallback + explicit-device paths).
- ``hf_seq2seq_backend._detect_default_device`` — MPS and CUDA positive
  branches (same reason).
- ``embedding_loader.load_embedding_model`` — thin wrapper delegation.
- ``nli_loader`` — helper edge cases (``_scalar_to_float`` list fallback,
  ``_entailment_class_index`` default, ``_softmax`` degenerate exps,
  ``NLIEvidenceBackend.predict_scores`` meta-device catch,
  module-level ``load_nli_model`` / ``get_nli_model`` /
  ``entailment_scores_batch`` wrappers).
"""

from __future__ import annotations

from typing import List
from unittest import mock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.module_ml_providers]


# ---- Device detection: CUDA / MPS branches -------------------------------


class _FakeTorchBackendsWithMPS:
    class mps:  # noqa: N801
        @staticmethod
        def is_available() -> bool:
            return True


class _FakeTorchBackendsNoMPS:
    pass  # no `mps` attribute at all


def _fake_torch_module(*, cuda_available: bool, mps_available: bool):
    m = mock.MagicMock()
    m.cuda.is_available.return_value = cuda_available
    if mps_available:
        m.backends = _FakeTorchBackendsWithMPS()
    else:
        m.backends = _FakeTorchBackendsNoMPS()
    return m


class TestResolveEvidenceDeviceCudaAndMps:
    def test_cuda_available_returns_cuda(self, monkeypatch):
        from podcast_scraper.providers.ml import hf_evidence_backend

        monkeypatch.setitem(
            __import__("sys").modules,
            "torch",
            _fake_torch_module(cuda_available=True, mps_available=False),
        )
        assert hf_evidence_backend.resolve_evidence_device(None) == "cuda"

    def test_mps_available_returns_mps_when_subclass_supports(self, monkeypatch):
        from podcast_scraper.providers.ml import hf_evidence_backend

        monkeypatch.setitem(
            __import__("sys").modules,
            "torch",
            _fake_torch_module(cuda_available=False, mps_available=True),
        )
        assert hf_evidence_backend.resolve_evidence_device(None, mps_supported=True) == "mps"

    def test_mps_available_but_subclass_disallows(self, monkeypatch):
        from podcast_scraper.providers.ml import hf_evidence_backend

        monkeypatch.setitem(
            __import__("sys").modules,
            "torch",
            _fake_torch_module(cuda_available=False, mps_available=True),
        )
        assert hf_evidence_backend.resolve_evidence_device(None, mps_supported=False) == "cpu"


class TestResolveSeq2SeqDeviceMpsAndCuda:
    def test_mps_available_returns_mps_first(self, monkeypatch):
        from podcast_scraper.providers.ml import hf_seq2seq_backend

        monkeypatch.setitem(
            __import__("sys").modules,
            "torch",
            _fake_torch_module(cuda_available=True, mps_available=True),
        )
        # MPS takes priority over CUDA for seq2seq (Apple Silicon story)
        assert hf_seq2seq_backend._detect_default_device() == "mps"

    def test_cuda_only_returns_cuda(self, monkeypatch):
        from podcast_scraper.providers.ml import hf_seq2seq_backend

        monkeypatch.setitem(
            __import__("sys").modules,
            "torch",
            _fake_torch_module(cuda_available=True, mps_available=False),
        )
        assert hf_seq2seq_backend._detect_default_device() == "cuda"


# ---- embedding_loader.load_embedding_model wrapper -----------------------


class TestLoadEmbeddingModelWrapper:
    def test_load_embedding_model_returns_backend_model(self, monkeypatch):
        from podcast_scraper.providers.ml import embedding_loader

        fake_model = mock.MagicMock(name="fake_st_model")

        def fake_ensure(self):
            self.model = fake_model

        monkeypatch.setattr(
            embedding_loader.EmbeddingEvidenceBackend, "_ensure_loaded", fake_ensure
        )
        result = embedding_loader.load_embedding_model(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_dir=None,
            allow_download=False,
        )
        assert result is fake_model


# ---- nli_loader helpers --------------------------------------------------


class TestNliScalarToFloat:
    def test_list_fallback_recurses_into_first_element(self):
        from podcast_scraper.providers.ml import nli_loader

        # Pass a list containing a scalar — recursion should extract the scalar.
        assert nli_loader._scalar_to_float([0.42], fallback=-1.0) == pytest.approx(0.42)

    def test_empty_container_returns_fallback(self):
        from podcast_scraper.providers.ml import nli_loader

        # __len__=0 → fallback path.
        assert nli_loader._scalar_to_float([], fallback=0.7) == 0.7


class TestNliEntailmentClassIndexFallback:
    def test_defaults_to_2_when_id2label_missing(self):
        from podcast_scraper.providers.ml import nli_loader

        model = mock.MagicMock()
        model.model = None  # no inner model -> no config -> no id2label
        assert nli_loader._entailment_class_index(model) == 2

    def test_defaults_to_2_when_id2label_lacks_entailment(self):
        from podcast_scraper.providers.ml import nli_loader

        inner = mock.MagicMock()
        inner.config.id2label = {0: "yes", 1: "no"}  # no "entailment"
        model = mock.MagicMock()
        model.model = inner
        assert nli_loader._entailment_class_index(model) == 2


class TestNliSoftmaxDegenerate:
    def test_softmax_empty(self):
        from podcast_scraper.providers.ml import nli_loader

        assert nli_loader._softmax([]) == []

    def test_softmax_uniform_extreme_values(self):
        from podcast_scraper.providers.ml import nli_loader

        # Softmax must sum to ~1 even for wildly negative logits.
        out = nli_loader._softmax([-1000.0, -1000.0, -1000.0])
        assert len(out) == 3
        assert sum(out) == pytest.approx(1.0)


class TestNliPredictScoresMetaCatch:
    def test_meta_device_runtime_error_returns_zeros(self):
        from podcast_scraper.providers.ml import nli_loader

        backend = nli_loader.NLIEvidenceBackend.__new__(nli_loader.NLIEvidenceBackend)
        # Stub the model to raise the meta-device RuntimeError shape.
        fake_model = mock.MagicMock()
        fake_model.predict.side_effect = RuntimeError("Cannot copy out of meta tensor; no data!")
        backend.model = fake_model
        scores = backend.predict_scores([("p1", "h1"), ("p2", "h2"), ("p3", "h3")])
        assert scores == [0.0, 0.0, 0.0]

    def test_non_meta_runtime_error_reraises(self):
        from podcast_scraper.providers.ml import nli_loader

        backend = nli_loader.NLIEvidenceBackend.__new__(nli_loader.NLIEvidenceBackend)
        fake_model = mock.MagicMock()
        fake_model.predict.side_effect = RuntimeError("something else")
        backend.model = fake_model
        with pytest.raises(RuntimeError, match="something else"):
            backend.predict_scores([("p", "h")])


class TestNliModuleLevelWrappers:
    def test_load_nli_model_returns_backend_model(self, monkeypatch):
        from podcast_scraper.providers.ml import nli_loader

        fake_model = mock.MagicMock(name="fake_cross_encoder")

        def fake_ensure(self):
            self.model = fake_model

        monkeypatch.setattr(nli_loader.NLIEvidenceBackend, "_ensure_loaded", fake_ensure)
        result = nli_loader.load_nli_model("nli-deberta-base", device="cpu")
        assert result is fake_model

    def test_get_nli_model_returns_cached_backend_model(self, monkeypatch):
        from podcast_scraper.providers.ml import nli_loader

        fake_model = mock.MagicMock(name="cached_cross_encoder")

        def fake_load(self):
            self.model = fake_model

        monkeypatch.setattr(nli_loader.NLIEvidenceBackend, "_load", fake_load)
        nli_loader.NLIEvidenceBackend.clear_cache()

        result = nli_loader.get_nli_model("nli-deberta-base", device="cpu")
        assert result is fake_model

    def test_entailment_scores_batch_delegates_to_backend(self, monkeypatch):
        from podcast_scraper.providers.ml import nli_loader

        def fake_load(self):
            self.model = mock.MagicMock()

        monkeypatch.setattr(nli_loader.NLIEvidenceBackend, "_load", fake_load)

        def fake_predict(self, pairs):
            return [0.9] * len(pairs)

        monkeypatch.setattr(nli_loader.NLIEvidenceBackend, "predict_scores", fake_predict)
        nli_loader.NLIEvidenceBackend.clear_cache()

        pairs: List[tuple[str, str]] = [("p1", "h1"), ("p2", "h2")]
        scores = nli_loader.entailment_scores_batch(
            pairs, model_id="nli-deberta-base", device="cpu"
        )
        assert scores == [0.9, 0.9]
