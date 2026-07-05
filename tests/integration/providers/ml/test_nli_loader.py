"""Integration tests for NLI loader (Issue #435, updated for #382 Phase E).

Post-#382 the module wraps :class:`NLIEvidenceBackend`. Tests install a
fake ``backend.model`` (CrossEncoder-like) and let ``backend.predict_scores``
run its real logic (softmax + id2label sniff + shape padding), so the
tests still exercise the interesting NLI post-processing paths.
"""

from unittest import mock

import pytest

from podcast_scraper.providers.ml import nli_loader
from podcast_scraper.providers.ml.nli_loader import NLIEvidenceBackend

pytestmark = [pytest.mark.integration]


def _install_fake_backend(monkeypatch, model_obj):
    """Route NLIEvidenceBackend.get_or_load to a backend with ``model=model_obj``."""
    backend = NLIEvidenceBackend.__new__(NLIEvidenceBackend)
    backend.model = model_obj
    backend.resolved_id = "test/nli"
    backend.device = "cpu"
    backend._loaded = True
    backend.extras = {}
    monkeypatch.setattr(
        NLIEvidenceBackend,
        "get_or_load",
        classmethod(lambda cls, *a, **kw: backend),
    )


class TestEntailmentScoreMocked:
    """Backend `predict_scores` runs real logic against a fake model."""

    def test_entailment_score_returns_float(self, monkeypatch):
        model = mock.Mock()
        model.predict.return_value = [0.85]  # scalar-row = clamped raw
        # Give it a minimal id2label so _entailment_class_index works.
        model.model = mock.Mock()
        model.model.config = mock.Mock()
        model.model.config.id2label = {0: "entailment"}
        _install_fake_backend(monkeypatch, model)

        score = nli_loader.entailment_score(
            premise="The cat sat on the mat.",
            hypothesis="A cat was on a mat.",
            model_id="nli-deberta-base",
        )
        assert isinstance(score, float)
        assert score == 0.85

    def test_entailment_scores_batch_returns_list(self, monkeypatch):
        model = mock.Mock()
        model.predict.return_value = [[0.7], [0.9]]
        model.model = mock.Mock()
        model.model.config = mock.Mock()
        model.model.config.id2label = {0: "entailment"}
        _install_fake_backend(monkeypatch, model)

        scores = nli_loader.entailment_scores_batch(
            pairs=[("p1", "h1"), ("p2", "h2")],
            model_id="nli-deberta-base",
        )
        assert len(scores) == 2
        assert scores[0] == 0.7
        assert scores[1] == 0.9

    def test_entailment_score_three_class_logits_1d(self, monkeypatch):
        """3-class logits → softmax P(entailment)."""
        model = mock.Mock()
        model.model = mock.Mock()
        model.model.config = mock.Mock()
        model.model.config.id2label = {0: "c", 1: "n", 2: "e"}
        model.predict.return_value = [[-2.0, -2.0, 4.0]]
        _install_fake_backend(monkeypatch, model)

        score = nli_loader.entailment_score("p", "h", "nli-deberta-base")
        assert score > 0.9

    def test_entailment_score_respects_id2label_entailment_index(self, monkeypatch):
        """Use config id2label to find entailment column (not always index 2)."""
        model = mock.Mock()
        model.model = mock.Mock()
        model.model.config = mock.Mock()
        model.model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        model.predict.return_value = [[0.0, 4.0, 0.0]]
        _install_fake_backend(monkeypatch, model)

        score = nli_loader.entailment_score("p", "h", "nli-deberta-base")
        assert score > 0.9

    def test_entailment_scores_batch_two_by_three_logits(self, monkeypatch):
        model = mock.Mock()
        model.model = mock.Mock()
        model.model.config = mock.Mock()
        model.model.config.id2label = {0: "c", 1: "n", 2: "e"}
        model.predict.return_value = [[-2.0, -2.0, 4.0], [4.0, -2.0, -2.0]]
        _install_fake_backend(monkeypatch, model)

        scores = nli_loader.entailment_scores_batch(
            [("a", "b"), ("c", "d")],
            "nli-deberta-base",
        )
        assert len(scores) == 2
        assert scores[0] > 0.9
        assert scores[1] < 0.1

    def test_entailment_score_meta_tensor_fallback(self, monkeypatch):
        """When model.predict raises meta-tensor RuntimeError, fall back to 0.0."""
        model = mock.Mock()
        model.predict.side_effect = RuntimeError("Tensor cannot be called on meta tensors")
        _install_fake_backend(monkeypatch, model)

        score = nli_loader.entailment_score(
            premise="Evidence.",
            hypothesis="Claim.",
            model_id="nli-deberta-base",
        )
        assert score == 0.0


class TestPredictOutputHelpers:
    """Direct module-helper tests — no backend involvement."""

    def test_predict_output_single_logit_row(self):
        class _M:
            model = type(
                "Inner",
                (),
                {"config": type("Cfg", (), {"id2label": {0: "entailment"}})()},
            )()

        scores = nli_loader.predict_output_to_entailment_scores([[0.25]], _M())
        assert scores == [0.25]

    def test_entailment_scores_batch_meta_runtime_returns_zeros(self, monkeypatch):
        model = mock.Mock()
        model.predict.side_effect = RuntimeError("meta tensor predict")
        _install_fake_backend(monkeypatch, model)

        out = nli_loader.entailment_scores_batch([("p", "h"), ("p2", "h2")], "nli-deberta-base")
        assert out == [0.0, 0.0]

    def test_entailment_scores_batch_pads_short_output(self, monkeypatch):
        """predict returns fewer rows than pairs — batch pads to len(pairs) with 0.0."""
        model = mock.Mock()
        model.model = mock.Mock()
        model.model.config = mock.Mock()
        model.model.config.id2label = {0: "c", 1: "n", 2: "e"}
        model.predict.return_value = [[-1.0, -1.0, 3.0]]  # 1 row for 3 pairs
        _install_fake_backend(monkeypatch, model)

        out = nli_loader.entailment_scores_batch(
            [("a", "b"), ("c", "d"), ("e", "f")],
            "nli-deberta-base",
        )
        assert len(out) == 3
        assert out[0] > 0.9
        assert out[1] == 0.0
        assert out[2] == 0.0
