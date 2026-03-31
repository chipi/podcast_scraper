"""Unit tests for NLI loader (Issue #435)."""

import pytest

from podcast_scraper.providers.ml import nli_loader

pytestmark = [pytest.mark.unit]


class TestEntailmentScoreMocked:
    """Tests for entailment_score() with mocked model."""

    def test_entailment_score_returns_float(self, monkeypatch):
        """entailment_score returns a float."""
        monkeypatch.setattr(
            nli_loader,
            "get_nli_model",
            lambda *args, **kwargs: type(
                "Model",
                (),
                {"predict": lambda self, pairs: [0.85]},
            )(),
        )
        score = nli_loader.entailment_score(
            premise="The cat sat on the mat.",
            hypothesis="A cat was on a mat.",
            model_id="nli-deberta-base",
        )
        assert isinstance(score, float)
        assert score == 0.85

    def test_entailment_scores_batch_returns_list(self, monkeypatch):
        """entailment_scores_batch returns list of floats (one scalar row per pair)."""
        monkeypatch.setattr(
            nli_loader,
            "get_nli_model",
            lambda *args, **kwargs: type(
                "Model",
                (),
                {"predict": lambda self, pairs: [[0.7], [0.9]]},
            )(),
        )
        scores = nli_loader.entailment_scores_batch(
            pairs=[("p1", "h1"), ("p2", "h2")],
            model_id="nli-deberta-base",
        )
        assert len(scores) == 2
        assert scores[0] == 0.7
        assert scores[1] == 0.9

    def test_entailment_score_three_class_logits_1d(self, monkeypatch):
        """NLI models return one row of 3 logits; map to softmax P(entailment)."""
        mock_model = type(
            "M",
            (),
            {
                "model": type(
                    "Inner",
                    (),
                    {"config": type("C", (), {"id2label": {0: "c", 1: "n", 2: "e"}})()},
                )(),
                "predict": lambda self, pairs: [[-2.0, -2.0, 4.0]],
            },
        )()
        monkeypatch.setattr(nli_loader, "get_nli_model", lambda *a, **k: mock_model)
        score = nli_loader.entailment_score("p", "h", "nli-deberta-base")
        assert score > 0.9

    def test_entailment_score_respects_id2label_entailment_index(self, monkeypatch):
        """Use config id2label to find entailment column (not always index 2)."""
        mock_model = type(
            "M",
            (),
            {
                "model": type(
                    "Inner",
                    (),
                    {
                        "config": type(
                            "Cfg",
                            (),
                            {
                                "id2label": {
                                    0: "contradiction",
                                    1: "entailment",
                                    2: "neutral",
                                }
                            },
                        )()
                    },
                )(),
                "predict": lambda self, pairs: [[0.0, 4.0, 0.0]],
            },
        )()
        monkeypatch.setattr(nli_loader, "get_nli_model", lambda *a, **k: mock_model)
        score = nli_loader.entailment_score("p", "h", "nli-deberta-base")
        assert score > 0.9

    def test_entailment_scores_batch_two_by_three_logits(self, monkeypatch):
        """Batch (n, 3) logits yields n entailment probabilities."""
        mock_model = type(
            "M",
            (),
            {
                "model": type(
                    "Inner",
                    (),
                    {"config": type("C", (), {"id2label": {0: "c", 1: "n", 2: "e"}})()},
                )(),
                "predict": lambda self, pairs: [
                    [-2.0, -2.0, 4.0],
                    [4.0, -2.0, -2.0],
                ],
            },
        )()
        monkeypatch.setattr(nli_loader, "get_nli_model", lambda *a, **k: mock_model)
        scores = nli_loader.entailment_scores_batch(
            [("a", "b"), ("c", "d")],
            "nli-deberta-base",
        )
        assert len(scores) == 2
        assert scores[0] > 0.9
        assert scores[1] < 0.1

    def test_entailment_score_meta_tensor_fallback(self, monkeypatch):
        """When model returns tensor on meta device, fallback to 0.0 (GIL + API-only)."""

        class MetaTensor:
            def item(self):
                raise RuntimeError("Tensor.item() cannot be called on meta tensors")

        monkeypatch.setattr(
            nli_loader,
            "get_nli_model",
            lambda *args, **kwargs: type(
                "Model",
                (),
                {"predict": lambda self, pairs: MetaTensor()},
            )(),
        )
        score = nli_loader.entailment_score(
            premise="Evidence.",
            hypothesis="Claim.",
            model_id="nli-deberta-base",
        )
        assert score == 0.0


try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers required for load path test",
)
class TestLoadNliModel:
    """Tests for load_nli_model (resolve + load wiring)."""

    def test_load_nli_model_resolves_alias(self, monkeypatch):
        """load_nli_model resolves alias via registry before loading."""
        from podcast_scraper.providers.ml.model_registry import ModelRegistry

        captured = []

        def fake_cross_encoder(model_id, device=None, **kwargs):
            captured.append(model_id)
            return type("Fake", (), {"predict": lambda self, pairs: [0.5]})()

        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            fake_cross_encoder,
            raising=False,
        )
        nli_loader.load_nli_model("nli-deberta-base", device="cpu")
        assert len(captured) == 1
        assert captured[0] == ModelRegistry.resolve_evidence_model_id("nli-deberta-base")


class TestPredictOutputHelpers:
    """Extra branches for ``predict_output_to_entailment_scores`` and batch padding."""

    def test_predict_output_single_logit_row(self) -> None:
        class _M:
            model = type(
                "Inner",
                (),
                {"config": type("Cfg", (), {"id2label": {0: "entailment"}})()},
            )()

        scores = nli_loader.predict_output_to_entailment_scores([[0.25]], _M())
        assert scores == [0.25]

    def test_entailment_scores_batch_meta_runtime_returns_zeros(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_meta(*_a: object, **_k: object) -> None:
            raise RuntimeError("meta tensor predict")

        monkeypatch.setattr(
            nli_loader,
            "get_nli_model",
            lambda *a, **k: type("M", (), {"predict": _raise_meta})(),
        )
        out = nli_loader.entailment_scores_batch([("p", "h"), ("p2", "h2")], "nli-deberta-base")
        assert out == [0.0, 0.0]

    def test_entailment_scores_batch_pads_short_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_model = type(
            "M",
            (),
            {
                "model": type(
                    "Inner",
                    (),
                    {"config": type("C", (), {"id2label": {0: "c", 1: "n", 2: "e"}})()},
                )(),
                "predict": lambda self, pairs: [[-1.0, -1.0, 3.0]],
            },
        )()
        monkeypatch.setattr(nli_loader, "get_nli_model", lambda *a, **k: mock_model)
        out = nli_loader.entailment_scores_batch(
            [("a", "b"), ("c", "d"), ("e", "f")],
            "nli-deberta-base",
        )
        assert len(out) == 3
        assert out[0] > 0.9
        assert out[1] == 0.0
        assert out[2] == 0.0
