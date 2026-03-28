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
        """entailment_scores_batch returns list of floats."""
        monkeypatch.setattr(
            nli_loader,
            "get_nli_model",
            lambda *args, **kwargs: type(
                "Model",
                (),
                {"predict": lambda self, pairs: [0.7, 0.9]},
            )(),
        )
        scores = nli_loader.entailment_scores_batch(
            pairs=[("p1", "h1"), ("p2", "h2")],
            model_id="nli-deberta-base",
        )
        assert len(scores) == 2
        assert scores[0] == 0.7
        assert scores[1] == 0.9

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

        def fake_cross_encoder(model_id, device=None):
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
