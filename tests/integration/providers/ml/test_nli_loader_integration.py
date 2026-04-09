"""Integration tests for NLI cross-encoder loader (Issue #435).

Exercises the full load → entailment_score flow with a fake CrossEncoder backend.
Requires ``sentence-transformers`` (``pip install -e '.[ml]'``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from podcast_scraper.providers.ml import nli_loader
from podcast_scraper.providers.ml.model_registry import ModelRegistry

pytestmark = [pytest.mark.integration]


def _make_fake_cross_encoder(captured):
    """Return a fake CrossEncoder class that records constructor kwargs."""

    class FakeCrossEncoder:
        def __init__(self, model_id, device=None, local_files_only=True, cache_folder=None, **kw):
            captured.append(
                {
                    "model_id": model_id,
                    "device": device,
                    "local_files_only": local_files_only,
                    "cache_folder": cache_folder,
                    **kw,
                }
            )
            self.model = type(
                "FakeModel",
                (),
                {
                    "config": type(
                        "Cfg", (), {"id2label": {0: "contradiction", 1: "neutral", 2: "entailment"}}
                    )()
                },
            )()

        def predict(self, pairs):
            return [[0.1, 0.2, 0.7]] * len(pairs)

    return FakeCrossEncoder


class TestNliLoaderIntegration:
    """Full wiring: alias → registry → CrossEncoder init → entailment scoring."""

    def test_entailment_score_returns_probability(self, monkeypatch, tmp_path):
        """entailment_score() resolves alias, loads model, returns float in [0, 1]."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            _make_fake_cross_encoder(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        nli_loader._nli_models.clear()

        score = nli_loader.entailment_score(
            premise="The cat sat on the mat.",
            hypothesis="A cat was on a mat.",
            model_id="nli-deberta-base",
            device="cpu",
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert captured[0]["model_id"] == ModelRegistry.resolve_evidence_model_id(
            "nli-deberta-base"
        )
        assert captured[0]["device"] == "cpu"
        assert captured[0]["local_files_only"] is True

    def test_entailment_scores_batch(self, monkeypatch, tmp_path):
        """entailment_scores_batch() returns one score per pair."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            _make_fake_cross_encoder(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        nli_loader._nli_models.clear()

        pairs = [
            ("Evidence A.", "Claim A."),
            ("Evidence B.", "Claim B."),
            ("Evidence C.", "Claim C."),
        ]
        scores = nli_loader.entailment_scores_batch(
            pairs,
            model_id="nli-deberta-base",
            device="cpu",
        )

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_get_nli_model_caches_instance(self, monkeypatch, tmp_path):
        """get_nli_model returns the same instance on repeated calls."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            _make_fake_cross_encoder(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        nli_loader._nli_models.clear()

        a = nli_loader.get_nli_model("nli-deberta-base", device="cpu")
        b = nli_loader.get_nli_model("nli-deberta-base", device="cpu")

        assert a is b
        assert len(captured) == 1

    def test_load_nli_model_resolves_alias(self, monkeypatch, tmp_path):
        """load_nli_model resolves alias via registry before loading."""
        captured: list = []
        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            _make_fake_cross_encoder(captured),
        )
        monkeypatch.setattr(
            "podcast_scraper.cache.get_transformers_cache_dir",
            lambda: tmp_path,
        )

        nli_loader.load_nli_model("nli-deberta-base", device="cpu")

        assert len(captured) == 1
        assert captured[0]["model_id"] == ModelRegistry.resolve_evidence_model_id(
            "nli-deberta-base"
        )
