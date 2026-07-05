"""Unit tests for `TransformersReduceBackend` (hybrid tier-1 REDUCE).

Post-#382 the backend delegates all loading and generation to
`HFSeq2SeqBackend`; these tests exercise the thin `TransformersReduceBackend`
wrapper — initialization, generation-config assembly, prompt shaping, and
cleanup — with the underlying backend fully mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# reduce() imports `from transformers import GenerationConfig`. Skip when the
# optional [ml] extras are not installed (no-ML dev venv used by test-unit in CI).
pytest.importorskip("transformers")

from podcast_scraper.providers.ml.hybrid_ml_provider import (
    HybridReduceResult,
    TransformersReduceBackend,
)


class TestTransformersReduceBackendInitialize:
    def test_initialize_constructs_backend_with_pinned_revision(self):
        b = TransformersReduceBackend(
            model_name="google/flan-t5-base",
            device="cpu",
            cache_dir=None,
        )
        with (
            patch("podcast_scraper.providers.ml.hybrid_ml_provider.HFSeq2SeqBackend") as mock_cls,
            patch("podcast_scraper.config_constants.get_pinned_revision_for_model") as mock_rev,
        ):
            mock_rev.return_value = "deadbeef1234"
            fake_backend = MagicMock()
            fake_backend._loaded = True
            fake_backend.device = "cpu"
            mock_cls.return_value = fake_backend

            b.initialize()

            mock_cls.assert_called_once_with(
                model_id="google/flan-t5-base",
                device="cpu",
                revision="deadbeef1234",
                cache_dir=None,
            )
            fake_backend.load.assert_called_once()
            assert b.device == "cpu"

    def test_initialize_mirrors_device_coercion_from_backend(self):
        """MPS→CPU fallback in backend must propagate to the wrapper."""
        b = TransformersReduceBackend(
            model_name="google/flan-t5-base",
            device="mps",
            cache_dir=None,
        )
        with (
            patch("podcast_scraper.providers.ml.hybrid_ml_provider.HFSeq2SeqBackend") as mock_cls,
            patch(
                "podcast_scraper.config_constants.get_pinned_revision_for_model",
                return_value=None,
            ),
        ):
            fake_backend = MagicMock()
            fake_backend._loaded = True
            fake_backend.device = "cpu"  # coerced from mps
            mock_cls.return_value = fake_backend

            b.initialize()
            assert b.device == "cpu"

    def test_initialize_skips_if_already_loaded(self):
        """Second `initialize()` call is a no-op when backend is loaded."""
        b = TransformersReduceBackend(
            model_name="google/flan-t5-base",
            device="cpu",
            cache_dir=None,
        )
        fake_backend = MagicMock()
        fake_backend._loaded = True
        b._backend = fake_backend

        with patch("podcast_scraper.providers.ml.hybrid_ml_provider.HFSeq2SeqBackend") as mock_cls:
            b.initialize()
            mock_cls.assert_not_called()
            fake_backend.load.assert_not_called()


class TestTransformersReduceBackendReduce:
    def _make_ready_backend(self, generated_text: str = "SUMMARY"):
        b = TransformersReduceBackend(
            model_name="google/flan-t5-base",
            device="cpu",
            cache_dir=None,
        )
        fake_backend = MagicMock()
        fake_backend._loaded = True
        fake_backend.generate.return_value = generated_text
        b._backend = fake_backend
        return b, fake_backend

    def test_reduce_returns_hybrid_reduce_result_with_metadata(self):
        b, fake = self._make_ready_backend("The market closed higher.")
        result = b.reduce(
            notes="Notes about the day.",
            instruction="Rewrite in one sentence.",
        )
        assert isinstance(result, HybridReduceResult)
        assert result.text == "The market closed higher."
        assert result.backend == "transformers"
        assert result.model == "google/flan-t5-base"

    def test_reduce_builds_generation_config_from_params(self):
        b, fake = self._make_ready_backend()
        b.reduce(
            notes="notes",
            instruction="instr",
            params={"max_new_tokens": 128, "num_beams": 2, "do_sample": True},
        )
        _, kwargs = fake.generate.call_args
        # generate() is called as generate(prompt, gen_cfg); grab positional args
        args, _ = fake.generate.call_args
        gen_cfg = args[1]
        assert gen_cfg.max_new_tokens == 128
        assert gen_cfg.num_beams == 2
        assert gen_cfg.do_sample is True

    def test_reduce_applies_defaults_when_params_missing(self):
        b, fake = self._make_ready_backend()
        b.reduce(notes="notes", instruction="instr")  # no params
        args, _ = fake.generate.call_args
        gen_cfg = args[1]
        assert gen_cfg.max_new_tokens == 600
        assert gen_cfg.num_beams == 4
        assert gen_cfg.do_sample is False

    def test_reduce_prompt_wraps_instruction_and_notes(self):
        b, fake = self._make_ready_backend()
        b.reduce(notes="  the notes  ", instruction="  the instruction  ")
        args, _ = fake.generate.call_args
        prompt = args[0]
        assert prompt == "the instruction\n\nNOTES:\nthe notes"

    def test_reduce_raises_when_not_initialized(self):
        b = TransformersReduceBackend(
            model_name="google/flan-t5-base",
            device="cpu",
            cache_dir=None,
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            b.reduce(notes="n", instruction="i")


class TestTransformersReduceBackendCleanup:
    def test_cleanup_unloads_and_drops_backend(self):
        b = TransformersReduceBackend(
            model_name="google/flan-t5-base",
            device="cpu",
            cache_dir=None,
        )
        fake_backend = MagicMock()
        b._backend = fake_backend

        b.cleanup()
        fake_backend.unload.assert_called_once()
        assert b._backend is None

    def test_cleanup_is_safe_when_never_initialized(self):
        b = TransformersReduceBackend(
            model_name="google/flan-t5-base",
            device="cpu",
            cache_dir=None,
        )
        # No initialize() call — _backend is None.
        b.cleanup()  # should not raise
        assert b._backend is None
