"""Unit tests for :mod:`podcast_scraper.providers.ml.hf_seq2seq_backend`.

Covers the shared loader + generator plumbing without loading a real
model — mocks are placed at the ``transformers.*`` and snapshot-lookup
seams, not at library internals. Concrete integration is covered by
the hybrid_provider tests + the Phase 7 parity gate.
"""

from unittest import mock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.module_ml_providers]

from podcast_scraper.providers.ml.hf_seq2seq_backend import (
    _default_use_safetensors,
    _detect_default_device,
    HFSeq2SeqBackend,
)


class TestDefaultUseSafetensors:
    """Model families with pytorch-only cache history should default to False."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "google/pegasus-cnn_dailymail",
            "allenai/led-base-16384",
            "allenai/led-large-16384",
            "google/flan-t5-base",
            "google/long-t5-tglobal-base",
            "google/long-t5-tglobal-large",
        ],
    )
    def test_no_safetensors_families(self, model_id):
        assert _default_use_safetensors(model_id) is False

    @pytest.mark.parametrize(
        "model_id",
        ["facebook/bart-base", "facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6"],
    )
    def test_safetensors_default_true(self, model_id):
        assert _default_use_safetensors(model_id) is True


class TestDetectDefaultDevice:
    def test_falls_back_to_cpu_without_torch(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        assert _detect_default_device() == "cpu"


class TestInitDefaults:
    def test_use_safetensors_auto_flanmapped(self):
        b = HFSeq2SeqBackend("google/flan-t5-base", device="cpu")
        assert b.use_safetensors is False

    def test_use_safetensors_auto_bart(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        assert b.use_safetensors is True

    def test_explicit_use_safetensors_wins(self):
        b = HFSeq2SeqBackend("google/flan-t5-base", device="cpu", use_safetensors=True)
        assert b.use_safetensors is True

    def test_low_cpu_mem_usage_default_false(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        # #539: default must stay False to avoid the meta-tensor breakage.
        assert b.low_cpu_mem_usage is False

    def test_generate_before_load_raises(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            b.generate("hello", gen_config=mock.Mock())

    def test_to_before_load_raises(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            b.to("cpu")

    def test_unload_is_idempotent(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        b.unload()  # never loaded — no-op
        assert b.model is None
        assert b.tokenizer is None


class TestLoadFallbacksToRepoIdWhenNoSnapshot:
    """When ``get_transformers_snapshot_path`` returns None, load should call
    ``from_pretrained(model_id, ...)`` directly with the standard kwargs."""

    def test_repo_id_path_taken(self, monkeypatch):
        fake_tokenizer = mock.Mock(name="tokenizer")
        fake_model = mock.Mock(name="model")
        fake_model.to = mock.Mock(return_value=fake_model)
        fake_model.eval = mock.Mock()

        fake_auto_tok = mock.Mock()
        fake_auto_tok.from_pretrained = mock.Mock(return_value=fake_tokenizer)
        fake_auto_model = mock.Mock()
        fake_auto_model.from_pretrained = mock.Mock(return_value=fake_model)

        fake_transformers = mock.Mock(
            AutoTokenizer=fake_auto_tok, AutoModelForSeq2SeqLM=fake_auto_model
        )
        import sys

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        # Stub the cache-dir + snapshot-lookup helpers so no real filesystem access.
        from podcast_scraper import cache as cache_pkg

        monkeypatch.setattr(
            cache_pkg,
            "get_transformers_cache_dir",
            lambda: mock.Mock(spec=[], __str__=lambda _: "/tmp/hf-cache"),
        )
        monkeypatch.setattr(cache_pkg, "get_transformers_snapshot_path", lambda *a, **kw: None)

        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu", cache_dir="/tmp/hf-cache")
        b.load()

        assert b._loaded is True
        assert b.model is fake_model
        assert b.tokenizer is fake_tokenizer
        # from_pretrained called with model_id + standard kwargs
        _, kwargs = fake_auto_model.from_pretrained.call_args
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["low_cpu_mem_usage"] is False


class TestGenerateWiresGenerationConfig:
    """Verify generate() feeds ``model.generate(**inputs, generation_config=gen_cfg)``
    and returns the decoded string stripped."""

    def test_generate_call_shape(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        # Simulate a loaded backend with mocked model + tokenizer.
        fake_tokenizer = mock.MagicMock()
        fake_tokenizer.return_value = {"input_ids": mock.Mock(), "attention_mask": mock.Mock()}
        for v in fake_tokenizer.return_value.values():
            v.to = mock.Mock(return_value=v)
        fake_tokenizer.decode = mock.Mock(return_value="  a summary  ")
        fake_model = mock.MagicMock()
        fake_model.config.max_position_embeddings = 1024
        fake_model.parameters = mock.Mock(return_value=iter([mock.Mock(device="cpu")]))
        fake_model.generate = mock.Mock(return_value=[[1, 2, 3]])
        b.model = fake_model
        b.tokenizer = fake_tokenizer
        b._loaded = True

        gen_cfg = mock.Mock()
        result = b.generate("input text", gen_cfg)

        assert result == "a summary"
        fake_model.generate.assert_called_once()
        _, kwargs = fake_model.generate.call_args
        assert kwargs["generation_config"] is gen_cfg


class TestSnapshotLoadWithFamilyClass:
    """Cover the snapshot-load path where family_class is set (LED/BART/Pegasus).

    This is the load branch used by SummaryModel for LED and BART checkpoints.
    Uncovered by TestLoadFallbacksToRepoIdWhenNoSnapshot because that test
    forces get_transformers_snapshot_path to return None.
    """

    def test_snapshot_path_taken_with_family_class(self, monkeypatch, tmp_path):
        """When a snapshot exists, load() calls family_class.from_pretrained
        on the resolved snapshot directory (not the repo id)."""
        fake_tokenizer = mock.Mock(name="tokenizer")
        fake_model = mock.Mock(name="model")
        fake_model.to = mock.Mock(return_value=fake_model)
        fake_model.eval = mock.Mock()

        fake_auto_tok = mock.Mock()
        fake_auto_tok.from_pretrained = mock.Mock(return_value=fake_tokenizer)
        fake_transformers = mock.Mock(AutoTokenizer=fake_auto_tok)
        import sys

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        family_class = mock.Mock()
        family_class.from_pretrained = mock.Mock(return_value=fake_model)

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        from podcast_scraper import cache as cache_pkg

        monkeypatch.setattr(cache_pkg, "get_transformers_cache_dir", lambda: tmp_path)
        monkeypatch.setattr(cache_pkg, "get_transformers_snapshot_path", lambda *a, **kw: snapshot)

        b = HFSeq2SeqBackend(
            "allenai/led-base-16384",
            device="cpu",
            cache_dir=str(tmp_path),
            family_class=family_class,
        )
        b.load()

        assert b._loaded is True
        assert b.model is fake_model
        assert b.tokenizer is fake_tokenizer
        family_class.from_pretrained.assert_called_once()
        call_args, _ = family_class.from_pretrained.call_args
        assert str(snapshot.resolve()) in call_args

    def test_snapshot_fallback_to_main_on_no_file_named(self, monkeypatch, tmp_path):
        """If the pinned snapshot lacks weights, load() retries against the
        main snapshot. Covers the OSError('no file named ...') branch."""
        fake_tokenizer = mock.Mock(name="tokenizer")
        fake_model = mock.Mock(name="model")
        fake_model.to = mock.Mock(return_value=fake_model)
        fake_model.eval = mock.Mock()

        fake_auto_tok = mock.Mock(from_pretrained=mock.Mock(return_value=fake_tokenizer))
        fake_transformers = mock.Mock(AutoTokenizer=fake_auto_tok)
        import sys

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        pinned = tmp_path / "pinned"
        pinned.mkdir()
        main_snap = tmp_path / "main"
        main_snap.mkdir()

        from podcast_scraper import cache as cache_pkg

        monkeypatch.setattr(cache_pkg, "get_transformers_cache_dir", lambda: tmp_path)

        def snapshot_lookup(model_id, revision=None, cache_dir=None):
            return pinned if revision else main_snap

        monkeypatch.setattr(cache_pkg, "get_transformers_snapshot_path", snapshot_lookup)

        family_class = mock.Mock()
        family_class.from_pretrained = mock.Mock(
            side_effect=[OSError("no file named model.safetensors"), fake_model]
        )

        b = HFSeq2SeqBackend(
            "google/flan-t5-base",
            device="cpu",
            revision="abc123",
            cache_dir=str(tmp_path),
            family_class=family_class,
        )
        b.load()

        assert b._loaded is True
        assert b.model is fake_model
        assert family_class.from_pretrained.call_count == 2


class TestToDeviceFallback:
    """Cover the OOM device-fallback branches on backend.to() after load."""

    def _make_loaded_backend(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="mps")
        b.model = mock.MagicMock()
        b.tokenizer = mock.MagicMock()
        b._loaded = True
        return b

    def test_to_oom_on_mps_falls_back_to_cpu(self):
        """MPS OOM → automatic CPU fallback (initial=False, from OOM recovery)."""
        b = self._make_loaded_backend()
        b.model.to = mock.Mock(
            side_effect=[
                RuntimeError("MPS out of memory"),
                b.model,
            ]
        )
        b.to("mps", initial=False)
        assert b.device == "cpu"
        assert b.model.to.call_count == 2

    def test_to_non_oom_error_reraises(self):
        """Non-OOM RuntimeError on post-load device move re-raises."""
        b = self._make_loaded_backend()
        b.model.to = mock.Mock(side_effect=RuntimeError("some other error"))
        with pytest.raises(RuntimeError, match="some other error"):
            b.to("mps", initial=False)

    def test_to_meta_tensor_error_reties_ties_weights(self):
        """meta-tensor NotImplementedError → tie_weights() + retry."""
        b = self._make_loaded_backend()
        b.model.to = mock.Mock(
            side_effect=[
                NotImplementedError("Cannot copy out of meta tensor"),
                b.model,
            ]
        )
        b.model.tie_weights = mock.Mock()
        b.to("cpu", initial=True)
        b.model.tie_weights.assert_called_once()
        assert b.model.to.call_count == 2

    def test_to_initial_load_falls_back_broadly(self):
        """During initial load, any RuntimeError → CPU fallback (broader than post-load)."""
        b = self._make_loaded_backend()
        b.model.to = mock.Mock(
            side_effect=[
                RuntimeError("something weird"),
                b.model,
            ]
        )
        b.to("mps", initial=True)
        assert b.device == "cpu"


class TestAdoptAndUnload:
    """Cover the adopt() + unload() surface added for SummaryModel Pegasus routing."""

    def test_adopt_marks_loaded_and_calls_eval(self):
        b = HFSeq2SeqBackend("google/pegasus-cnn_dailymail", device="cpu")
        fake_model = mock.MagicMock()
        fake_tokenizer = mock.MagicMock()
        b.adopt(fake_model, fake_tokenizer, device="cpu")
        assert b._loaded is True
        assert b.model is fake_model
        assert b.tokenizer is fake_tokenizer
        fake_model.eval.assert_called_once()

    def test_adopt_without_device_keeps_existing(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cuda")
        b.adopt(mock.MagicMock(), mock.MagicMock())
        assert b.device == "cuda"

    def test_unload_after_adopt_clears_state(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        b.adopt(mock.MagicMock(), mock.MagicMock())
        b.unload()
        assert b.model is None
        assert b.tokenizer is None
        assert b._loaded is False
