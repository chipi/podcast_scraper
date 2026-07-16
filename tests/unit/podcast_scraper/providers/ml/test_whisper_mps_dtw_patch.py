"""Tests for the Whisper DTW MPS float64 monkey-patch (transformers-v5 followup).

Backstory: ``openai-whisper``'s ``whisper.timing.dtw`` calls
``x.double().cpu().numpy()`` on the alignment tensor. On MPS, ``.double()`` is
rejected ("Cannot convert a MPS Tensor to float64 dtype…"). Swapping the order
to ``.cpu().double()`` is behaviour-neutral for CPU + CUDA and fixes MPS.

The patch lives in
``podcast_scraper.providers.ml.ml_provider._patch_whisper_dtw_for_mps`` and
is applied whenever ``_import_third_party_whisper`` runs. These tests are
device-agnostic — they exercise the patch's substitution logic without
needing an MPS device.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper.providers.ml import ml_provider

pytestmark = pytest.mark.unit


def _reset_patch_flag() -> None:
    """Force the patch to re-apply on next call."""
    ml_provider._whisper_dtw_patched = False


def test_patch_replaces_dtw_with_wrapper() -> None:
    """After ``_patch_whisper_dtw_for_mps``, ``whisper.timing.dtw`` is our
    ``_dtw_mps_safe`` wrapper, not the upstream function.
    """
    _reset_patch_flag()

    import whisper.timing as timing_mod

    original = timing_mod.dtw
    try:
        ml_provider._patch_whisper_dtw_for_mps(__import__("whisper"))
        assert timing_mod.dtw is not original
        assert timing_mod.dtw.__name__ == "_dtw_mps_safe"
    finally:
        timing_mod.dtw = original
        _reset_patch_flag()


def test_patch_is_idempotent() -> None:
    """Calling the patch twice does not re-wrap or otherwise misbehave."""
    _reset_patch_flag()

    import whisper.timing as timing_mod

    original = timing_mod.dtw
    try:
        ml_provider._patch_whisper_dtw_for_mps(__import__("whisper"))
        after_first = timing_mod.dtw
        ml_provider._patch_whisper_dtw_for_mps(__import__("whisper"))
        assert timing_mod.dtw is after_first
    finally:
        timing_mod.dtw = original
        _reset_patch_flag()


def test_patched_dtw_uses_cpu_then_double_order() -> None:
    """The wrapper must call ``.cpu().double()`` in that order on the
    non-CUDA path — the reordering IS the fix.
    """
    _reset_patch_flag()

    import whisper.timing as timing_mod

    original_dtw = timing_mod.dtw
    original_dtw_cpu = timing_mod.dtw_cpu
    try:
        # Fake tensor that records the call order (.cpu() then .double() then .numpy()).
        call_log: list[str] = []

        class FakeTensor:
            is_cuda = False

            def cpu(self):
                call_log.append("cpu")
                return self

            def double(self):
                call_log.append("double")
                return self

            def numpy(self):
                call_log.append("numpy")
                return "fake-numpy-array"

        called_with: list[str] = []

        def fake_dtw_cpu(arr):
            called_with.append(arr)
            return "dtw-result"

        timing_mod.dtw_cpu = fake_dtw_cpu
        ml_provider._patch_whisper_dtw_for_mps(__import__("whisper"))

        result = timing_mod.dtw(FakeTensor())
        assert result == "dtw-result"
        assert call_log == [
            "cpu",
            "double",
            "numpy",
        ], f"expected cpu→double→numpy order, got {call_log}"
        assert called_with == ["fake-numpy-array"]
    finally:
        timing_mod.dtw = original_dtw
        timing_mod.dtw_cpu = original_dtw_cpu
        _reset_patch_flag()


def test_patched_dtw_delegates_to_original_for_cuda() -> None:
    """When the tensor lives on CUDA, the wrapper delegates to the original
    ``dtw`` (which routes to Triton). We must not force it through the CPU
    path, or we'd lose the CUDA fast path.
    """
    _reset_patch_flag()

    import whisper.timing as timing_mod

    original_dtw = timing_mod.dtw
    original_dtw_cpu = timing_mod.dtw_cpu
    try:
        cuda_calls: list[str] = []

        def fake_original_dtw(x):
            cuda_calls.append("original")
            return "cuda-result"

        cpu_calls: list[str] = []

        def fake_dtw_cpu(arr):
            cpu_calls.append("cpu")
            return "cpu-result"

        # Install the fake original BEFORE patching so the patch captures it.
        timing_mod.dtw = fake_original_dtw
        timing_mod.dtw_cpu = fake_dtw_cpu
        ml_provider._patch_whisper_dtw_for_mps(__import__("whisper"))

        class FakeCudaTensor:
            is_cuda = True

        result = timing_mod.dtw(FakeCudaTensor())
        assert result == "cuda-result"
        assert cuda_calls == ["original"]
        assert cpu_calls == [], "CUDA path should not touch dtw_cpu"
    finally:
        timing_mod.dtw = original_dtw
        timing_mod.dtw_cpu = original_dtw_cpu
        _reset_patch_flag()


def test_patch_returns_early_when_timing_module_missing() -> None:
    """If ``whisper.timing`` cannot be imported, the patch degrades silently.

    Simulates a future openai-whisper reshuffle where the submodule moves.
    """
    _reset_patch_flag()

    import importlib

    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "whisper.timing":
            raise ImportError("simulated missing module")
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(importlib, "import_module", fake_import)
        # Should not raise.
        ml_provider._patch_whisper_dtw_for_mps(MagicMock())
    _reset_patch_flag()


def test_import_helper_patches_even_when_whisper_already_in_sys_modules() -> None:
    """Regression guard (ci-fast, 2026-07-16): when ``whisper`` is already in
    ``sys.modules`` (a common e2e-test setup — fixtures import whisper directly
    before the provider factory runs), ``_import_third_party_whisper`` used to
    return early without invoking the DTW patch. That let the MPS float64
    crash resurface on the transcription path.

    Assert the patch fires on the ``existing in sys.modules`` branch too.
    """
    _reset_patch_flag()

    import sys

    import whisper.timing as timing_mod

    original = timing_mod.dtw
    try:
        # Simulate the "already imported" state — whisper is guaranteed to be
        # in sys.modules right now anyway because we imported timing above.
        assert "whisper" in sys.modules

        # Now trigger the provider-side import helper.
        ml_provider._import_third_party_whisper()

        # The patch must have fired via the early-return branch.
        assert timing_mod.dtw is not original, (
            "_import_third_party_whisper returned early without applying the "
            "MPS DTW patch (see #1180 followup)"
        )
        assert timing_mod.dtw.__name__ == "_dtw_mps_safe"
    finally:
        timing_mod.dtw = original
        _reset_patch_flag()


def test_detect_whisper_device_returns_mps_when_auto_detected() -> None:
    """After the patch, MPS is a valid Whisper device again — ``_detect_whisper_device``
    should return ``mps`` on Apple Silicon under auto-detect (no explicit config).
    """
    from unittest.mock import MagicMock

    # Provider instance with an empty/no-explicit-device config.
    cfg = MagicMock()
    cfg.transcription_device = None
    cfg.whisper_device = None

    prov = MagicMock()
    prov.cfg = cfg
    # Bind the real method — we're testing the routing logic, not stubbing it.
    prov._detect_whisper_device = ml_provider.MLProvider._detect_whisper_device.__get__(prov)

    device = prov._detect_whisper_device()
    # On this test host MPS may or may not be available; assert we return one of the
    # valid device strings and NOT the historical CPU-only fallback path when MPS is
    # actually available.
    assert device in {"mps", "cuda", "cpu"}

    import torch

    if not torch.cuda.is_available():
        mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if mps_available:
            assert device == "mps", "MPS should be auto-selected on Apple Silicon post-patch"
