"""The GI embedding device resolver must never return mps (it flaky-SIGSEGVs on macOS)."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def _fresh(monkeypatch, **env):
    monkeypatch.delenv("PODCAST_EMBED_DEVICE", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import podcast_scraper.providers.ml.embedding_device as m

    return importlib.reload(m)


def test_never_mps_defaults_to_cpu_or_cuda(monkeypatch):
    m = _fresh(monkeypatch)
    dev = m.resolve_embedding_device()
    assert dev in ("cpu", "cuda")
    assert dev != "mps"


def test_env_override_wins(monkeypatch):
    m = _fresh(monkeypatch, PODCAST_EMBED_DEVICE="cuda:1")
    assert m.resolve_embedding_device() == "cuda:1"


def test_cuda_preferred_when_available(monkeypatch):
    monkeypatch.delenv("PODCAST_EMBED_DEVICE", raising=False)

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def is_available():
                return True

    # resolve_embedding_device imports torch lazily, so patching sys.modules is enough.
    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    from podcast_scraper.providers.ml.embedding_device import resolve_embedding_device

    assert resolve_embedding_device() == "cuda"
