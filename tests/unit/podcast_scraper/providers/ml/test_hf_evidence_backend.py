"""Unit tests for :mod:`podcast_scraper.providers.ml.hf_evidence_backend`.

Focus on the shared machinery — device resolution, standard load kwargs,
subclass cache isolation, get_or_load lock semantics. The concrete
subclasses (QA / NLI / embedding) are covered by their module tests +
the integration parity gate.
"""

import threading
from typing import ClassVar
from unittest import mock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.module_ml_providers]

from podcast_scraper.providers.ml.hf_evidence_backend import (
    HFEvidenceBackend,
    resolve_evidence_device,
    standard_hf_load_kwargs,
)


class DummyBackend(HFEvidenceBackend):
    """Minimal subclass for testing the base class in isolation.

    Uses a class-level ``load_count`` counter to prove ``_load()`` is
    called exactly once per cached instance.
    """

    kind = "dummy"
    mps_supported = True
    load_count: ClassVar[int] = 0

    _instances: ClassVar[dict] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def _load(self) -> None:
        type(self).load_count += 1
        self.model = mock.Mock(name=f"dummy_model_{self.resolved_id}")


class UnsupportedMPSBackend(HFEvidenceBackend):
    """Subclass with mps_supported=False, verifies MPS→CPU coercion."""

    kind = "no_mps"
    mps_supported = False

    _instances: ClassVar[dict] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def _load(self) -> None:
        self.model = mock.Mock()


@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset every subclass cache before each test — no cross-test leakage."""
    DummyBackend.clear_cache()
    UnsupportedMPSBackend.clear_cache()
    DummyBackend.load_count = 0
    yield
    DummyBackend.clear_cache()
    UnsupportedMPSBackend.clear_cache()


# ---- resolve_evidence_device --------------------------------------------


class TestResolveEvidenceDevice:
    def test_explicit_device_kept(self):
        assert resolve_evidence_device("cuda", mps_supported=True) == "cuda"
        assert resolve_evidence_device("cpu", mps_supported=True) == "cpu"

    def test_explicit_mps_coerced_when_unsupported(self):
        assert resolve_evidence_device("mps", mps_supported=False) == "cpu"
        assert resolve_evidence_device("MPS", mps_supported=False) == "cpu"

    def test_explicit_mps_kept_when_supported(self):
        assert resolve_evidence_device("mps", mps_supported=True) == "mps"

    def test_none_falls_back_to_cpu_when_no_torch(self, monkeypatch):
        # Force ImportError on torch to guarantee CPU fallback path.
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        assert resolve_evidence_device(None, mps_supported=True) == "cpu"

    def test_empty_string_treated_as_auto(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        assert resolve_evidence_device("  ", mps_supported=True) == "cpu"


# ---- standard_hf_load_kwargs --------------------------------------------


class TestStandardHFLoadKwargs:
    def test_shape_contract(self):
        kw = standard_hf_load_kwargs()
        assert set(kw.keys()) == {
            "cache_dir",
            "local_files_only",
            "trust_remote_code",
            "low_cpu_mem_usage",
        }
        assert kw["local_files_only"] is True
        assert kw["trust_remote_code"] is False
        assert kw["low_cpu_mem_usage"] is False
        assert isinstance(kw["cache_dir"], str)


# ---- cache + get_or_load ------------------------------------------------


class TestGetOrLoadCache:
    def test_load_happens_once_per_key(self):
        b1 = DummyBackend.get_or_load("minilm-l6", device="cpu")
        b2 = DummyBackend.get_or_load("minilm-l6", device="cpu")
        assert b1 is b2
        assert DummyBackend.load_count == 1

    def test_different_devices_are_different_keys(self):
        DummyBackend.get_or_load("minilm-l6", device="cpu")
        DummyBackend.get_or_load("minilm-l6", device="cuda")
        assert DummyBackend.load_count == 2

    def test_different_extras_are_different_keys(self):
        DummyBackend.get_or_load("minilm-l6", device="cpu", flavor="a")
        DummyBackend.get_or_load("minilm-l6", device="cpu", flavor="b")
        assert DummyBackend.load_count == 2

    def test_clear_cache_forces_reload(self):
        DummyBackend.get_or_load("minilm-l6", device="cpu")
        assert DummyBackend.load_count == 1
        DummyBackend.clear_cache()
        DummyBackend.get_or_load("minilm-l6", device="cpu")
        assert DummyBackend.load_count == 2

    def test_subclass_caches_are_independent(self):
        DummyBackend.get_or_load("minilm-l6", device="cpu")
        UnsupportedMPSBackend.get_or_load("minilm-l6", device="cpu")
        # Loading one does NOT populate the other's cache.
        DummyBackend.clear_cache()
        # UnsupportedMPSBackend cache still populated
        assert len(UnsupportedMPSBackend._instances) == 1
        assert len(DummyBackend._instances) == 0

    def test_mps_coerced_in_cache_key(self):
        """Two get_or_load calls with device='mps' and device='cpu' on a
        subclass with mps_supported=False must share the SAME cache entry
        (both resolve to 'cpu')."""
        b1 = UnsupportedMPSBackend.get_or_load("minilm-l6", device="mps")
        b2 = UnsupportedMPSBackend.get_or_load("minilm-l6", device="cpu")
        assert b1 is b2
        assert b1.device == "cpu"


# ---- Threading semantics ------------------------------------------------


class TestGetOrLoadThreadSafety:
    def test_concurrent_get_or_load_only_loads_once(self):
        """Under concurrent get_or_load calls with the same key, ``_load()``
        must run exactly once. The lock guards the whole load, so callers
        never see a half-constructed instance."""

        barrier = threading.Barrier(4)
        results: list = []

        def worker() -> None:
            barrier.wait()  # All 4 threads race the lock at the same instant.
            results.append(DummyBackend.get_or_load("minilm-l6", device="cpu"))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        # Exactly one load and every worker sees the same instance.
        assert DummyBackend.load_count == 1
        first = results[0]
        for other in results[1:]:
            assert other is first


# ---- Abstract contract --------------------------------------------------


class TestAbstractContract:
    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            HFEvidenceBackend("minilm-l6")  # type: ignore[abstract]

    def test_subclass_must_implement_load(self):
        class Broken(HFEvidenceBackend):
            kind = "broken"
            _instances: ClassVar[dict] = {}
            _instances_lock: ClassVar[threading.Lock] = threading.Lock()

        with pytest.raises(TypeError):
            Broken("minilm-l6")  # type: ignore[abstract]
