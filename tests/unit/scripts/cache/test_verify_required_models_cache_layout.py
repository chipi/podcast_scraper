"""Verify ``verify_required_models.py`` probes the REAL on-disk cache layout (#917).

This is the direct regression guard for the #897 MiniLM saga: the verifier gates CI
by probing the local caches for each manifest model. If its probe path logic ever
drifts from the layout ``preload_ml_models`` / the HF hub actually writes (e.g. a
wrong ``models--{org}--{name}`` join, or the whisper ``{id}.pt`` location), it would
either pass with a missing model (the saga) or fail with everything present. We build
a cache in the real HF-hub + whisper layout and assert the verifier passes, then
remove one model and assert it fails -- pinning the probe to the real layout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_required_models", _ROOT / "scripts" / "cache" / "verify_required_models.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.unit


def _write_hf_model(hub: Path, model_id: str) -> Path:
    # Real HF hub layout: models--{org}--{name}/snapshots/<rev>/<file> + blobs.
    model_dir = hub / f"models--{model_id.replace('/', '--')}"
    snap = model_dir / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"weights")
    return model_dir


def _build_real_cache(home: Path, hub: Path) -> None:
    import sys

    sys.path.insert(0, str(_ROOT / "src"))
    from podcast_scraper.providers.ml import model_manifest as mm

    whisper_cache = home / ".cache" / "whisper"
    whisper_cache.mkdir(parents=True)
    for spec in mm.models_for_tier("ci_artifact"):
        if spec.kind == "whisper":
            (whisper_cache / f"{spec.model_id}.pt").write_bytes(b"whisper-weights")
        elif spec.kind == "spacy":
            continue  # pip-delivered; verifier skips it
        else:
            _write_hf_model(hub, spec.model_id)


def test_verifier_passes_on_real_cache_layout(tmp_path, monkeypatch):
    home = tmp_path / "home"
    hub = tmp_path / "hub"
    home.mkdir()
    hub.mkdir()
    _build_real_cache(home, hub)

    verifier = _load_verifier()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("podcast_scraper.cache.directories.get_transformers_cache_dir", lambda: hub)

    assert verifier.main(["--tier", "ci_artifact"]) == 0


def test_verifier_fails_when_a_model_is_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    hub = tmp_path / "hub"
    home.mkdir()
    hub.mkdir()
    _build_real_cache(home, hub)

    # Remove MiniLM -- the exact model that went missing in #897.
    import shutil

    shutil.rmtree(hub / "models--sentence-transformers--all-MiniLM-L6-v2")

    verifier = _load_verifier()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("podcast_scraper.cache.directories.get_transformers_cache_dir", lambda: hub)

    assert verifier.main(["--tier", "ci_artifact"]) == 1
