"""``prune_unlisted_models.py`` reconciles the HF cache against the manifest.

The CI model cache only grows. A model dropped from ``REQUIRED_ML_MODELS`` stops being
preloaded, but the copy already cached is restored every run and re-uploaded in the
ml-models artifact — and ``restore-keys`` re-seed it even after the cache key is salted.
ADR-154 shipped 5.5 GB of retired hybrid models to every test job that way.

So the invariant is: what the tier does not list is not in the artifact.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]

pytestmark = pytest.mark.unit


def _load_pruner():
    spec = importlib.util.spec_from_file_location(
        "prune_unlisted_models", _ROOT / "scripts" / "cache" / "prune_unlisted_models.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model_dir(hub: Path, model_id: str) -> Path:
    d = hub / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc123"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"weights")
    return d.parent.parent


def _hub_with(tmp_path, monkeypatch, listed: str, unlisted: str):
    hub = tmp_path / "hub"
    hub.mkdir()
    _model_dir(hub, listed)
    _model_dir(hub, unlisted)
    pruner = _load_pruner()
    monkeypatch.setattr("podcast_scraper.cache.directories.get_transformers_cache_dir", lambda: hub)
    return pruner, hub


def _a_listed_model() -> str:
    import sys

    sys.path.insert(0, str(_ROOT / "src"))
    from podcast_scraper.providers.ml import model_manifest as mm

    return next(s.model_id for s in mm.models_for_tier("production") if s.kind == "summary")


def test_prunes_what_the_manifest_does_not_list(tmp_path, monkeypatch):
    listed = _a_listed_model()
    pruner, hub = _hub_with(tmp_path, monkeypatch, listed, "google/flan-t5-base")

    assert pruner.main(["--tier", "production"]) == 0

    assert (hub / pruner.dir_name_for(listed)).is_dir(), "a listed model must survive"
    assert not (hub / "models--google--flan-t5-base").exists(), "an unlisted model must go"


def test_dry_run_deletes_nothing(tmp_path, monkeypatch):
    listed = _a_listed_model()
    pruner, hub = _hub_with(tmp_path, monkeypatch, listed, "google/flan-t5-base")

    assert pruner.main(["--tier", "production", "--dry-run"]) == 0

    assert (hub / "models--google--flan-t5-base").is_dir(), "--dry-run must not delete"
    assert (hub / pruner.dir_name_for(listed)).is_dir()


def test_leaves_non_model_entries_alone(tmp_path, monkeypatch):
    """The hub root also holds `.locks`, `version.txt` and similar. Only models-- dirs."""
    listed = _a_listed_model()
    pruner, hub = _hub_with(tmp_path, monkeypatch, listed, "google/flan-t5-base")
    (hub / ".locks").mkdir()
    (hub / "version.txt").write_text("1")

    assert pruner.main(["--tier", "production"]) == 0

    assert (hub / ".locks").is_dir()
    assert (hub / "version.txt").is_file()
