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
    # Deliberately NO monkeypatch of get_transformers_cache_dir. Every test passes --hub
    # instead. Patching that function looks equivalent and is not: on the pre-install path
    # the script re-imports the module, gets a fresh unpatched copy, resolves the REAL
    # cache and deletes from it. That removed 4 GB from a developer machine before --hub
    # existed, so the redirect has to be an argument the script cannot import past.
    return pruner, hub


def _a_listed_model() -> str:
    import sys

    sys.path.insert(0, str(_ROOT / "src"))
    from podcast_scraper.providers.ml import model_manifest as mm

    return next(s.model_id for s in mm.models_for_tier("production") if s.kind == "summary")


def test_prunes_what_the_manifest_does_not_list(tmp_path, monkeypatch):
    listed = _a_listed_model()
    pruner, hub = _hub_with(tmp_path, monkeypatch, listed, "google/flan-t5-base")

    assert pruner.main(["--tier", "production", "--hub", str(hub)]) == 0

    assert (hub / pruner.dir_name_for(listed)).is_dir(), "a listed model must survive"
    assert not (hub / "models--google--flan-t5-base").exists(), "an unlisted model must go"


def test_dry_run_deletes_nothing(tmp_path, monkeypatch):
    listed = _a_listed_model()
    pruner, hub = _hub_with(tmp_path, monkeypatch, listed, "google/flan-t5-base")

    assert pruner.main(["--tier", "production", "--dry-run", "--hub", str(hub)]) == 0

    assert (hub / "models--google--flan-t5-base").is_dir(), "--dry-run must not delete"
    assert (hub / pruner.dir_name_for(listed)).is_dir()


def test_leaves_non_model_entries_alone(tmp_path, monkeypatch):
    """The hub root also holds `.locks`, `version.txt` and similar. Only models-- dirs."""
    listed = _a_listed_model()
    pruner, hub = _hub_with(tmp_path, monkeypatch, listed, "google/flan-t5-base")
    (hub / ".locks").mkdir()
    (hub / "version.txt").write_text("1")

    assert pruner.main(["--tier", "production", "--hub", str(hub)]) == 0

    assert (hub / ".locks").is_dir()
    assert (hub / "version.txt").is_file()


def test_works_when_the_package_is_not_importable(tmp_path, monkeypatch):
    """The path CI actually takes, and the one that shipped broken.

    This script can run BEFORE `pip install`, where `import podcast_scraper` fails and the
    loader must fall back to stubbing the parent packages and loading the leaf module by
    file. The first version copied that loader from verify_required_models.py and stubbed
    the parent with `__path__ = []` instead of the real source directory, so
    model_manifest's own `from podcast_scraper import config_constants` could not resolve:

        ImportError: cannot import name 'config_constants' from 'podcast_scraper'

    It passed every test here and died in CI, because locally the package IS importable —
    the normal import succeeds and the fallback never runs. So force it.
    """
    import importlib
    import sys

    listed = _a_listed_model()
    pruner, hub = _hub_with(tmp_path, monkeypatch, listed, "google/flan-t5-base")

    # Make the ordinary import fail exactly as it does on a runner without the package,
    # and drop anything already imported so the fallback is genuinely exercised.
    for name in [m for m in sys.modules if m.startswith("podcast_scraper")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    real = importlib.import_module

    def _no_package(name, *a, **kw):
        if name.startswith("podcast_scraper"):
            raise ImportError("simulated pre-install environment")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", _no_package)

    assert pruner.main(["--tier", "production", "--hub", str(hub)]) == 0
    assert not (hub / "models--google--flan-t5-base").exists()
    assert (hub / pruner.dir_name_for(listed)).is_dir()
