"""Verify the CI ml-models artifact carries every model the test jobs need (#917).

Reads the single manifest (``model_manifest.ci_artifact_model_ids``) and checks
each is present in the local caches. This replaces the hand-maintained,
~4x-duplicated bash arrays in ``.github/workflows/python-app.yml`` -- the drift
that let MiniLM go missing from CI and break the offline search tests (#897).

Usage (CI cache-validation step):
    python scripts/cache/verify_required_models.py

Exits non-zero and prints the missing models if any required model is absent.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"


def _ensure_src_on_path() -> None:
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))


def _load_light(module_name: str, rel_path: str):
    """Import a *lightweight* ``podcast_scraper`` submodule without the heavy deps.

    This verifier runs at the CI "validate cache" step BEFORE dependencies are
    installed (to fail fast before the expensive ``[ml]`` install), so importing
    the package normally explodes: ``podcast_scraper/__init__`` pulls ``config``
    (-> ``yaml``) and ``providers/ml/__init__`` pulls ``MLProvider`` (-> ``torch``).
    The data this script needs (the model manifest, the HF cache dir) lives in
    leaf modules that only use the stdlib. Try a normal import first (works once
    deps are installed -- the preload job, local use); on failure, register
    namespace stubs for the parent packages so the leaf module loads by file
    without executing any ``__init__``.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        parts = module_name.split(".")
        for i in range(1, len(parts)):
            pkg = ".".join(parts[:i])
            if pkg not in sys.modules:
                stub = types.ModuleType(pkg)
                stub.__path__ = [str(_SRC / Path(*parts[:i]))]  # type: ignore[attr-defined]
                sys.modules[pkg] = stub
        spec = importlib.util.spec_from_file_location(module_name, str(_SRC / rel_path))
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod


def _whisper_cached(name: str) -> bool:
    return (Path.home() / ".cache" / "whisper" / f"{name}.pt").is_file()


def _hf_cached(model_id: str) -> bool:
    """True if the HF hub cache has a non-empty dir for ``model_id``."""
    directories = _load_light(
        "podcast_scraper.cache.directories", "podcast_scraper/cache/directories.py"
    )
    cache_dir = directories.get_transformers_cache_dir()
    model_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    if not model_dir.is_dir():
        return False
    return any(p.is_file() for p in model_dir.rglob("*"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        default="ci_artifact",
        help=(
            "Manifest tier to verify (default ci_artifact, for the consumer test jobs). "
            "Nightly passes 'production' to also verify the pyannote diarization cache."
        ),
    )
    args = parser.parse_args(argv)

    _ensure_src_on_path()
    mm = _load_light(
        "podcast_scraper.providers.ml.model_manifest",
        "podcast_scraper/providers/ml/model_manifest.py",
    )

    # Kinds delivered via the ml-models artifact (HF hub cache / whisper cache).
    artifact_checks = {
        "whisper": _whisper_cached,
        "summary": _hf_cached,
        "embedding": _hf_cached,
        "qa": _hf_cached,
        "nli": _hf_cached,
        "diarization": _hf_cached,  # pyannote pipeline repo (HF cache)
    }
    # spaCy (en_core_web_sm) is delivered by `pip install .[ml]`, NOT the ml-models
    # artifact, and this verifier runs at the pre-install Validate step -- so it is
    # not importable yet. Don't gate on it here; the [ml] install guarantees it.
    pip_delivered_kinds = {"spacy"}

    missing: list[str] = []
    print(f"Verifying '{args.tier}' models (from model_manifest):")
    for spec in mm.models_for_tier(args.tier):
        if spec.kind in pip_delivered_kinds:
            print(f"  [skip] {spec.kind:9} {spec.model_id} (pip-delivered, not in artifact)")
            continue
        check = artifact_checks.get(spec.kind)
        if check is None:
            # An unrecognised ci_artifact kind is a manifest/verifier drift -- fail
            # loudly rather than silently passing it unchecked.
            print(f"  [ERR ] {spec.kind:9} {spec.model_id} (no artifact check for kind)")
            missing.append(f"{spec.kind}:{spec.model_id} (unknown kind)")
            continue
        ok = check(spec.model_id)
        print(f"  [{'OK ' if ok else 'MISS'}] {spec.kind:9} {spec.model_id}")
        if not ok:
            missing.append(f"{spec.kind}:{spec.model_id}")

    if missing:
        print(f"\nERROR: required models not cached: {', '.join(missing)}")
        return 1
    print(f"\nAll required '{args.tier}' models are cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
