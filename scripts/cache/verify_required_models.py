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

import importlib.util
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _whisper_cached(name: str) -> bool:
    return (Path.home() / ".cache" / "whisper" / f"{name}.pt").is_file()


def _spacy_cached(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _hf_cached(model_id: str) -> bool:
    """True if the HF hub cache has a non-empty dir for ``model_id``."""
    from podcast_scraper.cache.directories import get_transformers_cache_dir

    cache_dir = get_transformers_cache_dir()
    model_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    if not model_dir.is_dir():
        return False
    return any(p.is_file() for p in model_dir.rglob("*"))


def main() -> int:
    _ensure_src_on_path()
    from podcast_scraper.providers.ml import model_manifest as mm

    checks = {
        "whisper": _whisper_cached,
        "spacy": _spacy_cached,
        "summary": _hf_cached,
        "embedding": _hf_cached,
        "qa": _hf_cached,
        "nli": _hf_cached,
    }

    missing: list[str] = []
    print("Verifying CI artifact models (from model_manifest):")
    for spec in mm.models_for_tier("ci_artifact"):
        check = checks.get(spec.kind)
        if check is None:  # diarization etc. are not ci_artifact, but be safe
            continue
        ok = check(spec.model_id)
        print(f"  [{'OK ' if ok else 'MISS'}] {spec.kind:9} {spec.model_id}")
        if not ok:
            missing.append(f"{spec.kind}:{spec.model_id}")

    if missing:
        print(f"\nERROR: required models not cached: {', '.join(missing)}")
        return 1
    print("\nAll required ml-models-artifact models are cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
