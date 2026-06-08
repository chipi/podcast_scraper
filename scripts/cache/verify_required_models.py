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

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _whisper_cached(name: str) -> bool:
    return (Path.home() / ".cache" / "whisper" / f"{name}.pt").is_file()


def _hf_cached(model_id: str) -> bool:
    """True if the HF hub cache has a non-empty dir for ``model_id``."""
    from podcast_scraper.cache.directories import get_transformers_cache_dir

    cache_dir = get_transformers_cache_dir()
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
    from podcast_scraper.providers.ml import model_manifest as mm

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
