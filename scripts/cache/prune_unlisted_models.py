#!/usr/bin/env python3
"""Delete HF hub model directories the manifest does not list.

    python3 scripts/cache/prune_unlisted_models.py --tier production [--dry-run]

WHY
  The CI model cache only ever grows. A model removed from ``REQUIRED_ML_MODELS`` stops
  being preloaded, but the copy already in the cache is restored on every run and
  re-uploaded in the ml-models artifact, so the artifact keeps carrying it. Salting the
  cache key does not help either: ``restore-keys`` warm-start from the previous cache, so
  the stale model comes straight back and is saved under the new key.

  ADR-154 made this concrete. Retiring the hybrid summariser removed google/flan-t5-base
  (1895 MB in the cache) and google/long-t5-tglobal-base (3781 MB) from the manifest, and
  the artifact carried both regardless — 5.5 GB shipped to every test job for a provider
  that no longer exists.

  So the cache is reconciled against the manifest rather than trusted to shrink: whatever
  the tier does not list is not in the artifact. The manifest is the single source of
  truth (#917) for what CI carries, and this makes that true of the cache too.

SAFETY
  Only ``models--*`` directories directly under the hub root are considered, only the
  named tier decides, and ``--dry-run`` prints without deleting. Whisper lives in a
  separate cache and is never touched.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"


def _ensure_src_on_path() -> None:
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))


def _load_light(module_name: str, rel_path: str):
    """Import a leaf ``podcast_scraper`` module without the heavy deps.

    Same approach as verify_required_models.py: this may run before ``pip install``.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 — pre-install path: stub the parents, load the leaf by file
        parts = module_name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                stub = types.ModuleType(parent)
                stub.__path__ = []  # type: ignore[attr-defined]
                sys.modules[parent] = stub
        spec = importlib.util.spec_from_file_location(module_name, str(_SRC / rel_path))
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod


def dir_name_for(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", default="production", help="manifest tier that defines what to keep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    _ensure_src_on_path()
    mm = _load_light(
        "podcast_scraper.providers.ml.model_manifest",
        "podcast_scraper/providers/ml/model_manifest.py",
    )
    directories = _load_light(
        "podcast_scraper.cache.directories", "podcast_scraper/cache/directories.py"
    )
    hub = directories.get_transformers_cache_dir()
    if not hub.is_dir():
        print(f"no hub cache at {hub} — nothing to prune")
        return 0

    keep = {dir_name_for(s.model_id) for s in mm.models_for_tier(args.tier) if s.kind != "whisper"}
    freed = 0
    for entry in sorted(hub.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        if entry.name in keep:
            print(f"  keep   {entry.name}")
            continue
        size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        freed += size
        print(f"  PRUNE  {entry.name}  ({size / 1e6:.1f} MB){' [dry-run]' if args.dry_run else ''}")
        if not args.dry_run:
            shutil.rmtree(entry)
    print(f"\n{'would free' if args.dry_run else 'freed'}: {freed / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
