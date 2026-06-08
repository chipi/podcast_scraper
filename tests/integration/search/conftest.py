"""TEMP diagnostic conftest for the offline search tests.

The 6 search tests that genuinely load MiniLM fail in CI with
``existing_files = []`` from ``transformers.cached_files`` -- the model is on
disk and loads fine in the ensure-step process seconds earlier, yet the pytest
worker cannot find ``config.json`` for ``sentence-transformers/all-MiniLM-L6-v2``
in the (correct) cache dir.

This autouse fixture dumps the failing worker's cache reality to ``sys.stderr``
*per test* so it surfaces under "Captured stderr setup" in each failure report
(the channel that actually survives pytest's fd-level capture). Remove once the
CI MiniLM load is understood.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import pytest

_DUMPED = False


@pytest.fixture(autouse=True)
def _minilm_cache_probe():
    global _DUMPED
    if _DUMPED:
        yield
        return
    _DUMPED = True

    out: list[str] = ["", "==== MiniLM search-worker probe ===="]
    for k in (
        "HF_HUB_CACHE",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "SENTENCE_TRANSFORMERS_HOME",
        "XDG_CACHE_HOME",
        "HOME",
    ):
        out.append(f"  env {k}={os.environ.get(k)!r}")

    cdir = None
    try:
        from podcast_scraper.cache.directories import get_transformers_cache_dir

        cdir = Path(get_transformers_cache_dir())
        out.append(f"  get_transformers_cache_dir()={cdir}")
    except Exception:
        out.append("  get_transformers_cache_dir FAILED:")
        out.append(traceback.format_exc())

    if cdir is not None:
        mini = cdir / "models--sentence-transformers--all-MiniLM-L6-v2"
        out.append(f"  MiniLM dir exists={mini.is_dir()} path={mini}")
        refs_main = mini / "refs" / "main"
        if refs_main.exists():
            out.append(f"  refs/main -> {refs_main.read_text().strip()!r}")
        else:
            out.append("  refs/main MISSING")
        if mini.is_dir():
            for p in sorted(mini.rglob("*")):
                rel = p.relative_to(mini)
                try:
                    if p.is_symlink():
                        tgt = os.readlink(p)
                        resolved = (p.parent / tgt).resolve()
                        broken = "" if resolved.exists() else "  <<< BROKEN"
                        out.append(f"    {rel} -> {tgt}{broken}")
                    else:
                        out.append(f"    {rel} ({p.stat().st_size}B)")
                except OSError as exc:
                    out.append(f"    {rel} <stat-error {exc}>")

        # What transformers' own cache lookup returns for config.json.
        try:
            from transformers.utils.hub import try_to_load_from_cache

            for rev in ("main", None):
                res = try_to_load_from_cache(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "config.json",
                    cache_dir=str(cdir),
                    revision=rev,
                )
                out.append(f"  try_to_load_from_cache(config.json, rev={rev!r}) -> {res!r}")
        except Exception:
            out.append("  try_to_load_from_cache FAILED:")
            out.append(traceback.format_exc())

    # Live load via the exact path the tests use.
    try:
        from podcast_scraper import config_constants as _cc
        from podcast_scraper.providers.ml.embedding_loader import get_embedding_model

        get_embedding_model(_cc.DEFAULT_EMBEDDING_MODEL)
        out.append("  LOAD: OK")
    except Exception as exc:
        out.append(f"  LOAD: FAILED ({type(exc).__name__}: {exc})")

    out.append("==== end MiniLM search-worker probe ====")
    print("\n".join(out), file=sys.stderr, flush=True)
    yield
