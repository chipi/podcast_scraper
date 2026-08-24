"""Standalone corpus (re)indexing entry point — isolated from the main pipeline process.

Rebuild a corpus's LanceDB search index without re-transcribing:

    python -m podcast_scraper.search.reindex <corpus_dir> --config config/profiles/<p>.yaml

This module is also how the pipeline runs indexing: ``finalize_multi_feed_batch`` spawns it in
a subprocess via :func:`run_index_in_subprocess`. Isolation matters because the Arrow/LanceDB
layer can crash *natively*: pyarrow's bundled **mimalloc** allocator segfaults in
``mi_thread_init`` on a freshly-spawned LanceDB background-loop worker thread when pthread
thread-local storage is under pressure (many live threads: torch, gRPC/OTEL, sentry, langfuse).
Observed once as ``EXC_BAD_ACCESS`` in ``libarrow`` during a corpus index at the end of a full
run. A native signal is uncatchable, so an in-process ``index_corpus`` call takes the whole run
down with it — defeating the "non-fatal index" contract. Running it in a subprocess turns that
crash into a non-zero exit the parent logs and survives, and the corpus stays rebuildable.

The cause fix rides here too: the Arrow C++ default memory pool is chosen at first allocation
from ``ARROW_DEFAULT_MEMORY_POOL``. We pin it to ``system`` (off mimalloc) BEFORE pyarrow is
imported — which a fresh subprocess guarantees.
"""

from __future__ import annotations

import os

# MUST run before any transitive ``import pyarrow`` — Arrow reads this at first pool access, and
# pyarrow's default pool is created at import. ``setdefault`` respects an operator override.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

from podcast_scraper import config  # noqa: E402

logger = logging.getLogger(__name__)


def run_index_in_subprocess(
    corpus_parent: str,
    idx_cfg: "config.Config",
    *,
    rebuild: bool = False,
    timeout: float = 1800.0,
    backbone_changed_relpaths: Optional[Sequence[str]] = None,
) -> bool:
    """Build the corpus index in a clean subprocess. Return True on success.

    Never raises for an index failure: a Python error, a non-zero exit, OR a native crash
    (segfault → negative ``returncode``) all resolve to ``False`` so the caller can log it as
    non-fatal. This is what makes a native Arrow/LanceDB crash survivable — it can no longer take
    the parent process down.

    ``backbone_changed_relpaths`` (RFC-118) carries the orchestrator's corpus-delta scope across
    the process boundary as a temp JSON file — corpus-root-relative metadata paths of the
    episodes the backbone marked changed. ``None`` (no delta computed) omits the flag entirely.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".reindex-cfg.json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(idx_cfg.model_dump(mode="json"), fh)
        cfg_path = fh.name
    delta_path: Optional[str] = None
    if backbone_changed_relpaths is not None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".reindex-delta.json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(sorted(backbone_changed_relpaths), fh)
            delta_path = fh.name
    env = dict(os.environ)
    env["ARROW_DEFAULT_MEMORY_POOL"] = "system"
    argv = [
        sys.executable,
        "-m",
        "podcast_scraper.search.reindex",
        str(corpus_parent),
        "--config-json",
        cfg_path,
    ]
    if rebuild:
        argv.append("--rebuild")
    if delta_path is not None:
        argv.extend(["--backbone-changed-file", delta_path])
    try:
        proc = subprocess.run(argv, env=env, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("Corpus index subprocess timed out after %.0fs (non-fatal)", timeout)
        return False
    finally:
        for p in (cfg_path, delta_path):
            if p is None:
                continue
            try:
                os.unlink(p)
            except OSError:
                pass
    if proc.returncode != 0:
        # Negative returncode = killed by signal N (e.g. -11 == SIGSEGV): the isolation working.
        logger.warning(
            "Corpus index subprocess exited %s (non-fatal); corpus is rebuildable via "
            "`python -m podcast_scraper.search.reindex %s`",
            proc.returncode,
            corpus_parent,
        )
        return False
    return True


def _load_cfg(args: argparse.Namespace) -> "config.Config":
    if args.config_json:
        data = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        return config.Config.model_validate(data)
    if args.config:
        return config.Config.model_validate(config.load_config_file(args.config))
    raise SystemExit("reindex: one of --config or --config-json is required")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: rebuild a corpus's LanceDB search index from on-disk episode metadata."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper.search.reindex",
        description="Rebuild a corpus's LanceDB search index from on-disk episode metadata.",
    )
    parser.add_argument("corpus", help="Corpus directory (the parent that holds feeds/).")
    parser.add_argument("--config", help="Profile/config YAML or JSON (for standalone use).")
    parser.add_argument(
        "--config-json", help="Path to a serialized Config JSON (used by the pipeline)."
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop and rebuild the index from scratch."
    )
    parser.add_argument(
        "--backbone-changed-file",
        help="JSON list of corpus-root-relative metadata paths the RFC-118 backbone delta "
        "marked changed (written by run_index_in_subprocess).",
    )
    args = parser.parse_args(argv)

    cfg = _load_cfg(args)
    # Standalone --config gives a profile with no corpus binding; force it onto this corpus and
    # ensure the auto-index path is enabled (the same overrides finalize applies).
    cfg = cfg.model_copy(update={"output_dir": args.corpus, "skip_auto_vector_index": False})

    backbone_changed = None
    if args.backbone_changed_file:
        try:
            loaded = json.loads(Path(args.backbone_changed_file).read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                backbone_changed = {str(x) for x in loaded}
        except (OSError, ValueError) as exc:
            # The delta is observational — an unreadable file must not fail the reindex.
            logger.warning("could not read --backbone-changed-file (%s); ignoring delta", exc)

    from podcast_scraper.search.indexer import index_corpus

    stats = index_corpus(
        args.corpus, cfg, rebuild=args.rebuild, backbone_changed_relpaths=backbone_changed
    )
    logger.info("reindex complete: %s", stats)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
