#!/usr/bin/env python3
"""ADR-134 backfill — build ``.context.json`` for an already-processed corpus, no reprocess.

The context digest is a pure, deterministic denormalization of artifacts we already have
(`.gi.json` / `.kg.json` / `.metadata.json`), so it can be produced for every episode of a
processed corpus without re-running the pipeline or any LLM.

Target: the latest ``prod-v2.3-turbo`` corpus (default below). Walks every ``*.metadata.json``
under the root and writes a sibling ``*.context.json``.

Note: the ``voices`` split (``unknown``/``unidentified``) is NOT recoverable here — the per-voice
classification is not persisted in these artifacts (only aggregate ``by_voice_type`` counts). The
backfill therefore writes ``voices = {total, labels, unknown: null, unidentified: null}``; the split
is populated only on new runs (or once per-voice classification is persisted to the manifest).

Run from the repo root:

    .venv/bin/python scripts/dev/backfill_context_digests.py [--corpus-dir DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from podcast_scraper.builders.bridge_artifact_paths import (  # noqa: E402
    context_json_path_adjacent_to_metadata,
)
from podcast_scraper.builders.context_digest_builder import build_context_digest  # noqa: E402

_DEFAULT_CORPUS = _REPO_ROOT / ".test_outputs" / "manual" / "prod-v2.3-turbo"


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _backfill_one(md_path: Path, *, dry_run: bool) -> bool:
    """Build + write the digest for one episode. Returns True if a file was written."""
    md = _load(md_path)
    if not isinstance(md, dict):
        return False
    mp = str(md_path)
    if not mp.endswith(".metadata.json"):
        return False
    base = mp[: -len(".metadata.json")]
    gi = _load(Path(base + ".gi.json"))
    kg = _load(Path(base + ".kg.json"))
    episode_id = md.get("episode", {}).get("episode_id") or md_path.stem
    digest = build_context_digest(str(episode_id), gi_artifact=gi, kg_artifact=kg, metadata=md)
    out_path = Path(context_json_path_adjacent_to_metadata(str(md_path)))
    if dry_run:
        return False
    out_path.write_text(
        json.dumps(digest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = parser.parse_args()

    root: Path = args.corpus_dir
    md_files = sorted(root.rglob("*.metadata.json"))
    if not md_files:
        print(f"No .metadata.json under {root}", file=sys.stderr)
        return 1

    written = 0
    for md in md_files:
        if _backfill_one(md, dry_run=args.dry_run):
            written += 1
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"Done. {verb} {written if not args.dry_run else len(md_files)} .context.json "
        f"across {len(md_files)} episodes under {root}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
