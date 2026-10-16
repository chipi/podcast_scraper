#!/usr/bin/env python3
"""Build each episode's audio reference from the archived original (RFC-129, #2359).

For each selected episode:
1. Its original comes from the audio archive (``docs/recipes/prod-audio-archive.md``), only when
   the archive's provenance says it is byte-identical to the audio we transcribed. A later
   re-download carries other ads, and would make every offset wrong.
2. The ad regions are the ones our transcript already marks (``episode_ad_spans``).
3. The reference (fingerprints at checkpoints, ``podcast_scraper.audio_sync.reference``) is written
   next to the episode's transcript as ``<stem>.audioref.json``.

Read-only on the archive. Without ``--write`` it only reports. With ``--write`` each file is
written atomically; undo = delete the ``*.audioref.json`` files. No DGX, no model: CPU and ffmpeg.

    .venv/bin/python scripts/ops/audio_reference_build.py --corpus <root> \\
        (--rclone-remote NAME [--base-path P] | --local-root DIR) \\
        [--path <metadata path>]... [--write]

Without ``--path``, the episodes in ``<corpus>/audio_timing_drift.json`` (``audio_drift_check.py``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Set

REF_SUFFIX = ".audioref.json"


def exact_originals(corpus: Path) -> Set[str]:
    """GUIDs whose archived audio is byte-identical to what we transcribed, from every provenance
    file in the corpus (the pipeline writes one per run directory)."""
    good: Set[str] = set()
    pattern = str(corpus / "**" / ".podcast_scraper" / "audio-archive-provenance.jsonl")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("byte_identical_to_transcribed_audio") is True and row.get("guid"):
                    good.add(str(row["guid"]))
    return good


def reference_relpath(segments_relpath: str) -> str:
    base = segments_relpath[: -len(".segments.json")]
    return base.removesuffix(".adfree") + REF_SUFFIX


def _backend(args: argparse.Namespace):
    from podcast_scraper.utils.storage_backend import LocalStorageBackend, RcloneStorageBackend

    if args.rclone_remote:
        return RcloneStorageBackend(remote=args.rclone_remote, base_path=args.base_path or "")
    return LocalStorageBackend(Path(args.local_root))


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rclone-remote")
    src.add_argument("--local-root")
    ap.add_argument("--base-path", default="")
    ap.add_argument("--path", action="append", default=[], help="metadata path(s), repeatable")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    from podcast_scraper.audio_sync import fingerprint, reference
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_episode_search import episode_ad_spans, episode_segments_relpath
    from podcast_scraper.utils import audio_cache

    root: Path = args.corpus
    wanted = set(args.path)
    if not wanted:
        drift = json.loads((root / "audio_timing_drift.json").read_text(encoding="utf-8"))
        wanted = set(drift.get("episodes", {}))
    exact = exact_originals(root)
    backend = _backend(args)
    counts: Dict[str, int] = {}

    def tally(outcome: str, rel: str) -> None:
        counts[outcome] = counts.get(outcome, 0) + 1
        print(f"{outcome:24} {rel}")

    for row in cached_catalog(root):
        rel = row.metadata_relative_path
        if rel not in wanted:
            continue
        doc = json.loads((root / rel).read_text(encoding="utf-8"))
        guid = (doc.get("episode") or {}).get("guid")
        seg_rel = episode_segments_relpath(root, row)
        if not guid or seg_rel is None:
            tally("skip_no_guid_or_segments", rel)
            continue
        if guid not in exact:
            tally("skip_no_exact_original", rel)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "original.mp3")
            if not audio_cache.fetch_into(backend, guid, audio):
                tally("skip_not_in_archive", rel)
                continue
            built = reference.build(fingerprint.decode(audio), episode_ad_spans(root, row))
        built["guid"] = guid
        out = root / reference_relpath(seg_rel)
        if args.write:
            tmp_out = out.with_name(out.name + ".tmp")
            tmp_out.write_text(json.dumps(built), encoding="utf-8")
            os.replace(tmp_out, out)
        tally(
            "written" if args.write else "would_write",
            f"{out.relative_to(root)} " f"({len(built['checkpoints'])} checkpoints)",
        )
    print(" · ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no episodes selected")
    if not args.write:
        print("dry run: nothing written (add --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
