"""Build each episode's audio reference (RFC-129, #2359).

Every episode needs one: ads change within hours, so any episode can drift, and the app checks
the stream it plays against the reference. New episodes get theirs at ingest, from the audio just
transcribed (``write_for_episode``). This command backfills the rest from the archive.

For each selected episode:
1. Its original comes from the audio archive (prod-audio-archive (not in this repository)),
   only when the archive's provenance says it is byte-identical to the audio we transcribed. A later
   re-download carries other ads, and would make every offset wrong.
2. The ad regions are the ones our transcript already marks (``episode_ad_spans``).
3. The reference (fingerprints at checkpoints, ``podcast_scraper.audio_sync.reference``) is written
   next to the episode's transcript as ``<stem>.audioref.json``.

Read-only on the archive. Without ``--write`` it only reports. With ``--write`` each file is
written atomically; undo = delete the ``*.audioref.json`` files. No DGX, no model: CPU and ffmpeg.

    python -m podcast_scraper.audio_sync build-references --corpus <root> \\
        (--rclone-remote NAME [--base-path P] | --local-root DIR) \\
        [--path <metadata path>]... [--force] [--limit N] [--write]

Without ``--path``, every episode in the corpus. An episode that already has a reference is skipped
unless ``--force``, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set

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
    """``<stem>.audioref.json`` beside an episode's segments file (raw or ad-free)."""
    base = segments_relpath[: -len(".segments.json")]
    return base.removesuffix(".adfree") + REF_SUFFIX


def write_for_episode(root: Path, row: Any, audio_path: str, guid: Optional[str]) -> Optional[Path]:
    """Build and write one episode's reference from ``audio_path``, which must be the exact audio
    its transcript was made from. Returns the file written, or ``None`` when the episode has no
    transcript segments to sit next to."""
    from podcast_scraper.audio_sync import fingerprint, reference
    from podcast_scraper.server.app_episode_search import episode_ad_spans, episode_segments_relpath

    seg_rel = episode_segments_relpath(root, row)
    if seg_rel is None:
        return None
    built = reference.build(fingerprint.decode(audio_path), episode_ad_spans(root, row))
    if guid:
        built["guid"] = guid
    out = root / reference_relpath(seg_rel)
    tmp_out = out.with_name(out.name + ".tmp")
    tmp_out.write_text(json.dumps(built), encoding="utf-8")
    os.replace(tmp_out, out)
    return out


def write_for_run(corpus_root: str, run_dir: str) -> Dict[str, int]:
    """At the end of an ingest run: a reference for each of the run's episodes whose audio is still
    on disk, which is the exact audio its transcript was just made from. Best-effort per episode;
    returns counts by outcome."""
    import logging

    from podcast_scraper.archive.offload import _metadata_dir, _safe_media_path
    from podcast_scraper.server.catalog_cache import cached_catalog

    log = logging.getLogger(__name__)
    root = Path(corpus_root)
    counts: Dict[str, int] = {}
    rows = None
    for meta in sorted(_metadata_dir(run_dir).glob("*.metadata.json")):
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
            rel = str(meta.resolve().relative_to(root.resolve()))
        except (OSError, ValueError):
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue
        audio_relpath = str((doc.get("content") or {}).get("audio_relpath") or "")
        media = _safe_media_path(run_dir, audio_relpath) if audio_relpath else None
        if media is None or not media.is_file():
            counts["no_audio"] = counts.get("no_audio", 0) + 1
            continue
        if rows is None:
            rows = {r.metadata_relative_path: r for r in cached_catalog(root)}
        row = rows.get(rel)
        if row is None:
            counts["not_in_catalog"] = counts.get("not_in_catalog", 0) + 1
            continue
        try:
            guid = (doc.get("episode") or {}).get("guid")
            out = write_for_episode(root, row, str(media), str(guid) if guid else None)
            key = "written" if out else "no_segments"
        except Exception:  # noqa: BLE001 — one episode must never break the run
            log.warning("audio reference: failed for %s", rel, exc_info=True)
            key = "failed"
        counts[key] = counts.get(key, 0) + 1
    if counts:
        log.info("audio references (%s): %s", os.path.basename(run_dir), counts)
    return counts


def _backend(args: argparse.Namespace):
    from podcast_scraper.utils.storage_backend import LocalStorageBackend, RcloneStorageBackend

    if args.rclone_remote:
        return RcloneStorageBackend(remote=args.rclone_remote, base_path=args.base_path or "")
    return LocalStorageBackend(Path(args.local_root))


def main(argv: Optional[list] = None) -> int:
    """``build-references``: backfill references from the archive's exact originals."""
    ap = argparse.ArgumentParser(
        prog="podcast_scraper.audio_sync build-references",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rclone-remote")
    src.add_argument("--local-root")
    ap.add_argument("--base-path", default="")
    ap.add_argument("--path", action="append", default=[], help="metadata path(s), repeatable")
    ap.add_argument("--force", action="store_true", help="rebuild references that exist")
    ap.add_argument("--limit", type=int, default=0, help="at most this many built (0 = all)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    from podcast_scraper.server.app_episode_search import episode_segments_relpath
    from podcast_scraper.server.catalog_cache import cached_catalog
    from podcast_scraper.utils import audio_cache

    root: Path = args.corpus
    wanted = set(args.path)
    exact = exact_originals(root)
    backend = _backend(args)
    counts: Dict[str, int] = {}

    def tally(outcome: str, rel: str) -> None:
        counts[outcome] = counts.get(outcome, 0) + 1
        print(f"{outcome:24} {rel}")

    built = 0
    for row in cached_catalog(root):
        rel = row.metadata_relative_path
        if wanted and rel not in wanted:
            continue
        if args.limit and built >= args.limit:
            break
        doc = json.loads((root / rel).read_text(encoding="utf-8"))
        guid = (doc.get("episode") or {}).get("guid")
        seg_rel = episode_segments_relpath(root, row)
        if not guid or seg_rel is None:
            tally("skip_no_guid_or_segments", rel)
            continue
        if not args.force and (root / reference_relpath(seg_rel)).is_file():
            tally("skip_exists", rel)
            continue
        if guid not in exact:
            tally("skip_no_exact_original", rel)
            continue
        if not args.write:
            tally("would_write", reference_relpath(seg_rel))
            built += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "original.mp3")
            if not audio_cache.fetch_into(backend, guid, audio):
                tally("skip_not_in_archive", rel)
                continue
            out = write_for_episode(root, row, audio, guid)
        built += 1
        tally("written", str(out.relative_to(root)) if out else rel)
    print(" · ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no episodes selected")
    if not args.write:
        print("dry run: nothing written (add --write)")
    return 0
