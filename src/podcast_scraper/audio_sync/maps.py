"""Build ``<corpus>/audio_time_maps.json`` for drifted episodes (RFC-129, #2359).

For each episode in ``<corpus>/audio_timing_drift.json`` (``drift-check``), or each ``--path``:
1. The original comes from the audio archive, only when its provenance says it is byte-identical to
   what we transcribed (as ``build-references``).
2. The stream is fetched as a phone browser would fetch it now.
3. The two are lined up by fingerprint (``fingerprint.align_samples``).
4. The map is the offset of each stretch of talk from its start on: pairs
   ``[transcript_s, offset_s]``, as ``server.app_audio_drift.time_maps`` reads them.

A map measured from our server is right for listeners who get the same ads as the server, and
close for most others (measured 2026-10-16: within 0-13 s where no map is 4-76 s off). The api uses
it inside the episode only. Drifted episodes stay out of short listens and Learn whatever the map
says ("Estimated" in RFC-129).

Read-only on the archive and the hosts. Without ``--write`` it only reports. With ``--write``:
- in drift-file mode, the maps file is replaced by this run's maps, so episodes that no longer drift
  lose theirs;
- with ``--path``, this run's maps are merged into the file.
Writes are atomic. Undo: delete ``audio_time_maps.json``.

    python -m podcast_scraper.audio_sync build-maps --corpus <root> \\
        (--rclone-remote NAME [--base-path P] | --local-root DIR) \\
        [--path <metadata path>]... [--write]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

MIN_TALK_SHARE = 0.5


def main(argv: Optional[list] = None) -> int:
    from podcast_scraper.audio_sync import fingerprint
    from podcast_scraper.audio_sync.references import _backend, exact_originals
    from podcast_scraper.server.app_audio_drift import DRIFT_FILE, MAPS_FILE
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.utils import audio_cache

    ap = argparse.ArgumentParser(
        prog="podcast_scraper.audio_sync build-maps",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rclone-remote")
    src.add_argument("--local-root")
    ap.add_argument("--base-path", default="")
    ap.add_argument("--path", action="append", default=[], help="metadata path(s), repeatable")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    root: Path = args.corpus
    wanted = set(args.path)
    if not wanted:
        try:
            drift = json.loads((root / DRIFT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"no {DRIFT_FILE}: run drift-check --write first")
            return 1
        wanted = set(drift.get("episodes", {}))
    exact = exact_originals(root)
    backend = _backend(args)
    maps: Dict[str, Any] = {}
    counts: Dict[str, int] = {}

    def tally(outcome: str, rel: str, note: str = "") -> None:
        counts[outcome] = counts.get(outcome, 0) + 1
        print(f"{outcome:24} {rel} {note}".rstrip())

    for row in cached_catalog(root):
        rel = row.metadata_relative_path
        if rel not in wanted:
            continue
        doc = json.loads((root / rel).read_text(encoding="utf-8"))
        guid = (doc.get("episode") or {}).get("guid")
        url = (doc.get("content") or {}).get("media_url")
        if not guid or not isinstance(url, str) or not url.startswith(("http://", "https://")):
            tally("skip_no_guid_or_url", rel)
            continue
        if guid not in exact:
            tally("skip_no_exact_original", rel)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "original.mp3")
            if not audio_cache.fetch_into(backend, guid, audio):
                tally("skip_not_in_archive", rel)
                continue
            original = fingerprint.decode(audio)
        try:
            stream = fingerprint.decode(url, timeout_s=900)
        except Exception as exc:  # noqa: BLE001 — one host down must not stop the run
            tally("skip_stream_unreadable", rel, type(exc).__name__)
            continue
        talk = fingerprint.align_samples(original, stream)
        share = sum(s.seconds for s in talk) / max(1.0, len(original) / fingerprint.SR)
        if share < MIN_TALK_SHARE:
            tally("skip_poor_match", rel, f"talk found {share:.0%}")
            continue
        maps[rel] = {
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": "fingerprint",
            "talk_share": round(share, 3),
            "anchors": fingerprint.time_map(talk),
        }
        tally("mapped", rel, f"{len(talk)} stretches, talk found {share:.0%}")

    print(" · ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no episodes selected")
    if not args.write:
        print("dry run: nothing written (add --write)")
        return 0
    out = root / MAPS_FILE
    current: Dict[str, Any] = {}
    if args.path:
        try:
            current = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
    current.update(maps)
    tmp_out = out.with_name(out.name + ".tmp")
    tmp_out.write_text(json.dumps(current, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp_out, out)
    print(f"wrote {len(current)} maps to {out}")
    return 0
