"""Build ``<corpus>/audio_time_maps.json`` for drifted episodes (RFC-129, #2359).

For each episode in ``<corpus>/audio_timing_drift.json`` (``drift-check``), or each ``--path``:
1. Its audio reference (``<stem>.audioref.json`` next to the transcript, written at ingest or by the
   backfill): fingerprint windows of the audio we transcribed.
2. The stream as a listener's player gets it now (``drift.served_url``), decoded once.
3. Each checkpoint is found anywhere in the stream (``reference.locate_everywhere``); its offset
   holds from its transcript second on.

No archive: the archive is a one-time source for references (``build-references``), never part of
this daily job. An episode without a reference gets no map.

A map measured from our server is right for listeners who get the same ads, and close for most
others. The player prefers its own measurement; this is the fallback when it cannot measure.

Read-only on the hosts. Without ``--write`` it only reports. With ``--write``:
- in drift-file mode, the maps file is replaced by this run's maps, so episodes that no longer drift
  lose theirs;
- with ``--path``, this run's maps are merged into the file.
Writes are atomic. Undo: delete ``audio_time_maps.json``.

    python -m podcast_scraper.audio_sync build-maps --corpus <root> \\
        [--path <metadata path>]... [--write]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MIN_MATCH = 0.8
MIN_FOUND_SHARE = 0.5


def map_from_reference(doc: Dict[str, Any], stream: Any) -> Tuple[List[List[float]], float]:
    """(anchors, share of checkpoints found) for one decoded stream against one reference."""
    from podcast_scraper.audio_sync import fingerprint, reference

    spec = reference.prepared(fingerprint.spectra(stream))
    pairs: List[List[float]] = []
    found = 0
    checkpoints = doc.get("checkpoints") or []
    for cp in checkpoints:
        window = reference.decode_window(cp["frames"]) * (reference.SCALE / float(doc["scale"]))
        at, match = reference.locate_everywhere(window, spec)
        if match < MIN_MATCH:
            continue
        found += 1
        offset = round(at - float(cp["t"]), 2)
        if not pairs or abs(pairs[-1][1] - offset) > 0.05:
            pairs.append([round(float(cp["t"]), 2), offset])
    if pairs:
        pairs[0][0] = 0.0
    return pairs, found / max(1, len(checkpoints))


def main(argv: Optional[list] = None) -> int:
    """``build-maps``: a server time map per drifted episode, from its reference."""
    from podcast_scraper.audio_sync import fingerprint
    from podcast_scraper.audio_sync.drift import served_url
    from podcast_scraper.audio_sync.references import reference_relpath
    from podcast_scraper.server.app_audio_drift import DRIFT_FILE, MAPS_FILE
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_episode_search import episode_segments_relpath

    ap = argparse.ArgumentParser(
        prog="podcast_scraper.audio_sync build-maps",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", type=Path, required=True)
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
        url = (doc.get("content") or {}).get("media_url")
        seg_rel = episode_segments_relpath(root, row)
        if not isinstance(url, str) or not url.startswith(("http://", "https://")) or not seg_rel:
            tally("skip_no_url_or_transcript", rel)
            continue
        try:
            ref = json.loads((root / reference_relpath(seg_rel)).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tally("skip_no_reference", rel)
            continue
        try:
            stream = fingerprint.decode(served_url(url), timeout_s=900)
        except Exception as exc:  # noqa: BLE001 — one host down must not stop the run
            tally("skip_stream_unreadable", rel, type(exc).__name__)
            continue
        anchors, share = map_from_reference(ref, stream)
        if share < MIN_FOUND_SHARE or not anchors:
            tally("skip_poor_match", rel, f"checkpoints found {share:.0%}")
            continue
        maps[rel] = {
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": "reference",
            "found_share": round(share, 3),
            "anchors": anchors,
        }
        tally("mapped", rel, f"{len(anchors)} offsets, checkpoints found {share:.0%}")
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
