"""Which episodes' served audio no longer matches the audio we transcribed? (#2359, option C)

For every episode with a ``content.media_url``: the length the host serves NOW (``ffprobe`` on the
URL: a header read, plus a range request where the host needs one) against the length we
transcribed. That reference is ``source_audio_duration_s`` from the episode's ``.asr.json`` when
ingest recorded it (since d749f53b4), else the end of its last transcript segment.

An episode has drifted when:
- the transcribed length is known and the served one differs by more than ``--known-tolerance``
  seconds (default 5); or
- only the transcript's end is known and the stream runs more than ``--end-tolerance`` (default
  30) seconds past it, or ends more than 5 s before it. A plain trailing outro is shorter than 30 s
  (measured on 402 prod episodes, 2026-10-15).

Read-only on the hosts and on the corpus. Without ``--write`` it only reports. With ``--write`` it
writes ``<corpus>/audio_timing_drift.json`` (atomically), which the api reads to keep drifted
episodes out of short listens and Learn days (``server/app_audio_drift.py``). Undo: delete the file.

    python -m podcast_scraper.audio_sync drift-check --corpus <corpus root> [--write] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

DRIFT_FILE = "audio_timing_drift.json"


def _served_seconds(url: str, timeout: int) -> Optional[float]:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
        return float(out.strip().splitlines()[0])
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return None


def _reference(root: Path, row: Any) -> tuple[Optional[float], str]:
    """(seconds, kind): the recorded source length, else the last transcript segment's end."""
    from podcast_scraper.server.app_episode_search import _segment_dicts, episode_segments_relpath

    rel = episode_segments_relpath(root, row)
    if rel is None:
        return None, "none"
    seg_path = root / rel
    base = str(seg_path)[: -len(".segments.json")].removesuffix(".adfree")
    try:
        asr = json.loads(Path(base + ".asr.json").read_text(encoding="utf-8"))
        prov = asr.get("provenance", asr) if isinstance(asr, dict) else {}
        seconds = prov.get("source_audio_duration_s") if isinstance(prov, dict) else None
        if isinstance(seconds, (int, float)) and seconds > 0:
            return float(seconds), "transcribed"
    except (OSError, ValueError):
        pass
    segs = _segment_dicts(seg_path)
    ends = [float(s.get("end") or 0.0) for s in segs]
    return (max(ends), "transcript_end") if ends else (None, "none")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="podcast_scraper.audio_sync drift-check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--write", action="store_true", help=f"write <corpus>/{DRIFT_FILE}")
    ap.add_argument("--limit", type=int, default=0, help="at most this many episodes (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=45, help="seconds per ffprobe")
    ap.add_argument("--known-tolerance", type=float, default=5.0)
    ap.add_argument("--end-tolerance", type=float, default=30.0)
    args = ap.parse_args(argv)

    from podcast_scraper.server.app_catalog_cache import cached_catalog

    root = args.corpus
    rows = []
    for row in cached_catalog(root):
        try:
            doc = json.loads((root / row.metadata_relative_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        url = ((doc.get("content") or {}) if isinstance(doc, dict) else {}).get("media_url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            rows.append((row, url))
    if args.limit:
        rows = rows[: args.limit]

    def check(item: tuple[Any, str]) -> Optional[Dict[str, Any]]:
        row, url = item
        ref, kind = _reference(root, row)
        if ref is None:
            return None
        served = _served_seconds(url, args.timeout)
        if served is None:
            return None
        diff = served - ref
        drifted = (
            abs(diff) > args.known_tolerance
            if kind == "transcribed"
            else (diff > args.end_tolerance or diff < -5.0)
        )
        return {
            "path": row.metadata_relative_path,
            "host": urllib.parse.urlparse(url).netloc,
            "served_s": round(served, 1),
            "ref_s": round(ref, 1),
            "ref_kind": kind,
            "drifted": drifted,
        }

    with ThreadPoolExecutor(args.workers) as ex:
        results = [r for r in ex.map(check, rows) if r]
    drifted = {
        r["path"]: {k: r[k] for k in ("host", "served_s", "ref_s", "ref_kind")}
        for r in results
        if r["drifted"]
    }
    print(
        f"{len(rows)} episodes with audio · {len(results)} measured · {len(drifted)} drifted "
        f"· {len(rows) - len(results)} unreadable or without a reference"
    )
    by_host: Dict[str, int] = {}
    for v in drifted.values():
        by_host[v["host"]] = by_host.get(v["host"], 0) + 1
    for host, n in sorted(by_host.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n:4}  {host}")
    if not args.write:
        print("dry run: nothing written (add --write)")
        return 0
    out = root / DRIFT_FILE
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "episodes": drifted},
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, out)
    print(f"wrote {out}")
    return 0
