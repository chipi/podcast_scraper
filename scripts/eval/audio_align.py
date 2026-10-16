#!/usr/bin/env python3
"""Line up a served stream with the original we transcribed (RFC-129).

    .venv/bin/python scripts/eval/audio_align.py <original.mp3> <stream file or URL> [--json F]

Prints each stretch of the original's talk found in the stream: original seconds, and how far it
moved (positive: later in the stream). The original comes from the audio archive
(``docs/recipes/prod-audio-archive.md``), only where its provenance says it is byte-identical to
what we transcribed. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys

from podcast_scraper.audio_sync.fingerprint import align_files, time_map


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("original")
    ap.add_argument("stream", help="a file, or a URL ffmpeg can read")
    ap.add_argument("--json", help="write the stretches and the time map here")
    args = ap.parse_args()
    found = align_files(args.original, args.stream)
    for s in found:
        print(f"original {s.start:8.2f}-{s.end:8.2f}s  offset {s.offset:+8.2f}s")
    if not found:
        print("no talk of the original found in the stream")
    if args.json:
        doc = {
            "stretches": [{"start": s.start, "end": s.end, "offset": s.offset} for s in found],
            "audio_time_map": time_map(found),
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
