#!/usr/bin/env python3
"""Emit a silent, genuinely decodable MP3 — no encoder, no ffmpeg, no network.

Why this exists
---------------
The app-validation corpus shipped a 146-byte ``data:audio/mpeg`` URI as every episode's
``content.media_url``. It is an ID3 header with **no audio frames**, so no browser can decode it:
the player flipped to ``audioError`` and rendered the error panel instead of the transport. The e2e
suite worked around that by routing a synthetic WAV over the audio-source response
(``routeLoadableAudio``), which meant 8 specs depended on a mock in a suite whose whole contract is
"no mocks — bootstrap the real backend from real fixtures" (#1618).

The fix belongs in the fixture, not the test. This builds real MPEG-2.5 Layer III frames so the
committed corpus is directly playable.

Why hand-built frames
---------------------
Neither ffmpeg nor any Python MP3 encoder is available in this repo's environments (the API image
carries numpy/scipy only, and scipy writes WAV). Uncompressed WAV is not an option at this scale:
36 episodes × 30+ seconds of 8 kHz 8-bit mono is ~23 MB of base64 in committed JSON. MPEG-2.5 at
8 kbps is 1 KB per second, so the same corpus costs ~2 MB.

A Layer III frame whose side-info and main-data are zero decodes as silence — ``part2_3_length`` is
zero, so there is nothing to Huffman-decode and the granule renders as digital silence. That is the
standard construction, but "standard" is not evidence: `e2e/fixture-audio.spec.ts` loads the result
in real Chromium and asserts it reports a duration and advances, so a decoder that disagrees fails
the suite rather than silently reintroducing the error panel.

Which encoding, and how that was decided
----------------------------------------
Empirically, not by reading a spec. Five candidate headers were generated and loaded in real
Chromium; it accepted two and rejected three:

    mpeg1_44100_32k   OK        mpeg1_44100_64k   MEDIA_ERR_SRC_NOT_SUPPORTED
    mpeg2_22050_8k    OK        mpeg2_22050_32k   MEDIA_ERR_SRC_NOT_SUPPORTED
                                mpeg25_8000_8k    MEDIA_ERR_SRC_NOT_SUPPORTED

MPEG-2 / 22.05 kHz / 8 kbps is the cheapest of the two that work: 1 KB per second, so 45 s costs
~45 KB raw and ~60 KB as base64 in the committed JSON. The MPEG-1 alternative is 4× that. (My first
attempt was MPEG-2.5 at 8 kHz — the smallest on paper, and one of the three Chromium refuses. That
is the entire reason this module documents a measurement instead of a derivation.)

Header (4 bytes), MPEG-2 / Layer III / no CRC / 8 kbps / 22050 Hz / mono::

    0xFF  1111 1111    sync
    0xF3  1111 10 01 1 sync | version 10 (MPEG-2) | layer 01 (III) | protection 1 (no CRC)
    0x10  0001 00 0 0  bitrate 0001 (8 kbps) | rate 00 (22050 Hz) | pad 0 | private 0
    0xC0  11 00 0 0 00 mono | mode ext | copyright | original | emphasis

Frame = 576 samples / 8 × 8000 / 22050 = 26 bytes, i.e. 26.12 ms of audio.
"""

from __future__ import annotations

import argparse
import base64
import sys

FRAME_HEADER = bytes((0xFF, 0xF3, 0x10, 0xC0))
FRAME_BYTES = 26
"""Layer III MPEG-2: int(576 / 8 × 8000 / 22050) = 26 bytes per frame."""

SECONDS_PER_FRAME = 576 / 22050  # 0.02612 s


def silent_mp3(seconds: float) -> bytes:
    """Return `seconds` of decodable digital silence as MPEG-2 Layer III."""
    frames = max(1, round(seconds / SECONDS_PER_FRAME))
    body = FRAME_HEADER + bytes(FRAME_BYTES - len(FRAME_HEADER))
    return body * frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--out", help="write raw MP3 bytes here (otherwise print a data URI)")
    args = ap.parse_args()

    data = silent_mp3(args.seconds)
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(data)
        print(f"{args.out}: {len(data)} bytes, ~{len(data) / FRAME_BYTES * SECONDS_PER_FRAME:.1f}s")
    else:
        print("data:audio/mpeg;base64," + base64.b64encode(data).decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
