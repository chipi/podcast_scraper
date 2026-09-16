#!/usr/bin/env python3
"""Pull the UI-test screenshots out of an ``.xcresult`` under their logical names.

XCTest stores `XCTAttachment` screenshots inside the result bundle with opaque UUID filenames; the
human name we chose in the test (``t01-home``, ``05-add-to-collection``) survives only in the
manifest. This exports them and renames, so the output directory is directly reviewable and can be
stitched by ``contact_sheet.py``.

Xcode also writes a lot of automatic attachments into the same bundle — "UI Snapshot", "Synthesized
Event", "Screen Recording", "App UI hierarchy", "Debug description". Those are debugging aids, not
screens, and they outnumber the real frames several to one, so they are filtered out. The filter is
by EXCLUSION rather than by matching a numeric prefix: an earlier version required the name to start
with two digits and silently dropped every frame from the tour suite, which prefixes its frames
``t01``… (2026-09-16).

    python scripts/tools/export_xcresult_shots.py --xcresult <path> --out /tmp/shots
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Xcode's own attachments. Matched case-insensitively against the start of the attachment name.
NOISE_PREFIXES = (
    "ui snapshot",
    "synthesized event",
    "screen recording",
    "app ui hierarchy",
    "debug description",
    "kxctattachment",
)


def _walk(node: object):
    """Yield every attachment record in the manifest, whatever shape it nests them in."""
    if isinstance(node, dict):
        if "exportedFileName" in node and "suggestedHumanReadableName" in node:
            yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _is_screen(name: str) -> bool:
    lowered = name.strip().lower()
    return bool(lowered) and not lowered.startswith(NOISE_PREFIXES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xcresult", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.xcresult.exists():
        sys.exit(f"FAIL: no such .xcresult: {args.xcresult}")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    subprocess.run(
        [
            "xcrun",
            "xcresulttool",
            "export",
            "attachments",
            "--path",
            str(args.xcresult),
            "--output-path",
            str(args.out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    manifest = args.out / "manifest.json"
    if not manifest.is_file():
        sys.exit("FAIL: xcresulttool wrote no manifest.json — were there any attachments?")

    named = args.out / "named"
    named.mkdir(exist_ok=True)
    seen: set[str] = set()
    for record in _walk(json.loads(manifest.read_text())):
        raw = record["suggestedHumanReadableName"]
        if not _is_screen(raw):
            continue
        # XCTest appends "_<index>_<uuid>" when a name repeats across a run; the logical name is
        # everything before the first underscore. Keep the FIRST occurrence — a retried test would
        # otherwise overwrite the frame with a later, possibly failed, attempt.
        stem = raw.split("_")[0]
        if stem in seen:
            continue
        source = args.out / record["exportedFileName"]
        if not source.exists():
            continue
        shutil.copy(source, named / f"{stem}.png")
        seen.add(stem)

    if not seen:
        sys.exit("FAIL: no screenshots matched — every attachment looked like Xcode debug output")
    print(f"✓ {len(seen)} screenshots → {named}")
    for name in sorted(seen):
        print(f"    {name}")


if __name__ == "__main__":
    main()
