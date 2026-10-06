#!/usr/bin/env python3
"""Record every module-level English naming regex / word collection, as a reviewed golden file.

Operator rule (2026-10-05): the English naming path must not change. Moving English constants
into per-language maps is allowed; changing what they match is not. This script captures the
English values from a source tree and writes them as JSON, and
`tests/unit/podcast_scraper/test_english_patterns_are_mains.py` fails if the code drifts from
that file.

Regenerate ONLY when an English change has been asked for, from the tree that has it:

    git archive <ref> src | tar -x -C /tmp/ref
    python scripts/tools/capture_english_naming_patterns.py /tmp/ref/src \\
        tests/fixtures/naming/english_patterns.json --ref <ref>
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

MODULES = (
    "podcast_scraper.speaker_detectors.hosts",
    "podcast_scraper.speaker_detectors.resolution",
    "podcast_scraper.providers.ml.diarization.roster",
    "podcast_scraper.gi.speakers",
)


def encode(value: Any) -> Optional[Any]:
    """A JSON-comparable form of a regex / word collection / string, or None to skip it."""
    if isinstance(value, re.Pattern):
        return {"re": value.pattern, "flags": int(value.flags)}
    if isinstance(value, (set, frozenset)) and all(isinstance(x, str) for x in value):
        return {"set": sorted(value)}
    if isinstance(value, (tuple, list)) and value and all(isinstance(x, re.Pattern) for x in value):
        return {"patterns": [encode(x) for x in value]}
    if isinstance(value, (tuple, list)) and value and all(isinstance(x, str) for x in value):
        return {"strings": list(value)}
    if isinstance(value, str):
        return {"str": value}
    return None


def capture(src: Path) -> Dict[str, Dict[str, Any]]:
    for name in [k for k in sys.modules if k.startswith("podcast_scraper")]:
        del sys.modules[name]
    sys.path.insert(0, str(src))
    try:
        out: Dict[str, Dict[str, Any]] = {}
        for mod in MODULES:
            module = importlib.import_module(mod)
            rows = {}
            for name, value in sorted(vars(module).items()):
                # Skip modules, functions and classes. NOT "has __module__": a compiled regex has
                # one too ("re"), and that test silently dropped every pattern.
                if name.startswith("__") or callable(value) or isinstance(value, ModuleType):
                    continue
                enc = encode(value)
                if enc is not None:
                    rows[name] = enc
            out[mod] = rows
        return out
    finally:
        sys.path.remove(str(src))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="the src/ directory to capture from")
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", required=True, help="the git ref the src came from (recorded)")
    args = ap.parse_args()
    sha = subprocess.run(
        ["git", "rev-parse", args.ref], capture_output=True, text=True, check=True
    ).stdout.strip()
    doc = {"captured_from": args.ref, "sha": sha, "modules": capture(args.src)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{sum(len(v) for v in doc['modules'].values())} English values from {args.ref} @ {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
