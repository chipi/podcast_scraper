#!/usr/bin/env python3
"""Regenerate the registry-governed fields of every profile YAML from the model registry.

THE LOOP THIS EXISTS TO CLOSE
-----------------------------
An eval establishes a better value -> it goes in the REGISTRY (on a StageOption, with a
``research_ref`` naming the report that justified it) -> ``make profiles-materialize`` -> every
profile inherits it. That is the new default, everywhere, in one commit.

Before this script, the third step was "a human hand-copies the number into fourteen YAML files",
and that is exactly where the values went wrong:

  * ``gi_max_insights`` was **12** in the registry, **50** in the profiles and **20** in the
    ``Config`` default. Three doors, three answers, and production ran whichever one you came in
    through.
  * ``provider_chunked_gated_v3`` — the researched v3 tuning, temperature pin and all — was an
    ORPHAN: no preset pointed at it, so the measured configuration reached ZERO production profiles
    and the YAMLs hand-copied their way to something else.
  * ``gi_insight_temperature`` sat in the registry, unplumbed, while every scored bake-off arm
    sampled at 0.3 with a config that said 0.0.

A profile YAML is a downstream VIEW. It gets no vote on a registry-governed field. Hand-edit one and
``test_registry_is_the_source_of_truth`` fails — which is the point.

Edits are LINE-SURGICAL on purpose: the profiles are ~57% comments, and those comments carry the
reasoning. A full ``yaml.safe_dump`` round-trip would silently delete all of it, which is precisely
the kind of quiet loss this file exists to prevent.

USAGE
    python scripts/config/materialize_profiles.py            # rewrite the YAMLs in place
    python scripts/config/materialize_profiles.py --check    # CI: fail if any profile is stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.providers.ml.model_registry import (  # noqa: E402
    _PROFILE_PRESETS,
    REGISTRY_GOVERNED_FIELDS,
    resolve_profile_to_settings,
)

PROFILE_DIR = REPO / "config" / "profiles"

_BLOCK_HEADER = "# --- registry-materialized (make profiles-materialize) — do not hand-edit ---"


def governed_settings(name: str) -> Dict[str, Any]:
    """The registry's verdict for one profile, narrowed to the fields it OWNS.

    ``resolve_profile_to_settings`` also emits resolver-only keys (endpoints and such) that are not
    ``Config`` fields; writing those into a YAML would make it un-loadable.
    """
    resolved = resolve_profile_to_settings(name)
    return {k: resolved[k] for k in REGISTRY_GOVERNED_FIELDS if k in resolved}


def _fmt(value: Any) -> str:
    """Render a scalar the way PyYAML will read it back identically."""
    return yaml.safe_dump(value, default_flow_style=True).strip().rstrip("\n...").strip()


def _rewrite(text: str, want: Dict[str, Any], have: Dict[str, Any]) -> str:
    """Set every governed key to the registry's value, preserving comments and layout."""
    lines = text.splitlines()
    appended: List[str] = []

    for key, value in want.items():
        if have.get(key) == value:
            continue
        pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:.*$")
        for i, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                lines[i] = f"{m.group(1)}{key}: {_fmt(value)}"
                break
        else:
            appended.append(f"{key}: {_fmt(value)}")

    if appended:
        if _BLOCK_HEADER not in text:
            lines += ["", _BLOCK_HEADER]
        lines += appended
    return "\n".join(lines) + "\n"


def _stale_keys(name: str, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    want = governed_settings(name)
    return want, [k for k, v in want.items() if data.get(k) != v]


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize profiles from the model registry.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if any profile disagrees with the registry (for CI)",
    )
    args = ap.parse_args()

    stale: List[Tuple[str, List[str]]] = []
    checked = 0
    for path in sorted(PROFILE_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        # Only profiles naming a registry preset are governed. The rest are bespoke and are left
        # alone rather than silently rewritten into something they never claimed to be.
        if not isinstance(data, dict) or data.get("profile") not in _PROFILE_PRESETS:
            continue
        checked += 1
        want, changed = _stale_keys(str(data["profile"]), data)
        if not changed:
            continue
        stale.append((path.name, changed))
        if not args.check:
            path.write_text(_rewrite(text, want, data), encoding="utf-8")

    if not stale:
        print(f"All {checked} registry-governed profiles match the registry.")
        return 0

    verb = "STALE" if args.check else "Materialized"
    print(f"{verb}: {len(stale)} profile(s)\n")
    for name, keys in stale:
        print(f"  {name}")
        for k in keys:
            print(f"    - {k}")
    if args.check:
        print("\nRun `make profiles-materialize` to regenerate them from the registry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
