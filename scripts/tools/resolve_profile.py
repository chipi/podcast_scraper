"""Print the resolved settings dict for a registry profile preset.

Ops-debug tool for the #907 registry-driven config flow. Given a profile
name (e.g. ``cloud_with_dgx_primary``), prints what
``resolve_profile_to_settings()`` returns — the same dict that ``Config``
layers under the YAML / explicit data when constructing a config.

Useful for:
- Confirming a registry edit produced the routing you intended before
  rebuilding the pipeline image.
- Debugging "what would the runtime pick up if I set ``profile: X``?"
  without touching the actual pipeline.
- Sanity-checking that ``DGX_TAILNET_HOST`` env-var threading produces
  the expected endpoint URL on a given operator machine.

Usage::

    python scripts/tools/resolve_profile.py cloud_with_dgx_primary
    DGX_TAILNET_HOST=my-dgx.tailnet.ts.net \\
        python scripts/tools/resolve_profile.py local_dgx_balanced
    python scripts/tools/resolve_profile.py --list

Or via the make target::

    make profile-resolve PROFILE=cloud_with_dgx_primary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from podcast_scraper.providers.ml.model_registry import (
    _PROFILE_PRESETS,
    resolve_profile_to_settings,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "profile",
        nargs="?",
        help="Registry profile preset name (e.g. cloud_with_dgx_primary).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available preset names and exit.",
    )
    parser.add_argument(
        "--dgx-tailnet-host",
        default=None,
        help=(
            "Override DGX Tailscale MagicDNS host (otherwise reads "
            "DGX_TAILNET_HOST env var; falls back to a sentinel hostname)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="table",
        help="Output format. ``table`` is human-readable; ``json`` is machine-parseable.",
    )
    return parser


def _render_table(settings: Dict[str, Any]) -> str:
    width = max((len(k) for k in settings), default=0)
    lines = []
    # Group metadata (underscore-prefixed) at the bottom for legibility.
    payload = {k: v for k, v in settings.items() if not k.startswith("_")}
    metadata = {k: v for k, v in settings.items() if k.startswith("_")}
    for key in sorted(payload):
        lines.append(f"  {key:<{width}}  {payload[key]!r}")
    if metadata:
        lines.append("")
        lines.append("  metadata:")
        for key in sorted(metadata):
            lines.append(f"    {key:<{width}}  {metadata[key]!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list:
        for name in sorted(_PROFILE_PRESETS):
            print(name)
        return 0
    if not args.profile:
        print("error: profile name required (or pass --list)", file=sys.stderr)
        return 2
    if args.profile not in _PROFILE_PRESETS:
        print(
            f"error: unknown profile {args.profile!r}. " f"Known: {sorted(_PROFILE_PRESETS)}",
            file=sys.stderr,
        )
        return 2
    host = args.dgx_tailnet_host or os.environ.get("DGX_TAILNET_HOST")
    settings = resolve_profile_to_settings(args.profile, dgx_tailnet_host=host)
    if args.format == "json":
        print(json.dumps(settings, indent=2, sort_keys=True))
    else:
        print(f"Profile: {args.profile}")
        print(f"DGX host: {host or '(unset — sentinel fallback)'}")
        print()
        print(_render_table(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
