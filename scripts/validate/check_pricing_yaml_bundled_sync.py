#!/usr/bin/env python3
"""Fail when config and bundled pricing YAML diverge (#823)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML = REPO_ROOT / "config" / "pricing_assumptions.yaml"
BUNDLED_YAML = REPO_ROOT / "src" / "podcast_scraper" / "data" / "pricing_assumptions.yaml"


def main() -> int:
    if not CONFIG_YAML.is_file():
        print(f"missing {CONFIG_YAML}", file=sys.stderr)
        return 1
    if not BUNDLED_YAML.is_file():
        print(f"missing {BUNDLED_YAML}", file=sys.stderr)
        return 1
    config_text = CONFIG_YAML.read_text(encoding="utf-8")
    bundled_text = BUNDLED_YAML.read_text(encoding="utf-8")
    if config_text != bundled_text:
        print(
            "pricing_assumptions.yaml out of sync: "
            f"config ({CONFIG_YAML}) must match bundled ({BUNDLED_YAML})",
            file=sys.stderr,
        )
        return 1
    print("pricing_assumptions.yaml: config and bundled copies match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
