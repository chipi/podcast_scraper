#!/usr/bin/env python3
"""
Create a draft release notes file for the current version if it does not exist.

Reads version from pyproject.toml and creates docs/releases/RELEASE_vX.Y.Z.md
with a minimal template. Used by make release-docs-prep.

Usage:
    python scripts/tools/create_release_notes_draft.py
"""

import re
import sys
from pathlib import Path


def get_version_from_pyproject(root: Path) -> str:
    """Read version from pyproject.toml [project] section."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return ""
    text = pyproject.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else ""


def _check_compatibility_matrix(root: Path, version: str) -> bool:
    compat = root / "docs" / "COMPATIBILITY.md"
    if not compat.is_file():
        print(
            "create_release_notes_draft: missing docs/COMPATIBILITY.md — "
            "add a row for this release (#797)",
            file=sys.stderr,
        )
        return False
    body = compat.read_text(encoding="utf-8")
    if version not in body:
        print(
            f"create_release_notes_draft: docs/COMPATIBILITY.md must mention version {version!r}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    version = get_version_from_pyproject(root)
    if not version:
        print("Could not read version from pyproject.toml", file=sys.stderr)
        return 1

    if not _check_compatibility_matrix(root, version):
        return 1

    releases_dir = root / "docs" / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    filename = f"RELEASE_v{version}.md"
    filepath = releases_dir / filename

    if filepath.exists():
        print(f"Release notes already exist: docs/releases/{filename}")
        return 0

    # Minimal template
    from datetime import datetime

    month_year = datetime.utcnow().strftime("%B %Y")
    template = f"""# Release v{version}

**Release Date:** {month_year}
**Type:** Minor Release
**Last Updated:** {datetime.utcnow().strftime("%Y-%m-%d")}

## Summary

(Describe the main changes in this release.)

## Key Features

(Add sections for new features, improvements, and fixes.)

## Upgrade Notes

(Any breaking changes or migration steps.)

## Full Changelog

(Link: https://github.com/chipi/podcast_scraper/compare/vPREVIOUS...v{version}
 — replace vPREVIOUS with previous tag)
"""
    filepath.write_text(template, encoding="utf-8")
    print(f"Created draft: docs/releases/{filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
