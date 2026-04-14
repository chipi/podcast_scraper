#!/usr/bin/env python3
"""Bump package version in pyproject.toml and podcast_scraper.__init__.

Usage:
  python scripts/tools/bump_version.py X.Y.Z [--allow-dirty] [--force]

Refuses when:
  - VERSION is not semantic X.Y.Z
  - git working tree is dirty (unless --allow-dirty)
  - git tag vX.Y.Z already exists (unless --force)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_SEMVER = re.compile(r"^(?P<maj>0|[1-9]\d*)\.(?P<min>0|[1-9]\d*)\.(?P<pat>0|[1-9]\d*)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_porcelain(root: Path) -> str:
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        print("bump_version: git status failed — not a git repo?", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


def _tag_exists(root: Path, tag: str) -> bool:
    r = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.returncode == 0


def _write_pyproject_version(root: Path, version: str) -> None:
    path = root / "pyproject.toml"
    raw = path.read_text(encoding="utf-8")
    pattern = re.compile(r'^version\s*=\s*".*"', re.MULTILINE)
    new_raw, n = pattern.subn(f'version = "{version}"', raw, count=1)
    if n != 1:
        print("bump_version: could not find version = line in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    path.write_text(new_raw, encoding="utf-8")


def _write_init_version(root: Path, version: str) -> None:
    path = root / "src" / "podcast_scraper" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'^(__version__\s*=\s*)["\'][^"\']+["\']',
        rf'\1"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        print("bump_version: could not replace __version__ in __init__.py", file=sys.stderr)
        sys.exit(1)
    path.write_text(new_text, encoding="utf-8")


def bump(
    root: Path | None,
    version: str,
    *,
    allow_dirty: bool = False,
    force_tag: bool = False,
) -> None:
    """Update pyproject.toml and ``__init__.py`` version fields."""
    v = version.strip()
    if not _SEMVER.match(v):
        print("bump_version: VERSION must match X.Y.Z (semver numeric parts)", file=sys.stderr)
        raise SystemExit(1)

    base = root if root is not None else _repo_root()
    dirty = _git_porcelain(base)
    if dirty and not allow_dirty:
        print(
            "bump_version: working tree is dirty — commit/stash or pass --allow-dirty",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tag = f"v{v}"
    if _tag_exists(base, tag) and not force_tag:
        print(
            f"bump_version: tag {tag} already exists — choose a new version or pass --force",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _write_pyproject_version(base, v)
    _write_init_version(base, v)
    print(f"bump_version: set version to {v} in pyproject.toml and __init__.py")
    print("bump_version: next: make release-docs-prep, edit docs/releases/, make pre-release")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump project version in pyproject + __init__.py")
    parser.add_argument("version", help="Semantic version X.Y.Z (no v prefix)")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow non-empty git status (not recommended for releases)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow bump even if git tag vX.Y.Z already exists",
    )
    args = parser.parse_args()
    bump(
        None,
        args.version,
        allow_dirty=args.allow_dirty,
        force_tag=args.force,
    )


if __name__ == "__main__":
    main()
