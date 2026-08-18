#!/usr/bin/env python3
"""Enforce the repo's documentation structure: required READMEs, real content, live links.

Why this exists
---------------
Documentation here does not fail loudly. A directory quietly loses its README, a file is renamed
out from under a link, a README decays into a stub — and nothing breaks, so nobody notices until
someone (increasingly, an agent starting a fresh session) follows a pointer, finds nothing, and
concludes the thing does not exist.

That is not hypothetical. On 2026-08-13 an agent searched one directory for fixture audio, found
none, and hand-built an MP3 encoder rather than looking one level up where 46 real files sat. The
answer was already written in three documents it never opened, and the app's own README had **six
dead spec links** — PRD-035, PRD-038, PRD-039, RFC-099, UXS-011, PLATFORM_API — in the file a new
contributor opens first. A stale pointer is worse than a missing one, because it is trusted.

Checks
------
  1. Required READMEs exist
     REQUIRED_READMES lists the tree roots where someone lands and needs orientation. The list is
     explicit on purpose: adding a root is a deliberate decision, not something that sprawls.

  2. Required READMEs are not stubs
     A placeholder README is worse than none — it looks answered. Each must carry real prose and
     at least one outward pointer.

  3. Every relative link in every markdown file resolves
     Repo-wide, not just the structural docs, because the cost of a dead link is the same
     wherever it lives.

What this deliberately does NOT check
-------------------------------------
Whether the prose is still TRUE. Nothing can. That is why these documents are written as contracts,
reasons and pointers (slow to change) rather than restating code (fast to change) — see the READMEs
themselves. Style and formatting are already covered by ``make lint-markdown``.

Usage::

    python scripts/tools/check_doc_structure.py
    python scripts/tools/check_doc_structure.py --list   # show what is being enforced
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tree roots that must carry a README. Each is somewhere a contributor or agent lands and needs
# orientation: what is this, which file do I start from, what is related.
REQUIRED_READMES: tuple[str, ...] = (
    "tests",
    "tests/e2e",
    "tests/fixtures",
    "tests/fixtures/audio",
    "tests/fixtures/app-validation-corpus",
    "tests/stack-test",
    "web",
    "web/learning-player",
    "web/learning-player/e2e",
    "web/learning-player/src/stores",
    "web/learning-player/src/composables",
    "web/gi-kg-viewer",
    "web/gi-kg-viewer/e2e",
    "docker/mock-feeds",
)

# A README that exists but says nothing is a placeholder. These floors are deliberately low — the
# check is against emptiness, not a word count.
MIN_PROSE_LINES = 8
MIN_LINKS = 1

SKIP_DIRS = {"node_modules", ".venv", ".git", "dist", ".build", "htmlcov", ".pytest_cache"}


def _is_vendored(parts: tuple[str, ...]) -> bool:
    """Is this path inside someone else's package rather than our repo?

    The literal ``.venv`` in SKIP_DIRS missed ``.venv-dev``, so this gate reported 16 broken
    links — all 16 of them inside third-party site-packages (a vendored PHP parser, deepgram,
    nltk) and none in our own docs. A gate that can only fail on files we do not own and cannot
    edit is a gate that is always red, and an always-red gate is one nobody reads. Matching on
    the ``.venv`` PREFIX and on ``site-packages`` anywhere in the path covers every virtualenv
    naming convention instead of enumerating them one at a time.
    """
    return any(part.startswith(".venv") or part == "site-packages" for part in parts)


MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# Anything with a URI scheme (http:, mailto:, chrome:, vscode:) is not a repo path.
URI_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.I)


def markdown_files() -> list[Path]:
    out: list[Path] = []
    for path in REPO_ROOT.rglob("*.md"):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in SKIP_DIRS for part in parts) or _is_vendored(parts):
            continue
        out.append(path)
    return sorted(out)


def is_repo_path(target: str) -> bool:
    """Is this link target meant to be a file in this repo?

    Excludes URI schemes, anchors, and bare placeholders like ``[text](url)`` that appear in
    markdown style guides as examples — those have no separator and no extension.
    """
    if not target or target.startswith("#") or URI_SCHEME.match(target):
        return False
    path = target.split("#")[0]
    if not path:
        return False
    return "/" in path or "." in path


def check_required_readmes() -> list[str]:
    problems = []
    for rel in REQUIRED_READMES:
        readme = REPO_ROOT / rel / "README.md"
        if not readme.exists():
            problems.append(
                f"{rel}/README.md is missing. This tree is a place people land; it needs a short "
                f"purpose and pointers to what is related (see tests/fixtures/audio/README.md for "
                f"the shape). If this root no longer warrants one, remove it from REQUIRED_READMES "
                f"in {Path(__file__).name} — deliberately."
            )
    return problems


def check_not_stubs() -> list[str]:
    problems = []
    for rel in REQUIRED_READMES:
        readme = REPO_ROOT / rel / "README.md"
        if not readme.exists():
            continue  # already reported
        text = readme.read_text(encoding="utf-8")
        prose = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "|", "```", "-", ">"))
        ]
        links = [m for m in MD_LINK.findall(text)]
        if len(prose) < MIN_PROSE_LINES:
            problems.append(
                f"{rel}/README.md looks like a stub ({len(prose)} prose lines, want "
                f"≥{MIN_PROSE_LINES}). A placeholder is worse than nothing — it reads as answered."
            )
        if len(links) < MIN_LINKS:
            problems.append(
                f"{rel}/README.md has no links. Its job is to route the reader onward, not to "
                f"explain everything in place."
            )
    return problems


def check_links() -> list[str]:
    problems = []
    for path in markdown_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for target in MD_LINK.findall(text):
            if not is_repo_path(target):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                rel = path.relative_to(REPO_ROOT)
                problems.append(
                    f"{rel} → {target} does not exist. Fix the path, or drop the link and say "
                    f"plainly that the target is gone; a pointer to nothing sends the reader to "
                    f"the wrong conclusion."
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print what is enforced and exit")
    args = parser.parse_args()

    if args.list:
        print(f"Required READMEs ({len(REQUIRED_READMES)}):")
        for rel in REQUIRED_READMES:
            mark = "ok " if (REPO_ROOT / rel / "README.md").exists() else "MISSING"
            print(f"  {mark:8} {rel}/README.md")
        print(f"\nLink check covers {len(markdown_files())} markdown files.")
        return 0

    problems = check_required_readmes() + check_not_stubs() + check_links()

    if problems:
        print(f"Documentation structure: {len(problems)} problem(s)\n")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nThese are cheap to fix and expensive to ignore: every one of them ends with a "
            "reader concluding something is absent when it is not."
        )
        return 1

    print(
        f"Documentation structure OK "
        f"({len(REQUIRED_READMES)} required READMEs, {len(markdown_files())} markdown files linked "
        f"correctly)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
