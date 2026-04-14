#!/usr/bin/env python3
"""Remove decorative Unicode emoji from markdown under docs/ (text-only prose).

Strips pictographs, dingbats, emoticons, regional-indicator pairs (flags), ZWJ
sequences, and VS16. Intended for English project docs; does not use extra deps.

Run from repo root::

    .venv/bin/python3 scripts/docs_strip_emoji.py

Then::

    make fix-md && make lint-markdown

Scope: ``docs/**/*.md`` (includes ``docs/wip/``), root ``README.md``, ``CLAUDE.md``,
``.ai-coding-guidelines.md``, ``.github/**/*.md``, ``.cursorrules``, and ``.cursor/rules/*.mdc``.
Does not modify Python or YAML.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# One range (or literal) per pattern avoids CodeQL ``redos`` / overlapping-class noise.
_EMOJI_PATTERNS = tuple(
    re.compile(p, flags=re.UNICODE)
    for p in (
        r"[\U0001f1e6-\U0001f1ff]+",  # regional indicators
        r"[\U0001f300-\U0001faff]+",  # symbols, pictographs, supplemental
        r"[\U00002600-\U000026ff]+",  # misc symbols (e.g. warning sign)
        r"[\U00002700-\U000027bf]+",  # dingbats
        r"[\U000023e9-\U000023fa]+",  # media / UI glyphs
        r"[\U000025fc-\U000025ff]+",
        r"[\U00002b50-\U00002b59]+",  # star variants
        r"\U0000200d+",  # ZWJ runs
        r"\U0000fe0f+",  # emoji variation selector
        r"\U000020e3+",  # combining enclosing keycap
    )
)


def transform(text: str) -> str:
    for pat in _EMOJI_PATTERNS:
        text = pat.sub("", text)
    # Headings: "##  Foo" / "###   Bar" after strip
    text = re.sub(r"^(#{1,6}\s) +", r"\1", text, flags=re.MULTILINE)
    # Blockquotes: "> ** Important:**" left a gap after pictograph removal
    text = re.sub(r"(^> ?)\*\*\s+([A-Za-z0-9`])", r"\1**\2", text, flags=re.MULTILINE)
    # Lines that are only whitespace after strip
    text = re.sub(r"^[ \t]+\n", "\n", text, flags=re.MULTILINE)
    # Table rows: removing emoji can leave "|  |" gaps (MD060)
    lines_out: list[str] = []
    for line in text.splitlines(True):
        if line.lstrip().startswith("|"):
            prev = None
            while prev != line:
                prev = line
                line = re.sub(r"\|\s{2,}\|", "| |", line)
        lines_out.append(line)
    return "".join(lines_out)


def iter_paths() -> list[Path]:
    out: list[Path] = []
    docs = ROOT / "docs"
    if docs.is_dir():
        out.extend(sorted(docs.rglob("*.md")))
    for name in (
        "README.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "WORKTREE.md",
        ".ai-coding-guidelines.md",
        ".ai-coding-guidelines-quick.md",
    ):
        p = ROOT / name
        if p.is_file():
            out.append(p)
    viewer = ROOT / "web" / "gi-kg-viewer"
    if viewer.is_dir():
        for p in viewer.rglob("*.md"):
            if "node_modules" in p.parts:
                continue
            out.append(p)
    gh = ROOT / ".github"
    if gh.is_dir():
        out.extend(sorted(gh.rglob("*.md")))
    rules = ROOT / ".cursor" / "rules"
    if rules.is_dir():
        out.extend(sorted(rules.glob("*.mdc")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main() -> int:
    changed = 0
    for path in iter_paths():
        if path.resolve() == SELF.resolve():
            continue
        raw = path.read_text(encoding="utf-8")
        new = transform(raw)
        if new != raw:
            path.write_text(new, encoding="utf-8", newline="\n")
            print(f"updated: {path.relative_to(ROOT)}")
            changed += 1
    cr = ROOT / ".cursorrules"
    if cr.is_file():
        raw = cr.read_text(encoding="utf-8")
        new = transform(raw)
        if new != raw:
            cr.write_text(new, encoding="utf-8", newline="\n")
            print(f"updated: {cr.relative_to(ROOT)}")
            changed += 1
    print(f"done. {changed} file(s) updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
