#!/usr/bin/env python3
"""Strip checkmark-style emoji from markdown and .mdc policy docs.

Removes green tick, red cross, clipboard emoji, and decorative tick (U+2713) from
prose, lists, headings, and tables. Preserves meaning with plain words where helpful.

Run from repo root::

    .venv/bin/python3 scripts/docs_strip_checkmarks.py

Then::

    make fix-md && make lint-markdown

Does not modify this script file, Python sources, YAML workflows, or data logs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()


def transform(text: str) -> str:
    # RFC/PRD milestone style: "**M1 — Foo** ✅ (done):" -> "**M1 — Foo** (done):"
    text = re.sub(r"\s*✅\s*\(done\)", " (done)", text)

    # Status / prerequisite lines
    text = re.sub(r"(\*\*Status\*\*:\s*)✅\s*", r"\1", text)
    text = re.sub(r"✅\s*Implemented", "Implemented", text)
    text = re.sub(r"\s*✅\s*Completed\b", " Completed", text)
    text = re.sub(r"\(✅\s*", "(", text)

    # Module table header (module-boundaries.mdc)
    text = text.replace("| ✅ CAN Do | ❌ CANNOT Do |", "| Allowed | Forbidden |")
    text = text.replace("✅ CAN Do", "Allowed")
    text = text.replace("❌ CANNOT Do", "Forbidden")

    # Headings: ### ✅ GOOD: / ### ❌ BAD:
    text = re.sub(
        r"^(#+\s*)✅\s*GOOD:\s*",
        r"\1Good: ",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(
        r"^(#+\s*)❌\s*BAD:\s*",
        r"\1Bad: ",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(r"^(#+\s*)✅\s*", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^(#+\s*)❌\s*", r"\1", text, flags=re.MULTILINE)

    # Numbered lists
    text = re.sub(r"^(\s*\d+)\.\s*✅\s+", r"\1. ", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*\d+)\.\s*❌\s+", r"\1. **Never:** ", text, flags=re.MULTILINE)

    # Bullets
    text = re.sub(r"^(\s*)-\s*✅\s+", r"\1- ", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*)-\s*❌\s+", r"\1- **Never:** ", text, flags=re.MULTILINE)

    # Table cells
    text = re.sub(r"\|\s*✅\s+PASS\s*\|", "| Pass |", text)
    text = re.sub(r"\|\s*✅\s+([^|\n]+)\|", r"| Yes — \1|", text)
    text = re.sub(r"\|\s*✅\s*\|", "| Yes |", text)
    text = re.sub(r"\|\s*✅\s*", "| Yes ", text)
    text = re.sub(r"\|\s*❌\s+([^|\n]+)\|", r"| No — \1|", text)
    text = re.sub(r"\|\s*❌\s*\|", "| No |", text)

    # Inline arrows in checklists
    text = re.sub(r"→\s*❌\s*", "→ **Wrong:** ", text)
    text = re.sub(r"→\s*✅\s*", "→ **OK:** ", text)

    # Code / comment patterns (module-boundaries examples)
    text = re.sub(r"(\s#)\s*❌\s*NO!\s*", r"\1 Wrong: ", text)
    text = re.sub(r"(\s#)\s*✅\s*", r"\1 ", text)
    text = re.sub(r"#\s*✅\s*GOOD\b", "# Good", text)
    text = re.sub(r"#\s*❌\s*BAD\b", "# Bad", text)

    # Import-line comments:  # ✅ Works
    text = re.sub(r"\s+#\s*✅\s*Works\b", "  # Works", text)
    text = re.sub(r"\s+#\s*✅\s*Also works\b", "  # Also works", text)
    text = re.sub(r"\s+#\s*✅\s*Use proper module", "  # Use proper module", text)
    text = re.sub(r"\s+#\s*✅\s*Validation only", "  # Validation only", text)
    text = re.sub(r"\s+#\s*✅\s*Structured result", "  # Structured result", text)
    text = re.sub(r"\s+#\s*✅\s*Provider-specific", "  # Provider-specific", text)

    # **✅ GOOD:** / **❌ BAD:** inline (bold blocks)
    text = re.sub(r"\*\*✅\s*GOOD:?\*\*", "**Good:**", text, flags=re.I)
    text = re.sub(r"\*\*❌\s*BAD:?\*\*", "**Bad:**", text, flags=re.I)
    text = re.sub(r"\*\*❌\s*DON'T:\*\*", "**Do not:**", text, flags=re.I)

    # Remaining known words adjacent to marks
    text = re.sub(r"✅\s*Good:\s*", "Good: ", text, flags=re.I)
    text = re.sub(r"❌\s*Bad:\s*", "Bad: ", text, flags=re.I)

    # Clipboard section icons in headings (common in .mdc)
    text = re.sub(r"^##\s*📋\s*", "## ", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s*📁\s*", "## ", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s*📚\s*", "## ", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s*📝\s*", "## ", text, flags=re.MULTILINE)
    text = re.sub(r"^###\s*🚨\s*", "### ", text, flags=re.MULTILINE)

    # Strip any remaining forbidden characters (last resort)
    for ch in ("✅", "❌", "📋", "✓"):
        text = text.replace(ch, "")

    # Light cleanup: "  **Never:**" from double space after strip
    text = re.sub(r"^(\s*-\s)\s{2,}", r"\1", text, flags=re.MULTILINE)
    return text


def iter_doc_paths() -> list[Path]:
    out: list[Path] = []
    docs = ROOT / "docs"
    if docs.is_dir():
        out.extend(sorted(docs.rglob("*.md")))
    gh = ROOT / ".github"
    if gh.is_dir():
        out.extend(sorted(gh.rglob("*.md")))
    rules = ROOT / ".cursor" / "rules"
    if rules.is_dir():
        out.extend(sorted(rules.glob("*.mdc")))
    for name in (
        "README.md",
        "CLAUDE.md",
        ".ai-coding-guidelines.md",
        ".ai-coding-guidelines-quick.md",
    ):
        p = ROOT / name
        if p.is_file():
            out.append(p)
    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp == SELF.resolve():
            continue
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main() -> int:
    changed = 0
    for path in iter_doc_paths():
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        new = transform(raw)
        if new != raw:
            path.write_text(new, encoding="utf-8", newline="\n")
            print(f"updated: {path.relative_to(ROOT)}")
            changed += 1
    print(f"done. {changed} file(s) updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
