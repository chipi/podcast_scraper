#!/usr/bin/env python3
"""Strip green checkmark emoji from PRD/RFC markdown; use plain text.

Run from repo root: .venv/bin/python3 scripts/docs_strip_checkmarks.py
Then: make fix-md && make lint-markdown
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def transform(text: str) -> str:
    # RFC-062 style: "**M1 — Foo** ✅ (done):" -> "**M1 — Foo** (done):"
    text = re.sub(r"\s*✅\s*\(done\)", " (done)", text)

    # Status lines: **Status**: ✅ Implemented -> **Status**: Implemented
    text = re.sub(r"(\*\*Status\*\*:\s*)✅\s*", r"\1", text)

    # "Implemented" may follow without space issues
    text = re.sub(r"✅\s*Implemented", "Implemented", text)

    # Prerequisite lines: "✅ Completed" -> "Completed"
    text = re.sub(r"\s*✅\s*Completed\b", " Completed", text)

    # Checklist: (✅ Complete) -> (Complete)
    text = re.sub(r"\(✅\s*", "(", text)

    # Bullets: "- ✅ " -> "- "
    text = re.sub(r"^(\s*)-\s*✅\s+", r"\1- ", text, flags=re.MULTILINE)

    # Numbered: "1. ✅ " -> "1. "
    text = re.sub(r"^(\s*\d+)\.\s*✅\s+", r"\1. ", text, flags=re.MULTILINE)

    # Headings: "### ✅ Phase" -> "### Phase"
    text = re.sub(r"^(#+\s*)✅\s+", r"\1", text, flags=re.MULTILINE)

    # Table: | ✅  PASS | -> | Pass |
    text = re.sub(r"\|\s*✅\s+PASS\s*\|", "| Pass |", text)

    # Table cells starting with checkmark: | ✅ something | -> | Yes — something |
    text = re.sub(r"\|\s*✅\s+([^|\n]+)\|", r"| Yes — \1|", text)

    # Lone checkmark in cell: | ✅ |
    text = re.sub(r"\|\s*✅\s*\|", "| Yes |", text)

    # Remaining | ✅ without following content on same segment — conservative
    text = re.sub(r"\|\s*✅\s*", "| Yes ", text)

    # Red X in tables: | ❌ -> | No —
    text = re.sub(r"\|\s*❌\s+([^|\n]+)\|", r"| No — \1|", text)
    text = re.sub(r"\|\s*❌\s*\|", "| No |", text)

    # Any remaining ✅ (e.g. mid-sentence)
    text = text.replace("✅", "")

    return text


def main() -> None:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "docs" / "prd").glob("*.md")))
    paths.extend(sorted((ROOT / "docs" / "rfc").glob("RFC*.md")))
    paths.append(ROOT / "docs" / "rfc" / "RFC_TEMPLATE.md")

    for path in paths:
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        new = transform(raw)
        if new != raw:
            path.write_text(new, encoding="utf-8", newline="\n")
            print(f"updated: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
