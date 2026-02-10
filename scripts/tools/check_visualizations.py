#!/usr/bin/env python3
"""
Check that architecture diagrams are up-to-date with source code.

Exits with code 1 if any diagram is older than the source files it depends on.
Used by make check-visualizations and CI to fail PRs when diagrams are stale.

Usage:
    python scripts/tools/check_visualizations.py
"""

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    docs_arch = root / "docs" / "architecture"

    # Diagram -> list of source paths it depends on
    diagram_sources = {
        "dependency-graph.svg": [root / "src" / "podcast_scraper"],
        "dependency-graph-simple.svg": [root / "src" / "podcast_scraper"],
        "workflow-call-graph.svg": [
            root / "src" / "podcast_scraper" / "workflow" / "orchestration.py"
        ],
        "orchestration-flow.svg": [
            root / "src" / "podcast_scraper" / "workflow" / "orchestration.py"
        ],
        "service-flow.svg": [root / "src" / "podcast_scraper" / "service.py"],
    }

    stale = []
    for diagram_name, source_paths in diagram_sources.items():
        diagram_path = docs_arch / diagram_name
        if not diagram_path.exists():
            continue
        diagram_mtime = diagram_path.stat().st_mtime
        src_times = []
        for sp in source_paths:
            if sp.is_dir():
                for py in sp.rglob("*.py"):
                    if "__pycache__" not in str(py):
                        src_times.append(py.stat().st_mtime)
            elif sp.exists():
                src_times.append(sp.stat().st_mtime)
        newest_src = max(src_times) if src_times else 0.0
        if newest_src > diagram_mtime:
            stale.append((diagram_name, newest_src, diagram_mtime))

    if not stale:
        print("✓ Architecture diagrams are up to date")
        return 0

    print(
        "ERROR: Architecture diagrams are stale (source code is newer than diagrams):",
        file=sys.stderr,
    )
    for name, _src_t, _diag_t in stale:
        print(f"  - docs/architecture/{name}", file=sys.stderr)
    print("Run: make visualize", file=sys.stderr)
    print("Then commit updated docs/architecture/*.svg", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
