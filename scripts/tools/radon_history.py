#!/usr/bin/env python3
"""Per-commit code-complexity trend over git history (radon 6) — #1014.

Replaces the retired wily-based ``make complexity-track``. wily hard-pinned ``radon<5.2``
(#424), so it was dropped when radon moved to 6.x (#1015 / #1014); its one real feature —
walking the last N revisions and charting complexity per commit — is restored here using
radon directly.

For each of the last N commits on the current branch it extracts the package source at that
revision via ``git archive`` (no working-tree mutation, no checkout), runs radon cyclomatic
complexity (cc) + maintainability index (mi), and prints the trend oldest->newest. ``--output``
optionally writes the series as JSON.

Usage::

    python scripts/tools/radon_history.py [--revisions 50] [--package src/podcast_scraper] \
        [--output reports/complexity-history.json]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# Use the radon from the same environment as this interpreter (the venv), not whatever is on
# PATH — make targets invoke this via ``$(PYTHON)`` and PATH may not include the venv bin.
RADON = str(Path(sys.executable).parent / "radon")


def _commits(n: int) -> list[tuple[str, str, str]]:
    """Last *n* commits as ``(sha, iso-date, subject)``, newest first."""
    out = subprocess.run(
        ["git", "log", f"-{n}", "--format=%H%x09%cI%x09%s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        if line.strip():
            sha, date, subject = line.split("\t", 2)
            rows.append((sha, date, subject))
    return rows


def _extract(sha: str, package: str, dest: Path) -> Path | None:
    """Extract *package* at *sha* into *dest* via ``git archive``; return the path or None."""
    try:
        archive = subprocess.run(
            ["git", "archive", "--format=tar", sha, package],
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None  # path may not exist at that revision
    if not archive:
        return None
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest)  # noqa: S202 - our own git archive, trusted content
    p = dest / package
    return p if p.exists() else None


def _avg_complexity(path: Path) -> float | None:
    out = subprocess.run([RADON, "cc", str(path), "--json"], capture_output=True, text=True).stdout
    try:
        data = json.loads(out)
    except ValueError:
        return None
    comps = [
        b["complexity"]
        for blocks in data.values()
        if isinstance(blocks, list)
        for b in blocks
        if isinstance(b, dict) and "complexity" in b
    ]
    return round(statistics.mean(comps), 3) if comps else None


def _avg_mi(path: Path) -> float | None:
    out = subprocess.run([RADON, "mi", str(path), "--json"], capture_output=True, text=True).stdout
    try:
        data = json.loads(out)
    except ValueError:
        return None
    mis = [v["mi"] for v in data.values() if isinstance(v, dict) and "mi" in v]
    return round(statistics.mean(mis), 2) if mis else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-commit radon complexity/maintainability trend.")
    ap.add_argument("--revisions", type=int, default=50, help="recent commits to walk (default 50)")
    ap.add_argument("--package", default="src/podcast_scraper", help="package path to analyze")
    ap.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    rows: list[dict] = []
    print(f"{'date':<20} {'cc(avg)':>8} {'mi(avg)':>8}  commit")
    print("-" * 72)
    for sha, date, subject in reversed(_commits(args.revisions)):  # oldest -> newest
        with tempfile.TemporaryDirectory() as td:
            src = _extract(sha, args.package, Path(td))
            if src is None:
                continue
            cc = _avg_complexity(src)
            mi = _avg_mi(src)
        rows.append(
            {"sha": sha, "date": date, "subject": subject, "avg_complexity": cc, "avg_mi": mi}
        )
        cc_s = f"{cc:.3f}" if cc is not None else "-"
        mi_s = f"{mi:.2f}" if mi is not None else "-"
        print(f"{date[:19]:<20} {cc_s:>8} {mi_s:>8}  {sha[:8]} {subject[:50]}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {len(rows)} points -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
