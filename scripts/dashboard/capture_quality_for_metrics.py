#!/usr/bin/env python3
"""
Emit reports/docstrings.json, reports/vulture.json, and reports/codespell.txt for generate_metrics.py.

Interrogate 1.7+ no longer supports --output-format json; vulture 2.x no longer supports --json.
CI used invalid flags plus `cmd > f || echo fallback > f`, which overwrote good output when tools
exited non-zero. This script runs the tools and writes the JSON/text shapes generate_metrics expects.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_interrogate_coverage_percent(text: str) -> float:
    """Parse interrogate tabular / verbose output for overall coverage %."""
    m = re.search(r"actual:\s*([0-9.]+)\s*%", text)
    if m:
        return float(m.group(1))
    m = re.search(
        r"\|\s*TOTAL\s*\|[^|\n]*\|[^|\n]*\|[^|\n]*\|\s*([0-9.]+)\s*%",
        text,
    )
    if m:
        return float(m.group(1))
    return 0.0


def vulture_stdout_to_json_list(text: str) -> list:
    """Build a JSON list (generate_metrics counts len) from vulture text lines."""
    items: list = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Typical: path/to/file.py:42: unused import 'x' (90% confidence)
        if re.match(r"^[^\s].+\.py:\d+:", line):
            items.append({"line": line})
    return items


def run_interrogate(repo_root: Path, reports_dir: Path) -> None:
    out_txt = reports_dir / "interrogate_out.txt"
    reports_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "interrogate",
            "src/podcast_scraper/",
            "-o",
            str(out_txt),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    chunks: list[str] = []
    if out_txt.exists():
        chunks.append(out_txt.read_text(encoding="utf-8", errors="replace"))
    chunks.append(proc.stdout or "")
    chunks.append(proc.stderr or "")
    text = "\n".join(chunks)
    pct = parse_interrogate_coverage_percent(text)
    (reports_dir / "docstrings.json").write_text(
        json.dumps({"coverage_percent": pct}),
        encoding="utf-8",
    )
    print(f"docstrings.json: coverage_percent={pct} (interrogate exit {proc.returncode})")


def run_vulture(repo_root: Path, reports_dir: Path, min_confidence: int) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vulture",
            "src/podcast_scraper/",
            ".vulture_whitelist.py",
            f"--min-confidence={min_confidence}",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    items = vulture_stdout_to_json_list(text)
    (reports_dir / "vulture.json").write_text(json.dumps(items), encoding="utf-8")
    print(f"vulture.json: {len(items)} finding(s) (exit {proc.returncode})")


def run_codespell(repo_root: Path, reports_dir: Path) -> None:
    if shutil.which("codespell"):
        cmd: list[str] = [
            "codespell",
            "src/",
            "docs/",
            "--skip=*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "codespell",
            "src/",
            "docs/",
            "--skip=*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl",
        ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        (reports_dir / "codespell.txt").write_text("", encoding="utf-8")
        print("codespell not installed; wrote empty codespell.txt")
        return
    text = (proc.stdout or "") + (proc.stderr or "")
    (reports_dir / "codespell.txt").write_text(text, encoding="utf-8")
    print(f"codespell.txt: {len(text.splitlines())} line(s) (exit {proc.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for artefact files (default: reports)",
    )
    parser.add_argument(
        "--vulture-min-confidence",
        type=int,
        default=80,
        help="Vulture --min-confidence (nightly often uses 60)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of scripts/)",
    )
    args = parser.parse_args()
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    reports_dir = args.reports_dir
    if not reports_dir.is_absolute():
        reports_dir = repo_root / reports_dir

    run_interrogate(repo_root, reports_dir)
    run_vulture(repo_root, reports_dir, args.vulture_min_confidence)
    run_codespell(repo_root, reports_dir)


if __name__ == "__main__":
    main()
