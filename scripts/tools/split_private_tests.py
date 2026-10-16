#!/usr/bin/env python3
"""Run this private repo's tests inside a checkout of the public podcast_scraper repo (ADR-166).

``split_copy.py`` writes this file into every generated repo as ``scripts/run_tests.py``.

The tests keep the public repo's layout and read its committed fixtures (the corpora, goldens and
schemas under ``tests/fixtures``, ``docs/api``, ``config``), so they run from a throwaway git
worktree of the public checkout with this repo's ``tests``, ``scripts``, ``infra`` and ``docs``
laid over it. While the public repo still holds its own copy of the moved code (arc 1), its split
tools prune that copy first, so the code under test is this repo's and not the public duplicate.
The public checkout itself is never written to; the worktree is removed at the end.

Usage (from the private repo's root)::

    make test                                   # unit + integration, not ml_models / slow
    python scripts/run_tests.py -- -k digest    # anything after -- goes to pytest

``PODCAST_SCRAPER_ROOT`` names the public checkout; the default is two directories up, where the
``apps/<repo>`` mount puts it. The worktree is of its HEAD, so uncommitted public changes are not
seen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OVERLAY = ("tests", "scripts", "infra", "docs")


def public_root() -> Path:
    root = Path(os.environ.get("PODCAST_SCRAPER_ROOT") or HERE.parents[1]).resolve()
    if not (root / "src" / "podcast_scraper").is_dir() or not (root / ".git").exists():
        sys.exit(f"not a podcast_scraper checkout: {root} (set PODCAST_SCRAPER_ROOT)")
    return root


def private_repos(public: Path) -> list[Path]:
    """This repo first, then its siblings under the public checkout's apps/ (Player imports
    Common)."""
    siblings = sorted(p for p in (public / "apps").glob("*/src") if p.parent != HERE)
    return [HERE, *(p.parent for p in siblings)]


def entry_point_shims(repos: list[Path], dest: Path) -> list[Path]:
    """A dist-info per private repo declaring its extensions as its pyproject has them now.

    Written even when the repo is installed: an editable install records the entry points of
    the pyproject it was installed from, and ``importlib.metadata`` keeps the first distribution
    of a name on the path, which is this one (PYTHONPATH precedes site-packages)."""
    made = []
    for repo in repos:
        text = (repo / "pyproject.toml").read_text(encoding="utf-8")
        name = text.split('name = "', 1)[1].split('"', 1)[0]
        header = '[project.entry-points."podcast_scraper.extensions"]'
        if header not in text:
            continue
        eps = text.split(header, 1)[1].split("\n[", 1)[0].replace('"', "").strip()
        info = dest / f"{name.replace('-', '_')}-0.0.0.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.0\n")
        (info / "entry_points.txt").write_text(f"[podcast_scraper.extensions]\n{eps}\n")
        made.append(dest)
    return sorted(set(made))


def prune_moved_code(worktree: Path) -> None:
    """Arc 1: the public tree still has the moved code; delete it the way the split probe does."""
    tools = worktree / "scripts" / "tools"
    if not (tools / "split_probe.py").is_file():
        return
    sys.path.insert(0, str(tools))
    import split_copy  # type: ignore[import-not-found]
    import split_probe  # type: ignore[import-not-found]
    import yaml

    manifest = yaml.safe_load(split_copy.MANIFEST.read_text())
    mapping, _ = split_copy.build_mapping(manifest, split_copy.tracked_files())
    print(f"pruned {split_probe.prune(worktree, manifest, mapping)} public paths", flush=True)


def overlay(worktree: Path) -> list[str]:
    targets = []
    for top in OVERLAY:
        for f in (HERE / top).rglob("*") if (HERE / top).is_dir() else []:
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            dest = worktree / top / f.relative_to(HERE / top)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            if top == "tests" and f.name.startswith("test_") and f.suffix == ".py":
                targets.append(str(dest))
    return sorted(targets)


def main(argv: list[str]) -> int:
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    public = public_root()
    repos = private_repos(public)
    with tempfile.TemporaryDirectory(prefix=f"{HERE.name}-tests-") as tmp:
        worktree = Path(tmp) / "public"
        subprocess.run(
            ["git", "worktree", "add", "--quiet", "--detach", str(worktree), "HEAD"],
            cwd=public,
            check=True,
        )
        try:
            prune_moved_code(worktree)
            targets = overlay(worktree)
            paths = [worktree / "src", *(r / "src" for r in repos)]
            paths += entry_point_shims(repos, Path(tmp) / "dist")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))}
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                str(worktree / "pyproject.toml"),
                "--rootdir",
                str(worktree),
                "-p",
                "no:cacheprovider",
                *targets,
                *extra,
            ]
            return subprocess.run(cmd, cwd=worktree, env=env).returncode
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)], cwd=public, check=False
            )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
