"""#1058 chunk 4 — fixture-regen-stability gate.

Asserts that re-running ``build_fixture.py`` produces byte-identical
output to what's checked in. Without this gate a casual edit to the
fixture JSON (manually, or via a refactor that touched the script)
could silently drift the contract — and the only signal would be
``test_multi_show_fixture.py`` collapsing on a shape we lost.

Runs the script into a tmp dir, then compares byte-by-byte against
the checked-in tree. Skipped if the fixture tree or script are
missing.
"""

from __future__ import annotations

import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "connectivity-multi-show"
)
_SCRIPT = _FIXTURE_DIR / "build_fixture.py"


def _compare_trees(a: Path, b: Path) -> list[str]:
    """Recursive byte-compare. Returns list of paths that differ; empty
    list means trees match exactly."""
    diffs: list[str] = []
    cmp = filecmp.dircmp(a, b)
    diffs.extend(f"only-in-{a.name}: {x}" for x in cmp.left_only)
    diffs.extend(f"only-in-{b.name}: {x}" for x in cmp.right_only)
    diffs.extend(f"differ: {x}" for x in cmp.diff_files)
    for subdir in cmp.common_dirs:
        diffs.extend(_compare_trees(a / subdir, b / subdir))
    return diffs


def test_build_fixture_is_byte_stable(tmp_path: Path) -> None:
    """Re-running ``build_fixture.py`` against a fresh feeds/ dir
    produces the same JSON we've checked in. If this test fails,
    EITHER:

    1. The script was edited and the fixture is stale — run it
       against the real fixture dir (``.venv/bin/python
       tests/fixtures/connectivity-multi-show/build_fixture.py``)
       and commit the diff.
    2. The script's JSON-emit is non-deterministic — sort_keys=True
       + a fixed publish_date table should keep it deterministic;
       if it drifted, fix the script.
    """
    if not _SCRIPT.is_file():
        pytest.skip("build_fixture.py missing")
    if not _FIXTURE_DIR.is_dir():
        pytest.skip("checked-in fixture missing")

    sandbox = tmp_path / "fixture-sandbox"
    sandbox.mkdir()
    # Copy ONLY the script — _FEEDS resolves to <sandbox>/feeds
    # because Path(__file__).resolve().parent is the script's dir.
    shutil.copy(_SCRIPT, sandbox / "build_fixture.py")

    result = subprocess.run(
        [sys.executable, str(sandbox / "build_fixture.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"build_fixture.py failed: {result.stderr}"

    sandbox_feeds = sandbox / "feeds"
    checked_in_feeds = _FIXTURE_DIR / "feeds"
    assert sandbox_feeds.is_dir(), "script did not emit feeds/"

    diffs = _compare_trees(checked_in_feeds, sandbox_feeds)
    assert not diffs, "fixture drift detected — re-run build_fixture.py and commit:\n" + "\n".join(
        diffs[:10]
    )
