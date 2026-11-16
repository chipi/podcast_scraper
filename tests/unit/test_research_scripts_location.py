"""Research scripts belong in the private eval repository, not here (E04 IP extraction plan,
item 3).

The eval move (#2134, 2026-09-23) took research tradecraft out of this repository, but nothing
stopped new research scripts from landing here: 42 of 54 under ``scripts/measure``,
``scripts/eval`` and ``scripts/audit`` were added after it. This test refuses any file in those
directories that is not on the list below, so a new one goes to the eval repository instead.

What stays is public for good; the scripts that waited for E04's move went to the eval repository
with it. Shrink the list; never grow it without a reason written beside the entry.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
GUARDED = ("scripts/measure", "scripts/eval", "scripts/audit")

#: Public for good: the corpus change method's tooling (docs/guides/CORPUS_CHANGE_METHOD.md),
#: which public's upgrade framework runs, and the API memory measurement.
PUBLIC = {
    "scripts/measure/api_memory.py",
    "scripts/measure/build_outcome.py",
    "scripts/measure/corpus_change_gate.py",
    "scripts/measure/plan_diff_ro.py",
    "scripts/measure/review_verdicts_to_tsv.py",
    "scripts/measure/write_digest_ro.py",
    "scripts/eval/score/README.md",
}


def _guarded_files() -> set[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout: the tracked files cannot be listed")
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", *GUARDED], capture_output=True, text=True, check=True
    ).stdout.split()
    return {f for f in out if not f.endswith("__init__.py")}


def test_no_new_research_script_lands_here() -> None:
    extra = sorted(_guarded_files() - PUBLIC)
    assert not extra, (
        "research scripts belong in the private eval repository (E04 IP extraction plan, item 3); "
        "move these there, or add them to this test's list with the reason:\n" + "\n".join(extra)
    )


def test_the_list_names_only_files_that_exist() -> None:
    stale = sorted(PUBLIC - _guarded_files())
    assert not stale, "drop these from the list, they are gone:\n" + "\n".join(stale)
