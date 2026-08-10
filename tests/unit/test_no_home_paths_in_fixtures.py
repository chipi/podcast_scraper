"""Guard: committed fixtures / eval artifacts must not bake an absolute home path.

A macOS/Linux home path (e.g. ``/Users/<username>/…``) leaks the operator's OS username into the
public repo. Generators used to write the absolute working dir into corpus manifests
(``path`` / ``generated_from``) and the ``USER`` env into ``created_by`` (fixed 2026-08-10). This
test fails if any re-appear, catching a re-leak from ANY generator at PR time.
"""

from __future__ import annotations

import re
import subprocess

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.critical_path]

# Dirs where generated artifacts live + are committed.
_SCAN_DIRS = (
    "tests/fixtures",
    "data/eval/baselines",
    "data/eval/references",
    "data/perf",
    "autoresearch",
)
# A per-user home path, allowing the CI/runner + our placeholder usernames.
_HOME_RE = re.compile(r"/(?:Users|home)/(?!runner\b|operator\b|user\b)[A-Za-z0-9_.-]+/")


def test_no_absolute_home_paths_in_committed_fixtures() -> None:
    files = subprocess.check_output(["git", "ls-files", "--", *_SCAN_DIRS], text=True).splitlines()
    offenders = []
    for f in files:
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if _HOME_RE.search(text):
            offenders.append(f)
    assert not offenders, (
        "absolute home paths (OS-username leak) in committed fixtures/data — sanitize the "
        f"generator to write relative/placeholder paths: {offenders[:20]}"
    )
