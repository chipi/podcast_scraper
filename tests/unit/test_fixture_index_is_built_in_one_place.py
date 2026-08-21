"""The committed fixture's search index must be built by exactly one, locked, shared fixture.

``tests/fixtures/app-validation-corpus/v3/search/`` is gitignored: ``metadata.json`` and
``lance_index`` do not exist in a fresh checkout, and ``cli index-two-tier`` writes them into the
committed fixture tree at test time. Two integration modules each carried their own copy of that
build, and a third read the sidecar without declaring it needed one. Under xdist that is shared
mutable state with no ordering:

    2026-08-21 nightly, both at 69%:
      gw0  test_search_capability_against_fixture.py   (building the index)
      gw1  test_index_maps_tokens_to_episodes          (reading it)  -> {} -> FAILED

The same two landed on one worker in the PR run, builder first, and it passed. The code under test
was identical in both. Two writers were equally unserialised against each other.

So the build now lives once, in ``tests/integration/conftest.py``, behind a cross-process lock.
This test keeps it that way: a fourth module that copy-pastes the builder re-creates the race, and
the failure it produces looks like a product bug on whichever unlucky test reads first.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_TESTS = Path(__file__).resolve().parents[1]
_OWNER = _TESTS / "integration" / "conftest.py"

#: The CLI verb that writes an index.
_BUILD_CALL = re.compile(r'["\']index-two-tier["\']')
#: ...aimed at the COMMITTED fixture. Building into a tmp_path is fine and several tests do it —
#: `unit/search/test_index_two_tier_cli.py` and `e2e/test_upgrade_cli_e2e.py` exercise the verb
#: itself. Only a build that targets the shared committed tree is the race.
_COMMITTED_FIXTURE = re.compile(r"app-validation-corpus")


def _sources() -> list[Path]:
    return [p for p in _TESTS.rglob("*.py") if "__pycache__" not in p.parts]


def _targets_the_committed_fixture(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return bool(_BUILD_CALL.search(text) and _COMMITTED_FIXTURE.search(text))


def test_only_the_shared_fixture_builds_the_validation_index() -> None:
    offenders = sorted(
        str(p.relative_to(_TESTS))
        for p in _sources()
        if p != _OWNER and _targets_the_committed_fixture(p)
    )
    assert not offenders, (
        "these test files invoke `index-two-tier` themselves instead of requesting the "
        "`app_validation_search_index` fixture, which re-introduces the xdist build race: "
        f"{offenders}"
    )


def test_the_shared_fixture_actually_holds_a_lock() -> None:
    """A shared builder that does not serialise is the write/write half of the same bug."""
    src = _OWNER.read_text(encoding="utf-8")
    assert "def app_validation_search_index" in src
    assert "FileLock" in src, "the shared builder must serialise concurrent xdist workers"
    assert _BUILD_CALL.search(src), "the shared builder is where the build belongs"
