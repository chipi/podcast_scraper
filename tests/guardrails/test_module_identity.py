"""One module, one object: `sys.modules[...]` and the package attribute must agree.

THE BLINDNESS THIS GUARDS. A test that does ``del sys.modules["podcast_scraper.X"]`` and lets
the module be re-imported creates a SECOND module object, and Python rebinds the parent
package's ``X`` attribute to it. Restoring only the ``sys.modules`` entry leaves the two
pointing at different objects for the remainder of the process — and then:

    monkeypatch.setattr("podcast_scraper.X.fn", fake)   resolves via sys.modules  -> object A
    `from ...X import fn` inside the code under test    reads the package attr    -> object B

Different objects, so the patch is invisible, the REAL function runs, and the assertion fails
in a file with no connection to the one that caused it. Nothing raises; the patch simply does
nothing. That is the same shape as everything in ``test_guard_capability.py``: the check never
observed what it claimed to.

It is not hypothetical. ``test_workflow_helpers.py`` restored only ``sys.modules`` and cost
``TestHFSeq2SeqLoadBranches`` two tests in a full-suite run while both passed in isolation.
Order-dependent, so it stayed hidden until an unrelated new test file shifted collection order.

WHY A GUARD AND NOT JUST THE FIX. The fix is three lines in one file. This is the part that
survives: the next person to reach for ``del sys.modules[...]`` gets told here, in a test whose
failure message explains the trap, instead of by a stranger's test failing next month.
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = pytest.mark.unit


def _divergences() -> list[str]:
    """Submodules where ``sys.modules`` and the parent package attribute disagree."""
    out: list[str] = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("podcast_scraper.") or module is None:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        held = getattr(parent, attr, None)
        # A parent that never got the attribute is normal (submodule imported but not yet
        # bound, or a non-module attribute of the same name). Only DISAGREEMENT is the bug.
        if isinstance(held, types.ModuleType) and held is not module:
            out.append(f"{name}: sys.modules={id(module)} but {parent_name}.{attr}={id(held)}")
    return out


def test_no_module_is_present_as_two_objects() -> None:
    """Every imported ``podcast_scraper`` submodule is one object, seen one way."""
    divergences = _divergences()
    assert not divergences, (
        "A module exists twice — sys.modules and the parent package disagree:\n  "
        + "\n  ".join(divergences)
        + "\n\nA test almost certainly did `del sys.modules[...]` and restored only the "
        "sys.modules entry. Restore the package attribute too:\n"
        "    sys.modules['podcast_scraper.X'] = original\n"
        "    podcast_scraper.X = original\n"
        "Until then, monkeypatching 'podcast_scraper.X.fn' silently does nothing to code that "
        "reaches it through a relative import."
    )


def test_the_guard_actually_fires() -> None:
    """Prove the check can SEE the divergence, by building one and putting it back.

    A guard that has only ever been green is indistinguishable from no guard (see
    MULTILINGUAL_ARC.md §3.1). This constructs the exact state the real bug produced and
    asserts the detector reports it — then restores, and asserts it goes quiet again.
    """
    import podcast_scraper
    import podcast_scraper.cache  # noqa: F401 - a submodule known to be imported

    assert not _divergences(), "the process is already polluted; fix that before reading this"

    real = sys.modules["podcast_scraper.cache"]
    impostor = types.ModuleType("podcast_scraper.cache")
    try:
        podcast_scraper.cache = impostor  # type: ignore[attr-defined]
        found = _divergences()
        assert any(
            "podcast_scraper.cache" in d for d in found
        ), f"the guard did not see a divergence it was handed: {found}"
    finally:
        podcast_scraper.cache = real  # type: ignore[attr-defined]

    assert not _divergences(), "restoring the attribute did not clear the divergence"
