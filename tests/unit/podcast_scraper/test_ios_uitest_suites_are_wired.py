"""Every iOS XCUITest suite must be reachable from a make target.

`NativeCapabilityTests.swift` was written on 2026-09-16 — dictation, the native share sheet, push
permission, avatar upload — and no Makefile target ever referenced it. It had never run.

That is worse than having no test. A suite sitting on disk reads as coverage: it is what someone
greps for before asking "is this tested?", and the answer it gives is yes. On 2026-09-18 four native
features shipped broken (two exports delivered via `<a download>`, two via `window.open`, none of
which does anything in WKWebView) and the agent then asserted that "nothing in the suite runs in a
WKWebView" — an absence claim that was false, made while a share-sheet test sat unwired three
directories away.

So this pins the wiring, not the content. It cannot tell you a suite is *good*; it can tell you a
suite is *reachable*, which is the property that silently rotted.

Deliberately in the Python tier rather than Swift: it is a repo-structure invariant, it must run on
every CI box including the ones with no Xcode, and Swift tests cannot see the Makefile.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
UITESTS_DIR = REPO_ROOT / "web/learning-player/ios/uitests/Sources/UITests"
MAKEFILE = REPO_ROOT / "Makefile"

#: Swift files that are shared scaffolding, not suites — no `XCTestCase` of their own to run.
_SUPPORT_FILES = {"Journey.swift", "AppSession.swift"}

#: Suites that are deliberately not wired to a target, each with the reason.
#:
#: Empty on purpose. Add an entry only with a reason that survives being read aloud — "it is slow"
#: is not one (give it its own nightly target instead), and neither is "it is flaky" (fix it or
#: delete it; a quarantined suite that nobody runs is the exact thing this file exists to catch).
_UNWIRED_BY_DESIGN: dict[str, str] = {}


def _suite_names() -> list[str]:
    """Every XCUITest class name that has at least one `func test…`."""
    names: list[str] = []
    for path in sorted(UITESTS_DIR.glob("*.swift")):
        if path.name in _SUPPORT_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if not re.search(r"\bfunc\s+test\w*\s*\(", text):
            continue
        for match in re.finditer(r"\bclass\s+(\w+)\s*:\s*XCTestCase\b", text):
            names.append(match.group(1))
    return names


@pytest.mark.unit
def test_uitests_dir_is_found() -> None:
    """Guard the guard: a moved directory must fail loudly, not silently pass with zero suites.

    A glob that matches nothing is the failure mode this whole file is about — it would report
    "every suite is wired" having checked none of them.
    """
    assert UITESTS_DIR.is_dir(), f"iOS UITests directory not found at {UITESTS_DIR}"
    assert (
        _suite_names()
    ), f"no XCTestCase suites found under {UITESTS_DIR} — has the layout changed?"


@pytest.mark.unit
def test_every_ios_uitest_suite_is_reachable_from_a_make_target() -> None:
    """A suite with no `-only-testing:` reference is a suite that has never run."""
    makefile = MAKEFILE.read_text(encoding="utf-8")
    referenced = set(re.findall(r"-only-testing:\w+/(\w+)", makefile))

    orphaned = [
        name for name in _suite_names() if name not in referenced and name not in _UNWIRED_BY_DESIGN
    ]

    assert not orphaned, (
        "These XCUITest suites exist but no Makefile target runs them, so they have never "
        f"executed: {sorted(orphaned)}.\n\n"
        "An unreachable suite is worse than a missing one — it reads as coverage to anyone who "
        "greps for it, which is how a native share-sheet test sat unrun while four native "
        "features shipped broken (2026-09-18).\n\n"
        "Add a `-only-testing:OfflineSpikeUITests/<Suite>` target, or record it in "
        "_UNWIRED_BY_DESIGN with a reason."
    )


@pytest.mark.unit
def test_no_make_target_references_a_suite_that_does_not_exist() -> None:
    """The inverse: a target pointing at a deleted or renamed suite passes by running nothing.

    `xcodebuild -only-testing:` against an unknown identifier does not fail — it runs an empty set
    and reports success, so a renamed suite turns a green target into a no-op.
    """
    makefile = MAKEFILE.read_text(encoding="utf-8")
    referenced = set(re.findall(r"-only-testing:\w+/(\w+)", makefile))
    existing = set(_suite_names())

    dangling = sorted(referenced - existing)

    assert not dangling, (
        f"These Makefile targets run a suite that does not exist: {dangling}.\n"
        "xcodebuild does not fail on an unknown -only-testing identifier — it runs nothing and "
        "reports success, so the target is green while testing zero code."
    )
