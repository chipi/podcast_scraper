"""Every Android instrumented suite must be reachable from a make target.

The sibling of ``test_ios_uitest_suites_are_wired.py``, written at the same time as the Android
tier itself (#2139) rather than after the same thing had gone wrong twice.

The iOS version exists because ``NativeCapabilityTests.swift`` — dictation, the native share sheet,
push permission, avatar upload — sat on disk for days with no Makefile target referencing it. It
had never run. That is worse than having no test: a suite on disk reads as coverage, it is what
someone greps for before asking "is this tested?", and the answer it gives is yes. Four native
features then shipped broken while a share-sheet test sat unwired three directories away.

Android starts with the guard already in place. It pins the WIRING, not the content: it cannot tell
you a suite is good, only that it is reachable, which is the property that silently rotted.

Deliberately in the Python tier: it is a repo-structure invariant, it must run on every CI box
including the ones with no Android SDK, and an instrumented test cannot see the Makefile.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ANDROID_TESTS_DIR = (
    REPO_ROOT / "web/learning-player/android/app/src/androidTest/java/app/closelistening/player"
)
MAKEFILE = REPO_ROOT / "Makefile"

#: Java files that are shared scaffolding, not suites — no `@Test` of their own to run.
_SUPPORT_FILES = {"Journey.java", "AppSession.java", "UITestCase.java"}

#: Suites deliberately not wired to a target, each with the reason.
#:
#: Empty on purpose. Add an entry only with a reason that survives being read aloud — "it is slow"
#: is not one (give it its own nightly target instead), and neither is "it is flaky" (fix it or
#: delete it; a quarantined suite nobody runs is the exact thing this file exists to catch).
_UNWIRED_BY_DESIGN: dict[str, str] = {
    "PersonalisationTests": (
        "PARKED with the operator 2026-09-25, alongside ServerDegradedTests. Measured by running "
        "each platform's suite ALONE: iOS fails both too — PersonalisationTests.swift line 39 "
        "('no Play control and nothing playing') and line 71 ('no Topics tab on Profile'). So this "
        "is shared app behaviour, not an Android defect. Note the two tiers fail at DIFFERENT "
        "points in the same tests — iOS fails earlier, before reaching the assertion Android "
        "reaches — so the ports diverge as well. UNLIKE ServerDegradedTests this suite IS inside "
        "the iOS gate (`test-app-ios-journey-ui`), so parking the Android half does not make "
        "`test-ios` green; that tier is red on this too."
    ),
    "ServerDegradedTests": (
        "PARKED, matching iOS, which excludes `test-app-ios-server-degraded` from every gate "
        "(Makefile: 'ALSO EXCLUDED'). Measured 2026-09-25: the iOS test FAILS on the same two "
        "assertions as the Android port — the degraded banner and cache-survived, Swift lines 58 "
        "and 72. So this is shared app behaviour that regressed while the suite sat outside all "
        "gates, NOT an Android defect. I claimed it was one before checking iOS, and it was not. "
        "Parked deliberately with the operator until the app behaviour is addressed; both "
        "platforms come back together."
    ),
}


def _suite_names() -> list[str]:
    """Every instrumented class name that has at least one `@Test`."""
    names: list[str] = []
    for path in sorted(ANDROID_TESTS_DIR.glob("*.java")):
        if path.name in _SUPPORT_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if "@Test" not in text:
            continue
        # Matches the BASE class too. Every suite extends `UITestCase` for the per-suite account and
        # known-state setup, so a regex pinned to a runner annotation alone would go stale the
        # moment the base class grew — the same trap the iOS guard documents for `XCTestCase`.
        for match in re.finditer(r"\bclass\s+(\w+)\s+extends\s+UITestCase\b", text):
            names.append(match.group(1))
        runner = r"@RunWith\(AndroidJUnit4\.class\)\s*(?:/\*.*?\*/\s*)?public\s+class\s+(\w+)\s*\{"
        for match in re.finditer(runner, text, re.S):
            if match.group(1) not in names:
                names.append(match.group(1))
    return names


@pytest.mark.unit
def test_android_uitests_dir_is_found() -> None:
    """Guard the guard: a moved directory must fail loudly, not silently pass with zero suites.

    A glob that matches nothing is the failure mode this whole file is about — it would report
    "every suite is wired" having checked none of them.
    """
    assert ANDROID_TESTS_DIR.is_dir(), f"Android test directory not found at {ANDROID_TESTS_DIR}"
    assert (
        _suite_names()
    ), f"no instrumented suites found under {ANDROID_TESTS_DIR} — has the layout changed?"


@pytest.mark.unit
def test_every_android_suite_is_reachable_from_a_make_target() -> None:
    """A suite no `SUITE=` line names is a suite that has never run."""
    makefile = MAKEFILE.read_text(encoding="utf-8")
    referenced = set(re.findall(r"SUITE=(\w+)", makefile))

    orphaned = [
        name for name in _suite_names() if name not in referenced and name not in _UNWIRED_BY_DESIGN
    ]

    assert not orphaned, (
        "These Android instrumented suites exist but no Makefile target runs them, so they have "
        f"never executed: {sorted(orphaned)}.\n\n"
        "An unreachable suite is worse than a missing one — it reads as coverage to anyone who "
        "greps for it, which is how an iOS share-sheet test sat unrun while four native features "
        "shipped broken (2026-09-18).\n\n"
        "Add a `$(MAKE) android-suite SUITE=<Suite>` step to `test-android`, or record it in "
        "_UNWIRED_BY_DESIGN with a reason."
    )


@pytest.mark.unit
def test_android_device_tier_is_in_ci_ui_full() -> None:
    """The tier must be attached to a gate someone actually runs.

    Wiring a suite to a target is only half of it. `test-ios` itself was absent from every gate,
    which is how `DownloadThroughUITests.swift` stopped COMPILING at 71fc75965 and nobody noticed
    for weeks. A tier nothing runs is a tier nobody notices breaking.
    """
    makefile = MAKEFILE.read_text(encoding="utf-8")
    lines = makefile.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ci-ui-full:"))
    # A make recipe ends at the next line beginning a new target — NOT at the next blank line, which
    # is what an earlier version of this assertion used. Recipes here are one long continued command
    # containing blank `echo ""` lines, so that cut the body off after two lines and the guard
    # failed on a Makefile that was correct.
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if re.match(r"^[A-Za-z0-9_.\-/]+\s*:(?!=)", lines[i])
        ),
        len(lines),
    )
    body = "\n".join(lines[start:end])
    assert "$(MAKE) test-android" in body, (
        "`test-android` is not invoked by `ci-ui-full`. The Android device tier would then run "
        "only when someone remembered to run it by hand, which is the state the iOS tier was in "
        "when it silently stopped compiling."
    )
