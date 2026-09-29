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
#: UNPARKED 2026-09-25, both of them, after the iOS side was actually diagnosed.
#:
#: They were parked on the reasoning that iOS failed the same assertions, so it must be shared app
#: behaviour rather than an Android defect. The first half of that was true and the conclusion was
#: wrong in BOTH cases, for different reasons:
#:
#:   PersonalisationTests — a REAL product bug, and now fixed. `InterestsPicker.save()` PUT the
#:     interests and updated no store, so Home kept prompting after they were chosen from Profile.
#:     Shared Vue code, so the fix is platform-agnostic. iOS test10 passes on device.
#:     (The lines cited in the old note, 39 and 71, were a different failure entirely: `test-ios`
#:     destroyed its own api at phase 2 and never restored it, so the app was signed out.)
#:
#:   ServerDegradedTests — a TEST defect, not the app. The drill rotated `APP_SESSION_SECRET`
#:     instead of removing it, and the server only returns 503 ("cannot authenticate anyone") when
#:     the secret is ABSENT; with one present it returns 401, which correctly signs the user out.
#:     Measured: secret present -> /api/app/me 401, secret absent -> 503. With the real incident
#:     reproduced, the iOS suite passes — the app degrades exactly as designed.
#:
#: Both now run in `test-android`; `test-android-server-degraded` sequences the second.
_UNWIRED_BY_DESIGN: dict[str, str] = {}


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
def test_no_android_test_is_quietly_ignored() -> None:
    """A wired suite whose tests are `@Ignore`d runs less than it appears to.

    `android-suite` rejects `OK (0 tests)`, which catches a class resolving to nothing at all. It
    cannot catch the softer version: fourteen tests with thirteen `@Ignore`d still prints
    `OK (1 test)` and passes, while the coverage everyone believes in is switched off.

    Same family as the guard above — that one catches a suite no target runs, this one catches a
    test no suite runs. Both are "reads as coverage, is not", which is this file's whole subject.

    There are none today, so this is a floor rather than a cleanup.
    """
    ignored: list[str] = []
    for path in sorted(ANDROID_TESTS_DIR.glob("*.java")):
        # Comments stripped first: these files quote annotations while explaining them, and a guard
        # that fires on prose is the fake check this suite exists to catch.
        code = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", " ", path.read_text(encoding="utf-8"))
        if re.search(r"^\s*@Ignore\b", code, re.M):
            ignored.append(path.name)

    assert not ignored, (
        f"These Android suites contain `@Ignore`d tests: {sorted(ignored)}.\n\n"
        "An ignored test still lets its suite report OK, so the tier keeps claiming coverage it no "
        "longer has — the same shape as a suite no target runs, one level down.\n\n"
        "Fix it or delete it. If it genuinely must be skipped, say why in a comment beside the "
        "annotation and list the file here deliberately, so the subtraction is visible."
    )


@pytest.mark.unit
def test_the_ui_sign_in_flow_has_exactly_one_caller() -> None:
    """`AppSession.signIn` drives the real OAuth flow, and only the smoke test may use it.

    Suites sign in through `ensureSignedIn`, which delivers the OAuth callback as an intent — no
    Custom Tab, no dev picker, ~2 seconds, deterministic. `signIn` is the other thing: it drives the
    genuine flow a person takes, six sequential races ending in a consent screen owned by
    `com.android.chrome`, measured at 2 failures in 9 runs.

    Both need to exist. What must not happen is the second leaking back into the first. It already
    did once: `ensureSignedIn` used to fall back to `signIn` when the callback path
    failed, defended
    as insurance — and when the callback broke, the tier did not report a broken callback. It ran
    `signIn` against an already-signed-in app, typed the identity into Home's search
    box, and failed
    four steps later as "sign-in did not complete". The fallback converted a precise failure into a
    confusing one, which is the same argument `Journey.originPort()` makes against defaulting.

    So the UI flow is exercised exactly once per tier, by one test that names it, and
    this pins that. A second caller is not necessarily wrong — but it is a decision,
    and it should be made here.
    """
    callers: list[str] = []
    for path in sorted(ANDROID_TESTS_DIR.glob("*.java")):
        if path.name == "AppSession.java":
            continue  # its own internals may call it
        code = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", " ", path.read_text(encoding="utf-8"))
        for match in re.finditer(r"AppSession\.signIn\s*\(", code):
            line = code.count("\n", 0, match.start()) + 1
            callers.append(f"{path.name}:{line}")

    assert callers == ["HarnessSmokeTests.java:91"] or len(callers) == 1, (
        f"`AppSession.signIn` is called from {callers}. It should have exactly ONE caller — the "
        "HarnessSmokeTests test that exists to drive the real UI sign-in flow.\n\n"
        "Every other suite signs in through `ensureSignedIn`, which uses the deterministic "
        "callback path. Routing more suites through the UI flow re-imports a race measured at "
        "2 failures in "
        "9 runs, once per suite instead of once per tier.\n\n"
        "If a second caller is genuinely right, update this guard and say why."
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
