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
_SUPPORT_FILES = {"Journey.swift", "AppSession.swift", "UITestCase.swift"}

#: Suites that are deliberately not wired to a target, each with the reason.
#:
#: Empty on purpose. Add an entry only with a reason that survives being read aloud — "it is slow"
#: is not one (give it its own nightly target instead), and neither is "it is flaky" (fix it or
#: delete it; a quarantined suite that nobody runs is the exact thing this file exists to catch).
_UNWIRED_BY_DESIGN: dict[str, str] = {}

#: Suites reachable from a target but deliberately OUTSIDE the `test-ios` entry point.
#:
#: A much narrower door than `_UNWIRED_BY_DESIGN`: these still run, just never as part of the tier.
#: Every entry is about the preconditions being incompatible with the gate, not about the suite.
_OUTSIDE_THE_TIER: dict[str, str] = {
    "MagicLinkJourneyTests": (
        "Needs a REAL MAILBOX between its phases: M1 requests a link, the delivery worker emails "
        "it, and M2/M3 take that link as input. The tier has no mailbox and no worker, so folding "
        "it in would make every run fail for want of a link. "
        "`make test-app-ios-magic-link PHASE=M1|M2|M3`, run deliberately (#2272)."
    ),
    "MemoryWalkTests": (
        "A MEASUREMENT, not an assertion suite: it walks deep links and holds each screen while "
        "`ios-memory-walk.sh` samples the WebKit process with `footprint` from the host. It "
        "asserts nothing, and its numbers mean something only against a chosen backend and token. "
        "`make perf-ios`."
    ),
    "ProdTourTests": (
        "Points at the REAL production backend and wants NO session, where every step of the tier "
        "wants the fixture api and a seeded one. Folding it in would mean a prod outage reads as a "
        "native-shell regression. `make test-app-ios-prod-tour`, on purpose or not at all."
    ),
    "ScreenshotTourTests": (
        "A CAMERA, not an assertion suite — best-effort by design, it photographs whatever is on "
        "screen rather than failing. Gating on it would gate on screenshots. Run it with "
        "`make ios-contact-sheet`."
    ),
}


def _suite_names() -> list[str]:
    """Every XCUITest class name that has at least one `func test…`."""
    names: list[str] = []
    for path in sorted(UITESTS_DIR.glob("*.swift")):
        if path.name in _SUPPORT_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if not re.search(r"\bfunc\s+test\w*\s*\(", text):
            continue
        # Matches the BASE too: #2091 moved every suite onto `UITestCase` (which itself subclasses
        # XCTestCase for the per-suite account + known-state setUp), and a regex pinned to
        # `XCTestCase` then found zero suites. It failed loudly rather than passing vacuously —
        # `test_uitests_dir_is_found` exists for exactly that — but it is a reminder that a guard
        # naming one superclass goes stale the moment a base class appears.
        for match in re.finditer(r"\bclass\s+(\w+)\s*:\s*(?:XCTestCase|UITestCase)\b", text):
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


def _recipes() -> dict[str, list[str]]:
    """Map each Makefile target to its recipe lines."""
    recipes: dict[str, list[str]] = {}
    current: str | None = None
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^([A-Za-z0-9_.\-/]+)\s*:(?!=)", line)
        if header:
            current = header.group(1)
            recipes.setdefault(current, [])
            continue
        # A recipe line is TAB-indented; anything else at column 0 ends the recipe.
        if current is not None and (line.startswith("\t") or not line.strip()):
            recipes[current].append(line)
        elif line and not line[0].isspace():
            current = None
    return recipes


def _suites_reachable_from(entry: str) -> set[str]:
    """Every suite run by `entry`, following `$(MAKE) <target>` transitively."""
    recipes = _recipes()
    seen: set[str] = set()
    stack = [entry]
    suites: set[str] = set()
    while stack:
        target = stack.pop()
        if target in seen:
            continue
        seen.add(target)
        body = "\n".join(recipes.get(target, []))
        suites.update(re.findall(r"-only-testing:\w+/(\w+)", body))
        stack.extend(re.findall(r"\$\(MAKE\)\s+([A-Za-z0-9_.\-/]+)", body))
    return suites


@pytest.mark.unit
def test_no_ios_test_is_quietly_skipped() -> None:
    """A wired suite whose tests call `XCTSkip` runs less than it appears to.

    The twin of the Android guard. A skipped XCTest still lets the suite report success, so the tier
    keeps claiming coverage it no longer has — the same shape as a suite no target runs, one level
    down, and the same shape as `OK (0 tests)` passing on Android before that was closed.

    There are none today, so this is a floor rather than a cleanup. The bar for adding one is a
    reason written beside it: "flaky" is not a reason (fix it or delete it — a quarantined test
    nobody runs is exactly what this file exists to catch).
    """
    skipped: list[str] = []
    for path in sorted(UITESTS_DIR.glob("*.swift")):
        # Comments stripped first: these files discuss skipping while explaining decisions, and a
        # guard that fires on prose is the fake check this suite exists to catch.
        code = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", " ", path.read_text(encoding="utf-8"))
        if re.search(r"\bXCTSkip(?:If|Unless)?\s*\(", code):
            skipped.append(path.name)

    assert not skipped, (
        f"These iOS suites skip tests at runtime: {sorted(skipped)}.\n\n"
        "A skipped test still reports success, so the tier's green says more than it knows. Fix it "
        "or delete it; if it must be skipped, say why beside the call and list the file here."
    )


@pytest.mark.unit
def test_every_suite_is_reachable_from_the_test_ios_ENTRY_POINT() -> None:
    """Reachable from *a* target is not the same as reachable from the GATE.

    The original guard asked only whether some target mentioned the suite. `OfflinePlaybackTests`
    satisfied that for weeks while its only home, `test-app-ios-sim`, was called by nothing — so the
    guard was green about a suite that never ran. `ServerDegradedTests` was in the same position and
    its resulting failure got misread as an Android-vs-iOS product difference (2026-09-25).

    `test-ios` is the contract. A suite outside it runs only when someone remembers, which is the
    state this whole file exists to make impossible.
    """
    reachable = _suites_reachable_from("test-ios")
    # Guard the guard: zero reachable would report "everything is wired" having checked nothing.
    assert reachable, "no suites reachable from `test-ios` — did the target or the parser break?"

    missing = [
        name
        for name in _suite_names()
        if name not in reachable
        and name not in _OUTSIDE_THE_TIER
        and name not in _UNWIRED_BY_DESIGN
    ]

    assert not missing, (
        f"These suites exist but `make test-ios` does not run them: {sorted(missing)}.\n\n"
        "Being referenced by SOME target is not enough — `test-app-ios-sim` referenced "
        "OfflinePlaybackTests and was itself called by nothing, so the suite sat unrun while the "
        "wiring guard stayed green.\n\n"
        "Add it to a phase of `test-ios`, or record it in _OUTSIDE_THE_TIER with a reason."
    )
