"""The iOS and Android device harnesses are ports of one design, and ports drift.

The third guard in this family, after ``test_ios_uitest_suites_are_wired.py`` and
``test_android_uitest_suites_are_wired.py``. Those pin that a suite is REACHABLE. This pins that
the two tiers still agree about what the harness can DO.

## Why this exists

Every iOS defect fixed on 2026-09-28 was already fixed on Android and had simply never been
ported:

  * ``Journey.openProfile`` reached the masthead by a hardcoded ``["Your profile", "simtest",
    "uitest"]`` — a list from when every suite shared one account. Android took the labels as a
    parameter and its own comment recorded the flaw as latent on iOS
    (``Journey.java``), where it sat unfixed.
  * ``UITestCase.startClean`` normalised the offline switch BEFORE signing in. The switch is in
    Settings, Settings is behind the masthead avatar, and the avatar needs a session — so it could
    not reach the control it exists to set. Android documented the correct order and did it;
    iOS kept the broken one.

Neither was found by reading the iOS code. Both were found by an outside review noticing that the
two files disagreed. That is the failure mode here: a fix lands on one platform, the other keeps
the bug, and nothing in the repo knows the two are supposed to match.

The code genuinely cannot be shared — XCUITest is Swift and iOS-only, UI Automator is Java and
reads a different accessibility tree — so the only thing that can be shared is a LEDGER of what
each side has, checked mechanically.

## How to read the table

``BOTH`` is a parity pin: the capability exists on both tiers and losing it on either is a
regression. Most entries here got that status on 2026-09-28, when the gap was closed.

``ANDROID_ONLY`` / ``IOS_ONLY`` is a DECLARED GAP, and it is asserted in both directions — the
capability must be present on one side and absent on the other. That is deliberate. Porting one is
a good thing, and it must come with an edit here promoting the entry to ``BOTH``; the test failing
at that moment is the point, not an obstacle. It makes closing a gap visible, and it makes widening
one impossible to do by accident.

Deliberately in the Python tier, for the same reasons as its siblings: it is a repo-structure
invariant, it must run where there is no Android SDK and no Xcode, and neither harness can see the
other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ANDROID_DIR = (
    REPO_ROOT / "web/learning-player/android/app/src/androidTest/java/app/closelistening/player"
)
IOS_DIR = REPO_ROOT / "web/learning-player/ios/uitests/Sources/UITests"

BOTH = "both"
ANDROID_ONLY = "android-only"
IOS_ONLY = "ios-only"


@dataclass(frozen=True)
class Capability:
    """One behaviour the two harnesses are supposed to agree about.

    ``android`` / ``ios`` are (filename, regex) pairs. The regex is searched against the file's
    text; a match means the capability is present. They are deliberately narrow — a marker that
    matches a comment mentioning the feature would make this guard pass on prose, which is the
    exact class of fake check the rest of this suite exists to catch.
    """

    name: str
    android: tuple[str, str]
    ios: tuple[str, str]
    status: str
    why: str


CAPABILITIES: list[Capability] = [
    # ---------------------------------------------------------------- parity won on 2026-09-28
    Capability(
        name="openProfile accepts the CALLER's identity labels",
        android=("Journey.java", r"static boolean openProfile\(List<String> labels\)"),
        ios=("Journey.swift", r"func openProfile\(_ app: XCUIApplication, labels: \[String\]\)"),
        status=BOTH,
        why=(
            "The masthead link is named `auth.user?.name || t('profile.title')`, so its label is "
            "the ACCOUNT NAME once /me resolves. A hardcoded list cannot know a "
            "per-suite identity, "
            "and the failure is silent: startClean drives setOfflineMode through this and ignores "
            "the result, so the offline switch quietly stops being normalised."
        ),
    ),
    Capability(
        name="the base class exposes per-suite profile labels",
        android=("UITestCase.java", r"protected java\.util\.List<String> profileLabels\(\)"),
        ios=("UITestCase.swift", r"var profileLabels: \[String\]"),
        status=BOTH,
        why="Only the suite knows its own account name; the helper cannot guess it.",
    ),
    Capability(
        name="startClean signs in BEFORE normalising the offline switch",
        # Ordering, not proximity: comments are stripped before matching, so each identifier
        # appears exactly once and "ensureSignedIn ... setOfflineMode" means the sign-in comes
        # first. A windowed regex was tried and was wrong — the explanation between the two calls
        # is longer than any window worth hard-coding.
        android=("UITestCase.java", r"ensureSignedIn[\s\S]*setOfflineMode"),
        ios=("UITestCase.swift", r"ensureSignedIn[\s\S]*setOfflineMode"),
        status=BOTH,
        why=(
            "The offline switch lives in Settings, Settings is reached through the "
            "masthead avatar, "
            "and the avatar only exists when there is a session. The other order cannot reach the "
            "control at all and spends a minute of swipes discovering that on every test."
        ),
    ),
    # ---------------------------------------------------------------- declared gaps
    Capability(
        name="sign-in by delivering the OAuth callback directly (no Custom Tab)",
        android=("AppSession.java", r"static boolean signInViaCallback\(String identity\)"),
        ios=("AppSession.swift", r"func signInViaCallback\("),
        status=ANDROID_ONLY,
        why=(
            "Android mints a token over HTTP and delivers it as an ACTION_VIEW on "
            "closelistening://auth#token=, which lands in the same appUrlOpen listener production "
            "uses. It removed a sign-in flake measured at 2 failures in 9 runs. iOS cannot fire an "
            "intent; its equivalent would go through ASWebAuthenticationSession or a Safari "
            "hand-off, so it is a separate design rather than a port. iOS meanwhile still seeds a "
            "token by writing CapacitorStorage with `defaults write` (ios-journey-signin), which "
            "bypasses the app entirely — strictly worse than what Android now does."
        ),
    ),
    Capability(
        name="startClean cold-relaunches the app first",
        android=("UITestCase.java", r"AppSession\.relaunch\(\)"),
        ios=("UITestCase.swift", r"AppSession\.relaunch\("),
        status=ANDROID_ONLY,
        why=(
            "Android relaunches so what is on disk is re-read rather than re-activated, and waits "
            "for the WEB accessibility tree rather than the native shell. iOS relies on each suite "
            "launching the app itself, so a suite that does not is reading whatever the previous "
            "one left on screen."
        ),
    ),
    Capability(
        name="an accessible-name audit walks the app",
        android=("AccessibleNameAuditTests.java", r"class AccessibleNameAuditTests"),
        ios=("AccessibleNameAuditTests.swift", r"class AccessibleNameAuditTests"),
        status=ANDROID_ONLY,
        why=(
            "Every defect in the unnamed-control class so far was found by Android or by the "
            "static guard, so VoiceOver has no equivalent coverage. The two engines fail "
            "differently — WebKit DROPS an interactive element whose subtree is entirely "
            "aria-hidden, where Chromium keeps it and leaves it nameless — so Android being green "
            "says nothing about iOS."
        ),
    ),
    Capability(
        name="sign-out is verified, with a retry when the tap lands under the nav",
        android=("AppSession.java", r"for \(int attempt = 1; attempt <= \d+; attempt\+\+\)"),
        ios=("AppSession.swift", r"for attempt in 1\.\.\."),
        status=ANDROID_ONLY,
        why=(
            "Android measured the Sign-out tap landing under the bottom nav and retries, verifying "
            "afterwards. iOS taps the scrollTo result raw and sleeps — so when that tap misses, "
            "ensureSignedIn proceeds believing it signed out, and can certify a session belonging "
            "to the PREVIOUS suite's account. That defeats the per-suite isolation "
            "#2091 exists for."
        ),
    ),
]


#: `//` line comments and `/* */` blocks, in either language.
_COMMENTS = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")


def _code(directory: Path, filename: str) -> str:
    """The file with comments stripped.

    Markers are matched against CODE only, and that is load-bearing rather than tidy. Both of these
    files explain themselves at length, and they quote the very identifiers the markers look for —
    `UITestCase.swift` mentions `setOfflineMode` twice, once in a paragraph about the bug and once
    in the actual call. A marker that can be satisfied by prose would let this guard pass on a
    comment describing a capability the code no longer has, which is the exact shape of fake check
    the rest of this test suite exists to catch.

    It also makes the ordering markers sound: with comments gone, each identifier appears once, so
    "A appears before B" means what it says.
    """
    path = directory / filename
    if not path.is_file():
        return ""
    return _COMMENTS.sub(" ", path.read_text(encoding="utf-8"))


def _present(directory: Path, marker: tuple[str, str]) -> bool:
    filename, pattern = marker
    return bool(re.search(pattern, _code(directory, filename)))


@pytest.mark.unit
def test_both_harness_directories_are_found() -> None:
    """Guard the guard: a moved directory would make every marker 'absent' and read as total drift.

    Without this, renaming a directory turns every BOTH entry red and every gap entry green — a
    stampede of failures pointing at the wrong thing.
    """
    assert ANDROID_DIR.is_dir(), f"Android harness not found at {ANDROID_DIR}"
    assert IOS_DIR.is_dir(), f"iOS harness not found at {IOS_DIR}"


@pytest.mark.unit
@pytest.mark.parametrize("cap", CAPABILITIES, ids=lambda c: c.name)
def test_capability_matches_the_declared_parity(cap: Capability) -> None:
    """Each capability is on both tiers, or its absence is declared here with a reason."""
    on_android = _present(ANDROID_DIR, cap.android)
    on_ios = _present(IOS_DIR, cap.ios)

    if cap.status == BOTH:
        assert on_android and on_ios, (
            f"PARITY LOST: '{cap.name}' is on "
            f"{'Android' if on_android else 'iOS' if on_ios else 'NEITHER tier'} only.\n\n"
            f"Why it matters: {cap.why}\n\n"
            "Both tiers had this. Either restore it, or change its status in CAPABILITIES with a "
            "reason that survives being read aloud. Every iOS defect fixed on 2026-09-28 was a "
            "fix that had landed on Android and never been ported — this entry exists so that "
            "cannot happen again quietly."
        )
        return

    expected_android = cap.status == ANDROID_ONLY
    if on_android == expected_android and on_ios != expected_android:
        return

    holder, other = ("Android", "iOS") if expected_android else ("iOS", "Android")
    if on_android and on_ios:
        pytest.fail(
            f"GAP CLOSED — and that is good news that needs an edit: '{cap.name}' is now on BOTH "
            f"tiers, but CAPABILITIES still declares it {cap.status}.\n\n"
            "Promote it to BOTH so it becomes a regression pin. This test failing here is the "
            "mechanism working: closing a gap should be recorded, not silent."
        )
    pytest.fail(
        f"GAP WIDENED: '{cap.name}' is declared {cap.status} but is now missing from {holder} too "
        f"(android={on_android}, ios={on_ios}).\n\n"
        f"Why it matters: {cap.why}\n\n"
        f"Either restore it on {holder}, or remove the entry — but do not leave a ledger that "
        f"claims {other} is the only side lacking it when neither has it."
    )


@pytest.mark.unit
def test_every_declared_gap_has_a_real_reason() -> None:
    """A gap with a thin reason is a TODO pretending to be a decision.

    'Not ported yet' is not a reason — it is a restatement of the status. The bar is that someone
    reading it can tell whether the gap is a deliberate design difference (iOS cannot fire an
    intent) or an outstanding debt (iOS has no name audit), because those need different responses.
    """
    thin = [cap.name for cap in CAPABILITIES if cap.status != BOTH and len(cap.why) < 120]
    assert not thin, (
        f"These declared gaps have reasons too thin to act on: {thin}. Say what the capability "
        "buys, and why the other tier does not have it — design difference or outstanding debt."
    )
