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
        status=BOTH,
        why=(
            "Android measured the Sign-out tap landing UNDERNEATH the bottom nav: the tap hit "
            "'Discover', the app navigated there, and the session was of course still present. So "
            "it retries, re-resolves the node each round (one found before a scroll is stale after "
            "it), and goes through `tap` rather than clicking the raw node, because `tap` lifts a "
            "control clear of that overlap. iOS had the raw single tap until 2026-09-28; when it "
            "missed, `ensureSignedIn` proceeded believing it had signed out and could certify the "
            "PREVIOUS suite's account as this one's — defeating the isolation #2091 exists for."
        ),
    ),
    Capability(
        name="session state is read from the masthead, not a trip to Profile",
        android=("AppSession.java", r"private static boolean settles\(String label"),
        ios=("AppSession.swift", r"func settles\("),
        status=BOTH,
        why=(
            "Both platforms answer 'is this app signed in, and as whom' by reading the masthead — "
            "the profile link is labelled `auth.user?.name || t('profile.title')` and the "
            "notifications bell renders only under `auth.hasSession`. Both used to navigate to "
            "Profile (twelve swipes), scroll to 'Sign out', sleep 6s and scroll again: ~40s per "
            "call, on every startClean, to learn something already on screen. Android: "
            "HarnessSmokeTests 314.6s -> 177.9s. "
            "CLOSED 2026-09-28, and iOS turned out to be paying far more than the ~40s Android "
            "was. `NativeOnlySurfacesTests` spent 500+ seconds in a poll loop without reaching an "
            "assertion: it signed in, could not SEE that it had, and retried — five complete "
            "`auth/login` -> `auth/callback` pairs in the api log for ONE test. "
            "`openProfile`'s own diagnostic cleared the control twelve times over "
            "(`PROFILE_CTL link 'simtest' frame=(349.0, 62.0, 48.0, 18.0) hittable=true`), which "
            "is what made it a detection bug rather than a navigation one. After the port: in "
            "142.4s, sign-in resolved at t=16s on the first attempt, zero SETTLE markers. "
            "The temporary reason this entry carried — 'the iOS tier has not run since "
            "`ios-origin-up` began hanging (undiagnosed)' — was wrong on both counts: the target "
            "does not hang and the failure was in `ios-app-install`, a build failure from a "
            "half-pruned /tmp derived-data tree. Diagnosed and fixed the same day."
        ),
    ),
    Capability(
        name="suites sign in through the callback path ONLY, with no silent UI fallback",
        android=("AppSession.java", r"there is no fallback"),
        ios=("AppSession.swift", r"there is no fallback"),
        status=ANDROID_ONLY,
        why=(
            "Android's `ensureSignedIn` uses the OAuth-callback intent and FAILS if it does not "
            "land. It used to fall back to the UI flow, defended as insurance; when the callback "
            "actually broke on 2026-09-28 the fallback ran `signIn` against an already-signed-in "
            "app, typed the identity into Home's search box, and failed four steps later as "
            "'sign-in did not complete' — converting a precise failure into a confusing one. The "
            "real UI flow keeps its coverage in one dedicated HarnessSmokeTests test, pinned to a "
            "single caller by a guard. iOS has no callback path at all yet (it cannot fire an "
            "intent), so it necessarily still drives the UI flow everywhere."
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


@pytest.mark.unit
def test_the_seeded_identity_is_the_same_string_everywhere() -> None:
    """The shared account's name exists in four files and they must agree.

    Neither harness can read the Makefile, so the copies are unavoidable. What is avoidable is them
    drifting apart, and the reason to guard it is that a mismatch is SILENT: the make-level seeding
    would populate one account while the suite reads an empty one, and the suite would
    report missing content rather than a wrong name.

    That is not hypothetical. `ios-contact-sheet` runs the journey and personalisation
    suites first so the tour photographs a populated app; when per-suite identities
    landed (#2091) the seeders moved to their own accounts while
    `ScreenshotTourTests` stayed on the shared one, and the tour went
    back to photographing empty states. Nothing asserts on a contact sheet, so it stayed broken for
    weeks. This is the check that would have caught the same shape of mistake in a second.
    """
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    found: dict[str, str | None] = {}

    m = re.search(r"^IOS_SEED_IDENTITY \?= *(\S+)", makefile, re.M)
    found["Makefile IOS_SEED_IDENTITY"] = m.group(1) if m else None

    # The mint URL must use the variable, not a second literal — that copy is how this drifts.
    m = re.search(r"/api/app/auth/login\?as=([^&\"]+)&platform=native", makefile)
    found["Makefile ios-journey-signin as="] = m.group(1) if m else None

    m = re.search(r'sharedSeededIdentity\s*=\s*"([^"]+)"', _code(IOS_DIR, "UITestCase.swift"))
    found["iOS sharedSeededIdentity"] = m.group(1) if m else None

    m = re.search(r'SHARED_SEEDED_IDENTITY\s*=\s*"([^"]+)"', _code(ANDROID_DIR, "UITestCase.java"))
    found["Android SHARED_SEEDED_IDENTITY"] = m.group(1) if m else None

    missing = [k for k, v in found.items() if v is None]
    assert not missing, (
        f"could not find the seeded identity in: {missing}. A marker that matches nothing passes "
        f"silently, so this fails instead. Found: {found}"
    )

    # The Makefile's `as=` should be the VARIABLE reference, which is the whole point of having one.
    as_value = found["Makefile ios-journey-signin as="]
    assert as_value == "$(IOS_SEED_IDENTITY)", (
        f"`ios-journey-signin` mints for '{as_value}' as a literal instead of "
        "$(IOS_SEED_IDENTITY). That is a fifth copy of the account name, and the one that decides "
        "which account actually gets a token."
    )

    resolved = {k: v for k, v in found.items() if k != "Makefile ios-journey-signin as="}
    assert len(set(resolved.values())) == 1, (
        "the shared seeded identity is spelled differently in different places, so make-level "
        f"seeding and the suites would use different accounts: {resolved}"
    )
