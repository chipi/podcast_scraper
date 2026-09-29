import XCTest

/**
 * The base every native suite inherits (#2091).
 *
 * ## Why this exists
 *
 * The native tier had no shared setup, so each suite inherited whatever the previous one persisted.
 * `Journey.launch()` relaunches the app, which resets memory — but `localStorage`, Capacitor
 * `Preferences` and the account's data on the fixture API all survive it. Ordering decided results.
 *
 * Every native failure on 2026-09-16 was that, not a product bug: `ConfigOfflineToggleTests` left
 * forced-offline ON and three later tests failed with "no person row" / "no push cell" / an upload
 * rejection. A failure names the victim, never the culprit.
 *
 * The per-test discipline that landed then — assert the round trip, do not assume a starting state,
 * restore what you flip — is correct and stays. This removes the need for it to be *remembered*.
 *
 * ## The two kinds of leaked state, and the two fixes
 *
 * 1. **Account data** (favourites, queue, captures, interests) lives server-side against a user id.
 *    Fixed by giving each suite its OWN identity, so suites cannot collide at all. Same mechanism
 *    the browser tier uses (`signInIsolated` derives `${spec}-${project}`) — the mock provider mints
 *    an account for any name, so the dev picker is enough and no API change is needed.
 *
 * 2. **Device-local state** (`lp.forceOffline` in `localStorage`, accordion positions, dismissals)
 *    belongs to the APP, not the account, so a fresh identity does not clear it. Fixed by
 *    normalising the switches that are known to break later suites, in `setUp`.
 *
 * ## Suites that WANT shared state say so
 *
 * `make ios-contact-sheet` deliberately runs the journey + personalisation suites first so the tour
 * photographs a populated app, and the offline suites read downloads seeded by `seed-ios-download`
 * under the shared `simtest` account. That dependency is legitimate; it was just implicit. Those
 * suites override `accountIdentity` to `Self.sharedSeededIdentity`, which makes the coupling a
 * declaration rather than an accident.
 */
class UITestCase: XCTestCase {

  /// The account the make-level seeding targets mint (`ios-journey-signin`, `seed-ios-download`).
  /// A suite that reads seeded data must opt into it explicitly.
  static let sharedSeededIdentity = "simtest"

  /**
   * The account this suite signs in as. Per-suite by default, derived from the class name.
   *
   * Override with `Self.sharedSeededIdentity` in a suite that genuinely depends on make-seeded
   * state — and say why in a comment, because it re-enters the shared world this class exists to
   * leave.
   */
  var accountIdentity: String {
    // A RUN-LEVEL OVERRIDE, for the one job that needs several suites to share an account.
    //
    // `ios-contact-sheet` runs AppJourneyTests + PersonalisationTests first, specifically so the
    // tour photographs a populated app rather than a wall of empty states. That worked while every
    // suite shared one account. Per-suite identities (#2091) broke it silently: the seeders moved to
    // `appjourneytests` / `personalisationtests` while `ScreenshotTourTests` overrides to `simtest`,
    // so the tour began photographing an account nobody had seeded — back to the empty states the
    // seeding step was added to eliminate, and nothing asserts on a contact sheet, so nobody saw it.
    //
    // The target now passes `TEST_RUNNER_LP_FORCE_IDENTITY=simtest` to the seeding invocation, which
    // puts the seeders on the same account as the tour. Deliberately an explicit env var rather than
    // a default: it makes the sharing a declaration at the call site, which is the same reasoning
    // that made `sharedSeededIdentity` an explicit override rather than the norm.
    if let forced = ProcessInfo.processInfo.environment["LP_FORCE_IDENTITY"],
      !forced.trimmingCharacters(in: .whitespaces).isEmpty
    {
      return forced.trimmingCharacters(in: .whitespaces)
    }
    // `OfflineSpikeUITests.AppJourneyTests` -> `appjourneytests`. Lowercased and stripped to the
    // charset the mock provider accepts, matching how the browser tier derives its own ids.
    let raw = String(describing: type(of: self))
    let tail = raw.split(separator: ".").last.map(String.init) ?? raw
    return tail.lowercased().filter { $0.isLetter || $0.isNumber }
  }

  /**
   * Device-local switches that survive a relaunch AND an account change.
   *
   * Only `lp.forceOffline` today, because it is the one with a proven cross-suite failure. Add to
   * this deliberately: every entry costs time in every suite, so it should earn its place with a
   * real incident rather than a theory.
   */
  override func setUp() {
    super.setUp()
    // STOP AT THE FIRST FAILURE.
    //
    // This was `true`, so a test carried on after its first failed assertion — and these are device
    // tests, so "carried on" means it kept DRIVING THE APP and mutating server-side state that later
    // suites read. `DownloadThroughUITests` failed "the deep link did not land", then hunted "More
    // actions" for 20s on the wrong page and emitted a second, misleading failure about a control
    // that was never going to be there.
    //
    // Two costs, and the second is the expensive one: the log names a symptom several steps
    // downstream of the cause, and the account the run leaves behind is one nothing asked for.
    //
    // The trade is real and worth stating: the helpers in `Journey`/`AppSession` call `XCTFail`
    // themselves, so aborting at the first one can lose a caller's more specific follow-up message.
    // The helpers' messages are the specific ones — they carry the element inventory — so the
    // cheaper loss is the follow-up, not the corrupted state.
    continueAfterFailure = false
  }

  /**
   * How to reach Profile for THIS suite's account.
   *
   * The masthead link is named `auth.user?.name || t('profile.title')`, so the label is the account
   * name once `/me` resolves and the generic string only until then. Both have to be tried, and
   * only the suite knows the first one. Mirrors `UITestCase.java`'s `profileLabels()`.
   */
  var profileLabels: [String] { [accountIdentity, "Your profile"] }

  /**
   * Bring the app to a known state, then sign in as this suite's account.
   *
   * NOT in `setUp`: several suites deliberately start with the app or the API in an unusual state
   * (offline, server degraded, cold install), and a base class that force-launched the app would
   * fight them. Call it first from a test that wants the guarantee.
   */
  @discardableResult
  func startClean(_ app: XCUIApplication) -> Bool {
    // SIGN IN FIRST, then normalise the device switch. This was the other way round, and the
    // Android twin already documents why that is wrong (`UITestCase.java`): the offline switch
    // lives in Settings, Settings is reached through the masthead avatar, and the avatar only
    // exists when there IS a session. A signed-out app cannot reach the switch at all, so putting
    // it first spent a minute of swipes and timeouts discovering that on every single test.
    //
    // `Journey.setOfflineMode` carries the scar of the old order in its own comment: "On a fresh
    // simulator the app is signed out, Settings is unreachable, and the control the existence check
    // saw belonged to a page the app was already leaving (2026-09-24)." That is this bug, worked
    // around from the inside rather than fixed here.
    guard AppSession.ensureSignedIn(app, as: accountIdentity) else { return false }
    // Forced-offline OFF: device-local, so it survives a relaunch AND an account change, and left
    // ON every later network assertion fails for a reason that has nothing to do with the suite
    // reporting it. That was every native failure on 2026-09-16.
    //
    // THIS SUITE'S labels, not the hardcoded list. With per-suite identities the masthead reads the
    // account name, which `Journey.openProfile`'s default list does not contain — so for every
    // suite that does not override `accountIdentity` this call could not reach Settings at all, and
    // its return value is deliberately ignored, so it failed in silence. Ignoring the result is
    // still right (a suite that starts online must not be blocked by a switch that is already off)
    // but it is only safe once the call can actually succeed.
    _ = Journey.setOfflineMode(app, on: false, labels: profileLabels)
    // END ON HOME. `setOfflineMode` reaches the switch through Profile ▸ Settings and leaves the
    // app there, so "bring the app to a known state" was handing back an app parked on Settings —
    // and every caller then navigates as if it were on Home.
    //
    // MEASURED 2026-09-28: `test03TopicAndStoryline` failed "no topic row was tappable from the
    // Home rail" while its own inventory was plainly the SETTINGS page:
    //     Clear cache | Remove downloads | Wi-Fi only | Offline mode | ‹ Your profile
    // Only reachable once `startClean` was actually being called — before tonight no phase-4 test
    // called it at all, so nothing ever observed where it left the app.
    _ = Journey.openTab(app, "Home")
    return true
  }
}
