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
    continueAfterFailure = true
  }

  /**
   * Bring the app to a known state, then sign in as this suite's account.
   *
   * NOT in `setUp`: several suites deliberately start with the app or the API in an unusual state
   * (offline, server degraded, cold install), and a base class that force-launched the app would
   * fight them. Call it first from a test that wants the guarantee.
   */
  @discardableResult
  func startClean(_ app: XCUIApplication) -> Bool {
    // Forced-offline OFF first: it is device-local, so signing in as a different account does not
    // clear it, and with it ON every later network assertion fails for the wrong reason.
    _ = Journey.setOfflineMode(app, on: false)
    return AppSession.ensureSignedIn(app, as: accountIdentity)
  }
}
