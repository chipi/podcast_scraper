import XCTest

/**
 * Email magic-link sign-in, end to end on the device (#2272): create an account, land on its
 * profile, sign out, sign back in, land home.
 *
 * THREE PHASES, run one at a time, because the link travels through a real mailbox in between and
 * a test cannot read one. The operator (or a script) runs the delivery worker after M1 and M2, takes
 * the link from the outbox the app wrote, and passes it to the next phase as
 * `TEST_RUNNER_LP_MAGIC_LINK=<url>`:
 *
 *   M1  signed-out app → "Email me a sign-in link" → address → "Check your email"
 *   M2  open link 1 → NEW account lands on Profile → sign out → request link 2
 *   M3  with the app CLOSED, open link 2 → it launches the app → RETURNING account is signed in,
 *       NOT on Profile and NOT on the landing
 *
 * The address comes from `TEST_RUNNER_LP_MAGIC_EMAIL` and must be on the API's allowlist; the API
 * throttles one link per address per 60 s, so M2 must start at least a minute after M1.
 *
 * Not part of the default suite run (`test-ios` cannot supply a mailbox): run it through
 * `make test-app-ios-magic-link`. A phase without its inputs FAILS rather than skipping.
 */
final class MagicLinkJourneyTests: UITestCase {

  private var email: String? { env("LP_MAGIC_EMAIL") }
  private var link: String? { env("LP_MAGIC_LINK") }

  private func env(_ name: String) -> String? {
    let v = ProcessInfo.processInfo.environment[name]?.trimmingCharacters(in: .whitespaces)
    return (v?.isEmpty ?? true) ? nil : v
  }

  /// The three Profile tabs. Their presence together is what "landed on Profile" means — no other
  /// page shows all three.
  private static let profileTabs = ["Account", "Topics", "Stats"]

  private func onProfile(_ app: XCUIApplication, timeout: TimeInterval) -> Bool {
    Self.profileTabs.allSatisfy { Journey.find(app, labels: [$0], timeout: timeout) != nil }
  }

  /// From a signed-out app: reach /login, open the email form, send a link to `email`.
  private func requestLink(_ app: XCUIApplication, to email: String, tag: String) {
    guard app.links["Sign in"].firstMatch.waitForExistence(timeout: 20) else {
      Journey.inventory(app, "\(tag)-no-sign-in")
      return XCTFail("no signed-out 'Sign in' link — the app is not in a signed-out state")
    }
    app.links["Sign in"].firstMatch.tap()
    Journey.shot(self, "\(tag)-after-sign-in-tap")
    guard Journey.tap(app, labels: ["Email me a sign-in link"]) else {
      Journey.inventory(app, "\(tag)-no-magic-button")
      Journey.shot(self, "\(tag)-no-magic-button")
      return XCTFail("no magic-link button on /login")
    }
    Journey.shot(self, "\(tag)-after-magic-tap")
    // A web <input type=email> is exposed with its PLACEHOLDER as the value, not always the label,
    // so match either — and fall back to the only text field the form has.
    let byPlaceholder = app.textFields.matching(
      NSPredicate(format: "label == %@ OR placeholderValue == %@", "you@example.com",
        "you@example.com")
    ).firstMatch
    let field = byPlaceholder.waitForExistence(timeout: 10) ? byPlaceholder : app.textFields.firstMatch
    guard field.waitForExistence(timeout: 5) else {
      Journey.inventory(app, "\(tag)-no-email-field")
      Journey.shot(self, "\(tag)-no-email-field")
      return XCTFail("the email field did not appear")
    }
    field.tap()
    field.typeText(email)
    XCTAssertTrue(Journey.tap(app, labels: ["Send link"]), "no 'Send link' button")
    let sent = Journey.find(app, labels: ["Check your email"], timeout: 15)
    Journey.shot(self, "\(tag)-check-your-email")
    XCTAssertNotNil(sent, "the app never confirmed the link was sent")
  }

  /// Open `link` as Mail would: the system browser follows the verify redirect to
  /// `closelistening://auth#token=…`, and iOS asks before handing a custom scheme to an app.
  private func openLink(_ app: XCUIApplication, _ link: String) {
    guard let url = URL(string: link) else { return XCTFail("not a URL: \(link)") }
    XCUIDevice.shared.system.open(url)
    // The confirmation can come from Safari ("Open in …?") or SpringBoard depending on the hop, and
    // Safari's only appears after it has loaded the page and followed the redirect. Poll BOTH for
    // the whole window: checking each in turn with a short wait missed Safari's prompt, which came
    // up after its 8 s had run out (measured 2026-10-03) — and by then the link was spent.
    let prompts = ["com.apple.mobilesafari", "com.apple.springboard"].map {
      XCUIApplication(bundleIdentifier: $0).buttons["Open"]
    }
    let deadline = Date().addingTimeInterval(30)
    while Date() < deadline {
      if let open = prompts.first(where: { $0.exists }) {
        open.tap()
        break
      }
      if app.state == .runningForeground { break } // handed over with no prompt at all
      Thread.sleep(forTimeInterval: 0.5)
    }
    XCTAssertTrue(
      app.wait(for: .runningForeground, timeout: 20), "the app did not come to the foreground")
  }

  func testM1RequestALinkForANewAccount() throws {
    guard let email else { return XCTFail("missing TEST_RUNNER_LP_MAGIC_EMAIL — run via make test-app-ios-magic-link") }
    let app = XCUIApplication(bundleIdentifier: AppUnderTest.bundleId)
    app.launch()
    XCTAssertTrue(AppSession.signOut(app), "could not reach a signed-out state")
    requestLink(app, to: email, tag: "M1")
  }

  func testM2NewAccountLandsOnProfileThenSignsOut() throws {
    guard let email, let link else {
      return XCTFail("missing TEST_RUNNER_LP_MAGIC_EMAIL / _LINK — run via make test-app-ios-magic-link")
    }
    let app = XCUIApplication(bundleIdentifier: AppUnderTest.bundleId)
    app.launch()
    openLink(app, link)
    let landed = onProfile(app, timeout: 20)
    Journey.shot(self, "M2-landed")
    guard landed else {
      Journey.inventory(app, "M2-not-on-profile")
      return XCTFail("a NEW account must land on Profile (new=1), not Home")
    }
    XCTAssertTrue(AppSession.isSignedIn(app), "on Profile but not signed in")
    XCTAssertTrue(AppSession.signOut(app), "could not sign out")
    Journey.shot(self, "M2-signed-out")
    requestLink(app, to: email, tag: "M2")
  }

  func testM3ReturningAccountSignsInAndStaysOffProfile() throws {
    guard let link else { return XCTFail("missing TEST_RUNNER_LP_MAGIC_LINK — run via make test-app-ios-magic-link") }
    let app = XCUIApplication(bundleIdentifier: AppUnderTest.bundleId)
    // COLD: the link must be what LAUNCHES the app — the ordinary case of tapping "Sign in" in
    // Mail with the app closed. This phase first called `app.launch()` before opening the link,
    // which only ever tested a running app and hid that a launch-by-link dropped the token
    // (`initNativeAuth` did not read `getLaunchUrl`; fixed 2026-10-03).
    app.terminate()
    XCTAssertEqual(app.state, .notRunning, "the app must not be running before the link opens")
    openLink(app, link)
    XCTAssertTrue(AppSession.isSignedIn(app), "the returning account is not signed in")
    // Short timeout on purpose: this asserts ABSENCE, and the page has settled once signed in.
    let onProfileNow = onProfile(app, timeout: 3)
    // The landing's sign-up CTA. Signed in UNDER it was the real bug this phase first missed: the
    // masthead showed the avatar while the page still said "Create your free account" (2026-10-03).
    let stillOnLanding = Journey.find(app, labels: ["Create your free account"], timeout: 3) != nil
    Journey.shot(self, "M3-landed")
    XCTAssertFalse(onProfileNow, "a RETURNING account must not be sent to Profile")
    XCTAssertFalse(stillOnLanding, "signed in, but still on the signed-out landing page")
  }
}
