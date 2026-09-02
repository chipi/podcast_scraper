import XCTest

/**
 * Shared session helpers for the device tier.
 *
 * Both suites need the same question answered — "is this app signed in?" — and answering it too
 * early is what made the harness non-deterministic. Boot PAINTS the last known identity from the
 * device first, so an offline launch is signed in immediately, and only then revalidates. A token
 * minted by a previous api container is refused by the new one (the fixture api is recreated on
 * every `make app-e2e-api-up`), so "Sign out" can sit on screen for a few seconds and then vanish.
 *
 * The playback test used to decide inside that window: it saw the painted session, skipped its
 * sign-in, and then asserted against an app that had just signed itself out. The failure read as
 * "sign-in did not complete" when no sign-in had been attempted at all.
 */
enum AppSession {
  /// Whether the app is STILL signed in once the boot revalidation has landed.
  static func isSignedIn(_ app: XCUIApplication) -> Bool {
    guard app.buttons["Sign out"].firstMatch.waitForExistence(timeout: 12) else { return false }
    // The painted session is not the answer — the revalidation that follows it is. Six seconds is
    // the observed worst case for `refresh()` against the local fixture api plus a re-render.
    sleep(6)
    return app.buttons["Sign out"].firstMatch.exists
  }

  /// Open an episode by slug through the app's deep-link scheme (#1925).
  ///
  /// Replaces navigating by taps and blind swipes, which was the single flakiest thing in this
  /// tier: the shows and episode lists keep their scroll position between visits, so a
  /// swipe-and-look search walked past rows and — on one run — mis-tapped and downloaded an
  /// episode the test never asked for. A deep link addresses the episode directly, which is what
  /// the link is FOR; the test is now exercising a product capability rather than working around
  /// the lack of one.
  static func openEpisode(_ app: XCUIApplication, slug: String) {
    guard let url = URL(string: "closelistening://episode/\(slug)") else {
      XCTFail("could not build a deep link for \(slug)")
      return
    }
    // `XCUIDevice.system.openURL` hands the URL to the OS, which is the real path a shared link
    // takes — not a test-only shortcut into the router. (`Process` is not available here: a UI
    // test bundle runs ON the simulator, so it cannot shell out to `xcrun` on the host.)
    XCUIDevice.shared.system.open(url)
    _ = app.wait(for: .runningForeground, timeout: 15)
  }

  /// Sign in as the dedicated `uitest` identity through the dev picker. Idempotent-ish: call it
  /// only when `isSignedIn` is false.
  static func signIn(_ app: XCUIApplication, _ springboard: XCUIApplication) -> Bool {
    let signIn = app.links["Sign in"].firstMatch
    guard signIn.waitForExistence(timeout: 20) else {
      XCTFail("neither Sign in nor Sign out present")
      return false
    }
    signIn.tap()

    // Dev picker: a TextField placeholdered "or a custom name…"; its Sign in button stays
    // Disabled until the field has text.
    let input = app.textFields.firstMatch
    guard input.waitForExistence(timeout: 20) else {
      XCTFail("no dev identity input")
      return false
    }
    input.tap()
    input.typeText("uitest")

    let submit = app.buttons["Sign in"].firstMatch
    guard submit.waitForExistence(timeout: 10), submit.isEnabled else {
      XCTFail("submit stayed disabled after typing")
      return false
    }
    submit.tap()

    // ASWebAuthenticationSession shows a system consent sheet owned by Springboard.
    let consent = springboard.buttons["Continue"]
    if consent.waitForExistence(timeout: 10) { consent.tap() }

    return app.buttons["Sign out"].firstMatch.waitForExistence(timeout: 30)
      || app.links["Sign out"].firstMatch.waitForExistence(timeout: 5)
  }
}
