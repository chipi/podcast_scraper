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
