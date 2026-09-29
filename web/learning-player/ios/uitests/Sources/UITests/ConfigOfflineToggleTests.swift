import XCTest

/**
 * Drives the Settings → Config → "Offline mode" switch on device (operator request 2026-09-16).
 *
 * WHY a test rather than a tap: the switch persists to `localStorage` (`lp.forceOffline`), and the
 * host has no way to reach WKWebView's storage — the app's LocalStorage directory is empty until
 * WebKit materialises it, so it cannot be pre-seeded the way the native token can be via
 * `simctl spawn defaults write`. XCUITest is the only thing that can press it.
 *
 * Deliberately narrow: it flips the switch and asserts it flipped, nothing else. Because the flag
 * is persistent, the *observation* of what forced-offline does to the app is done from the host
 * afterwards (relaunch + screenshots + api access log) rather than inside XCUITest, where any
 * assertion about "no request was made" would be untestable from the device side.
 *
 * Preconditions: the app installed and ALREADY signed in (the host seeds the native bearer via
 * `CapacitorStorage.lp_native_token`), and the fixture api reachable on the origin the build was
 * pointed at.
 */
final class ConfigOfflineToggleTests: UITestCase {
  private func openSettings(_ app: XCUIApplication) -> Bool {
    // The masthead avatar is a link whose accessible name is the user's display name, falling back
    // to "Your profile" when the profile has no name.
    //
    // THIS SUITE'S identity, not a hardcoded list. What stood here was a FOURTH copy of
    // ["Your profile", "simtest", "uitest"] — the same literals `Journey.openProfile`,
    // `AppSession.signOut` and `Journey.setOfflineMode` each carried — and like the others it was
    // right only while every suite shared `simtest`. This suite signs in as
    // `configofflinetoggletests`, so once per-suite accounts became real none of the three matched
    // and it reported "could not reach Settings" about a Home screen with the avatar right there.
    let wanted = ([accountIdentity, "Your profile"])
      .map { "label == '\($0)'" }
      .joined(separator: " OR ")
    let profile = app.links.matching(NSPredicate(format: wanted)).firstMatch
    guard profile.waitForExistence(timeout: 25) else {
      print("=====NO_PROFILE_LINK_TREE_START====="); print(app.debugDescription); print("=====NO_PROFILE_LINK_TREE_END=====")
      return false
    }
    profile.tap()
    sleep(3)

    // Profile → the gear, aria-labelled with the Settings title.
    let gear = app.links.matching(NSPredicate(format: "label CONTAINS[c] 'Settings'")).firstMatch
    guard gear.waitForExistence(timeout: 20) else {
      print("=====NO_GEAR_TREE_START====="); print(app.debugDescription); print("=====NO_GEAR_TREE_END=====")
      return false
    }
    gear.tap()
    sleep(3)
    return true
  }

  func testTogglesForcedOfflineOn() throws {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))
    guard startClean(app) else {
      XCTFail("sign-in did not complete as \(accountIdentity)")
      return
    }
    sleep(6) // let boot revalidation land so the masthead has painted the signed-in state

    // PUT THE SWITCH BACK, whatever happens below.
    //
    // This test flips forced-offline ON and left it there. The switch is DEVICE-LOCAL — it survives
    // a relaunch and an account change, which is precisely why `UITestCase`'s docstring names
    // `lp.forceOffline` as the one piece of leaked state worth normalising — so every suite that
    // ran afterwards started offline.
    //
    // That is unrecoverable from inside a later test, not merely inconvenient: with the app
    // offline `/me` never resolves, so there is no account name, no "Sign out" (it is
    // `v-if="auth.isAuthenticated"`) and no "Sign in" either. Measured 2026-09-29 — the next
    // suite's `signIn` dumped an inventory reading `Offline mode is on — showing saved` with three
    // `Try again` buttons, and failed "neither Sign in nor Sign out present" on an app that simply
    // could not reach the server.
    //
    // `defer` rather than a trailing call: the assertions below can return early, and a restore
    // that only runs on the happy path is the one that will not run when it matters.
    defer { _ = Journey.setOfflineMode(app, on: false, labels: profileLabels) }

    guard openSettings(app) else { XCTFail("could not reach Settings"); return }

    // The control is a bare <input type="checkbox"> wrapped in a <label>, so its accessible name
    // is the label's text. WebKit surfaces it as a checkBox or a switch depending on version —
    // query both, plus a whole-tree fallback, rather than guessing one element type.
    let predicate = NSPredicate(format: "label CONTAINS[c] 'Offline mode'")
    var control = app.checkBoxes.matching(predicate).firstMatch
    if !control.waitForExistence(timeout: 10) {
      control = app.switches.matching(predicate).firstMatch
    }
    if !control.waitForExistence(timeout: 10) {
      control = app.descendants(matching: .any).matching(predicate).firstMatch
    }
    guard control.waitForExistence(timeout: 10) else {
      print("=====SETTINGS_TREE_START====="); print(app.debugDescription); print("=====SETTINGS_TREE_END=====")
      XCTFail("Offline mode control not found")
      return
    }

    print("=====OFFLINE_CONTROL label=\(control.label) value=\(String(describing: control.value)) type=\(control.elementType.rawValue)=====")
    let before = String(describing: control.value)

    // Scroll it clear of the tab bar before tapping — the same trap the playback suite documents.
    var tries = 0
    while control.frame.maxY > app.frame.height - 90 && tries < 6 {
      app.swipeUp(); sleep(1); tries += 1
    }
    control.tap()
    sleep(3)

    let after = String(describing: control.value)
    print("=====OFFLINE_TOGGLE before=\(before) after=\(after)=====")
    XCTAssertNotEqual(before, after, "the Offline mode checkbox did not change state")

    // RESTORE. The switch is PERSISTED (localStorage), so leaving it flipped hands every later test
    // an app in forced-offline: reads fast-fail, the surfaces empty out, and the failures read as
    // "no push cell" / "no person row" / a failed upload — three tests blamed for a fourth test's
    // side effect (2026-09-16). A test that mutates persisted state owns putting it back.
    if after != before {
      control.tap()
      sleep(2)
      let restored = String(describing: control.value)
      print("=====OFFLINE_RESTORED \(restored)=====")
      XCTAssertEqual(restored, before, "left the offline switch flipped for every later test")
    }
    print("=====SETTINGS_AFTER_TREE_START====="); print(app.debugDescription); print("=====SETTINGS_AFTER_TREE_END=====")

  }
}
