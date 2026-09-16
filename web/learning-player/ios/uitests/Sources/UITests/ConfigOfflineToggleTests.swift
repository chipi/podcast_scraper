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
final class ConfigOfflineToggleTests: XCTestCase {
  private func openSettings(_ app: XCUIApplication) -> Bool {
    // The masthead avatar is a link whose accessible name is the user's display name, falling back
    // to "Your profile" when the profile has no name. Accept either so the test does not depend on
    // which identity seeded the session.
    let profile = app.links.matching(
      NSPredicate(format: "label == 'Your profile' OR label == 'simtest' OR label == 'uitest'")
    ).firstMatch
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
    sleep(6) // let boot revalidation land so the masthead has painted the signed-in state

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
    print("=====SETTINGS_AFTER_TREE_START====="); print(app.debugDescription); print("=====SETTINGS_AFTER_TREE_END=====")
    XCTAssertNotEqual(before, after, "the Offline mode checkbox did not change state")
  }
}
