import XCTest

/**
 * Photograph the app running against PRODUCTION, signed out.
 *
 * Separate from `ScreenshotTourTests` because the preconditions are different in kind: that tour
 * needs a seeded, signed-in fixture account, and prod has neither a mock provider nor seeded data.
 * What this can cover is the login-first landing and whatever the tab bar reaches without a
 * session — which is exactly the surface a new user meets, and the one nobody photographs.
 *
 * Assertion-free on purpose: it is a camera pointed at a live system, and a surface that fails to
 * load against prod is the single most interesting thing it could capture.
 */
final class ProdTourTests: UITestCase {

  func testTourProdSignedOut() {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.terminate()
    app.launch()
    _ = app.wait(for: .runningForeground, timeout: 30)
    sleep(10) // live network, not a loopback fixture — give the first paint room

    Journey.shot(self, "p01-landing")
    Journey.inventory(app, "prod-landing")

    for (i, tab) in ["Discover", "Search", "Library", "Home"].enumerated() {
      if Journey.openTab(app, tab) {
        sleep(6)
        Journey.shot(self, String(format: "p0%d-%@", 2 + i, tab.lowercased()))
      } else {
        print("=====PROD_TAB_MISS \(tab)=====")
      }
    }

    // A show from the featured rail — the deepest read-only surface reachable signed out.
    if Journey.tap(app, labels: ["Featured this week"], contains: true, timeout: 6) { sleep(2) }
    let card = app.links.allElementsBoundByIndex.first { $0.label.count > 12 && $0.isHittable }
    if let card {
      card.tap()
      sleep(8)
      Journey.shot(self, "p06-show-or-episode")
      Journey.inventory(app, "prod-deep")
    }
  }
}
