import XCTest

/**
 * A full visual sweep of the app's surfaces, for eyeballing the whole product at once
 * (operator request 2026-09-16 — the shots are stitched into a single contact sheet by
 * `make ios-contact-sheet`).
 *
 * Deliberately NOT assertion-heavy: this is a CAMERA, not a gate. The journey suites
 * (`AppJourneyTests`, `PersonalisationTests`, `OfflineCacheTests`, `ServerDegradedTests`) own the
 * assertions; if this failed on a missing element it would abandon the tour half-way and produce a
 * partial sheet, which is the opposite of what it is for. Every step is therefore best-effort and
 * shoots whatever is on screen — a surface that failed to load is exactly the thing the sheet
 * should show.
 *
 * Numeric prefixes drive the order of tiles in the sheet.
 *
 * PRECONDITIONS: app installed, signed in (`make ios-journey-signin`), fixture api reachable.
 * Run AFTER the personalisation suite if you want Stats/Topics populated rather than empty.
 */
final class ScreenshotTourTests: XCTestCase {
  private let episodeSlug = "p09-a4bbb5dde3"

  override func setUp() {
    super.setUp()
    continueAfterFailure = true
  }

  /// Shoot the current screen; never fails the tour.
  private func frame(_ name: String) {
    Journey.shot(self, name)
  }

  func testTourEverySurface() {
    let app = Journey.launch()

    // --- primary tabs -------------------------------------------------------------------------
    frame("t01-home")

    if Journey.openTab(app, "Discover") { sleep(5); frame("t02-discover") }
    if Journey.openTab(app, "Search") { sleep(4); frame("t03-search") }

    // --- library, every tab -------------------------------------------------------------------
    if Journey.openTab(app, "Library") {
      sleep(4)
      frame("t04-library-following")
      for (i, tab) in ["Saved", "Boards", "Revisit"].enumerated() {
        if Journey.tap(app, labels: [tab], timeout: 10) {
          sleep(3)
          frame(String(format: "t0%d-library-%@", 5 + i, tab.lowercased()))
        }
      }
    }

    // --- profile, every tab, then settings ----------------------------------------------------
    if Journey.openProfile(app) {
      sleep(4)
      frame("t08-profile-account")
      if Journey.tap(app, labels: ["Topics"], timeout: 10) { sleep(3); frame("t09-profile-topics") }
      if Journey.tap(app, labels: ["Stats"], timeout: 10) { sleep(3); frame("t10-profile-stats") }
      if Journey.tap(app, labels: ["Settings"], contains: true, timeout: 10) {
        sleep(3)
        frame("t11-settings")
        // Config lives at the bottom — the offline switch + space reclaim.
        _ = Journey.scrollTo(app, labels: ["Offline mode"])
        frame("t12-settings-config")
      }
    }

    // --- episode + knowledge panel ------------------------------------------------------------
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    frame("t13-episode")
    if Journey.tap(app, labels: ["Insights"], contains: true, timeout: 12) {
      sleep(4)
      frame("t14-episode-insights")
      if Journey.tap(app, labels: ["Key points"], contains: true, timeout: 8) {
        sleep(3); frame("t15-episode-keypoints")
      }
      if Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 8) {
        sleep(3); frame("t16-episode-entities")
      }
    }

    // --- the overlays the operator asked to see rendered --------------------------------------
    Journey.dismissSheets(app)
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(5)
    if Journey.tap(app, labels: ["Share"], contains: true, timeout: 10) {
      sleep(3); frame("t17-share-popover")
      _ = Journey.tap(app, labels: ["Close", "Cancel"], contains: true, timeout: 4)
    }
    if Journey.tap(app, labels: ["Add to collection", "Add"], contains: true, timeout: 10) {
      sleep(3); frame("t18-add-to-collection")
      _ = Journey.tap(app, labels: ["Close", "Cancel"], contains: true, timeout: 4)
    }

    // --- entity surfaces ----------------------------------------------------------------------
    Journey.dismissSheets(app)
    Journey.openTab(app, "Home")
    sleep(5)
    if Journey.tap(app, labels: ["Topics"], timeout: 10) { sleep(3) }
    if Journey.tap(app, labels: ["systems thinking"], contains: true, timeout: 10) {
      sleep(5); frame("t19-topic")
    }
    Journey.openTab(app, "Home")
    sleep(4)
    if Journey.tap(app, labels: ["Storylines"], timeout: 10) {
      sleep(4)
      let rows = app.buttons.allElementsBoundByIndex.filter {
        $0.label.contains("momentum") && $0.label.contains("(")
      }
      if let first = rows.first, first.isHittable { first.tap(); sleep(5); frame("t20-storyline") }
    }

    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(5)
    _ = Journey.tap(app, labels: ["Insights"], contains: true, timeout: 10)
    sleep(2)
    _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 8)
    sleep(2)
    _ = Journey.scrollTo(app, labels: ["Open Dr. Elena Fischer", "Open Sam"])
    if Journey.tap(app, labels: ["Open Dr. Elena Fischer", "Open Sam"], contains: true, timeout: 8) {
      sleep(5); frame("t21-person")
    }
    Journey.dismissSheets(app)

    // --- saved colour picker (the row the operator reworked) ----------------------------------
    Journey.dismissSheets(app)
    Journey.openTab(app, "Library")
    sleep(4)
    if Journey.tap(app, labels: ["Saved"], contains: true, timeout: 10) {
      sleep(3)
      frame("t22-saved-row")
      if Journey.tap(app, labels: ["Colour", "Color"], contains: true, timeout: 8) {
        sleep(3); frame("t23-colour-popover")
      }
    }

    // --- offline, with the cache already warmed by everything above ----------------------------
    Journey.dismissSheets(app)
    if Journey.setOfflineMode(app, on: true) {
      Journey.openTab(app, "Home")
      sleep(5)
      frame("t24-home-offline")
      _ = Journey.setOfflineMode(app, on: false)
    }

    Journey.openTab(app, "Home")
    sleep(4)
    frame("t25-home-final")
  }
}
