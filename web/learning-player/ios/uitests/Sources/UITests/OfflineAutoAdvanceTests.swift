import XCTest

/**
 * The offline journey, end to end, with the api DOWN for the whole run (#1925 slice 3).
 *
 * None of this is observable from the web tiers: boot from a cached identity with no network,
 * render Library from the device, and play a downloaded episode off disk.
 *
 * SCOPE NOTE — auto-advance BETWEEN two downloaded episodes is deliberately not asserted here.
 * It needs two specific downloads seeded into the registry before launch, and the simulator's
 * defaults plumbing would not hold that seed reliably: `xcrun simctl spawn defaults read` reports
 * the seeded value while the app reads the previous one, so the test asserted a state that was
 * not the state under test. Rather than let a flaky harness pretend to be coverage, the resolver
 * itself is unit-tested (`src/App.offlineAdvance.test.ts`) and this file covers the journey the
 * device can prove. Re-attempting the two-episode seed is worth a follow-up, not a blocker.
 */
final class OfflineAutoAdvanceTests: XCTestCase {
  func testBootsAndPlaysADownloadedEpisodeWithNoNetwork() throws {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    // A COLD start is the point: launch() on an already-running app only activates it, so what is
    // on disk is never re-read.
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))

    // 1. Boot keeps the session with no network — the persisted identity, not a login bounce.
    XCTAssertTrue(
      app.buttons["Sign out"].firstMatch.waitForExistence(timeout: 30),
      "offline boot did not keep the session — the app fell back to signed-out"
    )

    // 2. The Downloaded list renders from the device registry, with zero successful requests.
    let downloadedHeading = app.staticTexts["Downloaded"]
    var navTries = 0
    while !downloadedHeading.exists && navTries < 6 {
      app.links["Library"].firstMatch.tap()
      _ = downloadedHeading.waitForExistence(timeout: 6)
      navTries += 1
    }
    XCTAssertTrue(downloadedHeading.exists, "Downloaded section did not render offline")

    // Whichever episode is downloaded — the journey is the assertion, not a specific slug.
    let episode = app.buttons["Downloaded — tap to remove"].firstMatch
    XCTAssertTrue(episode.waitForExistence(timeout: 15), "no downloaded episode listed offline")

    // 3. It opens and plays off disk.
    app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] 'Investing' OR label CONTAINS[c] 'Signal' OR label CONTAINS[c] 'Conversation'"))
      .firstMatch.tap()

    let play = app.buttons["Play"].firstMatch
    XCTAssertTrue(play.waitForExistence(timeout: 20), "no Play control offline")
    var scrolls = 0
    while play.frame.maxY > app.frame.height - 90 && scrolls < 6 {
      app.swipeUp(); sleep(1); scrolls += 1
    }
    play.tap()
    XCTAssertTrue(
      app.buttons["Pause"].firstMatch.waitForExistence(timeout: 20),
      "audio did not start from the downloaded file with no network"
    )
  }
}
