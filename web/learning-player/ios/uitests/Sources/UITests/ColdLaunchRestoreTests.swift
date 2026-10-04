import XCTest

/**
 * Cold-launch restore (#2278), on the device tier.
 *
 * iOS ends a backgrounded app — the whole process, or only the WebView's content process — and the
 * next open used to land on Home with an empty mini-player. The restore is native-only (`isNative()`
 * gates it), so no web tier can exercise it; this is the only place the real sequence runs:
 * load an episode, move to another screen, background the app, have it TERMINATED, launch again.
 *
 * `app.terminate()` after a Home press is the closest a test can come to a jetsam kill: the process
 * goes away while backgrounded and the next launch is cold. What it cannot reproduce is the timing
 * (iOS decides that), which is what `app_launch` (#2277) measures in prod.
 */
final class ColdLaunchRestoreTests: UITestCase {

  /// Same fixture episode as AppJourneyTests.
  private let episodeSlug = "p09-a4bbb5dde3"
  private let episodeTitle = "Risk Is a Systems Property"
  /// The Discover hub's catalogue heading (`browse.catalogTitle`) — rendered by BrowseView only.
  /// ("Discover" is also the tab label and "Trends" is on Home too, so neither says which screen.)
  private let discoverMarker = "Browse"

  /// The episode title in the MINI-PLAYER — at the bottom of the screen. Matching the title
  /// anywhere would pass on a Discover list row that happens to show the same episode.
  private func miniPlayerShows(_ app: XCUIApplication, timeout: TimeInterval) -> Bool {
    let height = app.windows.firstMatch.frame.height
    let deadline = Date().addingTimeInterval(timeout)
    repeat {
      let hits = app.descendants(matching: .any)
        .matching(NSPredicate(format: "label CONTAINS %@", episodeTitle)).allElementsBoundByIndex
      if hits.contains(where: { $0.exists && $0.frame.midY > height * 0.7 }) { return true }
      sleep(1)
    } while Date() < deadline
    return false
  }

  func testColdLaunchReopensTheLastScreenWithTheEpisodeLoaded() {
    let app = Journey.launch()
    guard startClean(app) else {
      XCTFail("sign-in did not complete as \(accountIdentity)")
      return
    }

    // 1. Load an episode into the player (PlayerView loads it on mount; no need to play).
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    XCTAssertNotNil(
      Journey.find(app, labels: [episodeTitle], contains: true, timeout: 20),
      "episode page did not render its title")

    // 2. Leave it for another screen — so the restore has a route AND an episode to bring back.
    XCTAssertTrue(Journey.openTab(app, "Discover"), "could not open Discover")
    sleep(4)
    XCTAssertNotNil(
      Journey.find(app, labels: [discoverMarker], contains: false, timeout: 15),
      "not on Discover before the kill")
    XCTAssertTrue(miniPlayerShows(app, timeout: 10), "the episode is not in the mini-player before the kill")
    Journey.shot(self, "restore-01-before-kill")

    // 3. Background, then end the process — what iOS does to an app it reclaims.
    XCUIDevice.shared.press(.home)
    sleep(3)
    app.terminate()

    // 4. Cold launch. The restore should put us back on Discover with the episode in the
    //    mini-player — not on Home with nothing loaded.
    let again = Journey.launch()
    Journey.shot(self, "restore-02-after-relaunch")
    XCTAssertNotNil(
      Journey.find(again, labels: [discoverMarker], contains: false, timeout: 15),
      "cold launch did not return to Discover. On screen: \(Journey.labelledInventory(again, limit: 10))")
    XCTAssertTrue(
      miniPlayerShows(again, timeout: 15),
      "the mini-player did not bring the episode back. On screen: \(Journey.labelledInventory(again, limit: 10))")
  }
}
