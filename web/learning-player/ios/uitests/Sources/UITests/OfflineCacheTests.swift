import XCTest

/**
 * The offline contract as the product actually intends it (operator 2026-09-16):
 *
 *   browse while online  →  the app caches what you saw  →  go offline  →  it shows the cached stuff.
 *
 * This exists because the first forced-offline check got that sequence WRONG: the switch was
 * flipped on a cold install that had never browsed, so there was nothing cached and every section
 * rendered empty. That is the test's fault, not the app's, and an empty result proves nothing.
 *
 * What it does legitimately expose, and what this asserts, is the COPY: with forced-offline on, the
 * app knows it is offline, yet sections render the generic
 * "Couldn't load this right now. Try again." — a retry affordance for a request the app has
 * deliberately refused to make. Offline state needs offline wording, not a retry prompt.
 *
 * The suite leaves the switch OFF whatever happens (`defer`), so a failure here cannot poison
 * every later run by stranding the app in forced-offline.
 */
final class OfflineCacheTests: UITestCase {
  private let episodeSlug = "p09-a4bbb5dde3"


  func test08BrowseThenOfflineShowsCachedContent() {
    let app = Journey.launch()

    // --- 1. WARM THE CACHE: browse the surfaces we will later assert on, online.
    Journey.shot(self, "08-a-home-online")
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    Journey.shot(self, "08-b-episode-online")
    Journey.openTab(app, "Library")
    sleep(4)
    Journey.shot(self, "08-c-library-online")
    Journey.openTab(app, "Home")
    sleep(5)

    // --- 2. GO OFFLINE via the real Config switch.
    guard Journey.setOfflineMode(app, on: true) else {
      XCTFail("could not turn Offline mode ON")
      return
    }
    defer { _ = Journey.setOfflineMode(app, on: false) }

    // --- 3. OBSERVE: cached content should be presented, not a wall of retry prompts.
    Journey.openTab(app, "Home")
    sleep(5)
    Journey.inventory(app, "offline-home")
    Journey.shot(self, "08-d-home-offline")

    // The offline banner must be up — that part already works.
    XCTAssertNotNil(
      Journey.find(app, labels: ["Offline"], contains: true, timeout: 10),
      "no offline banner while forced-offline"
    )

    // THE ASSERTION THAT MATTERS: no retry affordance while the app knows it is offline.
    let retry = Journey.find(app, labels: ["Try again"], contains: true, timeout: 5)
    if retry != nil {
      print("=====OFFLINE_COPY_DEFECT 'Try again' shown while forced-offline=====")
    }
    XCTAssertNil(
      retry,
      "offline copy defect: 'Try again' is offered for requests the app deliberately did not make"
    )

    // And the episode we browsed should still be reachable from cache.
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    Journey.inventory(app, "offline-episode")
    Journey.shot(self, "08-e-episode-offline")
  }
}
