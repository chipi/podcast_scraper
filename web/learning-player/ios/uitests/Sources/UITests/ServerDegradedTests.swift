import XCTest

/**
 * The production incident, on device (operator report 2026-09-16).
 *
 * A hardware reboot lost prod's secrets. The services came back UP and answered, but could not
 * authenticate anyone — so the app saw a healthy-looking server that rejected everything, and
 * rendered "a weird mix of cached things and broken things" without ever detecting that the server
 * was the problem.
 *
 * Two host-side conditions have to be created around these tests, which is why they are split into
 * an arrange half and an assert half, sequenced by `make test-app-ios-server-degraded`:
 *
 *   11a  runs while the api is HEALTHY — browses so the content cache is warm.
 *   11b  runs after the api has been restarted with a DIFFERENT signing secret, i.e. every stored
 *        token is now unverifiable. Nothing about the device changes; only the server does.
 *
 * What 11b asserts is the behaviour the incident lacked:
 *   - the app NOTICES (an offline/degraded banner, driven by `offlineReason === 'server'`)
 *   - it KEEPS the cache (it must not treat a server fault as "this user signed out" and wipe it)
 *   - it does not strand the user in a signed-out-looking shell on an admitted route
 *
 * Before the fix this test fails on the first assertion: with the network up and the server merely
 * broken, `isOffline()` was false, so no banner appeared at all.
 */
final class ServerDegradedTests: XCTestCase {
  private let episodeSlug = "p09-a4bbb5dde3"
  private let episodeTitle = "Risk Is a Systems Property"

  override func setUp() {
    super.setUp()
    continueAfterFailure = true
  }

  /// ARRANGE (api healthy): browse enough surfaces that there is a cache worth preserving.
  func test11aWarmTheCacheWhileHealthy() {
    let app = Journey.launch()
    Journey.shot(self, "11-a-home-healthy")
    XCTAssertNotNil(
      Journey.find(app, labels: ["Your profile", "simtest"], timeout: 20),
      "not signed in — run `make ios-journey-signin` before this target"
    )
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    Journey.shot(self, "11-a-episode-healthy")
    Journey.openTab(app, "Library")
    sleep(4)
    Journey.openTab(app, "Home")
    sleep(5)
    Journey.shot(self, "11-a-cache-warm")
  }

  /// ASSERT (api restarted with a different secret): the app degrades instead of self-destructing.
  func test11bDegradedServerIsDetectedAndCacheSurvives() {
    let app = Journey.launch()
    sleep(8) // let the boot reads fail and the degraded signal settle
    Journey.inventory(app, "degraded-home")
    Journey.shot(self, "11-b-home-degraded")

    // 1. NOTICED. The banner is the whole point: before this work the app stayed "online" because
    //    the device radio was fine, and said nothing at all.
    XCTAssertNotNil(
      Journey.find(app, labels: ["Can't reach the server", "Offline"], contains: true, timeout: 15),
      "no degraded/offline banner while the server cannot authenticate — the app did not notice"
    )

    // 2. HONEST. A server fault must not be reported as the user's credential going bad.
    XCTAssertNil(
      Journey.find(app, labels: ["Sign in"], timeout: 5),
      "app fell back to signed-out for a SERVER fault"
    )

    // 3. CACHE SURVIVED. `refresh()` used to clear the whole content cache on the 401 that a
    //    secret-less server returned, destroying the user's offline library over someone else's
    //    outage. Something previously browsed must still be on screen.
    XCTAssertNotNil(
      Journey.scrollTo(app, labels: [episodeTitle]),
      "cached content was wiped by a server-side fault"
    )
    Journey.shot(self, "11-b-cache-survived")
  }
}
