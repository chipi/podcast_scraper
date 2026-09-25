import XCTest

/**
 * The two surfaces that EXIST ONLY ON DEVICE (operator 2026-09-23).
 *
 * Both are gated on `isNative()`, so the browser tier cannot see either of them — a Playwright spec
 * would assert an empty row and an empty page and pass for the wrong reason. That is exactly the
 * shape of coverage gap this tier exists to close, and the operator's instruction when told the
 * controls were native-only: *"we actually have a native-only set of tests, so we should expand
 * that with a set of tests to test this."*
 *
 * 1. **Up next's inline download control.** `showDownload` promotes `DownloadButton` out of the ⋯
 *    and into the visible row, because "is this on the device?" is the question Up next answers —
 *    usually right before losing signal. Behind a menu, the answer was invisible.
 *
 * 2. **`/offline` — "On this device".** The only surface that works offline AND signed out. Every
 *    other route is behind the login-first guard, and signing in needs a network, so a lapsed
 *    session on a plane made already-downloaded episodes unreachable from the device holding them.
 *
 * ## What is asserted, and what deliberately is not
 *
 * Controls are found via `Journey.control`, never `app.buttons`: WebKit maps a `<button>` with
 * `aria-haspopup` to a PopUpButton and one with `aria-pressed` to a toggle, so a typed query
 * silently misses exactly the controls this suite is about.
 *
 * XCUITest reads the ACCESSIBILITY TREE, not the DOM. It can therefore see a control's NAME
 * ("Download for offline" vs "Downloaded — tap to remove") and whether it is hittable, which is all
 * the behaviour here needs. It CANNOT see colour, so the accent-when-downloaded styling is not
 * asserted — the state is carried by the accessible name as well, which is the part that matters to
 * a screen reader and the part a test can honestly check. The colour stays a screenshot-review
 * concern, the same call `CloseIcon` made for the same reason.
 *
 * PRECONDITIONS — `make test-app-ios-native`, which builds against the fixture api and runs
 * `seed-ios-download`. This suite reads those downloads, so it takes the SHARED account.
 */
final class NativeOnlySurfacesTests: UITestCase {

  /// SHARED, because this suite READS `seed-ios-download`'s registry. A per-suite identity (#2091)
  /// would give it an empty account and every assertion below would pass vacuously on "nothing is
  /// downloaded" — the failure mode this file exists to catch.
  override var accountIdentity: String { Self.sharedSeededIdentity }

  /// A real episode of "The Drift" (p06), read from the fixture corpus — the same one
  /// `DownloadThroughUITests` seeds. Invented titles are how an earlier seed stopped being able to
  /// notice the app disagreeing with the server.
  private let seeded = (slug: "p06-7217050bc6", title: "Signal, Noise, and the Space Between")

  // MARK: - 1. Up next carries the download control in the row

  func testUpNextShowsTheDownloadControlWithoutOpeningTheOverflow() throws {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))
    guard startClean(app) else { XCTFail("sign-in did not complete as \(accountIdentity)"); return }

    // Queue the seeded episode, so Up next has a row whose download state we already know.
    AppSession.openEpisode(app, slug: seeded.slug)
    if !Journey.tap(app, labels: ["Add to queue"], timeout: 10) {
      // Already queued from an earlier suite on the shared account — fine, that is the state we want.
      XCTAssertTrue(
        Journey.control(app, label: "Remove from queue").waitForExistence(timeout: 10),
        "neither queue control was reachable on the player")
    }

    // The masthead queue control (operator 2026-09-23). Asserted on the way in rather than taken
    // for granted: before it existed, `/queue` was reachable only from the player or Home's resume
    // hero, and finishing everything made the queue unreachable without starting an episode you did
    // not want to play. If this regresses, the surface under test becomes unreachable on device.
    // MATCH THE BADGE, not the bare word (2026-09-25 — this test had never completed a run).
    //
    // `App.vue` renders the masthead entry as `NavIconLink :label="t('queue.title')" :badge=
    // "queue.items.length"`, so its accessible name is "Queue" only while the queue is EMPTY. This
    // test queues an episode four lines above, so by the time it looks, the name is "Queue (1)" —
    // and `Journey.tap` defaults to `contains: false`. The test therefore changed the label it was
    // about to search for, and could never have passed. Measured: `[link] … Search | Queue (1) |
    // simtest …`.
    //
    // The prefix is "Queue (" and NOT "Queue": with `contains: true` a bare "Queue" also matches
    // the player's "Queue & recently played" BUTTON, and `find` checks buttons before links — so it
    // would tap the wrong control and open the wrong surface.
    guard Journey.tap(app, labels: ["Queue ("], contains: true, timeout: 15) else {
      Journey.inventory(app, "queue-entry-missing")
      XCTFail("the masthead queue control was not reachable"); return
    }

    // The claim: the download control is in the ROW, reachable without opening the ⋯.
    let downloadNames = ["Downloaded — tap to remove", "Download for offline"]
    let found = Journey.scrollTo(app, labels: downloadNames, contains: false)
    XCTAssertNotNil(
      found,
      "no download control on the queue row — it is still behind the ⋯, which is the bug this change fixed")

    // ...and the ⋯ is still there, because promoting download must not have REPLACED the overflow.
    XCTAssertTrue(
      Journey.control(app, label: "More actions").exists,
      "the overflow disappeared — download was meant to join the row, not take the ⋯'s place")
  }

  func testTheSeededEpisodeReadsAsDownloadedRatherThanOfferingToDownloadItAgain() throws {
    // The state half. A control that says "Download for offline" about a file already on disk is
    // the same failure as the Played filter matching nothing: the app holding the answer and
    // showing the opposite.
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))
    guard startClean(app) else { XCTFail("sign-in did not complete as \(accountIdentity)"); return }

    guard Journey.openTab(app, "Library") else { XCTFail("Library was not reachable"); return }
    guard Journey.scrollTo(app, labels: ["Downloaded"], contains: false) != nil else {
      Journey.inventory(app, "downloaded-section-missing")
      XCTFail("the Downloaded section was not reachable — seed-ios-download may not have run")
      return
    }
    XCTAssertTrue(
      app.staticTexts[seeded.title].firstMatch.waitForExistence(timeout: 10),
      "the seeded episode is missing from Downloaded")
    XCTAssertTrue(
      Journey.control(app, label: "Downloaded — tap to remove").exists,
      "a downloaded episode still offers 'Download for offline' — the state is not reaching the control")
  }

  // MARK: - 2. "On this device" — offline AND signed out

  func testOfflineAndSignedOutStillReachesTheDownloadedEpisodes() throws {
    /*
     * The operator's flight case: *"what if I was not logged in? I could not log in because I was
     * offline."* Signed out, offline, the landing page's two CTAs are both dead ends — so the link
     * to what is already on the device has to live there.
     *
     * Order matters. Offline is set FIRST, while still signed in, because the switch lives in
     * Settings and Settings is behind the guard: sign out first and there is no way back in to
     * flip it.
     */
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))
    guard startClean(app) else { XCTFail("sign-in did not complete as \(accountIdentity)"); return }

    XCTAssertTrue(Journey.setOfflineMode(app, on: true), "could not force offline mode")
    defer {
      // Device-local and survives a relaunch AND an account change — left on, it breaks every
      // later suite's network assertions for the wrong reason (the 2026-09-16 cross-suite leak).
      _ = Journey.setOfflineMode(app, on: false)
    }

    guard AppSession.signOut(app) else { XCTFail("could not sign out"); return }

    // The way in, from the one page a signed-out visitor can reach.
    guard Journey.tap(app, labels: ["Play what's downloaded"], timeout: 20) else {
      Journey.inventory(app, "offline-downloads-link-missing")
      XCTFail("the landing offered no route to the downloaded episodes while offline + signed out")
      return
    }

    XCTAssertTrue(
      app.staticTexts["On this device"].firstMatch.waitForExistence(timeout: 15),
      "the offline downloads page did not render")
    // The POINT of the page: the episodes are actually listed, not an empty shell that merely loads.
    XCTAssertTrue(
      app.staticTexts[seeded.title].firstMatch.waitForExistence(timeout: 15),
      "signed out + offline, the downloaded episode is still unreachable — the gap is not closed")
    // The operator reviews this surface by eye; it is the one screen they cannot reach in a browser.
    Journey.shot(self, "offline-on-this-device")
  }

  func testTheOfflinePageRefusesToRenderWhileONLINE() throws {
    /*
     * The privacy gate, not a nicety.
     *
     * Downloads are namespaced per account so a shared phone cannot show one person's listening
     * history to the next (#1905), and this route deliberately reads the LAST account's registry.
     * Offline that is the only door; online it would be a back-door. So online it must redirect to
     * the landing — where you can actually sign in — and this test is the boundary, not a smoke test.
     */
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))
    guard startClean(app) else { XCTFail("sign-in did not complete as \(accountIdentity)"); return }
    _ = Journey.setOfflineMode(app, on: false)
    guard AppSession.signOut(app) else { XCTFail("could not sign out"); return }

    // Signed out but ONLINE: the landing must not advertise the offline list at all.
    XCTAssertFalse(
      Journey.control(app, label: "Play what's downloaded").waitForExistence(timeout: 5),
      "the offline downloads link is offered while online, where signing in is the better answer")
    XCTAssertTrue(
      app.staticTexts["On this device"].firstMatch.exists == false,
      "the offline downloads page rendered while online")
  }
}
