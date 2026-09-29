import XCTest

/**
 * The offline journey, end to end, with the api DOWN for the whole run (#1925 slice 3).
 *
 * None of this is observable from the web tiers: boot from a cached identity with no network,
 * render Library from the device, and play a downloaded episode off disk.
 *
 * PRECONDITIONS — phase 3 of `make test-ios` (`test-app-ios-sim-offline` to run it alone), and only
 * AFTER phase 1 has installed the app and signed in as the shared `simtest` account. This test
 * asserts the app boots from a STORED session with the api down; a fresh install has no stored
 * session, so it cannot pass first. XCTest runs suites alphabetically and "A" sorts before "P", so
 * sharing one invocation with OfflinePlaybackTests would put this ahead of the sign-in it depends
 * on — which is why they are separate phases rather than one `-only-testing` list.
 *
 * SCOPE NOTE — auto-advance BETWEEN two downloaded episodes is deliberately not asserted here.
 * It needs two specific downloads seeded into the registry before launch, and the simulator's
 * defaults plumbing would not hold that seed reliably: `xcrun simctl spawn defaults read` reports
 * the seeded value while the app reads the previous one, so the test asserted a state that was
 * not the state under test. Rather than let a flaky harness pretend to be coverage, the resolver
 * itself is unit-tested (`src/App.offlineAdvance.test.ts`) and this file covers the journey the
 * device can prove. Re-attempting the two-episode seed is worth a follow-up, not a blocker.
 */
final class OfflineAutoAdvanceTests: UITestCase {

  /// SHARED account, deliberately: this suite reads the downloaded QUEUE `test-app-ios-sim-download` leaves behind.
  /// Per-suite isolation (#2091) would give it an empty account and the seed would be invisible.
  override var accountIdentity: String { Self.sharedSeededIdentity }
  func testBootsAndPlaysADownloadedEpisodeWithNoNetwork() throws {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))

    // TELL the app it is offline, as well as taking its network away (operator 2026-09-24:
    // "why not just move app to offline mode via config?").
    //
    // The api being down is a real network failure, and it is what the flight looks like from
    // outside. But on its own it leaves the app GUESSING: the episode page still reaches for its
    // detail over the network, fails, and can land in an error state — so playback never starts
    // and the failure reads as "audio did not start" when nothing about audio was wrong.
    //
    // The forced-offline switch is the app's own contract for this ("behave as if offline"), so
    // flipping it exercises the path the offline experience is actually built on. Both together is
    // the honest condition: no network AND the app knowing it. Using the switch ALONE would be
    // weaker — a code path that ignored the flag and hit the network would still pass.
    //
    // `lp.forceOffline` is device-local, so it survives the relaunch below.
    _ = Journey.setOfflineMode(app, on: true, labels: profileLabels)

    // A COLD start is the point: launch() on an already-running app only activates it, so what is
    // on disk is never re-read.
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))

    // 1. CASE 1 — "I was signed in; now I am offline."
    //
    //    The requirement is that the app opens into the APP, not a login wall: Library is
    //    reachable, so the episodes already on this device are too. That is what a listener needs
    //    on a plane.
    //
    //    It does NOT assert the masthead can display the account's name. That was the old
    //    assertion (`isSignedIn(as: identity)`) and it conflated two things: "my library is here"
    //    with "the app can render my identity". Offline no login is asked for and none is
    //    possible, so gating the journey on a name failed for something the listener never needed
    //    (operator 2026-09-24).
    //
    //    CASE 2 — never signed in, offline, reaching downloads through `/offline` — is a SEPARATE
    //    test with a separate contract, in `NativeOnlySurfacesTests`. Both are offline and they are
    //    otherwise unrelated; neither stands in for the other.
    guard Journey.openTab(app, "Library") else {
      XCTFail(
        "offline boot did not reach Library — the app put a wall in front of episodes that are "
          + "already on this device. On screen: \(Journey.labelledInventory(app, limit: 14))"
      )
      return
    }

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
    //
    // TAP THE TITLE ON THE DOWNLOADED ROW, not the first matching title on the page (2026-09-27).
    //
    // `.firstMatch` assumed the downloaded episode's title appears exactly once. It does not: with
    // anything in the queue, an Up next section renders carrying the SAME titles, and the first
    // match is then a queue row whose tap goes through a path that needs the network. Offline that
    // lands on "Couldn't load this episode." while the downloaded copy sits further down — so the
    // test reported "no Play control offline" about an episode it had never opened.
    //
    // MEASURED, from the inventory this failure now prints:
    //     Offline mode is on — showing saved | Couldn't load this episode. | … | Queue (1)
    //
    // `episode` above is the download toggle ON the downloaded row, so its vertical centre
    // identifies that row. Match the title sharing it.
    let rowY = episode.frame.midY
    let titles = app.staticTexts.matching(
      NSPredicate(format: "label CONTAINS[c] 'Investing' OR label CONTAINS[c] 'Signal' OR label CONTAINS[c] 'Conversation'"))
    var openedFromDownloaded = false
    for i in 0..<titles.count {
      let candidate = titles.element(boundBy: i)
      guard candidate.exists, abs(candidate.frame.midY - rowY) < 60 else { continue }
      candidate.tap()
      openedFromDownloaded = true
      break
    }
    if !openedFromDownloaded {
      Journey.inventory(app, "offline-no-downloaded-title")
    }
    XCTAssertTrue(openedFromDownloaded, "no episode title on the downloaded row")

    let play = app.buttons["Play"].firstMatch
    if !play.waitForExistence(timeout: 20) {
      // SAY WHAT IS ON SCREEN (2026-09-27). This named the control it could not find and nothing
      // about why, so "no Play control offline" reads identically whether the episode never opened,
      // the transport shows Pause because something auto-resumed, or the tap hit the wrong row.
      Journey.inventory(app, "offline-no-play")
    }
    XCTAssertTrue(play.exists, "no Play control offline")
    var scrolls = 0
    while play.frame.maxY > app.frame.height - 90 && scrolls < 6 {
      app.swipeUp(); sleep(1); scrolls += 1
    }
    play.tap()
    // Playing, OR ALREADY FINISHED. The fixture episodes are ~6 SECONDS long, so the transport can
    // run to the end and be replaced by the auto-advance end-card before this assertion catches
    // `Pause` — the state it was waiting for has already gone by.
    //
    // The end-card is not a weaker proof here, it is a stronger one: with no network and no
    // playable local file the episode could never reach its end at all, so "NEXT · IN" means the
    // audio ran from disk. Asserting only `Pause` was asserting on a race, and it reported "audio
    // did not start" about audio that had started AND finished (measured 2026-09-24: the page
    // carried the full transport, `Queue (2)`, and `NEXT · IN 0:06`).
    let playing = app.buttons["Pause"].firstMatch.waitForExistence(timeout: 20)
    let finished = Journey.control(app, label: "NEXT · IN 0:06").exists
      || app.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'NEXT'")).firstMatch.exists
    XCTAssertTrue(
      playing || finished,
      // What is ON the page matters more than the missing Pause. "Audio isn't available" or a
      // not-downloaded notice means the SOURCE failed to resolve (a stale container URI after
      // reinstall, say); a transport still sitting on Play means the tap never landed. Those need
      // opposite fixes, and the bare assertion could not tell them apart.
      "audio neither started nor completed from the downloaded file with no network. On screen: "
        + "\(Journey.labelledInventory(app, limit: 12))"
    )

    // Device-local and survives a relaunch AND an account change — left on, it breaks every later
    // suite's network assertions for the wrong reason (the 2026-09-16 cross-suite leak).
    _ = Journey.setOfflineMode(app, on: false, labels: profileLabels)
  }
}
