import XCTest

/**
 * Download an episode the way a user does — by tapping the control (#1925, decision 4).
 *
 * Every earlier device test SEEDED the registry with `xcrun simctl defaults write` and copied a
 * fixture file into place. That covered playback from disk, and nothing else: the whole path from
 * "the user taps Download" to "there is a playable file on the device" — audio-source resolution,
 * the size preflight, `Filesystem.downloadFile`, the artwork and transcript that ride along, and
 * the registry write — was never exercised anywhere. Two of the three production bugs this arc
 * found lived exactly there.
 *
 * The seed was not merely thinner, it was WRONG about the corpus: it wrote titles ("Signal Offline
 * One") that no episode has. A test asserting against invented data cannot notice the app
 * disagreeing with the server.
 *
 * PRECONDITIONS — `make test-app-ios-sim-download`, which stands up ONE origin serving both the
 * api and the episode audio. The fixture corpus stores `content.media_url` as a relative
 * `/audio/<id>.mp3`, and `resolveMediaUrl` absolutises it against the api base — so with the api
 * alone on :8011 every download 404s. That, not the defaults plumbing, is why nothing downloaded
 * here before.
 */
final class DownloadThroughUITests: XCTestCase {
  /// Real episodes of "The Drift" (p06) — titles read from the fixture corpus, not invented.
  private let firstEpisode = "Signal, Noise, and the Space Between"
  private let secondEpisode = "The Conversation About Conversations"

  func testDownloadsTwoEpisodesThroughTheUIAndQueuesThem() throws {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
    // Cold: launch() on a running app only ACTIVATES it, so a previous run could leave this one
    // deep inside a player page and the first Browse tap would be a no-op.
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))

    if !AppSession.isSignedIn(app) {
      guard AppSession.signIn(app, springboard) else {
        print("=====SIGNIN_TREE_START====="); print(app.debugDescription); print("=====SIGNIN_TREE_END=====")
        XCTFail("sign-in did not complete"); return
      }
    }

    // Downloaded in order so the QUEUE ends up [first, second] — auto-advance needs a known one,
    // and the queue button appends.
    download(app, episode: firstEpisode)
    download(app, episode: secondEpisode)

    // The end state, asserted by TITLE — unique per episode, so unlike the label-only download
    // states it cannot be satisfied by some other episode that happened to be downloaded already.
    var libTries = 0
    while !app.staticTexts["Downloaded"].exists && libTries < 6 {
      app.links["Library"].firstMatch.tap()
      _ = app.staticTexts["Downloaded"].waitForExistence(timeout: 6)
      libTries += 1
    }
    for title in [firstEpisode, secondEpisode] {
      let listed = app.staticTexts[title].firstMatch
      for _ in 0..<6 where !listed.exists { app.swipeDown() }
      var scrolls = 0
      while !listed.exists && scrolls < 8 {
        app.swipeUp()
        scrolls += 1
      }
      if !listed.exists {
        print("=====LIBRARY_TREE_START====="); print(app.debugDescription); print("=====LIBRARY_TREE_END=====")
        XCTFail("\(title) is not in the Downloaded list — the UI download did not land")
        return
      }
    }
  }

  // MARK: - steps

  private func openTheDrift(_ app: XCUIApplication) {
    let show = app.staticTexts["The Drift"].firstMatch
    // Browse -> the show. Retried because the tab bar can be mid-transition after a player push,
    // and SCROLLED because the shows list is longer than one screen — p06 sits below the fold, and
    // an element WKWebView has not rendered does not exist as far as XCUITest is concerned.
    var tries = 0
    while !show.exists && tries < 4 {
      app.links["Browse"].firstMatch.tap()
      _ = show.waitForExistence(timeout: 6)
      // Back to the TOP first. The list keeps its scroll position between visits, so a
      // swipe-up-only search walks away from a row that is sitting above the viewport — which is
      // why this passed on the first visit and failed on the second.
      for _ in 0..<6 where !show.exists { app.swipeDown() }
      var scrolls = 0
      while !show.exists && scrolls < 8 {
        app.swipeUp()
        scrolls += 1
      }
      tries += 1
    }
    if !show.exists {
      print("=====NO_SHOW_TREE_START====="); print(app.debugDescription); print("=====NO_SHOW_TREE_END=====")
      XCTFail("Browse never listed The Drift")
      return
    }
    show.tap()
  }

  /// Open one episode, tap Download, wait for the app to report it stored, and queue it.
  private func download(_ app: XCUIApplication, episode title: String) {
    // Navigate from the show every time rather than trusting Back to land there. After a player
    // push the back button can return to Browse instead, and then the row simply is not present —
    // which reads as "the episode is missing" when the truth is "we are on the wrong page".
    openTheDrift(app)

    let row = app.staticTexts[title].firstMatch
    _ = row.waitForExistence(timeout: 20)
    for _ in 0..<6 where !row.exists { app.swipeDown() }
    var scrolls = 0
    while !row.exists && scrolls < 8 {
      app.swipeUp()
      scrolls += 1
    }
    if !row.exists {
      print("=====NO_EPISODE_TREE_START====="); print(app.debugDescription); print("=====NO_EPISODE_TREE_END=====")
      XCTFail("episode not listed: \(title)")
      return
    }
    row.tap()

    // Start from NOT-downloaded whatever the device was left in. A previous run of this suite —
    // or of the old `seed-ios-offline-queue`, which seeds these very slugs — leaves the control
    // reading "Downloaded", and then this test would assert its own precondition and prove
    // nothing. Removing first is what makes it idempotent, which a device tier has to be: the
    // simulator's state persists across runs by design.
    // SCOPED to the episode page. `app.buttons[...].firstMatch` searches the whole app, and the
    // download states are label-only — so a DIFFERENT episode that is already downloaded (p05,
    // left by an earlier run) satisfied the query instantly and this test reported a success it
    // had not produced. That is the exact failure mode a device test must not have.
    let page = app.otherElements["main"]
    let already = page.buttons["Downloaded — tap to remove"].firstMatch
    if already.waitForExistence(timeout: 10) {
      already.tap()
      // Removal deletes the file and rewrites the registry; give it room, and CONFIRM the control
      // actually went back to offering a download rather than assuming the tap took.
      if !page.buttons["Download for offline"].firstMatch.waitForExistence(timeout: 30) {
        print("=====REMOVE_TREE_START====="); print(app.debugDescription); print("=====REMOVE_TREE_END=====")
        XCTFail("removing the existing download never restored the Download control for \(title)")
        return
      }
    }

    let downloadButton = page.buttons["Download for offline"].firstMatch
    if !downloadButton.waitForExistence(timeout: 20) {
      print("=====NO_DOWNLOAD_CONTROL_TREE_START====="); print(app.debugDescription)
      print("=====NO_DOWNLOAD_CONTROL_TREE_END=====")
      XCTFail("no Download control on \(title) — signed out, or the episode never loaded")
      return
    }
    downloadButton.tap()

    // The transfer is real: a few MB of fixture audio over the loopback proxy, plus artwork and
    // the transcript. "Downloaded" is the app's OWN report that the bytes are on disk and the
    // registry says so — the assertion the seed could never make.
    let done = page.buttons["Downloaded — tap to remove"].firstMatch
    if !done.waitForExistence(timeout: 90) {
      print("=====DOWNLOAD_TREE_START====="); print(app.debugDescription); print("=====DOWNLOAD_TREE_END=====")
      XCTFail("\(title) never reached the downloaded state")
      return
    }

    // Queue it, so the offline auto-advance run has somewhere to advance TO.
    let queue = page.buttons["Add to queue"].firstMatch
    if queue.waitForExistence(timeout: 10) { queue.tap() }

  }
}
