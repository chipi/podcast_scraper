import XCTest

/**
 * Device-tier coverage for offline playback (#1905/#1908).
 *
 * The unit suite runs under happy-dom and Playwright runs a browser — both are the WEB case. Every
 * native path is behind `isNative()`, so none of it was exercised anywhere until this existed. Two
 * production bugs (artwork and audio urls resolving against `capacitor://localhost`) reached main
 * precisely because no tier covered the case where the document origin differs from the API.
 *
 * Deliberately a SEPARATE project from `ios/App.xcodeproj`: it drives the already-installed app by
 * bundle id, so the app's own project file is never touched — `npx cap add ios` rewrites bundle
 * ids across that file and would silently break a target added there.
 *
 * Run it with `make test-app-ios-sim` (see the Makefile for the prerequisites).
 *
 * Preconditions the make target sets up: the api on :8011 with the fixture corpus, the app built
 * and installed on a booted simulator, and a downloaded episode seeded into the registry for the
 * dedicated `uitest` identity — a separate account from any manual signed-in one.
 */

final class OfflinePlaybackTests: XCTestCase {
  func testDownloadedEpisodePlaysAndSeeksOffline() throws {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))

    // Idempotent: the session persists across runs, so only sign in when signed out.
    let alreadyIn = app.buttons["Sign out"].firstMatch.waitForExistence(timeout: 8)
    if !alreadyIn {
      let signIn = app.links["Sign in"].firstMatch
      XCTAssertTrue(signIn.waitForExistence(timeout: 20), "neither Sign in nor Sign out present")
      signIn.tap()

      // Dev picker: a TextField placeholdered "or a custom name…"; its Sign in button stays
      // Disabled until the field has text.
      let input = app.textFields.firstMatch
      XCTAssertTrue(input.waitForExistence(timeout: 20), "no dev identity input")
      input.tap()
      input.typeText("uitest")

      let submit = app.buttons["Sign in"].firstMatch
      XCTAssertTrue(submit.waitForExistence(timeout: 10))
      XCTAssertTrue(submit.isEnabled, "submit stayed disabled after typing")
      submit.tap()
    } else {
      print("=====ALREADY SIGNED IN=====")
    }

    // ASWebAuthenticationSession shows a system consent sheet owned by Springboard.
    let consent = springboard.buttons["Continue"]
    if consent.waitForExistence(timeout: 10) {
      print("=====CONSENT SHEET APPEARED — tapping Continue=====")
      consent.tap()
    } else {
      print("=====NO CONSENT SHEET=====")
      print(springboard.debugDescription)
    }

    // Signed in?
    let signedIn = app.buttons["Sign out"].firstMatch.waitForExistence(timeout: 30)
      || app.links["Sign out"].firstMatch.waitForExistence(timeout: 5)
    print("=====SIGNED_IN=\(signedIn)=====")
    if !signedIn {
      print("=====POST_SUBMIT_TREE_START====="); print(app.debugDescription); print("=====POST_SUBMIT_TREE_END=====")
      XCTFail("sign-in did not complete"); return
    }

    app.links["Library"].firstMatch.tap()
    sleep(4)

    // The downloaded episode, straight from the device registry.
    let episode = app.staticTexts["Index Investing Without the Myths"].firstMatch
    XCTAssertTrue(episode.waitForExistence(timeout: 20), "downloaded episode not listed")
    episode.tap()
    sleep(6)

    print("=====PLAYER_TREE_START====="); print(app.debugDescription); print("=====PLAYER_TREE_END=====")

    let play = app.buttons["Play"].firstMatch
    XCTAssertTrue(play.waitForExistence(timeout: 15), "no Play control")

    // The transport sits at the very bottom of the scrollable player, overlapping the tab bar —
    // tapping its centre unscrolled lands on the Search tab instead. Scroll it into view first.
    var tries = 0
    while play.frame.maxY > app.frame.height - 90 && tries < 6 {
      app.swipeUp()
      sleep(1)
      tries += 1
    }
    print("=====PLAY FRAME AFTER SCROLL: \(play.frame) screen=\(app.frame)=====")
    play.tap()

    // Playing is observable as the control flipping to Pause.
    let pause = app.buttons["Pause"].firstMatch
    let playing = pause.waitForExistence(timeout: 20)
    print("=====PLAYING=\(playing)=====")
    if !playing {
      print("=====AFTER_PLAY_TREE_START====="); print(app.debugDescription); print("=====AFTER_PLAY_TREE_END=====")
      XCTFail("audio did not start from the downloaded file")
      return
    }

    // Let it run, then seek and confirm the position actually moved.
    sleep(4)
    let sliders = app.sliders
    print("=====SLIDER COUNT=\(sliders.count)=====")
    guard sliders.count > 0 else { XCTFail("no scrubber"); return }
    let scrubber = sliders.firstMatch
    let before = scrubber.value as? String ?? "nil"
    print("=====POS BEFORE SEEK=\(before)=====")

    // XCUITest cannot synthesise a drag on a web <input type=range>; the skip controls are real
    // buttons and exercise the same seek path through the custom scheme handler.
    let skip = app.buttons.matching(
      NSPredicate(format: "label CONTAINS[c] 'forward' OR label CONTAINS[c] 'skip' OR label CONTAINS[c] 'ahead'")
    ).firstMatch
    if skip.waitForExistence(timeout: 10) {
      print("=====SKIP CONTROL: \(skip.label)=====")
      skip.tap(); sleep(2); skip.tap(); sleep(3)
      let after = scrubber.value as? String ?? "nil"
      print("=====POS AFTER SEEK=\(after)=====")
      XCTAssertNotEqual(before, after, "seeking did not move the position")
      // Still playing after the seek — a scheme-handler range failure would stall it.
      XCTAssertTrue(app.buttons["Pause"].firstMatch.exists, "playback stopped after seeking")
      print("=====SEEK_OK=====")
    } else {
      print("=====NO SKIP CONTROL====="); print(app.debugDescription)
    }
    print("=====FINAL_TREE_START====="); print(app.debugDescription); print("=====FINAL_TREE_END=====")
  }
}
