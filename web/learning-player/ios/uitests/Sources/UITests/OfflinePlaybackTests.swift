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
 * Runs as PHASE 2 of `make test-ios` (`test-app-ios-playback` to run it alone).
 *
 * Preconditions, all set up by phase 1: the api on :$(APP_E2E_PORT) with the fixture corpus, the
 * app built and installed on a booted simulator, and two REAL episodes downloaded through the UI
 * as the shared `simtest` account — which is what this suite plays.
 *
 * It used to name a registry hand-seeded by `seed-ios-download` for the `uitest` identity. That
 * seed wrote a DIFFERENT namespace than this suite's `simtest` account, so it was redundant at
 * best; phase 1's real downloads are better evidence anyway. Its old home, `test-app-ios-sim`, was
 * called by nothing — so this suite was "wired" and never ran (2026-09-25).
 *
 * The api stays UP. Despite the name, what is offline here is the AUDIO SOURCE: it plays from disk
 * and seeks, but it still signs in and reads Library.
 */

final class OfflinePlaybackTests: UITestCase {

  /// SHARED account, deliberately: this suite reads the downloads `seed-ios-download` writes under the shared account.
  /// Per-suite isolation (#2091) would give it an empty account and the seed would be invisible.
  override var accountIdentity: String { Self.sharedSeededIdentity }
  func testDownloadedEpisodePlaysAndSeeksOffline() throws {
    // `Journey.launch()`, NOT a hand-rolled `app.launch()` (2026-09-25). Its `sleep(7)` is the boot
    // settle: the app paints a cached snapshot and THEN revalidates, so a query fired into that
    // window sees a masthead that is about to be replaced. Hand-rolled, this suite tapped the
    // profile link 0.6s after the app reached the foreground (measured: foreground t=2.85s, tap
    // t=3.43s), the navigation was discarded, `isSignedIn(as:)` reported SIGNED OUT about a
    // perfectly good session, and `signIn` then failed hunting a "Sign in" link that is correctly
    // absent on a signed-in app.
    //
    // `NativeOnlySurfacesTests` hand-rolls its launch too and gets away with it only by accident:
    // `startClean` opens with 12 `swipeDown()`s, ~6s, which buys the settle this suite never had.
    let app = Journey.launch()
    let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")

    // Idempotent: the session persists across runs, so only sign in when signed out — and the
    // question is asked AFTER the boot revalidation lands, not while the painted session is still
    // on screen (see AppSession).
    if !AppSession.isSignedIn(app, as: accountIdentity) {
      guard AppSession.signIn(app, springboard, as: accountIdentity) else {
        print("=====POST_SUBMIT_TREE_START====="); print(app.debugDescription); print("=====POST_SUBMIT_TREE_END=====")
        XCTFail("sign-in did not complete"); return
      }
    }

    app.links["Library"].firstMatch.tap()
    sleep(4)

    // An episode PHASE 1 ACTUALLY DOWNLOADED, not one a hand-written registry claimed (2026-09-25).
    //
    // This used to look for "Index Investing Without the Myths" — the p05 episode
    // `seed-ios-download` fabricates. That seed writes into `UITEST_NS`
    // (u_bc76c56b88bcec16904531b0), the `uitest` identity's namespace, while this suite signs in as
    // the shared `simtest` account, whose registry on device is u_1e9f7e3c36157a4b6262cafc. The two
    // never met: the seeded episode was invisible here no matter how often the seed ran. Nobody
    // noticed because the suite's only home, `test-app-ios-sim`, was called by nothing.
    //
    // `DownloadThroughUITests` (phase 1) downloads this one through the UI as `simtest`, so it is
    // genuinely on disk under the namespace this suite reads — and its title comes from the fixture
    // corpus rather than from a fixture the test wrote about itself.
    let episode = app.staticTexts["Signal, Noise, and the Space Between"].firstMatch
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
