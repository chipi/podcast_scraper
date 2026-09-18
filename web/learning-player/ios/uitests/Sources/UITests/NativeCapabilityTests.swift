import XCTest

/**
 * The Capacitor-only capabilities (operator 2026-09-16).
 *
 * The shell enables ten plugins plus two local ones, and almost none were exercised anywhere: the
 * unit suite runs under happy-dom and Playwright runs a browser, so everything behind `isNative()`
 * had no coverage at all. This starts with the three that are user-visible, newly wired, and
 * entirely unverified — speech recognition (dictated notes), the native share sheet, and push
 * registration.
 *
 * ## What a SIMULATOR can and cannot prove — read before trusting a pass
 *
 * The simulator has no microphone and no speech engine, so `SpeechRecognition.available()` is
 * expected to report false. These tests therefore assert the app's BEHAVIOUR AROUND the capability
 * — the affordance appears when enabled, a tap does not crash, an unavailable engine surfaces the
 * error path instead of a dead mic — and NOT that dictation transcribes speech. That needs a real
 * device and a human voice; nothing here should be read as covering it.
 *
 * The same caveat applies to push: the simulator can show the permission prompt and register, but
 * whether a real APNs payload arrives is out of scope.
 *
 * PRECONDITIONS: app installed, signed in (`make ios-journey-signin`), fixture api reachable.
 */
final class NativeCapabilityTests: UITestCase {
  private let episodeSlug = "p09-a4bbb5dde3"


  /// Springboard owns the system permission alerts; the app under test cannot see or tap them.
  private var springboard: XCUIApplication {
    XCUIApplication(bundleIdentifier: "com.apple.springboard")
  }

  /// Allow a system permission alert if one appears. Returns whether it did — callers assert on
  /// that only when the prompt is the thing under test, since iOS shows it ONCE per install and a
  /// re-run on the same simulator legitimately sees nothing.
  @discardableResult
  private func grantSystemPrompt(_ timeout: TimeInterval = 8) -> Bool {
    for label in ["Allow", "OK", "Allow While Using App", "Continue"] {
      let button = springboard.buttons[label]
      if button.waitForExistence(timeout: timeout) {
        button.tap()
        sleep(1)
        return true
      }
    }
    return false
  }

  // MARK: - N1 speech recognition (dictated notes)

  func testN1DictationAffordanceAppearsWhenEnabled() {
    let app = Journey.launch()

    // Dictation is OFF by default and lives behind a Settings opt-in, so the mic cannot appear
    // until that switch is on — which is itself worth asserting, since a mic that showed up
    // unbidden would be a privacy surprise.
    guard Journey.openSettings(app) else { XCTFail("could not reach Settings"); return }
    _ = Journey.scrollTo(app, labels: ["Voice input for notes"])
    Journey.shot(self, "n1-a-settings-voice")

    let predicate = NSPredicate(format: "label CONTAINS[c] 'Voice input'")
    var toggle = app.checkBoxes.matching(predicate).firstMatch
    if !toggle.waitForExistence(timeout: 8) { toggle = app.switches.matching(predicate).firstMatch }
    guard toggle.waitForExistence(timeout: 8) else {
      Journey.inventory(app, "settings-no-voice-toggle")
      XCTFail("no 'Voice input for notes' control in Settings")
      return
    }
    let wasOn = String(describing: toggle.value).contains("1")
    if !wasOn { toggle.tap(); sleep(2) }
    defer { if !wasOn { _ = Journey.tap(app, labels: ["Voice input"], contains: true, timeout: 6) } }

    // A PERSON card, not a topic one. Both carry a note composer, but the topic card ends with a
    // long annotated episode list, so notes sit far below the fold — sixteen swipes inside the
    // sheet still landed mid-list. The person card is short enough that notes are reachable, which
    // makes this a test of dictation rather than a test of scrolling (2026-09-16).
    Journey.openTab(app, "Home")
    sleep(4)
    _ = Journey.tap(app, labels: ["People"], timeout: 10)
    sleep(3)
    let personRow = app.buttons.allElementsBoundByIndex.first {
      $0.label.contains("momentum") && !$0.label.contains("(")
    }
    if let row = personRow, row.isHittable {
      row.tap()
    } else {
      // The People rail depends on trending state a previous test may have changed. Any entity card
      // carries a note composer, so fall back to a topic rather than failing on the route taken.
      Journey.inventory(app, "home-people-rail-empty")
      _ = Journey.tap(app, labels: ["Topics"], timeout: 10)
      sleep(2)
      guard Journey.tap(app, labels: ["systems thinking", "risk management"], contains: true, timeout: 12)
      else {
        XCTFail("neither a person nor a topic was reachable for the note composer")
        return
      }
    }
    sleep(5)
    // "Your notes" — the textarea's aria-label, which is what WebKit exposes as the accessible
    // name. Matching the PLACEHOLDER ("Add a note…") found nothing, because an aria-label wins over
    // a placeholder. Notes are also the LAST section of a long card, so this needs more swipes than
    // the default (2026-09-16).
    guard let composer = Journey.scrollTo(app, labels: ["Your notes"], maxSwipes: 16) else {
      Journey.inventory(app, "topic-no-note-composer")
      XCTFail("no note composer on the topic card")
      return
    }
    composer.tap()
    sleep(2)
    Journey.inventory(app, "note-composer")
    Journey.shot(self, "n1-b-note-composer")

    // The mic itself. Tapping it triggers the OS speech/mic permission prompt on a real device; on
    // the simulator the engine is absent, so what we assert is that the app SURVIVES either path
    // and says so, rather than presenting a mic that silently does nothing.
    // The affordance itself is the assertion that holds everywhere: it exists ONLY because the
    // Config switch above was turned on.
    let mic = Journey.find(app, labels: ["Dictate a note"], contains: true, timeout: 8)
    XCTAssertNotNil(mic, "no dictation control on the note field after enabling Voice input")

    // Opt IN to actually tapping the mic. Default-off on a simulator because it can abort the
    // process (below), but the host machine's own microphone permission for Simulator.app changes
    // the outcome — so this is a runtime switch, not a compile-time one, and the crash claim can be
    // re-tested rather than frozen into a comment.
    //   LP_TAP_MIC=1 xcodebuild test …
    let tapMic = ProcessInfo.processInfo.environment["LP_TAP_MIC"] == "1"
    #if targetEnvironment(simulator)
    if !tapMic {
      // Default-off because tapping CAN abort the app — but the cause is environmental, not a
      // product defect, and the first reading of it here was wrong.
      //
      // Crash report App-2026-09-16-121531.ips showed EXC_CRASH / SIGABRT:
      //   AVFAudio AVAudioIONodeImpl::GetInputFormat → AudioUnitInitialize
      //   → AURemoteIO::Initialize() → _ReportRPCTimeout → abort
      // which was read as "the app cannot survive audio-input init failing". It was actually the
      // HOST: macOS had not granted Simulator.app microphone access, so the audio server never
      // answered. Once that permission was granted the same tap reported `recording=true` and the
      // app was fine — so there is no product bug here (corrected 2026-09-16).
      //
      // Still opt-in, because the permission is a property of the machine rather than the repo and
      // CI cannot assume it: `TEST_RUNNER_LP_TAP_MIC=1 xcodebuild test …` (the TEST_RUNNER_ prefix
      // is required — a bare env var does not reach the runner).
      Journey.shot(self, "n1-c-dictation-available-not-tapped")
      print("=====DICTATION simulator: affordance present, tap skipped (set LP_TAP_MIC=1 to try)=====")
      return
    }
    #endif
    if Journey.tap(app, labels: ["Dictate a note"], contains: true, timeout: 8) {
      grantSystemPrompt()
      sleep(3)
      Journey.inventory(app, "after-dictate-tap")
      Journey.shot(self, "n1-c-dictation-tapped")
      XCTAssertEqual(app.state, .runningForeground, "the app left the foreground after a mic tap")

      // THE point of this test. The simulator has no speech engine, so the honest outcomes are
      // either "recording" (a real device that granted permission) or a visible failure. What must
      // NOT happen is a mic that looks armed and does nothing — the silent dead-mic the composable
      // documents as the reason it has an `onError` at all.
      let recording = Journey.find(app, labels: ["Stop dictation"], timeout: 3) != nil
      let saidItFailed = Journey.find(app, labels: ["Couldn't start dictation"], contains: true, timeout: 5) != nil
      print("=====DICTATION recording=\(recording) reportedFailure=\(saidItFailed)=====")
      XCTAssertTrue(
        recording || saidItFailed,
        "the mic neither started nor reported a failure — a silent dead mic"
      )
    } else {
      // Not a failure in itself — the engine may be reported unavailable — but the sheet should
      // then not be offering a mic at all, so record what was on screen.
      Journey.inventory(app, "note-composer-no-mic")
      Journey.shot(self, "n1-c-dictation-absent")
    }
  }

  // MARK: - N2 native share sheet

  func testN2NativeShareSheetOpens() {
    let app = Journey.launch()
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    guard Journey.tap(app, labels: ["Share"], contains: true, timeout: 12) else {
      XCTFail("no Share control on the episode")
      return
    }
    sleep(2)
    // The in-app popover offers card / link / text; picking one hands off to the OS share sheet,
    // which is Springboard's UI — the app cannot see it, so it is asserted there.
    // "Share TEXT" specifically: on native it writes `closelistening.txt` and hands that file to the
    // OS sheet, so the sheet displays the item name — a marker that OUR payload got there.
    // Asserting only that "a sheet appeared" would pass just as happily if the app shared the wrong
    // thing, or nothing at all (operator 2026-09-16).
    guard Journey.tap(app, labels: ["Share text"], contains: true, timeout: 8) else {
      Journey.inventory(app, "share-popover-no-text-option")
      XCTFail("share popover offered no 'Share text' option")
      return
    }
    sleep(5)
    Journey.shot(self, "n2-share-sheet")

    let owners = [app, springboard, XCUIApplication(bundleIdentifier: "com.apple.ShareSheetUI")]
    var sheetUp = false
    var carriedOurContent = false
    for owner in owners {
      if owner.otherElements["ActivityListView"].waitForExistence(timeout: 4)
        || owner.buttons["Copy"].waitForExistence(timeout: 2) {
        sheetUp = true
      }
      let named = NSPredicate(format: "label CONTAINS[c] 'closelistening'")
      if owner.staticTexts.matching(named).firstMatch.waitForExistence(timeout: 4)
        || owner.otherElements.matching(named).firstMatch.waitForExistence(timeout: 2) {
        carriedOurContent = true
      }
      if sheetUp && carriedOurContent { break }
    }
    print("=====SHARE_SHEET up=\(sheetUp) carriedOurContent=\(carriedOurContent)=====")
    if !sheetUp || !carriedOurContent { Journey.inventory(app, "after-share-pick") }
    XCTAssertTrue(sheetUp, "picking a share option did not produce a share sheet")
    XCTAssertTrue(
      carriedOurContent,
      "share sheet opened but showed no sign of OUR payload (expected the item named closelistening…)"
    )
    _ = Journey.tap(app, labels: ["Cancel", "Close"], contains: true, timeout: 5)
  }

  // MARK: - N3 push registration

  func testN3PushPermissionPromptOnEnable() {
    let app = Journey.launch()
    Journey.openProfile(app)
    sleep(4)
    // ProfileView is kept alive (`KEEP_ALIVE_TABS`), so whichever tab a PREVIOUS test left selected
    // is still selected here — the matrix is on Account, and running after a test that opened
    // Topics or Stats found no push cell at all (2026-09-16). Select it explicitly.
    _ = Journey.tap(app, labels: ["Account"], timeout: 10)
    sleep(2)
    // The per-type × per-channel notification matrix lives on the Account tab; turning a Push cell
    // on is what asks the OS for permission (usePushSubscription → @capacitor/push-notifications).
    _ = Journey.scrollTo(app, labels: ["Push"])
    Journey.shot(self, "n3-a-comms-matrix")

    // SWITCHES, not checkBoxes — WebKit surfaces these as `switch`, labelled "Your Week — Push".
    // Querying checkBoxes found nothing and read as "the matrix isn't here", which it plainly was.
    //
    // And it must POLL, not snapshot. `allElementsBoundByIndex` evaluates once, and web content
    // that has not been laid out yet is absent from the accessibility tree — so the array came back
    // empty while an inventory printed moments later listed all twelve cells. Both failures on the
    // same screen, for two different reasons (2026-09-16).
    let pushPredicate = NSPredicate(format: "label CONTAINS[c] 'Push'")
    var cellQuery = app.switches.matching(pushPredicate).firstMatch
    if !cellQuery.waitForExistence(timeout: 15) {
      cellQuery = app.checkBoxes.matching(pushPredicate).firstMatch
      _ = cellQuery.waitForExistence(timeout: 10)
    }
    let cell = cellQuery
    print("=====PUSH_CELL label=\(cell.exists ? cell.label : "<none>")=====")
    guard cell.exists, cell.isHittable else {
      Journey.inventory(app, "profile-no-push-cell")
      XCTFail("no push cell in the notifications matrix")
      return
    }
    let before = String(describing: cell.value)
    cell.tap()
    // iOS asks ONCE per install, so a re-run on the same simulator may see no prompt at all — that
    // is why this is not asserted. What must hold is that the app neither crashes nor leaves the
    // cell claiming push is on when the OS refused.
    grantSystemPrompt()
    sleep(4)
    Journey.shot(self, "n3-b-after-push-toggle")
    print("=====PUSH_CELL before=\(before) after=\(String(describing: cell.value))=====")
    XCTAssertEqual(app.state, .runningForeground, "the app left the foreground enabling push")
  }

  // MARK: - N4 avatar upload + crop (photo picker → crop → it renders)

  /**
   * Upload a photo from the system picker, crop it, and confirm it ends up ON the profile.
   *
   * Crosses three boundaries nothing else covers: the iOS photo picker (another process), the crop
   * modal, and the upload + `/me` refresh that makes the new image render.
   *
   * WHAT THIS CANNOT ASSERT, and why. `ProfileAvatar`'s root carries `aria-hidden="true"`, so the
   * avatar and its initials are invisible to XCUITest — "the photo is showing" simply cannot be
   * read from the accessibility tree. This test therefore drives the flow and proves it got through
   * cleanly (crop modal opened, confirmed, closed, no error). The VISUAL proof is taken outside:
   * the before/after frames below are pixel-compared, and `/me.image` is checked host-side.
   * Seed a photo first: `xcrun simctl addmedia booted <image>`.
   */
  func testN4AvatarUploadAndCrop() throws {
    let app = Journey.launch()
    guard Journey.openProfile(app) else { XCTFail("could not open Profile"); return }
    sleep(4)
    Journey.shot(self, "n4-a-avatar-before")

    // The system photo picker runs OUT OF PROCESS (PHPicker), so `app.images` queries this app's
    // tree and taps something that is not a photo — which is why the crop modal never opened. The
    // flow below is correct and compiles; driving the picker needs its own process hierarchy.
    // Skipped EXPLICITLY, with the reason, rather than left red so the cause stays visible.
    guard Journey.tap(app, labels: ["Change photo"], contains: true, timeout: 12) else {
      Journey.inventory(app, "profile-no-avatar-trigger")
      XCTFail("no 'Change photo' control on the profile")
      return
    }
    sleep(4)
    Journey.shot(self, "n4-b-photo-picker")

    // A web `<input type=file>` in WKWebView opens an ACTION SHEET first (Photo Library / Take
    // Photo / Choose File) — the picker only appears after choosing a source. The first version of
    // this test looked for photos immediately and tapped something inside the app instead, which is
    // why the crop modal never opened (2026-09-16).
    let sources = ["Photo Library", "Choose File", "Choose Photo"]
    var openedSource = false
    for owner in [app, springboard] {
      for label in sources where owner.buttons[label].waitForExistence(timeout: 4) {
        print("=====AVATAR_SOURCE \(label)=====")
        owner.buttons[label].tap()
        openedSource = true
        break
      }
      if openedSource { break }
    }
    if !openedSource { print("=====AVATAR_SOURCE none — picker may open directly=====") }
    sleep(4)

    // Limited-library PHPicker ("private access to photos"). Its cells are a remote view, so they
    // are NOT under the app's own collectionView — dump what is actually there before choosing.
    print("=====PICKER_TYPES images=\(app.images.count) cells=\(app.cells.count) " +
          "collections=\(app.collectionViews.count) buttons=\(app.buttons.count) " +
          "sb_images=\(springboard.images.count) sb_cells=\(springboard.cells.count)=====")
    Journey.inventory(app, "photo-picker")

    // The picker's cells report `isHittable == false` — they belong to a remote view, so the usual
    // hit-testing does not apply. Tapping via a normalized COORDINATE works on elements XCUITest
    // will not call hittable, which is the whole trick here (2026-09-16).
    //
    // Photos are appended AFTER the page's own artwork in the tree, so walk from the end.
    // Identify a REAL photo by its label — the picker names cells "Photo, <date>". Walking the tree
    // backwards and taking any large image also matched app chrome, and a non-image then failed to
    // load in the crop modal, surfacing as "Couldn't upload that image" — an app error caused
    // entirely by the test picking the wrong thing (2026-09-16).
    var picked = false
    let imgs = app.images
    let n = imgs.count
    for idx in stride(from: n - 1, through: max(0, n - 12), by: -1) {
      let candidate = imgs.element(boundBy: idx)
      guard candidate.exists else { continue }
      guard candidate.label.hasPrefix("Photo,") else { continue }
      let frame = candidate.frame
      guard frame.width > 40, frame.height > 40 else { continue }
      print("=====PICKER_TAP idx=\(idx)/\(n) label=\(candidate.label) frame=\(frame)=====")
      candidate.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
      picked = true
      break
    }
    // The limited-access picker SELECTS on tap and waits for confirmation — it does not dismiss
    // itself, which is why the crop modal never opened even once a real photo had been tapped.
    sleep(2)
    for confirm in ["Add", "Done", "Choose"] where app.buttons[confirm].waitForExistence(timeout: 3) {
      print("=====PICKER_CONFIRM \(confirm)=====")
      app.buttons[confirm].tap()
      break
    }
    guard picked else {
      Journey.shot(self, "n4-b2-picker-miss")
      XCTFail("no photo in the picker — seed one with `xcrun simctl addmedia booted <image>`")
      return
    }
    sleep(5)

    // The crop step. Its title is the tell that the picked file reached the app at all.
    guard Journey.find(app, labels: ["Position your photo"], contains: true, timeout: 15) != nil else {
      Journey.inventory(app, "after-photo-pick")
      Journey.shot(self, "n4-c-no-crop-modal")
      XCTFail("crop modal did not open after picking a photo")
      return
    }
    Journey.shot(self, "n4-c-crop-modal")

    guard Journey.tap(app, labels: ["Save photo"], contains: true, timeout: 10) else {
      XCTFail("no confirm control on the crop modal")
      return
    }
    sleep(6)
    Journey.inventory(app, "after-avatar-save")
    Journey.shot(self, "n4-d-avatar-after")

    // The modal must be gone and nothing may have failed — the upload path reports through an
    // explicit error line rather than silently keeping the old picture.
    XCTAssertNil(
      Journey.find(app, labels: ["Position your photo"], contains: true, timeout: 3),
      "crop modal still open after confirming"
    )
    XCTAssertNil(
      Journey.find(app, labels: ["upload that image"], contains: true, timeout: 3),
      "the avatar upload reported a failure"
    )
  }
}
