import XCTest

/**
 * Shared helpers for the native journey suite (operator request 2026-09-16).
 *
 * The app is a Capacitor WebView, so every "control" XCUITest sees is web content surfaced through
 * WebKit's accessibility bridge. Two consequences shape everything here:
 *
 * 1. **`data-testid` is NOT an accessibility identifier.** WebKit exposes the accessible NAME —
 *    an `aria-label`, or the element's text, or its wrapping `<label>`. So elements are addressed
 *    by label (from `src/i18n/locales/en.json`), never by testid.
 * 2. **Element TYPE is unstable across WebKit versions.** The same `<input type=checkbox>` can
 *    surface as `.checkBox` or `.switch`; a `<button>` can surface as `.button` or `.other`. Every
 *    lookup here therefore tries several types before giving up, and dumps a compact inventory
 *    when it fails so a broken selector is diagnosable from one run instead of five.
 *
 * Screenshots are XCTAttachments with `.keepAlways`; extract them from the `.xcresult` with
 * `xcrun xcresulttool export attachments`. See the `ios-journey-shots` Makefile target.
 */
enum Journey {
  // MARK: - evidence

  /// Attach a full-screen screenshot under a stable name. `.keepAlways` so it survives a PASS.
  static func shot(_ test: XCTestCase, _ name: String) {
    let img = XCUIScreen.main.screenshot()
    let att = XCTAttachment(screenshot: img)
    att.name = name
    att.lifetime = .keepAlways
    test.add(att)
    print("=====SHOT \(name)=====")
  }

  /// Compact element inventory — labels only, capped. `app.debugDescription` is thousands of lines
  /// and unreadable in a log; this prints what a selector could actually match.
  static func inventory(_ app: XCUIApplication, _ tag: String, limit: Int = 40) {
    func labels(_ q: XCUIElementQuery, _ kind: String) {
      let found = q.allElementsBoundByIndex.prefix(limit)
        .map { $0.label.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
      guard !found.isEmpty else { return }
      print("  [\(kind)] \(found.joined(separator: " | "))")
    }
    print("=====INVENTORY \(tag)=====")
    labels(app.buttons, "button")
    labels(app.links, "link")
    // checkBoxes/switches were MISSING here, and that blindness cost a diagnosis: the notifications
    // matrix is built entirely from checkboxes, so this printed an empty inventory for a screen
    // that was plainly full of controls, and the failure read as "the app is in the wrong state"
    // when the selector was simply wrong (2026-09-16). An inventory that omits an element type is
    // worse than none — it argues for the wrong conclusion.
    labels(app.checkBoxes, "checkBox")
    labels(app.switches, "switch")
    labels(app.textFields, "field")
    labels(app.staticTexts, "text")
    print("=====INVENTORY_END \(tag)=====")
  }

  // MARK: - lookup

  /// First element matching ANY of `labels` (exact, case-insensitive), across the types web
  /// content realistically surfaces as. `contains` widens to a substring match.
  static func find(
    _ app: XCUIApplication,
    labels: [String],
    contains: Bool = false,
    timeout: TimeInterval = 15
  ) -> XCUIElement? {
    // An EMPTY label list produces an empty format string, which NSPredicate rejects with
    // NSInvalidArgumentException — surfacing as a crash mid-test instead of a plain "not found"
    // (2026-09-16). Callers legitimately pass a computed list that can come back empty.
    guard !labels.isEmpty else { return nil }
    // Escape apostrophes before they reach the predicate. A label like "Couldn't start dictation"
    // closes the single-quoted literal early and NSPredicate throws `NSInvalidArgumentException`
    // mid-test — which surfaces as a crash in the helper rather than as "not found", so it reads
    // like the app broke rather than the query (2026-09-16). Plenty of this app's copy has
    // apostrophes, so this belongs here, not at each call site.
    let clauses = labels.map { label -> String in
      let safe = label.replacingOccurrences(of: "'", with: "\\'")
      return contains ? "label CONTAINS[c] '\(safe)'" : "label ==[c] '\(safe)'"
    }
    let predicate = NSPredicate(format: clauses.joined(separator: " OR "))
    // Ordered by how often each type is the right answer for this app's markup.
    let queries: [XCUIElementQuery] = [
      app.buttons, app.links, app.staticTexts, app.checkBoxes,
      app.switches, app.otherElements, app.descendants(matching: .any),
    ]
    let deadline = Date().addingTimeInterval(timeout)
    repeat {
      for q in queries {
        let el = q.matching(predicate).firstMatch
        if el.exists && el.isHittable { return el }
      }
      for q in queries {
        let el = q.matching(predicate).firstMatch
        if el.exists { return el }
      }
      usleep(400_000)
    } while Date() < deadline
    return nil
  }

  /// Find and tap, scrolling the element clear of the bottom tab bar first (the transport/tab-bar
  /// overlap trap the playback suite documents — an unscrolled tap lands on a nav tab instead).
  @discardableResult
  static func tap(
    _ app: XCUIApplication,
    labels: [String],
    contains: Bool = false,
    timeout: TimeInterval = 15
  ) -> Bool {
    guard let el = find(app, labels: labels, contains: contains, timeout: timeout) else {
      print("=====TAP_MISS \(labels)=====")
      return false
    }
    var tries = 0
    while el.frame.maxY > app.frame.height - 90 && tries < 6 {
      app.swipeUp(); usleep(700_000); tries += 1
    }
    guard el.isHittable else {
      print("=====TAP_NOT_HITTABLE \(labels) frame=\(el.frame)=====")
      return false
    }
    el.tap()
    return true
  }

  /// Scroll down until a matching element appears, or the page stops moving.
  ///
  /// The terminating condition is the PAGE ENDING, not a swipe count. It was a fixed 8, and on
  /// 2026-09-18 two suites failed because surfaces legitimately grew past it: Downloaded moved to
  /// the end of Saved, and Profile gained a capture-stats block above "Sign out". Both reported as
  /// product failures — "the UI download did not land", "sign-in did not complete" — when the app
  /// was working perfectly and the content was simply further down.
  ///
  /// A count encodes how long a page happens to be today, so every section added anywhere silently
  /// erodes it. Stalling does not: two identical frames mean scrolling achieves nothing more,
  /// whatever the page now holds. `maxSwipes` survives only as a runaway ceiling.
  static func scrollTo(
    _ app: XCUIApplication,
    labels: [String],
    contains: Bool = true,
    maxSwipes: Int = 40
  ) -> XCUIElement? {
    var lastSignature = ""
    var stalled = 0
    for _ in 0...maxSwipes {
      if let el = find(app, labels: labels, contains: contains, timeout: 2) { return el }
      app.swipeUp()
      usleep(800_000)
      // Label AND vertical position. Labels alone assume the first twelve change as you scroll —
      // true on these surfaces today (only "Skip to content" is fixed chrome) but nowhere written
      // down, so a sticky header would silently make every page look stalled after two swipes.
      // Frames move whenever the page does, which is the thing actually being detected.
      let signature = app.staticTexts.allElementsBoundByIndex
        .prefix(12).map { "\($0.label)@\(Int($0.frame.origin.y))" }.joined(separator: "|")
      if signature == lastSignature {
        stalled += 1
        if stalled >= 2 { break }
      } else {
        stalled = 0
      }
      lastSignature = signature
    }
    if let el = find(app, labels: labels, contains: contains, timeout: 2) { return el }

    // NOT FOUND — rewind to the top before giving up.
    //
    // Scrolling is a side effect, and a failed search used to leave the app wherever it stopped,
    // which is the bottom of the surface. That broke the very next step: `isSignedIn` scrolls
    // hunting "Sign out", does not find it, and `signIn` then looks for the masthead — by then at
    // y = -623, above the viewport — and reports "neither Sign in nor Sign out present" on a
    // perfectly healthy app (2026-09-18). A search that changes where you are standing has to put
    // you back when it finds nothing.
    for _ in 0..<12 { app.swipeDown() }
    return nil
  }

  // MARK: - navigation

  /// Launch the installed app and wait for first paint + boot revalidation.
  static func launch() -> XCUIApplication {
    let app = XCUIApplication(bundleIdentifier: "app.closelistening.player")
    app.launch()
    _ = app.wait(for: .runningForeground, timeout: 30)
    sleep(7) // boot paints the device snapshot, then revalidates; assert after that lands
    return app
  }

  /// Masthead avatar → Profile. Its accessible name is the user's display name, falling back to
  /// "Your profile" when the account has no name, so both are accepted.
  @discardableResult
  static func openProfile(_ app: XCUIApplication) -> Bool {
    // TOP first. The control is the masthead avatar, so it scrolls away with the page — and an
    // element above the viewport is in the accessibility tree with a NEGATIVE y, where `tap()`
    // lands on nothing. Whatever the previous step left on screen, the header is reachable from the
    // top (2026-09-18).
    for _ in 0..<12 { app.swipeDown() }
    return tap(app, labels: ["Your profile", "simtest", "uitest"], timeout: 25)
  }

  /// Bottom tab bar.
  @discardableResult
  static func openTab(_ app: XCUIApplication, _ name: String) -> Bool {
    tap(app, labels: [name], timeout: 20)
  }

  /// Dismiss any teleported sheet/popover that is still open.
  ///
  /// The sheets (entity card, storyline, share, colour) render over everything, so one left open
  /// silently swallows every later tap — which is how a screenshot sweep loses a run of frames in
  /// the middle and reports no error at all (five frames vanished this way on 2026-09-16). Cheap
  /// and idempotent: call it between sections rather than reasoning about which sheet is up.
  /// Tap the LAST element matching one of `labels`, not the first.
  ///
  /// Cards below the top of a stack stay in the accessibility tree, so an exact-label query can
  /// match a chip on a card that is visually buried. That is how "the storyline's topic row does
  /// not open anything" was diagnosed — the tap was landing on the `8 similar topics` chip of the
  /// TOPIC card underneath, and the row was never touched (2026-09-16). Later elements are the
  /// more recently mounted ones, which is the sheet on top.
  @discardableResult
  static func tapTopmost(_ app: XCUIApplication, labels: [String], timeout: TimeInterval = 10)
    -> Bool
  {
    guard !labels.isEmpty else { return false }
    let clauses = labels.map { "label ==[c] '\($0.replacingOccurrences(of: "'", with: "\\'"))'" }
    let predicate = NSPredicate(format: clauses.joined(separator: " OR "))
    let deadline = Date().addingTimeInterval(timeout)
    repeat {
      // Pool BOTH element types before choosing. Returning on the first query that had any hit
      // re-introduced the very bug this helper exists for: the storyline's member rows are
      // RouterLinks, the topic card's chips are buttons, so "buttons first" tapped a chip on the
      // card underneath every time.
      var hits: [XCUIElement] = []
      for q in [app.buttons, app.links] {
        hits += q.matching(predicate).allElementsBoundByIndex.filter { $0.isHittable }
      }
      // Lowest on screen wins: within one stack the top card is drawn over the others, so its rows
      // sit below the buried card's chips in the layout.
      if let target = hits.max(by: { $0.frame.minY < $1.frame.minY }) {
        target.tap()
        return true
      }
      usleep(400_000)
    } while Date() < deadline
    print("=====TAP_TOPMOST_MISS \(labels)=====")
    return false
  }

  /// Whether the app chrome is reachable — i.e. nothing modal is covering the tab bar. This is the
  /// only honest test of "did the sheets close": tapping a close control proves a tap happened, not
  /// that the sheet went away.
  static func chromeReachable(_ app: XCUIApplication) -> Bool {
    for label in ["Home", "Library", "Discover"] {
      let el = app.links[label].firstMatch
      if el.exists && el.isHittable { return true }
    }
    return false
  }

  /// `rounds` is per CARD, not per stack: a three-deep deck needs one pass each, and an entity card
  /// with its own back-stack needs one per entry. Six covers the deepest chain the tour builds.
  @discardableResult
  static func dismissSheets(_ app: XCUIApplication, rounds: Int = 6) -> Bool {
    // "Back" is in the label list because an entity card renders its dismiss control as Back
    // whenever `dismissAtRoot` is false (EntityCardBody: `t('ec.back')` vs `t('ec.close')`). Such a
    // card has NO control named Close at all, so a Close-only list could never dismiss it and the
    // tour reported SHEETS_STUCK on a card that was perfectly closable.
    //
    // The old version bailed via `guard … else { return }` the moment no close control was
    // hittable — which is exactly the stuck case. On the 2026-09-16 tour the person sheet stayed
    // open, the tab bar stayed covered, and every later step missed: 21 of 24 screens shot, still
    // reported as success. Silence about a stuck modal reads just like a clean screen.
    for _ in 0..<rounds {
      if chromeReachable(app) { return true }
      if let close = find(app, labels: ["Close", "Close panel", "✕", "Cancel", "Done", "Back"],
                          contains: false, timeout: 2),
         close.isHittable {
        close.tap()
        usleep(600_000)
        continue
      }
      // The close control usually EXISTS but has scrolled out of the viewport — observed at
      // y = -619 and y = -1289, i.e. the sheet's own body was scrolled down past its header. So
      // scroll the sheet back to the top and look again; that is also what a user does.
      for _ in 0..<6 {
        app.swipeDown()
        if let close = find(app, labels: ["Close", "Close panel", "✕", "Cancel", "Done", "Back"],
                            contains: false, timeout: 1),
           close.isHittable {
          close.tap()
          usleep(600_000)
          break
        }
      }
      if chromeReachable(app) { return true }
      // Last resort: the SCRIM. Deliberately NOT nearer the top than this — a pinned 92dvh sheet
      // leaves only an 8dvh strip above it and the upper part of that is the status bar / dynamic
      // island, where a tap never reaches the web view at all (which is why dy = 0.04 did nothing).
      app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.075)).tap()
      usleep(600_000)
    }
    let clear = chromeReachable(app)
    if !clear { print("=====SHEETS_STUCK chrome unreachable after \(rounds) rounds=====") }
    return clear
  }

  /// Profile → gear → Settings.
  @discardableResult
  static func openSettings(_ app: XCUIApplication) -> Bool {
    guard openProfile(app) else { return false }
    sleep(3)
    guard tap(app, labels: ["Settings"], contains: true, timeout: 20) else { return false }
    sleep(3)
    return true
  }

  /// Drive Settings → Config → "Offline mode" to an ABSOLUTE state (idempotent: a no-op when it
  /// already matches). The switch persists to `localStorage`, which the host cannot reach, so this
  /// is the only way to set it — see ConfigOfflineToggleTests for the standalone version.
  @discardableResult
  static func setOfflineMode(_ app: XCUIApplication, on wanted: Bool) -> Bool {
    guard openSettings(app) else { print("=====OFFLINE_SET no settings====="); return false }
    let predicate = NSPredicate(format: "label CONTAINS[c] 'Offline mode'")
    var control = app.checkBoxes.matching(predicate).firstMatch
    if !control.waitForExistence(timeout: 10) { control = app.switches.matching(predicate).firstMatch }
    if !control.waitForExistence(timeout: 10) {
      control = app.descendants(matching: .any).matching(predicate).firstMatch
    }
    guard control.waitForExistence(timeout: 10) else {
      inventory(app, "settings-no-offline-control")
      return false
    }
    // Web checkboxes surface their checked state as "0"/"1" in `value`.
    let isOn = String(describing: control.value).contains("1")
    print("=====OFFLINE_SET current=\(isOn) wanted=\(wanted)=====")
    if isOn == wanted { return true }
    var tries = 0
    while control.frame.maxY > app.frame.height - 90 && tries < 6 {
      app.swipeUp(); usleep(700_000); tries += 1
    }
    control.tap()
    sleep(2)
    let now = String(describing: control.value).contains("1")
    print("=====OFFLINE_SET now=\(now)=====")
    return now == wanted
  }
}
