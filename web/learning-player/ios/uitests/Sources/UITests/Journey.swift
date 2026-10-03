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
      // SAY SO when only a non-hittable match exists.
      //
      // The pass above requires `isHittable`; this one accepts mere existence, which for web content
      // is usually the inert copy — WebKit exposes each masthead control twice and only one of them
      // is interactive. `tapWithinScreen` then taps it, and its own docstring names the outcome: "a
      // silent no-op is the worst failure shape there is, because the symptom surfaces somewhere
      // else entirely as 'the control is not there'". That is what cost 2026-09-25.
      //
      // Not turned into a failure: the fallback earns its place, because an element can be genuinely
      // tappable by coordinate while reporting `isHittable == false`, and `tapWithinScreen` exists
      // precisely to handle that. The behaviour stands; the silence goes. The Android twin prints the
      // same marker from `tap`, so a "page never changed" failure has its cause in the same log on
      // either tier rather than several steps upstream.
      for q in queries {
        let el = q.matching(predicate).firstMatch
        if el.exists {
          print(
            "=====TAP_NONHITTABLE \(labels) — only a NON-hittable match exists "
              + "(frame=\(el.frame)). The tap may land on nothing; if a later step reports the "
              + "control is absent or the page did not change, this is why.====="
          )
          return el
        }
      }
      usleep(400_000)
    } while Date() < deadline
    return nil
  }

  /**
   * Find a control by its accessible LABEL, whatever XCUIElement type WebKit mapped it to.
   *
   * `app.buttons["…"]` is not enough for this app, and the way it fails is silent. WebKit maps an
   * HTML `<button>` to a Button only when it is a plain one: add `aria-haspopup` and it becomes a
   * PopUpButton, add `aria-pressed` and it becomes a toggle, and `role="menuitem"` becomes a
   * MenuItem. Every one of those is a11y-CORRECT markup, so the app is right and the query was
   * wrong — it just reported "the episode page never rendered its action row".
   *
   * Measured on the player page (2026-09-24): `app.buttons` returned 12 controls — Back, the tier
   * pill, summary, Insights, transcript, mark-moment, the transport — and NONE of favourite,
   * add-to-collection, share or ⋯, all four of which were plainly on screen. They carry
   * `aria-pressed` / `aria-haspopup`.
   *
   * `.any` is slower than a typed query; that cost is worth paying to stop a passing selector from
   * depending on which ARIA attributes a component happens to carry today.
   */
  static func control(_ app: XCUIApplication, label: String) -> XCUIElement {
    app.descendants(matching: .any)
      .matching(NSPredicate(format: "label == %@", label))
      .firstMatch
  }

  /**
   * A bounded, failure-PROOF inventory for diagnostic messages.
   *
   * This walked `descendants(matching: .any)` and blew up on its own snapshot — "No matches found
   * for Element at index 120": the tree mutates while you iterate it, and a diagnostic that can
   * fail is worse than no diagnostic, because it replaces the real failure with its own.
   *
   * So: a few CONCRETE types, small caps, and every access guarded. It is allowed to return less
   * than the whole truth. It is not allowed to throw.
   */
  static func labelledInventory(_ app: XCUIApplication, limit: Int = 8) -> String {
    // PER-TYPE caps, not one global budget. With a single budget the first type consumes it — Home
    // has a dozen buttons, so `links` never appeared, and the element actually being hunted (the
    // masthead avatar, a LINK) was invisible in every diagnostic it produced. A cap that hides the
    // thing you are looking for is worse than no cap, because it reads as evidence of absence.
    var out: [String] = []
    let sources: [(String, XCUIElementQuery)] = [
      ("button", app.buttons), ("link", app.links), ("menuItem", app.menuItems),
      ("popUp", app.popUpButtons), ("image", app.images), ("other", app.otherElements),
    ]
    for (kind, query) in sources {
      var taken = 0
      let n = min(query.count, 40)
      guard n > 0 else { continue }
      for i in 0..<n where taken < limit {
        let label = query.element(boundBy: i).label
        // UNLABELLED elements are reported, not skipped. Filtering them out was a blind spot in
        // exactly the case this helper exists for: an interactive element that IS in the tree but
        // carries no accessible name is invisible both here and to `find` (which matches on label),
        // so "absent from the inventory" was being read as "absent from the tree" — two very
        // different bugs with opposite fixes.
        out.append(label.isEmpty ? "<UNLABELLED>[\(kind)]" : "\(label)[\(kind)]")
        taken += 1
      }
    }
    return out.isEmpty ? "<nothing labelled>" : out.joined(separator: " | ")
  }

  /**
   * Everything in a REGION of the screen, identified by FRAME rather than by label.
   *
   * The masthead avatar could not be found by name, and that is precisely why name-based evidence
   * was useless: if an element is present but unnamed, a label query cannot see it, so "absent from
   * my list" never meant "absent from the tree". Position does not have that problem — the avatar
   * is top-right at ~32pt whatever it is called.
   *
   * Reports type, label (or <UNLABELLED>) and frame for every element intersecting `region`, so the
   * two hypotheses separate cleanly: something at that frame means exposed-but-unnamed; nothing
   * there means not rendered at all. Those need opposite fixes.
   */
  static func inventoryInRegion(_ app: XCUIApplication, _ region: CGRect, limit: Int = 14) -> String {
    // ATOMIC capture. Walking a query with `element(boundBy:)` and reading `.frame` raised
    // "Failed to get matching snapshot: No matches found for Element at index 25" — the tree
    // mutates while you iterate it, and an XCUITest snapshot failure is an XCTest failure, not a
    // Swift error, so it cannot be caught. It replaces the real failure with its own, which is the
    // second time a diagnostic of mine has done that.
    //
    // `debugDescription` is one snapshot of the whole tree, taken at once, and it carries the
    // element TYPE, its frame and its label — everything this needs, with nothing to go stale.
    let dump = app.debugDescription
    var out: [String] = []
    // Lines look like: "    Button, 0x…, {{338.0, 47.0}, {32.0, 32.0}}, label: 'Queue'"
    let frameRe = try? NSRegularExpression(pattern: #"\{\{([-\d.]+), ([-\d.]+)\}, \{([-\d.]+), ([-\d.]+)\}\}"#)
    for raw in dump.split(separator: "\n") {
      let line = String(raw).trimmingCharacters(in: .whitespaces)
      guard let re = frameRe,
        let m = re.firstMatch(in: line, range: NSRange(line.startIndex..., in: line))
      else { continue }
      func num(_ i: Int) -> CGFloat {
        guard let r = Range(m.range(at: i), in: line) else { return 0 }
        return CGFloat(Double(line[r]) ?? 0)
      }
      let f = CGRect(x: num(1), y: num(2), width: num(3), height: num(4))
      guard f.intersects(region), f.width > 0, f.height > 0 else { continue }
      // Keep the whole line: it already names the type and the label (or shows none at all, which
      // is exactly the case that matters here).
      out.append(line.replacingOccurrences(of: ", 0x", with: " 0x"))
      if out.count >= limit { break }
    }
    return out.isEmpty ? "<NOTHING intersecting \(region)>" : out.joined(separator: " || ")
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
    // ...and the MIRROR CASE, which was missing (2026-09-25). An element scrolled ABOVE the
    // viewport stays in the accessibility tree with a NEGATIVE y, where it is not hittable — so a
    // masthead control (profile, Queue, Search) is unreachable from any page that has been scrolled
    // down, which is most of them. Measured: `["Queue ("] frame=(265.0, -345.0, …)` and
    // `["simtest"] frame=(349.0, -619.0, …)`.
    //
    // `openProfile` already worked around this with 12 hand-rolled `swipeDown()`s. That fixed ONE
    // call site and left the shared helper wrong, which is how a1 then failed on the masthead Queue
    // control for the same reason months later. Fix it here, once, for every caller.
    var upTries = 0
    while el.frame.minY < 0 && upTries < 12 {
      app.swipeDown(); usleep(700_000); upTries += 1
    }
    // RE-RESOLVE after scrolling (2026-09-25). `el` is a QUERY (`…firstMatch`), not a snapshot:
    // every property access re-runs it. `find` only returns a HITTABLE element, so the thing we
    // scrolled into view was hittable when chosen — but scrolling changes layout, and this app
    // renders the masthead controls TWICE (measured: `Search | Queue (2) | simtest | … | Search |
    // Queue (2) | …`). After the scroll, `firstMatch` can resolve to the OTHER copy, which is not
    // the interactive one, and the tap then reports NOT_HITTABLE about an element that is plainly
    // on screen — measured `["Queue ("] frame=(265.0, 107.0, 39.0, 16.0)`, fully inside a 402pt
    // screen and still refused.
    //
    // Asking `find` again is the whole fix: it re-applies the hittable requirement against the
    // CURRENT tree and hands back whichever copy is actually usable now.
    guard let settled = find(app, labels: labels, contains: contains, timeout: 5) else {
      print("=====TAP_LOST_AFTER_SCROLL \(labels) — matched, scrolled, then no hittable match=====")
      return false
    }
    tapWithinScreen(app, settled, labels)
    return true
  }

  /// Tap a point guaranteed to be INSIDE the screen.
  ///
  /// `XCUIElement.tap()` taps the element's CENTRE. An element only partially on screen is therefore
  /// tapped at a coordinate outside the display, where the synthesized event lands on nothing — and
  /// NOTHING reports it: `exists` is true, `isHittable` is true, `tap()` returns normally, and the
  /// app simply does not react. A silent no-op is the worst failure shape there is, because the
  /// symptom surfaces somewhere else entirely as "the control is not there".
  ///
  /// That is what 2026-09-25 cost. The masthead profile link's accessible frame was driven by an
  /// unanchored `sr-only` text run — measured x=381 w=48 on a 402pt screen, centre at 405, three
  /// points past the edge. Every `openProfile` reported success and navigated nowhere, and all four
  /// `NativeOnlySurfacesTests` failed in `startClean` claiming "neither Sign in nor Sign out
  /// present" about an app that was signed in and sitting on Home.
  ///
  /// The app defect is fixed at cause (App.vue anchors the span). This exists so the NEXT one is
  /// LOUD: it re-centres into the visible part and says so, rather than tapping into space.
  private static func tapWithinScreen(
    _ app: XCUIApplication, _ el: XCUIElement, _ labels: [String]
  ) {
    let screen = app.frame
    // ONE read each: every property access re-resolves the query, and a query that has stopped
    // matching raises an XCTest failure rather than returning nil.
    let f = el.frame
    guard screen.width > 0, f.width > 0, f.height > 0 else { el.tap(); return }
    if screen.contains(CGPoint(x: f.midX, y: f.midY)) { el.tap(); return }

    let visible = f.intersection(screen)
    guard !visible.isNull, visible.width > 1, visible.height > 1 else {
      // Nothing to aim at. Tap anyway so behaviour is unchanged, but NAME it — silence here is
      // exactly what made this class of bug invisible.
      print("=====TAP_OFFSCREEN \(labels) frame=\(f) screen=\(screen) — no visible part=====")
      el.tap()
      return
    }
    let dx = (visible.midX - f.minX) / f.width
    let dy = (visible.midY - f.minY) / f.height
    print("=====TAP_RECENTRED \(labels) frame=\(f) screen=\(screen) offset=(\(dx), \(dy))=====")
    el.coordinate(withNormalizedOffset: CGVector(dx: dx, dy: dy)).tap()
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
      // HITTABLE, or keep scrolling. `find` returns a NON-hittable match through its fallback
      // branch, so accepting whatever it hands back made this return on the FIRST iteration
      // without ever swiping — which is the opposite of what a scroll-to-element helper is for.
      //
      // MEASURED 2026-09-28: `signOut` asked for "Sign out", got
      //     =====TAP_NONHITTABLE ["Sign out"] … frame=(19.0, 878.0, 364.0, 45.0)=====
      // on a ~874pt screen. The control is the LAST item on Profile by design (#1962), so it sits
      // just below the fold; this returned it unscrolled, the tap landed on nothing, and the
      // caller reported "signed in as another account and could not sign out after 3 attempts".
      // Two of four phase-4 failures were that one line.
      if let el = find(app, labels: labels, contains: contains, timeout: 2), el.isHittable {
        return el
      }
      app.swipeUp()
      usleep(800_000)
      // Label AND vertical position. Labels alone assume the first twelve change as you scroll —
      // true on these surfaces today (only "Skip to content" is fixed chrome) but nowhere written
      // down, so a sticky header would silently make every page look stalled after two swipes.
      // Frames move whenever the page does, which is the thing actually being detected.
      // BELOW THE STICKY CHROME (ported from Android's `Journey.signature()`, 2026-09-29).
      //
      // This took the first twelve staticTexts in TREE order, which on every page in this app is
      // the masthead — and the masthead is FIXED. The signature therefore never changed, every
      // surface read as stalled after two swipes, and this gave up long before reaching anything
      // below the fold. The comment directly above warns about "a sticky header" doing exactly
      // this; the warning was written and the bug was left in.
      //
      // Android met it (Settings reporting "no Offline mode row" while sitting on Settings with the
      // row three sections down, 2026-09-24), fixed it there, and its comment records that the iOS
      // twin still had it. Measured here: `signOut` on a Profile page — inventory `Change photo |
      // Account | Topics | Stats` — could not reach "Sign out", which is deliberately the LAST
      // control on that page (#1962) and so always just below the fold.
      //
      // 12%-88% excludes the masthead and the tab bar, leaving only nodes that move when the page
      // does.
      let top = app.frame.height * 0.12
      let bottom = app.frame.height * 0.88
      let signature = app.staticTexts.allElementsBoundByIndex
        .filter { $0.frame.midY > top && $0.frame.midY < bottom && !$0.label.isEmpty }
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
    let app = XCUIApplication(bundleIdentifier: AppUnderTest.bundleId)
    app.launch()
    _ = app.wait(for: .runningForeground, timeout: 30)
    sleep(7) // boot paints the device snapshot, then revalidates; assert after that lands
    return app
  }

  /// Masthead avatar → Profile, for a caller that knows WHICH account is signed in.
  ///
  /// The masthead link is named `auth.user?.name || t('profile.title')`, so its label is the
  /// ACCOUNT NAME once `/me` resolves and the generic "Your profile" only until then. The caller is
  /// the only one who knows the first of those, which is why the labels are a parameter.
  ///
  /// The label-less overload below keeps a hardcoded list, and that list is why this parameter
  /// exists: it holds `simtest` and `uitest` from when every suite shared one account. Per-suite
  /// identities landed (#2091) and the list was never updated, so a suite signed in as
  /// `appjourneytests` matched only the generic fallback — and once the name resolved, nothing.
  /// The Android twin already takes labels for exactly this reason and its own comment records the
  /// flaw as latent here (`Journey.java:499-507`).
  ///
  /// The consequence is worse than a slow timeout. `UITestCase.startClean` drives
  /// `setOfflineMode` through this, and that is the one guarantee `startClean` exists to give — so
  /// a miss here does not fail loudly, it leaves forced-offline in whatever state the previous
  /// suite left it, which is the 2026-09-16 cross-suite poisoning this base class was written to
  /// end.
  @discardableResult
  static func openProfile(_ app: XCUIApplication, labels: [String]) -> Bool {
    // TOP first. The control is the masthead avatar, so it scrolls away with the page — and an
    // element above the viewport is in the accessibility tree with a NEGATIVE y, where `tap()`
    // lands on nothing. Whatever the previous step left on screen, the header is reachable from the
    // top (2026-09-18).
    for _ in 0..<12 { app.swipeDown() }
    // DIAGNOSTIC (2026-09-25): say WHERE the control is before tapping it. A tap that "succeeds"
    // and does not navigate has two opposite causes — a frame above the viewport (negative y, the
    // hazard the swipes above exist to prevent) or a node being replaced under the tap — and
    // `TAP_MISS`/success alone cannot tell them apart. `.exists` first: reading `.frame` on a query
    // that matches nothing is an XCTest FAILURE, not a nil, and would replace the real result.
    for label in labels {
      for (kind, q) in [("link", app.links[label]), ("button", app.buttons[label])] {
        let e = q.firstMatch
        guard e.exists else { continue }
        print("=====PROFILE_CTL \(kind) '\(label)' frame=\(e.frame) hittable=\(e.isHittable)=====")
      }
    }
    return tap(app, labels: labels, timeout: 25)
  }


  // NO LABEL-LESS OVERLOADS (2026-09-29). `openProfile`, `openSettings` and `setOfflineMode`
  // each had one, and each hardcoded ["Your profile", "simtest", "uitest"] — right only while every
  // suite silently shared `simtest`. Once suites got their own accounts, every caller of those
  // overloads failed "could not reach Settings" on a Home screen showing the avatar. They were
  // found one tier run at a time (plus a FOURTH private copy in ConfigOfflineToggleTests); removing
  // the overloads makes the compiler find the rest. Pass `profileLabels` from the suite. The one
  // caller that genuinely cannot know the account, `AppSession.signOut`, discovers it from the
  // masthead instead.

  /// Bottom tab bar.
  @discardableResult
  static func openTab(_ app: XCUIApplication, _ name: String) -> Bool {
    tap(app, labels: [name], timeout: 20)
  }

  /// Run a search the way a phone does: from Discover's own search box.
  ///
  /// There is no "Search" tab and, since 2026-09-30, no magnifier in the phone header (no room for
  /// it). `openTab(app, "Search")` therefore matches nothing and returns false — a tour that used it
  /// silently lost its search frame. The box is labelled by its sr-only `<label>` ("Ask across every
  /// episode"); `searchFields` is the fallback because the input is `type="search"`.
  static func searchFromDiscover(_ app: XCUIApplication, _ query: String) -> Bool {
    guard openTab(app, "Discover") else { return false }

    // FIND THE INPUT BY TYPE, NOT BY LABEL (2026-10-03, measured).
    //
    // The label-first version matched the sr-only `<label>` — a StaticText — rather than the field
    // it names. Tapping a StaticText gives nothing keyboard focus, and `typeText` on an unfocused
    // element raises an XCTest failure that CANNOT be caught in Swift:
    //
    //     Failed to synthesize event: Neither element nor any descendant has keyboard focus.
    //     Event dispatch snapshot: StaticText, label: 'Ask across every episode'
    //
    // So this did not merely fail to search — it aborted the whole run. The screenshot tour is
    // explicitly best-effort per frame so one dead surface cannot cost the other twenty, and an
    // uncatchable throw in a helper defeats that: the sheet came back with 2 of 30 screens.
    //
    // `searchFields` first (the input is `type="search"`), then `textFields` for a build that
    // renders it plainly. The label match is gone entirely rather than kept as a fallback, because
    // it is precisely the thing that matched the wrong element.
    let search = app.searchFields.firstMatch
    let text = app.textFields.firstMatch
    let field = search.waitForExistence(timeout: 10) ? search
      : (text.waitForExistence(timeout: 4) ? text : nil)
    guard let field else { return false }

    field.tap()
    // Confirm focus BEFORE typing. Returning false leaves the caller to skip its frame and carry
    // on, which is the whole contract of a best-effort step.
    guard waitForKeyboardFocus(field, timeout: 5) else { return false }
    field.typeText(query + "\n")
    return true
  }

  /// Poll until the element actually holds keyboard focus.
  ///
  /// `tap()` returning is not the same as the field being focused — the WebView may still be
  /// settling, and typing into an unfocused element is an uncatchable XCTest failure rather than a
  /// recoverable one. So this is a guard, not a convenience.
  static func waitForKeyboardFocus(_ element: XCUIElement, timeout: TimeInterval) -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if element.value(forKey: "hasKeyboardFocus") as? Bool == true { return true }
      usleep(200_000)
    }
    return false
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

  /// Profile → gear → Settings, for a caller that knows which account is signed in.
  @discardableResult
  static func openSettings(_ app: XCUIApplication, labels: [String]) -> Bool {
    guard openProfile(app, labels: labels) else { return false }
    sleep(3)
    guard tap(app, labels: ["Settings"], contains: true, timeout: 20) else {
      // SAY WHAT PAGE WE ARE ON. `tap` logs only `TAP_MISS ["Settings"]`, which is indistinguishable
      // between "Profile opened and Settings moved" and "the Profile tap did nothing and we are
      // still on Home" — and those need opposite fixes. Without this the 2026-09-25 wedge was four
      // identical 157s failures with no way to tell which, from the log alone.
      inventory(app, "open-settings-miss")
      return false
    }
    sleep(3)
    return true
  }


  /// Drive Settings → Config → "Offline mode" to an ABSOLUTE state (idempotent: a no-op when it
  /// already matches). The switch persists to `localStorage`, which the host cannot reach, so this
  /// is the only way to set it — see ConfigOfflineToggleTests for the standalone version.
  @discardableResult
  static func setOfflineMode(_ app: XCUIApplication, on wanted: Bool, labels: [String]) -> Bool {
    guard openSettings(app, labels: labels) else {
      print("=====OFFLINE_SET no settings=====")
      return false
    }
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
    if isOn == wanted { return true }
    // RE-CHECK before touching `.frame`. Every property access re-resolves the query, and resolving
    // a query that now matches nothing is an XCTest FAILURE, not a nil — so it cannot be caught and
    // it aborts the test outright. `startClean` deliberately ignores this helper's return value,
    // which only works if the helper actually RETURNS on trouble: the caller's tolerance was being
    // defeated by a hard failure underneath it.
    //
    // It really can vanish between the two lines. On a fresh simulator the app is signed out,
    // Settings is unreachable, and the control the existence check saw belonged to a page the app
    // was already leaving (2026-09-24).
    guard control.exists else { return false }
    var tries = 0
    while control.exists && control.frame.maxY > app.frame.height - 90 && tries < 6 {
      app.swipeUp(); usleep(700_000); tries += 1
    }
    guard control.exists, control.isHittable else { return false }
    control.tap()
    sleep(2)
    guard control.exists else { return false }
    return String(describing: control.value).contains("1") == wanted
  }

}
