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
final class DownloadThroughUITests: UITestCase {

  /// SHARED account, deliberately: this suite is the target that CREATES the seed the two offline suites consume.
  /// Per-suite isolation (#2091) would give it an empty account and the seed would be invisible.
  override var accountIdentity: String { Self.sharedSeededIdentity }
  /// Real episodes of "The Drift" (p06) — slugs and titles read from the fixture corpus, not
  /// invented. The old seed wrote titles ("Signal Offline One") that no episode has, so it could
  /// not have noticed the app disagreeing with the server.
  private let first = (slug: "p06-7217050bc6", title: "Signal, Noise, and the Space Between")
  private let second = (slug: "p06-5416bc0968", title: "The Conversation About Conversations")

  func testDownloadsTwoEpisodesThroughTheUIAndQueuesThem() throws {
    let app = XCUIApplication(bundleIdentifier: AppUnderTest.bundleId)
    let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
    // Cold: launch() on a running app only ACTIVATES it, so a previous run could leave this one
    // deep inside a player page and the first Browse tap would be a no-op.
    app.terminate()
    app.launch()
    XCTAssertTrue(app.wait(for: .runningForeground, timeout: 30))

    // Known state, then THIS suite's account (#2091). `startClean` also forces offline OFF, which
    // is device-local and therefore survives both a relaunch and an account change — the exact leak
    // that made `ConfigOfflineToggleTests` break three unrelated suites on 2026-09-16.
    guard startClean(app) else {
      print("=====SIGNIN_TREE_START====="); print(app.debugDescription); print("=====SIGNIN_TREE_END=====")
      XCTFail("sign-in did not complete as \(accountIdentity)"); return
    }
    _ = springboard

    // Downloaded in order so the QUEUE ends up [first, second] — auto-advance needs a known one,
    // and the queue button appends.
    download(app, slug: first.slug, title: first.title)
    download(app, slug: second.slug, title: second.title)

    // QUEUE THE EPISODES THIS TEST DOWNLOADED, and prove the queue holds them (2026-10-11).
    //
    // The queue step used to run on the PLAYER page and tap the first "Add to queue" it found. The
    // player has no queue control for its own episode, so that was always a "More like this"
    // recommendation's toggle: the test queued an episode it never named and its flip check
    // ("Remove from queue" anywhere on screen) could not tell. Now each title is queued from its
    // OWN row on the show page, and the queue page must list both, in the order they were added.
    openShow(app, feed: "p06")
    for ep in [first, second] { queueFromItsRow(app, title: ep.title) }
    assertQueueHolds(app, [first.title, second.title])

    // The end state, asserted by TITLE — unique per episode, so unlike the label-only download
    // states it cannot be satisfied by some other episode that happened to be downloaded already.
    var libTries = 0
    while !app.staticTexts["Downloaded"].exists && libTries < 6 {
      app.links["Library"].firstMatch.tap()
      _ = app.staticTexts["Downloaded"].waitForExistence(timeout: 6)
      libTries += 1
    }
    // Downloaded moved to the END of Saved on 2026-09-18 — it had sat second, so a native build
    // opened Saved on a device-storage list instead of the user's own captures. It is now below
    // every capture the account holds, which the old fixed swipe budget could not reach: this suite
    // failed with "the UI download did not land" when the download had landed perfectly.
    //
    // `Journey.scrollTo` now terminates on the page ending rather than a count, so it stays correct
    // wherever the section sits next. Scroll to the top first — the tab keeps its prior position.
    for title in [first.title, second.title] {
      let listed = app.staticTexts[title].firstMatch
      for _ in 0..<6 where !listed.exists { app.swipeDown() }
      _ = Journey.scrollTo(app, labels: [title], contains: false)

      if !listed.exists {
        print("=====LIBRARY_TREE_START====="); print(app.debugDescription); print("=====LIBRARY_TREE_END=====")
        XCTFail(
          "\(title) is not in the Downloaded list after scrolling to the end of Saved — the UI "
            + "download did not land"
        )
        return
      }
    }
  }

  // MARK: - steps

  /// Kept for the Library assertion, which is a LIST check rather than a lookup by id.
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

  /// Open one episode, tap Download, and wait for the app to report it stored.
  private func download(_ app: XCUIApplication, slug: String, title: String) {
    AppSession.openEpisode(app, slug: slug)
    XCTAssertTrue(
      app.staticTexts[title].firstMatch.waitForExistence(timeout: 20),
      "the deep link did not land on \(title)"
    )

    // The Download control lives in the player's OVERFLOW menu (operator 2026-09-13), not in the
    // action row — so it must be opened before it exists at all: the panel is `v-if="open"`.
    //
    // The test was looking in the right place for where the button used to be. How the scope is
    // replaced is argued at the query below.
    guard Journey.control(app, label: "More actions").waitForExistence(timeout: 20) else {
      // The failure MESSAGE carries the diagnosis, not a `print`.
      //
      // This guard used to dump `app.debugDescription` to stdout, and xcodebuild does not reliably
      // forward test stdout — three separate investigations reached "the page never rendered" from
      // a truncated log while a screenshot of the same moment showed the page rendered perfectly,
      // ⋯ included. A failure that says what WAS on screen is worth more than one that repeats what
      // was not: the useful question here is what the accessibility bridge is actually exposing,
      // since that is the only thing XCUITest can see.
      let buttons = app.buttons.allElementsBoundByIndex.prefix(25).map {
        "\($0.label)\($0.isHittable ? "" : " [not hittable]")"
      }
      XCTFail(
        "no \"More actions\" button on \(title) after 20s. "
          + "Buttons present (\(app.buttons.count)): \(buttons.joined(separator: " | ")). "
          + "Links (\(app.links.count)), otherElements (\(app.otherElements.count))."
      )
      return
    }
    pageOverflow(app).tap()

    // `role="menuitem"` maps to MenuItem, not Button — see `Journey.control`.
    let already = waitForMenuItem(app, "Downloaded — tap to remove", timeout: 10)
    if already.exists {
      already.tap()
      // Removal deletes the file and rewrites the registry; give it room, and CONFIRM the control
      // actually went back to offering a download rather than assuming the tap took.
      //
      // RE-OPEN the menu first. The comment here used to say "the menu stays open — the item
      // re-renders in place", and that is not what the component does: both call sites render
      // `<DownloadButton variant="menuitem" @activated="close" />`, so activating an item CLOSES
      // the panel by design. The control we then waited 30s for was not slow, it was off-screen.
      _ = Journey.control(app, label: "More actions").waitForExistence(timeout: 10)
      pageOverflow(app).tap()
      if !waitForMenuItem(app, "Download for offline", timeout: 30).exists {
        print("=====REMOVE_TREE_START====="); print(app.debugDescription); print("=====REMOVE_TREE_END=====")
        XCTFail("removing the existing download never restored the Download control for \(title)")
        return
      }
    }

    // UNIQUENESS instead of a container. The old scope was `app.otherElements["main"]`, which
    // cannot work now for two reasons: the control moved into the overflow menu, and that menu
    // teleports to `<body>` to escape clipped ancestors — so it is not inside `main` even when
    // open. Scoping to the panel instead would mean selecting it by `data-testid`, and nothing in
    // this suite resolves that way: every selector that works here is an accessible NAME.
    //
    // So assert there is exactly ONE, which is strictly stronger than the scope it replaces. The
    // scope existed because an unscoped `firstMatch` once matched a DIFFERENT episode that was
    // already downloaded and reported a success it had not produced — `Library → Downloaded`
    // renders these controls bare, and that view is kept alive. A count of one cannot pick wrong,
    // and a count above one says so out loud instead of guessing.
    // Type-AGNOSTIC and matched on LABEL. This was `app.buttons.matching(identifier:)`, which is
    // wrong twice over for web content: the menu variant carries `role="menuitem"` so it is not a
    // Button, and `identifier` maps from `accessibilityIdentifier`, which the DOM does not set —
    // the accessible NAME is what `aria-label` produces. See `Journey.control`.
    //
    // ABOVE THE TRANSPORT only (2026-10-11). Download became a first-level action on every episode
    // CARD, so the "More like this" cards under the player each carry a "Download for offline"
    // control for their own episode — six of them. This episode's is the one in its ⋯ menu, which
    // drops down above the transport; count those: still exactly one, and a second would say so.
    let candidates = app.descendants(matching: .any)
      .matching(NSPredicate(format: "label == %@", "Download for offline"))
      .allElementsBoundByIndex
      .filter { $0.exists && $0.isHittable && $0.frame.maxY <= aboveTransport(app) }
    if candidates.count > 1 {
      print("=====AMBIGUOUS_DOWNLOAD_TREE_START====="); print(app.debugDescription)
      print("=====AMBIGUOUS_DOWNLOAD_TREE_END=====")
      XCTFail(
        "\(candidates.count) Download controls are on screen above the transport for \(title) — "
          + "another menu is open too, so tapping would be a guess"
      )
      return
    }
    let downloadButton = waitForMenuItem(app, "Download for offline", timeout: 20)
    if !downloadButton.exists {
      print("=====NO_DOWNLOAD_CONTROL_TREE_START====="); print(app.debugDescription)
      print("=====NO_DOWNLOAD_CONTROL_TREE_END=====")
      // No longer conflated (the issue asked for this split): reaching here means the page
      // rendered AND the menu opened, so neither "signed out" nor "never loaded" applies. The
      // control is native-only and self-hides on web — if it is absent here, the app does not
      // believe it is running natively.
      XCTFail(
        "the overflow menu opened on \(title) but carries no \"Download for offline\" control. "
          + "Download-ish elements: \(app.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS[c] %@", "download")).allElementsBoundByIndex.prefix(12).map { "\($0.elementType.rawValue)|\($0.label)|\(Int($0.frame.minY))|hit=\($0.isHittable)" }). "
          + "Ceiling: \(Int(aboveTransport(app))). Menu items: \(app.menuItems.allElementsBoundByIndex.map { $0.label }). "
          + "All elements: \(app.descendants(matching: .any).count). Other menu item present: "
          + "\(app.descendants(matching: .any).matching(NSPredicate(format: "label BEGINSWITH %@", "Mark as")).count). "
          // Not a walk of every element by index: a re-render mid-walk ("No matches found for
          // Element at index 130", 2026-10-14) threw from inside this message and hid the failure
          // it was explaining. The inventory snapshots once.
          + "Labelled elements on screen: \(Journey.labelledInventory(app))"
      )
      return
    }
    downloadButton.tap()

    // The transfer is real: a few MB of fixture audio over the loopback proxy, plus artwork and
    // the transcript. "Downloaded" is the app's OWN report that the bytes are on disk and the
    // registry says so — the assertion the seed could never make.
    //
    // POLL BY RE-OPENING. Activating a menu item closes the panel (`@activated="close"`), and every
    // download STATE label lives on the control inside that panel — so waiting 90s on a closed menu
    // watches an element that cannot appear. It reported "<none of the known states>", which was
    // true and completely misleading: the control was not in a bad state, it was not on screen.
    var settled = false
    let states = [
      "Downloaded — tap to remove", "Downloading", "Waiting for Wi-Fi",
      "Waiting for a connection", "Download failed — tap to retry", "Download for offline",
    ]
    var lastSeen = "<never observed>"
    for _ in 0..<9 {
      if waitForMenuItem(app, "Downloaded — tap to remove", timeout: 2).exists {
        settled = true
        break
      }
      // Re-open and read the state. A transfer in flight keeps the panel's item on "Downloading",
      // so this doubles as the progress observation.
      if Journey.control(app, label: "More actions").waitForExistence(timeout: 5) {
        pageOverflow(app).tap()
      }
      let seen = states.filter { menuItem(app, $0).exists }
      if !seen.isEmpty { lastSeen = seen.joined(separator: ", ") }
      if menuItem(app, "Downloaded — tap to remove").exists {
        settled = true
        break
      }
      sleep(10)
    }
    if !settled {
      // WHICH state it reached, not just "not downloaded". The control's accessible name encodes
      // the whole state machine, so naming the one on screen distinguishes "queued behind the
      // Wi-Fi policy" (the simulator has no radio, so `Network.getStatus()` can report `unknown`,
      // which `downloadScheduler.allows` deliberately refuses to start on) from a real transfer
      // failure or a tap that never registered.
      XCTFail(
        "\(title) never reached the downloaded state after ~90s. "
          + "Last state observed on the control: \(lastSeen). "
          + "Labelled elements: \(Journey.labelledInventory(app, limit: 20))"
      )
      return
    }
  }

  /// The show page by deep link, top of its episode list in view.
  private func openShow(_ app: XCUIApplication, feed: String) {
    AppSession.openLink(app, "podcast/\(feed)")
    sleep(4)
  }

  /// The queue toggle of THIS title's row, by its name: a card's controls name their episode
  /// ("Add to queue: <title>", 2026-10-12), so the row's toggle is the one named for it. (Before,
  /// every card's toggle was just "Add to queue" and the row had to be found by position.)
  private func queueToggle(_ app: XCUIApplication, rowOf title: String) -> XCUIElement? {
    let named = app.descendants(matching: .any).matching(NSPredicate(
      format: "label == %@ OR label == %@", "Add to queue: \(title)", "Remove from queue: \(title)"
    )).firstMatch
    return named.exists ? named : nil
  }

  private func queueFromItsRow(_ app: XCUIApplication, title: String) {
    guard Journey.scrollTo(app, labels: [title], contains: false) != nil,
      let toggle = queueToggle(app, rowOf: title)
    else {
      XCTFail("no row with a queue toggle for \(title) on the show page. Labelled elements: "
        + "\(Journey.labelledInventory(app, limit: 20))")
      return
    }
    var target = toggle
    // Clear of the tab bar and a docked mini-player: the toggle sits a card's artwork below the
    // title, so a title just in view can leave its toggle under them, and the tap lands on a tab.
    if target.frame.maxY > app.frame.height - 180 {
      app.swipeUp()
      sleep(1)
      guard let again = queueToggle(app, rowOf: title) else {
        XCTFail("lost the row of \(title) after scrolling it clear of the tab bar"); return
      }
      target = again
    }
    if target.label.hasPrefix("Add to queue") { target.tap() }
    // THAT row's toggle, re-read: the store drives its label, so the flip is the write confirmed.
    var flipped = false
    for _ in 0..<20 where !flipped {
      flipped = queueToggle(app, rowOf: title)?.label.hasPrefix("Remove from queue") == true
      if !flipped { usleep(500_000) }
    }
    XCTAssertTrue(flipped, "queued \(title) from its row and its toggle never read \"Remove from "
      + "queue\", so the queue was not written. Labelled elements: "
      + "\(Journey.labelledInventory(app, limit: 20))")
  }

  /// Up next lists exactly what was queued, in the order it was added.
  private func assertQueueHolds(_ app: XCUIApplication, _ titles: [String]) {
    for _ in 0..<8 { app.swipeDown() }
    sleep(1)
    guard Journey.tap(app, labels: ["Queue ("], contains: true, timeout: 15) else {
      XCTFail("the masthead queue control was not reachable — the queue is empty or unreachable. "
        + "Labelled elements: \(Journey.labelledInventory(app, limit: 20))")
      return
    }
    var tops: [CGFloat] = []
    for title in titles {
      guard let row = Journey.scrollTo(app, labels: [title], contains: false, maxSwipes: 10) else {
        XCTFail("\(title) was queued from its row but is not in Up next. Labelled elements: "
          + "\(Journey.labelledInventory(app, limit: 20))")
        return
      }
      tops.append(row.frame.minY)
    }
    XCTAssertEqual(tops, tops.sorted(), "Up next holds \(titles) out of the order they were added")
  }

  /// The PAGE's ⋯ — the topmost one on screen. Every "More like this" card under the player has a
  /// ⋯ of its own with the same name (2026-10-11), and a card's menu carries no Download item (on a
  /// card, Download is a button in the row), so the first match in the tree is not good enough.
  private func aboveTransport(_ app: XCUIApplication) -> CGFloat {
    let transport = Journey.control(app, label: "Skip back 15 seconds")
    return transport.exists ? transport.frame.minY : app.frame.height / 2
  }

  private func pageOverflow(_ app: XCUIApplication) -> XCUIElement {
    let all = app.descendants(matching: .any)
      .matching(NSPredicate(format: "label == %@", "More actions"))
      .allElementsBoundByIndex
    let onScreen = all.filter { $0.exists && $0.isHittable }
    return onScreen.min(by: { $0.frame.minY < $1.frame.minY })
      ?? Journey.control(app, label: "More actions")
  }

  /// This episode's download control: the item in its ⋯ menu, found by WHERE it is.
  ///
  /// Since 2026-10-11 every episode CARD also carries a Download control for its own episode, so
  /// the "More like this" cards under the player offer the same names. The element TYPE cannot
  /// separate them: the menu's `role="menuitem"` reached XCUITest as a MenuItem on one episode and
  /// not on the next (the recording showed the menu open with the item; `app.menuItems` was empty).
  /// Position can: the menu drops down under the page's ⋯, above the player's transport, and every
  /// card sits below it. So: a matching element that is on screen, above the transport.
  /**
   * `menuItem`, re-resolved until the item is on screen or `timeout` runs out.
   *
   * `menuItem` decides once, when it is called: if the panel has not rendered yet it hands back
   * its never-matching fallback, so `waitForExistence` on the result waits on nothing. Called
   * straight after tapping ⋯ — the menu still opening — that is what happened (2026-10-14): the
   * panel plainly showed "Downloaded — tap to remove" (screen recording at 2:05-2:10) while the
   * 10s wait for it failed, the removal was skipped, and the test then waited for a "Download for
   * offline" that could not appear.
   */
  private func waitForMenuItem(
    _ app: XCUIApplication, _ label: String, timeout: TimeInterval
  ) -> XCUIElement {
    let deadline = Date().addingTimeInterval(timeout)
    var item = menuItem(app, label)
    while !item.exists && Date() < deadline {
      _ = item.waitForExistence(timeout: 0.5)
      item = menuItem(app, label)
    }
    return item
  }

  private func menuItem(_ app: XCUIApplication, _ label: String) -> XCUIElement {
    let ceiling = aboveTransport(app)
    let all = app.descendants(matching: .any)
      .matching(NSPredicate(format: "label == %@", label))
    let pick = all.allElementsBoundByIndex.first {
      $0.exists && $0.isHittable && $0.frame.maxY <= ceiling
    }
    // Not found yet: hand back a query that never matches a card, so `waitForExistence` and
    // `exists` keep meaning "the menu's item", and the caller's timeout does the waiting.
    return pick ?? app.menuItems.matching(NSPredicate(format: "label == %@", label)).firstMatch
  }
}
