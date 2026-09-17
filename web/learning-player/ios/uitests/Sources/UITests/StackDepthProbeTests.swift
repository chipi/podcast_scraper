import XCTest

/**
 * Drive the card stack as deep as the content allows and photograph each level.
 *
 * The requirement is not "two sheets can overlap" — it is that the chain keeps working, each card
 * below staying readable by its title, however far you go (operator 2026-09-16: "if there's such a
 * navigation that you can do five, ten times, that's how it should go"). Two levels can be faked by
 * a single is-nested flag; four cannot.
 *
 * Entry is the player's Insights panel, which is also the hardest case: the panel is a modal
 * `<dialog>` in the browser's top layer, so anything opened from it has to be teleported INTO that
 * dialog or it renders invisibly underneath.
 */
final class StackDepthProbeTests: XCTestCase {
  private let episodeSlug = "p09-a4bbb5dde3"
  private let people = ["Dr. Elena Fischer", "Sam", "Skanda Amarnath", "Alex Morgan"]

  override func setUp() {
    super.setUp()
    continueAfterFailure = true
  }

  func testStackFourDeep() {
    let app = Journey.launch()
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)

    guard Journey.tap(app, labels: ["Insights"], contains: true, timeout: 15) else {
      XCTFail("could not open the knowledge panel"); return
    }
    sleep(3)

    let topicPre = NSPredicate(
      format: "label CONTAINS[c] 'Open systems thinking' OR label CONTAINS[c] 'Open risk management'"
    )
    if !app.buttons.matching(topicPre).firstMatch.waitForExistence(timeout: 4) {
      _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 12)
      sleep(3)
    }
    guard app.buttons.matching(topicPre).firstMatch.waitForExistence(timeout: 15) else {
      Journey.inventory(app, "probe-no-topic")
      XCTFail("no topic control in the insights panel"); return
    }

    // L1 — the topic, in the panel.
    app.buttons.matching(topicPre).firstMatch.tap()
    sleep(4)
    Journey.shot(self, "s1-topic")

    // L2 — its storyline, stacked on the topic.
    _ = Journey.scrollTo(app, labels: ["Part of a storyline"], maxSwipes: 8)
    guard Journey.tap(app, labels: ["Managing risk across domains"], contains: true, timeout: 12)
    else {
      Journey.inventory(app, "probe-no-storyline")
      XCTFail("storyline row not tappable"); return
    }
    sleep(4)
    Journey.shot(self, "s2-storyline-on-topic")

    // L3 — from inside the storyline sheet. BOTH its member-topic rows and its Related-people rows
    // must open a card; they go through the same `openEntity`. Dump what is on screen before and
    // after so a dead tap is distinguishable from a tap that hit nothing.
    _ = Journey.scrollTo(app, labels: ["Topics discussed together"], maxSwipes: 8)
    let tappedL3 = Journey.tapTopmost(app, labels: ["risk management", "safety practices"])
    print("=====L3_TAPPED \(tappedL3)=====")
    sleep(4)
    if tappedL3 {
      Journey.shot(self, "s3-topic-on-storyline")

      // L4 — a person from THAT topic's top voices. Four cards, kinds alternating, so the ladder
      // cannot be a special case for one pairing.
      _ = Journey.scrollTo(app, labels: ["Top voices"], maxSwipes: 6)
      if Journey.tapTopmost(app, labels: people) {
        sleep(4)
        Journey.shot(self, "s4-person-on-topic")
      } else {
        Journey.inventory(app, "probe-no-l4-person")
      }
    } else {
      Journey.inventory(app, "probe-no-l3-topic")
    }
  }
}

/**
 * The "Host of" prose links must actually reach the show page.
 *
 * Worth its own test because the control does two things at once: it emits `close` on the sheet AND
 * navigates. That exact pairing is what sent episode rows to Home instead of the episode — the
 * close owed a `router.back()` that ran before the route settled and popped the push (#2004). A
 * link that looks right and lands somewhere else is invisible in a screenshot.
 */
final class HostShowLinkTests: XCTestCase {
  private let episodeSlug = "p09-a4bbb5dde3"

  override func setUp() {
    super.setUp()
    continueAfterFailure = true
  }

  func testHostShowLinkOpensTheShow() {
    let app = Journey.launch()
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)

    guard Journey.tap(app, labels: ["Insights"], contains: true, timeout: 15) else {
      XCTFail("could not open the knowledge panel"); return
    }
    sleep(3)
    // The accordion REMEMBERS its state, so a blind tap CLOSES a section a previous test left
    // open — which is exactly how this failed with "no person to open" while test04 passed. Tap
    // only when the controls are not already reachable, and re-tap once if the first tap closed it.
    let personPre = NSPredicate(
      format: "label CONTAINS[c] 'Open Dr. Elena Fischer' OR label CONTAINS[c] 'Open Sam'"
    )
    if !app.buttons.matching(personPre).firstMatch.waitForExistence(timeout: 4) {
      _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 15)
      sleep(3)
      if !app.buttons.matching(personPre).firstMatch.waitForExistence(timeout: 6) {
        _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 10)
        sleep(3)
      }
    }
    _ = Journey.scrollTo(app, labels: ["Open Dr. Elena Fischer", "Open Sam"])
    guard Journey.tap(app, labels: ["Open Dr. Elena Fischer", "Open Sam"], contains: true, timeout: 10)
    else {
      Journey.inventory(app, "no-person-row"); XCTFail("no person to open"); return
    }
    sleep(5)

    // The show title inside the "Host of" sentence.
    _ = Journey.scrollTo(app, labels: ["Host of"], maxSwipes: 6)
    guard Journey.tapTopmost(app, labels: ["Cross-Show", "The Drift", "The Long View"]) else {
      Journey.inventory(app, "no-host-show-link")
      XCTFail("no host-show link in the person card"); return
    }
    sleep(6)
    Journey.shot(self, "h1-after-host-show-tap")
    Journey.inventory(app, "after-host-show-tap")

    // The show page states its own episode list — and critically we are NOT on Home, which is
    // where a close-then-back race lands.
    XCTAssertNotNil(
      Journey.find(app, labels: ["Episodes", "Latest", "Follow"], contains: true, timeout: 12),
      "tapping a hosted show did not land on a show page"
    )
    XCTAssertNil(
      Journey.find(app, labels: ["Continue listening", "Jump back in"], contains: true, timeout: 3),
      "landed on HOME — the sheet's close raced the navigation (the #2004 trap)"
    )
  }
}


/// Photograph the Boards tab with its reorder handles + cover thumbnails (CO.7).
final class BoardsShotTests: XCTestCase {
  func testBoardsTab() {
    let app = Journey.launch()
    Journey.openTab(app, "Library")
    sleep(4)
    _ = Journey.tap(app, labels: ["Boards"], contains: true, timeout: 12)
    sleep(4)
    Journey.shot(self, "b1-boards")
    // Grid ("board") view — the reordered sequence must read the same here as in the list.
    if Journey.tap(app, labels: ["Grid view", "Board view", "Tiles"], contains: true, timeout: 6) {
      sleep(3)
      Journey.shot(self, "b2-boards-grid")
    } else {
      Journey.inventory(app, "no-grid-toggle")
    }
  }
}
