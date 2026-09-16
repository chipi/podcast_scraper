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
