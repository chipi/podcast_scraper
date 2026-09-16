import XCTest

/**
 * Personalisation journeys (operator 2026-09-16).
 *
 * Both of these are about state the app only has AFTER the user does something, which is why the
 * first journey pass produced two empty panels and proved nothing:
 *
 *   - Stats said "Start listening to build your stats." because nothing had ever been played.
 *   - Topics said "No interests chosen yet." because no interests had ever been picked.
 *
 * An empty panel is only meaningful once the thing that fills it has happened. So these tests
 * PERFORM the action first (play episodes / choose interests) and then assert the panel changed —
 * and, for interests, that Home reacts, since interests are what feed its recommendations.
 */
final class PersonalisationTests: XCTestCase {
  private let episodes = ["p09-a4bbb5dde3", "p07-2aceab172c"]

  override func setUp() {
    super.setUp()
    continueAfterFailure = true
  }

  // MARK: - 09 play → stats

  func test09PlayFillsStats() {
    let app = Journey.launch()

    for (i, slug) in episodes.enumerated() {
      AppSession.openEpisode(app, slug: slug)
      sleep(6)
      // The transport sits under the tab bar until scrolled — the trap the playback suite documents.
      guard Journey.tap(app, labels: ["Play"], timeout: 15) else {
        Journey.inventory(app, "episode-no-play-\(i)")
        XCTFail("no Play control on \(slug)")
        continue
      }
      // Let real playback time accrue; stats are driven by reported position, not by opening a page.
      sleep(12)
      Journey.shot(self, "09-playing-\(i)")
      // Pausing flushes the position to the server in the same way backgrounding would.
      _ = Journey.tap(app, labels: ["Pause"], timeout: 10)
      sleep(3)
    }

    Journey.openProfile(app)
    sleep(3)
    _ = Journey.tap(app, labels: ["Stats"], timeout: 12)
    sleep(4)
    Journey.inventory(app, "profile-stats-after-play")
    Journey.shot(self, "09-profile-stats-after-play")

    XCTAssertNil(
      Journey.find(app, labels: ["Start listening to build your stats"], contains: true, timeout: 5),
      "Stats still shows its never-listened empty state after playing two episodes"
    )
  }

  // MARK: - 10 interests → chips + Home

  func test10InterestsRenderAndFeedHome() {
    let app = Journey.launch()
    Journey.openProfile(app)
    sleep(3)
    guard Journey.tap(app, labels: ["Topics"], timeout: 12) else {
      XCTFail("no Topics tab on Profile"); return
    }
    sleep(3)
    Journey.shot(self, "10-interests-before")

    guard Journey.tap(app, labels: ["Edit"], contains: true, timeout: 12) else {
      Journey.inventory(app, "interests-no-edit")
      XCTFail("no Edit control on the interests card"); return
    }
    sleep(4)
    Journey.inventory(app, "interests-picker")
    Journey.shot(self, "10-interests-picker")

    // Tap whatever the picker ACTUALLY offers, rather than guessing labels. The chips are named
    // after semantic CLUSTERS, not after the topic names on the trending rail — so the earlier
    // ["systems thinking", …] taps all missed, no interest was ever chosen, and the assertion below
    // then "failed" about a state the test had never created (2026-09-16).
    // WAIT for the picker's content. It fetches clusters, so the first snapshot of the button
    // array caught the sheet still on "Loading topics…" and came back empty — the same
    // snapshot-before-layout trap that cost a diagnosis on the notifications matrix.
    let ready = Journey.find(app, labels: ["Choose your interests"], contains: true, timeout: 20) != nil
    XCTAssertTrue(ready, "the interests picker never finished loading its topics")
    sleep(2)
    Journey.inventory(app, "interests-picker-loaded")

    let chrome: Set<String> = ["Close", "Cancel", "Save", "Done", "Skip", "Topics", "Storylines"]
    // Chips carry `aria-pressed`, which WebKit surfaces as a TOGGLE, not a button — so scanning
    // `app.buttons` found only the sheet's Cancel/Save and reported "nothing tappable" for a picker
    // full of chips. Exactly the push-matrix lesson: never assume the element type (2026-09-16).
    let chips = (app.switches.allElementsBoundByIndex
      + app.checkBoxes.allElementsBoundByIndex
      + app.buttons.allElementsBoundByIndex).filter {
      let l = $0.label.trimmingCharacters(in: .whitespacesAndNewlines)
      return !l.isEmpty && !chrome.contains(l) && $0.isHittable
    }
    print("=====INTERESTS_CHIPS \(chips.prefix(8).map { $0.label })=====")
    var picked = 0
    var chosen: [String] = []
    for chip in chips.prefix(3) {
      chosen.append(chip.label)
      chip.tap()
      picked += 1
      sleep(1)
    }
    print("=====INTERESTS_PICKED \(picked) \(chosen)=====")
    XCTAssertGreaterThan(picked, 0, "the interests picker offered nothing tappable")

    // Persist — the control is a Save/Done depending on the picker's state.
    _ = Journey.tap(app, labels: ["Save", "Done", "Save interests"], contains: true, timeout: 10)
    sleep(4)
    Journey.inventory(app, "interests-after")
    Journey.shot(self, "10-interests-after")

    XCTAssertNil(
      Journey.find(app, labels: ["No interests chosen yet"], contains: true, timeout: 5),
      "interests card still shows its empty state after choosing interests"
    )

    // Interests feed Home's recommendations, so Home must REFLECT them — the first cut of this test
    // only screenshotted Home afterwards, which proves nothing. Two observable consequences:
    // the "personalize your Home" prompt stops asking, and a chosen interest appears in the feed.
    Journey.openTab(app, "Home")
    sleep(6)
    Journey.inventory(app, "home-after-interests")
    Journey.shot(self, "10-home-after-interests")

    XCTAssertNil(
      Journey.find(app, labels: ["Choose interests"], contains: true, timeout: 5),
      "Home is still prompting to choose interests after interests were chosen"
    )
    // NOT asserted: that a chosen label appears verbatim on Home. Interests are semantic CLUSTERS
    // ("Show Themes"), while the Home rails list the TOPICS inside them ("systems thinking") — so
    // matching the cluster label on Home tests a relationship the product does not claim. The
    // honest Home-side consequence is the one asserted above: it stops asking you to choose.
    //
    // Where they DO render verbatim is the Profile Topics tab, which is what "see if they render"
    // means — so that is checked at the source.
    Journey.openProfile(app)
    sleep(3)
    _ = Journey.tap(app, labels: ["Topics"], timeout: 10)
    sleep(3)
    Journey.inventory(app, "profile-topics-after-pick")
    Journey.shot(self, "10-profile-topics-after-pick")
    XCTAssertNotNil(
      Journey.scrollTo(app, labels: chosen),
      "none of the interests just chosen (\(chosen)) render on the Profile Topics tab"
    )
  }
}
