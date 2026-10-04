import XCTest

/**
 * Personalisation journeys (operator 2026-09-16).
 *
 * Both of these are about state the app only has AFTER the user does something, which is why the
 * first journey pass produced two empty panels and proved nothing:
 *
 *   - Stats said "Start listening to build your stats." because nothing had ever been played.
 *   - Interests (then Topics) said "No interests chosen yet." because none had been picked.
 *
 * An empty panel is only meaningful once the thing that fills it has happened. So these tests
 * PERFORM the action first (play episodes / choose interests) and then assert the panel changed —
 * and, for interests, that Home reacts, since interests are what feed its recommendations.
 */
final class PersonalisationTests: UITestCase {
  private let episodes = ["p09-a4bbb5dde3", "p07-2aceab172c"]


  // MARK: - 09 play → stats

  func test09PlayFillsStats() {
    let app = Journey.launch()
    guard startClean(app) else {
      XCTFail("sign-in did not complete as \(accountIdentity)")
      return
    }

    for (i, slug) in episodes.enumerated() {
      AppSession.openEpisode(app, slug: slug)
      sleep(6)
      // The transport sits under the tab bar until scrolled — the trap the playback suite documents.
      // Scroll to it rather than hoping it is on screen.
      _ = Journey.scrollTo(app, labels: ["Play", "Pause", "Replay"], maxSwipes: 5)
      // The episode may ALREADY be playing, or already finished: playback position is persisted
      // server-side, so a slug another test has opened comes back resumed — the page then shows a
      // "NEXT · IN 0:06" auto-advance countdown and no Play control at all, which is how this test
      // started failing (2026-09-16). What it needs is playback time accruing, and "already
      // playing" satisfies that as well as "just started" does.
      if !Journey.tap(app, labels: ["Play"], timeout: 10) {
        guard Journey.find(app, labels: ["Pause"], contains: false, timeout: 5) != nil else {
          Journey.inventory(app, "episode-no-play-\(i)")
          XCTFail("no Play control and nothing playing on \(slug)")
          continue
        }
      }
      // Let real playback time accrue; stats are driven by reported position, not by opening a page.
      sleep(12)
      Journey.shot(self, "09-playing-\(i)")
      // Pausing flushes the position to the server in the same way backgrounding would.
      _ = Journey.tap(app, labels: ["Pause"], timeout: 10)
      sleep(3)
    }

    Journey.openProfile(app, labels: profileLabels)
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

  // MARK: - 10 interests → follows + Home

  /// Profile › Interests edits in place (2026-10-04): each section offers "Follow X" suggestions, a
  /// tap follows at once and moves the item to the section's followed row, whose control is "Stop
  /// following X". No picker, no Save. Suggestions never include what is already followed, so every
  /// "Follow …" control is safe to tap — the selected-state guessing the old toggle picker forced
  /// on this test has nothing left to guess about.
  func test10InterestsRenderAndFeedHome() {
    let app = Journey.launch()
    guard startClean(app) else {
      XCTFail("sign-in did not complete as \(accountIdentity)")
      return
    }
    Journey.openProfile(app, labels: profileLabels)
    sleep(3)
    guard Journey.tap(app, labels: ["Interests"], timeout: 12) else {
      XCTFail("no Interests tab on Profile"); return
    }
    // WAIT for the sections' suggestions — they are fetched, and a snapshot taken while they load
    // finds nothing to tap. i18n: interestSections.suggested = "Suggested".
    let ready = Journey.find(app, labels: ["Suggested"], contains: false, timeout: 20) != nil
    XCTAssertTrue(ready, "the Interests tab never finished loading its suggestions")
    sleep(2)
    Journey.shot(self, "10-interests-before")

    // i18n: interestSections.follow = "Follow {name}", interestSections.remove = "Stop following {name}"
    var followed: [String] = []
    for _ in 0..<3 {
      guard let control = app.buttons.allElementsBoundByIndex.first(where: {
        $0.label.hasPrefix("Follow ") && !$0.label.hasPrefix("Follow show") && $0.isHittable
      }) else { break }
      let label = String(control.label.dropFirst("Follow ".count))
      control.tap()
      followed.append(label)
      sleep(2)
    }
    print("=====INTERESTS_FOLLOWED \(followed)=====")
    XCTAssertFalse(followed.isEmpty, "no Follow suggestion could be tapped on the Interests tab")
    for label in followed {
      XCTAssertNotNil(
        Journey.scrollTo(app, labels: ["Stop following \(label)"]),
        "'\(label)' was tapped but is not shown as followed"
      )
    }
    Journey.shot(self, "10-interests-after")

    // Interests feed Home's recommendations, so Home must REFLECT them: it stops asking.
    Journey.openTab(app, "Home")
    sleep(6)
    Journey.shot(self, "10-home-after-interests")
    XCTAssertNil(
      Journey.find(app, labels: ["Choose interests"], contains: true, timeout: 5),
      "Home is still prompting to choose interests after interests were chosen"
    )

    // And they were WRITTEN, not just flipped on screen: back on Profile they are still there.
    Journey.openProfile(app, labels: profileLabels)
    sleep(3)
    _ = Journey.tap(app, labels: ["Interests"], timeout: 10)
    sleep(3)
    XCTAssertNotNil(
      Journey.scrollTo(app, labels: followed.map { "Stop following \($0)" }),
      "none of the interests just followed (\(followed)) are on the Profile Interests tab after leaving it"
    )
  }
}
