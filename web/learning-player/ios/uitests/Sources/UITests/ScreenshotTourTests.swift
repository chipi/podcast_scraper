import XCTest

/**
 * A full visual sweep of the app's surfaces, for eyeballing the whole product at once
 * (operator request 2026-09-16 — the shots are stitched into a single contact sheet by
 * `make ios-contact-sheet`).
 *
 * Deliberately NOT assertion-heavy: this is a CAMERA, not a gate. The journey suites
 * (`AppJourneyTests`, `PersonalisationTests`, `OfflineCacheTests`, `ServerDegradedTests`) own the
 * assertions; if this failed on a missing element it would abandon the tour half-way and produce a
 * partial sheet, which is the opposite of what it is for. Every step is therefore best-effort and
 * shoots whatever is on screen — a surface that failed to load is exactly the thing the sheet
 * should show.
 *
 * Numeric prefixes drive the order of tiles in the sheet.
 *
 * PRECONDITIONS: app installed, signed in (`make ios-journey-signin`), fixture api reachable.
 * Run AFTER the personalisation suite if you want Stats/Topics populated rather than empty.
 */
final class ScreenshotTourTests: UITestCase {

  /// SHARED account, deliberately: this suite photographs a populated app; `ios-contact-sheet` runs the journey + personalisation suites first on purpose.
  /// Per-suite isolation (#2091) would give it an empty account and the seed would be invisible.
  override var accountIdentity: String { Self.sharedSeededIdentity }
  private let episodeSlug = "p09-a4bbb5dde3"


  /// Every frame this tour is supposed to produce. The run FAILS if any is missing.
  ///
  /// The frames are best-effort by design — one unreachable surface should not cost the other
  /// twenty. But best-effort with no tally meant a stuck sheet silently dropped the last three
  /// screens and the tour still reported success; the gap only surfaced when a human noticed a
  /// screenshot they expected was not on the contact sheet (operator 2026-09-16). Shooting what it
  /// can and then declaring what it could not is the honest version.
  private static let expectedFrames = [
    "t01-home", "t02-discover", "t03-search",
    "t04-library-following", "t05-library-saved", "t06-library-boards", "t07-library-revisit",
    "t08-profile-account", "t09-profile-topics", "t10-profile-stats",
    "t11-settings", "t12-settings-config",
    "t13-episode", "t14-episode-insights", "t15-episode-keypoints", "t16-episode-entities",
    "t17-share-popover", "t18-add-to-collection",
    // Stacking from a STORYLINE root (opened from Home → Storylines).
    "t19-topic", "t20-storyline", "t20b-storyline-content", "t20c-storyline-topic-layered",
    "t20d-person-over-topic-over-storyline",
    "t21-person",
    // Stacking from a TOPIC root (opened from the player's Insights) — topic underneath in all
    // four, read them in order. Kept separate from the storyline root above on purpose: different
    // roots, and a regression can hit one without touching the other.
    "t22a-topic-sheet", "t22b-person-over-topic", "t22c-storyline-over-topic",
    "t22d-person-over-storyline-over-topic",
    "t23-saved-row", "t24-colour-popover", "t25-home-offline", "t26-home-final",
  ]

  /// People who appear as top-voice chips in the fixture corpus. Named explicitly because those
  /// chips carry the bare name as their accessible label, so there is no prefix to match on.
  private static let fixturePeople = [
    "Dr. Elena Fischer", "Sam", "Skanda Amarnath", "Alex Morgan",
  ]

  private var shotFrames: Set<String> = []

  /// Shoot the current screen. Records the name so the tour can report what it never reached.
  private func frame(_ name: String) {
    Journey.shot(self, name)
    shotFrames.insert(name)
  }

  /// Fail with the list of screens the tour never reached, rather than quietly shooting fewer.
  private func assertEveryFrameShot() {
    let missing = Self.expectedFrames.filter { !shotFrames.contains($0) }
    XCTAssertTrue(
      missing.isEmpty,
      "the tour never reached \(missing.count) screen(s): \(missing.joined(separator: ", ")) — "
        + "a sheet left open swallows every later tap, so check for SHEETS_STUCK above"
    )
  }

  func testTourEverySurface() {
    let app = Journey.launch()

    // --- primary tabs -------------------------------------------------------------------------
    frame("t01-home")

    if Journey.openTab(app, "Discover") { sleep(5); frame("t02-discover") }
    if Journey.openTab(app, "Search") { sleep(4); frame("t03-search") }

    // --- library, every tab -------------------------------------------------------------------
    if Journey.openTab(app, "Library") {
      sleep(4)
      frame("t04-library-following")
      for (i, tab) in ["Saved", "Boards", "Revisit"].enumerated() {
        if Journey.tap(app, labels: [tab], timeout: 10) {
          sleep(3)
          frame(String(format: "t0%d-library-%@", 5 + i, tab.lowercased()))
        }
      }
    }

    // --- profile, every tab, then settings ----------------------------------------------------
    if Journey.openProfile(app) {
      sleep(4)
      frame("t08-profile-account")
      if Journey.tap(app, labels: ["Topics"], timeout: 10) { sleep(3); frame("t09-profile-topics") }
      if Journey.tap(app, labels: ["Stats"], timeout: 10) { sleep(3); frame("t10-profile-stats") }
      if Journey.tap(app, labels: ["Settings"], contains: true, timeout: 10) {
        sleep(3)
        frame("t11-settings")
        // Config lives at the bottom — the offline switch + space reclaim.
        _ = Journey.scrollTo(app, labels: ["Offline mode"])
        frame("t12-settings-config")
      }
    }

    // --- episode + knowledge panel ------------------------------------------------------------
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    frame("t13-episode")
    if Journey.tap(app, labels: ["Insights"], contains: true, timeout: 12) {
      sleep(4)
      frame("t14-episode-insights")
      if Journey.tap(app, labels: ["Key points"], contains: true, timeout: 8) {
        sleep(3); frame("t15-episode-keypoints")
      }
      if Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 8) {
        sleep(3); frame("t16-episode-entities")
      }
    }

    // --- the overlays the operator asked to see rendered --------------------------------------
    Journey.dismissSheets(app)
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(5)
    if Journey.tap(app, labels: ["Share"], contains: true, timeout: 10) {
      sleep(3); frame("t17-share-popover")
      _ = Journey.tap(app, labels: ["Close", "Cancel"], contains: true, timeout: 4)
    }
    if Journey.tap(app, labels: ["Add to collection", "Add"], contains: true, timeout: 10) {
      sleep(3); frame("t18-add-to-collection")
      _ = Journey.tap(app, labels: ["Close", "Cancel"], contains: true, timeout: 4)
    }

    // --- entity surfaces ----------------------------------------------------------------------
    Journey.dismissSheets(app)
    Journey.openTab(app, "Home")
    sleep(5)
    if Journey.tap(app, labels: ["Topics"], timeout: 10) { sleep(3) }
    if Journey.tap(app, labels: ["systems thinking"], contains: true, timeout: 10) {
      sleep(5); frame("t19-topic")
    }
    Journey.openTab(app, "Home")
    sleep(4)
    if Journey.tap(app, labels: ["Storylines"], timeout: 10) {
      sleep(4)
      let rows = app.buttons.allElementsBoundByIndex.filter {
        $0.label.contains("momentum") && $0.label.contains("(")
      }
      if let first = rows.first, first.isHittable {
        first.tap()
        sleep(5)
        frame("t20-storyline")

        // Storyline sheet parity + layering (operator 2026-09-16): episodes, people and notes now
        // render IN the sheet, and a member topic/person opens as a sheet ON TOP rather than
        // navigating away. Shoot the layered state — it is the part nobody has actually looked at.
        _ = Journey.scrollTo(app, labels: ["Topics discussed together"], maxSwipes: 4)
        frame("t20b-storyline-content")
        let topicRow = app.links.allElementsBoundByIndex.first {
          !$0.label.isEmpty && $0.label.count < 40 && $0.isHittable
        }
        if let row = topicRow {
          row.tap()
          sleep(5)
          frame("t20c-storyline-topic-layered")

          // Three deep from THIS entry point — storyline at the bottom. Kept alongside the
          // topic-bottomed set below because the two exercise different roots: here the storyline
          // was opened from a page and everything stacks on it, there the topic is the root. A
          // regression could easily hit one and not the other.
          _ = Journey.scrollTo(app, labels: ["Top voices"], maxSwipes: 6)
          if Journey.tap(app, labels: Self.fixturePeople, contains: false, timeout: 8) {
            sleep(5)
            frame("t20d-person-over-topic-over-storyline")
          }
          Journey.dismissSheets(app)
        }
      }
    }

    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(5)
    _ = Journey.tap(app, labels: ["Insights"], contains: true, timeout: 10)
    sleep(2)
    _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 8)
    sleep(2)
    _ = Journey.scrollTo(app, labels: ["Open Dr. Elena Fischer", "Open Sam"])
    if Journey.tap(app, labels: ["Open Dr. Elena Fischer", "Open Sam"], contains: true, timeout: 8) {
      sleep(5); frame("t21-person")
    }
    Journey.dismissSheets(app)

    // --- the stacking sequence, TOPIC at the bottom -------------------------------------------
    //
    // Shot as an ordered set from ONE entry point so the four frames can be read against each
    // other: the same topic card is underneath in every one of them.
    //
    //   t22a  topic alone            depth 0
    //   t22b  person over topic      depth 1
    //   t22c  storyline over topic   depth 1   (the sibling case — same level, different kind)
    //   t22d  person / storyline / topic       depth 2, topic still at the bottom
    //
    // Entry is the episode's Topics & People, NOT Home → Topics: that route opens the topic as a
    // PAGE, the person then opens at depth 0 and covers it completely, and the frame comes out
    // indistinguishable from `t21-person`. Opened from Topics & People the topic is a SHEET, which
    // is what gives everything above it something to stack ON (operator 2026-09-16).
    func openTopicSheet() -> Bool {
      AppSession.openEpisode(app, slug: episodeSlug)
      sleep(5)
      _ = Journey.tap(app, labels: ["Insights"], contains: true, timeout: 10)
      sleep(2)
      _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 8)
      sleep(2)
      let opened = Journey.tap(
        app, labels: ["Open risk management", "Open systems thinking"], contains: true, timeout: 10
      )
      if opened { sleep(4) }
      return opened
    }

    if openTopicSheet() {
      frame("t22a-topic-sheet")
      // Top-voice chips are BUTTONS labelled with the bare person name (`:aria-label="p.name"`) —
      // no "Open " prefix, unlike the insights panel's rows.
      _ = Journey.scrollTo(app, labels: ["Top voices"], maxSwipes: 6)
      if Journey.tap(app, labels: Self.fixturePeople, contains: false, timeout: 8) {
        sleep(5)
        frame("t22b-person-over-topic")
      }
    }
    Journey.dismissSheets(app)

    if openTopicSheet() {
      _ = Journey.scrollTo(app, labels: ["Part of a storyline"], maxSwipes: 6)
      if Journey.tap(app, labels: ["Managing risk across domains"], contains: true, timeout: 10) {
        sleep(5)
        frame("t22c-storyline-over-topic")

        // Three deep, and the point of the whole exercise: a boolean "is nested" flag can express
        // level 1 and nothing beyond it. Each card below has to keep its title on screen.
        _ = Journey.scrollTo(app, labels: ["Topics discussed together", "Related people"],
                             maxSwipes: 6)
        if Journey.tap(app, labels: Self.fixturePeople, contains: false, timeout: 8) {
          sleep(5)
          frame("t22d-person-over-storyline-over-topic")
        }
      }
    }
    Journey.dismissSheets(app)

    // --- saved colour picker (the row the operator reworked) ----------------------------------
    Journey.dismissSheets(app)
    Journey.openTab(app, "Library")
    sleep(4)
    if Journey.tap(app, labels: ["Saved"], contains: true, timeout: 10) {
      sleep(3)
      frame("t23-saved-row")
      // EXACT match: the per-item picker is "Colour" and the filter group is "Filter by colour",
      // and a CONTAINS match would drive whichever came first in the tree.
      if Journey.tap(app, labels: ["Colour", "Color"], contains: false, timeout: 8) {
        sleep(3); frame("t24-colour-popover")
      }
    }

    // --- offline, with the cache already warmed by everything above ----------------------------
    Journey.dismissSheets(app)
    if Journey.setOfflineMode(app, on: true) {
      Journey.openTab(app, "Home")
      sleep(5)
      frame("t25-home-offline")
      _ = Journey.setOfflineMode(app, on: false)
    }

    Journey.openTab(app, "Home")
    sleep(4)
    frame("t26-home-final")

    assertEveryFrameShot()
  }
}
