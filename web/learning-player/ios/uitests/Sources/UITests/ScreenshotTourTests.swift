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
final class ScreenshotTourTests: XCTestCase {
  private let episodeSlug = "p09-a4bbb5dde3"

  override func setUp() {
    super.setUp()
    continueAfterFailure = true
  }

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
    "t19-topic", "t20-storyline", "t20b-storyline-content", "t20c-storyline-topic-layered",
    "t20d-stack-three-deep", "t21-person", "t21b-person-over-topic",
    "t22-saved-row", "t23-colour-popover", "t24-home-offline", "t25-home-final",
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

          // THREE deep — storyline → topic → person. The deck has to keep working past one level:
          // each card below stays visible by its title, each new one starts a title lower. Two
          // levels can be faked by a single "is nested" flag; three cannot (operator 2026-09-16).
          // Top-voice chips are BUTTONS labelled with the bare person name (`:aria-label="p.name"`)
          // — no "Open " prefix, unlike the insights panel's rows. Matching the panel's shape here
          // found nothing at all and cost this frame on the first attempt.
          _ = Journey.scrollTo(app, labels: ["Top voices"], maxSwipes: 6)
          if Journey.tap(app, labels: Self.fixturePeople, contains: false, timeout: 8) {
            sleep(5)
            frame("t20d-stack-three-deep")
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

    // --- person layered OVER a topic ----------------------------------------------------------
    // `t21-person` above opens a person from the insights panel, where it is the only sheet on
    // screen — which is why it photographs as a full-height card with nothing behind it and reads
    // as a page. The stacked case is a different surface and was never shot at all (operator
    // 2026-09-16: "I don't see a sheet where we open person overlaying the topic").
    Journey.openTab(app, "Home")
    sleep(4)
    if Journey.tap(app, labels: ["Topics"], timeout: 10) { sleep(3) }
    if Journey.tap(app, labels: ["systems thinking", "risk management"], contains: true, timeout: 10) {
      sleep(5)
      _ = Journey.scrollTo(app, labels: ["Top voices"], maxSwipes: 6)
      if Journey.tap(app, labels: Self.fixturePeople, contains: false, timeout: 8) {
        sleep(5)
        frame("t21b-person-over-topic")
      }
    }
    Journey.dismissSheets(app)

    // --- saved colour picker (the row the operator reworked) ----------------------------------
    Journey.dismissSheets(app)
    Journey.openTab(app, "Library")
    sleep(4)
    if Journey.tap(app, labels: ["Saved"], contains: true, timeout: 10) {
      sleep(3)
      frame("t22-saved-row")
      // EXACT match: the per-item picker is "Colour" and the filter group is "Filter by colour",
      // and a CONTAINS match would drive whichever came first in the tree.
      if Journey.tap(app, labels: ["Colour", "Color"], contains: false, timeout: 8) {
        sleep(3); frame("t23-colour-popover")
      }
    }

    // --- offline, with the cache already warmed by everything above ----------------------------
    Journey.dismissSheets(app)
    if Journey.setOfflineMode(app, on: true) {
      Journey.openTab(app, "Home")
      sleep(5)
      frame("t24-home-offline")
      _ = Journey.setOfflineMode(app, on: false)
    }

    Journey.openTab(app, "Home")
    sleep(4)
    frame("t25-home-final")

    assertEveryFrameShot()
  }
}
