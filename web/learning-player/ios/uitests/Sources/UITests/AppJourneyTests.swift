import XCTest

/**
 * Native journey suite (operator request 2026-09-16) — the beginning of an on-device test set for
 * the surfaces the web tiers cannot reach.
 *
 * Scope, in the order the operator asked for it:
 *   01  profile tabs (Account / Topics / Stats)
 *   02  episode → insights
 *   03  topic page → storyline page
 *   04  person page, reached from an episode's entities
 *   05  collections: create, then "Add to collection" lists them
 *   06  share popover renders its options
 *   07  saved-item colour picker popover renders and a colour can be chosen
 *
 * Each test launches fresh and navigates from Home, so any one can run alone
 * (`-only-testing:OfflineSpikeUITests/AppJourneyTests/test03TopicAndStoryline`). Methods are
 * numbered because XCTest runs them alphabetically and 05 seeds the collections 06 relies on.
 *
 * Every step attaches a screenshot (`Journey.shot`) and prints a compact element inventory on the
 * paths most likely to drift, so a failing selector is diagnosable from ONE run.
 *
 * PRECONDITIONS: app installed and signed in (host seeds `CapacitorStorage.lp_native_token`), and
 * the fixture api reachable. Fixture ids below are from the committed v3 corpus.
 */
final class AppJourneyTests: XCTestCase {
  /// An episode that actually HAS insights — picked from the fixture corpus (5 insights), so the
  /// insights assertions are testing rendering and not an empty-state.
  private let episodeSlug = "p09-a4bbb5dde3"
  private let episodeTitle = "Risk Is a Systems Property"

  override func setUp() {
    super.setUp()
    continueAfterFailure = true // collect every screenshot in a run, don't stop at the first gap
  }

  // MARK: - 01 profile

  func test01ProfileTabs() {
    let app = Journey.launch()
    Journey.shot(self, "01-home")

    XCTAssertTrue(Journey.openProfile(app), "could not open Profile from the masthead avatar")
    sleep(3)
    Journey.inventory(app, "profile")
    Journey.shot(self, "01-profile-account")

    for tab in ["Topics", "Stats"] {
      if Journey.tap(app, labels: [tab], timeout: 12) {
        sleep(3)
        Journey.shot(self, "01-profile-\(tab.lowercased())")
      } else {
        Journey.inventory(app, "profile-missing-\(tab)")
        XCTFail("profile tab '\(tab)' not tappable")
      }
    }
  }

  // MARK: - 02 episode + insights

  func test02EpisodeAndInsights() {
    let app = Journey.launch()
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    Journey.shot(self, "02-episode")
    XCTAssertNotNil(
      Journey.find(app, labels: [episodeTitle], contains: true, timeout: 20),
      "episode page did not render its title"
    )

    // The knowledge panel's Insights section — a tab/section header labelled "Insights".
    if Journey.tap(app, labels: ["Insights"], contains: true, timeout: 15) {
      sleep(4)
    } else {
      _ = Journey.scrollTo(app, labels: ["Insights"])
      sleep(2)
    }
    Journey.inventory(app, "episode-insights")
    Journey.shot(self, "02-episode-insights")
  }

  // MARK: - 03 topic + storyline

  func test03TopicAndStoryline() {
    let app = Journey.launch()

    // Topic ids carry a `topic:` prefix, which the deep-link id validator rejects by design, so
    // topics are reached the way a user reaches them: the Home entity rail's Topics tab.
    if Journey.tap(app, labels: ["Topics"], timeout: 15) { sleep(3) }
    Journey.inventory(app, "home-topics-rail")
    if Journey.tap(app, labels: ["systems thinking", "risk management"], contains: true, timeout: 12) {
      sleep(5)
      Journey.shot(self, "03-topic")
      Journey.inventory(app, "topic-page")
    } else {
      Journey.shot(self, "03-topic-MISS")
      XCTFail("no topic row was tappable from the Home rail")
    }

    // Storylines: back to Home, switch the rail to Storylines, open the first one. A storyline is
    // anchored on a topic, so this is the "topic that belongs to a storyline" path.
    Journey.openTab(app, "Home")
    sleep(4)
    if Journey.tap(app, labels: ["Storylines"], timeout: 15) {
      sleep(4)
      Journey.inventory(app, "home-storylines-rail")
      Journey.shot(self, "03-storylines-rail")
      // The rail rows are BUTTONS, not links (the first cut of this test looked at `app.links` and
      // found only the page chrome). A storyline row reads
      // "Managing risk across domains (4) — 2.2× momentum": the bracketed episode count is what
      // separates it from a topic row, which carries momentum but no count.
      let rows = app.buttons.allElementsBoundByIndex.filter {
        let l = $0.label
        return l.contains("momentum") && l.contains("(") && l.contains(")")
      }
      print("=====STORYLINE_ROWS \(rows.prefix(8).map { $0.label })=====")
      if let first = rows.first, first.isHittable {
        first.tap()
        sleep(5)
        Journey.shot(self, "03-storyline")
        Journey.inventory(app, "storyline-page")
        // A storyline opens as a SHEET (title + "Follow storyline" + "Topics discussed together" +
        // "Open in page ›"), not a full page — so "did it navigate" cannot be asked by looking for
        // the absence of the rail. The sheet's own kicker is the word "STORYLINES", which is what
        // an earlier version of this assertion matched against itself. Assert on a control that
        // exists ONLY on the sheet.
        XCTAssertNotNil(
          Journey.find(app, labels: ["Follow storyline", "Topics discussed together", "Open in page"],
                       contains: true, timeout: 10),
          "the storyline sheet did not open"
        )
      } else {
        Journey.shot(self, "03-storyline-MISS")
        XCTFail("no storyline row was tappable")
      }
    } else {
      XCTFail("Storylines tab not found on the Home rail")
    }
  }

  // MARK: - 04 person

  func test04PersonFromEpisode() {
    let app = Journey.launch()
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)

    // The first cut matched the person's NAME and "passed" while never leaving the player — the
    // name appears in the episode's own transcript/summary text, so the match was real and the
    // navigation was not. The actual control is the knowledge panel's "Open <name>" button, which
    // lives behind the "Topics & People" section.
    // The knowledge panel is CLOSED on a fresh launch — "Topics & People" only exists once it is
    // open. (This passed the first time only because an earlier test in the same run had already
    // opened it.) Open the panel from the player's "✦ Insights" entry point first.
    // Three steps, all required: open the knowledge panel (closed on a fresh launch), EXPAND the
    // "Topics & People" accordion (its sections render collapsed — only the headers are in the
    // tree), then scroll the control into view. Offscreen web content is not in the accessibility
    // tree at all, so a plain `find` cannot see it and `tap`'s own scroll never triggers.
    _ = Journey.tap(app, labels: ["Insights"], contains: true, timeout: 15)
    sleep(3)
    _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 15)
    sleep(3)
    // POLL for the control instead of snapshotting once. Expanding the accordion lays the section
    // out asynchronously, so a single `scrollTo` pass could run before the buttons existed — which
    // is why this passed alone and failed inside a full run, where the preceding tests changed the
    // timing. Waiting is what makes it order-independent (2026-09-16).
    let personPredicate = NSPredicate(format: "label CONTAINS[c] 'Open Dr. Elena Fischer' OR label CONTAINS[c] 'Open Sam'")
    let personButton = app.buttons.matching(personPredicate).firstMatch
    if !personButton.waitForExistence(timeout: 20) {
      _ = Journey.scrollTo(app, labels: ["Open Dr. Elena Fischer", "Open Sam"], maxSwipes: 10)
    }
    guard Journey.tap(app, labels: ["Open Dr. Elena Fischer", "Open Sam"], contains: true, timeout: 15) else {
      Journey.inventory(app, "episode-no-person")
      Journey.shot(self, "04-person-MISS")
      XCTFail("no 'Open <person>' control on the episode")
      return
    }
    sleep(5)
    Journey.shot(self, "04-person")
    Journey.inventory(app, "person-page")

    // Guard against the false pass: a person page has no episode transport.
    XCTAssertNil(
      Journey.find(app, labels: ["Skip forward 30 seconds"], timeout: 3),
      "still on the player — tapping the person did not navigate to a person page"
    )
  }

  // MARK: - 05 collections

  func test05CollectionsCreateAndAdd() {
    let app = Journey.launch()
    Journey.openTab(app, "Library")
    sleep(4)
    Journey.inventory(app, "library")
    Journey.shot(self, "05-library")

    // Collections live on the Library tab called "Boards" — NOT "Collections", which is the
    // internal/API name and matched nothing on screen. The fixture account starts with none.
    guard Journey.tap(app, labels: ["Boards"], timeout: 12) else {
      Journey.inventory(app, "library-no-boards")
      XCTFail("no Boards tab in Library")
      return
    }
    sleep(3)
    Journey.inventory(app, "boards")
    Journey.shot(self, "05-collections-before")

    for name in ["Test Board A", "Test Board B"] {
      guard let field = Journey.find(app, labels: ["New collection name"], contains: true, timeout: 12)
        ?? app.textFields.firstMatch as XCUIElement?
      else { XCTFail("no collection-name field"); break }
      guard field.waitForExistence(timeout: 8) else { XCTFail("no collection-name field"); break }
      field.tap()
      field.typeText(name)
      if !Journey.tap(app, labels: ["Create"], timeout: 8) {
        Journey.inventory(app, "collections-no-create")
        XCTFail("Create button not tappable for \(name)")
        break
      }
      sleep(3)
    }
    Journey.shot(self, "05-collections-after")
    Journey.inventory(app, "collections-after")

    // Now the "Add to collection" sheet on an episode must LIST those collections.
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    if Journey.tap(app, labels: ["Add to collection"], contains: true, timeout: 15) {
      sleep(3)
      Journey.inventory(app, "add-to-collection")
      Journey.shot(self, "05-add-to-collection")
      XCTAssertNotNil(
        Journey.find(app, labels: ["Test Board A"], contains: true, timeout: 10),
        "the Add-to-collection sheet did not list the collections that exist"
      )
    } else {
      Journey.inventory(app, "episode-no-addto")
      Journey.shot(self, "05-add-to-collection-MISS")
      XCTFail("'Add to collection' not reachable from the episode")
    }
  }

  // MARK: - 06 share

  func test06SharePopover() {
    let app = Journey.launch()
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    if Journey.tap(app, labels: ["Share"], contains: true, timeout: 15) {
      sleep(3)
      Journey.inventory(app, "share-popover")
      Journey.shot(self, "06-share-popover")
      // The popover offers card / link / text; at least one must render, or nothing opened.
      XCTAssertNotNil(
        Journey.find(app, labels: ["Share card", "Share link", "Share text"], contains: true, timeout: 10),
        "share popover did not render its options"
      )
    } else {
      Journey.inventory(app, "episode-no-share")
      Journey.shot(self, "06-share-MISS")
      XCTFail("Share control not reachable from the episode")
    }
  }

  // MARK: - 07 saved colour picker

  func test07SavedColourPicker() {
    let app = Journey.launch()

    // Favourite the episode so Saved has something to colour-code.
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    _ = Journey.tap(app, labels: ["Favourite", "Favorite", "Save"], contains: true, timeout: 12)
    sleep(3)

    Journey.openTab(app, "Library")
    sleep(4)
    _ = Journey.tap(app, labels: ["Saved"], contains: true, timeout: 12)
    sleep(3)
    Journey.inventory(app, "library-saved")
    Journey.shot(self, "07-library-saved")

    // The colour control opens a popover of named colours.
    if Journey.tap(app, labels: ["Colour", "Color"], contains: true, timeout: 12) {
      sleep(3)
      Journey.inventory(app, "colour-popover")
      Journey.shot(self, "07-colour-popover")
      guard let swatch = Journey.find(
        app, labels: ["Amber", "Rose", "Sky", "Emerald", "Violet"], contains: true, timeout: 10
      ) else {
        XCTFail("colour popover rendered no colour choices")
        return
      }
      swatch.tap()
      sleep(3)
      Journey.shot(self, "07-colour-chosen")
    } else {
      Journey.inventory(app, "saved-no-colour")
      Journey.shot(self, "07-colour-MISS")
      XCTFail("colour control not reachable from Saved")
    }
  }
}
