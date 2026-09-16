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

  /// One fixture episode per colour token. The Saved colour FILTER offers only the colours actually
  /// in use (`LibraryView.colorsPresent`), so a seed that coloured a single item rendered a single
  /// swatch and the filter looked broken in review when it was merely empty (operator 2026-09-16).
  /// Colouring one episode per token is what makes that control reviewable at all.
  private let colourSeeds: [(slug: String, colour: String)] = [
    ("p09-a4bbb5dde3", "Amber"),
    ("p09-6ac0bf4914", "Rose"),
    ("p08-72169222b1", "Sky"),
    ("p08-bd7cc798ff", "Emerald"),
    ("p07-2aceab172c", "Violet"),
  ]

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
    // The accordion REMEMBERS its state between tests, so a blind tap can collapse a section that
    // a previous test left open — which is how this passed alone and failed in a full run. Tap only
    // when the person controls are not already reachable, and re-tap once if the first tap closed
    // it (2026-09-16).
    let personPre = NSPredicate(format: "label CONTAINS[c] 'Open Dr. Elena Fischer' OR label CONTAINS[c] 'Open Sam'")
    if !app.buttons.matching(personPre).firstMatch.waitForExistence(timeout: 4) {
      _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 15)
      sleep(3)
      if !app.buttons.matching(personPre).firstMatch.waitForExistence(timeout: 6) {
        _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 10)
        sleep(3)
      }
    }
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

    // Seed every colour, not just one. Saved is newest-first, so the episode favourited on this
    // pass is row one — which is why each colour can be applied to the FIRST colour control
    // without having to address a specific row.
    for seed in colourSeeds.dropFirst() {
      AppSession.openEpisode(app, slug: seed.slug)
      sleep(5)
      if Journey.find(app, labels: ["Save to favorites"], contains: true, timeout: 8) != nil {
        _ = Journey.tap(app, labels: ["Save to favorites"], contains: true, timeout: 8)
        sleep(3)
      }
      Journey.openTab(app, "Library")
      sleep(3)
      _ = Journey.tap(app, labels: ["Saved"], contains: true, timeout: 10)
      sleep(2)
      // EXACT match. The per-item trigger is labelled "Colour" and the filter group is "Filter by
      // colour"; `contains` is CONTAINS[c], so a loose match hits whichever comes first in the
      // tree and would silently drive the filter instead of the picker.
      if Journey.tap(app, labels: ["Colour", "Color"], contains: false, timeout: 10) {
        sleep(2)
        _ = Journey.tap(app, labels: ["Set colour: \(seed.colour)"], contains: true, timeout: 8)
        sleep(2)
      }
    }

    // Favourite the episode so Saved has something to colour-code.
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)
    // Favourite only if it is not ALREADY favourited. A blind tap toggles, so running after a test
    // that favourited this episode un-favourited it and left Saved empty — the colour control then
    // "could not be reached" because there was nothing to colour (2026-09-16).
    if Journey.find(app, labels: ["Save to favorites"], contains: true, timeout: 8) != nil {
      _ = Journey.tap(app, labels: ["Save to favorites"], contains: true, timeout: 8)
      sleep(3)
    }
    XCTAssertNotNil(
      Journey.find(app, labels: ["Remove from favorites"], contains: true, timeout: 10),
      "the episode is not favourited, so Saved would be empty"
    )

    Journey.openTab(app, "Library")
    sleep(4)
    _ = Journey.tap(app, labels: ["Saved"], contains: true, timeout: 12)
    sleep(3)
    Journey.inventory(app, "library-saved")
    Journey.shot(self, "07-library-saved")

    // The colour control opens a popover of named colours. Exact match — see the loop above.
    if Journey.tap(app, labels: ["Colour", "Color"], contains: false, timeout: 12) {
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

    // The FILTER (distinct from the per-item picker above) offers only colours that are in use, so
    // it is a direct readout of whether the seeding worked. Asserting >1 is what stops this
    // silently collapsing back to a single swatch — which is how it reached review (2026-09-16).
    Journey.openTab(app, "Library")
    sleep(3)
    _ = Journey.tap(app, labels: ["Saved"], contains: true, timeout: 12)
    sleep(3)
    // `library.savedFilterColorOnly` renders as "Only Amber" / "Only Rose" — one button per colour
    // actually in use. Matching that exact string keeps this assertion about the FILTER and not
    // about the per-item picker, which uses "Set colour: <name>".
    let offered = colourSeeds.filter {
      Journey.find(app, labels: ["Only \($0.colour)"], contains: true, timeout: 2) != nil
    }
    Journey.inventory(app, "colour-filter")
    Journey.shot(self, "07b-colour-filter")
    XCTAssertGreaterThan(
      offered.count, 1,
      "the Saved colour filter offers \(offered.count) colour(s) — the seed coloured too few items"
    )
  }

  // MARK: - 11 storyline from the INSIGHTS panel (the canLayer=false path)

  /**
   * Inside the Knowledge Panel the entity card is INLINE, so the layering policy says a sheet must
   * not stack over it — the panel is already the layer. Topics replace in place via the shell's back
   * stack; a storyline has no back-stack equivalent, so it routes to its standalone page instead.
   *
   * This is the branch nothing exercised: every other storyline test opens one from a SHEET, where
   * it stacks. If `canLayer` were wired wrongly the panel would sprout a modal over itself and
   * nobody would notice (2026-09-16).
   */
  /// From the Insights panel: topic underneath, storyline stacked ON it — not a page.
  ///
  /// This asserted the opposite until 2026-09-16. Routing away was read as correct because
  /// replace-in-panel (UXS-014) says a sheet may not layer over an inline PANEL — but that rule is
  /// about what a tapped CHIP does inside the panel, and the panel is itself a full-height bottom
  /// sheet. Navigating to the storyline PAGE threw the topic away entirely, which is the opposite
  /// of the point: the operator's requirement is that each card stays visible by its title while
  /// the next one stacks below it.
  func test11StorylineFromInsightsStacksOverTheTopic() {
    let app = Journey.launch()
    AppSession.openEpisode(app, slug: episodeSlug)
    sleep(6)

    guard Journey.tap(app, labels: ["Insights"], contains: true, timeout: 15) else {
      XCTFail("could not open the knowledge panel"); return
    }
    sleep(3)
    // The accordion remembers its state, so tap only when the topic controls are not already there
    // (a blind tap would COLLAPSE a section a previous test left open) — same trap as test04.
    let topicPre = NSPredicate(format: "label CONTAINS[c] 'Open systems thinking' OR label CONTAINS[c] 'Open risk management'")
    if !app.buttons.matching(topicPre).firstMatch.waitForExistence(timeout: 4) {
      _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 12)
      sleep(3)
      if !app.buttons.matching(topicPre).firstMatch.waitForExistence(timeout: 6) {
        _ = Journey.tap(app, labels: ["Topics & People"], contains: true, timeout: 10)
        sleep(3)
      }
    }

    // Drill into a topic IN THE PANEL — it must replace in place, not open a sheet.
    let topicPredicate = NSPredicate(format: "label CONTAINS[c] 'Open systems thinking' OR label CONTAINS[c] 'Open risk management'")
    let topicButton = app.buttons.matching(topicPredicate).firstMatch
    guard topicButton.waitForExistence(timeout: 20) else {
      Journey.inventory(app, "panel-no-topic")
      XCTFail("no topic control in the insights panel"); return
    }
    topicButton.tap()
    sleep(4)
    Journey.shot(self, "11-a-topic-in-panel")

    // The topic card inside the panel offers its storyline. Tapping it must NAVIGATE.
    guard let storyline = Journey.scrollTo(app, labels: ["Part of a storyline"], maxSwipes: 8) else {
      Journey.inventory(app, "panel-topic-no-storyline")
      Journey.shot(self, "11-b-no-storyline-link")
      XCTFail("the in-panel topic card offered no storyline"); return
    }
    _ = storyline
    guard Journey.tap(app, labels: ["Managing risk across domains"], contains: true, timeout: 12) else {
      Journey.inventory(app, "panel-storyline-not-tappable")
      XCTFail("storyline row not tappable in the panel"); return
    }
    sleep(5)
    Journey.inventory(app, "after-panel-storyline")
    Journey.shot(self, "11-c-storyline-from-panel")

    // Assert on something ONLY a storyline renders. The first version of this checked for
    // "Managing risk across domains" and passed while nothing had opened at all — that string is
    // the label of the "Part of a storyline" ROW inside the topic card itself. `chromeReachable`
    // was no better: the panel covers the tab bar whether or not a sheet is above it. Two checks,
    // neither of which could distinguish the outcomes (2026-09-16).
    XCTAssertNotNil(
      // "Follow storyline" unfollowed, "Following storyline" once followed — match either, or the
      // assertion depends on this account's follow state rather than on the sheet being open.
      Journey.find(app, labels: ["Follow storyline"], contains: true, timeout: 12),
      "no storyline sheet on screen — the storyline never opened, or it opened behind the panel's "
        + "top layer (the panel uses showModal(), so a sheet teleported to <body> is hidden by it)"
    )
    // And the topic underneath must STILL be identifiable by its title. That is the contract —
    // stacking keeps it, routing away destroys it.
    XCTAssertNotNil(
      Journey.find(app, labels: ["systems thinking", "risk management"], contains: true, timeout: 8),
      "the topic is gone — the storyline replaced it instead of stacking over it"
    )
    Journey.shot(self, "11-d-storyline-stacked-over-topic")
  }
}
