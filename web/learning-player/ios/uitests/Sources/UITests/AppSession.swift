import XCTest

/**
 * Shared session helpers for the device tier.
 *
 * Both suites need the same question answered — "is this app signed in?" — and answering it too
 * early is what made the harness non-deterministic. Boot PAINTS the last known identity from the
 * device first, so an offline launch is signed in immediately, and only then revalidates. A token
 * minted by a previous api container is refused by the new one (the fixture api is recreated on
 * every `make app-e2e-api-up`), so "Sign out" can sit on screen for a few seconds and then vanish.
 *
 * The playback test used to decide inside that window: it saw the painted session, skipped its
 * sign-in, and then asserted against an app that had just signed itself out. The failure read as
 * "sign-in did not complete" when no sign-in had been attempted at all.
 */
enum AppSession {
  /// Whether the app is STILL signed in once the boot revalidation has landed.
  ///
  /// "Sign out" lives on the PROFILE page and nowhere else, but the app cold-boots to Home — so
  /// this asked a Home screen whether it had a Profile-only control and was answered "no" every
  /// time, whatever the session actually was. The offline suite reported that as "the app fell
  /// back to signed-out" on a device whose stored token was valid, and the download suite re-ran
  /// a sign-in it did not need. Go to Profile first, then read the answer.
  ///
  /// Opening Profile DELEGATES to `Journey.openProfile` rather than keeping a second copy of the
  /// selector. The header entry point is labelled with the signed-in DISPLAY NAME (`simtest`), not
  /// the static "Your profile" this used to hard-code — so once that label changed, this helper
  /// reported signed-out for every session and `signIn` then failed looking for a "Sign in" link
  /// that was correctly absent on a signed-in app. `Journey` already carried the fallback list;
  /// only this copy was stale. Two helpers knowing the same UI differently is the actual defect.
  /// Signed in AT ALL. Prefer `isSignedIn(_:as:)` — with per-suite accounts (#2091) "a session
  /// exists" is no longer the question worth asking.
  static func isSignedIn(_ app: XCUIApplication) -> Bool {
    _ = Journey.openProfile(app)
    // SCROLL to it. "Sign out" is deliberately the last control on Profile (#1962 — "quiet, last,
    // least weight"), so on any account with content it is below the fold. Waiting for it to exist
    // without scrolling asks whether it is on SCREEN, which is not the question: on 2026-09-18 a
    // capture-stats block was added above it and every suite that calls this reported "sign-in did
    // not complete" for a session that was perfectly valid and an app sitting on a signed-in Home.
    guard Journey.scrollTo(app, labels: ["Sign out"], contains: false) != nil else { return false }
    // The painted session is not the answer — the revalidation that follows it is. Six seconds is
    // the observed worst case for `refresh()` against the local fixture api plus a re-render.
    sleep(6)
    return Journey.scrollTo(app, labels: ["Sign out"], contains: false) != nil
  }

  /// Open an episode by slug through the app's deep-link scheme (#1925).
  ///
  /// Replaces navigating by taps and blind swipes, which was the single flakiest thing in this
  /// tier: the shows and episode lists keep their scroll position between visits, so a
  /// swipe-and-look search walked past rows and — on one run — mis-tapped and downloaded an
  /// episode the test never asked for. A deep link addresses the episode directly, which is what
  /// the link is FOR; the test is now exercising a product capability rather than working around
  /// the lack of one.
  static func openEpisode(_ app: XCUIApplication, slug: String) {
    guard let url = URL(string: "closelistening://episode/\(slug)") else {
      XCTFail("could not build a deep link for \(slug)")
      return
    }
    // `XCUIDevice.system.openURL` hands the URL to the OS, which is the real path a shared link
    // takes — not a test-only shortcut into the router. (`Process` is not available here: a UI
    // test bundle runs ON the simulator, so it cannot shell out to `xcrun` on the host.)
    XCUIDevice.shared.system.open(url)

    // iOS asks "Open in <app>?" before handing a custom scheme over from outside, and re-asks after
    // a fresh install. That alert belongs to SpringBoard, so while it is up the app under test is
    // not frontmost and EVERY accessibility query returns empty — which reads as "the page rendered
    // nothing". It cost a full diagnosis on 2026-09-16: an empty inventory on an app that was
    // fine, right after the app had been reinstalled.
    let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
    let confirm = springboard.buttons["Open"]
    if confirm.waitForExistence(timeout: 5) { confirm.tap() }

    _ = app.wait(for: .runningForeground, timeout: 15)
  }

  /// Sign in through the dev picker as `identity`. Idempotent-ish: call it only when
  /// `isSignedIn(as:)` is false.
  ///
  /// The identity is a PARAMETER because suites now get their own account (#2091) — a hard-coded
  /// `uitest` put every suite in one shared world, where one suite's favourites decided another
  /// suite's result. The mock provider mints an account for any name, so the picker is the whole
  /// mechanism; nothing server-side had to change.
  @discardableResult
  static func signIn(
    _ app: XCUIApplication, _ springboard: XCUIApplication, as identity: String = "uitest"
  ) -> Bool {
    let signIn = app.links["Sign in"].firstMatch
    guard signIn.waitForExistence(timeout: 20) else {
      // DUMP BEFORE FAILING. This message is reached in two opposite situations — genuinely signed
      // out with a broken login page, and perfectly SIGNED IN where "Sign in" is correctly absent
      // and the signed-in probe simply failed to see it. It has now been the latter three times
      // (label change; capture-stats block above "Sign out", 2026-09-18; the 2026-09-25 wedge), and
      // each time the log said only this sentence, which points at the wrong one.
      Journey.inventory(app, "signin-miss")
      XCTFail("neither Sign in nor Sign out present")
      return false
    }
    signIn.tap()

    // Dev picker: a TextField placeholdered "or a custom name…"; its Sign in button stays
    // Disabled until the field has text.
    //
    // RETRY the navigation, do not just wait longer. On a cold simulator the first tap can land
    // before the router is ready and is simply swallowed — waiting 20s then 40s on a page that was
    // never navigated to is waiting for the wrong thing. This failed as "no dev identity input" on
    // a fresh simulator and passed on the next run, and because every later suite depends on this
    // sign-in, the flake did not stay local: it left the app SIGNED OUT and downstream suites then
    // reported missing controls that were correctly absent (2026-09-24).
    //
    // The picker also needs `/auth/dev-users` to answer, so a cold API adds to the same window.
    let input = app.textFields.firstMatch
    var picker = input.waitForExistence(timeout: 20)
    if !picker {
      _ = Journey.tap(app, labels: ["Sign in"], contains: false, timeout: 10)
      picker = input.waitForExistence(timeout: 20)
    }
    guard picker else {
      // BEFORE failing: are we already signed in? `/login` correctly bounces an authenticated
      // visitor to Home, so an app that is ALREADY signed in can never show the picker — and this
      // reported "no dev identity input" for a session that was perfectly good. `isSignedIn(as:)`
      // hunts the masthead entry by the account NAME, which is not reliable when the identity has
      // not resolved, so a healthy session can fail detection and send us here.
      if isSignedIn(app) { return true }
      XCTFail(
        "no dev identity input after two attempts, and the app is not signed in either. "
          + "On screen: \(Journey.labelledInventory(app, limit: 10))"
      )
      return false
    }
    input.tap()
    input.typeText(identity)

    let submit = app.buttons["Sign in"].firstMatch
    guard submit.waitForExistence(timeout: 10), submit.isEnabled else {
      XCTFail("submit stayed disabled after typing")
      return false
    }
    submit.tap()

    // ASWebAuthenticationSession shows a system consent sheet owned by Springboard.
    let consent = springboard.buttons["Continue"]
    if consent.waitForExistence(timeout: 10) { consent.tap() }

    // Same trap as `isSignedIn`, one line further on: OAuth returns to HOME, and "Sign out" is on
    // Profile. This reported "sign-in did not complete" after sign-ins that had completed — the
    // minted token was in the simulator's preferences with an `iat` from that very run. Ask the
    // page that can actually answer.
    return isSignedIn(app)
  }

  /**
   * Signed in AS `identity` specifically.
   *
   * With one shared account "is there a session" was a sufficient question. With per-suite accounts
   * it is actively wrong: a session left by the PREVIOUS suite satisfies it, the suite proceeds
   * against someone else's data, and the collisions #2091 exists to remove come straight back — now
   * harder to see, because everything looks signed in.
   *
   * The masthead entry point is labelled with the display name, which for a dev-picker account IS
   * the identity, so Profile being reachable by that label answers both questions at once.
   */
  static func isSignedIn(_ app: XCUIApplication, as identity: String) -> Bool {
    // Open Profile by the identity label WHEN IT IS THERE, else the generic one.
    //
    // This used to reach the page by the identity alone, on the reasoning that "the masthead entry
    // is labelled with the display NAME, which for a dev-picker account IS the identity". That does
    // not hold: the avatar is `aria-label="auth.user?.name || 'Your profile'"`, so any account
    // without a resolved name is labelled generically — and the check then reported SIGNED OUT
    // about an app that was demonstrably signed in, with the avatar, the queue badge and the
    // offline stale-notice all on screen (2026-09-24).
    guard Journey.tap(app, labels: [identity], contains: true, timeout: 8)
      || Journey.tap(app, labels: ["Your profile"], contains: false, timeout: 8)
    else { return false }

    // The RIGHT account, checked FIRST — before the scroll below moves it off screen. Profile
    // prints the name and the email, and a dev identity appears in at least one of them
    // (`simtest` / `simtest@e2e.local`). This is what keeps per-suite isolation (#2091) honest:
    // "some session exists" is not the question, "whose" is.
    guard Journey.find(app, labels: [identity], contains: true, timeout: 10) != nil else {
      return false
    }

    // A session EXISTS — "Sign out" is deliberately the last control on Profile (#1962).
    guard Journey.scrollTo(app, labels: ["Sign out"], contains: false) != nil else { return false }
    // The painted session is not the answer — the revalidation that follows it is.
    sleep(6)
    return Journey.scrollTo(app, labels: ["Sign out"], contains: false) != nil
  }

  /**
   * Leave the app signed in as `identity`, whatever it was signed in as before.
   *
   * Signs OUT first when the session belongs to someone else. Without that the dev picker is
   * unreachable — the app is already signed in, so there is no "Sign in" link to tap — and the
   * suite would silently keep the previous suite's account.
   */
  /**
   * Sign out and stay out — the precondition for `/offline`, which only renders when signed out.
   *
   * `ensureSignedIn` already knew how to do this, but only as a step on the way BACK IN, so a suite
   * that wants the signed-out state had no way to ask for it. Extracted rather than copied: the
   * scroll-to-Sign-out dance has already broken twice when Profile grew (2026-09-18), and a second
   * copy would have to be fixed twice next time.
   */
  /// RETRIED, and through `Journey.tap` — ported from the Android twin, which measured why.
  ///
  /// This tapped the node `scrollTo` returned, once, and trusted it. Android hit the failure that
  /// makes that unsafe and fixed it there: "Sign out" is deliberately the LAST control on Profile
  /// (#1962 — "quiet, last, least weight"), so scrolling to it parks it at the bottom of the screen,
  /// measured UNDERNEATH the bottom nav. The tap landed on "Discover", the app navigated there, and
  /// the session was of course still present. `Journey.tap` exists precisely to lift a control clear
  /// of that overlap before touching it, and this call site skipped it — the same trap iOS's own
  /// `Journey` documents for the transport row.
  ///
  /// Re-resolving inside the loop matters as much as the retry: a node found before a scroll is
  /// stale afterwards, and tapping a stale node does nothing, silently.
  ///
  /// The app is not the suspect: `auth.logout()` drops the local identity in a `finally` so a
  /// sign-out with no network still works.
  @discardableResult
  static func signOut(_ app: XCUIApplication) -> Bool {
    guard isSignedIn(app) else { return true } // already out; the caller's precondition holds
    for attempt in 1...3 {
      _ = Journey.openProfile(app)
      guard Journey.scrollTo(app, labels: ["Sign out"], contains: false) != nil else {
        print(
          "=====SIGNOUT attempt \(attempt): no 'Sign out' on Profile :: "
            + "\(Journey.labelledInventory(app, limit: 10))====="
        )
        continue
      }
      let tapped = Journey.tap(app, labels: ["Sign out"], contains: false, timeout: 10)
      sleep(3)
      if !isSignedIn(app) { return true }
      print(
        "=====SIGNOUT attempt \(attempt) tapped=\(tapped) but a session is still present :: "
          + "\(Journey.labelledInventory(app, limit: 10))====="
      )
    }
    return false
  }

  @discardableResult
  static func ensureSignedIn(_ app: XCUIApplication, as identity: String) -> Bool {
    if isSignedIn(app, as: identity) { return true }
    let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
    // DELEGATE to `signOut`, do not re-implement it here.
    //
    // This carried its own copy of the scroll-and-tap dance — the second copy that `signOut`'s
    // docstring says it was extracted to prevent, and it had the raw-tap defect the loop above
    // exists to fix. So the path that runs when the session belongs to SOMEONE ELSE was the one
    // least able to get rid of them.
    //
    // And it must be able to FAIL. Before, a missed Sign-out tap fell through to `signIn`, which
    // cannot reach the dev picker on a signed-in app (there is no "Sign in" link) and whose own
    // fallback is `isSignedIn(app)` — any session. So the previous suite's account got certified as
    // this suite's, silently, which is exactly the per-suite isolation #2091 exists to provide.
    if isSignedIn(app), !signOut(app) {
      XCTFail(
        "signed in as another account and could not sign out after 3 attempts, so this suite "
          + "cannot get its own session. Continuing would certify the previous suite's account."
      )
      return false
    }
    // VERIFY THE IDENTITY, not merely that a session exists.
    //
    // `signIn` returns `isSignedIn(app)` — any session — deliberately, because hunting the masthead
    // by account name is unreliable before `/me` resolves, and being strict there caused false
    // negatives on healthy sessions. That tolerance is right inside `signIn` and wrong as this
    // function's contract: `ensureSignedIn(as:)` promises an account, so it checks for one.
    // Same fix as the Android twin.
    return signIn(app, springboard, as: identity) && isSignedIn(app, as: identity)
  }
}
