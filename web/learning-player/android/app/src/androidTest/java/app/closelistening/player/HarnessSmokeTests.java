package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * The harness proving itself, before any behaviour is asserted through it (#2139).
 *
 * Every other Android suite is worth exactly as much as this one: if sign-in, tab navigation and
 * the offline switch do not work, every downstream failure is the harness talking about itself
 * while appearing to talk about the app. The iOS tier learned that the expensive way — a stale
 * bearer token made the router bounce `/login` to Home, the dev picker never rendered, and a dozen
 * suites reported controls "missing" that were correctly absent for a signed-out app.
 *
 * So this suite is deliberately about the PLUMBING, and it is the first thing `test-android` runs.
 */
@RunWith(AndroidJUnit4.class)
public class HarnessSmokeTests extends UITestCase {

    @Test
    public void signsInNavigatesAndDrivesTheOfflineSwitch() {
        // The call FIRST, the message after. `assertTrue(msg, cond)` evaluates its arguments left to
        // right, so building the inventory inline describes the screen BEFORE the step ran — which
        // here meant reporting the launcher, because the app had not been launched yet. It sent a
        // diagnosis an hour down the wrong path (2026-09-24). A diagnostic about the wrong moment
        // is worse than none: it is confidently, specifically wrong.
        boolean signedIn = startClean();
        assertTrue(
                "sign-in did not complete as '" + accountIdentity() + "'. On screen: "
                        + Journey.labelledInventory(80),
                signedIn);

        // Tab navigation — Library is auth-gated, so reaching it proves the session is real to the
        // ROUTER, not just painted in the masthead.
        boolean library = Journey.openTab("Library");
        assertTrue(
                "Library tab unreachable. On screen: " + Journey.labelledInventory(80),
                library);

        // The offline switch drives to an ABSOLUTE state and reports what it observed afterwards.
        // Asserting the round trip is what stops it leaking into the next suite: a helper that
        // flips without verifying leaves forced-offline ON when the tap misses, and the next
        // suite's network assertions fail naming themselves.
        assertTrue("could not drive forced-offline ON", Journey.setOfflineMode(true, profileLabels()));
        assertTrue("could not drive forced-offline back OFF", Journey.setOfflineMode(false, profileLabels()));
    }

    /**
     * The REAL sign-in flow — dev picker, submit, Custom Tab consent — which nothing else runs now.
     *
     * `ensureSignedIn` tries the callback path first (mint a token over HTTP, deliver it through
     * `appUrlOpen`), because the UI flow's six sequential races cost the tier a failure roughly once
     * in nine runs and one of those races lives in `com.android.chrome`. That was the right trade
     * for the other thirteen suites. It is NOT a reason to stop testing the flow a real user takes,
     * and without this test that is precisely what happened: "Custom Tab launches: 0 across five
     * runs" was offered as evidence the fix worked, when it was equally evidence that the product's
     * actual sign-in had stopped being exercised by anything at all.
     *
     * The commit that introduced the callback path said this suite "should keep driving the UI path
     * FIRST" and then did not make it so. A "should" in a commit message is a TODO wearing a
     * claim's clothes; this is the edit that was missing.
     *
     * Deliberately ONE test, in the plumbing suite that already runs first: the consent flow is
     * paid once per tier run instead of once per suite, so the race is exercised where it can be
     * diagnosed rather than fourteen times where it cannot.
     *
     * If this goes flaky, that IS the finding. It is the path real users take, and a flake here is a
     * product or environment problem — not a second thing to route the harness around.
     */
    @Test
    public void theRealUISignInFlowStillWorks() {
        // A signed-OUT app is this test's precondition: the dev picker is unreachable while a
        // session exists, because a signed-in app has no "Sign in" link to tap.
        AppSession.relaunch();
        if (AppSession.hasAnySession()) {
            assertTrue(
                    "could not sign out to reach the signed-out state this test needs. On screen: "
                            + Journey.labelledInventory(80),
                    AppSession.signOut());
        }
        // `signIn`, NOT `ensureSignedIn` — the latter takes the callback shortcut, and this test
        // would then silently assert nothing whatsoever about the UI flow.
        assertTrue(
                "the real UI sign-in flow failed as '" + accountIdentity() + "'. On screen: "
                        + Journey.labelledInventory(80),
                AppSession.signIn(accountIdentity()));
        assertTrue(
                "the UI flow reported success but the app is not signed in as '"
                        + accountIdentity() + "'",
                AppSession.isSignedIn(accountIdentity()));
    }

    /**
     * Deep links reach a specific episode.
     *
     * This is the navigation every other suite depends on. Reaching an episode by tapping through
     * lists was the flakiest thing in the iOS tier — the lists keep their scroll position, so a
     * swipe-and-look search walks past rows and once mis-tapped into downloading an episode the
     * test never asked for.
     */
    @Test
    public void opensAnEpisodeByDeepLink() {
        boolean signedIn = startClean();
        assertTrue("sign-in did not complete", signedIn);
        AppSession.openEpisode("p06-7217050bc6");
        assertNotNull(
                "the deep link did not land on a player — no transport control. On screen: "
                        + Journey.labelledInventory(80),
                Journey.find(java.util.Arrays.asList("Play", "Pause"), false, 25_000));
    }
}
