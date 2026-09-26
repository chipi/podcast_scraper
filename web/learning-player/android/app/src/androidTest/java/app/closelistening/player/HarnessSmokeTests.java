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
