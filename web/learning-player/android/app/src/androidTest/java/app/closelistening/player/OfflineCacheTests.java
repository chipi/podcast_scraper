package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * The offline contract as the product actually intends it (#2139, porting the iOS suite).
 *
 *   browse while online  →  the app caches what you saw  →  go offline  →  it shows the cached stuff.
 *
 * This exists because the first forced-offline check got that sequence WRONG: the switch was flipped
 * on a cold install that had never browsed, so there was nothing cached and every section rendered
 * empty. That is the test's fault, not the app's, and an empty result proves nothing.
 *
 * What it does legitimately expose, and what this asserts, is the COPY: with forced-offline on, the
 * app knows it is offline, yet sections rendered the generic "Try again" — a retry affordance for a
 * request the app has deliberately refused to make. Offline state needs offline wording, not a
 * retry prompt.
 *
 * The suite restores the switch OFF in a `finally` block whatever happens, so a failure here cannot
 * poison every later run by stranding the app in forced-offline.
 */
@RunWith(AndroidJUnit4.class)
public class OfflineCacheTests extends UITestCase {

    // A real episode slug — "Risk Is a Systems Property" on feed p09.
    private static final String EPISODE_SLUG = "p09-a4bbb5dde3";

    @Test
    public void browseThenOfflineShowsCachedContent() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(14), ready);

        // --- 1. WARM THE CACHE: browse the surfaces we will later assert on, online.
        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        assertTrue("Library tab did not open", Journey.openTab("Library"));
        Journey.sleep(4_000);

        assertTrue("Home tab did not open", Journey.openTab("Home"));
        Journey.sleep(5_000);

        // --- 2. GO OFFLINE via the real Config switch.
        boolean wentOffline = Journey.setOfflineMode(true, profileLabels());
        assertTrue("could not turn Offline mode ON. On screen: " + Journey.labelledInventory(16),
                wentOffline);

        try {
            // --- 3. OBSERVE: cached content should be presented, not a wall of retry prompts.
            Journey.openTab("Home");
            Journey.sleep(5_000);

            // The offline banner must be up — "Offline mode is on — showing saved"
            // (i18n: offlineForced). Matched with `contains` so this also catches the shorter
            // network-offline variant "Offline — showing saved" (i18n: offlineNetwork) if the
            // platform decides to use that.
            UiObject2 banner = Journey.find("Offline", true, 10_000);
            assertNotNull(
                    "no offline banner while forced-offline. On screen: "
                            + Journey.labelledInventory(20),
                    banner);

            // THE ASSERTION THAT MATTERS: no retry affordance while the app knows it is offline.
            //
            // i18n key `staleRetry` = "Try again"; also checked under the episode player key
            // `retry` = "Try again". Both resolve to the same string, so one find covers them.
            // The iOS twin checked the same string; we keep parity.
            boolean retryShowing = Journey.find("Try again", false, 5_000) != null;
            assertNull(
                    "offline copy defect: 'Try again' is offered for requests the app deliberately "
                            + "did not make. On screen: " + Journey.labelledInventory(20),
                    retryShowing ? Journey.find("Try again", false, 1_000) : null);

            // The episode we browsed should still be reachable from cache.
            AppSession.openEpisode(EPISODE_SLUG);
            Journey.sleep(6_000);
            // The episode page renders something — the title, the description, anything content.
            // The iOS twin asserted just that the episode page opened; we match that.
            assertTrue(
                    "episode page did not open from cache while forced-offline. On screen: "
                            + Journey.labelledInventory(20),
                    Journey.find("Risk Is a Systems Property", true, 10_000) != null);

        } finally {
            // Device-local and surviving a relaunch and an account change. Left ON it breaks every
            // later suite — on Android it additionally blocks the next sign-in, because the switch
            // lives behind the masthead avatar and a signed-out app has no avatar.
            Journey.setOfflineMode(false, profileLabels());
        }
    }
}
