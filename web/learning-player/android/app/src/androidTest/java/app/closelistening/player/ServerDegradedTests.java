package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;

/**
 * The production incident, on device (#2139, porting the iOS suite of the same name).
 *
 * A hardware reboot lost prod's secrets. The services came back UP and answered, but could not
 * authenticate anyone — so the app saw a healthy-looking server that rejected everything, and
 * rendered "a weird mix of cached things and broken things" without ever detecting that the server
 * was the problem.
 *
 * Two host-side conditions have to be created around these tests, which is why they are split into
 * an arrange half and an assert half, sequenced by `make test-android-server-degraded`:
 *
 *   test11aWarmTheCacheWhileHealthy — runs while the api is HEALTHY: browses so the content cache
 *       is warm.
 *   test11bDegradedServerIsDetectedAndCacheSurvives — runs after the api has been restarted with a
 *       DIFFERENT signing secret, i.e. every stored token is now unverifiable. Nothing about the
 *       device changes; only the server does.
 *
 * What 11b asserts is the behaviour the incident lacked:
 *   - the app NOTICES (an offline/degraded banner, driven by `offlineReason === 'server'`)
 *   - it KEEPS the cache (it must not treat a server fault as "this user signed out" and wipe it)
 *   - it does not strand the user in a signed-out-looking shell on an admitted route
 *
 * Before the fix this test fails on the first assertion: with the network up and the server merely
 * broken, `isOffline()` was false, so no banner appeared at all.
 */
@RunWith(AndroidJUnit4.class)
public class ServerDegradedTests extends UITestCase {

    // Real episode on feed p09 — read from the fixture corpus, not invented.
    private static final String EPISODE_SLUG = "p09-a4bbb5dde3";
    private static final String EPISODE_TITLE = "Risk Is a Systems Property";

    /**
     * SHARED account, matching the iOS suite exactly.
     *
     * The iOS twin ASSERTS an existing `simtest` session rather than creating one ("not signed in —
     * run `make ios-journey-signin` before this target"). The session under test has to predate the
     * secret rotation: the whole scenario is a token the server can no longer verify. A per-suite
     * identity signed in by this test is a different arrangement, so it is not the same test.
     */
    @Override
    protected String accountIdentity() {
        return SHARED_SEEDED_IDENTITY;
    }

    /** ARRANGE (api healthy): browse enough surfaces that there is a cache worth preserving. */
    @Test
    public void test11aWarmTheCacheWhileHealthy() {
        // LAUNCH and ASSERT — do not sign in. Exactly what iOS does.
        AppSession.relaunch();
        assertTrue(
                "not signed in — run the download/seeding step before this one. On screen: "
                        + Journey.labelledInventory(80),
                Journey.find(Arrays.asList(accountIdentity(), "Your profile"), false, 20_000) != null);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        assertTrue("Library tab did not open", Journey.openTab("Library"));
        Journey.sleep(4_000);

        assertTrue("Home tab did not open", Journey.openTab("Home"));
        Journey.sleep(5_000);
        // Cache is now warm. The make target rotates the secret and runs 11b next.
    }

    /** ASSERT (api restarted with a different secret): the app degrades instead of self-destructing. */
    @Test
    public void test11bDegradedServerIsDetectedAndCacheSurvives() {
        AppSession.relaunch();
        // Let the boot reads fail and the degraded signal settle before asserting.
        Journey.sleep(8_000);

        // 1. NOTICED. The banner is the whole point: before this work the app stayed "online"
        //    because the device radio was fine, and said nothing at all.
        //
        //    i18n key `offlineServer` = "Can't reach the server — showing saved".
        //    Both that exact string and "Offline" are accepted, matching the iOS twin.
        UiObject2 banner = Journey.find(
                Arrays.asList("Can't reach the server", "Offline"), true, 15_000);
        assertNotNull(
                "no degraded/offline banner while the server cannot authenticate — the app did "
                        + "not notice. On screen: " + Journey.labelledInventory(80),
                banner);

        // 2. HONEST. A server fault must not be reported as the user's credential going bad.
        //    "Sign in" lives in the masthead when signed OUT.
        boolean signInShowing = Journey.find("Sign in", false, 5_000) != null;
        assertNull(
                "app fell back to signed-out for a SERVER fault. On screen: "
                        + Journey.labelledInventory(80),
                signInShowing ? Journey.find("Sign in", false, 1_000) : null);

        // 3. CACHE SURVIVED. `refresh()` used to clear the whole content cache on the 401 that a
        //    secret-less server returned, destroying the user's offline library over someone else's
        //    outage. The episode from 11a must still be on screen somewhere.
        UiObject2 cached = Journey.scrollTo(EPISODE_TITLE, false);
        assertNotNull(
                "cached content was wiped by a server-side fault. On screen: "
                        + Journey.labelledInventory(80),
                cached);
    }
}
