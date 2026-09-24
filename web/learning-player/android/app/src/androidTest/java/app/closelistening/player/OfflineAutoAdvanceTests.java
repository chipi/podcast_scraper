package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;

/**
 * CASE 1 — "I was signed in; now I am offline." (#2139, porting the iOS suite of the same name.)
 *
 * None of this is observable from the web tiers: boot from a cached identity with no network,
 * render Library from the device, and play a downloaded episode off disk.
 *
 * CASE 2 — never signed in, offline, reaching downloads through `/offline` — is a SEPARATE test
 * with a separate contract, in {@link NativeOnlySurfacesTests}. Both are offline and they are
 * otherwise unrelated; neither stands in for the other. An earlier iOS version merged them with an
 * `||`, which collapsed both into a weaker claim that could not say WHICH worked.
 *
 * PRECONDITIONS — run through `make test-android`, and only AFTER `DownloadThroughUITests` has
 * signed in and put real episodes on the device. This asserts the app boots from a STORED session
 * with the api down; a fresh install has no stored session, so it cannot pass first.
 *
 * SCOPE NOTE — auto-advance BETWEEN two downloaded episodes is deliberately not asserted here; the
 * resolver is unit-tested (`src/App.offlineAdvance.test.ts`) and this file covers the journey the
 * device can prove.
 */
@RunWith(AndroidJUnit4.class)
public class OfflineAutoAdvanceTests extends UITestCase {

    /**
     * SHARED account, deliberately: this suite reads the downloaded queue `DownloadThroughUITests`
     * leaves behind. Per-suite isolation would give it an empty account and the seed would be
     * invisible.
     */
    @Override
    protected String accountIdentity() {
        return SHARED_SEEDED_IDENTITY;
    }

    @Test
    public void bootsAndPlaysADownloadedEpisodeWithNoNetwork() {
        boolean ready = startClean();
        assertTrue("the precondition failed: this suite needs a stored session before the network "
                + "goes away. On screen: " + Journey.labelledInventory(20), ready);

        // TELL the app it is offline, as well as taking its network away at the make level.
        //
        // The api being down is a real network failure and it is what the flight looks like from
        // outside. On its own it leaves the app GUESSING: the episode page still reaches for its
        // detail, fails, and can land in an error state — so playback never starts and the failure
        // reads as "audio did not start" when nothing about audio was wrong.
        //
        // The forced-offline switch is the app's OWN contract for this ("behave as if offline"), so
        // flipping it exercises the path the offline experience is built on. Both together is the
        // honest condition: no network AND the app knowing it. The switch alone would be weaker — a
        // code path that ignored the flag and hit the network would still pass.
        assertTrue("could not force offline mode", Journey.setOfflineMode(true, profileLabels()));

        try {
            // A COLD start is the point: re-launching an already-running app only activates it, so
            // what is on disk is never re-read.
            AppSession.relaunch();

            // 1. The requirement is that the app opens into the APP, not a login wall: Library is
            //    reachable, so the episodes already on this device are too. That is what a listener
            //    needs on a plane.
            //
            //    It does NOT assert the masthead can display the account's NAME. That was the old
            //    iOS assertion and it conflated two things: "my library is here" with "the app can
            //    render my identity". Offline no login is asked for and none is possible, so gating
            //    the journey on a name failed for something the listener never needed.
            assertTrue("offline boot did not reach Library — the app put a wall in front of "
                            + "episodes that are already on this device. On screen: "
                            + Journey.labelledInventory(20),
                    Journey.openTab("Library"));

            // 2. The Downloaded list renders from the device registry, with zero successful
            //    requests.
            assertNotNull("the Downloaded section did not render offline. On screen: "
                    + Journey.labelledInventory(24), Journey.scrollTo("Downloaded", false));
            assertNotNull("no downloaded episode listed offline. On screen: "
                            + Journey.labelledInventory(24),
                    Journey.scrollTo("Downloaded — tap to remove", false));

            // 3. It opens and plays OFF DISK.
            AppSession.openEpisode("p06-7217050bc6");
            assertTrue("no Play control offline. On screen: " + Journey.labelledInventory(24),
                    Journey.tap("Play", false, 20_000));

            // Playing, OR ALREADY FINISHED. The fixture episodes are ~6 SECONDS long, so the
            // transport can run to the end and be replaced by the auto-advance end-card before this
            // assertion catches `Pause` — the state it waits for has already gone by. On iOS that
            // race reported "audio did not start" about audio that had started AND finished.
            //
            // The end-card is not a weaker proof, it is a stronger one: with no network and no
            // playable local file the episode could never reach its end at all, so "NEXT · IN"
            // means the audio ran from disk.
            boolean playing = Journey.find("Pause", false, 20_000) != null;
            boolean finished = Journey.find(Arrays.asList("NEXT"), true, 3_000) != null;
            assertTrue(
                    // What is ON the page matters more than the missing Pause. "Audio isn't
                    // available" or a not-downloaded notice means the SOURCE failed to resolve; a
                    // transport still sitting on Play means the tap never landed. Those need
                    // opposite fixes and a bare assertion cannot tell them apart.
                    "audio neither started nor completed from the downloaded file with no network. "
                            + "On screen: " + Journey.labelledInventory(24),
                    playing || finished);
        } finally {
            // Device-local and surviving both a relaunch and an account change. Left on, it breaks
            // every later suite — and on Android it blocks the next sign-in outright, because the
            // switch lives behind the masthead avatar and a signed-out app has no avatar.
            AppSession.relaunch();
            AppSession.ensureSignedIn(accountIdentity());
            Journey.setOfflineMode(false, profileLabels());
        }
    }
}
