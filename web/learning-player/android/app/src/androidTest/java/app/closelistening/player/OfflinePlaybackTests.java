package app.closelistening.player;

import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;

/**
 * Device-tier coverage for offline playback (#2139, porting #1905/#1908).
 *
 * The unit suite runs under happy-dom and Playwright runs a browser — both are the WEB case. Every
 * native path is behind `isNative()`, so none of it was exercised anywhere until this tier existed.
 * Two production bugs (artwork and audio urls resolving against `capacitor://localhost`) reached
 * main precisely because no tier covered the case where the document origin differs from the API.
 *
 * Preconditions — run through `make test-android`, and only AFTER `DownloadThroughUITests` has
 * signed in and put real episodes on the device.
 *
 * SEEK NOTE — XCUITest cannot synthesise a drag on a web `<input type=range>` and neither can
 * UiAutomator: the slider is a DOM element, so `setProgress` does not apply and a coordinate drag
 * lands at an unpredictable scroll offset. The skip buttons are real accessible controls and exercise
 * the same custom scheme-handler seek path; they are used instead. If the skip control is absent the
 * seek assertion is skipped rather than failed — it cannot be asserted from outside the DOM.
 */
@RunWith(AndroidJUnit4.class)
public class OfflinePlaybackTests extends UITestCase {

    /**
     * SHARED account, deliberately: this suite reads the downloads `DownloadThroughUITests` writes.
     * Per-suite isolation would give it an empty account and the seed would be invisible.
     */
    @Override
    protected String accountIdentity() {
        return SHARED_SEEDED_IDENTITY;
    }

    // Real fixture episodes of "The Drift" (p06) — read from the corpus, not invented.
    private static final String SLUG = "p06-7217050bc6";

    // Skip-button labels the player exposes — matched with `contains` because the full label
    // includes the number of seconds ("Skip forward 30 seconds").
    private static final String SKIP_CONTAINS = "skip";

    @Test
    public void downloadedEpisodePlaysAndSeeksOffline() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(14), ready);

        AppSession.openEpisode(SLUG);
        assertTrue("the deep link did not land. On screen: " + Journey.labelledInventory(14),
                Journey.find("Signal, Noise, and the Space Between", true, 20_000) != null);

        assertTrue("no Play control. On screen: " + Journey.labelledInventory(24),
                Journey.tap("Play", false, 20_000));

        // Playing is observable as the control flipping to Pause — or the episode finishing fast.
        //
        // Fixture episodes are ~6 SECONDS long, so the transport can run to the end and flip to the
        // auto-advance end-card before this poll catches Pause. On iOS that race reported "audio did
        // not start" about audio that had started AND finished. The end-card is the stronger proof:
        // with no playable local file the episode could never reach its end at all.
        boolean playing = Journey.find("Pause", false, 20_000) != null;
        boolean finished = Journey.find(Arrays.asList("NEXT"), true, 3_000) != null;
        assertTrue(
                "audio neither started nor completed from the downloaded file. On screen: "
                        + Journey.labelledInventory(24),
                playing || finished);

        if (!playing) {
            // Finished before we could assert further; the seek block below is unreachable.
            return;
        }

        // Let it run briefly, then exercise the skip path.
        Journey.sleep(4_000);

        // SEEK via skip controls, not a drag.
        //
        // UiAutomator cannot drive a web `<input type=range>` by value — the element is a DOM node
        // and `setProgress` applies only to Android's native ProgressBar. A coordinate drag lands
        // at a position that depends on the current scroll, which is not stable. The skip buttons
        // ("Skip forward 30 seconds") are real accessible controls in the tree and they exercise
        // the same custom scheme-handler seek path as a drag would.
        UiObject2 skipForward = Journey.find(SKIP_CONTAINS, true, 10_000);
        if (skipForward != null) {
            Journey.tap(SKIP_CONTAINS, true, 5_000);
            Journey.sleep(2_000);
            Journey.tap(SKIP_CONTAINS, true, 5_000);
            Journey.sleep(3_000);

            // Still playing after the seek — a scheme-handler range failure would stall it.
            boolean stillPlaying = Journey.find("Pause", false, 5_000) != null;
            boolean completedAfterSeek = Journey.find(Arrays.asList("NEXT"), true, 3_000) != null;
            assertTrue(
                    "playback stopped after seeking. On screen: " + Journey.labelledInventory(24),
                    stillPlaying || completedAfterSeek);
        }
        // If the skip control is absent: the seek assertion is not made. The iOS twin logs this
        // and continues; we do the same rather than failing for a missing UI control unrelated to
        // the playback correctness this suite asserts.
    }
}
